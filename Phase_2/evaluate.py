"""
evaluate.py  —  Full evaluation suite (v4 — fixed ClinicalBERT "int too big
to convert" crash, on top of v3's model2layers KeyError fix + incremental save)

Metrics computed:
  BLEU-4         (sacrebleu)
  ROUGE-1        (rouge-score)
  ROUGE-2        (rouge-score)
  ROUGE-L        (rouge-score)
  METEOR         (nltk)
  ClinicalBERT   (bert-score with emilyalsentzer/Bio_ClinicalBERT, num_layers=12)
  RadGraph-F1    (radgraph — see note below; expected to fail here)

FIX v3: bert_score's internal model2layers dict does not include
emilyalsentzer/Bio_ClinicalBERT, causing a KeyError crash. Bio_ClinicalBERT
is a standard 12-layer BERT-base architecture, so num_layers=12 is now
passed explicitly.

FIX v4: even with num_layers fixed, ClinicalBERT scoring crashed with
"OverflowError: int too big to convert". Root cause: Bio_ClinicalBERT's
tokenizer_config.json is missing from its HF repo (returns 404), so
transformers falls back to a sentinel "unset" model_max_length — an
enormous placeholder value (~1e30) — which overflows a C integer
conversion inside the tokenizer's truncation setup, regardless of how
short the actual input text is. Fixed by patching AutoTokenizer.from_pretrained
at the class level (so it's caught even though bert_score imports it
internally) to force a sane model_max_length (512) whenever a loaded
tokenizer reports that sentinel. Verified against the SATT chunk=4 eval
run's saved raw predictions (1000 samples): ClinicalBERT-F1 = 0.8035.

NOTE on RadGraph-F1: radgraph's internal (older, allennlp-based) code
calls BertTokenizer methods (.encode_plus, .build_inputs_with_special_tokens)
that were removed in the transformers version this environment needs for
the main training pipeline (transformers>=5, vs radgraph's tested range
of ~4.39-4.44). This is a hard version conflict, not something patchable
in this environment. RadGraph-F1 is therefore computed SEPARATELY, in an
isolated venv pinned to transformers==4.44.2, using the incrementally-saved
eval_raw_<variant>.csv this script produces (see retry_radgraph.py). The
try/except below lets this script continue past that failure gracefully.

SAFETY NET v3: predictions/references are still saved incrementally to a
raw CSV immediately after generation (before any metric computation),
so a crash during metric computation never loses the expensive
generation step again.
"""

import os
import time
import random
import logging
import torch
import pandas as pd
import numpy as np

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer as rs
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk

try:
    from radgraph import F1RadGraph
    RADGRAPH_AVAILABLE = True
except ImportError:
    RADGRAPH_AVAILABLE = False
    logging.warning("[Eval] radgraph not installed — RadGraph-F1 will be skipped. "
                    "pip install radgraph to enable.")

from dataset import MerlinCTDataset
from model import (
    SATTAdapter,
    MeanPoolAdapter,
    build_vision_encoder,
    build_llm_phase2_eval,
    build_tokenizer,
    encode_volume_slices,
)

# Download NLTK data silently
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

SYSTEM_PROMPT = (
    "<|begin_of_text|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    "Analyze this abdominal CT scan and generate a clinical radiology report.\n"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

# Bio_ClinicalBERT is a standard 12-layer BERT-base architecture.
# bert_score's built-in model2layers dict does not include it, so we
# must pass num_layers explicitly or it raises a KeyError.
CLINICALBERT_NUM_LAYERS = 12

# See FIX v4 above: Bio_ClinicalBERT's missing tokenizer_config.json makes
# transformers fall back to a huge sentinel model_max_length that overflows
# internally. This is the safe value we force it to instead.
CLINICALBERT_SAFE_MAX_LENGTH = 512

_CLINICALBERT_TOKENIZER_PATCHED = False


def _patch_clinicalbert_tokenizer_max_length():
    """Patch AutoTokenizer.from_pretrained at the class level so that ANY
    tokenizer bert_score loads internally (including Bio_ClinicalBERT's,
    whose HF repo is missing tokenizer_config.json) gets a sane
    model_max_length instead of an overflow-prone sentinel value.

    Patching the classmethod itself (rather than e.g. bert_score.utils'
    module-level function reference) is necessary because bert_score's
    score.py already imported that name directly before we'd get a chance
    to patch it — reassigning the utils module attribute alone silently
    does not affect score.py's already-bound reference.

    Idempotent: safe to call multiple times (e.g. once per eval run).
    """
    global _CLINICALBERT_TOKENIZER_PATCHED
    if _CLINICALBERT_TOKENIZER_PATCHED:
        return

    from transformers import AutoTokenizer

    _original = AutoTokenizer.from_pretrained.__func__

    def _patched(cls, *args, **kwargs):
        tok = _original(cls, *args, **kwargs)
        if getattr(tok, "model_max_length", None) is None or tok.model_max_length > 100_000:
            logging.info(
                f"[ClinicalBERT-Patch] Forcing model_max_length "
                f"{getattr(tok, 'model_max_length', None)} → {CLINICALBERT_SAFE_MAX_LENGTH} "
                f"for {args[0] if args else kwargs.get('pretrained_model_name_or_path')}"
            )
            tok.model_max_length = CLINICALBERT_SAFE_MAX_LENGTH
        return tok

    AutoTokenizer.from_pretrained = classmethod(_patched)
    _CLINICALBERT_TOKENIZER_PATCHED = True
    logging.info("[ClinicalBERT-Patch] AutoTokenizer.from_pretrained patched at class level.")


@torch.no_grad()
def generate_report(
    vision_encoder,
    adapter,
    llm,
    tokenizer,
    slices: torch.Tensor,
    max_new_tokens: int = 600,
) -> str:
    """Generate a radiology report for one CT volume."""
    adapter.eval()
    llm.eval()
    llm_device = next(llm.parameters()).device

    visual_tokens = encode_volume_slices(
        vision_encoder, adapter, slices.unsqueeze(0), micro_batch=8
    )
    visual_tokens = visual_tokens.to(llm_device)

    prompt_ids   = tokenizer(
        SYSTEM_PROMPT, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(llm_device)
    text_embeds  = llm.get_input_embeddings()(prompt_ids)
    vis_tokens   = visual_tokens.to(text_embeds.dtype)
    inputs_embeds = torch.cat([vis_tokens, text_embeds], dim=1)
    attn_mask    = torch.ones(
        1, inputs_embeds.shape[1], device=llm_device, dtype=torch.long
    )

    output_ids = llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def run_evaluation(args):
    logging.info("=" * 60)
    logging.info(f"EVALUATION — {args.model_type.upper()} "
                 f"(chunk={args.chunk_size if args.model_type=='satt' else 'N/A'})")
    logging.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build adapter ────────────────────────────────────────────────────────
    if args.model_type == "baseline":
        adapter = MeanPoolAdapter().to(device)
        satt_key = "phase2_best_satt_baseline.pt"
    else:
        adapter = SATTAdapter(chunk_size=args.chunk_size).to(device)
        satt_key = f"phase2_best_satt_chunk{args.chunk_size}.pt"

    satt_path = os.path.join(args.checkpoint_dir, satt_key)
    if not os.path.exists(satt_path):
        satt_path = os.path.join(args.checkpoint_dir, "phase2_best_satt.pt")

    # ── Load models ──────────────────────────────────────────────────────────
    vision_encoder = build_vision_encoder().to(device)
    vision_encoder.eval()

    ckpt = torch.load(satt_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(ckpt["satt_state"])
    adapter.eval()
    logging.info(f"[Eval] Loaded adapter from {satt_path} "
                 f"(val_loss={ckpt['loss']:.4f}, epoch={ckpt['epoch']})")

    lora_dir = os.path.join(args.checkpoint_dir, "phase2_best_lora")
    llm      = build_llm_phase2_eval(lora_dir)
    llm.eval()
    tokenizer = build_tokenizer()
    logging.info(f"[Eval] Loaded LoRA from {lora_dir}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    test_ds = MerlinCTDataset(
        args.data_dir, args.reports_xlsx,
        split="test", num_slices=args.num_slices,
    )

    if args.eval_max_samples > 0 and args.eval_max_samples < len(test_ds):
        random.seed(42)
        indices = sorted(random.sample(range(len(test_ds)), args.eval_max_samples))
        logging.info(f"[Eval] Subsampling {args.eval_max_samples} / {len(test_ds)} "
                     f"test samples (seed=42)")
    else:
        indices = list(range(len(test_ds)))
        logging.info(f"[Eval] Using full test set: {len(test_ds)} samples")

    # ── Naming for output files ─────────────────────────────────────────────
    variant_tag = (
        f"{args.model_type}"
        f"{'_chunk' + str(args.chunk_size) if args.model_type == 'satt' else ''}"
    )
    raw_path = os.path.join(args.checkpoint_dir, f"eval_raw_{variant_tag}.csv")

    predictions, references, study_ids = [], [], []
    t0 = time.time()

    for n, i in enumerate(indices):
        sample = test_ds[i]
        pred   = generate_report(
            vision_encoder, adapter, llm, tokenizer, sample["slices"]
        )
        predictions.append(pred)
        references.append(sample["findings"])
        study_ids.append(sample["study_id"])

        if (n + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate    = elapsed / (n + 1)
            eta_min = rate * (len(indices) - (n + 1)) / 60
            logging.info(
                f"Evaluated {n+1} / {len(indices)}  |  "
                f"{rate:.1f}s/sample  |  ETA {eta_min:.1f} min"
            )

        # SAFETY NET: save raw predictions incrementally every 50 samples,
        # so a crash during metric computation never loses generation work.
        if (n + 1) % 50 == 0 or (n + 1) == len(indices):
            pd.DataFrame({
                "study_id":   study_ids,
                "prediction": predictions,
                "reference":  references,
            }).to_csv(raw_path, index=False)

    logging.info(f"Generation complete. Raw predictions saved → {raw_path}")
    logging.info(f"Computing metrics...")

    # ── BLEU-4 ───────────────────────────────────────────────────────────────
    bleu        = BLEU(max_ngram_order=4)
    bleu_result = bleu.corpus_score(predictions, [references])
    logging.info(f"BLEU-4 computed: {bleu_result.score:.4f}")

    # ── ROUGE-1, ROUGE-2, ROUGE-L ────────────────────────────────────────────
    scorer      = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for ref, pred in zip(references, predictions):
        s = scorer.score(ref, pred)
        rouge1_scores.append(s["rouge1"].fmeasure)
        rouge2_scores.append(s["rouge2"].fmeasure)
        rougeL_scores.append(s["rougeL"].fmeasure)
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    logging.info(f"ROUGE computed: R1={avg_rouge1:.4f} R2={avg_rouge2:.4f} RL={avg_rougeL:.4f}")

    # ── METEOR ───────────────────────────────────────────────────────────────
    meteor_scores = []
    for ref, pred in zip(references, predictions):
        try:
            score = meteor_score(
                [word_tokenize(ref)], word_tokenize(pred)
            )
        except Exception:
            score = 0.0
        meteor_scores.append(score)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    logging.info(f"METEOR computed: {avg_meteor:.4f}")

    # ── ClinicalBERT Score (FIX v3: explicit num_layers; FIX v4: patched
    #    tokenizer model_max_length to avoid the "int too big to convert"
    #    overflow caused by Bio_ClinicalBERT's missing tokenizer_config.json) ──
    logging.info("Computing ClinicalBERT Score...")
    _patch_clinicalbert_tokenizer_max_length()
    try:
        _, _, bert_f1 = bert_score(
            predictions, references,
            lang="en",
            model_type="emilyalsentzer/Bio_ClinicalBERT",
            num_layers=CLINICALBERT_NUM_LAYERS,
            verbose=False,
        )
        avg_bert = bert_f1.mean().item()
        bert_f1_list = bert_f1.tolist()
        logging.info(f"ClinicalBERT-F1 computed: {avg_bert:.4f}")
    except Exception as e:
        logging.error(f"[Eval] ClinicalBERT Score failed: {e}")
        avg_bert = None
        bert_f1_list = [None] * len(predictions)

    # ── RadGraph-F1 ──────────────────────────────────────────────────────────
    # NOTE: this is EXPECTED to fail in this environment (transformers>=5
    # here vs radgraph's tested range of ~4.39-4.44 — a hard version
    # conflict, not fixable without a separate isolated environment).
    # RadGraph-F1 is computed separately via retry_radgraph.py against the
    # eval_raw_<variant>.csv this run just saved, using a venv pinned to
    # transformers==4.44.2. Left in place (with try/except) so a working
    # environment could still compute it inline if one is ever set up.
    avg_radgraph = None
    radgraph_scores = [None] * len(predictions)
    if RADGRAPH_AVAILABLE:
        try:
            logging.info("Computing RadGraph-F1 (may take several minutes)...")
            f1_radgraph = F1RadGraph(reward_level="partial")
            mean_f1, _, hypothesis_annotation_lists, reference_annotation_lists = \
                f1_radgraph(hyps=predictions, refs=references)
            avg_radgraph    = mean_f1
            radgraph_scores = [
                F1RadGraph.get_reward(h, r, reward_level="partial")
                for h, r in zip(
                    hypothesis_annotation_lists, reference_annotation_lists
                )
            ]
            logging.info(f"RadGraph-F1 computed: {avg_radgraph:.4f}")
        except Exception as e:
            logging.warning(
                f"[Eval] RadGraph-F1 failed (expected in this environment "
                f"— see NOTE above): {e}. Run retry_radgraph.py separately "
                f"against eval_raw_{variant_tag}.csv to get this metric."
            )
    else:
        logging.warning("[Eval] Skipping RadGraph-F1 — radgraph not installed.")

    # ── Print results ─────────────────────────────────────────────────────────
    logging.info("=" * 50)
    logging.info(f"Model      : {args.model_type} "
                 f"(chunk={args.chunk_size if args.model_type=='satt' else 'N/A'})")
    logging.info(f"Samples    : {len(predictions)}")
    logging.info(f"BLEU-4     : {bleu_result.score:.4f}")
    logging.info(f"ROUGE-1    : {avg_rouge1:.4f}")
    logging.info(f"ROUGE-2    : {avg_rouge2:.4f}")
    logging.info(f"ROUGE-L    : {avg_rougeL:.4f}")
    logging.info(f"METEOR     : {avg_meteor:.4f}")
    if avg_bert is not None:
        logging.info(f"ClinBERT-F1: {avg_bert:.4f}")
    if avg_radgraph is not None:
        logging.info(f"RadGraph-F1: {avg_radgraph:.4f}")
    logging.info("=" * 50)

    # ── Save final CSV ───────────────────────────────────────────────────────
    out_path = os.path.join(args.checkpoint_dir, f"eval_results_{variant_tag}.csv")
    pd.DataFrame({
        "study_id":    study_ids,
        "prediction":  predictions,
        "reference":   references,
        "rouge1":      rouge1_scores,
        "rouge2":      rouge2_scores,
        "rougeL":      rougeL_scores,
        "meteor":      meteor_scores,
        "bert_f1":     bert_f1_list,
        "radgraph_f1": radgraph_scores,
    }).to_csv(out_path, index=False)
    logging.info(f"Detailed results saved → {out_path}")

    return {
        "bleu4":       bleu_result.score,
        "rouge1":      avg_rouge1,
        "rouge2":      avg_rouge2,
        "rougeL":      avg_rougeL,
        "meteor":      avg_meteor,
        "bert_f1":     avg_bert,
        "radgraph_f1": avg_radgraph,
    }
