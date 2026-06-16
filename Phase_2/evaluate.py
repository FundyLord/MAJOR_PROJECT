"""
evaluate.py  —  BLEU-4 and ROUGE-L evaluation on the test split

Run after Phase 2 training:
    python main.py --mode eval --phase 2 --data_dir /data/.../merlin_data \
                   --reports_xlsx .../reports_final.xlsx \
                   --checkpoint_dir ./checkpoints

Requires:  pip install sacrebleu rouge-score
"""

import os
import time
import random
import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer as rs
from bert_score import score as bert_score

from dataset import MerlinCTDataset, merlin_collate_fn
from model import (
    SATTAdapter,
    build_vision_encoder,
    build_llm_phase2_eval,
    build_tokenizer,
    encode_volume_slices,
)
from train import SYSTEM_PROMPT, load_checkpoint


# ── Single-sample inference ───────────────────────────────────────────────────

@torch.no_grad()
def generate_report(
    vision_encoder,
    satt,
    llm,
    tokenizer,
    slices: torch.Tensor,           # (Z, 3, H, W)  — single sample, no batch dim
    max_new_tokens: int = 250,
) -> str:
    """Generate a radiology report for one CT volume."""
    satt.eval()
    llm.eval()

    llm_device = next(llm.parameters()).device

    # Encode volume
    visual_tokens = encode_volume_slices(
        vision_encoder, satt, slices.unsqueeze(0), micro_batch=8
    )                                                   # (1, T*N, llm_dim)
    visual_tokens = visual_tokens.to(llm_device)

    # Prompt embeddings
    prompt_ids    = tokenizer(
        SYSTEM_PROMPT, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(llm_device)
    text_embeds   = llm.get_input_embeddings()(prompt_ids)   # (1, L, llm_dim)

    vis_tokens    = visual_tokens.to(text_embeds.dtype)
    inputs_embeds = torch.cat([vis_tokens, text_embeds], dim=1)

    full_mask = torch.ones(
        1, inputs_embeds.shape[1],
        device=llm_device, dtype=torch.long
    )

    output_ids = llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,           # greedy — deterministic for evaluation
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ── Full evaluation pipeline ─────────────────────────────────────────────────

def run_evaluation(args):
    logging.info("=" * 60)
    logging.info("EVALUATION — BLEU-4 and ROUGE-L on test split")
    logging.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models
    vision_encoder = build_vision_encoder().to(device)
    satt           = SATTAdapter().to(device)
    lora_dir       = os.path.join(args.checkpoint_dir, "phase2_best_lora")
    llm            = build_llm_phase2_eval(lora_dir)
    tokenizer      = build_tokenizer()
    logging.info(f"[Eval] Loaded LoRA adapter from {lora_dir}")

    # Load Phase 2 best SATT weights directly (no optimizer needed for eval)
    best_satt_path = os.path.join(args.checkpoint_dir, "phase2_best_satt.pt")
    ckpt = torch.load(best_satt_path, map_location="cpu", weights_only=False)
    satt.load_state_dict(ckpt["satt_state"])
    logging.info(f"[Eval] Loaded SATT from {best_satt_path} (val_loss={ckpt['loss']:.4f})")

    # Dataset (test split, no shuffle)
    test_ds = MerlinCTDataset(
        args.data_dir, args.reports_xlsx,
        split="test", num_slices=args.num_slices,
    )

    if args.eval_max_samples > 0 and args.eval_max_samples < len(test_ds):
        random.seed(42)
        indices = sorted(random.sample(range(len(test_ds)), args.eval_max_samples))
        logging.info(f"[Eval] Subsampling {args.eval_max_samples} / {len(test_ds)} test samples (seed=42)")
    else:
        indices = list(range(len(test_ds)))

    predictions, references, study_ids = [], [], []
    t0 = time.time()

    for n, i in enumerate(indices):
        sample   = test_ds[i]
        pred     = generate_report(
            vision_encoder, satt, llm, tokenizer, sample["slices"]
        )
        predictions.append(pred)
        references.append(sample["findings"])
        study_ids.append(sample["study_id"])

        if (n + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (n + 1)
            eta_min = rate * (len(indices) - (n + 1)) / 60
            logging.info(
                f"Evaluated {n+1} / {len(indices)}  |  "
                f"{rate:.1f}s/sample  |  ETA {eta_min:.1f} min"
            )

    # ── BLEU-4 ──────────────────────────────────────────────────────────
    bleu        = BLEU(max_ngram_order=4)
    bleu_result = bleu.corpus_score(predictions, [references])

    # ── ROUGE-L ─────────────────────────────────────────────────────────
    scorer    = rs.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL_scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for ref, pred in zip(references, predictions)
    ]
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    # BERT Score — semantic similarity via contextual embeddings
    logging.info("Computing BERT Score (this may take a few minutes)...")
    _, _, bert_f1 = bert_score(
        predictions, references, lang="en", verbose=False,
        model_type="distilbert-base-uncased"
    )
    avg_bert_f1 = bert_f1.mean().item()

    logging.info("=" * 40)
    logging.info(f"BLEU-4     : {bleu_result.score:.4f}")
    logging.info(f"ROUGE-L    : {avg_rougeL:.4f}")
    logging.info(f"BERTScore-F1: {avg_bert_f1:.4f}")
    logging.info("=" * 40)

    # Save detailed CSV
    out_path = os.path.join(args.checkpoint_dir, "eval_results.csv")
    pd.DataFrame({
        "study_id":   study_ids,
        "prediction": predictions,
        "reference":  references,
        "rougeL":     rougeL_scores,
        "bert_f1":    bert_f1.tolist(),
    }).to_csv(out_path, index=False)
    logging.info(f"Detailed results saved → {out_path}")

    return {"bleu4": bleu_result.score, "rougeL": avg_rougeL, "bert_f1": avg_bert_f1}
