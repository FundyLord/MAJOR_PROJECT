"""
retry_clinicalbert.py — Standalone ClinicalBERT-Score retry (v2).

Loads the already-saved raw predictions/references from a previous eval run
(eval_raw_satt_chunk4.csv) and computes ONLY ClinicalBERT-Score, without
re-running generation.

Root cause of the "int too big to convert" error: Bio_ClinicalBERT's
tokenizer_config.json is missing from its HF repo (404), so transformers
falls back to a sentinel "unset" model_max_length (an enormous placeholder
value). That sentinel overflows a C integer conversion inside the
tokenizer's Rust-backed truncation setup.

FIX v2: v1 patched bert_score.utils.get_tokenizer, but bert_score/score.py
had already imported that name directly (`from .utils import get_tokenizer`)
before the patch ran, so reassigning the attribute on the utils module left
score.py's already-bound reference untouched — the patch silently didn't
apply. Fixed by patching AutoTokenizer.from_pretrained itself (the actual
classmethod), which every internal caller reaches through the class object
regardless of how they imported it.

Usage:
    python retry_clinicalbert.py --raw_csv /home/yashjadhav23/checkpoints/eval_raw_satt_chunk4.csv
"""

import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

CLINICALBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
CLINICALBERT_NUM_LAYERS = 12
SAFE_MAX_LENGTH = 512


def _patch_tokenizer_max_length():
    """Patch AutoTokenizer.from_pretrained at the class level so EVERY
    caller (including bert_score's internal, already-imported reference)
    gets a tokenizer with a sane model_max_length, no matter how they
    imported or called it."""
    from transformers import AutoTokenizer

    _original = AutoTokenizer.from_pretrained.__func__

    def _patched(cls, *args, **kwargs):
        tok = _original(cls, *args, **kwargs)
        if getattr(tok, "model_max_length", None) is None or tok.model_max_length > 100_000:
            log.info(f"[Patch] Forcing model_max_length {getattr(tok, 'model_max_length', None)} "
                     f"→ {SAFE_MAX_LENGTH} for {args[0] if args else kwargs.get('pretrained_model_name_or_path')}")
            tok.model_max_length = SAFE_MAX_LENGTH
        return tok

    AutoTokenizer.from_pretrained = classmethod(_patched)
    log.info("[Patch] AutoTokenizer.from_pretrained patched at class level.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    out_csv = args.out_csv or args.raw_csv.replace(".csv", "_clinicalbert.csv")

    log.info(f"Loading raw predictions from {args.raw_csv} ...")
    df = pd.read_csv(args.raw_csv)
    log.info(f"Loaded {len(df)} rows.")

    _patch_tokenizer_max_length()

    from bert_score import score as bert_score

    log.info("Computing ClinicalBERT-Score (patched tokenizer) ...")
    predictions = df["prediction"].astype(str).tolist()
    references  = df["reference"].astype(str).tolist()

    P, R, F1 = bert_score(
        predictions, references,
        model_type=CLINICALBERT_MODEL,
        num_layers=CLINICALBERT_NUM_LAYERS,
        lang="en",
        verbose=True,
    )

    mean_p, mean_r, mean_f1 = P.mean().item(), R.mean().item(), F1.mean().item()
    log.info("=" * 50)
    log.info(f"ClinicalBERT-Precision : {mean_p:.4f}")
    log.info(f"ClinicalBERT-Recall    : {mean_r:.4f}")
    log.info(f"ClinicalBERT-F1        : {mean_f1:.4f}")
    log.info("=" * 50)

    df["clinicalbert_precision"] = P.tolist()
    df["clinicalbert_recall"]    = R.tolist()
    df["clinicalbert_f1"]        = F1.tolist()
    df.to_csv(out_csv, index=False)
    log.info(f"Per-sample scores saved → {out_csv}")


if __name__ == "__main__":
    main()
