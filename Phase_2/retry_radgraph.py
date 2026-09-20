"""
retry_radgraph.py — Standalone RadGraph-F1 retry.

Loads the already-saved raw predictions/references from a previous eval run
(eval_raw_<variant>.csv) and computes ONLY RadGraph-F1, without re-running
generation.

Root cause of "BertTokenizer has no attribute encode_plus" / deeper
"build_inputs_with_special_tokens" errors: radgraph's internal (older,
allennlp-based) tokenizer-handling code depends on transformers API methods
that were removed/refactored in transformers v5.x. radgraph is only tested
against transformers>=4.39.0,<4.45 (the 4.x generation). This is a hard
version conflict with the main pipeline's transformers>=5 requirement —
not something patchable in the same environment.

FIX: run this script inside a completely separate venv pinned to
transformers==4.44.2 + CPU-only torch (radgraph_env), fully isolated from
the main training/eval environment (major_env). Confirmed working end to
end against eval_raw_satt_chunk4.csv (1000 samples, ~15 min on CPU):
  RadGraph-F1 (entities only)          = 0.2790
  RadGraph-F1 (entities + relations)   = 0.2442
  RadGraph-F1 (entities+relations avg) = 0.1646

FIX v2 (this version): the original per-sample CSV-save line assumed
reward_list had one entry per sample, but with reward_level="all",
F1RadGraph(...) returns a 3-element SUMMARY tuple in reward_list (not
one score per sample) — trying to zip it against 1000 rows crashed with
"ValueError: Length of values (3) does not match length of index (1000)".
This happened AFTER the actual scores were already computed and logged,
so it never affected correctness — just the optional per-sample CSV
column. Fixed by saving the run-level summary scores separately instead
of attempting a bogus per-row column.

Setup (run once, on a machine/venv separate from your main environment):
    python3 -m venv radgraph_env
    source radgraph_env/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers==4.44.2 radgraph

Usage:
    source radgraph_env/bin/activate
    python retry_radgraph.py --raw_csv /home/yashjadhav23/checkpoints/eval_raw_satt_chunk4.csv
"""

import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--model_type", default="radgraph-xl")
    args = parser.parse_args()

    out_csv = args.out_csv or args.raw_csv.replace(".csv", "_radgraph.csv")

    log.info(f"Loading raw predictions from {args.raw_csv} ...")
    df = pd.read_csv(args.raw_csv)
    log.info(f"Loaded {len(df)} rows.")

    predictions = df["prediction"].astype(str).tolist()
    references  = df["reference"].astype(str).tolist()

    from radgraph import F1RadGraph

    log.info(f"Initializing F1RadGraph (model_type={args.model_type}, reward_level=all) ...")
    f1radgraph = F1RadGraph(reward_level="all", model_type=args.model_type)

    log.info("Computing RadGraph-F1 (this can take several minutes on CPU) ...")
    mean_reward, reward_list, hyp_annotations, ref_annotations = f1radgraph(
        hyps=predictions, refs=references
    )

    # mean_reward is itself (simple, partial, complete) at the corpus level
    # when reward_level="all"; reward_list is the matching 3-element summary,
    # NOT one score per sample — do not try to zip it against df rows.
    log.info("=" * 50)
    if isinstance(mean_reward, (list, tuple)) and len(mean_reward) == 3:
        entities_only, entities_relations, entities_relations_avg = mean_reward
        log.info(f"RadGraph-F1 (entities only)          : {entities_only:.4f}")
        log.info(f"RadGraph-F1 (entities + relations)   : {entities_relations:.4f}")
        log.info(f"RadGraph-F1 (entities+relations avg) : {entities_relations_avg:.4f}")
    else:
        entities_only = entities_relations = entities_relations_avg = None
        log.info(f"RadGraph mean_reward (unexpected shape): {mean_reward}")
    log.info("=" * 50)

    # Save the run-level summary (not a bogus per-row column).
    summary_path = args.raw_csv.replace(".csv", "_radgraph_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"raw_csv: {args.raw_csv}\n")
        f.write(f"samples: {len(predictions)}\n")
        f.write(f"radgraph_f1_entities_only: {entities_only}\n")
        f.write(f"radgraph_f1_entities_relations: {entities_relations}\n")
        f.write(f"radgraph_f1_entities_relations_avg: {entities_relations_avg}\n")
    log.info(f"Summary saved → {summary_path}")

    # Still write out the annotations alongside predictions/references, in
    # case per-sample graph structure is useful later (this is NOT a scalar
    # per-sample score, so it's kept as its own columns rather than forced
    # into a "radgraph_f1" column).
    df["hyp_annotation"] = [str(a) for a in hyp_annotations]
    df["ref_annotation"] = [str(a) for a in ref_annotations]
    df.to_csv(out_csv, index=False)
    log.info(f"Per-sample annotations saved → {out_csv}")


if __name__ == "__main__":
    main()
