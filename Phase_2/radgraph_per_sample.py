"""
radgraph_per_sample.py — RadGraph-F1 with PER-SAMPLE scores saved.

Why this exists
---------------
retry_radgraph.py only saved the three corpus-level means. Bootstrap
confidence intervals need one score per report. With reward_level="all",
F1RadGraph returns reward_list as a tuple of THREE LISTS
(simple, partial, complete), each holding one score per report, and the
reported corpus score is simply the mean of each list. (The comment in
retry_radgraph.py calling reward_list a "3-element summary" was wrong: it
is three per-sample lists.)

Run inside radgraph_env (transformers==4.44.2, CPU torch), same as before:

    source ~/MAJOR_PROJECT/radgraph_env/bin/activate
    python radgraph_per_sample.py --raw_csv ~/checkpoints/eval_raw_satt_chunk4.csv

Output: <ckpt_dir>/eval_radgraph_persample_<variant>.csv with columns
    study_id, rg_simple, rg_partial, rg_complete

"partial" is the entities+relations level used in the paper.
Takes about 15-18 minutes per 1,000 reports on CPU.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", required=True,
                    help="eval_raw_<variant>.csv (study_id, prediction, reference)")
    ap.add_argument("--model_type", default="radgraph-xl")
    args = ap.parse_args()

    raw_csv = os.path.expanduser(args.raw_csv)
    base = os.path.basename(raw_csv)
    if not (base.startswith("eval_raw_") and base.endswith(".csv")):
        raise SystemExit(f"Expected a file named eval_raw_<variant>.csv, got {base}")
    variant = base[len("eval_raw_"):-len(".csv")]
    out_csv = os.path.join(os.path.dirname(raw_csv),
                           f"eval_radgraph_persample_{variant}.csv")

    df = pd.read_csv(raw_csv)
    n = len(df)
    log.info(f"Loaded {n} rows from {raw_csv} (variant={variant})")
    preds = df["prediction"].fillna("").astype(str).tolist()
    refs = df["reference"].fillna("").astype(str).tolist()
    n_empty = sum(1 for p, r in zip(preds, refs) if len(p) == 0 or len(r) == 0)
    if n_empty:
        log.warning(f"{n_empty} rows have an empty prediction or reference; "
                    f"RadGraph scores them as 0.")

    from radgraph import F1RadGraph
    log.info(f"Initializing F1RadGraph (model_type={args.model_type}, reward_level=all)")
    scorer = F1RadGraph(reward_level="all", model_type=args.model_type)

    log.info("Computing RadGraph-F1 ...")
    mean_reward, reward_list, _, _ = scorer(hyps=preds, refs=refs)

    ok = (isinstance(reward_list, (list, tuple)) and len(reward_list) == 3
          and all(len(x) == n for x in reward_list))
    if not ok:
        raise SystemExit(
            "Unexpected reward_list structure from this radgraph version: "
            f"type={type(reward_list)}, len={len(reward_list)}. "
            "Send this message back so the script can be adapted.")

    simple, partial, complete = (np.asarray(x, dtype=float) for x in reward_list)
    pd.DataFrame({
        "study_id": df["study_id"].astype(str),
        "rg_simple": simple,
        "rg_partial": partial,
        "rg_complete": complete,
    }).to_csv(out_csv, index=False)

    log.info("=" * 56)
    log.info(f"variant                              : {variant}")
    log.info(f"RadGraph-F1 simple   (entities only)  : {simple.mean():.4f}")
    log.info(f"RadGraph-F1 partial  (entities+rel.)  : {partial.mean():.4f}   <- used in paper")
    log.info(f"RadGraph-F1 complete                  : {complete.mean():.4f}")
    log.info(f"library-reported means               : "
             f"{tuple(round(float(m), 4) for m in mean_reward)}")
    log.info("=" * 56)
    log.info(f"Per-sample scores saved -> {out_csv}")


if __name__ == "__main__":
    main()
