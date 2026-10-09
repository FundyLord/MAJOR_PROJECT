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
    study_id, rg_simple, rg_partial, rg_complete, and finding counts:
    gen_present_n / ref_present_n   observations RadGraph tags "definitely present"
    present_overlap                 present findings named identically in both
    gen_absent_n / ref_absent_n     observations tagged "definitely absent"
    gen_uncertain_n / ref_uncertain_n

The finding counts measure under-reporting (fewer present findings than the
reference, low recall) and unsupported findings (low precision) directly.
Matching is exact on the entity text, so "effusion" and "effusions" count
as different; treat precision/recall as approximate.

"partial" is the entities+relations level used in the paper.
Takes about 15-18 minutes per 1,000 reports on CPU.
"""

import argparse
import logging
import os

from collections import Counter

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def finding_sets(annotation):
    """Split RadGraph observation entities into present / absent / uncertain."""
    present, absent, uncertain = set(), set(), set()
    entities = (annotation or {}).get("entities", {}) or {}
    for ent in entities.values():
        label = str(ent.get("label", "")).lower()
        text = " ".join(str(ent.get("tokens", "")).lower().split())
        if not text or not ("obs" in label or "observation" in label):
            continue
        if "definitely present" in label or label.endswith("-dp"):
            present.add(text)
        elif "definitely absent" in label or label.endswith("-da"):
            absent.add(text)
        elif "uncertain" in label or label.endswith("-u"):
            uncertain.add(text)
    return present, absent, uncertain


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
    mean_reward, reward_list, hyp_ann, ref_ann = scorer(hyps=preds, refs=refs)

    ok = (isinstance(reward_list, (list, tuple)) and len(reward_list) == 3
          and all(len(x) == n for x in reward_list))
    if not ok:
        raise SystemExit(
            "Unexpected reward_list structure from this radgraph version: "
            f"type={type(reward_list)}, len={len(reward_list)}. "
            "Send this message back so the script can be adapted.")

    simple, partial, complete = (np.asarray(x, dtype=float) for x in reward_list)

    # Annotations are returned only for pairs where both reports are non-empty.
    keep = [i for i, (p, r) in enumerate(zip(preds, refs)) if len(p) and len(r)]
    if len(keep) != len(hyp_ann) or len(keep) != len(ref_ann):
        raise SystemExit(f"Annotation count mismatch: {len(keep)} non-empty pairs, "
                         f"{len(hyp_ann)} / {len(ref_ann)} annotations.")
    counts = {k: np.zeros(n, dtype=int) for k in [
        "gen_present_n", "ref_present_n", "present_overlap", "gen_absent_n",
        "ref_absent_n", "gen_uncertain_n", "ref_uncertain_n"]}
    labels = Counter()
    for i, h, r in zip(keep, hyp_ann, ref_ann):
        for ann in (h, r):
            labels.update(str(e.get("label")) for e in (ann or {}).get("entities", {}).values())
        gp, ga, gu = finding_sets(h)
        rp, ra, ru = finding_sets(r)
        counts["gen_present_n"][i], counts["ref_present_n"][i] = len(gp), len(rp)
        counts["present_overlap"][i] = len(gp & rp)
        counts["gen_absent_n"][i], counts["ref_absent_n"][i] = len(ga), len(ra)
        counts["gen_uncertain_n"][i], counts["ref_uncertain_n"][i] = len(gu), len(ru)

    pd.DataFrame({
        "study_id": df["study_id"].astype(str),
        "rg_simple": simple,
        "rg_partial": partial,
        "rg_complete": complete,
        **counts,
    }).to_csv(out_csv, index=False)

    log.info("=" * 56)
    log.info(f"variant                              : {variant}")
    log.info(f"RadGraph-F1 simple   (entities only)  : {simple.mean():.4f}")
    log.info(f"RadGraph-F1 partial  (entities+rel.)  : {partial.mean():.4f}   <- used in paper")
    log.info(f"RadGraph-F1 complete                  : {complete.mean():.4f}")
    log.info(f"library-reported means               : "
             f"{tuple(round(float(m), 4) for m in mean_reward)}")
    log.info("=" * 56)
    g, r_, o = (counts[k].sum() for k in ("gen_present_n", "ref_present_n", "present_overlap"))
    prec = o / g if g else float("nan")
    rec = o / r_ if r_ else float("nan")
    log.info("FINDINGS (RadGraph observations, per report on average)")
    log.info(f"  present   generated {counts['gen_present_n'].mean():5.2f}   "
             f"reference {counts['ref_present_n'].mean():5.2f}")
    log.info(f"  absent    generated {counts['gen_absent_n'].mean():5.2f}   "
             f"reference {counts['ref_absent_n'].mean():5.2f}")
    log.info(f"  uncertain generated {counts['gen_uncertain_n'].mean():5.2f}   "
             f"reference {counts['ref_uncertain_n'].mean():5.2f}")
    log.info(f"  present findings, exact-text match: precision {prec:.3f}   recall {rec:.3f}")
    log.info(f"  entity labels seen: {dict(labels.most_common(8))}")
    log.info("=" * 56)
    log.info(f"Per-sample scores saved -> {out_csv}")


if __name__ == "__main__":
    main()