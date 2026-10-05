"""
bootstrap_ci.py — 95% bootstrap confidence intervals and paired comparisons
for the four SATT evaluation runs. CPU only, no model inference.

What it computes
----------------
1. For every variant and metric: the point estimate and a 95% percentile
   bootstrap confidence interval (resampling test reports with replacement).
2. For every pair of variants: the paired difference, its 95% CI and a
   two-sided bootstrap p-value. "Paired" means the SAME resampled reports
   are used for both variants, which is valid because all variants were
   evaluated on the same 1,000 scans.
3. For every metric: in what fraction of bootstrap resamples the full
   ordering  Mean-Pool < c=8 < c=4 < c=2  holds.

What it does NOT capture
------------------------
Only test-set sampling variability. Each model was trained once, so
seed-to-seed training variance is not measured by this analysis.

Inputs (all in --ckpt_dir)
--------------------------
  eval_results_<variant>.csv            per-sample rouge1, rouge2, rougeL,
                                        meteor, bert_f1 + prediction/reference
  eval_raw_<variant>_clinicalbert.csv   fallback for ClinicalBERT-F1
  eval_radgraph_persample_<variant>.csv from radgraph_per_sample.py (optional;
                                        RadGraph rows are skipped if missing)

Usage (major_env):
    python bootstrap_ci.py --ckpt_dir ~/checkpoints
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

VARIANTS = [                      # (file tag, display name), worst -> best
    ("baseline", "Mean-Pool"),
    ("satt_chunk8", "c=8"),
    ("satt_chunk4", "c=4"),
    ("satt_chunk2", "c=2"),
]
PAIRS = [                         # (better, worse) by the paper's ordering
    ("satt_chunk2", "satt_chunk4"),
    ("satt_chunk4", "satt_chunk8"),
    ("satt_chunk8", "baseline"),
    ("satt_chunk4", "baseline"),
    ("satt_chunk2", "satt_chunk8"),
]
MEAN_METRICS = [                  # (column, display name)
    ("rouge1", "ROUGE-1"),
    ("rouge2", "ROUGE-2"),
    ("rougeL", "ROUGE-L"),
    ("meteor", "METEOR"),
    ("bert_f1", "ClinicalBERT-F1"),
    ("rg_partial", "RadGraph-F1 (ent+rel)"),
    ("rg_simple", "RadGraph-F1 (ent only)"),
    ("rg_complete", "RadGraph-F1 (complete)"),
]


def load_variant(ckpt_dir, tag):
    path = os.path.join(ckpt_dir, f"eval_results_{tag}.csv")
    if not os.path.exists(path):
        sys.exit(f"MISSING: {path}")
    df = pd.read_csv(path)
    df["study_id"] = df["study_id"].astype(str)

    bert_ok = "bert_f1" in df.columns and df["bert_f1"].notna().all()
    if not bert_ok:
        alt = os.path.join(ckpt_dir, f"eval_raw_{tag}_clinicalbert.csv")
        if os.path.exists(alt):
            a = pd.read_csv(alt)
            a["study_id"] = a["study_id"].astype(str)
            df = df.drop(columns=[c for c in ["bert_f1"] if c in df.columns])
            df = df.merge(a[["study_id", "clinicalbert_f1"]]
                          .rename(columns={"clinicalbert_f1": "bert_f1"}),
                          on="study_id", how="left", validate="one_to_one")
            print(f"  [{tag}] ClinicalBERT-F1 taken from {os.path.basename(alt)}")
        else:
            print(f"  [{tag}] WARNING: no per-sample ClinicalBERT-F1 found; skipped")

    rg = os.path.join(ckpt_dir, f"eval_radgraph_persample_{tag}.csv")
    if os.path.exists(rg):
        r = pd.read_csv(rg)
        r["study_id"] = r["study_id"].astype(str)
        df = df.drop(columns=[c for c in ["radgraph_f1"] if c in df.columns])
        df = df.merge(r, on="study_id", how="left", validate="one_to_one")
    else:
        print(f"  [{tag}] NOTE: {os.path.basename(rg)} not found; RadGraph skipped")
    return df


def bleu_stats(preds, refs):
    """Per-report sufficient statistics for corpus BLEU-4 (sacrebleu)."""
    from sacrebleu.metrics import BLEU
    bleu = BLEU(max_ngram_order=4)
    stats = np.asarray(bleu._extract_corpus_statistics(preds, [refs]), dtype=np.int64)
    full = bleu._compute_score_from_stats(stats.sum(0).tolist()).score
    check = bleu.corpus_score(preds, [refs]).score
    if abs(full - check) > 1e-6:
        sys.exit(f"BLEU self-check failed: {full} vs {check}")
    return bleu, stats


def boot_bleu(bleu, stats, idx):
    out = np.empty(idx.shape[0])
    for b in range(idx.shape[0]):
        out[b] = bleu._compute_score_from_stats(stats[idx[b]].sum(0).tolist()).score
    return out


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def p_two_sided(d):
    """Two-sided bootstrap p-value for H0: difference = 0."""
    n = len(d)
    p = 2 * min((np.sum(d <= 0) + 1) / (n + 1), (np.sum(d >= 0) + 1) / (n + 1))
    return float(min(1.0, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="~/checkpoints")
    ap.add_argument("--n_boot", type=int, default=10000,
                    help="bootstrap resamples for per-report metrics")
    ap.add_argument("--n_boot_bleu", type=int, default=2000,
                    help="bootstrap resamples for corpus BLEU-4 (slower)")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    ckpt = os.path.expanduser(args.ckpt_dir)

    print("Loading per-sample results ...")
    data = {tag: load_variant(ckpt, tag) for tag, _ in VARIANTS}

    # ---- align all variants on the same reports, in the same order --------
    ids = None
    for tag, df in data.items():
        if df["study_id"].duplicated().any():
            sys.exit(f"[{tag}] duplicate study_id values")
        s = set(df["study_id"])
        ids = s if ids is None else ids & s
    order = sorted(ids)
    for tag in data:
        n_before = len(data[tag])
        data[tag] = data[tag].set_index("study_id").loc[order].reset_index()
        if n_before != len(order):
            print(f"  [{tag}] WARNING: {n_before} rows, {len(order)} shared with others")
    n = len(order)
    ref0 = data[VARIANTS[0][0]]["reference"].astype(str)
    for tag in data:
        same = (data[tag]["reference"].astype(str).values == ref0.values).mean()
        if same < 1.0:
            print(f"  [{tag}] WARNING: only {same:.1%} of references match the baseline file")
    print(f"Aligned {n} reports across {len(VARIANTS)} variants.\n")

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.n_boot, n))
    idx_bleu = idx[:args.n_boot_bleu]

    # ---- bootstrap distributions ------------------------------------------
    point, dist = {}, {}
    for col, _ in MEAN_METRICS:
        for tag, _name in VARIANTS:
            if col not in data[tag].columns or data[tag][col].isna().any():
                continue
            v = data[tag][col].to_numpy(dtype=float)
            point[(col, tag)] = float(v.mean())
            dist[(col, tag)] = v[idx].mean(axis=1)

    print(f"Bootstrapping corpus BLEU-4 ({args.n_boot_bleu} resamples per variant) ...")
    for tag, _name in VARIANTS:
        preds = data[tag]["prediction"].fillna("").astype(str).tolist()
        refs = data[tag]["reference"].fillna("").astype(str).tolist()
        bleu, stats = bleu_stats(preds, refs)
        point[("bleu4", tag)] = float(bleu._compute_score_from_stats(stats.sum(0).tolist()).score)
        dist[("bleu4", tag)] = boot_bleu(bleu, stats, idx_bleu)

    metrics = [("bleu4", "BLEU-4")] + MEAN_METRICS

    # ---- 1. per-variant CIs -------------------------------------------------
    rows = []
    print("\n" + "=" * 78)
    print("1. POINT ESTIMATES WITH 95% BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 78)
    for col, mname in metrics:
        have = [(t, nm) for t, nm in VARIANTS if (col, t) in dist]
        if not have:
            continue
        print(f"\n{mname}")
        for tag, name in have:
            lo, hi = ci(dist[(col, tag)])
            print(f"  {name:10s} {point[(col, tag)]:8.4f}   [{lo:.4f}, {hi:.4f}]")
            rows.append({"metric": mname, "variant": name, "estimate": point[(col, tag)],
                         "ci_low": lo, "ci_high": hi, "n": n})
    pd.DataFrame(rows).to_csv(os.path.join(ckpt, "bootstrap_ci_results.csv"), index=False)

    # ---- 2. paired differences ---------------------------------------------
    names = dict(VARIANTS)
    rows = []
    print("\n" + "=" * 78)
    print("2. PAIRED DIFFERENCES (A minus B), 95% CI, two-sided bootstrap p")
    print("   A CI that excludes 0 means the two variants are separable on this test set.")
    print("=" * 78)
    for col, mname in metrics:
        printed = False
        for a, b in PAIRS:
            if (col, a) not in dist or (col, b) not in dist:
                continue
            if not printed:
                print(f"\n{mname}")
                printed = True
            d = dist[(col, a)] - dist[(col, b)]
            lo, hi = ci(d)
            p = p_two_sided(d)
            pt = point[(col, a)] - point[(col, b)]
            verdict = "separable" if (lo > 0 or hi < 0) else "NOT separable"
            print(f"  {names[a]:9s} - {names[b]:9s} {pt:+8.4f}   [{lo:+.4f}, {hi:+.4f}]"
                  f"   p={p:.4f}   {verdict}")
            rows.append({"metric": mname, "A": names[a], "B": names[b], "difference": pt,
                         "ci_low": lo, "ci_high": hi, "p_value": p, "verdict": verdict})
    pd.DataFrame(rows).to_csv(os.path.join(ckpt, "bootstrap_pairwise.csv"), index=False)

    # ---- 3. probability that the full ordering holds ------------------------
    print("\n" + "=" * 78)
    print("3. SHARE OF RESAMPLES IN WHICH  Mean-Pool < c=8 < c=4 < c=2  HOLDS")
    print("=" * 78)
    rows = []
    for col, mname in metrics:
        tags = [t for t, _ in VARIANTS]
        if not all((col, t) in dist for t in tags):
            continue
        m = np.stack([dist[(col, t)] for t in tags], axis=1)
        share = float(np.mean(np.all(np.diff(m, axis=1) > 0, axis=1)))
        print(f"  {mname:26s} {share:6.1%}")
        rows.append({"metric": mname, "share_full_ordering": share})
    pd.DataFrame(rows).to_csv(os.path.join(ckpt, "bootstrap_ordering.csv"), index=False)

    print(f"\nSaved: bootstrap_ci_results.csv, bootstrap_pairwise.csv, "
          f"bootstrap_ordering.csv in {ckpt}")
    print(f"Settings: n={n} reports, {args.n_boot} resamples "
          f"({args.n_boot_bleu} for BLEU-4), seed={args.seed}")


if __name__ == "__main__":
    main()
