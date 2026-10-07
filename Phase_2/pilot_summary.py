"""
pilot_summary.py — one table comparing the decoding-pilot runs.

Reads every <ckpt_dir>/pilot_*/ folder written by start_decode_pilot.sh and
prints, per decoding setting: the overlap metrics, RadGraph-F1 (if
radgraph_per_sample.py has been run on that folder), report length, how often
a sentence is repeated three or more times (a decoding loop), and the
gallbladder / "Normal." patterns found in the main analysis.
All settings are compared on the same scans, so differences are paired.

Usage (major_env), after running radgraph_per_sample.py on each pilot folder:
    python pilot_summary.py --ckpt_dir ~/checkpoints
"""

import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU

from text_analysis import ABSENCE, COMPARE_RE, split_sections

TAG = "satt_chunk4"
PLAIN = re.compile(r"^\W*(normal|unremarkable)\W*$", re.IGNORECASE)
LATER = ["liver and biliary tree", "gallbladder", "spleen", "pancreas", "adrenal glands"]
GB_ABSENT = re.compile(ABSENCE["Gallbladder"][1], re.IGNORECASE)


def has_loop(text):
    sents = [s.strip().lower() for s in re.split(r"(?<=[.;])\s+", text) if len(s.split()) >= 5]
    return bool(sents) and max(Counter(sents).values()) >= 3


def plain_share(sections):
    hits = [bool(PLAIN.match(sec[k])) for sec in sections for k in LATER if k in sec]
    return float(np.mean(hits)) if hits else float("nan")


def load(folder):
    res = pd.read_csv(os.path.join(folder, f"eval_results_{TAG}.csv"))
    res["study_id"] = res["study_id"].astype(str)
    rg = os.path.join(folder, f"eval_radgraph_persample_{TAG}.csv")
    if os.path.exists(rg):
        r = pd.read_csv(rg)
        r["study_id"] = r["study_id"].astype(str)
        res = res.merge(r[["study_id", "rg_partial"]], on="study_id", how="left")
    cfg_path = os.path.join(folder, f"eval_config_{TAG}.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    return res.fillna({"prediction": "", "reference": ""}), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="~/checkpoints")
    args = ap.parse_args()
    ckpt = os.path.expanduser(args.ckpt_dir)
    folders = sorted(glob.glob(os.path.join(ckpt, "pilot_*")))
    folders = [f for f in folders if os.path.exists(os.path.join(f, f"eval_results_{TAG}.csv"))]
    if not folders:
        raise SystemExit(f"No finished pilot_* folders found in {ckpt}")

    runs = {os.path.basename(f)[len("pilot_"):]: load(f) for f in folders}
    ids = sorted(set.intersection(*(set(df["study_id"]) for df, _ in runs.values())))
    print(f"{len(runs)} decoding settings, {len(ids)} scans common to all "
          f"(split: {next(iter(runs.values()))[1].get('split', '?')})\n")

    rows, per_sample = [], {}
    for name, (df, cfg) in runs.items():
        df = df.set_index("study_id").loc[ids].reset_index()
        pred, ref = df["prediction"].astype(str), df["reference"].astype(str)
        psec, rsec = pred.map(split_sections), ref.map(split_sections)
        pairs = [(p["gallbladder"], r["gallbladder"]) for p, r in zip(psec, rsec)
                 if "gallbladder" in p and "gallbladder" in r]
        pa = np.array([bool(GB_ABSENT.search(p)) for p, _ in pairs])
        ra = np.array([bool(GB_ABSENT.search(r)) for _, r in pairs])
        dec = cfg.get("decoding", {})
        rows.append({
            "setting": name,
            "rep_pen": dec.get("repetition_penalty"),
            "ngram": dec.get("no_repeat_ngram_size"),
            "BLEU-4": BLEU(max_ngram_order=4).corpus_score(pred.tolist(), [ref.tolist()]).score,
            "ROUGE-2": df["rouge2"].mean(),
            "ROUGE-L": df["rougeL"].mean(),
            "METEOR": df["meteor"].mean(),
            "ClinBERT": df["bert_f1"].mean() if df["bert_f1"].notna().all() else np.nan,
            "RadGraph": df["rg_partial"].mean() if "rg_partial" in df else np.nan,
            "words": pred.str.split().str.len().mean(),
            "loop%": 100 * pred.map(has_loop).mean(),
            "sections": psec.map(len).mean(),
            "GB_absent%": 100 * pa.mean() if len(pairs) else np.nan,
            "GB_precision": (pa & ra).sum() / pa.sum() if pa.sum() else np.nan,
            "plainNormal%": 100 * plain_share(psec),
            "compare%": 100 * pred.map(lambda t: bool(COMPARE_RE.search(t))).mean(),
            "s/scan": cfg.get("seconds_per_scan"),
        })
        per_sample[name] = df
        ref_stats = (ref.str.split().str.len().mean(), rsec.map(len).mean(),
                     100 * ra.mean() if len(pairs) else np.nan, 100 * plain_share(rsec),
                     100 * ref.map(lambda t: bool(COMPARE_RE.search(t))).mean())

    table = pd.DataFrame(rows).set_index("setting")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.round(4).to_string())
    print(f"\nReference reports:  words {ref_stats[0]:.0f}   sections {ref_stats[1]:.1f}   "
          f"GB_absent {ref_stats[2]:.1f}%   plainNormal {ref_stats[3]:.1f}%   "
          f"compare {ref_stats[4]:.1f}%")
    print("\nColumns:  loop% = reports repeating one sentence 3+ times;  "
          "sections = organ sections found (of 13);\n"
          "  GB_absent% = generated reports calling the gallbladder surgically absent;  "
          "GB_precision = how often that claim is right;\n"
          "  plainNormal% = share of liver/gallbladder/spleen/pancreas/adrenal sections "
          "that are exactly 'Normal.';  compare% = reports with comparison-to-prior wording.")

    base = next((n for n in runs if n.startswith("A")), None)
    if base:
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(ids), size=(5000, len(ids)))
        print(f"\nPaired difference versus {base} (same scans), mean and 95% bootstrap interval")
        for col, label in [("rg_partial", "RadGraph-F1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")]:
            if col not in per_sample[base]:
                print(f"  {label}: not available yet (run radgraph_per_sample.py on each pilot folder)")
                continue
            for name, df in per_sample.items():
                if name == base or col not in df:
                    continue
                d = df[col].to_numpy(float) - per_sample[base][col].to_numpy(float)
                lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
                print(f"  {label:12s} {name:14s} {d.mean():+.4f}   [{lo:+.4f}, {hi:+.4f}]")
    table.to_csv(os.path.join(ckpt, "pilot_summary.csv"))
    print(f"\nSaved -> {os.path.join(ckpt, 'pilot_summary.csv')}")


if __name__ == "__main__":
    main()
