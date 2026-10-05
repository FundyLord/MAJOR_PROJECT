"""
text_analysis.py — quantifies the error patterns seen in generated reports
and checks whether the 1,000-scan evaluation subset resembles the full
test split. CPU only, reads CSV/XLSX, no model inference.

Part A (per variant, from eval_raw_<variant>.csv)
  A1. Organ-absence claims (e.g. "Gallbladder: Surgically absent"):
      how often the generated report claims it, how often the reference
      does, and precision / recall of the generated claim.
  A2. Comparison-to-prior language ("previously", "interval", "unchanged").
      The model sees one scan, so such a statement in a GENERATED report is
      never supported by its input, even when it happens to match.
  A3. Series/image citations ("series 3, image 4", "(602/87)"). The model
      receives 64 resampled slices and cannot know original image numbers.
  A4. Size measurements ("1.2 cm", "3 x 4 mm"): counted, as context only.
      A measurement is not automatically wrong; this needs manual review.
  Example rows for a case table are written to
      hallucination_examples_<variant>.csv

Part B (only if --reports_xlsx is given)
  Compares the evaluated subset against the rest of the test split on
  reference-report length and on how often common terms are mentioned.

Usage (major_env):
    python text_analysis.py --ckpt_dir ~/checkpoints \
        --reports_xlsx /data/yashjadhav23/merlin/reports_final.xlsx \
        --npy_dir /data/yashjadhav23/merlin/merlin_data_npy
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

VARIANTS = [("baseline", "Mean-Pool"), ("satt_chunk8", "c=8"),
            ("satt_chunk4", "c=4"), ("satt_chunk2", "c=2")]

SECTIONS = ["Lower thorax", "Liver and biliary tree", "Gallbladder", "Spleen",
            "Pancreas", "Adrenal glands", "Kidneys and ureters",
            "Gastrointestinal tract", "Peritoneal cavity", "Pelvic organs",
            "Vasculature", "Lymph nodes", "Musculoskeletal"]
_SEC_RE = re.compile(r"(" + "|".join(re.escape(s) for s in SECTIONS) + r")\s*:",
                     flags=re.IGNORECASE)

# organ -> (section holding the claim, pattern meaning "organ surgically absent")
ABSENCE = {
    "Gallbladder": ("gallbladder",
                    r"surgically absent|cholecystectomy|^\W*absent\b|has been removed|"
                    r"surgically removed"),
    "Spleen": ("spleen",
               r"surgically absent|splenectomy|^\W*absent\b|has been removed|"
               r"surgically removed"),
}

COMPARE_RE = re.compile(
    r"\b(prior|previous|previously|interval|unchanged|stable|compared|comparison|"
    r"redemonstrat\w*|again (?:seen|noted|demonstrated|identified)|similar to|"
    r"since the|dated)\b", flags=re.IGNORECASE)
CITE_RE = re.compile(
    r"series\s*\d+\s*[,;/]?\s*image\s*\d+|\bimage\s*\d+\b|\(\s*\d{1,4}\s*/\s*\d{1,4}\s*\)",
    flags=re.IGNORECASE)
MEASURE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:x\s*\d+(?:\.\d+)?\s*){0,2}(?:mm|cm)\b", flags=re.IGNORECASE)

FINDING_TERMS = {   # used only for the subset-representativeness check
    "gallbladder absent": r"cholecystectomy|gallbladder[^.]{0,30}(surgically )?absent",
    "ascites": r"\bascites\b",
    "pleural effusion": r"pleural effusion",
    "hydronephrosis": r"hydronephrosis",
    "cirrhosis": r"cirrho",
    "metastasis": r"metasta",
    "hernia": r"\bhernia",
    "stone/calculus": r"calcul|lithiasis|\bstones?\b",
    "mass": r"\bmass(es)?\b",
    "comparison language": COMPARE_RE.pattern,
}


def split_sections(text):
    """Return {lower-case section name: text} for a templated findings report."""
    text = "" if not isinstance(text, str) else text
    out, hits = {}, list(_SEC_RE.finditer(text))
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        out.setdefault(m.group(1).lower(), text[m.end():end].strip())
    return out


def first_sentence_with(text, regex):
    for sent in re.split(r"(?<=[.;])\s+", text if isinstance(text, str) else ""):
        if regex.search(sent):
            return sent.strip()[:300]
    return ""


def pct(x):
    return f"{100 * x:5.1f}%"


def analyse_variant(ckpt, tag, name, n_examples):
    path = os.path.join(ckpt, f"eval_raw_{tag}.csv")
    if not os.path.exists(path):
        print(f"\n[{name}] MISSING {path} - skipped")
        return None
    df = pd.read_csv(path)
    df["prediction"] = df["prediction"].fillna("").astype(str)
    df["reference"] = df["reference"].fillna("").astype(str)
    n = len(df)
    psec = df["prediction"].map(split_sections)
    rsec = df["reference"].map(split_sections)
    summary = {"variant": name, "n": n}
    examples = []

    print("\n" + "=" * 78)
    print(f"{name}   ({os.path.basename(path)}, n={n})")
    print("=" * 78)
    print(f"  words per report   generated: mean {df['prediction'].str.split().str.len().mean():.0f}"
          f"   reference: mean {df['reference'].str.split().str.len().mean():.0f}")
    print(f"  template sections found per report   generated: {psec.map(len).mean():.1f}"
          f"   reference: {rsec.map(len).mean():.1f}   (of {len(SECTIONS)})")

    # ---- A1 organ-absence claims -------------------------------------------
    print("\n  A1. Organ-absence claims")
    for organ, (sec, pat) in ABSENCE.items():
        rx = re.compile(pat, flags=re.IGNORECASE)
        both = [(i, p[sec], r[sec]) for i, (p, r) in enumerate(zip(psec, rsec))
                if sec in p and sec in r]
        if not both:
            print(f"    {organ}: section not found in both reports - skipped")
            continue
        pa = np.array([bool(rx.search(p)) for _, p, _ in both])
        ra = np.array([bool(rx.search(r)) for _, _, r in both])
        tp, fp = int((pa & ra).sum()), int((pa & ~ra).sum())
        fn, m = int((~pa & ra).sum()), len(both)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"    {organ} (both sections present in {m} of {n} pairs)")
        print(f"      reference says absent : {int(ra.sum()):4d}  ({pct(ra.mean())})")
        print(f"      generated says absent : {int(pa.sum()):4d}  ({pct(pa.mean())})")
        print(f"      generated 'absent' but reference shows organ present: "
              f"{fp:4d}  ({pct(fp / m)} of pairs)")
        print(f"      precision of generated 'absent' claim: {prec:.3f}   recall: {rec:.3f}")
        key = organ.lower()
        summary.update({f"{key}_pairs": m, f"{key}_ref_absent": int(ra.sum()),
                        f"{key}_gen_absent": int(pa.sum()), f"{key}_false_absent": fp,
                        f"{key}_precision": prec, f"{key}_recall": rec})
        for (i, p, r), is_fp in zip(both, pa & ~ra):
            if is_fp and sum(e["error_type"].startswith(organ) for e in examples) < n_examples:
                examples.append({"study_id": df["study_id"].iloc[i],
                                 "error_type": f"{organ}: false 'absent' claim",
                                 "reference_text": r[:300], "generated_text": p[:300]})

    # ---- A2-A4 pattern counts ----------------------------------------------
    def share_and_mean(series, rx):
        c = series.map(lambda t: len(rx.findall(t)))
        return float((c > 0).mean()), float(c.mean())

    print("\n  A2-A4. Share of reports containing the pattern (mean count per report)")
    print(f"    {'pattern':34s}{'generated':>18s}{'reference':>18s}")
    for label, rx, key in [("A2 comparison-to-prior language", COMPARE_RE, "compare"),
                           ("A3 series/image citation", CITE_RE, "cite"),
                           ("A4 size measurement (mm/cm)", MEASURE_RE, "measure")]:
        gs, gm = share_and_mean(df["prediction"], rx)
        rs_, rm = share_and_mean(df["reference"], rx)
        print(f"    {label:34s}{pct(gs):>9s} ({gm:4.1f}){pct(rs_):>10s} ({rm:4.1f})")
        summary.update({f"{key}_gen_share": gs, f"{key}_gen_mean": gm,
                        f"{key}_ref_share": rs_, f"{key}_ref_mean": rm})
    d = float(df["prediction"].str.contains("<DATE>", regex=False).mean())
    print(f"    generated reports containing a '<DATE>' placeholder: {pct(d)}")
    summary["date_placeholder_gen_share"] = d

    for label, rx in [("Comparison to a prior study (no prior in input)", COMPARE_RE),
                      ("Series/image citation (not available to model)", CITE_RE)]:
        k = 0
        for i, t in enumerate(df["prediction"]):
            s = first_sentence_with(t, rx)
            if s:
                examples.append({"study_id": df["study_id"].iloc[i], "error_type": label,
                                 "reference_text": "", "generated_text": s})
                k += 1
                if k >= n_examples:
                    break

    ex_path = os.path.join(ckpt, f"hallucination_examples_{tag}.csv")
    pd.DataFrame(examples).to_csv(ex_path, index=False)
    print(f"\n  {len(examples)} example rows -> {ex_path}")
    return summary


def subset_check(ckpt, reports_xlsx, npy_dir):
    from scipy import stats
    raw = os.path.join(ckpt, "eval_raw_baseline.csv")
    if not os.path.exists(raw):
        print("\nPart B skipped: eval_raw_baseline.csv not found")
        return
    sub_ids = set(pd.read_csv(raw)["study_id"].astype(str))
    df = pd.read_excel(os.path.expanduser(reports_xlsx), engine="openpyxl")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    test = df[df["split"].astype(str).str.strip().str.lower() == "test"].copy()
    test["study_id"] = test["study_id"].astype(str)
    n_listed = len(test)
    if npy_dir:
        npy_dir = os.path.expanduser(npy_dir)
        test = test[test["study_id"].map(
            lambda s: os.path.exists(os.path.join(npy_dir, f"{s}.npy")))]
    test["findings"] = test["findings"].fillna("").astype(str)
    in_sub = test["study_id"].isin(sub_ids)
    a, b = test[in_sub], test[~in_sub]

    print("\n" + "=" * 78)
    print("PART B. IS THE 1,000-SCAN SUBSET REPRESENTATIVE OF THE TEST SPLIT?")
    print("=" * 78)
    print(f"  test rows in spreadsheet: {n_listed}   usable (file on disk): {len(test)}")
    print(f"  evaluated subset: {len(a)}   not evaluated: {len(b)}")
    if len(a) != len(sub_ids):
        print(f"  WARNING: {len(sub_ids) - len(a)} evaluated study_ids not found in test split")

    la, lb = a["findings"].str.split().str.len(), b["findings"].str.split().str.len()
    ks = stats.ks_2samp(la, lb)
    print("\n  Reference report length (words)")
    print(f"    subset : mean {la.mean():6.1f}  sd {la.std():6.1f}  median {la.median():6.1f}")
    print(f"    rest   : mean {lb.mean():6.1f}  sd {lb.std():6.1f}  median {lb.median():6.1f}")
    print(f"    Kolmogorov-Smirnov D={ks.statistic:.3f}, p={ks.pvalue:.3f}")

    rows = []
    print("\n  Share of reference reports that MENTION each term (negated mentions included)")
    print(f"    {'finding':22s}{'subset':>10s}{'rest':>10s}{'chi-sq p':>11s}")
    for label, pat in FINDING_TERMS.items():
        rx = re.compile(pat, flags=re.IGNORECASE)
        ha = a["findings"].map(lambda t: bool(rx.search(t)))
        hb = b["findings"].map(lambda t: bool(rx.search(t)))
        table = [[int(ha.sum()), int((~ha).sum())], [int(hb.sum()), int((~hb).sum())]]
        try:
            p = stats.chi2_contingency(table)[1]
        except ValueError:
            p = float("nan")
        print(f"    {label:22s}{pct(ha.mean()):>10s}{pct(hb.mean()):>10s}{p:11.3f}")
        rows.append({"finding": label, "subset_prevalence": float(ha.mean()),
                     "rest_prevalence": float(hb.mean()), "chi2_p": p})
    rows.append({"finding": "report length (KS test)", "subset_prevalence": float(la.mean()),
                 "rest_prevalence": float(lb.mean()), "chi2_p": float(ks.pvalue)})
    out = os.path.join(ckpt, "subset_representativeness.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  {len(FINDING_TERMS) + 1} comparisons were made; about one in twenty falls below "
          f"p=0.05 by chance alone.\n  Saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="~/checkpoints")
    ap.add_argument("--reports_xlsx", default=None)
    ap.add_argument("--npy_dir", default=None,
                    help="preprocessed .npy folder, to drop test scans with no file")
    ap.add_argument("--n_examples", type=int, default=8)
    args = ap.parse_args()
    ckpt = os.path.expanduser(args.ckpt_dir)

    print("PART A. ERROR PATTERNS IN GENERATED REPORTS")
    rows = [s for tag, name in VARIANTS
            if (s := analyse_variant(ckpt, tag, name, args.n_examples)) is not None]
    if rows:
        out = os.path.join(ckpt, "hallucination_summary.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSummary for all variants -> {out}")

    if args.reports_xlsx:
        subset_check(ckpt, args.reports_xlsx, args.npy_dir)
    else:
        print("\nPart B skipped (pass --reports_xlsx to run it).")


if __name__ == "__main__":
    main()
