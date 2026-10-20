"""
case_examples.py — picks representative error cases for the paper's case table.

Reads the corrected c=4 test outputs and prints, organ section by organ section:
  A. false "gallbladder absent" claims (factual error)
  B. omissions: reference section reports a finding, generated section says only "Normal."
  C. comparisons with a prior exam the model never saw (unsupported information)

Run from Phase_2/ in major_env:
    python case_examples.py --raw_csv ~/checkpoints/eval_v2/eval_raw_satt_chunk4.csv
"""

import argparse
import os
import re

import pandas as pd

from text_analysis import ABSENCE, COMPARE_RE, split_sections

PLAIN = re.compile(r"^\W*(normal|unremarkable)\W*$", re.IGNORECASE)
FINDING = re.compile(
    r"mass|lesion|hydronephrosis|ascites|metasta|calcul|stone|"
    r"dilat|effusion|hernia|thicken|nodul|cyst|fracture",
    re.IGNORECASE,
)
GB_ABSENT = re.compile(ABSENCE["Gallbladder"][1], re.IGNORECASE)


def short(text, n=260):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", required=True)
    ap.add_argument("--k", type=int, default=3, help="examples per error type")
    args = ap.parse_args()

    df = pd.read_csv(os.path.expanduser(args.raw_csv)).fillna("")
    df = df.sort_values("study_id").reset_index(drop=True)
    P = df["prediction"].map(split_sections)
    R = df["reference"].map(split_sections)

    print("=" * 78)
    print("A. FALSE 'GALLBLADDER ABSENT' (generated says absent, reference describes it)")
    print("=" * 78)
    shown = 0
    for i in range(len(df)):
        p, r = P[i].get("gallbladder"), R[i].get("gallbladder")
        if p and r and GB_ABSENT.search(p) and not GB_ABSENT.search(r):
            print(f"\n[{df.study_id[i]}]\n  REFERENCE : {short(r)}\n  GENERATED : {short(p)}")
            shown += 1
            if shown >= args.k:
                break

    print("\n" + "=" * 78)
    print("B. OMISSIONS (reference reports a finding; generated section is only 'Normal.')")
    print("=" * 78)
    shown, used = 0, set()
    for i in range(len(df)):
        if i in used:
            continue
        for organ, rtext in R[i].items():
            ptext = P[i].get(organ)
            if ptext and PLAIN.match(ptext) and FINDING.search(rtext):
                print(
                    f"\n[{df.study_id[i]}] section: {organ}\n"
                    f"  REFERENCE : {short(rtext)}\n  GENERATED : {short(ptext)}"
                )
                shown += 1
                used.add(i)
                break
        if shown >= args.k:
            break

    print("\n" + "=" * 78)
    print("C. UNSUPPORTED COMPARISON WITH A PRIOR EXAM (model receives one scan only)")
    print("=" * 78)
    shown = 0
    for i in range(len(df)):
        for organ, ptext in P[i].items():
            m = COMPARE_RE.search(ptext)
            if m:
                print(
                    f"\n[{df.study_id[i]}] section: {organ}\n"
                    f"  REFERENCE : {short(R[i].get(organ, '(no such section)'))}\n"
                    f"  GENERATED : {short(ptext)}"
                )
                shown += 1
                break
        if shown >= args.k:
            break


if __name__ == "__main__":
    main()
