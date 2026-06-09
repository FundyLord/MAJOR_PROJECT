"""
preprocess_dataset.py  —  One-time Merlin .nii.gz  →  .npy conversion

Converts every CT volume to a pre-zoomed, normalised float32 numpy array
so the DataLoader never has to run scipy.ndimage.zoom during training.

    Input  :  <study_id>.nii.gz   raw NIfTI   ~80-160 MB each
    Output :  <study_id>.npy      float32      (64, 224, 224), ~12 MB each

Expected total output size: ~313 GB  (25,494 × 12 MB)

Run as a CPU-only SLURM job AFTER the rclone download finishes:
    sbatch start_preprocess.sh

The script is fully resumable — already-converted files are skipped.
"""

import os
import time
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import zoom
from multiprocessing import Pool, cpu_count
from functools import partial

# ── Must match dataset.py exactly ────────────────────────────────────────────
NUM_SLICES = 64
IMAGE_SIZE = 224
HU_MIN     = -200.0
HU_MAX     =  300.0


def preprocess_one(study_id: str, src_dir: str, dst_dir: str) -> str:
    """
    Load, zoom, clip, normalise one NIfTI file and save as .npy.
    Returns a one-line status string for logging.
    """
    src = os.path.join(src_dir, f"{study_id}.nii.gz")
    dst = os.path.join(dst_dir, f"{study_id}.npy")

    # Resumable: skip files already done
    if os.path.exists(dst):
        return f"SKIP  {study_id}"

    try:
        nii = nib.load(src)
        vol = nii.get_fdata(dtype=np.float32)       # (X, Y, Z)
        vol = np.transpose(vol, (2, 1, 0))           # (Z, Y, X)

        factors = (
            NUM_SLICES / vol.shape[0],
            IMAGE_SIZE / vol.shape[1],
            IMAGE_SIZE / vol.shape[2],
        )
        vol = zoom(vol, factors, order=3)            # (64, 224, 224)
        vol = np.clip(vol, HU_MIN, HU_MAX)
        vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)    # [0, 1] float32

        np.save(dst, vol.astype(np.float32))
        return f"OK    {study_id}"

    except Exception as exc:
        return f"ERROR {study_id} — {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir",      required=True,
                    help="merlin_data/ folder with .nii.gz files")
    ap.add_argument("--dst_dir",      required=True,
                    help="Output folder for .npy files  (created if missing)")
    ap.add_argument("--reports_xlsx", required=True,
                    help="reports_final.xlsx  (used to get all study IDs)")
    ap.add_argument("--num_workers",  type=int, default=cpu_count())
    args = ap.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    # All study IDs present on disk
    df = pd.read_excel(args.reports_xlsx, engine="openpyxl")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    all_ids = df["study_id"].astype(str).tolist()
    ids = [
        sid for sid in all_ids
        if os.path.exists(os.path.join(args.src_dir, f"{sid}.nii.gz"))
    ]

    already_done = sum(
        1 for sid in ids
        if os.path.exists(os.path.join(args.dst_dir, f"{sid}.npy"))
    )

    print(f"Total  : {len(ids)} files")
    print(f"Done   : {already_done} already converted (will skip)")
    print(f"Todo   : {len(ids) - already_done} to process")
    print(f"Workers: {args.num_workers}")
    print(f"Output : {args.dst_dir}")
    print(f"Est. output size: ~{len(ids) * 12 / 1024:.0f} GB\n")

    fn = partial(preprocess_one, src_dir=args.src_dir, dst_dir=args.dst_dir)
    t0 = time.time()
    errors = []

    with Pool(args.num_workers) as pool:
        for i, result in enumerate(
            pool.imap_unordered(fn, ids, chunksize=4), start=1
        ):
            if "ERROR" in result:
                errors.append(result)
                print(result)
            elif i % 500 == 0:
                elapsed = time.time() - t0
                rate    = i / elapsed
                eta_h   = (len(ids) - i) / rate / 3600
                print(
                    f"[{i:5d}/{len(ids)}]  rate={rate:.1f} files/s  "
                    f"ETA={eta_h:.1f}h"
                )

    elapsed_h = (time.time() - t0) / 3600
    print(f"\nFinished in {elapsed_h:.2f} hours")
    print(f"Errors : {len(errors)}")
    if errors:
        for e in errors:
            print(f"  {e}")
    print(f"Output : {args.dst_dir}")


if __name__ == "__main__":
    main()
