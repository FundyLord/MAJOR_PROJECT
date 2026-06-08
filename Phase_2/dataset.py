"""
dataset.py  —  Merlin Abdominal CT Dataset Loader

Preprocessing matches the validated notebook (new_check.ipynb):
  1. Load NIfTI  →  (X, Y, Z)
  2. Transpose   →  (Z, Y, X)   — depth axis first
  3. 3D cubic zoom on RAW HU values to target shape
  4. HU clip [-200, 300]  AFTER zoom  (avoids interpolation artefacts at boundary)
  5. Normalize  [0, 1]
  6. Expand to 3 channels, apply SigLIP normalisation  →  [-1, 1]
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom


class MerlinCTDataset(Dataset):
    """
    Args:
        data_dir     : flat folder containing  <study_id>.nii.gz  files
        reports_xlsx : path to  reports_final.xlsx
        split        : 'train' | 'val' | 'test'
        num_slices   : target depth after 3-D zoom  (Z axis)
        image_size   : target H and W after zoom    (SigLIP expects 224)
        hu_min       : soft-tissue window lower bound
        hu_max       : soft-tissue window upper bound
    """

    def __init__(
        self,
        data_dir:     str,
        reports_xlsx: str,
        split:        str   = "train",
        num_slices:   int   = 64,
        image_size:   int   = 224,
        hu_min:       float = -200.0,
        hu_max:       float =  300.0,
    ):
        self.data_dir   = data_dir
        self.num_slices = num_slices
        self.image_size = image_size
        self.hu_min     = hu_min
        self.hu_max     = hu_max

        # Load reports spreadsheet
        df = pd.read_excel(reports_xlsx, engine="openpyxl")
        df.columns = (
            df.columns.str.strip()
                      .str.lower()
                      .str.replace(" ", "_", regex=False)
        )
        # Expected columns after normalisation: study_id, findings, split, few_shot

        self.df = (
            df[df["split"].str.strip().str.lower() == split.lower()]
            .reset_index(drop=True)
        )

        # Drop rows whose NIfTI file is missing
        exists = self.df["study_id"].apply(
            lambda sid: os.path.exists(os.path.join(data_dir, f"{sid}.nii.gz"))
        )
        n_missing = int((~exists).sum())
        if n_missing:
            print(f"[Dataset] WARNING: {n_missing} NIfTI files not found on disk — skipped.")
        self.df = self.df[exists].reset_index(drop=True)

        print(
            f"[Dataset] split={split!r}  |  samples={len(self.df)}  |  "
            f"num_slices={num_slices}  |  image_size={image_size}  |  "
            f"HU=[{hu_min}, {hu_max}]"
        )

    def _load_and_preprocess(self, path: str) -> np.ndarray:
        """Returns float32 ndarray (num_slices, image_size, image_size) in [0, 1]."""
        nii = nib.load(path)
        vol = nii.get_fdata(dtype=np.float32)       # (X, Y, Z)

        # (X, Y, Z) -> (Z, Y, X)
        vol = np.transpose(vol, (2, 1, 0))

        # 3-D cubic zoom on raw HU — matches notebook zoom() call
        factors = (
            self.num_slices / vol.shape[0],
            self.image_size / vol.shape[1],
            self.image_size / vol.shape[2],
        )
        vol = zoom(vol, factors, order=3)           # (num_slices, H, W)

        # Clip AFTER zoom, then normalise
        vol = np.clip(vol, self.hu_min, self.hu_max)
        vol = (vol - self.hu_min) / (self.hu_max - self.hu_min)

        return vol                                   # float32, [0, 1]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row      = self.df.iloc[idx]
        study_id = str(row["study_id"])
        findings = str(row["findings"])

        vol = self._load_and_preprocess(
            os.path.join(self.data_dir, f"{study_id}.nii.gz")
        )                                            # (Z, H, W), [0, 1]

        # Grayscale -> 3 channels: (Z, 3, H, W)
        vol_3ch = np.stack([vol, vol, vol], axis=1)
        slices  = torch.from_numpy(vol_3ch).float()

        # SigLIP normalisation: [0, 1] -> [-1, 1]
        slices = (slices - 0.5) / 0.5

        return {
            "study_id": study_id,
            "slices":   slices,     # (num_slices, 3, H, W)
            "findings": findings,   # raw string — tokenised in training loop
        }


def merlin_collate_fn(batch: list) -> dict:
    """Stack slices into a tensor; keep findings as a list of strings."""
    return {
        "study_ids": [b["study_id"] for b in batch],
        "slices":    torch.stack([b["slices"] for b in batch]),
        # shape: (B, num_slices, 3, H, W)
        "findings":  [b["findings"] for b in batch],
    }
