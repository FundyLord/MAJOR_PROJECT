"""
dataset.py  —  Merlin Abdominal CT Dataset Loader
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom


class MerlinCTDataset(Dataset):
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
        self.df = (
            df[df["split"].str.strip().str.lower() == split.lower()]
            .reset_index(drop=True)
        )

        # Auto-detect .npy cache FIRST
        npy_dir = data_dir.rstrip("/") + "_npy"
        self.use_npy = os.path.isdir(npy_dir)
        self.npy_dir = npy_dir

        # Filter missing files — check .npy if cache exists, .nii.gz otherwise
        # This correctly catches files that failed preprocessing
        if self.use_npy:
            exists = self.df["study_id"].apply(
                lambda sid: os.path.exists(
                    os.path.join(npy_dir, f"{sid}.npy")
                )
            )
        else:
            exists = self.df["study_id"].apply(
                lambda sid: os.path.exists(
                    os.path.join(data_dir, f"{sid}.nii.gz")
                )
            )

        n_missing = int((~exists).sum())
        if n_missing:
            print(f"[Dataset] WARNING: {n_missing} files not found on disk — skipped.")
        self.df = self.df[exists].reset_index(drop=True)

        if self.use_npy:
            print(f"[Dataset] .npy cache found  →  fast path ({npy_dir})")
        else:
            print(f"[Dataset] No .npy cache  →  slow path (raw .nii.gz + zoom)")

        print(
            f"[Dataset] split={split!r}  |  samples={len(self.df)}  |  "
            f"num_slices={num_slices}  |  image_size={image_size}  |  "
            f"HU=[{hu_min}, {hu_max}]"
        )

    def _load_and_preprocess(self, path: str) -> np.ndarray:
        nii = nib.load(path)
        vol = nii.get_fdata(dtype=np.float32)
        vol = np.transpose(vol, (2, 1, 0))
        factors = (
            self.num_slices / vol.shape[0],
            self.image_size / vol.shape[1],
            self.image_size / vol.shape[2],
        )
        vol = zoom(vol, factors, order=3)
        vol = np.clip(vol, self.hu_min, self.hu_max)
        vol = (vol - self.hu_min) / (self.hu_max - self.hu_min)
        return vol

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row      = self.df.iloc[idx]
        study_id = str(row["study_id"])
        findings = str(row["findings"])

        if self.use_npy:
            vol = np.load(os.path.join(self.npy_dir, f"{study_id}.npy"))
        else:
            vol = self._load_and_preprocess(
                os.path.join(self.data_dir, f"{study_id}.nii.gz")
            )

        vol_3ch = np.stack([vol, vol, vol], axis=1)
        slices  = torch.from_numpy(vol_3ch).float()
        slices  = (slices - 0.5) / 0.5

        return {
            "study_id": study_id,
            "slices":   slices,
            "findings": findings,
        }


def merlin_collate_fn(batch: list) -> dict:
    return {
        "study_ids": [b["study_id"] for b in batch],
        "slices":    torch.stack([b["slices"] for b in batch]),
        "findings":  [b["findings"] for b in batch],
    }
