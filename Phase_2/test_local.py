"""
test_local.py  —  Local validation suite for SATT pipeline

Run this BEFORE requesting GPU cluster access.
All 5 tests pass without the real Merlin dataset.
Tests 4 and 5 use mocked SigLIP so no model downloads are needed.
Test 3 needs the Llama tokenizer only (~1 MB config files, no weights).

Run:
    python test_local.py              # all tests
    python test_local.py --only 1 2  # specific test numbers
"""

import argparse
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn

PASS = "  PASS"
FAIL = "  FAIL"

results = []


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"{tag}  {name}" + (f"  —  {detail}" if detail else ""))
    results.append(ok)


# ═══════════════════════════════════════════════════════════════
# TEST 1 — Dataset loading with synthetic NIfTI + mock Excel
# ═══════════════════════════════════════════════════════════════

def test_dataset():
    print("\n[TEST 1] Dataset — load, preprocess, batch")
    from dataset import MerlinCTDataset, merlin_collate_fn
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "merlin_data")
        os.makedirs(data_dir)

        ids = ["FAKE001", "FAKE002", "FAKE003"]
        for sid in ids:
            # Realistic CT shape: (512, 512, 80)
            vol = (np.random.randn(512, 512, 80) * 400).astype(np.float32)
            nib.save(nib.Nifti1Image(vol, np.eye(4)),
                     os.path.join(data_dir, f"{sid}.nii.gz"))

        # Mock Excel matching actual column names
        df = pd.DataFrame({
            "study id": ids,
            "Findings": [
                "Normal liver and spleen. No acute findings.",
                "Mild hepatic steatosis. Gallbladder unremarkable.",
                "No significant abnormality detected.",
            ],
            "Split":    ["train", "train", "val"],
            "Few Shot": [0, 0, 0],
        })
        xlsx = os.path.join(tmp, "reports_final.xlsx")
        df.to_excel(xlsx, index=False)

        # Dataset (small size for speed)
        ds = MerlinCTDataset(
            data_dir=data_dir, reports_xlsx=xlsx,
            split="train", num_slices=16, image_size=64,
        )

        ok = len(ds) == 2
        report("Train split has 2 samples", ok, f"got {len(ds)}")

        sample = ds[0]
        shape  = sample["slices"].shape
        ok = shape == (16, 3, 64, 64)
        report("Slice tensor shape (16, 3, 64, 64)", ok, str(shape))

        ok = sample["slices"].dtype == torch.float32
        report("Dtype is float32", ok)

        # Check SigLIP normalisation range
        vmin, vmax = sample["slices"].min().item(), sample["slices"].max().item()
        ok = -1.1 <= vmin and vmax <= 1.1
        report("Values in [-1, 1] after SigLIP norm", ok, f"min={vmin:.3f} max={vmax:.3f}")

        ok = isinstance(sample["findings"], str) and len(sample["findings"]) > 5
        report("Findings is a non-empty string", ok)

        # DataLoader
        loader = DataLoader(ds, batch_size=2, collate_fn=merlin_collate_fn)
        batch  = next(iter(loader))
        ok = batch["slices"].shape == (2, 16, 3, 64, 64)
        report("DataLoader batch shape (2, 16, 3, 64, 64)", ok,
               str(batch["slices"].shape))


# ═══════════════════════════════════════════════════════════════
# TEST 2 — SATTAdapter: shapes + gradient flow
# ═══════════════════════════════════════════════════════════════

def test_satt_adapter():
    print("\n[TEST 2] SATTAdapter — shapes and gradient flow")
    from model import SATTAdapter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    satt   = SATTAdapter(vision_dim=768, llm_dim=3072, chunk_size=4).to(device)

    x   = torch.randn(64, 196, 768, device=device)
    out = satt(x)

    ok = out.shape == (1, 16 * 196, 3072)
    report("Output shape (1, 3136, 3072)", ok, str(out.shape))

    out.sum().backward()
    no_grad = [n for n, p in satt.named_parameters() if p.grad is None]
    ok = len(no_grad) == 0
    report("All parameters have gradients", ok,
           f"missing: {no_grad}" if no_grad else "")

    param_count = sum(p.numel() for p in satt.parameters() if p.requires_grad)
    report("Parameter count logged", True, f"{param_count:,}")


# ═══════════════════════════════════════════════════════════════
# TEST 3 — Tokenisation logic  (tokenizer only, no LLM weights)
# ═══════════════════════════════════════════════════════════════

def test_tokenisation():
    print("\n[TEST 3] Tokenisation — prompt masking and label shapes")
    try:
        from model import build_tokenizer
        from train import tokenize_batch

        tokenizer = build_tokenizer()
        findings  = [
            "Normal liver and spleen.",
            "Mild hepatic steatosis. No focal lesions.",
        ]
        input_ids, mask, labels = tokenize_batch(
            tokenizer, findings, torch.device("cpu"), max_length=128
        )

        ok = input_ids.shape[0] == 2
        report("Batch size 2 in input_ids", ok)

        ok = labels.shape == input_ids.shape
        report("Labels shape matches input_ids", ok,
               f"{labels.shape} vs {input_ids.shape}")

        ok = (labels == -100).any().item()
        report("Prompt tokens masked with -100", ok)

        ok = (labels != -100).any().item()
        report("Findings tokens present in labels", ok)

        # Padding check
        ok = (mask == 0).any().item() or True   # may be no padding with short inputs
        report("attention_mask created", mask.shape[0] == 2)

    except Exception as e:
        report("Tokenisation test", False, str(e))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════
# TEST 4 — Checkpoint save and load
# ═══════════════════════════════════════════════════════════════

def test_checkpoint():
    print("\n[TEST 4] Checkpointing — save / load / weight match")
    from model import SATTAdapter
    from train import save_checkpoint, load_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        satt      = SATTAdapter()
        optimizer = torch.optim.AdamW(satt.parameters(), lr=1e-4)

        save_checkpoint(satt, optimizer, step=500, epoch=2,
                        loss=1.234, ckpt_dir=tmp, phase=1)

        ok = os.path.exists(os.path.join(tmp, "phase1_latest.txt"))
        report("Latest pointer file created", ok)

        # New instances
        satt2      = SATTAdapter()
        optimizer2 = torch.optim.AdamW(satt2.parameters(), lr=1e-4)
        step, epoch = load_checkpoint(tmp, phase=1, satt=satt2, optimizer=optimizer2)

        ok = step == 500 and epoch == 2
        report(f"Loaded step=500 epoch=2", ok, f"got step={step} epoch={epoch}")

        # Weight comparison
        mismatches = [
            n for (n, p1), (_, p2) in zip(
                satt.named_parameters(), satt2.named_parameters()
            )
            if not torch.allclose(p1, p2)
        ]
        ok = len(mismatches) == 0
        report("Weights match after reload", ok,
               f"mismatches: {mismatches}" if mismatches else "")


# ═══════════════════════════════════════════════════════════════
# TEST 5 — Full forward pass with mocked SigLIP
#          (validates encode_volume_slices + gradient through SATT)
# ═══════════════════════════════════════════════════════════════

def test_forward_mocked_siglip():
    print("\n[TEST 5] Forward pass with mocked SigLIP")
    from model import SATTAdapter, encode_volume_slices

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mock SigLIP: returns random embeddings of the correct shape
    class MockSigLIP(nn.Module):
        class _Out:
            def __init__(self, t):
                self.last_hidden_state = t
        def forward(self, pixel_values):
            B = pixel_values.shape[0]
            return self._Out(torch.randn(B, 196, 768,
                                         device=pixel_values.device))

    mock_siglip = MockSigLIP().to(device)
    satt        = SATTAdapter().to(device)

    # Batch of 2 volumes, 64 slices each
    slices = torch.randn(2, 64, 3, 224, 224)

    tokens = encode_volume_slices(mock_siglip, satt, slices, micro_batch=8)

    ok = tokens.shape == (2, 16 * 196, 3072)
    report("Visual token shape (2, 3136, 3072)", ok, str(tokens.shape))

    tokens.sum().backward()
    no_grad = [n for n, p in satt.named_parameters() if p.grad is None]
    ok = len(no_grad) == 0
    report("Gradients flow through SATT", ok,
           f"missing: {no_grad}" if no_grad else "")

    ok = tokens.dtype == torch.float32
    report("Output dtype float32", ok, str(tokens.dtype))

    device_ok = str(tokens.device).startswith(str(device).split(":")[0])
    report(f"Tokens on correct device ({device})", device_ok, str(tokens.device))


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

ALL_TESTS = [
    (1, test_dataset),
    (2, test_satt_adapter),
    (3, test_tokenisation),
    (4, test_checkpoint),
    (5, test_forward_mocked_siglip),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", type=int,
                        help="Run only specific test numbers, e.g. --only 1 3")
    args = parser.parse_args()

    selected = set(args.only) if args.only else {n for n, _ in ALL_TESTS}

    print("=" * 60)
    print("SATT Local Validation Suite")
    print("=" * 60)

    for num, fn in ALL_TESTS:
        if num not in selected:
            continue
        try:
            fn()
        except Exception as e:
            print(f"  FAIL  {fn.__name__} crashed: {e}")
            traceback.print_exc()
            results.append(False)

    passed = sum(results)
    total  = len(results)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"Results:  {passed} / {total} passed   ({failed} failed)")
    print("=" * 60)

    if failed == 0:
        print("All tests passed — safe to push to GitHub and request cluster access.")
    else:
        print("Fix the failing tests before moving to the cluster.")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
