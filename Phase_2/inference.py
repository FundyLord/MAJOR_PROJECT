"""
inference.py  —  Standalone SATT inference engine (laptop/RTX 4060)

Usage:
    # Single scan from command line:
    python inference.py ../sample_scans/AC42413f4.nii.gz

    # Or import and use in code:
    from inference import SATTInference
    engine = SATTInference(checkpoint_dir="./checkpoints")
    report = engine.generate_report("../sample_scans/AC42413f4.nii.gz")
    print(report)

Keep this file in the same folder as model.py (Phase_2/).
Checkpoints folder should contain:
    phase2_best_satt.pt
    phase2_best_lora/
"""

import os
import sys
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

# model.py must be in the same folder
from model import (
    SATTAdapter,
    build_vision_encoder,
    build_tokenizer,
    encode_volume_slices,
    LLAMA_ID,
)

# ── Must match training preprocessing exactly ────────────────────────────────
NUM_SLICES = 64
IMAGE_SIZE = 224
HU_MIN     = -200.0
HU_MAX     =  300.0

SYSTEM_PROMPT = (
    "<|begin_of_text|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    "Analyze this abdominal CT scan and generate a clinical radiology report.\n"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)


def preprocess_nifti(path: str) -> torch.Tensor:
    """
    Load a .nii.gz file and apply the exact same preprocessing used in training.
    Returns: (64, 3, 224, 224) float32 tensor, values in [-1, 1]
    """
    print(f"  Loading NIfTI: {path}")
    nii = nib.load(path)
    vol = nii.get_fdata(dtype=np.float32)       # (X, Y, Z)
    print(f"  Original shape: {vol.shape}")

    vol = np.transpose(vol, (2, 1, 0))           # (Z, Y, X)

    factors = (
        NUM_SLICES / vol.shape[0],
        IMAGE_SIZE / vol.shape[1],
        IMAGE_SIZE / vol.shape[2],
    )
    print(f"  Zoom factors: {factors[0]:.2f}, {factors[1]:.2f}, {factors[2]:.2f}")
    vol = zoom(vol, factors, order=3)            # (64, 224, 224)
    vol = np.clip(vol, HU_MIN, HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)    # [0, 1]

    vol_3ch = np.stack([vol, vol, vol], axis=1)  # (64, 3, 224, 224)
    slices  = torch.from_numpy(vol_3ch).float()
    slices  = (slices - 0.5) / 0.5              # [-1, 1]
    print(f"  Preprocessed shape: {slices.shape}  range: [{slices.min():.2f}, {slices.max():.2f}]")
    return slices


class SATTInference:
    """End-to-end inference engine: CT scan (.nii.gz) -> radiology report (str)."""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[SATTInference] Device: {self.device}")
        print("[SATTInference] Loading models — this takes ~60s on first run...\n")

        # ── SigLIP vision encoder (frozen) ────────────────────────────
        print("  Loading SigLIP...")
        self.vision_encoder = build_vision_encoder().to(self.device)
        self.vision_encoder.eval()
        print("  SigLIP ready.")

        # ── SATT adapter — your trained weights ───────────────────────
        print("  Loading SATT adapter...")
        self.satt = SATTAdapter().to(self.device)
        satt_path = os.path.join(checkpoint_dir, "phase2_best_satt.pt")
        ckpt = torch.load(satt_path, map_location=self.device, weights_only=False)
        self.satt.load_state_dict(ckpt["satt_state"])
        self.satt.eval()
        print(f"  SATT ready  (val_loss={ckpt['loss']:.4f}, epoch={ckpt['epoch']})")

        # ── Tokenizer ──────────────────────────────────────────────────
        self.tokenizer = build_tokenizer()

        # ── Llama 4-bit + trained LoRA adapter ────────────────────────
        print("  Loading Llama 3.2-3B in 4-bit NF4...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_llm = AutoModelForCausalLM.from_pretrained(
            LLAMA_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )
        lora_dir = os.path.join(checkpoint_dir, "phase2_best_lora")
        print(f"  Applying LoRA adapter from {lora_dir}...")
        self.llm = PeftModel.from_pretrained(base_llm, lora_dir)
        self.llm.eval()
        self.llm_device = next(self.llm.parameters()).device
        print("  Llama + LoRA ready.")
        print("\n[SATTInference] All models loaded. Ready to generate reports.\n")

    @torch.no_grad()
    def generate_report(self, nifti_path: str, max_new_tokens: int = 250) -> str:
        """
        Generate a radiology report for a single CT scan (.nii.gz).
        Returns the generated report as a string.
        """
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(nifti_path)}")
        print(f"{'='*60}")

        # Step 1: Preprocess CT scan
        slices = preprocess_nifti(nifti_path)   # (64, 3, 224, 224)

        # Step 2: Encode through SigLIP + SATT
        print("  Running SigLIP + SATT encoding...")
        visual_tokens = encode_volume_slices(
            self.vision_encoder, self.satt,
            slices.unsqueeze(0), micro_batch=8
        )                                        # (1, 3136, 3072)
        visual_tokens = visual_tokens.to(self.llm_device)
        print(f"  Visual tokens shape: {visual_tokens.shape}")

        # Step 3: Build prompt embeddings
        prompt_ids = self.tokenizer(
            SYSTEM_PROMPT, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.llm_device)
        text_embeds = self.llm.get_input_embeddings()(prompt_ids)

        # Step 4: Concatenate visual + text tokens
        vis_tokens    = visual_tokens.to(text_embeds.dtype)
        inputs_embeds = torch.cat([vis_tokens, text_embeds], dim=1)
        attn_mask     = torch.ones(
            1, inputs_embeds.shape[1],
            device=self.llm_device, dtype=torch.long
        )

        # Step 5: Generate report
        print("  Generating report (greedy decode)...")
        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        report = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return report


if __name__ == "__main__":
    scan_path = sys.argv[1] if len(sys.argv) > 1 else "../sample_scans/AC42413f4.nii.gz"

    # checkpoint_dir relative to Phase_2/
    engine = SATTInference(checkpoint_dir="./checkpoints")
    report = engine.generate_report(scan_path)

    print(f"\n{'='*60}")
    print("GENERATED REPORT:")
    print(f"{'='*60}")
    print(report)
    print(f"{'='*60}\n")

# Execution command: 
# python inference.py ./sample_scans/AC42413f4.nii.gz
# python inference.py ./sample_scans/AC42136e9.nii.gz    