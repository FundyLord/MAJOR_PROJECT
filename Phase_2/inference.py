"""
inference.py  --  SATT inference engine: CT scan (.nii.gz) -> radiology report

Research prototype. NOT a medical device and not for clinical use. The model
omits most of the findings a radiologist reports, and it sometimes states
things that are wrong or cannot be known from a single scan (see the paper).

Decoding matches the paper's test-set evaluation exactly: greedy search,
repetition penalty 1.1, no n-gram blocking, at most 600 new tokens. (The
earlier version of this file used 4-gram blocking, which the paper shows
produces stereotyped, partly unsupported reports.)

Command line (run from Phase_2/):
    python inference.py demo_scans/AC42136e9.nii.gz
    python inference.py demo_scans/*.nii.gz --compare_csv demo_scans/demo_refs.csv
    python inference.py scan.nii.gz --variant baseline        # Mean-Pool baseline

--compare_csv takes a CSV with columns study_id, prediction, reference (such as
eval_raw_satt_chunk4.csv from the evaluation). For each scan it prints the
radiologist's report and the report saved by the paper's evaluation, and says
whether this run reproduced it.

From Python (the model is loaded once, then reused for any number of scans):
    from inference import SATTInference
    engine = SATTInference(checkpoint_dir="./checkpoints")
    report = engine.generate_report("demo_scans/AC42136e9.nii.gz")
    print(engine.last_stats)        # seconds, peak GPU memory, token count

Checkpoint files (in --checkpoint_dir), resolved by eval_paths.py so adapter
and LoRA always come from the same training run:
    SATT c=4 (default) : phase2_best_satt.pt + phase2_best_lora/
    other variants     : phase2_best_satt_<variant>.pt + phase2_best_lora_<variant>/
Keep this file in the same folder as model.py and eval_paths.py.
"""

import argparse
import os
import sys
import time

import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

from model import (
    SATTAdapter,
    MeanPoolAdapter,
    ChunkPoolAdapter,
    build_vision_encoder,
    build_tokenizer,
    encode_volume_slices,
    LLAMA_ID,
)
from eval_paths import resolve_checkpoints

# -- Must match training preprocessing exactly --------------------------------
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

# -- Decoding used for every test-set result in the paper ----------------------
DEFAULT_MAX_NEW_TOKENS     = 600
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_NO_REPEAT_NGRAM    = 0

DISCLAIMER = ("Research prototype. Not for clinical use. Generated reports omit "
              "most findings and may contain errors.")

# variant name -> (model_type, chunk_size)
VARIANTS = {
    "satt_chunk4":      ("satt", 4),
    "satt_chunk2":      ("satt", 2),
    "satt_chunk8":      ("satt", 8),
    "chunkpool_chunk4": ("chunkpool", 4),
    "baseline":         ("baseline", None),
}


def build_adapter(model_type, chunk_size):
    if model_type == "baseline":
        return MeanPoolAdapter()
    if model_type == "chunkpool":
        return ChunkPoolAdapter(chunk_size=chunk_size)
    return SATTAdapter(chunk_size=chunk_size)


def preprocess_nifti(path: str, verbose: bool = True) -> torch.Tensor:
    """
    Load a .nii.gz file and apply the exact preprocessing used in training.
    Returns: (64, 3, 224, 224) float32 tensor, values in [-1, 1]
    """
    log = print if verbose else (lambda *a, **k: None)
    log(f"  Loading NIfTI: {path}")
    nii = nib.load(path)
    vol = nii.get_fdata(dtype=np.float32)       # (X, Y, Z)
    log(f"  Original shape: {vol.shape}")

    vol = np.transpose(vol, (2, 1, 0))           # (Z, Y, X)
    factors = (
        NUM_SLICES / vol.shape[0],
        IMAGE_SIZE / vol.shape[1],
        IMAGE_SIZE / vol.shape[2],
    )
    vol = zoom(vol, factors, order=3)            # (64, 224, 224)
    vol = np.clip(vol, HU_MIN, HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)    # [0, 1]

    vol_3ch = np.stack([vol, vol, vol], axis=1)  # (64, 3, 224, 224)
    slices  = torch.from_numpy(vol_3ch).float()
    slices  = (slices - 0.5) / 0.5              # [-1, 1]
    log(f"  Preprocessed shape: {tuple(slices.shape)}  "
        f"range: [{slices.min():.2f}, {slices.max():.2f}]")
    return slices


class SATTInference:
    """End-to-end inference engine: CT scan (.nii.gz) -> radiology report (str)."""

    def __init__(self, checkpoint_dir: str = "./checkpoints",
                 variant: str = "satt_chunk4"):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(VARIANTS)}")
        self.variant = variant
        model_type, chunk_size = VARIANTS[variant]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.last_stats = {}

        adapter_path, lora_dir, source = resolve_checkpoints(
            checkpoint_dir, model_type, chunk_size if chunk_size else 4)
        print(f"\n[SATTInference] Variant: {variant}   Device: {self.device}")
        print(f"[SATTInference] Weights: {source}")
        print("[SATTInference] Loading models -- about a minute on first run...\n")

        # -- SigLIP vision encoder (frozen) -------------------------------------
        print("  Loading SigLIP...")
        self.vision_encoder = build_vision_encoder().to(self.device)
        self.vision_encoder.eval()

        # -- Adapter: the trained weights ---------------------------------------
        print("  Loading adapter...")
        self.adapter = build_adapter(model_type, chunk_size).to(self.device)
        ckpt = torch.load(adapter_path, map_location=self.device, weights_only=False)
        rec_type, rec_chunk = ckpt.get("model_type"), ckpt.get("chunk_size")
        if rec_type is not None and rec_type != model_type:
            raise SystemExit(f"{adapter_path} was trained as '{rec_type}', not '{model_type}'.")
        if (rec_chunk is not None and chunk_size is not None
                and int(rec_chunk) != int(chunk_size)):
            raise SystemExit(f"{adapter_path} has chunk_size={rec_chunk}, not {chunk_size}.")
        self.adapter.load_state_dict(ckpt["satt_state"])
        self.adapter.eval()
        self.val_loss = ckpt.get("loss")
        vl = f"{self.val_loss:.4f}" if self.val_loss is not None else "n/a"
        print(f"  Adapter ready  (val_loss={vl}, epoch={ckpt.get('epoch')})")

        # -- Tokenizer ------------------------------------------------------------
        self.tokenizer = build_tokenizer()

        # -- Llama 4-bit + trained LoRA adapter -------------------------------------
        print("  Loading Llama 3.2-3B in 4-bit NF4...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_llm = AutoModelForCausalLM.from_pretrained(
            LLAMA_ID, quantization_config=bnb_config, device_map="auto")
        print(f"  Applying LoRA adapter from {lora_dir}...")
        self.llm = PeftModel.from_pretrained(base_llm, lora_dir)
        self.llm.eval()
        self.llm_device = next(self.llm.parameters()).device
        print("\n[SATTInference] All models loaded. Ready to generate reports.\n")

    @torch.no_grad()
    def generate_report(
        self,
        nifti_path: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM,
        verbose: bool = True,
    ) -> str:
        """Generate a radiology report for one CT scan (.nii.gz)."""
        log = print if verbose else (lambda *a, **k: None)
        log(f"\n{'=' * 60}\nProcessing: {os.path.basename(nifti_path)}\n{'=' * 60}")
        use_cuda = self.device == "cuda"
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        slices = preprocess_nifti(nifti_path, verbose)           # (64, 3, 224, 224)
        t_pre = time.time()

        log("  Running SigLIP + adapter encoding...")
        visual_tokens = encode_volume_slices(
            self.vision_encoder, self.adapter, slices.unsqueeze(0), micro_batch=8
        ).to(self.llm_device)
        log(f"  Visual tokens: {tuple(visual_tokens.shape)}")

        prompt_ids = self.tokenizer(
            SYSTEM_PROMPT, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.llm_device)
        text_embeds   = self.llm.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([visual_tokens.to(text_embeds.dtype), text_embeds], dim=1)
        attn_mask = torch.ones(1, inputs_embeds.shape[1],
                               device=self.llm_device, dtype=torch.long)

        log(f"  Generating report (greedy, repetition penalty {repetition_penalty}, "
            f"n-gram blocking {no_repeat_ngram_size})...")
        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.time()

        self.last_stats = {
            "scan": os.path.basename(nifti_path),
            "variant": self.variant,
            "preprocess_seconds": round(t_pre - t0, 1),
            "total_seconds": round(t1 - t0, 1),
            "generated_tokens": int(output_ids.shape[1]),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if use_cuda else None,
        }
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


# -- Comparison with the paper's saved output ----------------------------------------

def _norm(text: str) -> str:
    return " ".join(str(text).split())


def compare_with_saved(report: str, saved: str):
    """Return (identical, shared_prefix_fraction) after whitespace normalisation."""
    a, b = _norm(report), _norm(saved)
    if a == b:
        return True, 1.0
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return False, n / max(len(a), len(b), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scans", nargs="+", help="one or more .nii.gz CT volumes")
    ap.add_argument("--variant", default="satt_chunk4", choices=list(VARIANTS))
    ap.add_argument("--checkpoint_dir", default="./checkpoints")
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--repetition_penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=DEFAULT_NO_REPEAT_NGRAM)
    ap.add_argument("--compare_csv", default=None,
                    help="CSV with study_id, prediction, reference columns")
    ap.add_argument("--out_dir", default=None, help="also save each report as a .txt file here")
    args = ap.parse_args()

    ref_rows = {}
    if args.compare_csv:
        import pandas as pd
        df = pd.read_csv(args.compare_csv).fillna("")
        ref_rows = {str(r.study_id): r for r in df.itertuples()}
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    engine = SATTInference(checkpoint_dir=args.checkpoint_dir, variant=args.variant)

    for path in args.scans:
        sid = os.path.basename(path).replace(".nii.gz", "")
        report = engine.generate_report(
            path,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        bar = "=" * 60
        print(f"\n{bar}\nGENERATED REPORT  ({sid})\n{bar}\n{report}\n{bar}")
        print(f"[{engine.last_stats}]")
        print(f"[{DISCLAIMER}]")

        if sid in ref_rows:
            row = ref_rows[sid]
            print(f"\n{bar}\nRADIOLOGIST'S REPORT (reference)  ({sid})\n{bar}\n{row.reference}\n{bar}")
            same, frac = compare_with_saved(report, row.prediction)
            if same:
                print("Reproduction check: IDENTICAL to the report saved by the paper's evaluation.")
            else:
                print(f"Reproduction check: differs from the paper's saved report "
                      f"(first {frac:.0%} of the text is identical). Small differences are "
                      f"possible across GPUs and library versions; the paper's report is:\n"
                      f"{row.prediction}")
        if args.out_dir:
            with open(os.path.join(args.out_dir, f"{sid}.txt"), "w") as f:
                f.write(report + "\n")


if __name__ == "__main__":
    main()

# Execution command: 
# python inference.py ./sample_scans/AC42413f4.nii.gz
# python inference.py ./sample_scans/AC42136e9.nii.gz    