"""
app.py -- interactive demo for SATT (Slice-Aware Temporal Tokenization).

Pick one of the demo CT scans (or upload a .nii.gz), see what the model sees,
and generate a radiology report in the browser. For demo scans from the test
set, the radiologist's report is shown next to the generated one.

Research prototype. NOT a medical device and not for clinical use.

Run from Phase_2/ (same folder as inference.py, model.py, eval_paths.py):
    python app.py
The browser opens at http://127.0.0.1:7860 once the model has loaded
(about a minute). Each report then takes about 13-26 s on an RTX 4060.

Options:
    --scans_dir demo_scans               folder of .nii.gz files for the dropdown
    --refs demo_scans/demo_refs.csv      CSV with study_id, prediction, reference
    --checkpoint_dir ./checkpoints
    --port 7860
Works offline: start it with  HF_HUB_OFFLINE=1 python app.py
"""

import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # no telemetry calls

import argparse
import glob
import re
import threading

import numpy as np
import nibabel as nib
import pandas as pd
import gradio as gr

from inference import SATTInference, DISCLAIMER, HU_MIN, HU_MAX, compare_with_saved


# ---------------------------------------------------------------------------
# Display helpers (they change only how text and images are shown, never the
# model input or the generated text itself)
# ---------------------------------------------------------------------------

SECTION_HEADERS = [
    "FINDINGS", "IMPRESSION", "Lower thorax", "Liver and biliary tree",
    "Gallbladder", "Spleen", "Pancreas", "Adrenal glands", "Kidneys and ureters",
    "Gastrointestinal tract", "Bowel", "Peritoneal cavity", "Bladder",
    "Prostate and seminal vesicles", "Uterus and ovaries", "Uterus", "Ovaries",
    "Vasculature", "Lymph nodes", "Abdominal wall", "Musculoskeletal",
]
_HEADER_RE = re.compile(
    r"\s*\b(" + "|".join(re.escape(h) for h in sorted(SECTION_HEADERS, key=len, reverse=True))
    + r"):\s*")
_NUMBERED_RE = re.compile(r"\s+(\d{1,2})\.\s+")


def format_report(text: str) -> str:
    """Put each organ section and each impression item on its own line."""
    t = " ".join(str(text).split())
    if not t:
        return ""
    t = _HEADER_RE.sub(lambda m: "\n" + m.group(1) + ": ", t)
    t = _NUMBERED_RE.sub(lambda m: "\n" + m.group(1) + ". ", t)
    t = t.strip()
    t = re.sub(r"^FINDINGS:\s*", "FINDINGS:\n", t)
    t = re.sub(r"\nIMPRESSION:\s*", "\n\nIMPRESSION:\n", t)
    return t.strip()


def scan_id(path: str) -> str:
    name = os.path.basename(str(path))
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def make_preview(path: str, n: int = 6):
    """
    Montage of n axial slices from the scan, windowed to the model's HU range
    [-200, 300]. Shown in radiological convention (anterior up, patient's right
    on the left), most superior slice first. Display only: the model itself
    uses inference.preprocess_nifti, unchanged.
    """
    img = nib.as_closest_canonical(nib.load(str(path)))      # RAS orientation
    data = np.asanyarray(img.dataobj)                         # (X, Y, Z)
    Z = data.shape[2]
    zs = np.linspace(0.15 * (Z - 1), 0.85 * (Z - 1), n).round().astype(int)[::-1]
    tiles = []
    for z in zs:
        s = data[:, :, z].astype(np.float32)
        s = np.clip((s - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0)
        s = s[::-1, ::-1].T                                   # anterior up, R on left
        step = max(1, s.shape[0] // 256)
        tiles.append((s[::step, ::step] * 255).astype(np.uint8))
    h, w = tiles[0].shape
    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    gap = 4
    grid = np.zeros((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * (h + gap): r * (h + gap) + h, c * (w + gap): c * (w + gap) + w] = t
    zooms = img.header.get_zooms()[:3]
    info = (f"**{scan_id(path)}**: {data.shape[0]} x {data.shape[1]} x {Z} voxels, "
            f"voxel size {zooms[0]:.2f} x {zooms[1]:.2f} x {zooms[2]:.2f} mm. "
            f"Six axial slices from top (upper abdomen) to bottom (pelvis), window "
            f"[{HU_MIN:.0f}, {HU_MAX:.0f}] HU. The model sees 64 slices like these.")
    return np.stack([grid] * 3, axis=-1), info


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

ABOUT_MD = """
### How a report is generated
1. **Preprocessing.** The CT volume is resampled to **64 axial slices of 224 x 224** and windowed to **[-200, 300] HU**.
2. **Vision encoder (frozen).** SigLIP ViT-B/16 turns each slice into 196 patch features: **64 x 196 x 768**.
3. **SATT adapter (trained).** Groups of **4 consecutive slices** are averaged into 16 chunks. A 2-layer Transformer
   then lets the **16 x 196 = 3,136 tokens** attend to each other across chunks, and an MLP projects them
   to Llama's 3,072-dimensional embedding space.
4. **Language model.** Llama 3.2-3B-Instruct (4-bit NF4) with a trained **LoRA** adapter reads the 3,136 visual
   tokens plus a short instruction and writes the report (greedy decoding, repetition penalty 1.1, at most 600 tokens).

### Test-set results (1,000 Merlin test scans)
| Adapter | RadGraph-F1 | BLEU-4 | ROUGE-L | METEOR |
|---|---|---|---|---|
| Mean-Pool (baseline) | 0.312 | 10.39 | 0.283 | 0.258 |
| ChunkPool c=4 (SATT without the Transformer) | 0.308 | 11.85 | 0.277 | 0.275 |
| **SATT c=4 (this demo)** | **0.331** | **14.20** | **0.291** | **0.286** |

### Limitations
Generated reports are fluent and well structured but **omit most findings**, can **contradict** the radiologist
(for example, "no evidence of recurrent disease" when metastases are present), and can include details that
are not supported by the scan. Each model was trained once, so the confidence intervals reflect test-scan sampling only.
"""


def build_app(engine, scans, refs):
    gpu_lock = threading.Lock()
    choices = [(f"{scan_id(p)}{'   (test set, reference available)' if scan_id(p) in refs else ''}", p)
               for p in scans]
    default = next((p for p in scans if scan_id(p) == "AC423da3a"), scans[0] if scans else None)

    def pick(choice, upload):
        if upload:
            return str(upload)
        return choice

    def on_input(choice, upload):
        path = pick(choice, upload)
        if not path:
            return None, "Choose a demo scan or upload a .nii.gz file.", "", "", "", ""
        try:
            image, info = make_preview(path)
        except Exception as e:                       # bad or unsupported file
            return None, f"Could not read this file as a NIfTI volume: {e}", "", "", "", ""
        return image, info, "", "", "", ""

    def on_generate(choice, upload):
        path = pick(choice, upload)
        if not path:
            raise gr.Error("Choose a demo scan or upload a .nii.gz file first.")
        if not str(path).endswith((".nii.gz", ".nii")):
            raise gr.Error("Please upload a NIfTI file (.nii.gz or .nii).")
        sid = scan_id(path)
        yield "", "", f"Generating the report for **{sid}** ... about 15-25 s.", ""

        with gpu_lock:                               # one scan on the GPU at a time
            try:
                report = engine.generate_report(str(path), verbose=True)
            except Exception as e:
                raise gr.Error(f"Generation failed: {e}")
        s = engine.last_stats
        gpu = f" | peak GPU memory {s['peak_gpu_gb']} GB" if s.get("peak_gpu_gb") else ""
        stats = (f"**{sid}** | {s['total_seconds']} s total ({s['preprocess_seconds']} s "
                 f"preprocessing) | {s['generated_tokens']} tokens{gpu} | model: SATT c=4")

        if sid in refs:
            row = refs[sid]
            reference = format_report(row["reference"])
            same, frac = compare_with_saved(report, row["prediction"])
            if same:
                repro = "Reproduction check: **identical** to the report saved by the paper's test-set evaluation."
            else:
                repro = (f"Reproduction check: matches the paper's saved report for the first "
                         f"{frac:.0%} of the text, then differs in small details "
                         f"(numerical differences between GPUs under greedy decoding).")
        else:
            reference = "No radiologist report is available for this scan (it is not one of the demo test scans)."
            repro = ""
        yield format_report(report), reference, stats, repro

    with gr.Blocks(title="SATT: CT to Radiology Report") as demo:
        gr.Markdown(
            "# SATT: Abdominal CT to Radiology Report\n"
            "Slice-Aware Temporal Tokenization for 3D CT report generation "
            "(SigLIP + SATT adapter + Llama 3.2-3B with LoRA, trained on Merlin).")
        gr.Markdown(f"> **{DISCLAIMER}**")

        with gr.Row():
            with gr.Column(scale=5):
                scan_dd = gr.Dropdown(choices=choices, value=default,
                                      label="Demo scan (Merlin test set)")
                upload = gr.File(label="... or upload a CT scan (.nii.gz)", type="filepath",
                                 file_count="single")
                run_btn = gr.Button("Generate report", variant="primary")
                preview = gr.Image(label="What the model sees (6 of 64 slices)",
                                   type="numpy", interactive=False, height=420)
                preview_info = gr.Markdown()
            with gr.Column(scale=7):
                status = gr.Markdown()
                with gr.Row():
                    gen_box = gr.Textbox(label="Generated report (SATT c=4)", lines=22,
                                         max_lines=40, interactive=False)
                    ref_box = gr.Textbox(label="Radiologist's report (reference)", lines=22,
                                         max_lines=40, interactive=False)
                repro_md = gr.Markdown()

        with gr.Accordion("How it works, results and limitations", open=False):
            gr.Markdown(ABOUT_MD)

        outs_preview = [preview, preview_info, gen_box, ref_box, status, repro_md]
        scan_dd.change(on_input, [scan_dd, upload], outs_preview)
        upload.change(on_input, [scan_dd, upload], outs_preview)
        demo.load(on_input, [scan_dd, upload], outs_preview)
        run_btn.click(on_generate, [scan_dd, upload], [gen_box, ref_box, status, repro_md])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scans_dir", default="demo_scans")
    ap.add_argument("--refs", default="demo_scans/demo_refs.csv")
    ap.add_argument("--checkpoint_dir", default="./checkpoints")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    scans = sorted(glob.glob(os.path.join(args.scans_dir, "*.nii.gz")))
    print(f"[app] {len(scans)} demo scans in {args.scans_dir}/")
    refs = {}
    if os.path.exists(args.refs):
        df = pd.read_csv(args.refs).fillna("")
        refs = {str(r["study_id"]): {"reference": r["reference"], "prediction": r["prediction"]}
                for _, r in df.iterrows()}
        print(f"[app] {len(refs)} reference reports from {args.refs}")
    else:
        print(f"[app] {args.refs} not found: reference reports will not be shown")

    engine = SATTInference(checkpoint_dir=args.checkpoint_dir, variant="satt_chunk4")
    demo = build_app(engine, scans, refs)
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port,
                        inbrowser=True, share=False)


if __name__ == "__main__":
    main()
