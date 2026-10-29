"""
app.py -- interactive demo for SATT (Slice-Aware Temporal Tokenization).

A scrolling, step-by-step page:
    1. Upload a CT scan (.nii.gz)
    2. 3D view of the scan (rotate / zoom)
    3. 2D slice browser (axial, coronal, sagittal)
    4. What the model sees: 64 slices grouped into 16 chunks of 4
    5. Generate the report with the SATT model
    6. How SATT works, results and limitations

Research prototype. NOT a medical device and not for clinical use.

Run from Phase_2/ (same folder as inference.py, model.py, eval_paths.py):
    HF_HUB_OFFLINE=1 python app.py
The browser opens at http://127.0.0.1:7860 once the model has loaded
(about a minute).

The 3D view needs plotly and scikit-image; without them the rest of the app
still works and the 3D section says what is missing.
"""

import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # no telemetry calls

import argparse
import glob
import html
import inspect
import re
import threading

import numpy as np
import nibabel as nib
import pandas as pd
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from inference import SATTInference, DISCLAIMER, HU_MIN, HU_MAX, compare_with_saved

try:                                    # optional: 3D view
    import plotly.graph_objects as go
    from skimage import measure
    HAVE_3D = True
except Exception:
    HAVE_3D = False

OUT_DIR = os.path.abspath("demo_out")   # downloadable reports (git-ignored)


# =============================================================================
# Volume loading and display helpers (display only: the model itself always
# uses inference.preprocess_nifti, unchanged)
# =============================================================================

_CACHE = {}


def scan_id(path: str) -> str:
    name = os.path.basename(str(path))
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def load_volume(path: str):
    """Scan in RAS orientation as int16 HU, plus voxel size in mm. Cached."""
    path = str(path)
    if path in _CACHE:
        return _CACHE[path]
    img = nib.as_closest_canonical(nib.load(path))
    arr = np.asanyarray(img.dataobj)
    if arr.ndim == 4:
        arr = arr[..., 0]
    vol = np.clip(np.round(arr), -32768, 32767).astype(np.int16)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    _CACHE.clear()                                  # keep only the current scan
    _CACHE[path] = (vol, zooms)
    return vol, zooms


def to_uint8(s):
    s = np.clip((s.astype(np.float32) - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0)
    return (s * 255).astype(np.uint8)


PLANES = {   # plane -> (array axis that is fixed, slider caption)
    "Axial": (2, "feet  →  head"),
    "Coronal": (1, "back  →  front"),
    "Sagittal": (0, "patient's right  →  left"),
}


def render_slice(vol, zooms, plane: str, pos: float, height: int = 520):
    """
    One slice in radiological convention (patient's right on the viewer's left,
    anterior / superior at the top), resized to true physical proportions.
    """
    axis = PLANES[plane][0]
    n = vol.shape[axis]
    idx = int(round(float(pos) / 100.0 * (n - 1)))
    s = np.take(vol, idx, axis=axis)                 # 2D, remaining axes in order
    img = s[::-1, ::-1].T                            # same rule works for all 3 planes
    rest = [a for a in range(3) if a != axis]
    width_mm = vol.shape[rest[0]] * zooms[rest[0]]
    height_mm = vol.shape[rest[1]] * zooms[rest[1]]
    w = max(1, int(round(height * width_mm / height_mm)))
    pil = Image.fromarray(to_uint8(img)).resize((w, height), Image.BILINEAR)
    return np.asarray(pil.convert("RGB")), f"{plane} slice {idx + 1} of {n}"


CHUNK_COLORS = ["#14b8a6", "#0ea5e9", "#6366f1", "#f59e0b"]


def model_input_grid(vol, tile: int = 104):
    """
    The 64 axial slices the model reads (evenly spaced over the scan), grouped
    into the 16 chunks of 4 that SATT pools. Most superior slice first.
    Illustration: the model's own preprocessing uses cubic resampling.
    """
    Z = vol.shape[2]
    zs = np.linspace(0, Z - 1, 64).round().astype(int)[::-1]
    tiles = []
    for z in zs:
        s = vol[:, :, z][::-1, ::-1].T
        tiles.append(Image.fromarray(to_uint8(s)).resize((tile, tile), Image.BILINEAR))

    gap, chunk_gap, band = 3, 18, 22
    per_row = 2                                       # chunks per row
    chunk_w = 4 * tile + 3 * gap
    W = per_row * chunk_w + (per_row - 1) * chunk_gap
    rows = 16 // per_row
    H = rows * (band + tile) + (rows - 1) * chunk_gap
    canvas = Image.new("RGB", (W, H), (12, 16, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for c in range(16):
        r, k = divmod(c, per_row)
        x0 = k * (chunk_w + chunk_gap)
        y0 = r * (band + tile + chunk_gap)
        color = CHUNK_COLORS[c % len(CHUNK_COLORS)]
        draw.rectangle([x0, y0, x0 + chunk_w - 1, y0 + band - 4], fill=color)
        draw.text((x0 + 8, y0 + 4), f"Chunk {c + 1}   slices {4 * c + 1}-{4 * c + 4}",
                  fill=(255, 255, 255), font=font)
        for j in range(4):
            canvas.paste(tiles[4 * c + j], (x0 + j * (tile + gap), y0 + band))
    return np.asarray(canvas)


def _largest_component(mask):
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def _mesh(mask, spacing, sigma, step):
    smooth = ndimage.gaussian_filter(mask.astype(np.float32), sigma)
    verts, faces, _, _ = measure.marching_cubes(smooth, 0.5, spacing=spacing, step_size=step)
    return np.round(verts, 1), faces


def build_3d(vol, zooms):
    """
    Interactive 3D model of the scan: the body surface (translucent) and
    structures brighter than 250 HU (bone, plus contrast-filled vessels and
    kidneys). Built at about 3 mm resolution so it is quick to compute and draw.
    """
    sx = max(1, int(3.0 // zooms[0]))
    sz = max(1, int(3.0 // zooms[2]))
    small = vol[::sx, ::sx, ::sz]
    spacing = (zooms[0] * sx, zooms[1] * sx, zooms[2] * sz)

    body = small > -500
    for z in range(body.shape[2]):                  # fill lungs / bowel gas per slice
        body[:, :, z] = ndimage.binary_fill_holes(body[:, :, z])
    body = _largest_component(body)                 # drops the CT table

    dense = (small > 250) & ndimage.binary_dilation(body, iterations=2)
    lab, n = ndimage.label(dense)
    if n:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= 40
        keep[0] = False
        dense = keep[lab]

    fig = go.Figure()
    light = dict(ambient=0.45, diffuse=0.85, specular=0.35, roughness=0.5, fresnel=0.2)
    if body.any():
        v, f = _mesh(body, spacing, 1.0, 2)
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color="#f0b89a", opacity=0.16, name="Body surface",
                                hoverinfo="skip", lighting=light, showlegend=True))
    if dense.any():
        step = 1 if dense.sum() < 250_000 else 2
        v, f = _mesh(dense, spacing, 0.8, step)
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color="#f5efe0", opacity=1.0, name="Bone and contrast (> 250 HU)",
                                hoverinfo="skip", lighting=light, showlegend=True))
    hidden = dict(visible=False, showbackground=False)
    fig.update_layout(
        height=640, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden, aspectmode="data",
                   bgcolor="rgba(0,0,0,0)",
                   camera=dict(eye=dict(x=0.0, y=1.9, z=0.35), up=dict(x=0, y=0, z=1))),
        legend=dict(x=0.01, y=0.99, font=dict(color="#94a3b8", size=13),
                    bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# =============================================================================
# Report formatting
# =============================================================================

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


def report_to_html(text: str, title: str, eyebrow: str, sid: str, note: str = "") -> str:
    """Render a report as a card: a findings table and a numbered impression."""
    findings, impression, loose = [], [], []
    in_impression = False
    for line in format_report(text).split("\n"):
        line = line.strip()
        if not line or line == "FINDINGS:":
            continue
        if line == "IMPRESSION:":
            in_impression = True
            continue
        m_num = re.match(r"^(\d{1,2})\.\s*(.*)$", line)
        m_hdr = re.match(r"^([A-Za-z ]+):\s*(.*)$", line)
        if m_num:
            impression.append(m_num.group(2))
        elif in_impression:
            impression.append(line)
        elif m_hdr and m_hdr.group(1) in SECTION_HEADERS:
            findings.append((m_hdr.group(1), m_hdr.group(2)))
        else:
            loose.append(line)

    e = html.escape
    parts = [f'<div class="report-card"><div class="report-head"><div>'
             f'<div class="eyebrow">{e(eyebrow)}</div><h3>{e(title)}</h3></div>'
             f'<div class="scan-tag">{e(sid)}</div></div>']
    if findings or loose:
        parts.append('<div class="report-title">Findings</div><table class="findings">')
        for k, v in findings:
            parts.append(f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>")
        for v in loose:
            parts.append(f'<tr><td colspan="2">{e(v)}</td></tr>')
        parts.append("</table>")
    if impression:
        parts.append('<div class="report-title">Impression</div><ol class="impression">')
        parts.extend(f"<li>{e(v)}</li>" for v in impression)
        parts.append("</ol>")
    if note:
        parts.append(f'<div class="report-foot">{note}</div>')
    parts.append("</div>")
    return "".join(parts)


def chips(items):
    return '<div class="chips">' + "".join(
        f'<div class="chip"><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>'
        for k, v in items) + "</div>"


def section_head(num, title, sub):
    return (f'<div class="section-head"><div class="step">{num}</div><h2>{title}</h2></div>'
            f'<p class="section-sub">{sub}</p>')


# =============================================================================
# Page text and style
# =============================================================================

HERO = f"""
<div id="hero">
  <div class="hero-eyebrow">Final-year Major Project &middot; SPIT</div>
  <h1>From a 3D abdominal CT scan<br>to a radiology report</h1>
  <p>SATT (Slice-Aware Temporal Tokenization) reads all 64 slices of a CT scan,
     groups neighbouring slices into chunks, lets the chunks attend to each other,
     and hands the result to a language model that writes the report.</p>
  <div>
    <span class="pill">SigLIP ViT-B/16 (frozen)</span>
    <span class="pill">SATT adapter</span>
    <span class="pill">Llama 3.2-3B + LoRA</span>
    <span class="pill">Trained on Merlin</span>
  </div>
  <div class="warn">&#9888; {html.escape(DISCLAIMER)}</div>
</div>
"""

PIPELINE = """
<div class="pipeline">
  <div class="stage"><div class="stage-k">Input</div><b>CT volume</b>
    <p>Resampled to 64 axial slices of 224&times;224, window [-200, 300] HU</p></div>
  <div class="arrow">&rarr;</div>
  <div class="stage"><div class="stage-k">Frozen</div><b>SigLIP ViT-B/16</b>
    <p>196 patch features per slice: 64&times;196&times;768</p></div>
  <div class="arrow">&rarr;</div>
  <div class="stage trained"><div class="stage-k">Trained (ours)</div><b>SATT adapter</b>
    <p>Average every 4 slices &rarr; 16 chunks; 2-layer Transformer over 3,136 tokens;
       MLP to 3,072-d</p></div>
  <div class="arrow">&rarr;</div>
  <div class="stage trained"><div class="stage-k">4-bit + LoRA</div><b>Llama 3.2-3B</b>
    <p>Reads 3,136 visual tokens + instruction; greedy decoding</p></div>
  <div class="arrow">&rarr;</div>
  <div class="stage"><div class="stage-k">Output</div><b>Report</b>
    <p>Findings by organ and an impression</p></div>
</div>
"""

RESULTS_MD = """
#### Test-set results (1,000 Merlin test scans)
| Adapter | RadGraph-F1 | BLEU-4 | ROUGE-L | METEOR |
|---|---|---|---|---|
| Mean-Pool (baseline) | 0.312 | 10.39 | 0.283 | 0.258 |
| ChunkPool c=4 (SATT without the Transformer) | 0.308 | 11.85 | 0.277 | 0.275 |
| **SATT c=4 (this demo)** | **0.331** | **14.20** | **0.291** | **0.286** |

#### Limitations
Generated reports are fluent and well structured but **omit most findings**, can **contradict**
the radiologist (for example, "no evidence of recurrent disease" when metastases are present),
and can include details that are not supported by the scan. Each model was trained once.
"""

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
#hero { background: linear-gradient(135deg, #0f766e 0%, #0e7490 45%, #1e3a8a 100%);
        border-radius: 24px; padding: 64px 48px 44px; color: #fff;
        box-shadow: 0 20px 50px rgba(14, 116, 144, .25); margin-top: 8px; }
#hero h1 { font-size: 2.7rem; line-height: 1.12; margin: 6px 0 16px; color: #fff;
           letter-spacing: -0.02em; font-weight: 800; }
#hero p { color: rgba(255,255,255,.9); font-size: 1.1rem; max-width: 780px; margin: 0 0 18px; }
.hero-eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .8rem;
                color: rgba(255,255,255,.75); }
.pill { display: inline-block; padding: 7px 14px; border-radius: 999px; margin: 6px 8px 0 0;
        background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
        font-size: .88rem; color: #fff; }
.warn { margin-top: 26px; padding: 11px 16px; border-radius: 12px; font-size: .92rem;
        background: rgba(0,0,0,.25); color: #fff; display: inline-block; }
.section-head { display: flex; align-items: center; gap: 14px; margin: 56px 0 4px; }
.section-head h2 { margin: 0 !important; font-size: 1.6rem; font-weight: 750; }
.step { width: 40px; height: 40px; border-radius: 50%; flex: none; display: flex;
        align-items: center; justify-content: center; font-weight: 800; color: #fff;
        background: linear-gradient(135deg, #14b8a6, #0ea5e9); }
.section-sub { color: var(--body-text-color-subdued); margin: 4px 0 14px 54px !important;
               font-size: 1rem; }
#upload-box { min-height: 230px; }
.chips { display: flex; flex-wrap: wrap; gap: 10px; margin: 6px 0; }
.chip { padding: 10px 16px; border-radius: 14px; background: var(--block-background-fill);
        border: 1px solid var(--border-color-primary); }
.chip span { display: block; font-size: .72rem; text-transform: uppercase; letter-spacing: .08em;
             color: var(--body-text-color-subdued); }
.chip b { font-size: 1.05rem; }
.scroll-hint { margin-top: 12px; color: #14b8a6; font-weight: 600; }
#generate-btn { font-size: 1.2rem !important; padding: 20px !important; border-radius: 16px !important;
                border: none !important; color: #fff !important;
                background: linear-gradient(135deg, #14b8a6, #0ea5e9) !important;
                box-shadow: 0 12px 30px rgba(14, 165, 233, .3); }
.status { display: flex; align-items: center; gap: 14px; padding: 18px 20px; border-radius: 16px;
          border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
          font-size: 1.02rem; }
.spinner { width: 24px; height: 24px; border-radius: 50%; flex: none;
           border: 3px solid rgba(20,184,166,.25); border-top-color: #14b8a6;
           animation: satt-spin .9s linear infinite; }
@keyframes satt-spin { to { transform: rotate(360deg); } }
.report-card { border-radius: 20px; padding: 28px 30px; margin-top: 8px;
               border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
               box-shadow: 0 10px 30px rgba(0,0,0,.12); }
.report-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;
               border-bottom: 2px solid #14b8a6; padding-bottom: 14px; margin-bottom: 8px; }
.report-head h3 { margin: 2px 0 0 !important; font-size: 1.45rem; }
.eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .74rem; color: #14b8a6;
           font-weight: 700; }
.scan-tag { font-family: ui-monospace, monospace; padding: 6px 12px; border-radius: 10px;
            background: rgba(20,184,166,.12); color: #14b8a6; font-weight: 600; }
.report-title { margin: 20px 0 6px; font-weight: 800; text-transform: uppercase; letter-spacing: .1em;
                font-size: .8rem; color: var(--body-text-color-subdued); }
table.findings { width: 100%; border-collapse: collapse; }
table.findings tr { border-bottom: 1px solid var(--border-color-primary); }
table.findings th { text-align: left; vertical-align: top; width: 32%; padding: 9px 14px 9px 0;
                    font-weight: 650; }
table.findings td { padding: 9px 0; line-height: 1.5; }
ol.impression { margin: 4px 0 0 20px; }
ol.impression li { padding: 6px 0; line-height: 1.55; font-weight: 550; }
.report-foot { margin-top: 18px; font-size: .86rem; color: var(--body-text-color-subdued); }
.pipeline { display: flex; flex-wrap: wrap; align-items: stretch; gap: 10px; }
.stage { flex: 1 1 150px; padding: 16px; border-radius: 16px; background: var(--block-background-fill);
         border: 1px solid var(--border-color-primary); }
.stage.trained { border: 2px solid #14b8a6; }
.stage p { margin: 6px 0 0 !important; font-size: .9rem; color: var(--body-text-color-subdued); }
.stage-k { font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; color: #14b8a6;
           font-weight: 700; margin-bottom: 4px; }
.arrow { align-self: center; font-size: 1.5rem; color: var(--body-text-color-subdued); }
.caption { color: var(--body-text-color-subdued); font-size: .92rem; }
.footer { text-align: center; margin: 60px 0 30px; color: var(--body-text-color-subdued);
          font-size: .9rem; }
"""


def make_theme():
    try:
        return gr.themes.Soft(primary_hue="teal", secondary_hue="cyan", neutral_hue="slate",
                              font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif",
                                    "system-ui", "sans-serif"])
    except Exception:
        return None


# =============================================================================
# App
# =============================================================================

def build_app(engine, scans, refs):
    gpu_lock = threading.Lock()
    demo_choices = [(scan_id(p), p) for p in scans]
    demo_default = next((p for p in scans if scan_id(p) == "AC423da3a"),
                        scans[0] if scans else None)

    # ---- callbacks -----------------------------------------------------------
    def load_scan(path):
        """Generator: show the 2D views at once, then the 3D model."""
        if not path:
            raise gr.Error("Please upload a CT scan first.")
        path = str(path)
        if not path.endswith((".nii.gz", ".nii")):
            raise gr.Error("Please upload a NIfTI file (.nii.gz or .nii).")
        try:
            vol, zooms = load_volume(path)
        except Exception as e:
            raise gr.Error(f"Could not read this file as a CT volume: {e}")

        sid = scan_id(path)
        X, Y, Z = vol.shape
        info = chips([
            ("Scan", sid),
            ("Size", f"{X} x {Y} x {Z} voxels"),
            ("Voxel size", f"{zooms[0]:.2f} x {zooms[1]:.2f} x {zooms[2]:.2f} mm"),
            ("Coverage", f"{Z * zooms[2] / 10:.1f} cm head-to-feet"),
        ]) + '<div class="scroll-hint">Scan loaded. Scroll down to explore it in 3D and 2D &darr;</div>'
        slice_img, slice_cap = render_slice(vol, zooms, "Axial", 50)
        grid = model_input_grid(vol)
        note_3d = ('<p class="caption">Building the 3D model ... a few seconds.</p>' if HAVE_3D else
                   '<p class="caption">3D view unavailable: install <code>plotly</code> and '
                   '<code>scikit-image</code>.</p>')
        yield (path, info, gr.update(visible=True), None, note_3d,
               "Axial", 50, slice_img, slice_cap, grid,
               "", "", None, "")
        if HAVE_3D:
            try:
                fig = build_3d(vol, zooms)
                note_3d = ('<p class="caption">Drag to rotate, scroll to zoom. Translucent: body '
                           'surface. Solid: bone, plus contrast-filled vessels and kidneys '
                           '(&gt; 250 HU). Built at about 3 mm resolution for speed.</p>')
            except Exception as e:
                fig, note_3d = None, f'<p class="caption">Could not build the 3D view: {html.escape(str(e))}</p>'
            yield (gr.update(), gr.update(), gr.update(), fig, note_3d,
                   gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                   gr.update(), gr.update(), gr.update(), gr.update())

    def on_slice(path, plane, pos):
        if not path:
            return None, ""
        vol, zooms = load_volume(path)
        return render_slice(vol, zooms, plane, pos)

    def on_plane(plane):
        return gr.update(label=f"Position ({PLANES[plane][1]})", value=50)

    def generate(path):
        if not path:
            raise gr.Error("Please upload a CT scan first.")
        sid = scan_id(path)
        waiting = ('<div class="status"><div class="spinner"></div><div><b>Generating the report '
                   f'for {html.escape(sid)}</b><br>Encoding 64 slices with SigLIP and SATT, then '
                   'writing the report with Llama. About 15-25 seconds.</div></div>')
        yield waiting, "", "", None, ""

        with gpu_lock:                               # one scan on the GPU at a time
            try:
                report = engine.generate_report(path, verbose=True)
            except Exception as e:
                raise gr.Error(f"Generation failed: {e}")
        s = engine.last_stats

        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, f"SATT_report_{sid}.txt")
        with open(out_path, "w") as f:
            f.write(f"SATT generated report for {sid}\n{DISCLAIMER}\n\n{format_report(report)}\n")

        card = report_to_html(report, "Radiology report", "Generated by SATT (c=4)", sid,
                              note=f"&#9888; {html.escape(DISCLAIMER)}")
        stat_items = [("Total time", f"{s['total_seconds']} s"),
                      ("Preprocessing", f"{s['preprocess_seconds']} s"),
                      ("Tokens generated", str(s["generated_tokens"]))]
        if s.get("peak_gpu_gb"):
            stat_items.append(("Peak GPU memory", f"{s['peak_gpu_gb']} GB"))
        stats = chips(stat_items)

        if sid in refs:
            row = refs[sid]
            same, frac = compare_with_saved(report, row["prediction"])
            repro = ("identical to the report saved by the paper's test-set evaluation." if same else
                     f"matches the paper's saved report for the first {frac:.0%} of the text, then "
                     "differs in small details (numerical differences between GPUs).")
            ref = report_to_html(row["reference"], "Radiologist's report", "Merlin test set (reference)",
                                 sid, note=f"Reproduction check: this run's report is {repro}")
        else:
            ref = ('<p class="caption">No radiologist report is available for this scan. '
                   'References exist only for the demo test-set scans.</p>')
        status = ('<div class="status"><b>&#10003; Report ready</b>'
                  '<span class="caption">Scroll down to read it.</span></div>')
        yield status, card, stats, out_path, ref

    # ---- layout --------------------------------------------------------------
    blocks_kwargs, launch_kwargs = {"title": "SATT: CT to Radiology Report"}, {}
    style = {"css": CSS, "theme": make_theme()}
    launch_params = inspect.signature(gr.Blocks.launch).parameters
    blocks_params = inspect.signature(gr.Blocks.__init__).parameters
    for key, val in style.items():
        if val is None:
            continue
        if key in launch_params:          # Gradio 6: styling is passed to launch()
            launch_kwargs[key] = val
        elif key in blocks_params:        # Gradio 4/5: styling is passed to Blocks()
            blocks_kwargs[key] = val

    with gr.Blocks(**blocks_kwargs) as demo:
        cur_path = gr.State(None)
        gr.HTML(HERO)

        # 1. Upload
        gr.HTML(section_head(1, "Upload a CT scan",
                             "An abdominal CT volume in NIfTI format (.nii.gz). "
                             "Nothing leaves this computer."))
        upload = gr.File(label="Drop a .nii.gz file here, or click to browse", type="filepath",
                         file_count="single", elem_id="upload-box")
        with gr.Accordion("No scan at hand? Load one of the demo test scans", open=False):
            with gr.Row():
                demo_dd = gr.Dropdown(choices=demo_choices, value=demo_default,
                                      label="Demo scan (Merlin test set)", scale=3)
                demo_btn = gr.Button("Load demo scan", scale=1)
        scan_info = gr.HTML()

        with gr.Column(visible=False) as results_col:
            # 2. 3D view
            gr.HTML(section_head(2, "3D view", "The whole scan as a 3D model. "
                                 "Drag to rotate, scroll to zoom."))
            plot3d = gr.Plot(show_label=False)
            note3d = gr.HTML()

            # 3. 2D slices
            gr.HTML(section_head(3, "2D slices", "Browse the scan slice by slice in any of "
                                 "the three standard planes (patient's right is on your left)."))
            with gr.Row():
                with gr.Column(scale=1, min_width=240):
                    plane = gr.Radio(list(PLANES), value="Axial", label="Plane")
                    pos = gr.Slider(0, 100, value=50, step=1,
                                    label=f"Position ({PLANES['Axial'][1]})")
                    slice_cap = gr.Markdown()
                with gr.Column(scale=3):
                    slice_img = gr.Image(show_label=False, type="numpy", interactive=False,
                                         height=540)

            # 4. What the model sees
            gr.HTML(section_head(4, "What the model sees",
                                 "The scan becomes 64 slices. SATT averages every 4 neighbouring "
                                 "slices into one chunk (16 chunks) and a Transformer relates the "
                                 "chunks to each other before the language model reads them."))
            grid_img = gr.Image(show_label=False, type="numpy", interactive=False)
            gr.HTML('<p class="caption">Illustration of the model input; most superior slice first. '
                    'The model\'s own preprocessing resamples the scan with cubic interpolation.</p>')

            # 5. Generate
            gr.HTML(section_head(5, "Generate the report",
                                 "Runs the full SATT pipeline on this scan on the laptop GPU."))
            gen_btn = gr.Button("Generate report", variant="primary", elem_id="generate-btn")
            status = gr.HTML()
            report_out = gr.HTML()
            stats_out = gr.HTML()
            dl_file = gr.File(label="Download the report (.txt)", interactive=False)
            with gr.Accordion("Compare with the radiologist's report (demo test scans only)",
                              open=False):
                ref_out = gr.HTML()

        # 6. How it works
        gr.HTML(section_head(6, "How SATT works", "From voxels to words in four stages."))
        gr.HTML(PIPELINE)
        gr.Markdown(RESULTS_MD)
        gr.HTML('<div class="footer">SATT &middot; Slice-Aware Temporal Tokenization for 3D CT '
                'report generation &middot; Research prototype, not for clinical use</div>')

        # ---- events ----------------------------------------------------------
        load_outs = [cur_path, scan_info, results_col, plot3d, note3d,
                     plane, pos, slice_img, slice_cap, grid_img,
                     status, report_out, dl_file, ref_out]
        upload.upload(load_scan, upload, load_outs)
        demo_btn.click(load_scan, demo_dd, load_outs)
        upload.clear(lambda: (None, "", gr.update(visible=False)), None,
                     [cur_path, scan_info, results_col])

        plane.change(on_plane, plane, pos)
        plane.change(on_slice, [cur_path, plane, pos], [slice_img, slice_cap])
        pos.change(on_slice, [cur_path, plane, pos], [slice_img, slice_cap],
                   trigger_mode="always_last")

        gen_btn.click(generate, cur_path, [status, report_out, stats_out, dl_file, ref_out])
    return demo, launch_kwargs


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
    print(f"[app] 3D view {'enabled' if HAVE_3D else 'DISABLED (pip install plotly scikit-image)'}")

    engine = SATTInference(checkpoint_dir=args.checkpoint_dir, variant="satt_chunk4")
    demo, launch_kwargs = build_app(engine, scans, refs)
    os.makedirs(OUT_DIR, exist_ok=True)
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port, inbrowser=True,
                        share=False, allowed_paths=[OUT_DIR], **launch_kwargs)


if __name__ == "__main__":
    main()