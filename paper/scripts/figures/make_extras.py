#!/usr/bin/env python3
"""Figure S3 (training curves), Figure 1 panel C, and compose.py.

1A and 1B are live interface captures and cannot be produced here; a shot list
is written instead so they can be taken in one sitting.
"""
from __future__ import annotations
import sys as _release_sys
from pathlib import Path as _ReleasePath

_RELEASE_SCRIPTS = _ReleasePath(__file__).resolve().parents[1]
if str(_RELEASE_SCRIPTS) not in _release_sys.path:
    _release_sys.path.insert(0, str(_RELEASE_SCRIPTS))
from release_paths import (
    ACORN_SOURCE_DIR, ADDITIONAL_INFO_DIR, BUILD_DIR, FIGURE_SCRIPTS_DIR,
    MODEL_CHECKPOINTS_DIR, MODEL_RUNS_DIR, RAW_CRYO_DIR, RAW_SEM_DIR, WORK_DIR,
)

import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COL_MM, FULL_MM, MM, TRANSMISSION, SCANNING, REFERENCE,
                    TRUE_POS, FALSE_POS, MISSED, GREY, RULE, save)

HERE = Path(__file__).resolve().parent
RUNS = MODEL_RUNS_DIR

def fig_S3():
    out = HERE / "si" / "S3"
    meta = {}
    runs = [("transmission", "nanoparticles_tem", TRANSMISSION),
            ("scanning", "spores_sem", SCANNING)]
    for tag, name, col in runs:
        rows = list(csv.DictReader(open(RUNS / name / "results.csv")))
        ep = [float(r["epoch"]) for r in rows]
        gk = lambda sub: next((k for k in rows[0] if sub in k), None)
        kb, km = gk("mAP50(B)"), gk("mAP50(M)")
        fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM * 0.72))
        ax = fig.add_subplot(111)
        # One modality, so one colour: box against mask is line style.
        ax.plot(ep, [float(r[kb]) for r in rows], "-", color=col, lw=1.3, label="box mAP@50")
        ax.plot(ep, [float(r[km]) for r in rows], "--", color=col, lw=1.3, label="mask mAP@50")
        ax.axvline(len(rows), color=GREY, lw=.7, ls=":")
        ax.set_xlabel("epoch"); ax.set_ylabel("mAP@50 on the validation split")
        ax.set_ylim(0, 1.02); ax.grid(alpha=.15, lw=.4)
        ax.legend(frameon=False, loc="lower right")
        save(fig, out, f"S3{'A' if tag=='transmission' else 'B'}_{tag}")
        meta[tag] = dict(epochs_run=len(rows), csv=str(RUNS / name / "results.csv"),
                         final_box_map50=float(rows[-1][kb]),
                         final_mask_map50=float(rows[-1][km]))
    # S3C loss components
    fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM * 0.72)); ax = fig.add_subplot(111)
    for tag, name, col in runs:
        rows = list(csv.DictReader(open(RUNS / name / "results.csv")))
        ep = [float(r["epoch"]) for r in rows]
        for sub, ls in (("box_loss", "-"), ("seg_loss", "--")):
            k = next((c for c in rows[0] if sub in c and "train" in c), None)
            if k: ax.plot(ep, [float(r[k]) for r in rows], ls, color=col, lw=1.2,
                          label=f"{tag} {sub.replace('_',' ')}")
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.grid(alpha=.15, lw=.4); ax.legend(frameon=False, fontsize=6.0)
    save(fig, out, "S3C_loss_components")
    meta["reported_epoch_rule"] = (
        "ultralytics best.pt, the epoch with the highest validation fitness, not "
        "the final epoch; the dotted line marks where training stopped")
    (out / "S3_metadata.json").write_text(json.dumps(meta, indent=1))
    print("  S3 built:", {k: v.get("epochs_run") for k, v in meta.items() if isinstance(v, dict)})

def fig_1C():
    """Panel C: the four ways to run ACORN. Drawn, not captured."""
    out = HERE / "fig1"
    fig = plt.figure(figsize=(FULL_MM * MM, FULL_MM * MM * 0.20))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 22); ax.axis("off")
    items = [("desktop application", "point and click"),
             ("command line", "batch and scripting"),
             ("Python library", "import acorn"),
             ("natural language", "drives the same operations")]
    w, gap = 21.0, 4.0
    for i, (title, sub) in enumerate(items):
        x = 2 + i * (w + gap)
        nat = (i == 3)
        ax.add_patch(FancyBboxPatch((x, 5), w, 12,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor="white" if not nat else "#f4f6f7",
            edgecolor=TRANSMISSION, lw=1.1 if not nat else 1.4,
            linestyle="-" if not nat else (0, (4, 2))))
        ax.text(x + w/2, 12.4, title, ha="center", va="center", fontsize=8,
                color=TRANSMISSION, fontweight="bold" if nat else "normal")
        ax.text(x + w/2, 8.6, sub, ha="center", va="center", fontsize=6.6, color=GREY)
    ax.annotate("", xy=(2 + 3*(w+gap) + w/2, 4.4), xytext=(2 + 3*(w+gap) + w/2, 1.6),
                arrowprops=dict(arrowstyle="-", lw=0))
    ax.text(50, 2.0, "one application, one calibration, one annotation store — "
            "nothing is uploaded", ha="center", va="center", fontsize=6.8, color=GREY)
    save(fig, out, "1C")

def write_compose():
    (HERE / "compose.py").write_text('''#!/usr/bin/env python3
"""Assemble plates from individual panels, adding letters in a white band.

Panels are produced without letters so a plate can be re-laid without
regenerating anything. Rows are lists; a row given as a single-element list of
lists is treated as ONE panel spanning that row, which is how Figure 1's 1B
strip gets a single letter instead of five.

    python compose.py fig4
"""
from __future__ import annotations
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BAND, PAD, LETTER = 34, 10, 26

PLATES = {
    "fig2": [["fig2/2A.png"], ["fig2/2B.png"], ["fig2/2C.png"], ["fig2/2D.png"]],
    "fig3": [["fig3/3A.png"], ["fig3/3B.png"], ["fig3/3C.png"], ["fig3/3D.png"]],
    "fig4": [["fig4/4A_cryoTEM.png", "fig4/4A_SEM.png"],
             ["fig4/4B_cryoTEM.png", "fig4/4B_SEM.png"],
             ["fig4/4C.png"], ["fig4/4D.png"]],
    "fig5": [["fig5/5A_uncorrected.png", "fig5/5B_motion_corrected.png"],
             ["fig5/5C_trajectory.png", "fig5/5D_dose_dependence.png"]],
    # Figure 1: the 1B strip is ONE panel, so it is nested one level deeper.
    "fig1": [["fig1/1A.png"],
             [["fig1/1B_1_prompt.png", "fig1/1B_2_queue.png", "fig1/1B_3_train.png",
               "fig1/1B_4_predict.png", "fig1/1B_5_measure.png"]],
             ["fig1/1C.png"]],
}

def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

def _strip(paths, height):
    ims = []
    for p in paths:
        im = Image.open(HERE / p).convert("RGB")
        im = im.resize((int(im.width * height / im.height), height), Image.LANCZOS)
        ims.append(im)
    w = sum(i.width for i in ims) + PAD * (len(ims) - 1)
    out = Image.new("RGB", (w, height), "white"); x = 0
    for i in ims:
        out.paste(i, (x, 0)); x += i.width + PAD
    return out

def compose(name, letters=True, width=2400):
    rows = PLATES[name]
    built = []
    for row in rows:
        if len(row) == 1 and isinstance(row[0], list):     # a spanning strip
            built.append(_strip(row[0], 520))
        else:
            ims = [Image.open(HERE / p).convert("RGB") for p in row]
            h = min(i.height for i in ims)
            built.append(_strip(row, h))
    scaled = []
    for im in built:
        scaled.append(im.resize((width, int(im.height * width / im.width)), Image.LANCZOS))
    H = sum(i.height + (BAND if letters else 0) for i in scaled) + PAD * (len(scaled) - 1)
    plate = Image.new("RGB", (width, H), "white")
    d = ImageDraw.Draw(plate); f = _font(LETTER)
    y = 0
    for i, im in enumerate(scaled):
        if letters:
            d.text((4, y + 2), chr(ord("A") + i), fill="black", font=f)
            y += BAND
        plate.paste(im, (0, y)); y += im.height + PAD
    out = HERE / "plates"; out.mkdir(exist_ok=True)
    plate.save(out / f"{name}.png", dpi=(600, 600))
    print("wrote", out / f"{name}.png", plate.size)

if __name__ == "__main__":
    for n in (sys.argv[1:] or ["fig2", "fig3", "fig4", "fig5"]):
        compose(n)
''')
    print("  compose.py written")

if __name__ == "__main__":
    print("S3 training curves:"); fig_S3()
    print("Figure 1 panel C:");   fig_1C()
    print("composer:");           write_compose()
