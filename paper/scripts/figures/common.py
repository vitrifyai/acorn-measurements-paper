"""Shared drawing rules for every panel in figures 2, 3 and 4."""
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

import hashlib, subprocess
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

DPI = 600
# 7.0 in = 177.8 mm full width; a 2x2 grid is two panels plus one 2.5 mm
# gutter, so each panel is 87.65 mm. At 600 dpi the plate is 4200 px.
FULL_MM = 177.8
GUTTER_MM = 2.5
COL_MM = (FULL_MM - GUTTER_MM) / 2
MM = 1 / 25.4
BG = "white"                      # one background everywhere

# Colour comes from palette.py, which fixes what colour is allowed to mean:
# modality and nothing else. The names below are ROLES, not hues, so a call site
# reads as "this is scanning data" rather than "this is orange".
from palette import P, apply_style
TRANSMISSION, SCANNING = P.TRANSMISSION, P.SCANNING
REFERENCE, TRUE_POS, FALSE_POS, MISSED = P.REFERENCE, P.TRUE_POS, P.FALSE_POS, P.MISSED
GREY, RULE, BAND = P.GREY, P.RULE, P.BAND
# Legacy aliases, each mapped to the role it actually plays at its call site.
BLUE, ORANGE = TRANSMISSION, SCANNING
GREEN, PURPLE, VERM, SKY, YELLOW = REFERENCE, MISSED, FALSE_POS, TRANSMISSION, SCANNING
TRACE = FALSE_POS

apply_style()

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def git_commit():
    try:
        return subprocess.run(["git", "-C", str(ACORN_SOURCE_DIR),
                               "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return "unknown"

def stretch(a, lo_pct=0.5, hi_pct=99.5):
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    return ((np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8),
            float(lo), float(hi))

_LEGEND_LOG = BUILD_DIR / "qc" / "legend_audit.jsonl"

def _legend_audit(fig, name):
    """Does any legend sit on top of plotted data?

    Eyeballing composites missed this once already (S1A), so it is measured:
    the legend's drawn box is compared against the actual device coordinates of
    every line vertex, scatter point and bar in the same axes.
    """
    import json
    fig.canvas.draw()
    hits = []
    for i, ax in enumerate(fig.axes):
        leg = ax.get_legend()
        if leg is None:
            continue
        bb = leg.get_window_extent()
        n = 0
        for ln in ax.lines:
            d = ln.get_xydata()
            if d is None or len(d) == 0:
                continue
            pts = ax.transData.transform(d)
            n += int(((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) &
                      (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)).sum())
        for col in ax.collections:
            try:
                off = col.get_offsets()
            except Exception:
                continue
            if off is None or len(off) == 0:
                continue
            pts = ax.transData.transform(off)
            n += int(((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) &
                      (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)).sum())
        for pa in ax.patches:
            try:
                pbb = pa.get_window_extent()
            except Exception:
                continue
            if pbb.overlaps(bb):
                n += 1
        hits.append(dict(panel=name, axes=i, data_points_under_legend=n,
                         loc=str(leg._loc)))
    if hits:
        _LEGEND_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_LEGEND_LOG, "a") as fh:
            for h in hits:
                fh.write(json.dumps(h) + "\n")


def save(fig, outdir, name):
    _legend_audit(fig, name)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=DPI, facecolor=BG, bbox_inches="tight",
                pad_inches=0.01)
    fig.savefig(outdir / f"{name}.pdf", facecolor=BG, bbox_inches="tight",
                pad_inches=0.01)
    plt.close(fig)
    print("    wrote", name)

def image_fig(arr8, width_mm=COL_MM):
    H, W = arr8.shape
    w_in = width_mm * MM
    fig = plt.figure(figsize=(w_in, w_in * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    return fig, ax

def scalebar(ax, W_px, px_nm, frac=0.30):
    """Length written on the bar, never left to the caption."""
    nice = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    L_nm = min(nice, key=lambda v: abs(v - W_px * frac * px_nm))
    L_px = L_nm / px_nm
    H = ax.get_ylim()[0]
    x0, y0 = W_px * 0.05, H * 0.945
    ax.add_patch(Rectangle((x0, y0 - H * 0.016), L_px, H * 0.016, facecolor="white",
                           edgecolor="black", lw=0.4, zorder=6))
    lbl = f"{L_nm/1000:g} µm" if L_nm >= 1000 else f"{L_nm:g} nm"
    ax.text(x0 + L_px / 2, y0 - H * 0.026, lbl, color="white", fontsize=7,
            ha="center", va="bottom", zorder=7,
            path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
    return L_nm

def prompt_label(ax, prompt, quote=True):
    """Text drawn on the mask panel. With quote=True it is the literal prompt
    string, monospace, so it can be retyped exactly. With quote=False it is a
    specimen name, which is not a quoted literal and so is not typeset as one.
    Everything else -- font, size, fill, outline, position -- is identical."""
    ax.text(0.5, 0.975, f'"{prompt}"' if quote else prompt,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7, family="monospace", color="white", zorder=7,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])

def trace_marker(ax, xy, size=1.0):
    """One consistent marker for the traced object, used in every panel."""
    ax.plot([xy[0]], [xy[1]], marker="o", ms=13 * size, mfc="none", mec=TRACE,
            mew=1.6, zorder=8)
