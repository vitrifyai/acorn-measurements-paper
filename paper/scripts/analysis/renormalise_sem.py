#!/usr/bin/env python3
"""Re-prepare the SEM micrographs without crushing the acquisition's black level.

The published preparation (`build_real_sem.py`) stretches each image against its
own 0.5-99.5 percentiles. The acquired TIFFs are already uint8 0-255 from the
microscope, so that stretch adds nothing and takes something away: it drives
60-85 % of every frame below DN 32, by an amount that depends on how much empty
substrate happens to be in the field. Across the held-out set that
per-image black level is the strongest single predictor of the model's score
(Spearman rho -0.88), and it does not exist in the raw data, where about 1 % of
pixels sit below DN 32 in every image.

This writes two alternative preparations of the same 26 micrographs, with the
same banner crop and therefore the same labels, so intensity is the only
variable:

  raw     the microscope's own 8-bit values, untouched
  global  one percentile window computed once over the pooled histogram of all
          26 cropped images and applied identically to each

`raw` asks whether the stretch costs anything. `global` separates "the black
level was crushed" from "the window varied per image", because it uses the full
8-bit range like the published version but applies one window to everything.

    python renormalise_sem.py --out work/renorm
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


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

SRC = RAW_SEM_DIR
WORK = WORK_DIR


def crop_banner(img: np.ndarray) -> np.ndarray:
    """The banner crop from build_real_sem.py, reproduced exactly.

    Zeiss burns a bright info band into the bottom of the frame. The rule is
    data-dependent, so it is copied rather than re-invented: applied to the same
    raw array it returns the same cut, which is what keeps the existing labels
    valid for these images.
    """
    if img.ndim == 3:
        img = img.mean(-1)
    h = img.shape[0]
    rowmean = img.mean(axis=1)
    thresh = img.mean() + 1.5 * img.std()
    cut = h
    for y in range(h - 1, int(h * 0.80), -1):
        if rowmean[y] > thresh:
            cut = y
    return img[:cut] if cut < h else img


def dark_frac(a: np.ndarray) -> float:
    return float((a < 32).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    man = json.loads((WORK / "real_sem/manifest.json").read_text())
    published = WORK / "real_sem/images"

    cropped: dict[str, np.ndarray] = {}
    for rec in man:
        stem = rec["stem"]
        raw = tifffile.imread(SRC / f"{stem}.tif").astype(np.float32)
        c = crop_banner(raw)
        # the crop has to agree with the published image, or the labels move
        pub = np.array(Image.open(published / f"{stem}.png").convert("L"))
        if c.shape != pub.shape:
            print(f"  {stem}: crop {c.shape} disagrees with published "
                  f"{pub.shape} -- skipped")
            continue
        cropped[stem] = c

    pool = np.concatenate([c.ravel() for c in cropped.values()])
    g_lo, g_hi = np.percentile(pool, [0.5, 99.5])
    print(f"  {len(cropped)} images, pooled window {g_lo:.1f}-{g_hi:.1f} DN\n")

    arms = {
        "raw": lambda c: np.clip(c, 0, 255).astype(np.uint8),
        "global": lambda c: (np.clip((c - g_lo) / max(g_hi - g_lo, 1e-6), 0, 1)
                             * 255).astype(np.uint8),
    }

    stats = []
    for arm, fn in arms.items():
        d = args.out / arm / "images"
        d.mkdir(parents=True, exist_ok=True)
        for stem, c in cropped.items():
            Image.fromarray(fn(c)).save(d / f"{stem}.png")
        # labels are unchanged: same crop, same frame, same normalised vertices
        lab = args.out / arm / "labels"
        lab.mkdir(parents=True, exist_ok=True)
        for stem in cropped:
            (lab / f"{stem}.txt").write_text(
                (WORK / "real_sem/labels" / f"{stem}.txt").read_text())
        (args.out / arm / "manifest.json").write_text(json.dumps(
            [r for r in man if r["stem"] in cropped], indent=1))

    print("  %-16s %10s %10s %10s" % ("preparation", "mean DN", "<DN32", "p1"))
    for label, get in (("published", lambda s: np.array(
                            Image.open(published / f"{s}.png").convert("L")).astype(np.float32)),
                       ("raw", lambda s: arms["raw"](cropped[s]).astype(np.float32)),
                       ("global", lambda s: arms["global"](cropped[s]).astype(np.float32))):
        m = [get(s) for s in cropped]
        row = dict(preparation=label,
                   mean_dn=float(np.mean([a.mean() for a in m])),
                   frac_below_32=float(np.mean([dark_frac(a) for a in m])),
                   p1=float(np.mean([np.percentile(a, 1) for a in m])),
                   spread_of_frac_below_32=float(np.std([dark_frac(a) for a in m])))
        stats.append(row)
        print("  %-16s %10.1f %9.0f%% %10.1f" %
              (label, row["mean_dn"], 100 * row["frac_below_32"], row["p1"]))
    print("\n  spread of the per-image dark fraction (sd across the 26 images):")
    for r in stats:
        print("    %-10s %.3f" % (r["preparation"], r["spread_of_frac_below_32"]))

    (args.out / "intensity_stats.json").write_text(json.dumps(
        {"pooled_window_dn": [float(g_lo), float(g_hi)],
         "n_images": len(cropped), "by_preparation": stats}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
