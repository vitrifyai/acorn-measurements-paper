#!/usr/bin/env python3
"""Why does one unseen isolate transfer badly when two others transfer well?

The per-field held-out scores split the SEM isolates into three unseen groups:
27E1 (F1 0.42-0.50) against RO-NN-1 and TU-B-10 (0.76). All three were absent
from training, so "the model never saw it" cannot be the explanation on its own.
This measures the candidate explanations against the training distribution:

  geometry     object size and shape, from the reviewed reference at the
               calibrated pixel size
  scale        pixel size and the resulting object size in pixels
  intensity    the black-level and dynamic-range statistics that the manuscript
               already identifies as the largest single sim-to-real gap
  reference    the share of the reference that is a single object by geometry
  crowding     objects per unit area

    python isolate_characterisation.py --work work/sem --out results/sem
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ISOLATE_RE = re.compile(r"^([A-Za-z0-9\-]+?)_?\d{3}$")


def isolate(stem: str) -> str:
    m = ISOLATE_RE.match(stem)
    return m.group(1) if m else stem


def per_field(src: dict) -> dict:
    """Everything measurable about one micrograph and its reference."""
    from acorn.core.measurements import polygon_metrics

    prepared = Path(src["prepared"])
    px = float(src["pixel_size_nm"])
    img = np.array(Image.open(prepared).convert("L")).astype(np.float32)
    h, w = img.shape[:2]
    sc = json.loads((prepared.parent / f".{prepared.stem}.acorn.json").read_text())

    ecd, circ, solid, area_um2 = [], [], [], []
    for a in sc["annotations"]:
        if a.get("type") != "roi":
            continue
        v = [(float(x), float(y)) for x, y in a["vertices"]]
        m = polygon_metrics(v, px)
        ecd.append(m["ecd_nm"])
        circ.append(m["circularity"])
        area_um2.append(m["area_nm2"] / 1e6)
        solid.append(m.get("solidity", float("nan")))

    ecd = np.asarray(ecd)
    circ = np.asarray(circ)
    area_um2 = np.asarray(area_um2)
    confident = ((area_um2 >= 0.3) & (area_um2 <= 3.0) & (circ > 0.5))
    fov_um2 = (h * px / 1e3) * (w * px / 1e3)

    return {
        "stem": src["stem"],
        "isolate": isolate(src["stem"]),
        "split": src["split"],
        "pixel_size_nm": px,
        "shape": f"{h}x{w}",
        "n_reference": int(len(ecd)),
        "confident_frac": float(confident.mean()) if len(ecd) else float("nan"),
        "ecd_nm_median": float(np.median(ecd)) if len(ecd) else float("nan"),
        "ecd_nm_p10": float(np.percentile(ecd, 10)) if len(ecd) else float("nan"),
        "ecd_nm_p90": float(np.percentile(ecd, 90)) if len(ecd) else float("nan"),
        "ecd_px_median": float(np.median(ecd) / px) if len(ecd) else float("nan"),
        "circularity_median": float(np.median(circ)) if len(ecd) else float("nan"),
        "density_per_um2": float(len(ecd) / fov_um2),
        "intensity_mean": float(img.mean()),
        "intensity_std": float(img.std()),
        "intensity_p1": float(np.percentile(img, 1)),
        "intensity_p99": float(np.percentile(img, 99)),
        "frac_below_dn32": float((img < 32).mean()),
        "frac_above_dn223": float((img > 223).mean()),
        "_ecd": ecd,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    man = json.loads((args.work / "split_manifest.json").read_text())
    fields = [per_field(s) for s in man["sources"]]

    # per-isolate roll-up, and the training distribution to compare against
    by_iso: dict[str, list] = defaultdict(list)
    for f in fields:
        by_iso[f["isolate"]].append(f)
    train_ecd = np.concatenate([f["_ecd"] for f in fields
                                if f["split"] == "train" and len(f["_ecd"])])

    keys = ["pixel_size_nm", "confident_frac", "ecd_nm_median", "ecd_px_median",
            "circularity_median", "density_per_um2", "intensity_mean",
            "intensity_p1", "intensity_p99", "frac_below_dn32",
            "frac_above_dn223"]

    rows = []
    for iso in sorted(by_iso):
        fs = by_iso[iso]
        ecd = np.concatenate([f["_ecd"] for f in fs if len(f["_ecd"])])
        row = {
            "isolate": iso,
            "n_fields": len(fs),
            "n_train_fields": sum(1 for f in fs if f["split"] == "train"),
            "n_test_fields": sum(1 for f in fs if f["split"] == "test"),
            "n_reference": int(sum(f["n_reference"] for f in fs)),
        }
        for k in keys:
            row[k] = float(np.mean([f[k] for f in fs]))
        # how far this isolate's size distribution sits from the training pool
        if len(ecd) and len(train_ecd):
            from scipy import stats as st
            row["ks_vs_train_ecd"] = float(st.ks_2samp(ecd, train_ecd).statistic)
            row["ks_p"] = float(st.ks_2samp(ecd, train_ecd).pvalue)
            row["median_ecd_ratio_to_train"] = float(
                np.median(ecd) / np.median(train_ecd))
        rows.append(row)

    with open(args.out / "isolate_characterisation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(args.out / "field_characterisation.csv", "w", newline="") as fh:
        cols = [k for k in fields[0] if not k.startswith("_")]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: f[k] for k in cols} for f in fields])

    # ── printed comparison, unseen isolates against the training pool ────────
    ev = json.loads((args.out / "heldout_evaluation.json").read_text())
    per = list(csv.DictReader(open(args.out / "heldout_per_field.csv")))
    f1_by_iso: dict[str, list] = defaultdict(list)
    for r in per:
        f1_by_iso[isolate(r["stem"])].append(float(r["f1"]))

    print(f"  training pool: {len(train_ecd)} objects, "
          f"median ECD {np.median(train_ecd):.0f} nm\n")
    hdr = ("isolate", "flds", "tr", "F1", "px_nm", "ECD_nm", "ECD_px", "circ",
           "conf%", "dens", "p1", "p99", "<32%", "KS")
    print("  %-11s %4s %2s %6s %6s %7s %7s %5s %5s %5s %5s %5s %5s %5s" % hdr)
    for r in sorted(rows, key=lambda r: np.mean(f1_by_iso.get(r["isolate"], [9]))):
        f1 = f1_by_iso.get(r["isolate"])
        print("  %-11s %4d %2d %6s %6.1f %7.0f %7.0f %5.2f %4.0f%% %5.2f %5.0f %5.0f %4.0f%% %5.2f"
              % (r["isolate"], r["n_fields"], r["n_train_fields"],
                 f"{np.mean(f1):.3f}" if f1 else "--",
                 r["pixel_size_nm"], r["ecd_nm_median"], r["ecd_px_median"],
                 r["circularity_median"], 100 * r["confident_frac"],
                 r["density_per_um2"], r["intensity_p1"], r["intensity_p99"],
                 100 * r["frac_below_dn32"], r.get("ks_vs_train_ecd", float("nan"))))
    print("\n  KS = two-sample Kolmogorov-Smirnov statistic of that isolate's "
          "object-size\n  distribution against the pooled training distribution "
          "(0 = identical).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
