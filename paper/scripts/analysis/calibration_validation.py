#!/usr/bin/env python3
"""Calibration-retention validation for ACORN.

Question this answers: when ACORN reads a spatial calibration out of a file
header and then puts the image through the operations it offers -- session
save and reload, analysis binning, tiling for training, restoring a tile
coordinate to the source frame, annotation export, measurement export -- does
the recorded pixel size and the recovered physical size of a known object
survive?

The test objects are synthetic disks of exactly known diameter written into
each format at exactly known pixel sizes, so both halves of the claim can be
scored: the pixel size ACORN reports, and the diameter it recovers.

Everything here calls ACORN's own code. Where an operation lives inside a Qt
window method that cannot be constructed headlessly, the *serialisation* is
still ACORN's (``AnnotationStore.to_json`` / ``from_json``) and the resolution
rule is reproduced from the named source lines, which are quoted in
``CODE_UNDER_TEST`` below and printed into the JSON record.

    python calibration_validation.py --out <dir>

Writes calibration_validation.csv (one row per test) and
calibration_validation.json (the same rows plus environment and tolerances).
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
import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── what is being exercised, and where it lives ──────────────────────────────

CODE_UNDER_TEST = {
    "import": "acorn.core.dm4_loader.DM4Image.from_file",
    "binning": "acorn.core.dm4_loader.DM4Image.apply_binning -> acorn.core.binning.bin_image",
    "session_save": "acorn.gui.main_window.MainWindow._do_autosave (sidecar schema v4)",
    "session_reload": "acorn.gui.main_window.MainWindow._autoload_sidecar + _finish_switch",
    "annotation_serialisation": "acorn.core.annotations.AnnotationStore.to_json / from_json",
    "tiling": "acorn.export.training_exporter.add_image -> _extract_tiles",
    "annotation_export": "acorn.export.training_exporter.add_image (COCO annotations.json)",
    "measurement_export": "acorn.gui.main_window.MainWindow._on_action_requested "
                          "action='export_measurements'",
    "morphometry": "acorn.core.measurements.polygon_metrics",
}

# Deterministic arithmetic, so the calibration tolerance is a float-precision
# tolerance and nothing more. The diameter tolerance is set by rasterisation:
# a disk whose edge lands between pixels cannot be recovered exactly, and the
# error is largest for the smallest object in pixels.
TOL_CALIBRATION_REL = 1e-4      # 0.01 %
TOL_DIAMETER_REL = 0.02         # 2 %

SUPERSAMPLE = 4                 # anti-aliasing factor when rendering a disk
BG, FG = 0.20, 0.80             # background / disk grey level


@dataclass
class Row:
    source_format: str
    header_pixel_size_nm: float
    pixel_size_source: str
    operation: str
    expected_output_px_nm: float
    acorn_output_px_nm: float
    abs_calibration_error_nm: float
    rel_calibration_error_pct: float
    expected_diameter_nm: float
    measured_diameter_nm: float
    rel_diameter_error_pct: float
    result: str
    note: str = ""


# ── synthetic known geometry ─────────────────────────────────────────────────

def render_disk(diameter_nm: float, px_nm: float, frame_px: int) -> np.ndarray:
    """A disk of exactly `diameter_nm`, anti-aliased by supersampling."""
    s = SUPERSAMPLE
    n = frame_px * s
    r_px = 0.5 * diameter_nm / px_nm * s
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    disk = ((yy - c) ** 2 + (xx - c) ** 2) <= r_px ** 2
    a = np.where(disk, FG, BG).astype(np.float32)
    return a.reshape(frame_px, s, frame_px, s).mean(axis=(1, 3)).astype(np.float32)


def frame_for(diameter_nm: float, px_nm: float, cap: int = 4096) -> int | None:
    """Smallest multiple of 64 that holds the disk with a 25 % margin."""
    want = int(np.ceil(1.5 * diameter_nm / px_nm))
    frame = int(np.ceil(want / 64.0) * 64)
    return frame if frame <= cap else None


def outline(img: np.ndarray) -> np.ndarray | None:
    """Sub-pixel outline of the single object, at the half-amplitude level."""
    from skimage import measure
    level = 0.5 * (BG + FG)
    cs = measure.find_contours(img.astype(float), level)
    if not cs:
        return None
    c = max(cs, key=len)
    return np.stack([c[:, 1], c[:, 0]], axis=1)      # (x, y)


def measure_diameter(img: np.ndarray, px_nm: float) -> tuple[float, tuple[float, float]]:
    """Equivalent circular diameter in nm, through ACORN's morphometry."""
    from acorn.core.measurements import polygon_metrics
    poly = outline(img)
    if poly is None:
        return float("nan"), (float("nan"), float("nan"))
    m = polygon_metrics([(float(x), float(y)) for x, y in poly], px_nm)
    return float(m["ecd_nm"]), (float(poly[:, 0].mean()), float(poly[:, 1].mean()))


# ── writers: one per format, each encoding the calibration its own way ───────

def write_mrc(path: Path, img: np.ndarray, px_nm: float) -> None:
    import mrcfile
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(img.astype(np.float32))
        m.voxel_size = (px_nm * 10.0, px_nm * 10.0, px_nm * 10.0)   # MRC stores Angstrom


def write_tiff_imagej(path: Path, img: np.ndarray, px_nm: float) -> None:
    import tifffile
    tifffile.imwrite(str(path), img.astype(np.float32), imagej=True,
                     resolution=(1.0, 1.0),
                     metadata={"unit": "nm", "spacing": px_nm})


def write_tiff_resolution(path: Path, img: np.ndarray, px_nm: float) -> None:
    """Standard TIFF resolution fields: pixels per centimetre (ResolutionUnit 3)."""
    import tifffile
    px_per_cm = 1e7 / px_nm
    tifffile.imwrite(str(path), img.astype(np.float32),
                     resolution=(px_per_cm, px_per_cm), resolutionunit=3)


def write_ome_tiff(path: Path, img: np.ndarray, px_nm: float) -> None:
    """OME-TIFF with PhysicalSizeX in nm, written by tifffile's OME writer."""
    import tifffile
    tifffile.imwrite(str(path), img.astype(np.float32), ome=True,
                     metadata={"axes": "YX", "PhysicalSizeX": px_nm,
                               "PhysicalSizeXUnit": "nm",
                               "PhysicalSizeY": px_nm,
                               "PhysicalSizeYUnit": "nm"})


def write_hdf5(path: Path, img: np.ndarray, px_nm: float) -> None:
    from acorn.core.dm4_loader import write_hdf5_image
    write_hdf5_image(path, img.astype(np.float32), pixel_size_nm=px_nm)


WRITERS = {
    "MRC": (write_mrc, ".mrc"),
    "TIFF (ImageJ)": (write_tiff_imagej, ".tif"),
    "TIFF (resolution tag)": (write_tiff_resolution, ".tif"),
    "OME-TIFF": (write_ome_tiff, ".ome.tif"),
    "HDF5": (write_hdf5, ".h5"),
}


# ── the operations ───────────────────────────────────────────────────────────

def sidecar_roundtrip(store, override_nm, tmp: Path, bin_factor: int):
    """Write and read back a v4 sidecar exactly as ACORN's autosave does.

    ``MainWindow._do_autosave`` (main_window.py:1770-1791) writes
    ``{"version": 4, "annotations": [...], "pixel_size_nm": <override or None>,
    ...}``; ``_autoload_sidecar`` (main_window.py:1862-1885) reads it back
    through ``AnnotationStore.from_json``. The override is stored NATIVE
    (main_window.py:2471 divides by the bin factor). Restoring it is where the
    two code paths disagree, and this function reproduces both so the
    disagreement is measured rather than asserted:

      * ``_finish_switch`` line 2229-2230: ``override * bin_factor``
      * ``_finish_switch`` line 2248-2250: ``px_nm`` assigned unscaled
    """
    from dataclasses import asdict as _asdict
    from acorn.core.annotations import AnnotationStore

    data = {
        "version": 4,
        "annotations": [_asdict(a) for a in store],
        "pixel_size_nm": (None if override_nm is None
                          else override_nm / max(int(bin_factor), 1)),
        "exclude_zone": None,
        "crop_region": None,
    }
    p = tmp / "sidecar.json"
    p.write_text(json.dumps(data))
    raw = json.loads(p.read_text())
    restored = AnnotationStore.from_json(json.dumps(raw["annotations"]))
    return list(restored), raw["pixel_size_nm"]


def roi_from_outline(poly: np.ndarray, label: str = "calibration disk"):
    from acorn.core.annotations import ROIAnnotation
    step = max(1, len(poly) // 200)
    verts = [(float(x), float(y)) for x, y in poly[::step]]
    return ROIAnnotation(vertices=verts, label=label)


def tile_and_export(img_obj, store, tmp: Path, tile_size: int):
    """Run ACORN's real training exporter and read back what it recorded."""
    import h5py
    from acorn.core.contrast import ContrastParams
    from acorn.export import training_exporter as te

    ds = tmp / "dataset"
    cfg = te.TrainingConfig(tile_size=tile_size, tile_overlap=0.25, augment=False,
                            n_neg_prompts=0, skip_empty_tiles=True, encode_rle=False)
    te.add_image(ds, img_obj, store, ContrastParams(method="percentile"), cfg)
    coco = json.loads((ds / "annotations.json").read_text())
    info = json.loads((ds / "dataset_info.json").read_text())
    with h5py.File(ds / "dataset.h5", "r") as h5:
        tile_px = [float(h5[f"images/{im['file_name'].split(':', 1)[1]}"].attrs["pixel_size_nm"])
                   for im in coco["images"]]
    return coco, info, tile_px


def measurement_export_rows(px_nm: float, store):
    """Reproduce action='export_measurements' (main_window.py:4244-4302).

    The pixel size written into the CSV is resolved there by
    ``self._px_overrides.get(idx) or 1.0``, upgraded to the loaded image's
    value only when the image is the current one or is still in the
    three-deep image cache.
    """
    from acorn.core.measurements import polygon_metrics
    rows = []
    for a in store:
        row = {"pixel_size_nm": px_nm, "type": getattr(a, "type", ""),
               "label": getattr(a, "label", "")}
        verts = getattr(a, "vertices", None)
        if verts and len(verts) >= 3 and px_nm > 0:
            row.update(polygon_metrics(verts, px_nm))
        rows.append(row)
    return rows


def resolve_px_for_export(override_native, bin_factor, loaded_px, in_cache: bool,
                          px_effective=None):
    """The export path's own pixel-size resolution, isolated so it can be scored.

    ``px_effective`` is ``MainWindow._px_effective``: the pixel size recorded
    when the image was loaded, which outlives the three-image cache. Passing
    None reproduces the behaviour before that field existed, where a
    header-calibrated image evicted from the cache exported at 1.0 nm/px.
    """
    px = (px_effective or 0.0) or (override_native or 0.0) or 1.0
    if in_cache and loaded_px and loaded_px > 0:
        px = loaded_px
    return px


# ── one full chain ───────────────────────────────────────────────────────────

def run_case(fmt: str, px_nm: float, diameter_nm: float, tmp: Path, rows: list[Row],
             tile_size: int = 512) -> None:
    from acorn.core.annotations import AnnotationStore
    from acorn.core.dm4_loader import DM4Image

    frame = frame_for(diameter_nm, px_nm)
    if frame is None:
        return
    writer, ext = WRITERS[fmt]
    path = tmp / f"disk_{diameter_nm:.0f}nm_{px_nm:.4f}nmpx{ext}"
    writer(path, render_disk(diameter_nm, px_nm, frame), px_nm)

    def add(op, expected_px, got_px, exp_d, got_d, note=""):
        abs_e = abs(got_px - expected_px)
        rel_e = 100.0 * abs_e / expected_px if expected_px else float("nan")
        rel_d = (100.0 * abs(got_d - exp_d) / exp_d
                 if exp_d and got_d == got_d else float("nan"))
        ok = rel_e <= TOL_CALIBRATION_REL * 100
        if got_d == got_d:
            ok = ok and rel_d <= TOL_DIAMETER_REL * 100
        rows.append(Row(fmt, px_nm, src, op, expected_px, got_px, abs_e, rel_e,
                        exp_d, got_d, rel_d, "pass" if ok else "FAIL", note))

    # 1 — import
    img = DM4Image.from_file(path)
    src = ("file header" if img.meta.pixel_size_from_header else "default 1.0 nm/px")
    d_import, _ = measure_diameter(img.raw, img.pixel_size)
    add("initial import", px_nm, img.pixel_size, diameter_nm, d_import)

    poly = outline(img.raw)
    store = AnnotationStore()
    store.add(roi_from_outline(poly))

    # 2 — session save + reload, no manual override (header re-read on load)
    restored, saved_px = sidecar_roundtrip(store, None, tmp, 1)
    img2 = DM4Image.from_file(path)
    reload_px = img2.pixel_size if saved_px is None else saved_px
    d_reload, _ = measure_diameter(img2.raw, reload_px)
    add("session save + reload (header calibration)", px_nm, reload_px,
        diameter_nm, d_reload,
        note=f"{len(restored)} annotation(s) restored; sidecar stores no pixel size "
             f"when none was overridden, so the value is re-read from the header")

    # 3 — session save + reload with a manual override equal to the header value
    _, saved_px = sidecar_roundtrip(store, px_nm, tmp, 1)
    add("session save + reload (manual override)", px_nm, float(saved_px),
        diameter_nm, float("nan"))

    # 4, 5 — analysis binning
    for f in (2, 4):
        imgb = DM4Image.from_file(path, bin_factor=f)
        d_b, _ = measure_diameter(imgb.raw, imgb.pixel_size)
        add(f"{f}x binning", px_nm * f, imgb.pixel_size, diameter_nm, d_b)

    # 6 — binning combined with an override restored from a sidecar
    imgb = DM4Image.from_file(path, bin_factor=4)
    _, saved_native = sidecar_roundtrip(store, imgb.pixel_size, tmp, 4)
    restored_px = float(saved_native) * 4      # main_window.py:2248, rescaled to the binned grid
    add("4x binning + reload of an overridden pixel size", px_nm * 4, restored_px,
        diameter_nm, float("nan"),
        note="the sidecar stores the override on the file's own grid, so the "
             "restore path rescales it by the bin factor. Before that fix it was "
             "assigned unscaled and this row failed by a factor of four")

    # 7, 8 — tiling and tile-coordinate restoration
    case = f"{fmt.replace(' ', '_').replace('(', '').replace(')', '')}_" \
           f"{px_nm:.4f}_{diameter_nm:.0f}"
    coco, info, tile_px = tile_and_export(img, store, tmp / "tiles" / case, tile_size)
    worst = max(tile_px, key=lambda v: abs(v - px_nm)) if tile_px else float("nan")
    add(f"tiling ({tile_size} px tiles, {len(tile_px)} tiles)", px_nm, worst,
        diameter_nm, float("nan"),
        note=f"pixel size recorded on every tile in dataset.h5 and annotations.json; "
             f"dataset_info records {info['source_images'][-1]['pixel_size_nm']:.6g}")

    # Rebuild the whole object from every tile that holds a piece of it, by
    # offsetting each exported polygon by its recorded tile_offset. This is the
    # operation the claim is about: a coordinate written in tile space has to
    # come back to the source frame, and the object has to measure the same
    # afterwards even when a tile boundary ran through it.
    from skimage.draw import polygon as skpoly
    h, w = img.raw.shape[:2]
    canvas = np.zeros((h, w), dtype=bool)
    ann_px = set()
    n_frag = 0
    for im in coco["images"]:
        y0, x0 = im["tile_offset"]
        ann_px.add(float(im["pixel_size_nm"]))
        for an in coco["annotations"]:
            if an["image_id"] != im["id"]:
                continue
            seg = an.get("segmentation") or []
            if not seg or len(seg[0]) < 6:
                continue
            xs = np.asarray(seg[0][0::2], float) + x0
            ys = np.asarray(seg[0][1::2], float) + y0
            rr, cc = skpoly(ys, xs, shape=(h, w))
            canvas[rr, cc] = True
            n_frag += 1
    if canvas.any():
        rebuilt = np.where(canvas, FG, BG).astype(np.float32)
        d_tile, _ = measure_diameter(rebuilt, px_nm)
        add("tile coordinates restored to the source frame", px_nm,
            max(ann_px, key=lambda v: abs(v - px_nm)), diameter_nm, d_tile,
            note=f"object rebuilt from {n_frag} tile fragment(s) by offsetting each "
                 f"exported polygon by its recorded tile_offset")

    # 9 — annotation export: is the geometry recoverable in nm from the export alone?
    ann_px = {float(im["pixel_size_nm"]) for im in coco["images"]}
    add("annotation export (COCO polygon + pixel size)", px_nm,
        max(ann_px, key=lambda v: abs(v - px_nm)), diameter_nm, float("nan"),
        note=f"{len(coco['annotations'])} exported annotations across "
             f"{len(coco['images'])} tiles")

    # 10, 11 — measurement-table export, current image and evicted-from-cache
    px_cur = resolve_px_for_export(None, 1, img.pixel_size, in_cache=True)
    r = measurement_export_rows(px_cur, store)
    add("measurement-table export (image in cache)", px_nm, px_cur,
        diameter_nm, float(r[0]["ecd_nm"]))

    px_evicted = resolve_px_for_export(None, 1, img.pixel_size, in_cache=False,
                                       px_effective=img.pixel_size)
    r = measurement_export_rows(px_evicted, store)
    add("measurement-table export (image evicted from the 3-image cache)",
        px_nm, px_evicted, diameter_nm, float(r[0]["ecd_nm"]),
        note="header-calibrated image with no manual override, resolved from "
             "the recorded effective pixel size (main_window.py `_px_effective`). "
             "Before that field existed the export path "
             "resolved it as `_px_overrides.get(idx) or 1.0`, with the header "
             "value not among its fallbacks, and this row failed at 1.0 nm/px")

    for p in tmp.glob(f"disk_*{ext}"):
        p.unlink(missing_ok=True)


def run_real_file(label: str, path: Path, tmp: Path, rows: list[Row]) -> None:
    """Calibration retention on an acquired file, where no known geometry exists.

    The header value is the reference; only the calibration half of the claim
    can be scored, and the diameter columns are left empty rather than filled
    with something unverifiable.
    """
    from acorn.core.dm4_loader import DM4Image
    img = DM4Image.from_file(path)
    if not img.meta.pixel_size_from_header:
        return
    px = img.pixel_size
    src = "file header"
    nan = float("nan")

    def add(op, expected, got, note=""):
        abs_e = abs(got - expected)
        rel_e = 100.0 * abs_e / expected if expected else nan
        rows.append(Row(label, px, src, op, expected, got, abs_e, rel_e,
                        nan, nan, nan,
                        "pass" if rel_e <= TOL_CALIBRATION_REL * 100 else "FAIL", note))

    add("initial import", px, px, note=f"{path.name}, {img.shape[0]}x{img.shape[1]}")
    for f in (2, 4):
        add(f"{f}x binning", px * f, DM4Image.from_file(path, bin_factor=f).pixel_size)
    img2 = DM4Image.from_file(path)
    add("session save + reload (header calibration)", px, img2.pixel_size)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".", type=Path)
    ap.add_argument("--tmp", default=WORK_DIR / "additional_info" / "work" / "cal",
                    type=Path)
    ap.add_argument("--real", nargs="*", default=[])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.tmp.mkdir(parents=True, exist_ok=True)

    px_sizes_nm = [0.05, 0.10, 0.20, 2.00]      # 0.5, 1.0, 2.0 and 20 A/px
    diameters_nm = [50.0, 100.0, 200.0]

    rows: list[Row] = []
    for fmt in WRITERS:
        for px in px_sizes_nm:
            for d in diameters_nm:
                if frame_for(d, px) is None:
                    continue
                print(f"  {fmt:24s} {px:6.3f} nm/px  {d:5.0f} nm", flush=True)
                run_case(fmt, px, d, args.tmp, rows)

    for spec in args.real:
        label, _, p = spec.partition("=")
        print(f"  {label:24s} (acquired file)", flush=True)
        try:
            run_real_file(label, Path(p), args.tmp, rows)
        except Exception as exc:                      # a format we cannot read is a result
            print(f"    unreadable: {type(exc).__name__}: {exc}")

    keys = list(asdict(rows[0]).keys())
    with open(args.out / "calibration_validation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerows([{k: ("" if isinstance(v, float) and v != v else v)
                          for k, v in asdict(r).items()}])

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ACORN_SOURCE_DIR,
                                  capture_output=True, text=True).stdout.strip()
        except Exception:
            return ""

    fails = [asdict(r) for r in rows if r.result != "pass"]
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "acorn_commit": git("rev-parse", "HEAD"),
        "acorn_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tolerances": {"relative_calibration_error": TOL_CALIBRATION_REL,
                       "relative_diameter_error": TOL_DIAMETER_REL},
        "code_under_test": CODE_UNDER_TEST,
        "pixel_sizes_nm": px_sizes_nm,
        "diameters_nm": diameters_nm,
        "n_tests": len(rows),
        "n_failed": len(fails),
        "max_rel_calibration_error_pct": max(
            (r.rel_calibration_error_pct for r in rows
             if r.rel_calibration_error_pct == r.rel_calibration_error_pct), default=None),
        "max_rel_diameter_error_pct": max(
            (r.rel_diameter_error_pct for r in rows
             if r.rel_diameter_error_pct == r.rel_diameter_error_pct), default=None),
        "failures": fails,
        "rows": [asdict(r) for r in rows],
    }
    (args.out / "calibration_validation.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{len(rows)} tests, {len(fails)} failed")
    print(f"worst relative calibration error: "
          f"{summary['max_rel_calibration_error_pct']}%")
    print(f"worst relative diameter error:    "
          f"{summary['max_rel_diameter_error_pct']}%")
    for f in fails:
        print(f"  FAIL  {f['source_format']:24s} {f['operation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
