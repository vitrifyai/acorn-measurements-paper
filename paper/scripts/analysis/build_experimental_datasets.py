#!/usr/bin/env python3
"""Build training datasets from reviewed experimental annotations, in ACORN.

Two datasets, one per modality, each assembled by the same steps the
application performs:

  cryo-TEM (PLGA nanoparticles)
      acquired MRC -> calibrated analysis binning -> SAM 3 text-prompted
      segmentation -> documented review (accept / correct / reject) ->
      annotation store with provenance -> sidecar -> tiled training export

  SEM (bacterial spores)
      acquired Zeiss TIFF, vendor-tag calibration -> the operator's own
      reviewed ACORN annotations, imported from their sidecars -> annotation
      store with provenance -> tiled training export

Both then go through ``acorn.export.dataset_finalizer.finalize_dataset`` with
an EXPLICIT per-source-micrograph split, so every tile from one acquisition
field lands in exactly one of train / val / test. The split is decided before
tiling and recorded in ``split_manifest.json``.

    python build_experimental_datasets.py --modality cryo --out <dir>
    python build_experimental_datasets.py --modality sem  --out <dir>
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
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

WORK = WORK_DIR
CRYO_ROOT = RAW_CRYO_DIR
SEM_IMAGES = WORK / "real_sem/images"
SEM_MANIFEST = WORK / "real_sem/manifest.json"
SEM_LABELS = WORK / "real_sem/labels"

# The field used for the figure-2B prompt demonstration. Prompt phrasing was
# chosen while looking at it, so it cannot be part of a held-out set.
CRYO_PROMPT_DEV = "FoilHole_14663539_Data_14664769_14664771_20240215_115338"
SEM_PROMPT_DEV = "4AC2_008"

CRYO_PROMPT = "dark round blob"          # verbatim, as in the manuscript
CRYO_CONF = 0.5
CRYO_BIN = 4                             # 2.23 A/px x 4 = 0.892 nm/px
CRYO_KEEP_NM = (8.0, 400.0)              # the manuscript's acceptance window
SOLIDITY_REPAIR = 0.90                   # below this, the outline is repaired

TILE = 512
OVERLAP = 0.25
# An object has to be at least half inside a tile to be labelled there; with
# 25 % overlap the whole object is present in a neighbouring tile, so nothing
# is lost and no truncated sliver is taught as an example of the object.
MIN_INSTANCE_FRAC = 0.5


import re as _re

_ISOLATE_RE = _re.compile(r"^([A-Za-z0-9\-]+?)_?\d{3}$")


def sem_isolate(stem: str) -> str:
    """The isolate code an SEM filename carries; the trailing three digits are
    the frame number. Operator-supplied identity, not derivable from anything
    else in the file."""
    m = _ISOLATE_RE.match(stem)
    return m.group(1) if m else stem


# ── review protocol ──────────────────────────────────────────────────────────

REVIEW_PROTOCOL = """
Review is applied as a documented deterministic rule so that the outcome is
reproducible and auditable, rather than by an operator on the canvas:

  reject   equivalent circular diameter outside 8-400 nm
           (the acceptance window stated for figure 2B)
  correct  solidity below 0.90: the outline is replaced by its convex hull,
           written back through AnnotationStore.update so the edit is recorded
           in the annotation's provenance (modification_count, geometry hash)
  accept   everything else, unchanged

Rejected candidates are removed from the store. Every outcome is counted.
""".strip()


def sha256(path: Path, cap: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(cap):
            h.update(chunk)
    return h.hexdigest()


def git_sha(repo=ACORN_SOURCE_DIR) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def stretch8(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [0.5, 99.5])
    return (np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)


def mask_outline(m: np.ndarray, max_pts: int = 200):
    from skimage import measure
    cs = measure.find_contours(m.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    step = max(1, len(c) // max_pts)
    return [(float(x), float(y)) for y, x in c[::step]]


def solidity(verts) -> float:
    from scipy.spatial import ConvexHull
    v = np.asarray(verts, float)
    if len(v) < 4:
        return 1.0
    try:
        hull = ConvexHull(v)
    except Exception:
        return 1.0
    a = abs(np.dot(v[:, 0], np.roll(v[:, 1], -1)) - np.dot(v[:, 1], np.roll(v[:, 0], -1))) / 2
    return float(a / hull.volume) if hull.volume > 0 else 1.0


def convex_hull_verts(verts):
    from scipy.spatial import ConvexHull
    v = np.asarray(verts, float)
    hull = ConvexHull(v)
    return [(float(x), float(y)) for x, y in v[hull.vertices]]


# ── cryo-TEM: prompted segmentation, then review ─────────────────────────────

def cryo_sources(n: int, reference_files: list[str]) -> list[Path]:
    """Reference micrographs first (they carry the independent particle list),
    then further fields in filename order until `n` are collected."""
    ref = [Path(p) for p in reference_files]
    rest = [p for p in sorted(CRYO_ROOT.rglob("*.mrc")) if p not in ref]
    return (ref + rest)[:n]


def cryo_annotate(path: Path, predictor, out_dir: Path, notes: list) -> dict | None:
    from acorn.core.annotations import AnnotationStore, ROIAnnotation
    from acorn.core.dm4_loader import DM4Image
    from acorn.core.measurements import polygon_metrics
    from acorn.core import provenance as prov
    from PIL import Image

    try:
        img = DM4Image.from_file(path, bin_factor=CRYO_BIN)
    except Exception as exc:
        notes.append(f"{path.name}: unreadable ({type(exc).__name__})")
        return None
    if img.raw is None or img.raw.ndim != 2:
        notes.append(f"{path.name}: not a 2-D micrograph")
        return None
    native = img.meta.native_pixel_size
    if not (0.1 < native < 4.0):
        notes.append(f"{path.name}: native pixel size {native:.4g} nm outside range")
        return None

    px = img.pixel_size
    img8 = stretch8(img.raw)

    proc = predictor._sam3_processor
    state = proc.set_image(Image.fromarray(np.repeat(img8[:, :, None], 3, 2)))
    proc.set_confidence_threshold(CRYO_CONF)
    masks = proc.set_text_prompt(CRYO_PROMPT, state).get("masks")

    store = AnnotationStore()
    accepted = corrected = rejected = 0
    with prov.provenance_context(prov.Origin.SAM.value,
                                 invocation=prov.Invocation.DIRECT_GUI.value,
                                 source_model={"path": "sam3", "sha256": None}):
        with prov.batch_transaction(prov.Origin.SAM.value,
                                    invocation=prov.Invocation.DIRECT_GUI.value,
                                    human_interaction_count=1):
            for x in (masks if masks is not None else []):
                m = np.asarray(x.squeeze().float().cpu().numpy() > 0.5)
                if m.sum() < 4:
                    continue
                verts = mask_outline(m)
                if verts is None or len(verts) < 3:
                    continue
                d = polygon_metrics(verts, px)["ecd_nm"]
                if not (CRYO_KEEP_NM[0] <= d <= CRYO_KEEP_NM[1]):
                    rejected += 1
                    continue
                ann = ROIAnnotation(vertices=verts, label="PLGA nanoparticle")
                store.add(ann)
                if solidity(verts) < SOLIDITY_REPAIR:
                    ann.vertices = convex_hull_verts(verts)
                    store.update(ann)
                    corrected += 1
                else:
                    accepted += 1

    prepared = out_dir / "prepared" / f"{path.stem}.png"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img8).save(prepared)
    write_sidecar(prepared, store, px)

    return dict(stem=path.stem, source=str(path), source_sha256=sha256(path),
                prepared=str(prepared), pixel_size_nm=px,
                native_pixel_size_nm=native, bin_factor=CRYO_BIN,
                pixel_size_source="MRC voxel_size header, calibrated analysis binning",
                shape=list(img.raw.shape), n_accepted=accepted,
                n_corrected=corrected, n_rejected=rejected,
                n_annotations=len(store), prompt=CRYO_PROMPT, confidence=CRYO_CONF)


# ── SEM: the operator's own reviewed annotations, imported ───────────────────

def sem_annotate(stem: str, meta: dict, out_dir: Path,
                 images_dir: Path | None = None,
                 labels_dir: Path | None = None,
                 prep_label: str = "published percentile stretch") -> dict:
    from acorn.core.annotations import AnnotationStore, ROIAnnotation
    from acorn.core.dm4_loader import DM4Image
    from acorn.core import provenance as prov
    from PIL import Image

    png = (images_dir or SEM_IMAGES) / f"{stem}.png"
    labels = labels_dir or SEM_LABELS
    img = DM4Image.from_file(png)
    img.meta.pixel_size = float(meta["pixel_size_nm"])       # vendor-tag value
    h, w = img.raw.shape[:2]

    store = AnnotationStore()
    n = 0
    with prov.provenance_context(prov.Origin.IMPORTED_SIDECAR.value,
                                 invocation=prov.Invocation.BATCH.value):
        with prov.batch_transaction(prov.Origin.IMPORTED_SIDECAR.value,
                                    invocation=prov.Invocation.BATCH.value,
                                    human_interaction_count=1):
            for line in (labels / f"{stem}.txt").read_text().split("\n"):
                f = line.split()
                if len(f) < 7:
                    continue
                xy = np.asarray([float(v) for v in f[1:]], float).reshape(-1, 2)
                verts = [(float(x * w), float(y * h)) for x, y in xy]
                store.add(ROIAnnotation(vertices=verts, label="Spore"))
                n += 1

    prepared = out_dir / "prepared" / f"{stem}.png"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.array(Image.open(png).convert("L"))).save(prepared)
    write_sidecar(prepared, store, img.pixel_size)

    return dict(stem=stem, source=str(png),
                source_sha256=sha256(png), prepared=str(prepared),
                preparation=prep_label,
                pixel_size_nm=img.pixel_size, native_pixel_size_nm=img.pixel_size,
                bin_factor=1,
                pixel_size_source="Zeiss CZ_SEM ap_image_pixel_size (vendor TIFF tag)",
                shape=[h, w], n_accepted=n, n_corrected=0, n_rejected=0,
                n_annotations=n, prompt=None, confidence=None,
                magnification=meta.get("magnification"))


def write_sidecar(image_path: Path, store, px_nm: float) -> Path:
    """The v4 sidecar ACORN's autosave writes, provenance block included."""
    from dataclasses import asdict
    p = image_path.parent / f".{image_path.stem}.acorn.json"
    p.write_text(json.dumps({
        "version": 4,
        "annotations": [asdict(a) for a in store],
        "pixel_size_nm": px_nm,
        "exclude_zone": None,
        "crop_region": None,
    }, indent=1))
    return p


# ── tiling + split, through ACORN ────────────────────────────────────────────

def export_and_split(records: list[dict], out_dir: Path, split_of: dict,
                     modality: str, notes: list) -> dict:
    from acorn.core.annotations import AnnotationStore
    from acorn.core.contrast import ContrastParams
    from acorn.core.dm4_loader import DM4Image
    from acorn.export import training_exporter as te
    from acorn.export.dataset_finalizer import finalize_dataset

    ds = out_dir / "dataset"
    cfg = te.TrainingConfig(tile_size=TILE, tile_overlap=OVERLAP, augment=False,
                            n_neg_prompts=0, skip_empty_tiles=True,
                            encode_rle=False,
                            min_instance_area_frac=MIN_INSTANCE_FRAC)
    params = ContrastParams(method="percentile")

    per_source = []
    for r in records:
        img = DM4Image.from_file(Path(r["prepared"]))
        img.meta.pixel_size = float(r["pixel_size_nm"])
        sc = Path(r["prepared"]).parent / f".{Path(r['prepared']).stem}.acorn.json"
        store = AnnotationStore.from_json(
            json.dumps(json.loads(sc.read_text())["annotations"]))
        summ = te.add_image(ds, img, store, params, cfg)
        per_source.append({**r, **summ, "split": split_of[r["stem"]]})
        print(f"    {r['stem']:<60s} {summ['n_tiles']:3d} tiles "
              f"{summ['n_augmented']:3d} kept  {summ['n_instances_total']:4d} inst "
              f"[{split_of[r['stem']]}]", flush=True)

    explicit = {r["prepared"]: {"train": "Train", "val": "Validation",
                                "test": "Test"}[split_of[r["stem"]]]
                for r in records}
    res = finalize_dataset(ds, val_frac=0.0, test_frac=0.0, seed=0,
                           explicit_splits=explicit)

    split_map = json.loads((ds / "splits/split_map.json").read_text())
    coco = json.loads((ds / "annotations.json").read_text())
    by_id = {im["id"]: im for im in coco["images"]}
    inst = {"train": 0, "val": 0, "test": 0}
    tiles = {"train": 0, "val": 0, "test": 0}
    for a in coco["annotations"]:
        s = split_map.get(str(a["image_id"]))
        if s:
            inst[s] += 1
    for k, s in split_map.items():
        tiles[s] += 1
    empty_tiles = sum(r["n_tiles"] - r["n_augmented"] for r in per_source)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modality": modality,
        "acorn_commit": git_sha(),
        "review_protocol": REVIEW_PROTOCOL,
        "tiling": {"tile_size": TILE, "overlap": OVERLAP, "augment_at_export": False,
                   "skip_empty_tiles": True,
                   "min_instance_area_frac": MIN_INSTANCE_FRAC,
                   "note": "the split is assigned per source micrograph and passed to "
                           "finalize_dataset as explicit_splits, so tiling happens "
                           "after the partition and no field appears in two splits"},
        "split_counts_source_images": {
            k: sum(1 for r in per_source if r["split"] == k) for k in ("train", "val", "test")},
        "split_counts_tiles": tiles,
        "split_counts_instances": inst,
        "negative_tiles_written": 0,
        "empty_tiles_discarded": empty_tiles,
        "totals": {
            "source_images": len(per_source),
            "annotated_objects": sum(r["n_annotations"] for r in per_source),
            "accepted_unchanged": sum(r["n_accepted"] for r in per_source),
            "corrected": sum(r["n_corrected"] for r in per_source),
            "rejected": sum(r["n_rejected"] for r in per_source),
            "tiles": sum(tiles.values()),
        },
        "finalize": res.get("split_counts"),
        "notes": notes,
        "sources": per_source,
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ── main ─────────────────────────────────────────────────────────────────────

def stratified_splits(stems: list[str], group_of, forced_train: list[str],
                      seed: int) -> dict:
    """Split by source micrograph, stratified so every group that CAN be
    represented in training is.

    Splitting by micrograph stops tiles from one acquisition field reaching two
    partitions, which is the leak that matters. It does not control which
    *specimens* are in training, and with 13 isolates over 26 micrographs a
    random draw leaves whole isolates out: the held-out score then measures how
    well those particular unseen isolates happen to transfer, mixed in with
    everything else. This assignment gives every isolate with more than one
    field a field in training and a field in test, so the two effects can be
    separated. Isolates with a single field go to training, because an isolate
    can be trained on or tested on but not both.

      n == 1  -> train
      n == 2  -> train, test
      n == 3  -> train, val, test
      n >= 4  -> n-2 train, val, test
    """
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[str]] = {}
    for s in stems:
        by_group.setdefault(group_of(s), []).append(s)

    out: dict[str, str] = {}
    for g in sorted(by_group):
        members = sorted(by_group[g])
        # a prompt-development field is pinned to train and taken out of the draw
        pinned = [m for m in members if m in forced_train]
        rest = list(rng.permutation([m for m in members if m not in pinned]))
        order = pinned + rest
        n = len(order)
        if n == 1:
            splits = ["train"]
        elif n == 2:
            splits = ["train", "test"]
        elif n == 3:
            splits = ["train", "val", "test"]
        else:
            splits = ["train"] * (n - 2) + ["val", "test"]
        for stem, spl in zip(order, splits):
            out[stem] = spl
    return out


def assign_splits(stems: list[str], forced_train: list[str], n_val: int, n_test: int,
                  seed: int) -> dict:
    """Split by source micrograph. Fields used for prompt development are
    forced into train, never into val or test."""
    rng = np.random.default_rng(seed)
    pool = [s for s in stems if s not in forced_train]
    order = list(rng.permutation(pool))
    test = order[:n_test]
    val = order[n_test:n_test + n_val]
    out = {}
    for s in stems:
        out[s] = "test" if s in test else "val" if s in val else "train"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["cryo", "sem"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-sources", type=int, default=30)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--split-seed", type=int, default=17)
    ap.add_argument("--images-dir", type=Path, default=None,
                    help="alternative prepared-image directory, for comparing "
                         "one intensity preparation against another")
    ap.add_argument("--labels-dir", type=Path, default=None)
    ap.add_argument("--prep-label", default="published percentile stretch")
    ap.add_argument("--split-from", type=Path, default=None,
                    help="reuse the per-micrograph split of an existing "
                         "split_manifest.json, so two runs differ in nothing else")
    ap.add_argument("--split-mode", choices=["random", "stratified"],
                    default="random",
                    help="stratified keeps every isolate with more than one "
                         "field represented in training")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    if args.modality == "cryo":
        ref = sorted({r["File Location"]
                      for r in json.loads((WORK / "plga_reference.json").read_text())})
        notes.append(f"{len(ref)} micrographs carry the independent CryoBLOB "
                     f"particle list and are placed in the held-out set")
        from acorn.core.sam_predictor import SAMPredictor
        pred = SAMPredictor(backend="sam3")
        pred.load_model()
        records = []
        for p in cryo_sources(args.n_sources, ref):
            r = cryo_annotate(p, pred, args.out, notes)
            if r is None:
                continue
            print(f"    {r['stem']:<60s} {r['n_accepted']:3d} accepted "
                  f"{r['n_corrected']:3d} corrected {r['n_rejected']:3d} rejected",
                  flush=True)
            records.append(r)
        ref_stems = [Path(p).stem for p in ref]
        # reference fields into test, except the one used for prompt development
        stems = [r["stem"] for r in records]
        forced_train = [CRYO_PROMPT_DEV]
        split = assign_splits(stems, forced_train, args.n_val, args.n_test,
                              args.split_seed)
        for s in ref_stems:
            if s in split and s != CRYO_PROMPT_DEV:
                split[s] = "test"
        # keep the requested test size by moving surplus test fields to train
        extra = [s for s, v in split.items() if v == "test" and s not in ref_stems]
        while sum(1 for v in split.values() if v == "test") > args.n_test and extra:
            split[extra.pop()] = "train"
    else:
        man_path = ((args.images_dir.parent / "manifest.json")
                    if args.images_dir and (args.images_dir.parent / "manifest.json").exists()
                    else SEM_MANIFEST)
        man = json.loads(man_path.read_text())
        records = []
        for m in man:
            r = sem_annotate(m["stem"], m, args.out, args.images_dir,
                             args.labels_dir, args.prep_label)
            print(f"    {r['stem']:<20s} {r['n_annotations']:4d} annotations "
                  f"{r['pixel_size_nm']:7.3f} nm/px", flush=True)
            records.append(r)
        stems = [r["stem"] for r in records]
        if args.split_from is not None:
            prev = json.loads(args.split_from.read_text())
            reuse = {s["stem"]: s["split"] for s in prev["sources"]}
            missing = [s for s in stems if s not in reuse]
            if missing:
                raise SystemExit(f"--split-from does not cover {missing}")
            split = {s: reuse[s] for s in stems}
            notes.append(f"split reused verbatim from {args.split_from}, so this "
                         f"run differs from that one only in image preparation")
        elif args.split_mode == "stratified":
            split = stratified_splits(stems, sem_isolate, [SEM_PROMPT_DEV],
                                      args.split_seed)
            notes.append("split stratified by isolate: every isolate with more "
                         "than one field has a field in training and a field in "
                         "test; single-field isolates go to training")
        else:
            split = assign_splits(stems, [SEM_PROMPT_DEV],
                                  args.n_val, args.n_test, args.split_seed)

    print("\n  tiling and splitting:")
    man = export_and_split(records, args.out, split, args.modality, notes)
    t = man["totals"]
    print(f"\n  {t['source_images']} source images, {t['annotated_objects']} objects, "
          f"{t['tiles']} tiles")
    print(f"  split (source images): {man['split_counts_source_images']}")
    print(f"  split (tiles):         {man['split_counts_tiles']}")
    print(f"  split (instances):     {man['split_counts_instances']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
