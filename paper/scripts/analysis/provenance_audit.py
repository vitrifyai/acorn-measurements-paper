#!/usr/bin/env python3
"""Provenance audit of one completed ACORN project, end to end.

The project audited here is the SEM bacterial-spore project: 26 acquired
micrographs annotated three ways, reviewed, measured, analysed, and plotted.
Every annotation carries the provenance block ACORN stamps at creation, so the
counts below are read out of the record rather than reconstructed:

  1. the operator's own annotations, imported from their ACORN sidecars
  2. SAM 3 text-prompted candidates for what the import missed
  3. predictions from the task-specific model trained in this work

Each candidate from steps 2 and 3 is reviewed under one documented rule set and
ends as accepted unchanged, corrected, or rejected. Accepted annotations are
measured at the calibrated pixel size, the measurements are analysed by isolate,
and the analysis produces one figure. One plotted value is then traced all the
way back to its source micrograph and that micrograph's calibration.

    python provenance_audit.py --work work/sem --checkpoint <best.pt> \
        --runs results/sem/training_runs.json --out provenance --device 0
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
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROMPT = "oval object"            # verbatim, as in the manuscript's figure 3B
CONF = 0.5
ACCEPT_AREA_UM2 = (0.05, 10.0)    # review window for a single spore
SOLIDITY_REPAIR = 0.90
DUP_IOU = 0.50                    # a candidate this close to an accepted mask is a duplicate

STRAIN_RE = re.compile(r"^([A-Za-z0-9\-]+?)_?\d{3}$")

REVIEW_RULES = {
    "accepted": "kept with its geometry unchanged",
    "corrected": f"solidity below {SOLIDITY_REPAIR}; outline replaced by its convex "
                 f"hull through AnnotationStore.update, which increments "
                 f"modification_count and makes the current geometry hash differ "
                 f"from geometry_hash_at_creation",
    "rejected": f"projected area outside {ACCEPT_AREA_UM2[0]}-{ACCEPT_AREA_UM2[1]} um^2, "
                f"or mask IoU above {DUP_IOU} with an annotation already accepted "
                f"for that micrograph (a duplicate of an object already recorded)",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
    return h.hexdigest()


def git_sha(repo=ACORN_SOURCE_DIR) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def rasterise(verts, h, w):
    from skimage.draw import polygon as skpoly
    v = np.asarray(verts, float)
    m = np.zeros((h, w), dtype=bool)
    rr, cc = skpoly(v[:, 1], v[:, 0], shape=(h, w))
    m[rr, cc] = True
    return m


def iou(a, b) -> float:
    inter = np.count_nonzero(a & b)
    return inter / float(np.count_nonzero(a | b)) if inter else 0.0


def mask_outline(m, max_pts=200):
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
    a = abs(np.dot(v[:, 0], np.roll(v[:, 1], -1))
            - np.dot(v[:, 1], np.roll(v[:, 0], -1))) / 2
    return float(a / hull.volume) if hull.volume > 0 else 1.0


def convex_hull_verts(verts):
    from scipy.spatial import ConvexHull
    v = np.asarray(verts, float)
    return [(float(x), float(y)) for x, y in v[ConvexHull(v).vertices]]


def tiles_for(h, w, tile, overlap):
    if h < tile or w < tile:
        return [(0, 0)]
    stride = max(1, int(tile * (1 - overlap)))
    ys, y0 = [], 0
    while True:
        ys.append(min(y0, h - tile))
        if y0 + tile >= h:
            break
        y0 += stride
    xs, x0 = [], 0
    while True:
        xs.append(min(x0, w - tile))
        if x0 + tile >= w:
            break
        x0 += stride
    return [(y, x) for y in dict.fromkeys(ys) for x in dict.fromkeys(xs)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    from ultralytics import YOLO
    from acorn.core.annotations import AnnotationStore, ROIAnnotation
    from acorn.core.dm4_loader import DM4Image
    from acorn.core.measurements import polygon_metrics
    from acorn.core import provenance as prov
    from acorn.core.sam_predictor import SAMPredictor

    man = json.loads((args.work / "split_manifest.json").read_text())
    runs = json.loads(args.runs.read_text())
    run = next(r for r in runs["runs"] if Path(r["checkpoint"]) == args.checkpoint)

    ds = args.work / "dataset"
    dataset_version = sha256_text((ds / "annotations.json").read_text(),
                                  (ds / "splits/split_map.json").read_text(),
                                  (args.work / "split_manifest.json").read_text())[:16]
    ckpt_hash_full = sha256_file(args.checkpoint)
    model = YOLO(str(args.checkpoint))
    sam = SAMPredictor(backend="sam3")
    sam.load_model()
    sproc = sam._sam3_processor

    annotations: list[dict] = []
    measurements: list[dict] = []
    review_log: list[dict] = []
    images: list[dict] = []

    for s in man["sources"]:
        stem = s["stem"]
        prepared = Path(s["prepared"])
        px = float(s["pixel_size_nm"])
        img8 = np.array(Image.open(prepared).convert("L"))
        h, w = img8.shape[:2]
        loaded = DM4Image.from_file(prepared)
        loaded.meta.pixel_size = px

        images.append({
            "image_id": stem,
            "source_file": s["source"],
            "source_sha256": s["source_sha256"],
            "prepared_file": str(prepared),
            "shape": [h, w],
            "pixel_size_nm": px,
            "pixel_size_source": s["pixel_size_source"],
            "split": s["split"],
        })

        store = AnnotationStore()
        accepted_masks: list[np.ndarray] = []

        def review(cand_verts, origin, source_model, parent_id=None):
            """One documented review decision, recorded in the provenance block."""
            m = polygon_metrics([(float(x), float(y)) for x, y in cand_verts], px)
            a_um2 = m["area_nm2"] / 1e6
            mask = rasterise(cand_verts, h, w)
            if not (ACCEPT_AREA_UM2[0] <= a_um2 <= ACCEPT_AREA_UM2[1]):
                return "rejected", "outside the area window", None
            dup = max((iou(mask, k) for k in accepted_masks), default=0.0)
            if dup > DUP_IOU:
                return "rejected", f"duplicate of an accepted annotation (IoU {dup:.2f})", None
            with prov.provenance_context(origin,
                                         invocation=prov.Invocation.DIRECT_GUI.value,
                                         source_model=source_model,
                                         parent_id=parent_id):
                ann = ROIAnnotation(vertices=[(float(x), float(y)) for x, y in cand_verts],
                                    label="Spore")
                store.add(ann)
                outcome = "accepted"
                if solidity(cand_verts) < SOLIDITY_REPAIR:
                    ann.vertices = convex_hull_verts(cand_verts)
                    store.update(ann)
                    outcome = "corrected"
                    mask = rasterise(ann.vertices, h, w)
            accepted_masks.append(mask)
            return outcome, "", ann

        # ── 1. the operator's own annotations, imported ───────────────────────
        n_imported = 0
        with prov.provenance_context(prov.Origin.IMPORTED_SIDECAR.value,
                                     invocation=prov.Invocation.BATCH.value):
            sc = json.loads((prepared.parent / f".{prepared.stem}.acorn.json").read_text())
            for a in sc["annotations"]:
                if a.get("type") != "roi":
                    continue
                ann = ROIAnnotation(vertices=[(float(x), float(y)) for x, y in a["vertices"]],
                                    label=a.get("label") or "Spore")
                store.add(ann)
                accepted_masks.append(rasterise(ann.vertices, h, w))
                n_imported += 1

        # ── 2. SAM 3 text-prompted candidates ────────────────────────────────
        state = sproc.set_image(Image.fromarray(np.repeat(img8[:, :, None], 3, 2)))
        sproc.set_confidence_threshold(CONF)
        masks = sproc.set_text_prompt(PROMPT, state).get("masks")
        n_sam = {"accepted": 0, "corrected": 0, "rejected": 0}
        for x in (masks if masks is not None else []):
            m = np.asarray(x.squeeze().float().cpu().numpy() > 0.5)
            if m.sum() < 4:
                continue
            verts = mask_outline(m)
            if verts is None or len(verts) < 3:
                continue
            outcome, why, _ = review(verts, prov.Origin.SAM.value,
                                     {"path": "sam3", "sha256": None,
                                      "prompt": PROMPT, "confidence": CONF})
            n_sam[outcome] += 1
            review_log.append({"image_id": stem, "origin": "sam", "outcome": outcome,
                               "reason": why})

        # ── 3. predictions from the model trained in this work ───────────────
        rgb = np.repeat(img8[:, :, None], 3, 2)
        n_yolo = {"accepted": 0, "corrected": 0, "rejected": 0}
        for (y0, x0) in tiles_for(h, w, 512, 0.25):
            crop = rgb[y0:y0 + 512, x0:x0 + 512]
            if crop.shape[0] < 512 or crop.shape[1] < 512:
                crop = np.pad(crop, ((0, 512 - crop.shape[0]), (0, 512 - crop.shape[1]),
                                     (0, 0)), mode="reflect")
            r = model.predict(crop, verbose=False, conf=CONF, device=args.device,
                              imgsz=512)[0]
            if r.masks is None:
                continue
            for xy in r.masks.xy:
                if len(xy) < 3:
                    continue
                verts = (np.asarray(xy, float) + np.array([x0, y0])).tolist()
                outcome, why, _ = review(verts, prov.Origin.YOLO.value,
                                         {"path": str(args.checkpoint),
                                          "sha256": ckpt_hash_full[:16],
                                          "sha256_full": ckpt_hash_full,
                                          "seed": run["seed"],
                                          "dataset_version": dataset_version})
                n_yolo[outcome] += 1
                review_log.append({"image_id": stem, "origin": "yolo",
                                   "outcome": outcome, "reason": why})

        # ── record every annotation, with its provenance as stored ───────────
        for ann in store:
            p = ann.provenance
            annotations.append({
                "annotation_id": p.id,
                "image_id": stem,
                "origin": p.origin,
                "invocation": p.invocation,
                "created_at": p.created_at,
                "created_by": p.created_by,
                "modification_count": p.modification_count,
                "reviewed": p.reviewed,
                "geometry_hash_at_creation": p.geometry_hash_at_creation,
                "geometry_hash_current": prov.geometry_hash(ann),
                "edited_after_creation": (p.geometry_hash_at_creation is not None
                                          and prov.geometry_hash(ann)
                                          != p.geometry_hash_at_creation),
                "batch_id": p.batch_id,
                "parent_id": p.parent_id,
                "source_model": p.source_model,
                "n_vertices": len(ann.vertices),
                # kept so a traced annotation can be drawn from the audit record
                # alone, without going back to the store that produced it
                "vertices": [[round(float(x), 2), round(float(y), 2)]
                             for x, y in ann.vertices],
            })
            m = polygon_metrics([(float(x), float(y)) for x, y in ann.vertices], px)
            measurements.append({
                "measurement_id": "M-" + sha256_text(p.id, stem)[:12],
                "annotation_id": p.id,
                "image_id": stem,
                "pixel_size_nm": px,
                "pixel_size_source": s["pixel_size_source"],
                "area_nm2": m["area_nm2"], "ecd_nm": m["ecd_nm"],
                "feret_nm": m["feret_nm"], "circularity": m["circularity"],
            })

        print(f"  {stem:<16s} imported {n_imported:4d}  sam {n_sam}  yolo {n_yolo}",
              flush=True)

    # ── statistical analysis on the measurement set ──────────────────────────
    from scipy import stats as st

    def strain(stem):
        m = STRAIN_RE.match(stem)
        return m.group(1) if m else stem

    groups: dict[str, list] = {}
    for r in measurements:
        groups.setdefault(strain(r["image_id"]), []).append(r)
    top = sorted(groups, key=lambda k: -len(groups[k]))[:5]
    samples = [[r["ecd_nm"] for r in groups[g]] for g in top]
    kw = st.kruskal(*samples)
    analysis_id = "A-" + sha256_text(json.dumps(top),
                                     json.dumps([len(x) for x in samples]))[:12]
    analysis = {
        "analysis_id": analysis_id,
        "kind": "Kruskal-Wallis on equivalent circular diameter by isolate",
        "groups": top,
        "n_per_group": {g: len(groups[g]) for g in top},
        "statistic": float(kw.statistic), "p_value": float(kw.pvalue),
        "median_ecd_nm": {g: float(np.median([r["ecd_nm"] for r in groups[g]]))
                          for g in top},
        "measurement_ids": [r["measurement_id"] for g in top for r in groups[g]],
        "dataset_version": dataset_version,
    }

    # ── the plotted values that get traced ──────────────────────────────────
    # Two traces. The first is the object nearest its group's median in the most
    # populous group -- a value a reader can point at, chosen by a rule and not
    # by what it happens to link to. Whatever that object's origin is, is what
    # the record says. The second is the nearest-median object among the ones
    # the trained model produced, so the checkpoint, training run and dataset
    # version links are shown with real values as well.
    g0 = top[0]
    med = analysis["median_ecd_nm"][g0]
    ann_by_id = {a["annotation_id"]: a for a in annotations}
    img_by_id = {i["image_id"]: i for i in images}

    def build_trace(rec, rule):
        a = ann_by_id[rec["annotation_id"]]
        img = img_by_id[rec["image_id"]]
        is_model = a["origin"] == "yolo"
        return {
            "plot_point": {
                "figure": "provenance_audit_figure.pdf",
                "panel": "diameter by isolate",
                "group": g0,
                "value_nm": rec["ecd_nm"],
                "selection_rule": rule,
            },
            "statistical_analysis_record": {
                "analysis_id": analysis_id, "kind": analysis["kind"],
                "p_value": analysis["p_value"], "n_in_group": len(groups[g0]),
                "group_median_nm": med,
            },
            "measurement_record": rec,
            "annotation_record": a,
            "prediction": a["source_model"],
            "checkpoint": {
                "path": str(args.checkpoint), "sha256": ckpt_hash_full,
                "bytes": args.checkpoint.stat().st_size,
            } if is_model else None,
            "training_run": {
                "seed": run["seed"],
                "architecture": runs["hyperparameters"]["architecture"],
                "epochs": runs["hyperparameters"]["epochs"],
                "batch": runs["hyperparameters"]["batch"],
                "imgsz": runs["hyperparameters"]["imgsz"],
                "training_seconds": run.get("training_seconds"),
            } if is_model else None,
            "dataset_version": {
                "id": dataset_version,
                "definition": "sha256 over the dataset's annotations.json, its "
                              "splits/split_map.json and the build manifest",
                "partitions": man["split_counts_source_images"],
            },
            "source_image": img,
            "calibration": {
                "pixel_size_nm": img["pixel_size_nm"],
                "source": img["pixel_size_source"],
            },
        }

    traced = min(groups[g0], key=lambda r: abs(r["ecd_nm"] - med))
    trace = build_trace(traced, "the object nearest its group's median diameter")

    model_in_group = [r for r in groups[g0]
                      if ann_by_id[r["annotation_id"]]["origin"] == "yolo"]
    trace_model = None
    if model_in_group:
        tm = min(model_in_group, key=lambda r: abs(r["ecd_nm"] - med))
        trace_model = build_trace(
            tm, "the model-generated object nearest its group's median diameter")

    # ── counts for the audit table ───────────────────────────────────────────
    def count(origin, edited=None):
        return sum(1 for a in annotations
                   if a["origin"] == origin
                   and (edited is None or a["edited_after_creation"] == edited))

    n_total = len(annotations)
    rejected = {"sam": sum(1 for r in review_log
                           if r["origin"] == "sam" and r["outcome"] == "rejected"),
                "yolo": sum(1 for r in review_log
                            if r["origin"] == "yolo" and r["outcome"] == "rejected")}
    counts = {
        "total_annotations": n_total,
        "total_source_images": len(images),
        "by_origin": {o: count(o) for o in sorted({a["origin"] for a in annotations})},
        "manual": count("manual"),
        "prompt_generated_accepted": count("sam", edited=False),
        "prompt_generated_corrected": count("sam", edited=True),
        "prompt_generated_rejected": rejected["sam"],
        "model_generated_accepted": count("yolo", edited=False),
        "model_generated_corrected": count("yolo", edited=True),
        "model_generated_rejected": rejected["yolo"],
        "imported": count("imported_sidecar"),
        "measurements": len(measurements),
        "statistical_outputs": 1,
    }
    counts["percentages"] = {k: (100.0 * v / n_total if n_total else 0.0)
                             for k, v in counts["by_origin"].items()}

    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": "SEM bacterial spores, 26 acquired micrographs",
        "acorn_commit": git_sha(),
        "dataset_version": dataset_version,
        "checkpoint_sha256": ckpt_hash_full,
        "review_rules": REVIEW_RULES,
        "origin_categories": {
            "manual": "drawn on the canvas with no model involved",
            "sam": "SAM 3 text-prompted segmentation",
            "yolo": "the task-specific model trained in this work",
            "imported_sidecar": "read from an ACORN annotation sidecar written by "
                                "the operator in an earlier session",
        },
        "geometry_serialisation": "acorn.core.provenance.geometry_signature: the "
                                 "annotation type followed by its vertices rounded "
                                 "to 1e-3 px, JSON-serialised with sorted keys",
        "hashing": "sha256, truncated to 16 hex characters for geometry and to a "
                   "size+first-MB+last-MB fingerprint for checkpoints; the full "
                   "sha256 of the checkpoint is recorded here as well",
        "counts": counts,
        "analysis": analysis,
        "trace": trace,
        "trace_model_generated": trace_model,
        "images": images,
    }
    (args.out / "provenance_audit.json").write_text(json.dumps(audit, indent=1))
    (args.out / "annotations.json").write_text(json.dumps(annotations, indent=1))
    (args.out / "measurements.json").write_text(json.dumps(measurements, indent=1))
    (args.out / "review_log.json").write_text(json.dumps(review_log, indent=1))

    import csv as _csv
    with open(args.out / "measurements.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(measurements[0]))
        w.writeheader()
        w.writerows(measurements)

    print("\n  counts:", json.dumps(counts, indent=1))
    print("  traced:", trace["plot_point"], "->", trace["annotation_record"]["annotation_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
