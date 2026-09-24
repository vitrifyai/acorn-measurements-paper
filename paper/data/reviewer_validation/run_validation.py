#!/usr/bin/env python3
"""Frozen reviewer validation for the ACORN synthetic TEM and SEM datasets.

This script only reads existing datasets/checkpoints and writes beneath its
own output directory. Evaluation settings reproduce the recorded runs; the
object-level benchmark uses a prespecified greedy class-matched mask-IoU rule.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("ACORN_WORK_DIR", REPO_ROOT / "work"))
DOI_ROOT = Path(
    os.environ.get("ACORN_DOI_DIR", REPO_ROOT / "external" / "doi")
)
RUNS = DOI_ROOT / "models"
BOOTSTRAP_SEED = 20260921
BOOTSTRAP_REPS = 2000
PREDICTION_CONFIDENCE = 0.5
MATCH_IOU = 0.5

DATASETS = {
    "TEM": {
        "slug": "nanoparticles_tem",
        "root": ROOT / "sim/nanoparticles_tem",
        "checkpoint": RUNS / "nanoparticles_tem/best.pt",
        "reported": ROOT / "models/nanoparticles_tem_test_metrics.json",
        "target": "nanoparticle",
    },
    "SEM": {
        "slug": "spores_sem",
        "root": ROOT / "sim/spores_sem",
        "checkpoint": RUNS / "spores_sem/best.pt",
        "reported": ROOT / "models/spores_sem_test_metrics.json",
        "target": "spore",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_manifest(rows: list[dict], path: Path, role: str, modality: str = "",
                 partition: str = "") -> None:
    resolved = path.resolve()
    rows.append({
        "role": role,
        "modality": modality,
        "partition": partition,
        "path": str(resolved),
        "bytes": resolved.stat().st_size if resolved.exists() else "",
        "sha256": sha256(resolved) if resolved.is_file() else "",
        "status": "present" if resolved.exists() else "missing",
    })


def build_input_manifest(out: Path) -> list[dict]:
    rows: list[dict] = []
    for modality, cfg in DATASETS.items():
        add_manifest(rows, cfg["checkpoint"], "selected_checkpoint", modality)
        add_manifest(rows, cfg["reported"], "reported_metrics", modality)
        add_manifest(rows, cfg["root"] / "data.yaml", "dataset_configuration", modality)
        add_manifest(rows, cfg["root"] / "manifest_val.json", "model_selection_manifest", modality, "val")
        add_manifest(rows, cfg["root"] / "manifest_test.json", "untouched_test_manifest", modality, "test")
        for kind in ("images", "labels"):
            for path in sorted((cfg["root"] / kind / "test").glob("*")):
                if path.is_file():
                    add_manifest(rows, path, f"test_{kind[:-1]}", modality, "test")
    for path, role in (
        (ROOT / "additional_info/work/sem/split_manifest.json", "experimental_split_manifest"),
        (ROOT / "additional_info/work/cryo/split_manifest.json", "experimental_split_manifest"),
        (ROOT / "real_ft/split.json", "initialization_split_manifest"),
        (ROOT / "sim_all_runs.json", "initialization_results"),
        (ROOT / "additional_info/provenance/provenance_audit.json", "provenance_audit"),
        (ROOT / "figures_v2/fig5/fig5_replicates.json", "motion_correction_results"),
    ):
        add_manifest(rows, path, role)
    write_csv(out / "INPUT_MANIFEST.csv", rows,
              ["role", "modality", "partition", "path", "bytes", "sha256", "status"])
    return rows


def polygon_mask(poly: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(poly) >= 3:
        cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 1)
    return mask.astype(bool)


def load_truth(label_path: Path, shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    objects = []
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 7:
            continue
        cls = int(fields[0])
        vals = np.asarray([float(x) for x in fields[1:]], dtype=float).reshape(-1, 2)
        vals[:, 0] *= w
        vals[:, 1] *= h
        objects.append({"class_id": cls, "polygon": vals,
                        "mask": polygon_mask(vals, shape)})
    return objects


def measure(poly: np.ndarray, px_nm: float) -> dict[str, float]:
    from acorn.core.measurements import polygon_metrics
    metrics = polygon_metrics([(float(x), float(y)) for x, y in poly], px_nm)
    return {
        "area_nm2": float(metrics["area_nm2"]),
        "ecd_nm": float(metrics["ecd_nm"]),
        "feret_nm": float(metrics["feret_nm"]),
        "circularity": float(metrics["circularity"]),
    }


def greedy_match(pred: list[dict], truth: list[dict]) -> list[tuple[int, int, float]]:
    candidates = []
    for pi, p in enumerate(pred):
        for ti, t in enumerate(truth):
            if p["class_id"] != t["class_id"]:
                continue
            inter = np.logical_and(p["mask"], t["mask"]).sum()
            union = np.logical_or(p["mask"], t["mask"]).sum()
            iou = float(inter / union) if union else 0.0
            if iou >= MATCH_IOU:
                candidates.append((iou, pi, ti))
    used_p, used_t, matched = set(), set(), []
    for iou, pi, ti in sorted(candidates, reverse=True):
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
            matched.append((pi, ti, iou))
    return matched


def metric_scope(result, scope: str, target_class: int = 0) -> dict:
    box, seg = result.box, result.seg
    if scope == "all_classes_macro":
        bp, br = float(box.mp), float(box.mr)
        sp, sr = float(seg.mp), float(seg.mr)
        bm50, bm = float(box.map50), float(box.map)
        sm50, sm = float(seg.map50), float(seg.map)
    else:
        # Ultralytics class_result is [precision, recall, AP50, AP50-95].
        bp, br, bm50, bm = map(float, box.class_result(target_class))
        sp, sr, sm50, sm = map(float, seg.class_result(target_class))
    return {
        "box_precision": bp,
        "box_recall": br,
        "box_map50": bm50,
        "box_map50_95": bm,
        "precision": sp,
        "recall": sr,
        "f1": 2 * sp * sr / (sp + sr) if sp + sr else 0.0,
        "mask_map50": sm50,
        "mask_map50_95": sm,
    }


def bootstrap_ci(rows: list[dict], value_fn, rng: np.random.Generator) -> tuple[float, float]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_image[row["stem"]].append(row)
    names = sorted(by_image)
    if len(names) < 2:
        return float("nan"), float("nan")
    values = []
    for _ in range(BOOTSTRAP_REPS):
        sampled = rng.choice(names, size=len(names), replace=True)
        sample = [row for name in sampled for row in by_image[name]]
        values.append(value_fn(sample))
    return tuple(map(float, np.percentile(values, [2.5, 97.5])))


def summarize_measurements(pairs: list[dict]) -> list[dict]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups: list[tuple[str, str, list[dict]]] = []
    for modality in ("TEM", "SEM"):
        subset = [r for r in pairs if r["modality"] == modality]
        groups.append((modality, "all", subset))
        for condition in sorted({r["condition"] for r in subset}):
            groups.append((modality, condition,
                           [r for r in subset if r["condition"] == condition]))
    output = []
    for modality, condition, rows in groups:
        for metric in ("area_nm2", "ecd_nm", "feret_nm", "circularity"):
            errors = np.asarray([r[f"pred_{metric}"] - r[f"truth_{metric}"] for r in rows])
            truth = np.asarray([r[f"truth_{metric}"] for r in rows])
            if not len(errors):
                continue
            abs_err = np.abs(errors)
            rel = abs_err / np.maximum(np.abs(truth), 1e-12) * 100
            bias_ci = bootstrap_ci(rows, lambda x: float(np.mean([
                r[f"pred_{metric}"] - r[f"truth_{metric}"] for r in x])), rng)
            mae_ci = bootstrap_ci(rows, lambda x: float(np.mean([
                abs(r[f"pred_{metric}"] - r[f"truth_{metric}"]) for r in x])), rng)
            output.append({
                "component": "segmentation_boundary",
                "modality": modality,
                "condition": condition,
                "measurement": metric,
                "n_source_images": len({r["stem"] for r in rows}),
                "n_matched_objects": len(rows),
                "bias": float(errors.mean()),
                "bias_ci95_low": bias_ci[0],
                "bias_ci95_high": bias_ci[1],
                "mae": float(abs_err.mean()),
                "mae_ci95_low": mae_ci[0],
                "mae_ci95_high": mae_ci[1],
                "median_absolute_error": float(np.median(abs_err)),
                "mean_absolute_relative_error_pct": float(rel.mean()),
            })
        px_errors = [float(r["calibration_relative_error_pct"]) for r in rows]
        output.append({
            "component": "calibration_retention",
            "modality": modality,
            "condition": condition,
            "measurement": "pixel_size_nm",
            "n_source_images": len({r["stem"] for r in rows}),
            "n_matched_objects": len(rows),
            "bias": float(np.mean(px_errors)) if px_errors else float("nan"),
            "bias_ci95_low": 0.0 if px_errors else float("nan"),
            "bias_ci95_high": 0.0 if px_errors else float("nan"),
            "mae": float(np.mean(np.abs(px_errors))) if px_errors else float("nan"),
            "mae_ci95_low": 0.0 if px_errors else float("nan"),
            "mae_ci95_high": 0.0 if px_errors else float("nan"),
            "median_absolute_error": float(np.median(np.abs(px_errors))) if px_errors else float("nan"),
            "mean_absolute_relative_error_pct": float(np.mean(np.abs(px_errors))) if px_errors else float("nan"),
        })
    return output


def run_synthetic(out: Path, device: str) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    from ultralytics import YOLO
    aggregate, per_image, predictions, pairs = [], [], [], []
    for modality, cfg in DATASETS.items():
        manifest = json.loads((cfg["root"] / "manifest_test.json").read_text())
        metadata = {row["stem"]: row for row in manifest}
        model = YOLO(str(cfg["checkpoint"]))
        result = model.val(
            data=str((cfg["root"] / "data.yaml").resolve()),
            split="test", device=device, plots=False, verbose=False,
            project=str(out / "ultralytics"), name=cfg["slug"], exist_ok=False,
        )
        reported = json.loads(cfg["reported"].read_text())
        for scope in ("all_classes_macro", "target_class"):
            row = {
                "row_type": "aggregate",
                "modality": modality,
                "dataset": cfg["slug"],
                "scope": scope,
                "n_images": len(manifest),
                "n_objects": sum(int(r["n_instances"]) for r in manifest),
                "n_target_objects": sum(int(r.get("n_particles", r.get("n_spores_labelled", 0)))
                                        for r in manifest),
                "checkpoint": str(cfg["checkpoint"]),
                "checkpoint_sha256": sha256(cfg["checkpoint"]),
                **metric_scope(result, scope),
            }
            if scope == "all_classes_macro":
                row["reported_mask_map50"] = reported["mask_map50"]
                row["reported_mask_map50_95"] = reported["mask_map5095"]
                row["reproduced_within_1e-6"] = all(
                    abs(row[key] - reported[reported_key]) <= 1e-6
                    for key, reported_key in (
                        ("box_map50", "box_map50"),
                        ("box_map50_95", "box_map5095"),
                        ("mask_map50", "mask_map50"),
                        ("mask_map50_95", "mask_map5095"),
                    )
                )
            aggregate.append(row)

        image_paths = sorted((cfg["root"] / "images/test").glob("*.png"))
        results = model.predict([str(p) for p in image_paths], conf=PREDICTION_CONFIDENCE,
                                device=device, verbose=False, stream=True)
        for image_path, pred_result in zip(image_paths, results):
            image = np.asarray(Image.open(image_path).convert("L"))
            shape = image.shape
            truth = load_truth(cfg["root"] / "labels/test" / f"{image_path.stem}.txt", shape)
            pred = []
            if pred_result.masks is not None:
                classes = pred_result.boxes.cls.cpu().numpy().astype(int)
                confs = pred_result.boxes.conf.cpu().numpy()
                boxes = pred_result.boxes.xyxy.cpu().numpy()
                for idx, (poly, cls, conf, box) in enumerate(
                        zip(pred_result.masks.xy, classes, confs, boxes)):
                    poly = np.asarray(poly, dtype=float)
                    item = {"class_id": int(cls), "confidence": float(conf),
                            "polygon": poly, "mask": polygon_mask(poly, shape),
                            "box": box}
                    pred.append(item)
            matches = greedy_match(pred, truth)
            matched_pred = {p for p, _, _ in matches}
            matched_truth = {t for _, t, _ in matches}
            meta = metadata[image_path.stem]
            px_nm = float(meta["px_a"]) / 10.0 if modality == "TEM" else float(meta["pixel_size_nm"])
            condition = (f"dose={float(meta['dose']):g} e-/A2" if modality == "TEM"
                         else ("coated" if float(meta["coating_nm"]) > 0
                               else "uncoated+charging" if float(meta["charging"]) > 0
                               else "uncoated"))
            target_truth = sum(t["class_id"] == 0 for t in truth)
            target_pred = sum(p["class_id"] == 0 for p in pred)
            per_image.append({
                "row_type": "per_image_count",
                "modality": modality,
                "dataset": cfg["slug"],
                "scope": "target_class",
                "stem": image_path.stem,
                "condition": condition,
                "n_truth": target_truth,
                "n_pred": target_pred,
                "count_error": target_pred - target_truth,
                "count_error_pct": 100 * (target_pred - target_truth) / max(target_truth, 1),
                "n_matched_iou50": sum(pred[p]["class_id"] == 0 for p, _, _ in matches),
                "fixed_confidence": PREDICTION_CONFIDENCE,
            })
            for idx, p in enumerate(pred):
                measurements = measure(p["polygon"], px_nm)
                predictions.append({
                    "modality": modality, "stem": image_path.stem,
                    "prediction_id": idx, "class_id": p["class_id"],
                    "confidence": p["confidence"], "matched": idx in matched_pred,
                    "pixel_size_nm": px_nm,
                    "bbox_xyxy": json.dumps([round(float(v), 4) for v in p["box"]]),
                    "polygon_xy": json.dumps(np.round(p["polygon"], 3).tolist()),
                    **measurements,
                })
            for pi, ti, iou in matches:
                if pred[pi]["class_id"] != 0:
                    continue
                pm, tm = measure(pred[pi]["polygon"], px_nm), measure(truth[ti]["polygon"], px_nm)
                pairs.append({
                    "modality": modality, "stem": image_path.stem,
                    "condition": condition, "pred_id": pi, "truth_id": ti,
                    "mask_iou": iou, "pixel_size_nm": px_nm,
                    "calibration_relative_error_pct": 0.0,
                    **{f"pred_{k}": v for k, v in pm.items()},
                    **{f"truth_{k}": v for k, v in tm.items()},
                })
    return aggregate, per_image, predictions, pairs


def initialization_rows() -> list[dict]:
    source = ROOT / "sim_all_runs.json"
    if not source.exists():
        return [{"status": "blocked", "reason": "sim_all_runs.json missing"}]
    rows = json.loads(source.read_text())
    output = []
    for row in rows:
        copied = dict(row)
        copied["status"] = "legacy_result_not_independent_simulation_pretraining"
        copied["independent_simulation_seed_range"] = False
        copied["reason"] = (
            "Simulation-start runs reuse one spores_sem pretrained checkpoint; "
            "they vary fine-tuning subset/seed, not independently generated simulation pretraining."
        )
        output.append(copied)
    return output


def versions() -> dict:
    packages = {}
    for name in ("acorn", "ultralytics", "torch", "numpy", "opencv-python", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=False, capture_output=True, text=True,
        ).stdout.splitlines()[0]
    except Exception:
        gpu = "unavailable"
    return {"python": sys.version.replace("\n", " "), "platform": platform.platform(),
            "packages": packages, "gpu": gpu}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    protected = [
        "INPUT_MANIFEST.csv", "synthetic_test_metrics.csv",
        "synthetic_predictions.csv", "measurement_pairs.csv",
        "measurement_accuracy.csv", "initialization_replicates.csv",
        "run_record.json",
    ]
    existing = [args.out / name for name in protected if (args.out / name).exists()]
    if existing:
        raise SystemExit("Refusing to overwrite existing outputs: " +
                         ", ".join(str(path) for path in existing))

    manifest = build_input_manifest(args.out)
    missing = [r for r in manifest if r["status"] == "missing"]
    if missing:
        (args.out / "BLOCKED.json").write_text(json.dumps(missing, indent=2))
        raise SystemExit("Required input missing; see BLOCKED.json")

    aggregate, per_image, predictions, pairs = run_synthetic(args.out, args.device)
    write_csv(args.out / "synthetic_test_metrics.csv", aggregate + per_image)
    write_csv(args.out / "synthetic_predictions.csv", predictions)
    write_csv(args.out / "measurement_pairs.csv", pairs)
    summaries = summarize_measurements(pairs)
    write_csv(args.out / "measurement_accuracy.csv", summaries)
    write_csv(args.out / "initialization_replicates.csv", initialization_rows())

    run_record = {
        "generated_utc": utc_now(),
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "seeds": {"bootstrap": BOOTSTRAP_SEED},
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "fixed_prediction_confidence": PREDICTION_CONFIDENCE,
        "matching": "greedy one-to-one, descending mask IoU, class matched",
        "matching_iou_threshold": MATCH_IOU,
        "software": versions(),
        "checkpoint_hashes": {m: sha256(c["checkpoint"]) for m, c in DATASETS.items()},
        "input_manifest_sha256": sha256(args.out / "INPUT_MANIFEST.csv"),
    }
    (args.out / "run_record.json").write_text(json.dumps(run_record, indent=2))
    print(json.dumps({"output": str(args.out), "aggregate": aggregate,
                      "matched_pairs": len(pairs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
