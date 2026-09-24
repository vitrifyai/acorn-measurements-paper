#!/usr/bin/env python3
"""Score the trained models on held-out micrographs, per source field.

Tile-level mAP is what a detector trainer reports; it is not what a microscopist
gets. This scores whole held-out micrographs: tiled inference, predictions
mapped back to the source frame with the tile offsets, duplicates across the
overlap merged, then matched to the reference annotations at IoU 0.5.

Also written, per modality:
  * timing: inference seconds per tile and per whole micrograph
  * cryo-TEM only: the same predictions scored against the independent
    CryoBLOB verified particle list, which was never used for training
  * SEM only: the same predictions scored against the confident-spore subset
    of the reference, as a bracket on the reference's own composition

    python evaluate_heldout.py --modality sem --work work/sem \
        --runs results/sem/training_runs.json --out results/sem --device 0
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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

IOU_MATCH = 0.5          # a prediction matches a reference at IoU >= this
MERGE_IOU = 0.60         # two predictions from overlapping tiles are the same object
# Containment, not IoU, is what catches a tile-boundary fragment. A mask cut in
# half by a tile edge has an IoU of only about 0.5 with the whole object seen in
# the neighbouring tile, so an IoU rule alone leaves the half behind as a second
# detection sitting on top of the first. Its intersection over its OWN area is
# near 1, which is the quantity that identifies it as a piece of something
# already detected rather than a new object.
MERGE_CONTAINMENT = 0.70
CONF = 0.5               # detection confidence threshold
TILE = 512
OVERLAP = 0.25


def tiles_for(h: int, w: int, tile: int, overlap: float):
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


def rasterise(verts, h, w) -> np.ndarray:
    from skimage.draw import polygon as skpoly
    v = np.asarray(verts, float)
    m = np.zeros((h, w), dtype=bool)
    rr, cc = skpoly(v[:, 1], v[:, 0], shape=(h, w))
    m[rr, cc] = True
    return m


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    return inter / float(np.count_nonzero(a | b))


def containment(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the area of the smaller mask."""
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    return inter / float(min(np.count_nonzero(a), np.count_nonzero(b)))


def is_duplicate(m: np.ndarray, kept: list) -> bool:
    return any(iou(m, k) > MERGE_IOU or containment(m, k) > MERGE_CONTAINMENT
               for k in kept)


def predict_field(model, img8: np.ndarray, device):
    """Tiled inference, predictions returned in source-frame coordinates."""
    h, w = img8.shape[:2]
    rgb = np.repeat(img8[:, :, None], 3, 2)
    preds, t_tiles = [], []
    for (y0, x0) in tiles_for(h, w, TILE, OVERLAP):
        crop = rgb[y0:y0 + TILE, x0:x0 + TILE]
        if crop.shape[0] < TILE or crop.shape[1] < TILE:
            crop = np.pad(crop, ((0, TILE - crop.shape[0]), (0, TILE - crop.shape[1]),
                                 (0, 0)), mode="reflect")
        t = time.perf_counter()
        r = model.predict(crop, verbose=False, conf=CONF, device=device, imgsz=TILE)[0]
        t_tiles.append(time.perf_counter() - t)
        if r.masks is None:
            continue
        confs = r.boxes.conf.cpu().numpy()
        for xy, c in zip(r.masks.xy, confs):
            if len(xy) < 3:
                continue
            v = np.asarray(xy, float) + np.array([x0, y0])
            preds.append({"verts": v, "conf": float(c)})
    # merge duplicates from the overlap, highest confidence first
    preds.sort(key=lambda p: -p["conf"])
    kept: list[dict] = []
    masks: list[np.ndarray] = []
    for p in preds:
        m = rasterise(p["verts"], h, w)
        if not m.any():
            continue
        if is_duplicate(m, masks):
            continue
        kept.append(p)
        masks.append(m)
    return kept, masks, t_tiles


def match(pred_masks, ref_masks, thr=IOU_MATCH):
    """Greedy one-to-one matching at an IoU threshold. Returns tp/fp/fn indices."""
    pairs = []
    for i, pm in enumerate(pred_masks):
        for j, rm in enumerate(ref_masks):
            v = iou(pm, rm)
            if v >= thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_p, used_r, tp = set(), set(), []
    for v, i, j in pairs:
        if i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        tp.append((i, j, v))
    fp = [i for i in range(len(pred_masks)) if i not in used_p]
    fn = [j for j in range(len(ref_masks)) if j not in used_r]
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall and F1, with the degenerate cases named rather than
    left as nan.

    A field where the model fired and matched nothing has precision 0 and
    recall 0, so its F1 is 0 -- a complete miss, not a missing value. Returning
    nan there quietly dropped the worst field out of every per-field average
    that skipped nans, which flatters exactly the case one wants to see. nan is
    kept only for the genuinely undefined case: no reference objects and no
    predictions, where there is nothing to score.
    """
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    if tp + fp + fn == 0:
        f = float("nan")
    elif p == p and r == r and (p + r) > 0:
        f = 2 * p * r / (p + r)
    else:
        f = 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def load_reference(prepared: Path):
    sc = prepared.parent / f".{prepared.stem}.acorn.json"
    d = json.loads(sc.read_text())
    return [a["vertices"] for a in d["annotations"] if a.get("type") == "roi"], d


def confident_subset(verts_list, px_nm):
    """The reference subset that is a single object by calibrated geometry.

    Same rule as the manuscript's reference-composition analysis: projected
    area 0.3-3.0 um^2 and circularity above 0.5.
    """
    from acorn.core.measurements import polygon_metrics
    keep = []
    for i, v in enumerate(verts_list):
        m = polygon_metrics([(float(x), float(y)) for x, y in v], px_nm)
        a_um2 = m["area_nm2"] / 1e6
        if 0.3 <= a_um2 <= 3.0 and m["circularity"] > 0.5:
            keep.append(i)
    return keep


def cryoblob_reference(test_stems: set[str]):
    """The independent verified particle list, restricted to held-out fields."""
    rows = json.loads(WORK_DIR / "plga_reference.json".read_text())
    out: dict[str, list] = {}
    for r in rows:
        stem = Path(r["File Location"]).stem
        if stem not in test_stems:
            continue
        out.setdefault(stem, []).append({
            "cx_nm": float(r["Center X (nm)"]), "cy_nm": float(r["Center Y (nm)"]),
            "diameter_nm": float(r["Size (nm)"]),
            "px_nm": float(r["Pixel Size X (nm/px)"]),
            "particle_id": r["Particle ID"],
        })
    return out


def tile_level_maps(checkpoint: str, project_dir: Path, device: str) -> dict:
    """Tile-level mAP on the held-out tiles, from the same checkpoint.

    Reported alongside the per-micrograph numbers because tile mAP is what a
    detector paper usually quotes, and the two are not interchangeable: a tile
    is a crop, and an object cut by a tile edge is scored twice at half size.
    """
    from ultralytics import YOLO
    yaml = project_dir / "data" / "dataset_test.yaml"
    if not yaml.exists():
        yaml = project_dir / "data" / "dataset.yaml"
    if not yaml.exists():
        return {}
    r = YOLO(checkpoint).val(data=str(yaml), split="test", device=device,
                             verbose=False, plots=False)
    out = {}
    for name, m in (("box", getattr(r, "box", None)), ("mask", getattr(r, "seg", None))):
        if m is None:
            continue
        out[f"{name}_map50"] = float(getattr(m, "map50", float("nan")))
        out[f"{name}_map50_95"] = float(getattr(m, "map", float("nan")))
        out[f"{name}_precision"] = float(np.mean(getattr(m, "p", [float("nan")])))
        out[f"{name}_recall"] = float(np.mean(getattr(m, "r", [float("nan")])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["cryo", "sem"], required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    man = json.loads((args.work / "split_manifest.json").read_text())
    runs = json.loads(args.runs.read_text())
    test = [s for s in man["sources"] if s["split"] == "test"]
    test_stems = {s["stem"] for s in test}
    print(f"  {len(test)} held-out micrographs")

    cb_ref = cryoblob_reference(test_stems) if args.modality == "cryo" else {}

    per_seed, per_field_rows, obj_rows = [], [], []
    for run in runs["runs"]:
        seed = run["seed"]
        model = YOLO(run["checkpoint"])
        agg = {"tp": 0, "fp": 0, "fn": 0}
        agg_conf = {"tp": 0, "fp": 0, "fn": 0}
        cb_agg = {"found": 0, "total": 0, "pred_with_ref": 0, "pred_total": 0}
        t_tiles_all, t_field_all = [], []
        for s in test:
            prepared = Path(s["prepared"])
            from PIL import Image
            img8 = np.array(Image.open(prepared).convert("L"))
            ref_verts, _ = load_reference(prepared)
            px = float(s["pixel_size_nm"])
            h, w = img8.shape[:2]

            t0 = time.perf_counter()
            preds, pmasks, t_tiles = predict_field(model, img8, args.device)
            t_field = time.perf_counter() - t0
            t_tiles_all += t_tiles
            t_field_all.append(t_field)

            rmasks = [rasterise(v, h, w) for v in ref_verts]
            tp, fp, fn = match(pmasks, rmasks)
            agg["tp"] += len(tp); agg["fp"] += len(fp); agg["fn"] += len(fn)
            row = {"seed": seed, "stem": s["stem"], "pixel_size_nm": px,
                   "n_reference": len(rmasks), "n_predicted": len(pmasks),
                   **prf(len(tp), len(fp), len(fn)),
                   "inference_seconds": t_field, "n_tiles": len(t_tiles)}

            if args.modality == "sem":
                keep = confident_subset(ref_verts, px)
                ctp, cfp, cfn = match(pmasks, [rmasks[i] for i in keep])
                agg_conf["tp"] += len(ctp); agg_conf["fp"] += len(cfp)
                agg_conf["fn"] += len(cfn)
                row["n_reference_confident"] = len(keep)
                row["recall_confident"] = prf(len(ctp), len(cfp), len(cfn))["recall"]
            else:
                refs = cb_ref.get(s["stem"], [])
                hit = 0
                pred_hit = set()
                for rp in refs:
                    # the reference is a centre and a size, so a particle counts
                    # as found when its centre falls inside a predicted mask
                    cx = rp["cx_nm"] / px
                    cy = rp["cy_nm"] / px
                    j = None
                    for k, m in enumerate(pmasks):
                        if 0 <= int(cy) < h and 0 <= int(cx) < w and m[int(cy), int(cx)]:
                            j = k
                            break
                    if j is not None:
                        hit += 1
                        pred_hit.add(j)
                cb_agg["found"] += hit
                cb_agg["total"] += len(refs)
                cb_agg["pred_with_ref"] += len(pred_hit)
                cb_agg["pred_total"] += len(pmasks)
                row["n_cryoblob_reference"] = len(refs)
                row["cryoblob_found"] = hit

            per_field_rows.append(row)
            if seed == runs["runs"][0]["seed"]:
                from acorn.core.measurements import polygon_metrics
                matched_p = {i for i, _, _ in tp}
                for i, p in enumerate(preds):
                    m = polygon_metrics([(float(x), float(y)) for x, y in p["verts"]], px)
                    obj_rows.append({"stem": s["stem"], "kind":
                                     "true_positive" if i in matched_p else "false_positive",
                                     "confidence": p["conf"], "ecd_nm": m["ecd_nm"],
                                     "area_nm2": m["area_nm2"],
                                     "circularity": m["circularity"]})
            print(f"    seed {seed}  {s['stem'][:44]:<44s} "
                  f"ref {row['n_reference']:4d} pred {row['n_predicted']:4d}  "
                  f"P {row['precision']:.3f} R {row['recall']:.3f} "
                  f"F1 {row['f1']:.3f}  {t_field:.2f}s", flush=True)

        maps = tile_level_maps(run["checkpoint"], Path(run["project_dir"]), args.device)
        entry = {"seed": seed, "checkpoint": run["checkpoint"],
                 "tile_level_maps": maps,
                 "checkpoint_sha256": run["checkpoint_sha256"],
                 "training_seconds": run.get("training_seconds"),
                 "pooled": prf(agg["tp"], agg["fp"], agg["fn"]),
                 "tile_level_test_metrics": run.get("test_metrics"),
                 "tile_level_val_metrics": run.get("best_metrics"),
                 "inference_seconds_per_tile_mean": float(np.mean(t_tiles_all)),
                 "inference_seconds_per_micrograph_mean": float(np.mean(t_field_all))}
        if args.modality == "sem":
            entry["pooled_confident_reference"] = prf(agg_conf["tp"], agg_conf["fp"],
                                                      agg_conf["fn"])
        else:
            entry["cryoblob_reference"] = {
                **cb_agg,
                "recall": cb_agg["found"] / cb_agg["total"] if cb_agg["total"] else None,
                "fraction_of_predictions_on_a_reference_particle":
                    cb_agg["pred_with_ref"] / cb_agg["pred_total"]
                    if cb_agg["pred_total"] else None}
        per_seed.append(entry)
        print(f"    seed {seed} pooled: {entry['pooled']}", flush=True)

    def ms(key, sub="pooled"):
        v = [e[sub][key] for e in per_seed if e[sub][key] == e[sub][key]]
        return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "n": len(v), "values": v}

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modality": args.modality,
        "matching": {"criterion": "mask IoU, greedy one-to-one",
                     "iou_threshold": IOU_MATCH,
                     "tile_merge_iou": MERGE_IOU,
                     "tile_merge_containment": MERGE_CONTAINMENT,
                     "tile_merge_note": "a prediction is discarded as a duplicate "
                                        "when its mask IoU with an already kept "
                                        "prediction exceeds the IoU threshold OR "
                                        "its intersection over its own area "
                                        "exceeds the containment threshold; the "
                                        "second is what removes a fragment cut "
                                        "off by a tile boundary",
                     "confidence": CONF,
                     "inference": f"{TILE} px tiles, {OVERLAP:.2f} overlap, "
                                  f"predictions mapped to the source frame"},
        "n_test_micrographs": len(test),
        "n_test_reference_objects": sum(s["n_annotations"] for s in test),
        "test_micrographs": sorted(test_stems),
        "per_seed": per_seed,
        "across_seeds": {k: ms(k) for k in ("precision", "recall", "f1")},
        "inference_seconds_per_micrograph": {
            "mean": float(np.mean([e["inference_seconds_per_micrograph_mean"]
                                   for e in per_seed]))},
        "inference_seconds_per_tile": {
            "mean": float(np.mean([e["inference_seconds_per_tile_mean"]
                                   for e in per_seed]))},
        "training_seconds": {
            "mean": float(np.mean([e["training_seconds"] for e in per_seed
                                   if e["training_seconds"]]))},
    }
    if args.modality == "sem":
        summary["across_seeds_confident_reference"] = {
            k: ms(k, "pooled_confident_reference") for k in ("precision", "recall", "f1")}
    else:
        summary["cryoblob_reference_across_seeds"] = {
            "recall_mean": float(np.mean([e["cryoblob_reference"]["recall"]
                                          for e in per_seed])),
            "recall_sd": float(np.std([e["cryoblob_reference"]["recall"]
                                       for e in per_seed], ddof=1)),
            "n_reference_particles": per_seed[0]["cryoblob_reference"]["total"],
        }

    (args.out / "heldout_evaluation.json").write_text(json.dumps(summary, indent=1))
    with open(args.out / "heldout_per_field.csv", "w", newline="") as fh:
        keys = sorted({k for r in per_field_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(per_field_rows)
    with open(args.out / "heldout_objects_seed0.csv", "w", newline="") as fh:
        if obj_rows:
            w = csv.DictWriter(fh, fieldnames=list(obj_rows[0]))
            w.writeheader()
            w.writerows(obj_rows)
    print("\n  across seeds:", json.dumps(summary["across_seeds"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
