#!/usr/bin/env python3
"""Additional supplementary figures and tables for the ACORN methods paper.

Built in the same style as the existing supplementary set: individual panels,
600 dpi PNG plus vector PDF, no panel letters burned in, and colour that
encodes imaging modality only. `common.py` and `palette.py` are imported from
the paper's own figure directory so nothing can drift.

  figS6   held-out predictions from the models trained on reviewed
          experimental annotations, both modalities, with an error overlay
  figS7   one plotted value traced back through the provenance record to its
          source micrograph and that micrograph's calibration
  fig1_inset
          worst-case relative calibration error by operation
  tableS10  experimental training datasets, configuration, held-out performance
  tableS11  calibration retention
  tableS12  provenance-audit counts and linked records

    python make_additional_si.py --root <additional_info dir> --out <deposit dir>
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

FIGDIR = FIGURE_SCRIPTS_DIR
sys.path.insert(0, str(FIGDIR))

import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.patches import Polygon as MplPoly                  # noqa: E402
from common import (COL_MM, FULL_MM, MM, TRANSMISSION, SCANNING,   # noqa: E402
                    REFERENCE, TRUE_POS, FALSE_POS, MISSED, GREY, RULE,
                    git_commit, save, image_fig, scalebar)

META: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "git_commit": git_commit(), "panels": {}, "tables": {}}


def fmt(v, n=3):
    if v is None or (isinstance(v, float) and v != v):
        return "--"
    return f"{v:.{n}f}"


def mean_sd(d, n=3):
    """mean +- sd across seeds, or just the mean when there is one seed."""
    if d["n"] < 2:
        return fmt(d["mean"], n)
    return f"{d['mean']:.{n}f} +- {d['sd']:.{n}f}"


def csv_pdf(tb: Path, name, header, rows, caption, colw=None, wide=False):
    tb.mkdir(parents=True, exist_ok=True)
    with open(tb / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    h = 0.26 * (len(rows) + 1) + 0.5
    width = FULL_MM if (wide or len(header) > 4) else COL_MM
    fig = plt.figure(figsize=(width * MM, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    t = ax.table(cellText=[[str(c) for c in r] for r in rows], colLabels=header,
                 loc="center", cellLoc="right", colWidths=colw)
    t.auto_set_font_size(False)
    t.set_fontsize(6.6)
    t.scale(1, 1.25)
    for (r_, c_), cell in t.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor(RULE)
        if r_ == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eef0f3")
        if c_ == 0:
            cell.set_text_props(ha="left")
    save(fig, tb, name)
    META["tables"][name] = dict(header=header, rows=rows, caption=caption)


# ── S6: held-out predictions ─────────────────────────────────────────────────

def outlines_from_sidecar(prepared: Path):
    d = json.loads((prepared.parent / f".{prepared.stem}.acorn.json").read_text())
    return [np.asarray(a["vertices"], float) for a in d["annotations"]
            if a.get("type") == "roi"]


def predict_for_figure(checkpoint: str, img8: np.ndarray, device="0"):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_heldout import predict_field, rasterise, match
    from ultralytics import YOLO
    model = YOLO(checkpoint)
    preds, pmasks, _ = predict_field(model, img8, device)
    return preds, pmasks, rasterise, match


def fig_S6(root: Path, out: Path, device="0") -> dict:
    """Four panels per modality: micrograph, reference, prediction, errors.

    The field shown is the held-out micrograph whose F1 is nearest the median
    across the held-out set, so it is representative rather than flattering,
    and it appeared in no training, validation or prompt-development step.
    """
    from evaluate_heldout import rasterise, match
    rec = {}
    for tag, colour, sub in (("transmission", TRANSMISSION, "cryo"),
                             ("scanning", SCANNING, "sem")):
        ev = json.loads((root / f"results/{sub}/heldout_evaluation.json").read_text())
        man = json.loads((root / f"work/{sub}/split_manifest.json").read_text())
        rows = list(csv.DictReader(open(root / f"results/{sub}/heldout_per_field.csv")))
        seed0 = [r for r in rows if int(r["seed"]) == ev["per_seed"][0]["seed"]]
        f1s = np.array([float(r["f1"]) for r in seed0])
        med = float(np.median(f1s))
        pick = seed0[int(np.argmin(np.abs(f1s - med)))]
        src = next(s for s in man["sources"] if s["stem"] == pick["stem"])
        prepared = Path(src["prepared"])
        px = float(src["pixel_size_nm"])
        img8 = np.array(Image.open(prepared).convert("L"))
        h, w = img8.shape[:2]

        ref = outlines_from_sidecar(prepared)
        preds, pmasks, _, _ = predict_for_figure(
            ev["per_seed"][0]["checkpoint"], img8, device)
        rmasks = [rasterise(v, h, w) for v in ref]
        tp, fp, fn = match(pmasks, rmasks)
        matched_p = {i for i, _, _ in tp}

        # A / E — the micrograph
        fig, ax = image_fig(img8)
        scalebar(ax, w, px)
        save(fig, out, f"S6_{tag}_micrograph")

        # B / F — the reference annotations
        fig, ax = image_fig(img8)
        for v in ref:
            ax.add_patch(MplPoly(v, closed=True, fill=False, ec=REFERENCE, lw=0.7))
        scalebar(ax, w, px)
        save(fig, out, f"S6_{tag}_reference")

        # C / G — what the trained model predicted
        fig, ax = image_fig(img8)
        for p in preds:
            ax.add_patch(MplPoly(p["verts"], closed=True, fill=False, ec=colour, lw=0.7))
        scalebar(ax, w, px)
        save(fig, out, f"S6_{tag}_prediction")

        # D / H — the errors, by category
        fig, ax = image_fig(img8)
        for i, p in enumerate(preds):
            c = TRUE_POS if i in matched_p else FALSE_POS
            ls = "-" if i in matched_p else (0, (2, 1.2))
            ax.add_patch(MplPoly(p["verts"], closed=True, fill=False, ec=c, lw=0.8,
                                 ls=ls))
        for j in fn:
            ax.add_patch(MplPoly(ref[j], closed=True, fill=False, ec=MISSED, lw=0.8,
                                 ls=(0, (1, 1.2))))
        scalebar(ax, w, px)
        save(fig, out, f"S6_{tag}_errors")

        rec[tag] = {
            "field": pick["stem"],
            "selection_rule": "held-out field whose F1 is nearest the median "
                              "across the held-out set, seed 0",
            "dataset_median_f1": med,
            "field_f1": float(pick["f1"]),
            "n_reference": len(ref), "n_predicted": len(preds),
            "tp": len(tp), "fp": len(fp), "missed": len(fn),
            "pixel_size_nm": px,
            "pixel_size_source": src["pixel_size_source"],
            "source_file": src["source"], "source_sha256": src["source_sha256"],
            "checkpoint": ev["per_seed"][0]["checkpoint"],
            "checkpoint_sha256": ev["per_seed"][0]["checkpoint_sha256"],
            "confidence": 0.5, "iou_threshold": 0.5,
            "never_used_for": ["training", "model selection", "prompt development"],
        }

    # one shared legend, drawn once
    fig = plt.figure(figsize=(COL_MM * MM, 0.30))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.legend(handles=[
        Line2D([], [], color=REFERENCE, lw=1.2, label="reference"),
        Line2D([], [], color=TRUE_POS, lw=1.2, label="true positive"),
        Line2D([], [], color=FALSE_POS, lw=1.2, ls=(0, (2, 1.2)), label="false positive"),
        Line2D([], [], color=MISSED, lw=1.2, ls=(0, (1, 1.2)), label="undetected"),
    ], loc="center", ncol=4, frameon=False, handlelength=2.0, columnspacing=1.2)
    save(fig, out, "S6_legend")
    META["panels"]["S6"] = rec
    return rec


# ── S7: the provenance trace ─────────────────────────────────────────────────

def _trace_rows(tr, a):
    """The chain as label/value pairs, printed exactly as recorded."""
    m = tr["measurement_record"]
    img = tr["source_image"]
    rows = [
        ("plotted value",
         f"{tr['plot_point']['group']}   ECD {m['ecd_nm']:.1f} nm"),
        ("statistical analysis",
         f"{tr['statistical_analysis_record']['analysis_id']}   "
         f"Kruskal-Wallis p = {tr['statistical_analysis_record']['p_value']:.2e}   "
         f"n = {tr['statistical_analysis_record']['n_in_group']}"),
        ("measurement record",
         f"{m['measurement_id']}   area {m['area_nm2']:.0f} nm2   "
         f"circularity {m['circularity']:.2f}"),
        ("annotation",
         f"{a['annotation_id']}"),
        ("origin",
         f"{a['origin']}   {a['invocation']}   "
         f"{'edited after creation' if a['edited_after_creation'] else 'unedited'}"),
        ("geometry hash",
         f"created {a['geometry_hash_at_creation']}   "
         f"current {a['geometry_hash_current']}"),
    ]
    if tr.get("checkpoint"):
        rows += [
            ("checkpoint sha256", tr["checkpoint"]["sha256"]),
            ("training run",
             f"seed {tr['training_run']['seed']}   "
             f"{tr['training_run']['architecture']}   "
             f"{tr['training_run']['epochs']} epochs   "
             f"batch {tr['training_run']['batch']}   "
             f"{tr['training_run']['imgsz']} px"),
        ]
    else:
        sm = a.get("source_model") or {}
        rows += [("model / prompt",
                  sm.get("prompt") or sm.get("path") or "none: not model-generated")]
    rows += [
        ("dataset version",
         f"{tr['dataset_version']['id']}   partitions "
         + ", ".join(f"{k} {v}" for k, v in tr["dataset_version"]["partitions"].items())),
        ("source micrograph", Path(img["source_file"]).name),
        ("source sha256", img["source_sha256"]),
        ("calibration",
         f"{img['pixel_size_nm']:.3f} nm/px   {img['pixel_size_source']}"),
    ]
    return rows


def _trace_panel(rows, out: Path, name: str):
    fig = plt.figure(figsize=(FULL_MM * MM, 0.235 * len(rows) + 0.30))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(rows))
    for i, (label, value) in enumerate(rows):
        y = len(rows) - i - 0.5
        ax.text(0.004, y, label, fontsize=6.0, va="center", ha="left", color=GREY)
        ax.text(0.245, y, value, fontsize=5.6, va="center", ha="left",
                family="monospace")
        ax.plot([0.0, 1.0], [y - 0.5, y - 0.5], color=RULE, lw=0.35)
        if i < len(rows) - 1:
            ax.annotate("", xy=(0.232, y - 0.52), xytext=(0.232, y - 0.18),
                        arrowprops=dict(arrowstyle="-|>", lw=0.5, color=SCANNING))
    save(fig, out, name)


def fig_S7(root: Path, out: Path) -> dict:
    """The real trace: plotted values back to a mask and a source micrograph.

    Values and hashes are printed as recorded, so nothing here is illustrative.
    Two chains are shown: the object nearest its group's median, whatever its
    origin turned out to be, and the nearest-median object that the trained
    model produced, which is the one that carries a checkpoint and a training
    run to link to.
    """
    audit = json.loads((root / "provenance/provenance_audit.json").read_text())
    anns = {a["annotation_id"]: a
            for a in json.loads((root / "provenance/annotations.json").read_text())}

    panels = {}
    for key, name in (("trace", "S7A_trace_median_object"),
                      ("trace_model_generated", "S7B_trace_model_generated")):
        tr = audit.get(key)
        if not tr:
            continue
        a = anns[tr["annotation_record"]["annotation_id"]]
        _trace_panel(_trace_rows(tr, a), out, name)
        panels[key] = {"panel": name, "selection_rule": tr["plot_point"]["selection_rule"],
                       "annotation_id": a["annotation_id"], "origin": a["origin"],
                       "value_nm": tr["plot_point"]["value_nm"]}

    # the mask that value was measured from, on its own source micrograph
    tr = audit["trace_model_generated"] or audit["trace"]
    a = anns[tr["annotation_record"]["annotation_id"]]
    img = tr["source_image"]
    img8 = np.array(Image.open(img["prepared_file"]).convert("L"))
    h, w = img8.shape[:2]
    verts = np.asarray(a["vertices"], float)

    fig, ax = image_fig(img8)
    ax.add_patch(MplPoly(verts, closed=True, fill=False, ec=FALSE_POS, lw=1.0))
    scalebar(ax, w, img["pixel_size_nm"])
    save(fig, out, "S7C_source_micrograph")

    cx, cy = verts[:, 0].mean(), verts[:, 1].mean()
    span = max(np.ptp(verts[:, 0]), np.ptp(verts[:, 1]))
    pad = int(max(40, 0.9 * span))
    x0, x1 = max(0, int(cx - pad)), min(w, int(cx + pad))
    y0, y1 = max(0, int(cy - pad)), min(h, int(cy + pad))
    fig, ax = image_fig(img8[y0:y1, x0:x1], width_mm=COL_MM * 0.55)
    ax.add_patch(MplPoly(verts - np.array([x0, y0]), closed=True, fill=False,
                         ec=FALSE_POS, lw=1.2))
    save(fig, out, "S7D_traced_mask")

    rec = {"panels": panels,
           "mask_shown_for": tr["annotation_record"]["annotation_id"],
           "mask_shown_origin": a["origin"],
           "crop_centre_px": [float(cx), float(cy)],
           "counts": audit["counts"],
           "dataset_version": audit["dataset_version"],
           "checkpoint_sha256": audit["checkpoint_sha256"],
           "review_rules": audit["review_rules"]}
    META["panels"]["S7"] = rec
    return rec


# ── figure 1 inset: worst calibration error by operation ─────────────────────

OP_GROUPS = {
    "import": ["initial import"],
    "session reload": ["session save + reload (header calibration)",
                       "session save + reload (manual override)"],
    "binning": ["2x binning", "4x binning",
                "4x binning + reload of an overridden pixel size"],
    "tiling": ["tiling", "tile coordinates restored to the source frame"],
    "export": ["annotation export (COCO polygon + pixel size)",
               "measurement-table export (image in cache)",
               "measurement-table export (image evicted from the 3-image cache)"],
}


def group_of(op: str):
    for g, pats in OP_GROUPS.items():
        for p in pats:
            if op.startswith(p) or op == p:
                return g
    return None


def fig1_inset(root: Path, out: Path) -> dict:
    """The inset for Figure 1: what each operation does to the calibration.

    A bar chart of the pixel-size error would be five bars of exactly zero,
    which is the result but not a picture. So the bars carry the quantity that
    is not zero -- the error in the physical size recovered from a known
    object -- and the pixel-size column states the zero as a number, once per
    operation, which is what it is.
    """
    rows = list(csv.DictReader(open(root / "calibration/calibration_validation.csv")))
    worst_cal: dict[str, float] = {}
    worst_dia: dict[str, float] = {}
    n: dict[str, int] = {}
    for r in rows:
        g = group_of(r["operation"])
        if g is None:
            continue
        n[g] = n.get(g, 0) + 1
        if r["rel_calibration_error_pct"]:
            worst_cal[g] = max(worst_cal.get(g, 0.0),
                               float(r["rel_calibration_error_pct"]))
        if r["rel_diameter_error_pct"]:
            worst_dia[g] = max(worst_dia.get(g, 0.0),
                               float(r["rel_diameter_error_pct"]))

    order = ["import", "session reload", "binning", "tiling", "export"]
    fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM * 0.52))
    ax = fig.add_subplot(111)
    y = np.arange(len(order))
    dia = [worst_dia.get(k, 0.0) for k in order]
    ax.barh(y, dia, color=GREY, height=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlim(0, max(dia) * 2.05)
    ax.set_xlabel("worst error in the recovered object size (%)")
    ax.grid(axis="x", alpha=.15, lw=.4, zorder=0)
    for i, k in enumerate(order):
        cal = worst_cal.get(k, 0.0)
        ax.text(dia[i] + max(dia) * 0.04, i,
                f"{dia[i]:.2g}%   pixel size {cal:.0f}%   n={n[k]}",
                va="center", fontsize=5.8, color=GREY)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, out, "fig1_inset_calibration")

    rec = {"worst_rel_calibration_error_pct_by_operation": worst_cal,
           "worst_rel_diameter_error_pct_by_operation": worst_dia,
           "n_tests_by_operation": n,
           "note": "the pixel-size error is exactly zero for every operation, so "
                   "the bars show the error in the recovered physical size and "
                   "the zero is written as a number beside each bar"}
    META["panels"]["fig1_inset"] = rec
    return rec


# ── tables ───────────────────────────────────────────────────────────────────

def table_S10(root: Path, tb: Path):
    """Dataset, configuration and held-out performance, one row per modality."""
    def resolved_optimizer(root: Path, sub: str) -> str:
        """The optimiser Ultralytics actually chose, from the training log.

        `optimizer=auto` in the argument set is a request, not a record: the
        trainer picks the optimiser, learning rate and momentum from the
        iteration count and prints what it chose. Quoting the request would
        misreport lr0=0.01 when 0.002 was used.
        """
        for seed in (0, 1, 2):
            log = root / f"results/{sub}/seed{seed}/train.log"
            for cand in (log, root / f"logs/train_{sub}.log"):
                if not cand.exists():
                    continue
                for line in cand.read_text(errors="ignore").splitlines():
                    if "with parameter groups" in line:
                        return line.split("m ")[-1].split(" with")[0].strip()
        return "--"

    rows = []
    hyper = []
    for label, sub in (("Cryo-TEM", "cryo"), ("SEM", "sem")):
        man = json.loads((root / f"work/{sub}/split_manifest.json").read_text())
        ev = json.loads((root / f"results/{sub}/heldout_evaluation.json").read_text())
        runs = json.loads((root / f"results/{sub}/training_runs.json").read_text())
        t = man["totals"]
        sc = man["split_counts_source_images"]
        tl = man["split_counts_tiles"]
        maps = [r.get("tile_level_maps") or {} for r in ev["per_seed"]]

        def m(key):
            v = [x.get(key) for x in maps
                 if x.get(key) is not None and x.get(key) == x.get(key)]
            return float(np.mean(v)) if v else float("nan")

        rows.append([
            label, t["source_images"], f"{sc['train']} / {tl['train']}",
            f"{sc['val']} / {tl['val']}", f"{sc['test']} / {tl['test']}",
            t["annotated_objects"], runs["hyperparameters"]["architecture"],
            len(ev["per_seed"]),
            mean_sd(ev["across_seeds"]["precision"]),
            mean_sd(ev["across_seeds"]["recall"]),
            mean_sd(ev["across_seeds"]["f1"]),
            fmt(m("box_map50")),
            fmt(m("mask_map50")),
            fmt(m("mask_map50_95")),
        ])
        h = runs["hyperparameters"]
        hyper.append([
            label, h["architecture"], f"{h['epochs']}", f"{h['batch']}",
            f"{h['imgsz']}", ", ".join(str(s) for s in runs["seeds"]),
            resolved_optimizer(root, sub),
            "HSV 0.015/0.7/0.4, translate 0.1, scale 0.5, fliplr 0.5, "
            "mosaic 1.0 (off for the last 10 epochs), erasing 0.4, "
            "randaugment; no rotation, shear, perspective, flipud, mixup "
            "or copy-paste",
            f"{ev['training_seconds']['mean'] / 60:.1f} min",
            f"{ev['inference_seconds_per_micrograph']['mean'] * 1000:.0f} ms",
            f"{ev['inference_seconds_per_tile']['mean'] * 1000:.0f} ms",
            runs["hardware"].get("gpu") or "--",
        ])

    csv_pdf(tb, "tableS10",
            ["Modality", "Source images", "Train img / tiles", "Val img / tiles",
             "Test img / tiles", "Annotated objects", "Model", "Seeds",
             "Precision", "Recall", "F1", "Box mAP50", "Mask mAP50",
             "Mask mAP50-95"],
            rows,
            "Experimental training datasets, partitioned by source micrograph, "
            "and held-out performance of the models trained on the reviewed "
            "annotations. Precision, recall and F1 are per held-out micrograph "
            "at a mask IoU of 0.5, pooled over the held-out set and reported as "
            "mean +- s.d. across training seeds; the mAP columns are tile-level "
            "values from the same checkpoints on the same held-out fields, "
            "given because they are the quantity a detection paper usually "
            "reports and are not interchangeable with the per-micrograph "
            "numbers.", wide=True)

    csv_pdf(tb, "tableS10b",
            ["Modality", "Architecture", "Epochs", "Batch", "Input size", "Seeds",
             "Optimiser (resolved)", "Augmentations", "Training time",
             "Inference / micrograph", "Inference / tile", "GPU"],
            hyper,
            "Training configuration for Table S10. The optimiser column is what "
            "Ultralytics resolved `optimizer=auto` to and the learning rate and "
            "momentum it chose, not the requested defaults, which were "
            "lr0 = 0.01 and momentum = 0.937 and were overridden. Weight decay "
            "5e-4, 3 warmup epochs, no cosine schedule, patience 100 (so no run "
            "early-stopped). Augmentations are the Ultralytics segmentation "
            "defaults; the complete resolved argument set for every run is "
            "deposited verbatim.", wide=True)


def table_S11(root: Path, tb: Path):
    """One row per format and operation: the worst case over every test in it.

    Reporting the worst rather than a representative row is the point of a
    retention test -- a mean would hide the one combination that failed.
    """
    rows = list(csv.DictReader(open(root / "calibration/calibration_validation.csv")))
    summary = json.loads((root / "calibration/calibration_validation.json").read_text())

    def op_key(op):
        return op.split(" (")[0] if op.startswith("tiling") else op

    order, agg = [], {}
    for r in rows:
        k = (r["source_format"], op_key(r["operation"]))
        if k not in order:
            order.append(k)
        a = agg.setdefault(k, {"n": 0, "cal": 0.0, "diam": None, "px": set(),
                               "fail": 0, "note": r["note"]})
        a["n"] += 1
        a["px"].add(float(r["header_pixel_size_nm"]))
        if r["rel_calibration_error_pct"]:
            a["cal"] = max(a["cal"], float(r["rel_calibration_error_pct"]))
        if r["rel_diameter_error_pct"]:
            v = float(r["rel_diameter_error_pct"])
            a["diam"] = v if a["diam"] is None else max(a["diam"], v)
        if r["result"] != "pass":
            a["fail"] += 1

    out = []
    for k in order:
        a = agg[k]
        px = sorted(a["px"])
        out.append([
            k[0], k[1], a["n"],
            f"{px[0]:g}" if len(px) == 1 else f"{px[0]:g}-{px[-1]:g}",
            f"{a['cal']:.3g}",
            "--" if a["diam"] is None else f"{a['diam']:.3g}",
            "pass" if a["fail"] == 0 else f"{a['fail']} of {a['n']} outside tolerance",
        ])
    csv_pdf(tb, "tableS11",
            ["Format", "Operation", "Tests", "Input px (nm)",
             "Worst rel. calibration error (%)",
             "Worst rel. diameter error (%)", "Result"],
            out,
            "Spatial calibration and physical object dimensions are retained "
            "during image transformations and export. Each row is the worst case "
            "over every test of that format and operation, across "
            f"{len(summary['pixel_sizes_nm'])} pixel sizes and "
            f"{len(summary['diameters_nm'])} known object diameters; "
            f"{summary['n_tests']} tests in total, of which "
            f"{summary['n_failed']} fall outside tolerance. Tolerances: "
            f"{summary['tolerances']['relative_calibration_error'] * 100:g} % on "
            "the recorded pixel size and "
            f"{summary['tolerances']['relative_diameter_error'] * 100:g} % on the "
            "recovered diameter. Rows with no diameter entry are operations that "
            "move a calibration without producing a new measurable image. The "
            "complete machine-readable result is deposited as "
            "calibration_validation.csv.", wide=True)


def table_S12(root: Path, tb: Path):
    a = json.loads((root / "provenance/provenance_audit.json").read_text())
    c = a["counts"]
    tot = c["total_annotations"]

    def pct(n):
        return f"{100.0 * n / tot:.1f}" if tot else "--"

    rows = [
        ["Manual annotations", c["manual"], pct(c["manual"]), "Source image"],
        ["Prompt-generated, accepted", c["prompt_generated_accepted"],
         pct(c["prompt_generated_accepted"]), "Prompt and model"],
        ["Prompt-generated, corrected", c["prompt_generated_corrected"],
         pct(c["prompt_generated_corrected"]), "Parent annotation"],
        ["Prompt-generated, rejected", c["prompt_generated_rejected"], "--",
         "Review log"],
        ["Trained-model, accepted", c["model_generated_accepted"],
         pct(c["model_generated_accepted"]), "Checkpoint"],
        ["Trained-model, corrected", c["model_generated_corrected"],
         pct(c["model_generated_corrected"]), "Checkpoint and parent"],
        ["Trained-model, rejected", c["model_generated_rejected"], "--", "Review log"],
        ["Imported", c["imported"], pct(c["imported"]), "Import record"],
        ["Final measurements", c["measurements"], "--", "Annotation and calibration"],
        ["Statistical outputs", c["statistical_outputs"], "--", "Measurement set"],
    ]
    csv_pdf(tb, "tableS12",
            ["Record category", "Count", "Percentage", "Linked record"], rows,
            "Annotation origins, review outcomes, and downstream records in the "
            f"provenance audit of the {a['project']}. Percentages are of the "
            f"{tot} annotations in the project; rejected candidates never enter "
            "the store and so have no share of it. Dataset version "
            f"{a['dataset_version']}; checkpoint sha256 {a['checkpoint_sha256']}.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=WORK_DIR / "additional_info")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()
    figs = args.out / "figures"
    tb = args.out / "tables"

    want = lambda k: (not args.only) or k in args.only     # noqa: E731
    if want("S6"):
        print("  figure S6")
        fig_S6(args.root, figs / "figS6", args.device)
    if want("S7"):
        print("  figure S7")
        fig_S7(args.root, figs / "figS7")
    if want("inset"):
        print("  figure 1 inset")
        fig1_inset(args.root, figs / "fig1_inset")
    if want("S10"):
        print("  table S10")
        table_S10(args.root, tb)
    if want("S11"):
        print("  table S11")
        table_S11(args.root, tb)
    if want("S12"):
        print("  table S12")
        table_S12(args.root, tb)

    mp = args.out / "additional_si_metadata.json"
    old = json.loads(mp.read_text()) if mp.exists() else {}
    old.setdefault("panels", {}).update(META["panels"])
    old.setdefault("tables", {}).update(META["tables"])
    old["generated_utc"] = META["generated_utc"]
    old["git_commit"] = META["git_commit"]
    mp.write_text(json.dumps(old, indent=1))
    print(f"\n  wrote {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
