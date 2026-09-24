#!/usr/bin/env python3
"""The software-comparison table, by named application rather than by category.

This is a documentation-based assessment, not a benchmark. Every entry is taken
from the cited documentation at the version and access date given in the table;
nothing here is a measured performance claim. A capability delivered only by a
third-party plugin is marked *partial* and the plugin is named in the footnote,
because "the ecosystem can do it" and "the application does it" are different
statements for anyone deciding what to install.

    python software_comparison.py --out <deposit dir>
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

FIGDIR = FIGURE_SCRIPTS_DIR
sys.path.insert(0, str(FIGDIR))

import matplotlib.pyplot as plt                                   # noqa: E402
from common import COL_MM, FULL_MM, MM, RULE, save, git_commit    # noqa: E402

ACCESSED = "2026-09-08"

Y, P, N = "yes", "partial", "no"

CAPABILITIES = [
    "Local or hosted",
    "Native EM metadata",
    "Calibration tracking",
    "Manual annotation",
    "Prompted segmentation",
    "In-application training",
    "Batch inference",
    "Calibrated measurement",
    "Statistical analysis",
    "Simulation",
    "Training-to-result provenance",
    "GUI / CLI / library",
]

# Each row: name, version, documentation URL, then one entry per capability.
# A footnote key in braces attaches the note printed underneath.
ROWS = [
    dict(
        name="Fiji / ImageJ",
        version="Fiji stable distribution; ImageJ 1.54t (16 May 2026)",
        docs="https://imagej.net/",
        cells=["local", f"{P}{{fiji-meta}}", f"{P}{{fiji-cal}}", Y,
               f"{P}{{fiji-prompt}}", f"{P}{{fiji-train}}", f"{P}{{fiji-batch}}",
               Y, f"{P}{{fiji-stats}}", N, N, "GUI / CLI / library"],
    ),
    dict(
        name="CVAT",
        version="2.74.0 (August 2026)",
        docs="https://docs.cvat.ai/",
        cells=["local or hosted", N, N, Y, f"{Y}{{cvat-prompt}}", N,
               f"{P}{{cvat-batch}}", N, N, N, f"{P}{{cvat-prov}}",
               "GUI / CLI / library"],
    ),
    dict(
        name="Label Studio",
        version="1.23 (2026)",
        docs="https://labelstud.io/guide/",
        cells=["local or hosted", N, N, Y, f"{P}{{ls-prompt}}", f"{P}{{ls-train}}",
               f"{P}{{ls-batch}}", N, N, N, f"{P}{{ls-prov}}",
               "GUI / CLI / library"],
    ),
    dict(
        name="Cellpose",
        version="4.2.1",
        docs="https://cellpose.readthedocs.io/",
        cells=["local", N, f"{P}{{cp-cal}}", Y, N, Y, Y, N, N, N, N,
               "GUI / CLI / library"],
    ),
    dict(
        name="napari",
        version="0.9.0 (25 August 2026)",
        docs="https://napari.org/stable/",
        cells=["local", f"{P}{{np-meta}}", f"{P}{{np-cal}}", Y, f"{P}{{np-prompt}}",
               f"{P}{{np-train}}", f"{P}{{np-batch}}", f"{P}{{np-measure}}",
               f"{P}{{np-stats}}", N, N, f"GUI / -- / library{{np-cli}}"],
    ),
    dict(
        name="RELION",
        version="5.0.1 (local build, commit f2c1a3)",
        docs="https://relion.readthedocs.io/",
        cells=["local", Y, Y, f"{P}{{re-manual}}", N, f"{P}{{re-train}}", Y,
               f"{P}{{re-measure}}", f"{P}{{re-stats}}", f"{P}{{re-sim}}",
               f"{P}{{re-prov}}", f"GUI / CLI / {P}{{re-lib}}"],
    ),
    dict(
        name="cryoSPARC",
        version="5.0.7 (14 August 2026)",
        docs="https://guide.cryosparc.com/",
        cells=["local (browser client)", Y, Y, f"{P}{{cs-manual}}", N,
               f"{P}{{cs-train}}", Y, f"{P}{{cs-measure}}", f"{P}{{cs-stats}}",
               f"{P}{{cs-sim}}", f"{P}{{cs-prov}}", f"GUI / CLI / library"],
    ),
    dict(
        name="ACORN (this work)",
        version="0.3.0 (commit be8aa3e plus the fixes reported here)",
        docs="https://github.com/vitrifyai/acorn_software",
        cells=["local", Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, "GUI / CLI / library"],
    ),
]

FOOTNOTES = {
    "fiji-meta": "reads TIFF, DM3/DM4 and MRC through Bio-Formats; the vendor "
                 "SEM tags that carry the true scale (Zeiss CZ_SEM, "
                 "Thermo/FEI FEI_HELIOS) are not read by the standard readers, "
                 "which is the failure documented in this manuscript",
    "fiji-cal": "an image carries a pixel size and a unit, and measurements use "
                "them; where the value came from is not recorded, so a header "
                "value and a typed-in value are indistinguishable afterwards",
    "fiji-prompt": "through third-party plugins (SAMJ, Labkit), not in the base "
                   "distribution",
    "fiji-train": "pixel classifiers only (Trainable Weka Segmentation, Labkit); "
                  "no instance-segmentation training",
    "fiji-batch": "macro batch mode and headless execution; no model-serving "
                  "batch inference in the application",
    "fiji-stats": "a results table, histograms and curve fitting; no hypothesis "
                  "testing or effect sizes",
    "cvat-prompt": "interactive SAM-family segmentors are built in",
    "cvat-batch": "automatic annotation through separately deployed serverless "
                  "model functions",
    "cvat-prov": "per-job annotation history and user attribution; no link from "
                 "an annotation to a model checkpoint or a training run",
    "ls-prompt": "requires a machine-learning backend the user deploys",
    "ls-train": "the ML-backend interface exposes a fit() hook; the training "
                "itself is code the user supplies and runs outside the app",
    "ls-batch": "predictions are requested from a deployed ML backend",
    "ls-prov": "annotation history and task versions; no model or calibration "
               "lineage",
    "cp-cal": "an object diameter in pixels conditions the model; there is no "
              "physical pixel size and no measurement in physical units",
    "np-meta": "through reader plugins for individual formats; layer scale must "
               "usually be set by the caller",
    "np-cal": "layers carry scale and units and viewers honour them; the origin "
              "of the value is not tracked",
    "np-prompt": "through plugins (napari-sam, micro-sam)",
    "np-train": "through plugins (micro-sam finetuning)",
    "np-batch": "scripted through the library; no batch-inference UI",
    "np-measure": "through plugins (napari-skimage-regionprops), which honour "
                  "layer scale",
    "np-stats": "through plugins",
    "re-manual": "manual particle picking, not general annotation",
    "re-train": "training is confined to specific built-in tasks (the wrapped "
                "Topaz picker, the class ranker); a user cannot train a model "
                "for their own object class",
    "re-measure": "resolution, per-particle and map statistics; no 2-D object "
                  "morphometry in physical units",
    "re-stats": "job-level statistics such as FSC and per-class distributions; "
                "no general hypothesis testing on measurements",
    "re-sim": "relion_project produces projections of a reference map; there is "
              "no dose, detector or specimen forward model",
    "re-prov": "the pipeline STAR files record every job with its inputs and "
               "outputs, which is a strong process record; individual "
               "annotations are not linked to a model checkpoint hash",
    "cs-manual": "manual picking and curation within the processing workflow",
    "cs-train": "training within built-in jobs (Topaz wrappers, deep picker); "
                "not for a user-defined object class",
    "cs-measure": "map and particle statistics; no calibrated 2-D morphometry",
    "cs-stats": "job-level statistics; no general hypothesis testing",
    "cs-sim": "synthetic data generation from a reference volume, without a "
              "detector or dose model",
    "cs-prov": "a complete job graph with inputs, outputs and parameters; "
               "annotations are not linked to checkpoint hashes",
    "re-lib": "a Python module exists for reading and writing STAR files and "
              "some job control; the reconstruction code itself is a C++ "
              "executable driven by command line or GUI",
    "np-cli": "no command-line entry point for analysis; napari is driven from "
              "the GUI or as a Python library",
}


def strip_keys(cell: str) -> tuple[str, list[str]]:
    keys = []
    out = cell
    while "{" in out:
        i, j = out.index("{"), out.index("}")
        keys.append(out[i + 1:j])
        out = out[:i] + out[j + 1:]
    return out, keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tb = args.out / "tables"
    tb.mkdir(parents=True, exist_ok=True)

    header = ["Software", "Version", "Documentation accessed"] + CAPABILITIES
    csv_rows, disp_rows, used = [], [], {}
    for r in ROWS:
        cells, marks = [], []
        for c in r["cells"]:
            text, keys = strip_keys(c)
            cells.append(text)
            for k in keys:
                used.setdefault(k, len(used) + 1)
            marks.append("".join(f"({used[k]})" for k in keys))
        csv_rows.append([r["name"], r["version"], ACCESSED] + cells)
        disp_rows.append([r["name"], r["version"], ACCESSED]
                         + [f"{t}{m}" for t, m in zip(cells, marks)])

    with open(tb / "software_comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(csv_rows)
        w.writerow([])
        w.writerow(["Documentation URLs"])
        for r in ROWS:
            w.writerow([r["name"], r["docs"]])
        w.writerow([])
        w.writerow(["Footnotes"])
        for k, n in sorted(used.items(), key=lambda kv: kv[1]):
            w.writerow([f"({n})", FOOTNOTES[k]])

    # a portrait rendering: capabilities down the page, software across it,
    # because twelve capability columns do not fit a journal page width
    body = [[cap] + [disp_rows[i][3 + j] for i in range(len(ROWS))]
            for j, cap in enumerate(CAPABILITIES)]
    top = [["Version"] + [r["version"] for r in ROWS],
           ["Documentation accessed"] + [ACCESSED for _ in ROWS]]
    rows = top + body
    cols = ["Capability"] + [r["name"] for r in ROWS]

    h = 0.30 * (len(rows) + 1) + 0.6
    fig = plt.figure(figsize=(FULL_MM * MM, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    t = ax.table(cellText=[[str(c) for c in r] for r in rows], colLabels=cols,
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(5.4)
    t.scale(1, 1.2)
    for (r_, c_), cell in t.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor(RULE)
        if r_ == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eef0f3")
        if c_ == 0:
            cell.set_text_props(ha="left")
        if r_ in (1, 2):
            cell.set_text_props(fontsize=4.8)
    save(fig, tb, "software_comparison")

    tex = [r"\begin{tabular}{l" + "c" * len(ROWS) + "}", r"\hline",
           " & ".join(["Capability"] + [r["name"] for r in ROWS]) + r" \\", r"\hline"]
    for r in rows:
        tex.append(" & ".join(str(c).replace("--", r"\textendash{}")
                              for c in r) + r" \\")
    tex += [r"\hline", r"\end{tabular}"]
    (tb / "software_comparison.tex").write_text("\n".join(tex))

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "documentation_accessed": ACCESSED,
        "assessment_basis": "the cited documentation at the stated version; not a "
                            "benchmark. A capability provided only by a "
                            "third-party plugin is marked partial and the plugin "
                            "is named.",
        "software": [{k: r[k] for k in ("name", "version", "docs")} for r in ROWS],
        "capabilities": CAPABILITIES,
        "footnotes": {str(n): FOOTNOTES[k] for k, n in
                      sorted(used.items(), key=lambda kv: kv[1])},
    }
    (tb / "software_comparison_metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"  wrote {tb / 'software_comparison.csv'} and .tex / .pdf / .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
