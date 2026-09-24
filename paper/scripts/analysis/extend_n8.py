#!/usr/bin/env python3
"""Extend the n=8 condition to ten seeds per arm.

Four seeds gave a bit-identical mask mAP@50 for the simulation arm despite
different weights and different training curves, i.e. an AP plateau rather than
the seed having no effect. Ten seeds test whether the plateau holds.
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

import json, os
from pathlib import Path
os.environ["YOLO_VERBOSE"] = "0"
from ultralytics import YOLO

W = WORK_DIR
RUNS = MODEL_RUNS_DIR
DATA = str(W / "burden_replicates/ds_8_0/data.yaml")     # all eight, fixed val
ARMS = (("simulation", str(MODEL_CHECKPOINTS_DIR / "spores_sem/best.pt")),
        ("natural image", "yolo11s-seg.pt"))
OUTJ = W / "burden_n8_extended.json"

def main():
    res = json.loads(OUTJ.read_text()) if OUTJ.exists() else []
    have = {(r["arm"], r["rep"]) for r in res}
    # carry the first four over from the main replicate file
    for r in json.loads((W / "burden_replicates.json").read_text()):
        if r["n"] == 8 and (r["arm"], r["rep"]) not in have:
            res.append(r); have.add((r["arm"], r["rep"]))
    jobs = [(a, i, init) for a, init in ARMS for i in range(4, 10)
            if (a, i) not in have]
    print(f"{len(jobs)} new runs (target 10 seeds x 2 arms)", flush=True)
    for arm, rep, init in jobs:
        tag = f"rep_{arm.split()[0]}_8_{rep}"
        wt = RUNS / tag / "weights/best.pt"
        if not wt.exists():
            YOLO(init).train(data=DATA, epochs=120, batch=4, imgsz=640, device=1,
                             seed=rep, project=str(RUNS), name=tag, exist_ok=True,
                             patience=40, workers=4, verbose=False, plots=False)
        v = YOLO(str(wt)).val(data=DATA, split="val", verbose=False,
                              plots=False, device=1)
        res.append(dict(n=8, rep=rep, arm=arm, mask_map50=float(v.seg.map50),
                        box_map50=float(v.box.map50)))
        print(f"  {tag}  mask {v.seg.map50:.6f}", flush=True)
        OUTJ.write_text(json.dumps(res, indent=1))
    print("done", flush=True)

if __name__ == "__main__":
    main()
