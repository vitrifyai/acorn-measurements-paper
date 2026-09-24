#!/usr/bin/env python3
"""Per-execution record for every simulation-initialisation run."""
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

import hashlib, json, os, subprocess
from pathlib import Path
os.environ["YOLO_VERBOSE"]="0"
import numpy as np
from ultralytics import YOLO

W=WORK_DIR; R=MODEL_RUNS_DIR
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()
def commit():
    try: return subprocess.run(["git", "-C", str(ACORN_SOURCE_DIR),"rev-parse","HEAD"],
                               capture_output=True,text=True).stdout.strip() or "NOT AVAILABLE"
    except Exception: return "NOT AVAILABLE"
GC=commit()

reps=json.loads((W/"burden_replicates.json").read_text())
ext=json.loads((W/"burden_n8_extended.json").read_text())
subsets={(r["n"],r["rep"]):r.get("subset") for r in reps if r.get("subset")}
jobs=[]
for r in reps:
    if r["n"] in (2,4): jobs.append((r["arm"],r["n"],r["rep"]))
for r in ext:
    if r["n"]==8: jobs.append((r["arm"],r["n"],r["rep"]))
jobs=sorted(set(jobs))
print(len(jobs),"runs",flush=True)

VAL=sorted(p.stem for p in (W/"real_ft/images/val").glob("*.png"))
out=[]
for arm,n,rep in jobs:
    tag=f"rep_{arm.split()[0]}_{n}_{rep}"
    d=R/tag; wt=d/"weights/best.pt"
    ds=W/f"burden_replicates/ds_{n}_{rep}" if n!=8 else W/"burden_replicates/ds_8_0"
    data=str(ds/"data.yaml")
    m=YOLO(str(wt))
    v=m.val(data=data,split="val",verbose=False,plots=False,device=1)
    # detection totals on the same val fields
    nd,cs=0,0.0
    for s in VAL:
        r0=m.predict(str(ds/f"images/val/{s}.png"),verbose=False,conf=0.25)[0]
        nd+=len(r0.boxes); cs+=float(r0.boxes.conf.sum()) if len(r0.boxes) else 0.0
    import csv as _csv
    rows=list(_csv.DictReader(open(d/"results.csv")))
    p=float(v.seg.mp); rc=float(v.seg.mr)
    out.append(dict(
        initialization=arm, number_acquired_micrographs=n, replicate=rep,
        training_micrograph_ids=";".join(subsets.get((n,rep)) or
            sorted(x.stem for x in (ds/"images/train").glob("*.png"))),
        subset_id=f"ds_{n}_{rep}" if n!=8 else "ds_8_full_pool",
        seed=rep, deterministic=True, checkpoint_hash=sha(wt),
        number_epochs=len(rows),
        early_stopping_epoch=(len(rows) if len(rows)>=120 else len(rows)),
        validation_micrograph_ids=";".join(VAL),
        mask_map50=float(v.seg.map50), mask_map50_95=float(v.seg.map),
        precision=p, recall=rc, f1=float(2*p*rc/(p+rc)) if (p+rc) else 0.0,
        number_detections=int(nd), confidence_sum=round(float(cs),6),
        software_version="ultralytics "+__import__("ultralytics").__version__,
        git_commit=GC))
    print(f"  {tag}  mAP50={v.seg.map50:.6f}  dets={nd}",flush=True)
    (W/"sim_all_runs.json").write_text(json.dumps(out,indent=1))
print("done")
