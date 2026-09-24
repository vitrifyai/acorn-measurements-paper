#!/usr/bin/env python3
"""Replicate the annotation-burden curve so figure 4D can carry error bars.

The single-seed curve fixed both the training seed AND which images were
annotated. With a pool of only eight acquired micrographs, *which* two or four
you happen to annotate is the larger source of variation, so replicates resample
the subset as well as the seed. At n=8 the pool is exhausted, so the spread
there reflects training stochasticity only -- a real limitation, recorded.
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

import json, os, shutil, random
from pathlib import Path
os.environ["YOLO_VERBOSE"] = "0"
from ultralytics import YOLO

W = WORK_DIR
RUNS = MODEL_RUNS_DIR
POOL = sorted(p.stem for p in (W / "real_ft/images/train").glob("*.png"))
SIM_INIT = str(MODEL_CHECKPOINTS_DIR / "spores_sem/best.pt")
NAT_INIT = "yolo11s-seg.pt"
REPS, OUT = 4, W / "burden_replicates"
OUT.mkdir(exist_ok=True)

def build(subset, dst):
    """A dataset with `subset` as train and the SAME fixed 18-image val split."""
    if dst.exists(): shutil.rmtree(dst)
    for s in ("train", "val"):
        (dst / "images" / s).mkdir(parents=True); (dst / "labels" / s).mkdir(parents=True)
    for stem in subset:
        shutil.copy(W / f"real_ft/images/train/{stem}.png", dst / f"images/train/{stem}.png")
        shutil.copy(W / f"real_ft/labels/train/{stem}.txt", dst / f"labels/train/{stem}.txt")
    for p in (W / "real_ft/images/val").glob("*.png"):
        shutil.copy(p, dst / f"images/val/{p.name}")
        shutil.copy(W / f"real_ft/labels/val/{p.stem}.txt", dst / f"labels/val/{p.stem}.txt")
    (dst / "data.yaml").write_text(
        f"path: {dst}\ntrain: images/train\nval: images/val\n"
        "nc: 2\nnames: ['spore', 'debris']\n")
    return dst / "data.yaml"

def main():
    results, done = [], 0
    jobs = []
    for n in (2, 4, 8):
        for rep in range(REPS):
            r = random.Random(1000 * n + rep)
            sub = POOL if n == 8 else sorted(r.sample(POOL, n))
            for arm, init in (("simulation", SIM_INIT), ("natural image", NAT_INIT)):
                jobs.append((n, rep, sub, arm, init))
    print(f"{len(jobs)} runs\n", flush=True)
    prior = {}
    pj = W / "burden_replicates.json"
    if pj.exists():
        for r in json.loads(pj.read_text()):
            prior[(r["n"], r["rep"], r["arm"])] = r
    for n, rep, sub, arm, init in jobs:
        tag = f"rep_{arm.split()[0]}_{n}_{rep}"
        data = build(sub, OUT / f"ds_{n}_{rep}")
        if (n, rep, arm) in prior:                      # resume
            results.append(prior[(n, rep, arm)]); done += 1
            print(f"[{done}/{len(jobs)}] {tag}  cached", flush=True); continue
        wt = RUNS / tag / "weights/best.pt"
        if not wt.exists():
                YOLO(init).train(data=str(data), epochs=120, batch=4, imgsz=640, device=1,
                             seed=rep, project=str(RUNS), name=tag, exist_ok=True,
                             patience=40, workers=4, verbose=False, plots=False)
        m = YOLO(str(wt))
        v = m.val(data=str(data), split="val", verbose=False, plots=False, device=1)
        results.append(dict(n=n, rep=rep, arm=arm, subset=sub,
                            mask_map50=float(v.seg.map50), box_map50=float(v.box.map50)))
        done += 1
        print(f"[{done}/{len(jobs)}] {tag}  mask mAP50={v.seg.map50:.3f}", flush=True)
        (W / "burden_replicates.json").write_text(json.dumps(results, indent=1))
    print("\ndone")

if __name__ == "__main__":
    main()
