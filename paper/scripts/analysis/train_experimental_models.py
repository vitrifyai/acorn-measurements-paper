#!/usr/bin/env python3
"""Train task-specific models from the reviewed experimental annotations.

One model per modality, three training seeds each, through ACORN's own
in-application trainer (``acorn.core.yolo_trainer.YOLOTrainer``) on the dataset
built by ``build_experimental_datasets.py``. Everything a rerun needs is
recorded: seed, hyperparameters, the resolved Ultralytics argument set,
hardware, training time, and the checkpoint hash.

    python train_experimental_models.py --dataset work/sem/dataset --tag sem \
        --seeds 0 1 2 --device 2 --out results/sem
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
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def hardware(device: str) -> dict:
    import torch
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device_arg": device}
    try:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count_visible"] = torch.cuda.device_count()
    except Exception:
        info["gpu"] = None
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    return info


def git_sha(repo=ACORN_SOURCE_DIR) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--base-model", default="yolo11s-seg.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from acorn.core.yolo_trainer import YOLOTrainer

    runs = []
    for seed in args.seeds:
        proj = args.out / f"seed{seed}"
        if (proj / "training_info.json").exists():
            print(f"  seed {seed}: already trained, reusing", flush=True)
            runs.append(json.loads((proj / "training_info.json").read_text())
                        | {"seed": seed, "project_dir": str(proj)})
            continue
        print(f"  seed {seed}: training {args.base_model} "
              f"{args.epochs} epochs, imgsz {args.imgsz}, batch {args.batch}", flush=True)
        log_path = proj / "train.log"
        proj.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "w", buffering=1)
        t0 = time.time()
        tr = YOLOTrainer(dataset_dir=args.dataset, base_model=args.base_model,
                         epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                         devices=[int(args.device)] if args.device.isdigit() else args.device,
                         seed=seed, project_dir=proj,
                         log_cb=lambda m: fh.write(str(m) + "\n"))
        best = tr.train()
        wall = time.time() - t0
        fh.close()
        info = json.loads((proj / "training_info.json").read_text())
        # the resolved Ultralytics argument set, so augmentations and the
        # optimiser are recorded as used rather than as intended
        arg_files = sorted(proj.rglob("args.yaml"))
        resolved = arg_files[0].read_text() if arg_files else ""
        info |= {"seed": seed, "project_dir": str(proj),
                 "training_seconds": wall,
                 "checkpoint": str(best),
                 "checkpoint_sha256": sha256_file(Path(best)),
                 "checkpoint_bytes": Path(best).stat().st_size,
                 "resolved_ultralytics_args": resolved}
        (proj / "run_record.json").write_text(json.dumps(info, indent=1))
        runs.append(info)
        print(f"    {wall / 60:.1f} min -> {best}", flush=True)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": args.tag,
        "dataset": str(args.dataset),
        "acorn_commit": git_sha(),
        "hyperparameters": {
            "architecture": args.base_model,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "trainer": "acorn.core.yolo_trainer.YOLOTrainer (Ultralytics backend)",
            "note": "optimiser, learning rate and augmentations are Ultralytics "
                    "defaults for segmentation; the resolved set is stored "
                    "verbatim per run under resolved_ultralytics_args",
        },
        "hardware": hardware(args.device),
        "seeds": args.seeds,
        "runs": runs,
    }
    (args.out / "training_runs.json").write_text(json.dumps(out, indent=1))
    print(f"\n  {len(runs)} runs -> {args.out / 'training_runs.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
