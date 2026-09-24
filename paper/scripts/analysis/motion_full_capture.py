#!/usr/bin/env python3
"""Per-movie motion-correction record for the handoff package.

The earlier replicate run kept only summary RMSE lists. This one keeps every
per-movie quantity the deposit CSV asks for, including the failures: a movie
whose alignment blew up keeps its measured error rather than becoming a blank.
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

import json, subprocess, sys
from pathlib import Path
import numpy as np
FIG = FIGURE_SCRIPTS_DIR
sys.path.insert(0, str(FIG))
from make_fig5 import make_movie, PX_A, N_FRAMES, TOTAL_DRIFT_PX
from acorn.core.frame_processor import motion_correct_frames, mean_average

lap = lambda a: float(np.var(np.gradient(a.astype(np.float32))[0]))
REPS, DOSES, NOMINAL = 6, [100.0, 20.0, 4.0, 1.0, 0.25, 0.06, 0.015], 300.0
SUCCESS_PX = 1.0          # sub-pixel recovery
def commit():
    try:
        return subprocess.run(["git", "-C", str(ACORN_SOURCE_DIR),"rev-parse","HEAD"],
                              capture_output=True,text=True).stdout.strip() or "NOT AVAILABLE"
    except Exception: return "NOT AVAILABLE"

def one(e, k, cond, panel):
    f, t, _ = make_movie(e_per_px_frame=e, seed=100+k, scene_seed=4+k, traj_seed=k)
    c, s = motion_correct_frames(f)
    r = -(s - s[0]); tt = t - t[0]
    err = np.hypot(*(r - tt).T)
    before, after = lap(mean_average(f)), lap(c)
    rmse = float(np.sqrt((err**2).mean()))
    return dict(
        movie_id=f"{cond}_rep{k}", dose_condition=cond,
        electrons_per_pixel_per_frame=e, replicate=k,
        random_seed=100+k, specimen_seed=4+k, trajectory_seed=k, noise_seed=100+k,
        number_of_frames=N_FRAMES, pixel_size=f"{PX_A} A/px",
        input_motion_px=float(TOTAL_DRIFT_PX),
        alignment_rmse_px=rmse, maximum_error_px=float(err.max()),
        sharpness_before=before, sharpness_after=after,
        sharpness_gain_percent=float(100*(after/max(before,1e-12)-1)),
        alignment_success=bool(rmse < SUCCESS_PX),
        failure_definition=f"alignment_success = alignment_rmse_px < {SUCCESS_PX} px",
        figure_panel=panel, source_file=str(FIG/"make_fig5.py"),
        software_version="acorn-sim-playground working tree",
        git_commit=commit())

def main():
    rows = [one(NOMINAL, k, "nominal_300", "2A/2B/2C") for k in range(REPS)]
    print(f"nominal done ({len(rows)})", flush=True)
    for e in DOSES:
        for k in range(REPS):
            rows.append(one(e, k, f"dose_{e:g}", "2D"))
        print(f"  dose {e:g} done", flush=True)
        WORK_DIR / "motion_all_movies.json".write_text(json.dumps(rows, indent=1))
    WORK_DIR / "motion_all_movies.json".write_text(json.dumps(rows, indent=1))
    print("total movies:", len(rows))

if __name__ == "__main__":
    main()
