#!/usr/bin/env python3
"""Figure 2 panel D rebuilt: every movie shown, bimodal failure made visible."""
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

import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(FIGURE_SCRIPTS_DIR))
from common import COL_MM, MM, TRANSMISSION, MISSED, GREY, apply_style
apply_style()
H=BUILD_DIR
OUT=BUILD_DIR/"build/fig2_motion"; OUT.mkdir(parents=True,exist_ok=True)
rows=json.load(open(WORK_DIR / "motion_all_movies.json"))
DOSES=[100.0,20.0,4.0,1.0,0.25,0.06,0.015]
fig=plt.figure(figsize=(COL_MM*MM, COL_MM*MM*0.80)); ax=fig.add_subplot(111)
rng=np.random.default_rng(0)
for e in DOSES:
    v=[r for r in rows if r["dose_condition"]==f"dose_{e:g}"]
    y=np.array([r["alignment_rmse_px"] for r in v])
    ok=np.array([r["alignment_success"] for r in v])
    x=e*np.exp(rng.normal(0,0.045,len(v)))          # jitter in log space
    # Filled = sub-pixel recovery, open = failed. Every movie is plotted, so the
    # split at 1 e/px/frame is visible rather than averaged away.
    ax.scatter(x[ok],y[ok],s=17,color=TRANSMISSION,zorder=3,lw=0)
    ax.scatter(x[~ok],y[~ok],s=17,facecolors="none",edgecolors=TRANSMISSION,lw=0.9,zorder=3)
    med=float(np.median(y))
    ax.plot([e*0.78,e*1.28],[med]*2,"-",color=MISSED,lw=1.5,zorder=4)
ax.set_xscale("log"); ax.set_yscale("log")
ax.axhline(1.0,color=GREY,lw=0.8,ls=":")
ax.axhspan(30,1e4,color=GREY,alpha=0.12,lw=0)
ax.set_xticks(DOSES); ax.set_xticklabels([f"{d:g}" for d in DOSES],fontsize=6.2)
ax.minorticks_off(); ax.invert_xaxis()
ax.set_xlabel("electrons per pixel per frame")
ax.set_ylabel("drift recovery error, RMSE (pixels)")
ax.set_ylim(0.03,3e3); ax.grid(alpha=0.15,lw=0.4,which="major")
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([],[],marker="o",ls="none",color=TRANSMISSION,ms=4,label="sub-pixel recovery"),
    Line2D([],[],marker="o",ls="none",mfc="none",mec=TRANSMISSION,ms=4,label="failed (RMSE > 1 px)"),
    Line2D([],[],color=MISSED,lw=1.5,label="median of 6 movies")],
    frameon=False,fontsize=5.4,loc="upper left")
for e in ("pdf","png"):
    fig.savefig(OUT/f"2D_dose_dependence.{e}",dpi=600,bbox_inches="tight")
print("wrote 2D (individual movies, median bars, no connecting line)")
