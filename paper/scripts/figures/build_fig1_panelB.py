#!/usr/bin/env python3
"""Figure 1 panel B: what the handoffs cost, and what removing them looks like.

The old version was three stacked rows of boxes asserting an architecture. The
introduction makes a sharper point -- calibration, provenance and the definition
of ground truth are lost in the handoffs BETWEEN tools, and the loss is
invisible downstream -- so the panel is drawn as that contrast, with the three
losses this work documents marked on the handoffs where they happen.
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

import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, str(FIGURE_SCRIPTS_DIR))
from common import FULL_MM, MM
from palette import P, apply_style
apply_style()
OUT=BUILD_DIR / "main_figures" / "panels"

STAGES=["acquire","annotate","train","predict","measure"]
# the three instances the introduction names, placed at the handoff each occurs in
LOSSES={0:("pixel size\nnot read","§2.2"),
        1:("ground truth\nnot the target","§5.4"),
        2:("intensity\nconvention","§5.5")}

fig=plt.figure(figsize=(FULL_MM*MM, FULL_MM*MM*0.40))
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,40); ax.axis("off")
w,gap,x0=15.6,3.4,4.0

def row(y,h,fc,ec,labels,dashed=False):
    for i,t in enumerate(labels):
        x=x0+i*(w+gap)
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.45,rounding_size=0.9",
            facecolor=fc,edgecolor=ec,lw=1.15,
            linestyle=(0,(4,2)) if dashed else "-"))
        ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=7.6,color=ec,fontweight="bold")

# ---- top: separate tools, and what falls out of the handoffs ----------------
yt=27.5; h=6.4
row(yt,h,"white",P.GREY,STAGES)
ax.text(x0-1.6,yt+h/2,"separate\ntools",ha="right",va="center",fontsize=7.0,
        color=P.GREY,fontweight="bold",linespacing=1.3)
for i in range(len(STAGES)-1):
    xa=x0+i*(w+gap)+w; xb=xa+gap
    ax.add_patch(FancyArrowPatch((xa,yt+h/2),(xb,yt+h/2),arrowstyle="-|>",
        mutation_scale=8,lw=1.0,color=P.GREY,shrinkA=0,shrinkB=0))
    if i in LOSSES:
        txt,sec=LOSSES[i]
        ax.plot([xa+gap/2,xa+gap/2],[yt+h/2+0.9,yt+h+3.2],lw=0.8,color=P.FALSE_POS)
        ax.text(xa+gap/2,yt+h+3.6,txt,ha="center",va="bottom",fontsize=5.5,
                color=P.FALSE_POS,linespacing=1.25)
        ax.text(xa+gap/2,yt+h+7.6,sec,ha="center",va="bottom",fontsize=5.0,color=P.GREY)
        ax.text(xa+gap/2,yt+h/2,"×",ha="center",va="center",fontsize=11,
                color=P.FALSE_POS,fontweight="bold")
ax.text(50,yt-2.6,"each handoff is a place for calibration, provenance or the definition of "
        "ground truth to be lost — and the loss is invisible downstream",
        ha="center",va="top",fontsize=6.0,color=P.GREY)

# ---- bottom: one workspace --------------------------------------------------
yb=10.0
row(yb,h,"white",P.TRANSMISSION,STAGES)
ax.text(x0-1.6,yb+h/2,"ACORN",ha="right",va="center",fontsize=7.4,
        color=P.TRANSMISSION,fontweight="bold")
xL,xR=x0,x0+len(STAGES)*(w+gap)-gap
ax.add_patch(FancyBboxPatch((xL,1.6),xR-xL,6.0,boxstyle="round,pad=0.45,rounding_size=0.9",
    facecolor=P.BAND,edgecolor=P.TRANSMISSION,lw=1.15))
ax.text((xL+xR)/2,5.6,"one calibration  ·  one annotation store  ·  one provenance record",
        ha="center",va="center",fontsize=7.0,color=P.TRANSMISSION,fontweight="bold")
ax.text((xL+xR)/2,3.0,"reached from the desktop, the command line, Python, or plain language",
        ha="center",va="center",fontsize=6.1,color=P.GREY)
for i in range(len(STAGES)):
    x=x0+i*(w+gap)+w/2
    ax.plot([x,x],[7.7,yb],ls=(0,(2,2)),lw=0.9,color=P.TRANSMISSION,zorder=0)
for i in range(len(STAGES)-1):
    xa=x0+i*(w+gap)+w
    ax.add_patch(FancyArrowPatch((xa,yb+h/2),(xa+gap,yb+h/2),arrowstyle="-|>",
        mutation_scale=8,lw=1.0,color=P.TRANSMISSION,shrinkA=0,shrinkB=0))
for e in ("pdf","png"):
    fig.savefig(OUT/f"1B_handoffs.{e}",dpi=600,bbox_inches="tight")
print("wrote 1B_handoffs")
