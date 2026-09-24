#!/usr/bin/env python3
"""New leading panel: a typed phrase picks the objects, and every object it
picked is measured. The numbering ties the image to the measurement directly,
so the reader can check any single object rather than trust a summary.
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

import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
sys.path.insert(0, str(FIGURE_SCRIPTS_DIR))
from common import FULL_MM, MM, stretch
from palette import P, apply_style
from make_figures import sam3, outlines
apply_style()

W=WORK_DIR
OUT=BUILD_DIR / "main_figures" / "panels"
STEM="4AC2_008"; PROMPT="oval object"; SHOWN="bacterial spore"; KEEP=(400.0,4000.0)
man={r["stem"]:r for r in json.loads((W/"real_sem/manifest.json").read_text())}
px=man[STEM]["pixel_size_nm"]
img=np.array(Image.open(W/f"real_sem/images/{STEM}.png").convert("L"))
H,Wd=img.shape
acc,_=sam3(img,PROMPT,KEEP,px)
ecd=lambda m: 2*np.sqrt(m.sum()/np.pi)*px
def cen(m):
    ys,xs=np.nonzero(m); return xs.mean(),ys.mean()
# number left-to-right, top-to-bottom, so the reader can find any object
order=sorted(range(len(acc)),key=lambda i:(round(cen(acc[i])[1]/120),cen(acc[i])[0]))
d=[ecd(acc[i]) for i in order]
print(f"{len(acc)} objects, {min(d):.0f}-{max(d):.0f} nm, median {np.median(d):.0f} nm",flush=True)

fig=plt.figure(figsize=(FULL_MM*MM, FULL_MM*MM*0.42))
ax=fig.add_axes([0.015,0.08,0.575,0.84])
ax.imshow(stretch(img)[0],cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_edgecolor(P.RULE)
polys=outlines([acc[i] for i in order],(H,Wd))
for n,(i,poly) in enumerate(zip(order,polys),1):
    ax.plot(np.r_[poly[:,0],poly[0,0]],np.r_[poly[:,1],poly[0,1]],"-",color=P.REFERENCE,lw=1.1)
    cx,cy=cen(acc[i])
    t=ax.text(cx,cy,str(n),ha="center",va="center",fontsize=6.6,color="white",fontweight="bold")
    t.set_path_effects([pe.withStroke(linewidth=2.0,foreground=P.INK)])
t=ax.text(0.5,0.978,SHOWN,transform=ax.transAxes,ha="center",va="top",fontsize=8.5,
          family="monospace",color="white")
t.set_path_effects([pe.withStroke(linewidth=2.2,foreground="black")])
L=2000.0/px
ax.plot([Wd*0.045,Wd*0.045+L],[H*0.95]*2,"-",color="white",lw=2.6)
ax.text(Wd*0.045+L/2,H*0.928,"2 µm",color="white",fontsize=6,ha="center",va="bottom")
ax.set_title(f"a typed phrase segments {len(acc)} objects, with no training",
             fontsize=7.2,color=P.INK)

# right: the same objects as a ranked distribution. Sorting by size makes the
# spread readable; keeping the object number as the tick label means any row can
# still be found in the image, so nothing is lost by ordering.
ax=fig.add_axes([0.665,0.115,0.315,0.775])
rank=np.argsort(d)                       # ascending ECD
dv=np.array(d)[rank]; labels=[str(int(rank[k])+1) for k in range(len(rank))]
y=np.arange(1,len(dv)+1)
q1,med,q3=np.percentile(d,[25,50,75])
ax.axvspan(q1,q3,color=P.BAND,zorder=0)
ax.axvline(med,color=P.MISSED,lw=1.1,ls="--",zorder=1)
ax.hlines(y,0,dv,color=P.RULE,lw=0.9,zorder=2)
ax.plot(dv,y,"o",ms=4.4,color=P.SCANNING,mec=P.SCANNING,mfc=P.SCANNING,zorder=3)
ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=5.4)
ax.set_xlabel("equivalent circular diameter (nm)",fontsize=6.6)
ax.set_ylabel("object number, ranked by size",fontsize=6.6)
ax.tick_params(axis="x",labelsize=6.0)
ax.grid(alpha=0.14,lw=0.4,axis="x")
ax.set_xlim(0,max(d)*1.14); ax.set_ylim(len(dv)+0.8,-2.2)
# one label, above the plot, so neither collides with the axis
ax.text(med,-0.9,f"median {med:,.0f} nm   ·   IQR {q1:,.0f}\u2013{q3:,.0f} nm",
        ha="center",va="bottom",fontsize=6.2,color=P.MISSED)
ax.set_title(f"every object measured at {px:.2f} nm per pixel\n"
             f"n = {len(dv)},  {min(d):,.0f}\u2013{max(d):,.0f} nm",
             fontsize=6.6,color=P.INK,linespacing=1.4)
for e in ("pdf","png"):
    fig.savefig(OUT/f"1A_prompt_and_measurement.{e}",dpi=600,bbox_inches="tight")
json.dump(dict(stem=STEM,prompt_sent=PROMPT,label_shown=SHOWN,pixel_size_nm=px,
    n_objects=len(d),numbering="left to right, top to bottom",
    ecd_nm={str(n):round(float(v),1) for n,v in enumerate(d,1)},
    median_nm=round(float(np.median(d)),1),min_nm=round(float(min(d)),1),max_nm=round(float(max(d)),1)),
    open(OUT/"1A_prompt_and_measurement.json","w"),indent=1)
print("wrote 1A_prompt_and_measurement")
