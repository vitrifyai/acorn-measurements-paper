#!/usr/bin/env python3
"""Figure S10: SEM intensity preparation, and what it does and does not explain."""
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

import csv, json, sys
from pathlib import Path
import numpy as np, tifffile
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.path.insert(0, str(FIGURE_SCRIPTS_DIR))
sys.path.insert(0, str(ADDITIONAL_INFO_DIR / "code"))
from common import COL_MM, FULL_MM, MM, SCANNING, REFERENCE, TRUE_POS, FALSE_POS, MISSED, GREY, apply_style
from renormalise_sem import crop_banner, dark_frac

apply_style()
W=WORK_DIR; AI=ADDITIONAL_INFO_DIR
SRC = RAW_SEM_DIR
H=BUILD_DIR
OUT=H/"supplementary_figures"; OUT.mkdir(parents=True,exist_ok=True)
PREPS=[("per-image 0.5-99.5 %ile","sem"),("raw 8-bit","sem_raw"),("global window","sem_global")]

man=json.loads((W/"real_sem/manifest.json").read_text())
cropped={}
for rec in man:
    s=rec["stem"]
    c=crop_banner(tifffile.imread(SRC/f"{s}.tif").astype(np.float32))
    p=np.array(Image.open(W/f"real_sem/images/{s}.png").convert("L"))
    if c.shape==p.shape: cropped[s]=(c,p)
pool=np.concatenate([c.ravel() for c,_ in cropped.values()])
glo,ghi=np.percentile(pool,[0.5,99.5])
prep={"sem":lambda c,p:p,"sem_raw":lambda c,p:np.clip(c,0,255).astype(np.uint8),
      "sem_global":lambda c,p:(np.clip((c-glo)/max(ghi-glo,1e-6),0,1)*255).astype(np.uint8)}
corr=json.load(open(W/"sem_intensity_correlations.json"))
fc={r["stem"]:r for r in csv.DictReader(open(AI/"data/results/sem/field_characterisation.csv"))}
def ev(sub):
    o=json.load(open(H/f"data/sem_intensity/{sub}_heldout_evaluation.json"))
    return o["across_seeds"]["f1"], o["across_seeds_confident_reference"]["f1"]
def perfield(sub):
    b={}
    for r in csv.DictReader(open(AI/f"data/results/{sub}/heldout_per_field.csv")):
        b.setdefault(r["stem"],[]).append(float(r["f1"]))
    return {k:float(np.mean(v)) for k,v in b.items()}

fig=plt.figure(figsize=(FULL_MM*MM, FULL_MM*MM*0.78))
gs=fig.add_gridspec(3,3,hspace=0.55,wspace=0.35)
demo=sorted(cropped)[0]
# A-C: the same field under the three preparations
for i,(lbl,sub) in enumerate(PREPS):
    ax=fig.add_subplot(gs[0,i]); a=prep[sub](*cropped[demo])
    ax.imshow(a,cmap="gray",vmin=0,vmax=255); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{lbl}\n{100*dark_frac(a):.0f}% of pixels below DN 32",fontsize=6)
# D: histograms
ax=fig.add_subplot(gs[1,0])
for (lbl,sub),c in zip(PREPS,(SCANNING,TRUE_POS,REFERENCE)):
    v=np.concatenate([prep[sub](*cropped[s]).ravel()[::37] for s in cropped])
    ax.hist(v,bins=64,range=(0,255),histtype="step",color=c,lw=1.1,density=True,label=lbl)
ax.axvline(32,color=MISSED,lw=0.8,ls="--")
ax.set_xlabel("8-bit value (DN)"); ax.set_ylabel("density"); ax.legend(frameon=False,fontsize=4.8)
# E: F1 by preparation
ax=fig.add_subplot(gs[1,1])
x=np.arange(3)
m=[ev(s)[0]["mean"] for _,s in PREPS]; sd=[ev(s)[0]["sd"] for _,s in PREPS]
mc=[ev(s)[1]["mean"] for _,s in PREPS]
ax.bar(x-0.2,m,0.4,yerr=sd,capsize=2.5,color=SCANNING,label="all reference")
ax.bar(x+0.2,mc,0.4,facecolor="white",edgecolor=SCANNING,hatch="////",label="confident only")
ax.set_xticks(x); ax.set_xticklabels([l.replace(" ","\n") for l,_ in PREPS],fontsize=4.8)
ax.set_ylabel("held-out F1"); ax.set_ylim(0,0.9); ax.legend(frameon=False,fontsize=4.8)
# F: between-field spread
ax=fig.add_subplot(gs[1,2])
for i,(lbl,sub) in enumerate(PREPS):
    v=list(perfield(sub).values())
    ax.plot(np.full(len(v),i)+np.random.default_rng(0).normal(0,.05,len(v)),v,"o",
            ms=3,mfc="none",color=SCANNING,alpha=.8)
    ax.plot([i-.22,i+.22],[np.mean(v)]*2,"-",color=MISSED,lw=1.4)
ax.set_xticks(range(3)); ax.set_xticklabels([l.replace(" ","\n") for l,_ in PREPS],fontsize=4.8)
ax.set_ylabel("per-field F1"); ax.set_title("bar = mean; between-field SD ~0.11-0.14 in every preparation",fontsize=5)
# G-H: the two exploratory relationships
for j,(sub,lbl) in enumerate((("sem","per-image %ile"),("sem_raw","raw 8-bit"))):
    ax=fig.add_subplot(gs[2,j]); pf=perfield(sub); stems=sorted(pf)
    d=[dark_frac(prep[sub](*cropped[s])) for s in stems]
    ax.plot(d,[pf[s] for s in stems],"o",ms=4,color=SCANNING,mfc="none")
    c=corr[sub]; ax.set_xlabel("fraction of pixels below DN 32"); ax.set_ylabel("per-field F1")
    ax.set_title(f"{lbl}: rho={c['dark_rho']}, p={c['dark_p']}",fontsize=5.4)
ax=fig.add_subplot(gs[2,2]); pf=perfield("sem_raw"); stems=[s for s in sorted(pf) if s in fc]
ax.plot([float(fc[s]["confident_frac"]) for s in stems],[pf[s] for s in stems],"o",ms=4,
        color=TRUE_POS,mfc="none")
c=corr["sem_raw"]; ax.set_xlabel("confident share of the reference"); ax.set_ylabel("per-field F1")
ax.set_title(f"raw 8-bit: rho={c['conf_rho']}, p={c['conf_p']}",fontsize=5.4)
for e in ("pdf","png"):
    fig.savefig(OUT/f"figureS10_sem_intensity.{e}",dpi=600,bbox_inches="tight")
print("wrote figureS10")
meta=dict(panels="A-C prepared field; D pooled histograms; E F1 by preparation; "
    "F per-field spread; G-H dark fraction vs score; I confident share vs score",
    correlations=corr, pooled_global_window_dn=[float(glo),float(ghi)],
    n_images=len(cropped),
    exploratory_note=("All correlations are Spearman over 8 held-out fields and are "
        "EXPLORATORY, not causal. The dark-fraction relationship is strong and significant "
        "ONLY in the published per-image preparation (rho=-0.881, p=0.004) and disappears in "
        "the raw (rho=+0.147, p=0.73) and global (rho=+0.098, p=0.82) preparations. "
        "Between-field SD is 0.14, 0.11 and 0.13 respectively, so changing the intensity "
        "preparation removed the dark-fraction artefact WITHOUT reducing the dominant "
        "between-field variation."))
(OUT/"figureS10_metadata.json").write_text(json.dumps(meta,indent=1))
print("wrote metadata")
