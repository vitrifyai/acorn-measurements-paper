#!/usr/bin/env python3
"""Vector-where-possible PDF twins of the raster figures.

Same geometry as the raster build, but each panel is placed as its own PDF, so
plot panels stay vector (text selectable, lines resolution-independent) and only
the micrograph content inside a panel remains raster. Panel letters are drawn as
real text in Helvetica-Bold.
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
import fitz
from PIL import Image

B=BUILD_DIR
DPI=600; PT=72.0
FULL_PT=7.0*PT
GUT_PT=0.035*PT
MARGIN_PT=0.02*PT
LETTER_PT=7.0

def panel_pdf(png: Path) -> Path:
    """The PDF twin of a panel PNG, if the generator wrote one."""
    p=png.with_suffix(".pdf")
    return p if p.exists() else None

def aspect(png: Path):
    im=Image.open(png); return im.width/im.height

def build_pdf(name, rows, out_dir, skip_letters=(), row_frac=None, letters=True):
    fracs=row_frac or [1.0]*len(rows)
    # geometry identical to the raster build: equal height per row, widths by aspect
    laid=[]
    for row,fr in zip(rows,fracs):
        width=FULL_PT*fr
        avail=width-GUT_PT*(len(row)-1)
        ar=[aspect(p) for p in row]
        h=avail/sum(ar)
        laid.append([(p, h*a, h) for p,a in zip(row,ar)])
    H=sum(max(h for _,_,h in row) for row in laid)+GUT_PT*(len(laid)-1)
    doc=fitz.open(); page=doc.new_page(width=FULL_PT, height=H)
    y=0.0; k=0; placed=[]
    vector=raster=0
    for row in laid:
        rw=sum(w for _,w,_ in row)+GUT_PT*(len(row)-1)
        x=(FULL_PT-rw)/2
        rh=max(h for _,_,h in row)
        for png,w,h in row:
            rect=fitz.Rect(x,y,x+w,y+h)
            src=panel_pdf(png)
            if src:
                sp=fitz.open(src); page.show_pdf_page(rect,sp,0); sp.close(); vector+=1
            else:
                page.insert_image(rect,filename=str(png)); raster+=1
            if letters and k not in skip_letters:
                ch=chr(ord("A")+len(placed))
                # same adaptive rule as the raster build: a black letter is
                # invisible on a dark micrograph, so sample the corner it will
                # sit on and stroke it in the opposite tone.
                im=Image.open(png).convert("L")
                box=int(min(im.width,im.height)*0.05)
                patch=np.asarray(im.crop((0,0,max(box,1),max(box,1))))
                dark = patch.mean() < 128 if patch.size else False
                ink = (1,1,1) if dark else (0,0,0)
                # render_mode 0 paints with `color`; mode 2 wrote the glyph into
                # the text layer without painting it, so the letter was extractable
                # but invisible.
                page.insert_text(fitz.Point(x+MARGIN_PT, y+MARGIN_PT+LETTER_PT),
                                 ch, fontname="hebo", fontsize=LETTER_PT,
                                 color=ink, render_mode=0)
                placed.append(ch)
            x+=w+GUT_PT; k+=1
        y+=rh+GUT_PT
    out_dir.mkdir(parents=True,exist_ok=True)
    out=out_dir/f"{name}.pdf"
    doc.save(str(out),deflate=True,garbage=4); doc.close()
    return dict(name=name,pt=[round(FULL_PT,1),round(H,1)],
                inches=[7.0,round(H/PT,3)],vector_panels=vector,raster_panels=raster,
                letters="".join(placed),kb=round(out.stat().st_size/1024))
