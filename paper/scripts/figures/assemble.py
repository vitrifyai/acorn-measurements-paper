#!/usr/bin/env python3
"""Assemble every main and SI figure at 600 dpi, 7.0 in wide, from panels that
were generated at that size -- nothing here is enlarged.

Rules enforced:
  * plate width exactly 4200 px (7.0 in) or 1950 px (3.25 in) at 600 dpi
  * panels are placed at their generated resolution; only sub-pixel fitting
    rescales, never an upscale of a small source
  * panel letters are bold 7 pt at final size (58 px at 600 dpi), upper-left,
    inside a margin so they never sit on axes, scale bars or specimen
  * legends and colour keys are NOT panels and get no letter
  * TIFF is written with LZW (lossless); PNG carries 600 dpi metadata
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPI=600
FULL_PX=int(round(7.0*DPI))        # 4200
COL_PX=int(round(3.25*DPI))        # 1950
GUT=int(round(0.035*DPI))          # ~21 px gutter
MARGIN=int(round(0.02*DPI))        # 12 px
LETTER_PT=7.0
FONT_CANDIDATES=["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                 "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"]
def letter_font():
    px=int(round(LETTER_PT/72.0*DPI))          # 58 px
    for c in FONT_CANDIDATES:
        if Path(c).exists(): return ImageFont.truetype(c,px), px
    return ImageFont.load_default(), px

SCALES=[]

def fit_row(paths, width, gutter=GUT):
    """Scale a row of panels to share `width`, preserving every aspect ratio."""
    ims=[Image.open(p).convert("RGB") for p in paths]
    n=len(ims); avail=width-gutter*(n-1)
    # common height so the row is flush; width follows each aspect ratio
    ar=[im.width/im.height for im in ims]
    h=avail/sum(ar)
    out=[]
    for src,im,a in zip(paths,ims,ar):
        w=int(round(h*a)); hh=int(round(h))
        SCALES.append(dict(panel=Path(src).name, src_px=[im.width,im.height],
                           placed_px=[max(w,1),max(hh,1)],
                           scale=round(w/im.width,4)))
        out.append(im.resize((max(w,1),max(hh,1)),Image.LANCZOS))
    # correct rounding drift on the last panel
    drift=width-(sum(i.width for i in out)+gutter*(n-1))
    if drift and out:
        last=out[-1]; out[-1]=last.resize((max(last.width+drift,1),last.height),Image.LANCZOS)
    return out

def build(name, rows, out_dir, width=FULL_PX, letters=True, skip_letters=(),
          row_frac=None):
    """rows: list of lists of panel paths. skip_letters: indices (global panel
    order) that are legends/keys and must NOT receive a letter. row_frac: per-row
    fraction of the plate width, so a lone panel need not be blown up to full
    width and drive the figure past a page height."""
    f,fpx=letter_font()
    fracs=row_frac or [1.0]*len(rows)
    built=[fit_row(r,int(round(width*fr))) for r,fr in zip(rows,fracs)]
    H=sum(max(i.height for i in row) for row in built)+GUT*(len(built)-1)
    plate=Image.new("RGB",(width,H),"white")
    d=ImageDraw.Draw(plate)
    y=0; k=0; placed=[]
    for row in built:
        rw=sum(i.width for i in row)+GUT*(len(row)-1)
        x=(width-rw)//2                      # centre a narrower row
        rh=max(i.height for i in row)
        for im in row:
            plate.paste(im,(x,y))
            if letters and k not in skip_letters:
                ch=chr(ord("A")+len(placed))
                # A black letter vanishes on a dark micrograph. Sample the patch the
                # letter will sit on and pick the ink that contrasts with it, with a
                # thin opposite-colour stroke so it reads on mixed backgrounds too.
                patch=np.asarray(im.crop((MARGIN, MARGIN,
                                          min(MARGIN+fpx, im.width),
                                          min(MARGIN+fpx, im.height))).convert("L"))
                dark = patch.mean() < 128 if patch.size else False
                ink, halo = ("white", "black") if dark else ("black", "white")
                d.text((x+MARGIN,y+MARGIN), ch, fill=ink, font=f,
                       stroke_width=max(1,fpx//24), stroke_fill=halo)
                placed.append(ch)
            x+=im.width+GUT; k+=1
        y+=rh+GUT
    out_dir.mkdir(parents=True,exist_ok=True)
    png=out_dir/f"{name}.png"; tif=out_dir/f"{name}.tif"
    plate.save(png,dpi=(DPI,DPI))
    plate.save(tif,dpi=(DPI,DPI),compression="tiff_lzw")
    global SCALES
    scales=SCALES[:]; SCALES.clear()
    worst=min((x["scale"] for x in scales), default=1.0)
    return dict(name=name,px=[plate.width,plate.height],
                panel_scales=scales, worst_panel_scale=worst,
                effective_text_pt=round(7.0*worst,2),
                inches=[round(plate.width/DPI,3),round(plate.height/DPI,3)],
                panels=len(placed),letters="".join(placed),
                letter_pt=LETTER_PT,letter_px=fpx)
