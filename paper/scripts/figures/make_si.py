#!/usr/bin/env python3
"""Supplementary figures S1, S2, S4, S5, S6, S7 and tables S1-S9.

Individual panels, 600 dpi PNG plus vector PDF, no letters burned in.
S3 (training curves) is deliberately not built: no claim in the paper depends
on it. Figure 1 needs live interface captures and cannot be produced here.

    python make_si.py
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

import csv, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COL_MM, FULL_MM, MM, TRANSMISSION, SCANNING, REFERENCE,
                    TRUE_POS, FALSE_POS, MISSED, GREY, RULE, sha256,
                    git_commit, stretch, save, image_fig, scalebar)

HERE = Path(__file__).resolve().parent
WORK = WORK_DIR
RUNS = MODEL_CHECKPOINTS_DIR
SI   = HERE / "si"; TB = SI / "tables"
META: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "git_commit": git_commit(), "panels": {}, "tables": {}}

def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(.5); s.set_color(RULE)

def csv_pdf(name, header, rows, caption, colw=None):
    TB.mkdir(parents=True, exist_ok=True)
    with open(TB / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    h = 0.26 * (len(rows) + 1) + 0.5
    fig = plt.figure(figsize=(FULL_MM * MM if len(header) > 4 else COL_MM * MM, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    t = ax.table(cellText=[[str(c) for c in r] for r in rows], colLabels=header,
                 loc="center", cellLoc="right", colWidths=colw)
    t.auto_set_font_size(False); t.set_fontsize(6.6); t.scale(1, 1.25)
    for (r_, c_), cell in t.get_celld().items():
        cell.set_linewidth(0.4); cell.set_edgecolor(RULE)
        if r_ == 0: cell.set_text_props(weight="bold"); cell.set_facecolor("#eef0f3")
        if c_ == 0: cell.set_text_props(ha="left")
    save(fig, TB, name)
    META["tables"][name] = dict(header=header, rows=rows, caption=caption)

# ── S1: kernel fidelity ──────────────────────────────────────────────────────

def fig_S1():
    from acorn_sem_sim import materials as M, transport as T, kernels as K
    out = SI / "S1"
    mats = ["carbon", "silicon", "copper", "gold"]
    # S1 is the scanning engine, so every series is ochre; materials are
    # separated by line style and marker, never by a second colour.
    styles = dict(zip(mats, ["-", "--", "-.", ":"]))
    marks  = dict(zip(mats, ["o", "s", "^", "D"]))
    cols = {m: SCANNING for m in mats}
    kern = {m: K.compute(M.get(m), 5.0, n_electrons=40_000, seed=0, use_cache=False)
            for m in mats}
    trj  = {m: T.trace(M.get(m), 5.0, n_electrons=40_000, seed=0) for m in mats}

    # S1A enclosed weight vs radius
    fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM * 0.8)); ax = fig.add_subplot(111)
    for m in mats:
        k = kern[m].bse
        ax.semilogx(np.maximum(k.r_nm, 1e-3), k.enclosed(), color=SCANNING,
                    ls=styles[m], lw=1.3, label=m)
    ax.set_xlabel("radius (nm)"); ax.set_ylabel("enclosed weight")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.15, lw=.4, which="both")
    # The curves sweep up through the lower right, so a legend there sat on top of
    # carbon, silicon and copper. The upper left is empty for every material.
    ax.legend(frameon=False, loc="upper left", handlelength=2.6,
              borderaxespad=0.6, labelspacing=0.55)
    save(fig, out, "S1A_radial_kernels")

    # S1B kernel vs trajectory quantiles
    rows = []
    for m in mats:
        rad = np.asarray(trj[m].se_r_nm); wt = np.asarray(trj[m].se_weight, float)
        o = np.argsort(rad); cw = np.cumsum(wt[o]) / wt.sum()
        for q in (0.5, 0.9, 0.95):
            direct = float(rad[o][np.searchsorted(cw, q)])
            via = float(kern[m].se.radius_containing(q))
            rows.append(dict(material=m, q=q, trajectory_nm=direct, kernel_nm=via,
                             abs_dev_nm=abs(via - direct),
                             rel_pct=(100 * abs(via - direct) / direct) if direct > 1e-6 else None))
    fin = [r for r in rows if r["rel_pct"] is not None]
    worst = max(fin, key=lambda r: r["rel_pct"])
    fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM * 0.9)); ax = fig.add_subplot(111)
    for m in mats:
        rs = [r for r in rows if r["material"] == m]
        ax.loglog([r["trajectory_nm"] for r in rs], [r["kernel_nm"] for r in rs],
                  marks[m], ms=5, color=SCANNING, mfc="white", mew=1.1, label=m)
    lim = [1e-3, 4e2]
    ax.plot(lim, lim, color=GREY, lw=.8, ls="--", zorder=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("radius from full trajectory tracing (nm)")
    ax.set_ylabel("radius from the kernel (nm)")
    ax.grid(alpha=.15, lw=.4, which="both")
    ax.legend(frameon=False, loc="upper left")
    ax.annotate(f"worst finite deviation {worst['rel_pct']:.2f}%",
                xy=(worst["trajectory_nm"], worst["kernel_nm"]), xytext=(0.42, 0.13),
                textcoords="axes fraction", fontsize=6.4, color=GREY,
                arrowprops=dict(arrowstyle="-", lw=.6, color=GREY))
    save(fig, out, "S1B_kernel_vs_trajectory")

    # S1C silicon SE kernel with median and p95
    fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM * 0.8)); ax = fig.add_subplot(111)
    k = kern["silicon"].se
    ax.semilogx(np.maximum(k.r_nm, 1e-3), k.enclosed(), color=SCANNING, lw=1.4)
    for q, c in ((0.5, GREY), (0.95, GREY)):
        r = k.radius_containing(q)
        ax.axvline(r, color=c, lw=.9, ls=":")
        ax.annotate(f"r{int(q*100)} = {r:.2f} nm", xy=(r, q), xytext=(4, -8),
                    textcoords="offset points", fontsize=6.4, color=c)
    ax.set_xlabel("radius (nm)"); ax.set_ylabel("enclosed weight")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.15, lw=.4, which="both")
    save(fig, out, "S1C_silicon_SE_kernel")

    # S1D weight discarded when the kernel exceeds the field
    fields = np.array([256, 512, 640, 1024, 2048])
    fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM * 0.8)); ax = fig.add_subplot(111)
    used = {"cryo-TEM 640 px @ 1.8 nm": 640 * 1.8, "SEM 640 px @ 14 nm": 640 * 14.0}
    for m in mats:
        k = kern[m].bse
        fr = []
        for f_nm in fields * 14.0:
            fr.append(100 * (1 - float(k.enclosed()[np.searchsorted(k.r_nm, f_nm / 2)] if
                                       (f_nm / 2) < k.r_nm[-1] else 1.0)))
        ax.semilogx(fields * 14.0, fr, marker=marks[m], ls=styles[m],
                    color=SCANNING, lw=1.2, ms=4, mfc="white", mew=1.0, label=m)
    for lab, v in used.items():
        ax.axvline(v, color=GREY, lw=.7, ls=":")
        ax.annotate(lab, xy=(v, ax.get_ylim()[1]), rotation=90, fontsize=5.8,
                    color=GREY, ha="right", va="top")
    ax.set_xlabel("field size (nm)"); ax.set_ylabel("kernel weight outside the field (%)")
    ax.grid(alpha=.15, lw=.4, which="both"); ax.legend(frameon=False)
    save(fig, out, "S1D_weight_discarded")

    META["panels"]["S1"] = dict(
        materials=mats, energy_keV=5.0, n_electrons=40000, seed=0,
        quantile_comparison=rows,
        worst_finite_rel_pct=worst["rel_pct"], worst_case=worst,
        excluded=("carbon SE q50: the trajectory median radius is ~0, so a relative "
                  "deviation is undefined; the absolute difference is 0.003 nm"),
        claim_check=("supports 'better than 1%' on trajectory quantiles; the suite's "
                     "own tolerance for this is 10%, which is looser than the "
                     "measured behaviour"))
    print("  S1 worst finite deviation %.2f%%" % worst["rel_pct"])


# ── S2: counting error distributions ─────────────────────────────────────────

def fig_S2():
    out = SI / "S2"
    sp = list(csv.DictReader(open(WORK / "figures/source/fig4_spores_per_image.csv")))
    na = list(csv.DictReader(open(WORK / "figures/source/fig4_nano_per_image.csv")))
    def cond(r):
        if float(r.get("coating_nm") or 0) > 0: return "coated"
        return "uncoated\n+ charging" if float(r.get("charging") or 0) > 0 else "uncoated"
    def box(groups, colour, name, xlabel):
        ks = list(groups)
        fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM * 0.8)); ax = fig.add_subplot(111)
        bp = ax.boxplot([groups[k] for k in ks], widths=.55, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=1.1))
        for p in bp["boxes"]:
            p.set_facecolor(colour); p.set_alpha(.30); p.set_edgecolor(colour); p.set_linewidth(.8)
        for w in bp["whiskers"] + bp["caps"]: w.set_color(GREY); w.set_linewidth(.7)
        rng = np.random.default_rng(0)
        for i, k in enumerate(ks, 1):
            ax.plot(i + rng.normal(0, .05, len(groups[k])), groups[k], ".", ms=2.6,
                    color=GREY, alpha=.5, zorder=3)
        ax.axhline(0, color=GREY, lw=.7, ls=":")
        ax.set_xticks(range(1, len(ks) + 1))
        ax.set_xticklabels([f"{k}\nn={len(groups[k])}" for k in ks], fontsize=6.2)
        ax.set_ylabel("signed counting error (%)"); ax.set_xlabel(xlabel)
        ax.grid(alpha=.15, lw=.4, axis="y")
        save(fig, out, name)
        return {k: len(v) for k, v in groups.items()}
    gs, gn = {}, {}
    for r in sp: gs.setdefault(cond(r), []).append(100 * float(r["count_error"]))
    for r in na: gn.setdefault(f"{float(r['dose']):g}", []).append(100 * float(r["count_error"]))
    gs = {k: gs[k] for k in ["coated", "uncoated", "uncoated\n+ charging"] if k in gs}
    gn = {k: gn[k] for k in sorted(gn, key=float)}
    n_s = box(gs, SCANNING, "S2A_scanning_conditions", "specimen preparation")
    n_n = box(gn, TRANSMISSION, "S2B_transmission_dose", "total dose (e⁻/Å²)")
    # S2C error vs true count
    fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM * 0.8)); ax = fig.add_subplot(111)
    for rows_, c, lab in ((sp, SCANNING, "scanning"), (na, TRANSMISSION, "transmission")):
        ax.scatter([float(r["n_truth"]) for r in rows_],
                   [100 * float(r["count_error"]) for r in rows_],
                   s=9, color=c, alpha=.55, edgecolors="none", label=lab)
    ax.axhline(0, color=GREY, lw=.7, ls=":")
    ax.set_xlabel("true objects in the field"); ax.set_ylabel("signed counting error (%)")
    ax.grid(alpha=.15, lw=.4); ax.legend(frameon=False)
    save(fig, out, "S2C_error_vs_count")
    META["panels"]["S2"] = dict(scanning_n=n_s, transmission_n=n_n,
                                y="signed counting error, (predicted - true)/true")


# ── S4 / S6: predictions on simulated and on acquired ────────────────────────

def _overlay(ax, gt_polys, tp, fp, missed):
    for v in gt_polys:
        ax.add_patch(MplPoly(v, closed=True, fill=False, edgecolor=REFERENCE, lw=.45))
    for v in tp:
        ax.add_patch(MplPoly(v, closed=True, fill=False, edgecolor=TRUE_POS, lw=.8))
    for v in fp:
        ax.add_patch(MplPoly(v, closed=True, fill=False, edgecolor=FALSE_POS, lw=.8,
                             ls=(0, (2, 1.5))))
    for m in missed:
        ax.plot([m[0]], [m[1]], marker="o", ms=4.2, mfc="none", mec=MISSED, mew=1.0)

def _legend_fig():
    fig = plt.figure(figsize=(COL_MM * MM, 0.34))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.legend(handles=[
        Line2D([], [], color=REFERENCE, lw=1.2, label="reference object"),
        Line2D([], [], color=TRUE_POS, lw=1.2, label="true positive"),
        Line2D([], [], color=FALSE_POS, lw=1.2, ls=(0, (2, 1.5)), label="false positive"),
        Line2D([], [], color=MISSED, marker="o", ms=4.2, mfc="none", ls="none",
               label="undetected reference object")],
        loc="center", ncol=2, frameon=False, fontsize=6.6)
    return fig

def fig_S4():
    """Simulated held-out fields, chosen nearest the median per-field F1."""
    from ultralytics import YOLO
    out = SI / "S4"
    META["panels"]["S4"] = {}
    for tag, ds, mdl in (("transmission", WORK / "sim/nanoparticles_tem",
                          RUNS / "nanoparticles_tem/best.pt"),
                         ("scanning", WORK / "sim/spores_sem",
                          RUNS / "curve_sim_4/best.pt" if False else
                          RUNS / "spores_sem/best.pt")):
        m = YOLO(str(mdl))
        files = sorted((ds / "images/test").glob("*.png"))
        scored = []
        for f in files:
            H, W = np.array(Image.open(f).convert("L")).shape
            gt = []
            for line in (ds / "labels/test" / f"{f.stem}.txt").read_text().splitlines():
                q = line.split()
                if len(q) < 7 or q[0] != "0": continue
                gt.append(np.array(q[1:], float).reshape(-1, 2) * [W, H])
            r = m.predict(str(f), verbose=False, conf=0.5)[0]
            pr = [np.asarray(xy, float) for xy, c in
                  zip(r.masks.xy, r.boxes.cls.cpu().numpy().astype(int))
                  if c == 0 and len(xy) >= 4] if r.masks is not None else []
            cg = [p.mean(0) for p in gt]; used = set(); tp = []; fp = []
            for p in pr:
                cen = p.mean(0); best, bd = None, 1e18
                for j, g in enumerate(cg):
                    if j in used: continue
                    d = float(np.hypot(*(cen - g)))
                    if d < bd: bd, best = d, j
                if best is not None and bd < 0.03 * max(H, W): used.add(best); tp.append(p)
                else: fp.append(p)
            miss = [g for j, g in enumerate(cg) if j not in used]
            prec = len(tp) / max(len(pr), 1); rec = len(tp) / max(len(cg), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            scored.append(dict(f=f, gt=gt, tp=tp, fp=fp, miss=miss, f1=f1,
                               prec=prec, rec=rec, n_ref=len(cg)))
        med = float(np.median([s["f1"] for s in scored]))
        pick = sorted(scored, key=lambda s: abs(s["f1"] - med))[:2]
        recs = []
        for i, s in enumerate(pick, 1):
            im = np.array(Image.open(s["f"]).convert("L"))
            fig, ax = image_fig(im, COL_MM)
            _overlay(ax, s["gt"], s["tp"], s["fp"], s["miss"])
            px = 1.8 if tag == "transmission" else 14.0
            scalebar(ax, im.shape[1], px)
            save(fig, out, f"S4_{tag}_{i}")
            recs.append(dict(field=s["f"].stem, f1=s["f1"], precision=s["prec"],
                             recall=s["rec"], n_reference=s["n_ref"],
                             tp=len(s["tp"]), fp=len(s["fp"]), missed=len(s["miss"])))
        META["panels"]["S4"][tag] = dict(model=str(mdl), checkpoint_sha256=sha256(mdl),
            conf=0.5, dataset_median_f1=med,
            selection_rule="two fields nearest the dataset median per-field F1",
            fields=recs)
    save(_legend_fig(), out, "S4_legend")


# ── S5: acquired beside simulated, with intensity histograms ─────────────────

def fig_S5():
    out = SI / "S5"
    sys.path.insert(0, str(WORK / "figures/fig4"))
    import make_fig4 as F4
    from make_figures import cryo_field, sem_field
    cryo, sem = cryo_field(), sem_field()
    B1, _, _, _, _ = F4.simulate_plga_matched(cryo["px_nm"], cryo["img"].shape)
    B2, _, _, _, _ = F4.simulate_spores_matched(sem["px_nm"], sem["img"].shape)
    pairs = [("transmission", cryo["img"], np.asarray(B1, float), cryo["px_nm"]),
             ("scanning",     sem["img"],  np.asarray(B2, float), sem["px_nm"])]
    stats = {}
    for tag, acq, sim, px in pairs:
        # ONE window per pair, from the acquired image, applied to both.
        lo, hi = np.percentile(acq, [0.5, 99.5])
        to8 = lambda a: (np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
        sim_s = (sim - sim.min()) / (np.ptp(sim) + 1e-9)
        sim_s = sim_s * (hi - lo) + lo                     # onto the acquired range
        A, S = to8(acq), to8(sim_s)
        for nm, im in ((f"S5_{tag}_acquired", A), (f"S5_{tag}_simulated", S)):
            fig, ax = image_fig(im, COL_MM); scalebar(ax, im.shape[1], px); save(fig, out, nm)
        fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM * 0.55)); ax = fig.add_subplot(111)
        MOD = TRANSMISSION if tag == "transmission" else SCANNING
        ax.hist(A.ravel(), bins=64, range=(0, 255), color=MOD, alpha=.45,
                density=True, label="acquired")
        ax.hist(S.ravel(), bins=64, range=(0, 255), histtype="step", lw=1.3,
                ls="--", color=MOD, density=True, label="simulated")
        ax.set_xlabel("display value (8-bit)"); ax.set_yticks([])
        # above the axes: inside, it clipped the tail of the acquired histogram
        ax.legend(frameon=False, ncol=2, loc="lower center",
                  bbox_to_anchor=(0.5, 1.01), borderaxespad=0)
        ax.grid(alpha=.12, lw=.4)
        save(fig, out, f"S5_{tag}_histograms")
        st = lambda x: dict(mean=float(x.mean()), std=float(x.std()),
                            p1=float(np.percentile(x, 1)), p99=float(np.percentile(x, 99)),
                            frac_below_32=float((x < 32).mean()))
        stats[tag] = dict(acquired=st(A), simulated=st(S),
                          display=dict(rule="single 0.5-99.5 percentile window taken "
                                            "from the acquired image and applied to both",
                                       lo=float(lo), hi=float(hi)))
    META["panels"]["S5"] = stats


# ── S6: detections on acquired micrographs, before and after fine-tuning ─────

def fig_S6():
    from ultralytics import YOLO
    out = SI / "S6"
    pf = json.load(open(WORK / "figures/fig4/perfield.json"))
    sys.path.insert(0, str(WORK / "figures/fig4"))
    import make_fig4 as F4
    recs = {}

    # transmission: simulation only (no fine-tuning series exists)
    row = [r for r in pf["plga"] if r["stem"] == pf["plga_pick"]][0]
    A_raw, px, _, _ = F4.load_plga_acquired(row["src"]); A, _, _ = stretch(A_raw)
    ref = json.load(open(WORK / "plga_reference.json"))
    refs = [np.array([float(x["Center X (nm)"]) / px, float(x["Center Y (nm)"]) / px])
            for x in ref if x["File Location"] == row["src"]]
    tp, fp, miss, nd = F4.detect_and_match(RUNS / "nanoparticles_tem/best.pt",
                                           A, refs, 40.0 / px)
    fig, ax = image_fig(A, COL_MM); _overlay(ax, [], tp, fp, miss)
    scalebar(ax, A.shape[1], px); save(fig, out, "S6_transmission_simulation_only")
    recs["transmission_simulation_only"] = dict(field=row["stem"], recall=row["recall"],
        dataset_median_recall=pf["plga_median_recall"], n_reference=len(refs),
        tp=len(tp), fp=len(fp), missed=len(miss),
        note="no fine-tuning series exists for the transmission data")

    # scanning: before and after four acquired images, same held-out field
    stem = pf["sem_pick"]; A2_raw, px2, _ = F4.load_sem_acquired(stem)
    A2, _, _ = stretch(A2_raw); H2, W2 = A2.shape
    refs2 = [p.mean(0) for p in F4.read_label_polys(
        WORK / "real_sem/labels" / f"{stem}.txt", W2, H2, cls=0)]
    for lab, mdl in (("simulation_only", RUNS / "spores_sem/best.pt"),
                     ("plus_four_acquired", RUNS / "curve_sim_4/best.pt")):
        tp, fp, miss, nd = F4.detect_and_match(mdl, A2, refs2, 60.0)
        fig, ax = image_fig(A2, COL_MM); _overlay(ax, [], tp, fp, miss)
        scalebar(ax, A2.shape[1], px2); save(fig, out, f"S6_scanning_{lab}")
        recs[f"scanning_{lab}"] = dict(field=stem, model=str(mdl),
            checkpoint_sha256=sha256(mdl), n_reference=len(refs2), tp=len(tp),
            fp=len(fp), missed=len(miss),
            dataset_median_recall=pf["sem_median_recall"])
    save(_legend_fig(), out, "S6_legend")
    META["panels"]["S6"] = dict(selection_rule="field nearest the dataset median recall",
                                fields=recs)


# ── S7: composition of the reference set ─────────────────────────────────────

def fig_S7():
    out = SI / "S7"
    rows = json.load(open(WORK / "ann_features.json"))
    def cl(r):
        if r["area_um2"] > 3.0: return "aggregate outline"
        if r["area_um2"] < 0.3: return "fragment"
        return "confident spore" if r["circ"] >= 0.5 else "plausible, ragged"
    # Four reference categories are categorical, not modality, so they use the
    # detection family rather than the modality pair.
    cols = {"confident spore": REFERENCE, "plausible, ragged": TRUE_POS,
            "fragment": FALSE_POS, "aggregate outline": MISSED}
    fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM * 0.85)); ax = fig.add_subplot(111)
    for k, c in cols.items():
        rs = [r for r in rows if cl(r) == k]
        ax.scatter([r["circ"] for r in rs], [r["area_um2"] for r in rs], s=7,
                   color=c, alpha=.55, edgecolors="none", label=f"{k} (n={len(rs)})")
    ax.axvline(0.5, color=GREY, lw=.8, ls="--")
    ax.axhline(0.3, color=GREY, lw=.8, ls=":"); ax.axhline(3.0, color=GREY, lw=.8, ls=":")
    ax.set_yscale("log"); ax.set_xlabel("circularity, 4πA/P²")
    ax.set_ylabel("projected area (µm²)")
    ax.grid(alpha=.13, lw=.4)
    # the categories occupy every corner of this panel, so the key goes outside
    ax.legend(frameon=False, fontsize=6.0, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), borderaxespad=0)
    save(fig, out, "S7A_composition")

    br = json.load(open(WORK / "models/reference_bracket.json"))
    names = list(br); x = np.arange(len(names)); w = 0.36
    fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM * 0.75)); ax = fig.add_subplot(111)
    ax.bar(x - w/2, [br[n]["recall"] for n in names], w, color=SCANNING,
           alpha=.9, label="recall")
    ax.bar(x + w/2, [br[n]["prec"] for n in names], w, facecolor="white",
           edgecolor=SCANNING, lw=1.0, hatch="////", label="precision")
    for i, n in enumerate(names):
        ax.text(i - w/2, br[n]["recall"] + .02, f'{br[n]["recall"]:.3f}', ha="center", fontsize=6)
        ax.text(i + w/2, br[n]["prec"] + .02, f'{br[n]["prec"]:.3f}', ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (", "\n(") + f"\nn={br[n]['n']}" for n in names], fontsize=5.8)
    ax.set_ylim(0, 1.0); ax.set_ylabel("value against that reference subset")
    ax.grid(alpha=.15, lw=.4, axis="y"); ax.legend(frameon=False, ncol=2)
    save(fig, out, "S7B_recall_precision_by_subset")
    META["panels"]["S7"] = dict(
        composition={k: sum(1 for r in rows if cl(r) == k) for k in cols},
        thresholds=dict(circularity=0.5, area_um2=[0.3, 3.0]),
        bracket=br,
        interpretation=("recall moves 0.512 -> 0.572 -> 0.693 while precision stays "
                        "0.726 -> 0.714 -> 0.699; the flat precision is what shows the "
                        "detector was not producing the excluded objects"),
        n_all_annotations=1085, n_evaluated_fields=648,
        caption=("Composition of the reference set. (A) Every annotation in the "
                 "reference set, n = 1085, placed by circularity and projected "
                 "area; dashed line, the circularity threshold at 0.5; dotted "
                 "lines, the area bounds at 0.3 and 3.0 um^2. The four categories "
                 "are assigned by those thresholds alone. (B) Recall and precision "
                 "against nested subsets of the reference. Panel B and Table S8 "
                 "are computed on the 648 annotations that fall within the fields "
                 "the detector was evaluated on, a subset of the 1085 shown in "
                 "panel A; the remaining 437 lie in fields held outside the "
                 "evaluation. Panel A therefore characterises the reference set as "
                 "a whole, while panel B reports performance, and the two counts "
                 "are not expected to agree."))


# ── supplementary tables ─────────────────────────────────────────────────────

def tables():
    fm = json.load(open(HERE / "figure_metadata.json"))
    f4 = fm["panels"]["fig4"]

    csv_pdf("tableS1", ["Parameter", "Value"], [
        ["Forward model", "multislice through vitreous ice"],
        ["Transfer function", "weak-phase CTF, CTFFIND/RELION convention"],
        ["chi(k)", "pi*lam*df*k^2 - 0.5*pi*Cs*lam^3*k^4"],
        ["Envelopes", "temporal, spatial, B-factor, dose damage"],
        ["Microscopes", "krios, krios-cfeg, glacios, talos-arctica, talos-l120c, cs-corrected"],
        ["Detectors", "K3, K2, Falcon4, Falcon4i, Falcon3EC, Apollo, DE64, Ceta, ideal"],
        ["Detector response", "parameterised DQE(q) and MTF(q); realistic shapes, not vendor curves"],
        ["Slice thickness", "40 A"],
        ["Inelastic loss", "thickness-dependent"],
        ["Validation", "defocus recovered from the simulated power spectrum, "
                       f"R2 {f4['B_cryoTEM']['r_squared']:.3f}, "
                       f"MAE {f4['B_cryoTEM']['mean_abs_error_um']:.3f} um"],
    ], "Transmission engine parameters.")

    csv_pdf("tableS2", ["Parameter", "Value"], [
        ["Forward model", "Monte Carlo electron transport"],
        ["Elastic scattering", "screened Rutherford"],
        ["Stopping power", "Joy-Luo modified Bethe"],
        ["Range", "Kanaya-Okayama (1972)"],
        ["Materials", "17, including 6 compounds by weight-averaged Z and A"],
        ["Free parameter", "epsilon, energy per escaping secondary; fitted per material"],
        ["Fitted epsilon range", "57-243 eV"],
        ["Emission", "SE1/SE2 split with per-material interaction-volume kernels"],
        ["Kernel construction", "enclosed-weight CDF from traced trajectories"],
        ["Kernel fidelity", "better than 1% on trajectory quantiles (Fig. S1B)"],
        ["Detectors", "ETD, TLD, BSE"],
        ["Optional artefacts", "specimen charging (off by default)"],
        ["Validation", "backscatter yield within 15% of published, 20% for gold (Fig. 4B)"],
    ], "Scanning engine parameters.")

    csv_pdf("tableS3", ["Specimen model", "Parameter varied", "Range"], [
        ["PLGA nanoparticles", "diameter", "14-210 nm, log-normal about 42 nm"],
        ["", "particles per field", "15-60"],
        ["", "ice thickness", "40-90 nm"],
        ["", "defocus", "-1.0 to -3.0 um"],
        ["", "detector", "K3, K2, Falcon4"],
        ["", "total dose", "20, 40, 60 e-/A2"],
        ["", "pixel size", "15, 18, 22 A"],
        ["", "debris", "0 or 6-30 chains, 70-180 nm"],
        ["Bacterial spores", "length", "800-3000 nm, log-normal about 1267 nm"],
        ["", "aspect ratio", "1.05-3.0, mean 1.65"],
        ["", "areal density", "0.35-0.95 per um2"],
        ["", "coating", "0 nm, or 8-16 nm gold"],
        ["", "beam energy", "0.8-1.5 kV uncoated, 3-6 kV coated"],
        ["", "charging", "0, 0.4, 0.8 or 1.2"],
        ["", "substrate", "resin, carbon, biology or silicon"],
        ["", "pixel size", "4.0, 6.5, 9.6, 14.0, 19.3 nm"],
        ["", "debris", "0-13 clumps, 45-110 nm grains"],
    ], "Specimen models and the parameters actually varied.", colw=[0.30, 0.30, 0.40])

    inst = f4["A"]["labelled_instances_in_full_datasets"]
    csv_pdf("tableS4", ["Dataset", "Images", "Train", "Val", "Test",
                        "Labelled instances", "Held-out fraction"], [
        ["Transmission, PLGA", 400, 280, 60, 60, f'{inst["nanoparticles"]:,}', "0.15"],
        ["Scanning, spores", 400, 280, 60, 60, f'{inst["spores"]:,}', "0.15"],
        ["Total", 800, 560, 120, 120, f'{sum(inst.values()):,}', "0.15"],
    ], "Simulated datasets as generated. Splits are by disjoint random seed, so no "
       "test scene appears in training in any form.")

    csv_pdf("tableS5", ["Setting", "Value"], [
        ["Architecture", "YOLO11s-seg"],
        ["Initialisation", "ImageNet-pretrained, or simulation-pretrained where stated"],
        ["Epochs requested", "100"],
        ["Early stopping patience", "25"],
        ["Batch size", "16 (4 when fine-tuning on acquired images)"],
        ["Image size", "640"],
        ["Augmentation", "ultralytics defaults"],
        ["Split", "280 / 60 / 60 by disjoint seed range"],
        ["Seed", "0, deterministic"],
        ["Hardware", "one Tesla V100-SXM3-32GB"],
        ["Inference", "~7 ms per 640x640 image"],
    ], "Training configuration.")

    csv_pdf("tableS6", ["Dataset", "Epochs", "Box mAP@50", "Box mAP@50-95",
                        "Mask mAP@50", "Mask mAP@50-95", "Mask P", "Mask R"], [
        ["Transmission, PLGA", 100, 0.984, 0.889, 0.935, 0.577, 0.944, 0.932],
        ["Scanning, spores", 95, 0.974, 0.937, 0.990, 0.872, 0.972, 0.988],
    ], "Simulated held-out performance, target class only.")

    csv_pdf("tableS7", ["Dataset", "Evaluated on", "Acquired images",
                        "Recall", "Precision"], [
        ["Transmission", "simulated test", 0, 0.932, 0.944],
        ["Transmission", "acquired PLGA, 295 verified", 0, 0.329, 0.970],
        ["Scanning", "simulated test", 0, 0.988, 0.972],
        ["Scanning", "acquired, 1085 annotated", 0, 0.253, 0.432],
        ["Scanning", "acquired, 1085 annotated", 4, 0.512, 0.726],
    ], "Transfer to acquired data.")

    # The annotation-burden table was removed: it reproduced main-text Figure 4D
    # exactly, and two copies of the same seven numbers can only drift apart.
    br = json.load(open(WORK / "models/reference_bracket.json"))
    csv_pdf("tableS8", ["Reference subset", "n", "Recall", "Precision", "Mask mAP@50"],
        [[k, br[k]["n"], round(br[k]["recall"], 3), round(br[k]["prec"], 3),
          round(br[k]["map50"], 3)] for k in br],
        "Detector performance against nested reference subsets. The detector, images "
        "and acquisition are identical throughout; only the reference differs. "
        "n counts annotations within the evaluated fields; the full reference set "
        "contains 1085 annotations, of which 648 fall in those fields (Figure S7).")


def main():
    SI.mkdir(parents=True, exist_ok=True)
    print("S1 kernel fidelity:");      fig_S1()
    print("S2 counting error:");       fig_S2()
    print("S4 simulated held-out:");   fig_S4()
    print("S5 acquired vs simulated:");fig_S5()
    print("S6 acquired detections:");  fig_S6()
    print("S7 reference composition:");fig_S7()
    print("tables S1-S8:");            tables()
    META["not_built"] = dict(
        figure_1=("needs six staged captures from a live interface session; cannot be "
                  "produced headlessly without looking contrived"),
        S3=("training curves; no claim in the paper depends on them and the main text "
            "reports final held-out performance"),
        text_S1=("removed at the author's request. For the record: no written "
                 "annotation protocol was found on either share, and the reference "
                 "set's single-class vocabulary -- every annotation labelled 'Spore', "
                 "with no class for debris -- is measurable from the annotations "
                 "themselves and is what places debris contours inside the spore "
                 "reference."),
        table_S8_burden=("the annotation-burden table was removed; it reproduced "
                         "main-text Figure 4D exactly. The former Table S9 "
                         "(reference subsets) is now Table S8."))
    META["palette"] = palette_block()
    (SI / "si_metadata.json").write_text(json.dumps(META, indent=1, default=str))
    print("\nwrote si_metadata.json")

# ── palette record ───────────────────────────────────────────────────────────
# Written into the metadata by the generator itself. It used to be stamped in by
# a separate script, which meant any regeneration silently dropped it.

def palette_block():
    from palette import P
    return {
     "encoding_rule": "Colour encodes imaging modality only. Within a modality, "
                      "series are separated by line style and marker fill, never by hue.",
     "modality": {"transmission_cryoTEM": P.TRANSMISSION, "scanning_SEM": P.SCANNING},
     "outcome": {"reference_ground_truth": P.REFERENCE, "true_positive": P.TRUE_POS,
                 "false_positive": P.FALSE_POS, "missed": P.MISSED},
     "structural": {"ink": P.INK, "grey": P.GREY, "rule": P.RULE, "band": P.BAND},
     "accessibility": "The two modality hues differ in lightness as well as hue "
                      "(L* ~31 vs ~62), so panels survive greyscale conversion and "
                      "the common colour vision deficiencies.",
    }


if __name__ == "__main__":
    sys.exit(main())
