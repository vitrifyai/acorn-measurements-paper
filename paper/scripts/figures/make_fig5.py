#!/usr/bin/env python3
"""Figure 5: movie handling — motion correction, its validation, and dose
fractionation. Individual panels, 600 dpi PNG plus vector PDF, no letters.

Built on a SIMULATED movie because the trajectory is then known exactly, which
turns a demonstration into a measurement. See README for why the acquired
movies in this project cannot serve: their consecutive frames correlate at
0.002, so there is no frame-to-frame signal to align on.

    python make_fig5.py
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.ndimage import shift as ndshift
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COL_MM, FULL_MM, MM, TRANSMISSION, SCANNING, REFERENCE,
                    TRUE_POS, FALSE_POS, MISSED, GREY, RULE, stretch, save,
                    image_fig, scalebar, git_commit)

HERE = Path(__file__).resolve().parent
OUT = HERE / "fig5"
PX_A = 18.0            # 1.8 nm/px, the scale the acquired PLGA is analysed at
N_FRAMES = 24
E_PER_PX_FRAME = 300.0
TOTAL_DRIFT_PX = 26.0   # ~470 nm of early beam-induced motion; comparable to
                        # the particle diameter, which is the regime that blurs

def true_trajectory(n, total_px=TOTAL_DRIFT_PX, seed=0):
    """Beam-induced motion: fast at first, then settling, with small jitter."""
    r = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    mag = total_px * (1 - np.exp(-4.5 * t))
    ang = np.deg2rad(35.0)
    x = mag * np.cos(ang) + np.cumsum(r.normal(0, 0.12, n))
    y = mag * np.sin(ang) + np.cumsum(r.normal(0, 0.12, n))
    return np.stack([y - y[0], x - x[0]], 1)

def make_movie(e_per_px_frame=E_PER_PX_FRAME, n=N_FRAMES, seed=1,
               scene_seed=4, traj_seed=0):
    from acorn_tem_sim import engine as E
    from acorn_tem_sim.engine.scene import Scene, Nanoparticles, simulate_scene
    from acorn_tem_sim.engine.detector import to_display
    cfg = E.resolve(answers={"microscope": "krios", "detector_model": "K3",
        "pixel_size_a": PX_A, "image_size_px": 1024, "voltage_kv": "300",
        "total_dose_e_per_a2": 45.0, "defocus_min_um": -2.0,
        "defocus_max_um": -2.0, "energy_filter_ev": 0.0})
    # Thinner ice and a denser, more polydisperse population: closer to the
    # acquired PLGA fields, and higher particle contrast.
    sc = Scene(ice_thickness_nm=62.0, seed=scene_seed, solvent_noise=9.0,
               components=[Nanoparticles(n=110, diameter_nm_mean=42.0, diameter_nm_sd=15.0)])
    ideal, _ = simulate_scene(cfg, sc, defocus_um=-2.0, dz_a=40.0)
    base = np.asarray(to_display(ideal), float)
    base = (base - base.min()) / (np.ptp(base) + 1e-9)
    traj = true_trajectory(n, seed=traj_seed)
    rng = np.random.default_rng(seed)
    frames = np.stack([
        rng.poisson(np.clip(ndshift(base, s, order=1, mode="nearest"), 0, None)
                    * e_per_px_frame) / e_per_px_frame
        for s in traj]).astype(np.float32)
    return frames, traj, cfg

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from acorn.core.frame_processor import motion_correct_frames, mean_average
    lap = lambda a: float(np.var(np.gradient(a.astype(np.float32))[0]))

    frames, traj, cfg = make_movie()
    uncorr = mean_average(frames)
    corr, shifts = motion_correct_frames(frames)
    px_nm = PX_A / 10.0

    # ---- 5A / 5B: the same movie averaged without and with correction ----
    # One intensity scaling for both, from the uncorrected average, so the
    # comparison is of sharpness and not of contrast handling.
    lo, hi = np.percentile(uncorr, [0.5, 99.5])
    to8 = lambda a: (np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    for nm, arr in (("5A_uncorrected", uncorr), ("5B_motion_corrected", corr)):
        fig, ax = image_fig(to8(arr), COL_MM)
        scalebar(ax, arr.shape[1], px_nm)
        save(fig, OUT, nm)

    # ---- 5C: recovered trajectory against the known one ----
    rec = -(shifts - shifts[0]); tru = traj - traj[0]
    err = np.hypot(*(rec - tru).T)
    fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM * 0.92)); ax = fig.add_subplot(111)
    ax.plot(tru[:, 1], tru[:, 0], "-", color=GREY, lw=1.6, label="applied", zorder=2)
    ax.plot(rec[:, 1], rec[:, 0], "o-", color=TRANSMISSION, lw=1.0, ms=3.4,
            label="recovered", zorder=3)
    ax.plot([tru[0, 1]], [tru[0, 0]], "s", color=TRANSMISSION, mfc="white", mew=1.2, ms=6, zorder=4)
    ax.plot([tru[-1, 1]], [tru[-1, 0]], "^", color=TRANSMISSION, ms=7, zorder=4)
    ax.set_xlabel("x drift (pixels)"); ax.set_ylabel("y drift (pixels)")
    ax.set_aspect("equal"); ax.grid(alpha=0.15, lw=0.4)
    ax.legend(frameon=False, loc="lower right")
    save(fig, OUT, "5C_trajectory")

    # ---- 5D: recovery accuracy against per-frame dose ----
    # Down to the point where alignment fails. At 1.8 nm/px each pixel
    # integrates a lot of signal, so recovery stays sub-pixel far below any
    # realistic fractionation; the breakdown is what connects this panel to the
    # acquired movies, whose per-frame signal is below the noise entirely.
    doses = [100.0, 20.0, 4.0, 1.0, 0.25, 0.06, 0.015]
    rmse, gain = [], []
    for e in doses:
        f2, t2, _ = make_movie(e_per_px_frame=e)
        c2, s2 = motion_correct_frames(f2)
        r2 = -(s2 - s2[0]); tt = t2 - t2[0]
        rmse.append(float(np.sqrt((np.hypot(*(r2 - tt).T) ** 2).mean())))
        m2 = mean_average(f2)
        gain.append(100 * (lap(c2) / max(lap(m2), 1e-12) - 1))
    fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM * 0.78))
    ax = fig.add_subplot(111)
    # Both axes log: recovery is sub-pixel over three decades of dose and then
    # fails by four orders of magnitude within one. The shaded region is where
    # the returned shifts exceed the image itself and are meaningless.
    ax.loglog(doses, rmse, "o-", color=TRANSMISSION, lw=1.4, ms=5, zorder=3)
    ax.axhspan(30, 1e4, color=GREY, alpha=0.13, lw=0)
    ax.axhline(1.0, color=GREY, lw=0.7, ls=":")
    ax.set_xticks(doses); ax.set_xticklabels([f"{d:g}" for d in doses], fontsize=6.2)
    ax.minorticks_off()
    ax.set_xlabel("electrons per pixel per frame")
    ax.set_ylabel("drift recovery error, RMSE (pixels)")
    ax.grid(alpha=0.15, lw=0.4, which="major")
    ax.set_ylim(0.03, 3e3); ax.invert_xaxis()
    save(fig, OUT, "5D_dose_dependence")

    meta = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=git_commit(),
        why_simulated=("Acquired movies in this project cannot serve. Their "
                       "consecutive frames correlate at 0.0017 -- the per-frame "
                       "signal is below the shot noise -- so there is no "
                       "frame-to-frame signal to align on, and frame-to-frame "
                       "phase correlation returns spurious shifts of ~1000 px on "
                       "a 2048 px frame. That is expected for such data, not a "
                       "defect; production pipelines align to a running average "
                       "instead. A simulated movie also supplies the true "
                       "trajectory, which makes 5C a measurement rather than a "
                       "demonstration."),
        movie=dict(frames=N_FRAMES, electrons_per_px_per_frame=E_PER_PX_FRAME,
                   pixel_size_A=PX_A, image_size_px=1024, defocus_um=-2.0,
                   microscope="krios", detector="K3", scene_seed=4, noise_seed=1,
                   applied_drift_px=float(TOTAL_DRIFT_PX),
                   trajectory_model="fast initial motion settling exponentially, "
                                    "plus a small random walk"),
        A_B=dict(display="single 0.5-99.5 percentile window taken from the "
                         "uncorrected average and applied to both, so the panels "
                         "differ in sharpness only",
                 sharpness_var_of_gradient=dict(uncorrected=lap(uncorr),
                                                corrected=lap(corr),
                                                change_pct=100*(lap(corr)/max(lap(uncorr),1e-12)-1))),
        C=dict(applied_px=tru.tolist(), recovered_px=rec.tolist(),
               rmse_px=float(np.sqrt((err ** 2).mean())), max_error_px=float(err.max())),
        D=dict(doses_e_per_px_frame=doses, rmse_px=rmse, sharpness_gain_pct=gain),
        )
    (OUT / "fig5_metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    print("\n5C  drift RMSE %.2f px, max %.2f px" % (meta["C"]["rmse_px"], meta["C"]["max_error_px"]))
    print("5A/B sharpness %+.0f%%" % meta["A_B"]["sharpness_var_of_gradient"]["change_pct"])
    print("5D  RMSE by dose:", ["%.2f" % r for r in rmse])

if __name__ == "__main__":
    sys.exit(main())
