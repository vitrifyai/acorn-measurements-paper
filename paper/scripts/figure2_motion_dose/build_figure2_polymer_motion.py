#!/usr/bin/env python3
"""Build revised ACORN Figure 2 from a deterministic synthetic movie.

The specimen is a generic nanoparticle-loaded, phase-separated polymer thin
film. Its dose response is phenomenological and is not parameterized to a
specific chemistry. Ground truth separates:

1. coherent whole-field stage drift,
2. local polymer-interface deformation, and
3. nanoparticle motion relative to the interface.

The script uses ACORN's production rigid alignment and annotation tracker.
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


import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, shift as nd_shift
from skimage.feature import peak_local_max

ACORN = ACORN_SOURCE_DIR
sys.path.insert(0, str(ACORN / "src"))
from acorn.analysis.tracking import track_annotations
from acorn.core.frame_processor import align_frames, dose_series


OUT = Path(__file__).resolve().parent
SEED = 20260923
DPI = 600
WIDTH_IN = 7.0
HEIGHT_IN = 8.6
H = W = 512
N_FRAMES = 100
DOSE_PER_FRAME = 10.0
TOTAL_DOSE = N_FRAMES * DOSE_PER_FRAME
PIXEL_SIZE_NM = 0.5

SLATE = "#2F4858"
OCHRE = "#C08A2E"
TEAL = "#1B9E77"
MAGENTA = "#E7298A"
GREY = "#5B6670"
LIGHT = "#D6DADE"
INK = "#1B1F24"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


@dataclass
class Circle:
    cx: float
    cy: float
    area_nm2: float
    type: str = "circle"


class Store(list):
    pass


def centerline(x: np.ndarray, fraction: float) -> np.ndarray:
    base = 257.0 + 31.0 * np.sin(2 * np.pi * x / 205.0 + 0.35)
    local_bend = 20.0 * fraction * np.exp(-((x - 325.0) / 105.0) ** 2)
    ripple = 5.5 * fraction * np.sin(2 * np.pi * x / 92.0 + 2.2 * fraction)
    return base + local_bend + ripple


def particle_positions(fraction: float) -> np.ndarray:
    x0 = np.array([84.0, 154.0, 232.0, 318.0, 402.0, 463.0])
    offsets = np.array([-28.0, 19.0, -17.0, 25.0, -23.0, 14.0])
    # Some particles follow the polymer, while others diffuse relative to it.
    dx = np.array([2.0, 12.0, -8.0, 17.0, -14.0, 7.0]) * fraction
    rel_y = np.array([1.0, -13.0, 18.0, -7.0, 15.0, -19.0]) * fraction
    x = x0 + dx
    y = centerline(x, fraction) + offsets + rel_y
    return np.column_stack([x, y])


def noiseless_specimen(
    fraction: float,
    texture: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:H, :W]
    line = centerline(np.arange(W), fraction)
    distance = yy - line[np.newaxis, :]
    damage = np.exp(-0.72 * fraction)
    width = 4.5 + 3.5 * fraction
    interface = -0.17 * damage * np.exp(-(distance ** 2) / (2 * width ** 2))
    phase = 0.035 * damage * np.tanh(distance / 11.0)
    image = 1.0 + 0.055 * texture + interface + phase

    particles = particle_positions(fraction)
    for j, (px, py) in enumerate(particles):
        sigma = 3.2 + 0.35 * (j % 3)
        image += (0.34 - 0.045 * fraction) * np.exp(
            -((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2)
        )
    return image.astype(np.float32), line.astype(np.float32), particles.astype(np.float32)


def build_movie():
    rng = np.random.default_rng(SEED)
    texture = gaussian_filter(rng.normal(size=(H, W)), 2.2)
    texture += 0.55 * gaussian_filter(rng.normal(size=(H, W)), 10.0)
    texture /= np.std(texture)

    steps = rng.normal(0.0, 0.28, size=(N_FRAMES, 2))
    drift = np.cumsum(steps, axis=0)
    drift[:, 0] += np.linspace(0, 6.5, N_FRAMES)
    drift[:, 1] += 2.8 * np.sin(np.linspace(0, 2.2 * np.pi, N_FRAMES))
    drift -= drift[0]

    frames = np.empty((N_FRAMES, H, W), dtype=np.float32)
    lines = np.empty((N_FRAMES, W), dtype=np.float32)
    particles = np.empty((N_FRAMES, 6, 2), dtype=np.float32)
    clean = []
    count_scale = 58.0

    for i in range(N_FRAMES):
        fraction = i / (N_FRAMES - 1)
        specimen, line, pts = noiseless_specimen(fraction, texture)
        moved = nd_shift(specimen, drift[i], order=1, mode="reflect")
        counts = rng.poisson(np.clip(moved, 0.05, None) * count_scale)
        frames[i] = counts.astype(np.float32) / count_scale
        lines[i] = line
        particles[i] = pts
        if i in (0, N_FRAMES - 1):
            clean.append(specimen)

    return frames, drift.astype(np.float32), lines, particles, clean


def center_trajectory(path: np.ndarray) -> np.ndarray:
    return path - path[0]


def detect_interface(image: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray:
    smooth = gaussian_filter(image, 2.0)
    estimate = np.empty(W, dtype=float)
    if previous is None:
        previous = np.full(W, H / 2.0)
    for x in range(W):
        yc = int(round(previous[x]))
        lo = max(12, yc - 48)
        hi = min(H - 12, yc + 49)
        estimate[x] = lo + int(np.argmin(smooth[lo:hi, x]))
    return gaussian_filter(estimate, 6.0)


def detect_particles(image: np.ndarray) -> np.ndarray:
    smooth = gaussian_filter(image, 1.2)
    highpass = smooth - gaussian_filter(smooth, 9.0)
    peaks = peak_local_max(
        highpass,
        min_distance=24,
        threshold_abs=0.075,
        num_peaks=6,
        exclude_border=16,
    )
    if len(peaks) == 0:
        return np.empty((0, 2), dtype=float)
    # peak_local_max returns (row, col); tracking expects (x, y).
    strengths = highpass[peaks[:, 0], peaks[:, 1]]
    order = np.argsort(strengths)[::-1][:6]
    return peaks[order][:, ::-1].astype(float)


def nearest_detection(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.array([np.nan, np.nan])
    return points[np.argmin(np.linalg.norm(points - target, axis=1))]


def membrane_cnr(image: np.ndarray, line: np.ndarray) -> float:
    yy, _ = np.mgrid[:H, :W]
    distance = yy - line[np.newaxis, :]
    band = np.abs(distance) <= 5
    background = (np.abs(distance) >= 28) & (np.abs(distance) <= 55)
    return float(abs(image[band].mean() - image[background].mean()) /
                 max(image[background].std(), 1e-9))


def build_measurements(aligned, true_lines, true_particles):
    bins, ranges = dose_series(aligned, 20)  # 50 e-/A^2 per analysis bin
    doses = []
    true_mem = []
    measured_mem = []
    true_rel = []
    measured_rel = []
    contours = []
    stores = []
    previous = None
    p_index = 3

    for avg, (s, e) in zip(bins, ranges):
        dose_mid = ((s + e) / 2.0) * DOSE_PER_FRAME
        truth_line = true_lines[s:e].mean(axis=0)
        contour = detect_interface(avg, previous)
        previous = contour
        detections = detect_particles(avg)
        stores.append(Store([
            Circle(float(x), float(y), np.pi * (3.5 * PIXEL_SIZE_NM) ** 2)
            for x, y in detections
        ]))

        truth_p = true_particles[s:e, p_index].mean(axis=0)
        detected_p = nearest_detection(detections, truth_p)
        truth_local_line = np.interp(truth_p[0], np.arange(W), truth_line)
        detected_local_line = (
            np.interp(detected_p[0], np.arange(W), contour)
            if np.all(np.isfinite(detected_p)) else np.nan
        )

        doses.append(dose_mid)
        true_mem.append(float(np.mean(truth_line[72:440])))
        measured_mem.append(float(np.mean(contour[72:440])))
        true_rel.append(float(truth_p[1] - truth_local_line))
        measured_rel.append(float(detected_p[1] - detected_local_line))
        contours.append(contour)

    # Exercise ACORN's production annotation linker on the detected centroids.
    tracks = track_annotations(
        stores,
        pixel_size_nm=PIXEL_SIZE_NM,
        max_displacement_nm=18.0,
        min_frames=3,
        max_gap=1,
    )

    arrays = [np.asarray(v, dtype=float) for v in
              (doses, true_mem, measured_mem, true_rel, measured_rel)]
    arrays[1] = (arrays[1] - arrays[1][0]) * PIXEL_SIZE_NM
    arrays[2] = (arrays[2] - arrays[2][0]) * PIXEL_SIZE_NM
    arrays[3] = (arrays[3] - arrays[3][0]) * PIXEL_SIZE_NM
    arrays[4] = (arrays[4] - arrays[4][0]) * PIXEL_SIZE_NM
    return (*arrays, bins, ranges, contours, tracks)


def bin_width_comparison(aligned, true_lines, true_particles):
    center = N_FRAMES // 2
    widths_dose = [20, 100, 250]
    rows = []
    images = []
    p_index = 3
    for width_dose in widths_dose:
        n = max(1, int(round(width_dose / DOSE_PER_FRAME)))
        s = max(0, center - n // 2)
        e = min(N_FRAMES, s + n)
        s = e - n
        avg = aligned[s:e].mean(axis=0)
        truth_line = true_lines[s:e].mean(axis=0)
        contour = detect_interface(avg)
        detections = detect_particles(avg)
        truth_p = true_particles[s:e, p_index].mean(axis=0)
        detected_p = nearest_detection(detections, truth_p)
        loc_error = (
            float(np.linalg.norm(detected_p - truth_p) * PIXEL_SIZE_NM)
            if np.all(np.isfinite(detected_p)) else np.nan
        )
        rows.append({
            "bin_width_e_per_A2": width_dose,
            "start_dose_e_per_A2": s * DOSE_PER_FRAME,
            "end_dose_e_per_A2": e * DOSE_PER_FRAME,
            "membrane_cnr": membrane_cnr(avg, truth_line),
            "particle_localization_error_nm": loc_error,
        })
        images.append((avg, truth_line, truth_p, detected_p))
    return rows, images


def add_panel_label(ax, letter):
    ax.text(
        0.015, 0.985, letter,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9, fontweight="bold", color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
        zorder=20,
    )


def image_limits(images):
    values = np.concatenate([im.ravel() for im in images])
    return tuple(np.percentile(values, [0.5, 99.5]))


def draw_panel_a(ax, initial_image, true_lines, true_particles, truth_drift):
    """Draw programmed motion components without conflating local motions."""
    ax.imshow(
        initial_image, cmap="gray", vmin=0.76, vmax=1.30,
        interpolation="nearest",
    )
    x = np.arange(W)
    initial_line, = ax.plot(x, true_lines[0], color=SLATE, lw=1.35)
    final_line, = ax.plot(x, true_lines[-1], color=SLATE, lw=1.35, ls="--")
    for line in (initial_line, final_line):
        line.set_path_effects([
            pe.Stroke(linewidth=2.7, foreground="white"),
            pe.Normal(),
        ])

    p0 = true_particles[0]
    p1 = true_particles[-1]
    relative_dx = p1[:, 0] - p0[:, 0]
    relative_dy = (
        p1[:, 1] - centerline(p1[:, 0], 1.0)
        - (p0[:, 1] - centerline(p0[:, 0], 0.0))
    )
    relative_vectors = np.column_stack([relative_dx, relative_dy])
    relative_vector_scale = 2.0
    relative_end = p0 + relative_vector_scale * relative_vectors
    ax.scatter(
        p0[:, 0], p0[:, 1], s=24, facecolors="none",
        edgecolors="white", lw=1.8, zorder=5,
    )
    ax.scatter(
        p0[:, 0], p0[:, 1], s=16, facecolors="none",
        edgecolors=MAGENTA, lw=1.0, zorder=6,
    )
    for start, end in zip(p0, relative_end):
        arrow = ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle="->", color=MAGENTA, lw=1.25,
                mutation_scale=8, shrinkA=2, shrinkB=1,
            ),
        )
        arrow.arrow_patch.set_path_effects([
            pe.Stroke(linewidth=2.7, foreground="white"),
            pe.Normal(),
        ])

    drift_xy = np.array([truth_drift[-1, 1], truth_drift[-1, 0]])
    arrow_scale = 15.0
    drift_start = np.array([48.0, 55.0])
    drift_end = drift_start + arrow_scale * drift_xy
    drift_arrow = ax.annotate(
        "", xy=drift_end, xytext=drift_start,
        arrowprops=dict(
            arrowstyle="->", color=OCHRE, lw=1.7,
            mutation_scale=10, shrinkA=0, shrinkB=0,
        ),
    )
    drift_arrow.arrow_patch.set_path_effects([
        pe.Stroke(linewidth=2.8, foreground="white"),
        pe.Normal(),
    ])

    bar_px = 50.0 / PIXEL_SIZE_NM
    bar_x, bar_y = 30, 478
    ax.plot([bar_x, bar_x + bar_px], [bar_y, bar_y], color="white", lw=2.2)
    ax.text(
        bar_x + bar_px / 2, bar_y - 10, "50 nm",
        color="white", fontsize=6.0, ha="center", va="bottom",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=0.8),
    )

    motion_handles = [
        Line2D([0], [0], color=SLATE, lw=1.4,
               label="membrane, initial"),
        Line2D([0], [0], color=SLATE, lw=1.4, ls="--",
               label="membrane, final"),
        Line2D([0], [0], color=MAGENTA, lw=1.3, marker=">",
               label="nanoparticle motion direction"),
        Line2D([0], [0], color=OCHRE, lw=1.5, marker=">",
               label="global motion direction"),
    ]
    ax.legend(
        handles=motion_handles, loc="lower right", frameon=True,
        facecolor="white", edgecolor=LIGHT, framealpha=0.92,
        fontsize=5.3, handlelength=2.3, borderpad=0.5,
    )
    ax.set_title("Programmed motion", fontsize=7.0)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")


def draw_panel_b(ax, truth_drift, recovered_drift, drift_rmse):
    """Draw the known and recovered rigid-drift trajectories."""
    truth_x, truth_y = truth_drift[:, 1], truth_drift[:, 0]
    recovered_x, recovered_y = recovered_drift[:, 1], recovered_drift[:, 0]
    ax.plot(
        truth_x, truth_y, color=SLATE, lw=1.8,
        label="ground truth", zorder=2,
    )
    ax.plot(
        recovered_x, recovered_y, color=OCHRE, lw=1.55,
        ls="--", label="ACORN recovered", zorder=3,
    )
    ax.scatter(
        truth_x[0], truth_y[0], s=52, marker="o",
        facecolor=INK, edgecolor="white", lw=1.2, zorder=5,
    )
    ax.scatter(
        truth_x[-1], truth_y[-1], s=56, marker="s",
        facecolor=INK, edgecolor="white", lw=1.0, zorder=5,
    )
    ax.annotate(
        "start", (truth_x[0], truth_y[0]), xytext=(7, -10),
        textcoords="offset points", fontsize=6.2, color=INK,
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=0.8),
    )
    ax.annotate(
        "end", (truth_x[-1], truth_y[-1]), xytext=(7, 6),
        textcoords="offset points", fontsize=6.2, color=INK,
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=0.8),
    )

    all_x = np.concatenate([truth_x, recovered_x])
    all_y = np.concatenate([truth_y, recovered_y])
    x_pad = max(0.45, 0.08 * np.ptp(all_x))
    y_pad = max(0.45, 0.08 * np.ptp(all_y))
    ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
    ax.set_ylim(all_y.max() + y_pad, all_y.min() - y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x drift (pixels)")
    ax.set_ylabel("y drift (pixels)")
    ax.set_title(
        f"Global drift recovery (RMSE = {drift_rmse:.2f} pixels)",
        fontsize=7.0, pad=4,
    )
    ax.grid(color=LIGHT, lw=0.45, alpha=0.65)
    ax.legend(frameon=False, loc="upper left")


def draw_panel_c(ax, frames, aligned):
    """Compare matched regions before and after rigid motion correction."""
    ax.axis("off")
    ax.set_title("Effect of global motion correction", fontsize=7.0, pad=5)
    raw_avg = frames.mean(axis=0)
    corrected_avg = aligned.mean(axis=0)
    vmin, vmax = image_limits([raw_avg, corrected_avg])
    crop = np.s_[36:476, 36:476]
    for j, (image, label) in enumerate([
        (raw_avg[crop], "Uncorrected average"),
        (corrected_avg[crop], "Motion-corrected average"),
    ]):
        tile = ax.inset_axes([0.01 + 0.5 * j, 0.05, 0.48, 0.84])
        tile.imshow(
            image, cmap="gray", vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        tile.set_title(label, fontsize=6.3, pad=3)
        tile.axis("off")
        bar_px = 50.0 / PIXEL_SIZE_NM
        bar_x, bar_y = 24, image.shape[0] - 24
        scale_line, = tile.plot(
            [bar_x, bar_x + bar_px], [bar_y, bar_y],
            color="white", lw=2.2,
        )
        scale_line.set_path_effects([
            pe.Stroke(linewidth=3.8, foreground="black"),
            pe.Normal(),
        ])
        tile.text(
            bar_x + bar_px / 2, bar_y - 9, "50 nm",
            color="white", fontsize=5.6, ha="center", va="bottom",
            bbox=dict(
                facecolor="black", alpha=0.78,
                edgecolor="white", linewidth=0.35, pad=0.8,
            ),
        )


def draw_panel_d(ax, aligned):
    """Show matched specimen regions at four accumulated-dose intervals."""
    ax.axis("off")
    ax.set_title(
        "Dose-resolved structure after motion correction", fontsize=7.0, pad=5,
    )
    selected = [(0, 10), (30, 40), (60, 70), (90, 100)]
    images = [aligned[s:e].mean(axis=0) for s, e in selected]
    vmin, vmax = image_limits(images)
    for j, ((start, end), image) in enumerate(zip(selected, images)):
        row, col = divmod(j, 2)
        tile = ax.inset_axes([
            0.02 + 0.5 * col,
            0.52 - 0.47 * row,
            0.46,
            0.39,
        ])
        crop = image[96:416, 96:416]
        tile.imshow(
            crop, cmap="gray", vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        tile.set_title(
            f"{start * DOSE_PER_FRAME:.0f}-{end * DOSE_PER_FRAME:.0f} "
            + r"$\mathrm{e}^{-}\,\mathrm{\AA}^{-2}$",
            fontsize=5.8, pad=2,
        )
        tile.axis("off")

        bar_px = 50.0 / PIXEL_SIZE_NM
        bar_x, bar_y = 16, crop.shape[0] - 18
        scale_line, = tile.plot(
            [bar_x, bar_x + bar_px], [bar_y, bar_y],
            color="white", lw=2.0,
        )
        scale_line.set_path_effects([
            pe.Stroke(linewidth=3.5, foreground="black"),
            pe.Normal(),
        ])
        tile.text(
            bar_x + bar_px / 2, bar_y - 8, "50 nm",
            color="white", fontsize=5.1, ha="center", va="bottom",
            bbox=dict(
                facecolor="black", alpha=0.78,
                edgecolor="white", linewidth=0.3, pad=0.7,
            ),
        )


def draw_panel_e(ax, width_images, bin_rows):
    """Compare one dose state using three temporal integration widths."""
    ax.axis("off")
    ax.text(
        0.5, 0.99, "Dose-bin width: signal versus temporal averaging",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.0,
    )
    images = [item[0] for item in width_images]
    vmin, vmax = image_limits(images)
    for j, (image, row) in enumerate(zip(images, bin_rows)):
        tile = ax.inset_axes([0.01 + j / 3.0, 0.24, 0.32, 0.62])
        crop = image[96:416, 96:416]
        tile.imshow(
            crop, cmap="gray", vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        tile.set_title(
            f"{row['bin_width_e_per_A2']} "
            + r"$\mathrm{e}^{-}\,\mathrm{\AA}^{-2}$ bin",
            fontsize=5.8, pad=2,
        )
        tile.axis("off")

        bar_px = 50.0 / PIXEL_SIZE_NM
        bar_x, bar_y = 16, crop.shape[0] - 18
        scale_line, = tile.plot(
            [bar_x, bar_x + bar_px], [bar_y, bar_y],
            color="white", lw=2.0,
        )
        scale_line.set_path_effects([
            pe.Stroke(linewidth=3.5, foreground="black"),
            pe.Normal(),
        ])
        tile.text(
            bar_x + bar_px / 2, bar_y - 8, "50 nm",
            color="white", fontsize=5.0, ha="center", va="bottom",
            bbox=dict(
                facecolor="black", alpha=0.78,
                edgecolor="white", linewidth=0.3, pad=0.7,
            ),
        )
        tile.text(
            0.5, -0.035,
            f"membrane CNR {row['membrane_cnr']:.2f}\n"
            f"particle error {row['particle_localization_error_nm']:.2f} nm",
            transform=tile.transAxes, ha="center", va="top",
            fontsize=5.4, clip_on=False,
        )


def draw_panel_f(
    ax,
    doses,
    true_mem,
    measured_mem,
    true_rel,
    measured_rel,
    membrane_mae,
    particle_mae,
):
    """Compare known and measured local motion after global drift removal."""
    ax.axis("off")
    plot_ax = ax.inset_axes([0.13, 0.08, 0.84, 0.56])
    plot_ax.plot(
        doses, true_mem, color=SLATE, lw=1.7,
        label="membrane, ground truth",
    )
    plot_ax.plot(
        doses, measured_mem, color=SLATE, lw=1.2, ls="--",
        marker="o", ms=3.0, markerfacecolor="white",
        label="membrane, ACORN",
    )
    plot_ax.plot(
        doses, true_rel, color=MAGENTA, lw=1.7,
        label="nanoparticle, ground truth",
    )
    plot_ax.plot(
        doses, measured_rel, color=MAGENTA, lw=1.2, ls="--",
        marker="s", ms=2.8, markerfacecolor="white",
        label="nanoparticle, ACORN",
    )
    plot_ax.axhline(0, color=LIGHT, lw=0.7, zorder=0)
    plot_ax.set_xlabel(
        r"Cumulative dose ($\mathrm{e}^{-}\,\mathrm{\AA}^{-2}$)",
        fontsize=6.0,
    )
    plot_ax.set_ylabel("Displacement (nm)", fontsize=6.0)
    plot_ax.tick_params(labelsize=5.5)
    plot_ax.grid(color=LIGHT, alpha=0.65, lw=0.45)
    handles, labels = plot_ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, frameon=False, ncol=2,
        loc="upper center", bbox_to_anchor=(0.55, 0.80),
        fontsize=4.8, columnspacing=0.9, handlelength=2.1,
    )
    ax.text(
        0.5, 0.99, "Dose-resolved motion after global-drift correction",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.0,
    )
    ax.text(
        0.5, 0.89,
        f"Membrane MAE {membrane_mae:.2f} nm | "
        f"nanoparticle-relative MAE {particle_mae:.2f} nm",
        transform=ax.transAxes, ha="center", va="top", fontsize=5.2,
    )


def save_outputs(fig, metadata):
    png = OUT / "figure2_motion_dose_tracking.png"
    pdf = OUT / "figure2_motion_dose_tracking.pdf"
    tif = OUT / "figure2_motion_dose_tracking.tif"
    fig.savefig(png, dpi=DPI, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    with Image.open(png) as image:
        image.save(tif, dpi=(DPI, DPI), compression="tiff_lzw")
    metadata["outputs"] = {
        "png": png.name,
        "pdf": pdf.name,
        "tif": tif.name,
        "dpi": DPI,
        "width_inches": WIDTH_IN,
        "height_inches": HEIGHT_IN,
    }
    (OUT / "figure2_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    frames, true_drift, true_lines, true_particles, clean = build_movie()
    aligned, shifts = align_frames(frames)

    recovered_drift = center_trajectory(-shifts)
    truth_drift = center_trajectory(true_drift)
    drift_error = recovered_drift - truth_drift
    drift_rmse = float(np.sqrt(np.mean(np.sum(drift_error ** 2, axis=1))))

    (
        doses,
        true_mem,
        measured_mem,
        true_rel,
        measured_rel,
        analysis_bins,
        analysis_ranges,
        contours,
        tracks,
    ) = build_measurements(aligned, true_lines, true_particles)

    bin_rows, width_images = bin_width_comparison(
        aligned, true_lines, true_particles
    )

    membrane_mae = float(np.nanmean(np.abs(measured_mem - true_mem)))
    particle_mae = float(np.nanmean(np.abs(measured_rel - true_rel)))

    with open(OUT / "frame_trajectories.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "frame", "cumulative_dose_e_per_A2",
            "true_dy_px", "true_dx_px",
            "recovered_dy_px", "recovered_dx_px",
        ])
        for i in range(N_FRAMES):
            writer.writerow([
                i + 1, (i + 1) * DOSE_PER_FRAME,
                *truth_drift[i], *recovered_drift[i],
            ])

    with open(OUT / "dose_resolved_measurements.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dose_midpoint_e_per_A2",
            "true_membrane_displacement_nm",
            "measured_membrane_displacement_nm",
            "true_particle_relative_displacement_nm",
            "measured_particle_relative_displacement_nm",
        ])
        writer.writerows(zip(doses, true_mem, measured_mem, true_rel, measured_rel))

    with open(OUT / "bin_width_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bin_rows[0]))
        writer.writeheader()
        writer.writerows(bin_rows)

    if hasattr(tracks, "to_csv"):
        tracks.to_csv(OUT / "acorn_particle_tracks.csv", index=False)

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), constrained_layout=False)
    outer_gs = fig.add_gridspec(
        1, 2,
        left=0.055, right=0.985, bottom=0.055, top=0.975,
        wspace=0.19,
    )
    left_gs = outer_gs[0, 0].subgridspec(
        3, 1, hspace=0.25, height_ratios=[1.0, 1.0, 1.0],
    )
    right_gs = outer_gs[0, 1].subgridspec(
        3, 1, hspace=0.25, height_ratios=[1.0, 1.28, 0.72],
    )

    # A: independently programmed global and specimen-relative motion.
    ax = fig.add_subplot(left_gs[0, 0])
    draw_panel_a(ax, clean[0], true_lines, true_particles, truth_drift)

    review_fig, review_ax = plt.subplots(figsize=(3.5, 3.5))
    draw_panel_a(review_ax, clean[0], true_lines, true_particles, truth_drift)
    review_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
    review_fig.savefig(
        OUT / "panelA_review_current.png", dpi=400, facecolor="white",
    )
    plt.close(review_fig)

    # B: trajectory recovery.
    ax = fig.add_subplot(right_gs[0, 0])
    draw_panel_b(ax, truth_drift, recovered_drift, drift_rmse)

    review_fig, review_ax = plt.subplots(figsize=(3.5, 3.5))
    draw_panel_b(review_ax, truth_drift, recovered_drift, drift_rmse)
    review_fig.subplots_adjust(left=0.17, right=0.96, bottom=0.14, top=0.84)
    review_fig.savefig(
        OUT / "panelB_review_current.png", dpi=400, facecolor="white",
    )
    plt.close(review_fig)

    # C: uncorrected versus corrected average as equally sized image tiles.
    ax = fig.add_subplot(left_gs[1, 0])
    draw_panel_c(ax, frames, aligned)

    review_fig, review_ax = plt.subplots(figsize=(5.0, 2.9))
    draw_panel_c(review_ax, frames, aligned)
    review_fig.subplots_adjust(left=0.02, right=0.99, bottom=0.03, top=0.86)
    review_fig.savefig(
        OUT / "panelC_review_current.png", dpi=400, facecolor="white",
    )
    plt.close(review_fig)

    # D: progression at fixed 100 e-/A^2 bins in a 2 x 2 tile grid.
    ax = fig.add_subplot(right_gs[1, 0])
    draw_panel_d(ax, aligned)

    review_fig, review_ax = plt.subplots(figsize=(4.4, 4.7))
    draw_panel_d(review_ax, aligned)
    review_fig.subplots_adjust(left=0.03, right=0.99, bottom=0.02, top=0.90)
    review_fig.savefig(
        OUT / "panelD_review_current.png", dpi=400, facecolor="white",
    )
    plt.close(review_fig)

    # E: same central dose, different integration widths in equal tiles.
    ax = fig.add_subplot(left_gs[2, 0])
    panel_e_top = ax.get_position().y1
    draw_panel_e(ax, width_images, bin_rows)

    review_fig, review_ax = plt.subplots(figsize=(5.6, 3.1))
    draw_panel_e(review_ax, width_images, bin_rows)
    review_fig.subplots_adjust(left=0.02, right=0.99, bottom=0.03, top=0.86)
    panel_e_png = OUT / "panelE_review_current.png"
    panel_e_pdf = OUT / "panelE_review_current.pdf"
    panel_e_tif = OUT / "panelE_review_current.tif"
    review_fig.savefig(panel_e_png, dpi=600, facecolor="white")
    review_fig.savefig(panel_e_pdf, facecolor="white")
    plt.close(review_fig)
    with Image.open(panel_e_png) as image:
        image.save(panel_e_tif, dpi=(600, 600), compression="tiff_lzw")

    # F: recovered membrane and relative-particle movement.
    ax = fig.add_subplot(right_gs[2, 0])
    panel_f_position = ax.get_position()
    ax.set_position([
        panel_f_position.x0,
        panel_e_top - panel_f_position.height,
        panel_f_position.width,
        panel_f_position.height,
    ])
    draw_panel_f(
        ax, doses, true_mem, measured_mem, true_rel, measured_rel,
        membrane_mae, particle_mae,
    )

    review_fig, review_ax = plt.subplots(figsize=(5.0, 3.6))
    draw_panel_f(
        review_ax, doses, true_mem, measured_mem, true_rel, measured_rel,
        membrane_mae, particle_mae,
    )
    review_fig.subplots_adjust(left=0.13, right=0.98, bottom=0.15, top=0.82)
    review_fig.savefig(
        OUT / "panelF_review_current.png", dpi=400, facecolor="white",
    )
    plt.close(review_fig)

    left_label_x = outer_gs[0, 0].get_position(fig).x0
    right_label_x = outer_gs[0, 1].get_position(fig).x0 - 0.045
    row_tops = [
        left_gs[row, 0].get_position(fig).y1 for row in range(3)
    ]
    for letter, x, y in zip(
        "ABCDEF",
        [left_label_x, right_label_x] * 3,
        [row_tops[0], row_tops[0], row_tops[1],
         row_tops[1], row_tops[2], row_tops[2]],
    ):
        fig.text(x, y, letter, ha="left", va="top", fontsize=9,
                 fontweight="bold", color=INK, zorder=30)
    metadata = {
        "title": "Motion-corrected dose binning and feature tracking in a synthetic polymer nanocomposite",
        "simulation_scope": (
            "Generic nanoparticle-loaded phase-separated polymer thin-film phantom. "
            "Dose-dependent changes are phenomenological, not calibrated to a named material."
        ),
        "seed": SEED,
        "shape": [N_FRAMES, H, W],
        "pixel_size_nm": PIXEL_SIZE_NM,
        "dose_per_frame_e_per_A2": DOSE_PER_FRAME,
        "total_dose_e_per_A2": TOTAL_DOSE,
        "motion_correction": "ACORN two-pass rigid phase cross-correlation",
        "particle_linking": "ACORN Hungarian assignment with 18 nm maximum displacement",
        "analysis_bin_width_e_per_A2": 50,
        "drift_trajectory_rmse_px": drift_rmse,
        "membrane_displacement_mae_nm": membrane_mae,
        "particle_relative_displacement_mae_nm": particle_mae,
        "n_acorn_track_rows": int(len(tracks)),
        "bin_width_metrics": bin_rows,
        "source_script": Path(__file__).name,
    }
    save_outputs(fig, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
