"""
plot_attractors.py

Generates attractor landscape visualizations from tail of training rollouts:
- Figure 5A: 2D Attractor Landscape (Breath Focus vs Pending Tasks)
- Figure 5B: 3D Attractor Landscape with Free Energy surface
"""

from __future__ import annotations

import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from typing import Dict, List

from config.meditation_config import STATES, THOUGHTSEEDS
from . import plotting_utils as pu

# Use constants from plotting_utils
STATE_COLORS = pu.STATE_COLORS
STATE_SHORT = pu.STATE_SHORT_NAMES

TS_BF = THOUGHTSEEDS.index("attend_breath")
TS_PT = THOUGHTSEEDS.index("pending_tasks")

def _load_cohort_series(cohort: str, tail: int | None = 200) -> Dict[str, np.ndarray]:
    ts_data, _, stats_data = pu.load_json_data(cohort)
    
    # Use pu.get_tail_stats to slice
    tail_stats = pu.get_tail_stats(stats_data, tail=tail)
    
    activations = np.asarray(tail_stats.get("activations_history", []), dtype=float)
    free_energy = np.asarray(tail_stats.get("free_energy_history", []), dtype=float)
    states = tail_stats.get("state_history", [])
    
    if activations.ndim != 2 or activations.shape[0] == 0:
        raise ValueError(f"Activation history missing or malformed for {cohort}")

    return {
        "cohort": cohort,
        "activations": activations,
        "free_energy": free_energy,
        "states": states,
        "activation_means": ts_data.get("activation_means_by_state", {}),
    }

def _state_centroids(
    activations: np.ndarray,
    states: List[str],
    means_by_state: Dict[str, Dict[str, float]],
) -> Dict[str, np.ndarray]:
    centroids: Dict[str, np.ndarray] = {}
    for state in STATES:
        idx = [i for i, st in enumerate(states) if st == state]
        if idx:
            subset = activations[idx][:, [TS_BF, TS_PT]]
            centroids[state] = subset.mean(axis=0)
        elif state in means_by_state:
            means = means_by_state[state]
            centroids[state] = np.array([
                float(means.get("attend_breath", 0.5)),
                float(means.get("pending_tasks", 0.5)),
            ])
    return centroids

def _smooth_path(series: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    if series.size < 3:
        return series.copy()
    if kernel is None:
        kernel = np.array([0.2, 0.6, 0.2], dtype=float)
    kernel = kernel / kernel.sum()
    pad = len(kernel) // 2
    padded = np.pad(series, pad_width=pad, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(float)

def plot_attractor_2d(
    novice: Dict[str, np.ndarray],
    expert: Dict[str, np.ndarray],
    save_path: Path | None = None,
) -> None:
    """
    Figure 5A: 2D Attractor Landscape showing trajectory through thoughtseed phase space
    (Breath Focus vs Pending Tasks activations, colored by Free Energy)
    """
    pu.set_plot_style()

    # Calculate global min/max for consistent normalization
    nov_fe = novice["free_energy"]
    exp_fe = expert["free_energy"]
    
    all_fe = np.concatenate([nov_fe, exp_fe])
    fe_min, fe_max = all_fe.min(), all_fe.max()
    fe_range = fe_max - fe_min + 1e-10

    cmap = LinearSegmentedColormap.from_list(
        "fe_gradient", ["#0072B2", "#009E73", "#D55E00"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, cohort_data, title in zip(axes, (novice, expert), ("Novice", "Expert")):
        acts = cohort_data["activations"]
        fe = cohort_data["free_energy"]
        states = cohort_data["states"]
        means = cohort_data["activation_means"]

        x = acts[:, TS_BF]
        y = acts[:, TS_PT]

        x_smooth = _smooth_path(x)
        y_smooth = _smooth_path(y)

        # Normalize using global bounds
        if fe.size:
            fe_norm = (fe - fe_min) / fe_range
        else:
            fe_norm = np.zeros_like(x)

        for i in range(len(x) - 1):
            ax.plot(
                x_smooth[i : i + 2],
                y_smooth[i : i + 2],
                color=cmap(fe_norm[i]),
                linewidth=1.6,
                alpha=0.9,
            )

        centroids = _state_centroids(acts, states, means)
        for state, centre in centroids.items():
            ax.text(
                centre[0],
                centre[1],
                STATE_SHORT.get(state, state),
                color=STATE_COLORS.get(state, "#000000"),
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2),
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Breath Focus activation", fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(False)

    axes[0].set_ylabel("Pending Tasks activation", fontweight="bold")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.32, top=0.9, wspace=0.12)

    cbar_ax = fig.add_axes([0.12, 0.1, 0.76, 0.03])
    sm = mpl.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalized Free Energy", fontweight="bold")
    # Set ticks to 0 (Low) and 1 (High) to indicate relative scale
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    for tick in cbar.ax.get_xticklabels():
        tick.set_fontweight("bold")

    fig.suptitle("Thoughtseed Attractor Trajectories", fontsize=16, fontweight="bold")

    out_path = save_path or (Path(pu.PLOT_DIR) / "attractor_landscape_2d.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _kernel_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    bandwidth: float = 0.12,
    max_samples: int = 1200,
) -> np.ndarray:
    if z.size == 0:
        return np.full_like(grid_x, np.nan, dtype=float)

    step = max(1, z.size // max_samples)
    xs = x[::step][:, None]
    ys = y[::step][:, None]
    zs = z[::step][:, None]

    coords = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
    dx2 = (coords[:, 0:1] - xs.T) ** 2
    dy2 = (coords[:, 1:2] - ys.T) ** 2
    dist2 = dx2 + dy2

    denom = 2.0 * max(1e-6, bandwidth**2)
    weights = np.exp(-dist2 / denom)
    weighted = weights @ zs
    normaliser = weights.sum(axis=1, keepdims=True) + 1e-12
    surface = (weighted / normaliser).reshape(grid_x.shape)
    z_min, z_max = float(np.nanmin(z)), float(np.nanmax(z))
    return np.clip(surface, z_min, z_max)

def plot_attractor_landscape_3d(
    novice: Dict[str, np.ndarray],
    expert: Dict[str, np.ndarray],
    save_path: Path | None = None,
    bandwidth: float = 0.12,
) -> None:
    """
    Figure 5B: 3D Attractor Landscape showing Free Energy surface over thoughtseed phase space
    (Pending Tasks vs Breath Focus, with kernel-smoothed Free Energy as Z-axis)
    """
    pu.set_plot_style()

    xg = np.linspace(0.0, 1.0, 120)
    yg = np.linspace(0.0, 1.0, 120)
    grid_x, grid_y = np.meshgrid(xg, yg)

    def _prepare_surface(data: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        acts = data["activations"]
        fe = data["free_energy"]
        x = acts[:, TS_BF]
        y = acts[:, TS_PT]
        return x, y, fe

    nov_x, nov_y, nov_fe = _prepare_surface(novice)
    exp_x, exp_y, exp_fe = _prepare_surface(expert)

    # Global Normalization
    all_fe = np.concatenate([nov_fe, exp_fe])
    fe_min, fe_max = all_fe.min(), all_fe.max()
    fe_range = fe_max - fe_min + 1e-10

    nov_fe_norm = (nov_fe - fe_min) / fe_range
    exp_fe_norm = (exp_fe - fe_min) / fe_range

    nov_surface = _kernel_surface(nov_x, nov_y, nov_fe_norm, grid_x, grid_y, bandwidth)
    exp_surface = _kernel_surface(exp_x, exp_y, exp_fe_norm, grid_x, grid_y, bandwidth)

    z_min, z_max = 0.0, 1.0

    fig = plt.figure(figsize=(14, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])
    ax_left = fig.add_subplot(gs[0, 0], projection="3d")
    ax_right = fig.add_subplot(gs[0, 1], projection="3d")
    cax = fig.add_subplot(gs[0, 2])

    for ax in (ax_left, ax_right):
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(z_min, z_max)
        ax.view_init(elev=22, azim=140)
        ax.set_box_aspect((1, 1, 0.6))
        # Labels: Breath Focus (x) and Pending Tasks (y) to match Fig5A
        ax.set_xlabel("Breath Focus activation", labelpad=8, fontweight="bold")
        ax.set_ylabel("Pending Tasks activation", labelpad=8, fontweight="bold")
        ax.set_zlabel("Normalized Free Energy", labelpad=8, fontweight="bold")
        ax.grid(False)
        ax.set_zticks([])

    cmap = plt.get_cmap("viridis")
    # Normalization for mapping z-values to colors (used for adaptive label contrast)
    norm = mpl.colors.Normalize(vmin=z_min, vmax=z_max)
    ax_left.plot_surface(grid_x, grid_y, nov_surface, cmap=cmap, vmin=z_min, vmax=z_max, linewidth=0, antialiased=True, alpha=0.65)
    ax_right.plot_surface(grid_x, grid_y, exp_surface, cmap=cmap, vmin=z_min, vmax=z_max, linewidth=0, antialiased=True, alpha=0.65)

    levels = np.linspace(z_min, z_max, 8)
    ax_left.contour(grid_x, grid_y, nov_surface, levels=levels, colors="k", linestyles="--", offset=z_min, alpha=0.4)
    ax_right.contour(grid_x, grid_y, exp_surface, levels=levels, colors="k", linestyles="--", offset=z_min, alpha=0.4)

    for ax, data, title in (
        (ax_left, novice, "Novice"),
        (ax_right, expert, "Expert"),
    ):
        centroids = _state_centroids(data["activations"], data["states"], data["activation_means"])
        # Choose the corresponding surface for z-height lookup
        surface = nov_surface if title == "Novice" else exp_surface
        for state, centre in centroids.items():
            # Find nearest grid index for centroid coordinates
            xi = int(np.abs(xg - centre[0]).argmin())
            yi = int(np.abs(yg - centre[1]).argmin())
            try:
                z_val = float(surface[yi, xi])
            except Exception:
                z_val = float(z_min)

            z_plot = min(0.98, z_val + 0.02)

            # Render text with a white stroke for contrast over the surface
            # Choose text color adaptively for contrast against the surface
            # Map z value to cmap color and compute perceived luminance
            cmap_rgba = cmap(norm(z_val)) if 'cmap' in locals() else (0.0, 0.0, 0.0, 1.0)
            # Perceived luminance approximation from sRGB
            r, g, b = cmap_rgba[0], cmap_rgba[1], cmap_rgba[2]
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

            # If surface under label is dark, use white text; otherwise use dark blue
            if luminance < 0.5:
                text_color = "#FFFFFF"
            else:
                text_color = "#003366"  # dark blue for good contrast

            txt = ax.text(
                centre[0],
                centre[1],
                z_plot + 0.01,
                STATE_SHORT.get(state, state),
                color=text_color,
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
            )
            txt.set_path_effects([
                patheffects.Stroke(linewidth=3, foreground="#000000" if text_color=="#FFFFFF" else "white"),
                patheffects.Normal()
            ])
        ax.set_title(title, fontsize=13, fontweight="bold")

    norm = mpl.colors.Normalize(vmin=z_min, vmax=z_max)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, shrink=0.45, aspect=16)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle("Free-energy Landscape over Thoughtseed Phase Space", fontsize=16, fontweight="bold")

    # Update colorbar
    norm = mpl.colors.Normalize(vmin=z_min, vmax=z_max)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, shrink=0.45, aspect=16)
    cbar.set_label("Normalized Free Energy", fontweight="bold")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    cbar.ax.tick_params(labelsize=9)

    out_path = save_path or (Path(pu.PLOT_DIR) / "attractor_landscape_3d.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def generate_plots(tail: int | None = 200) -> None:
    novice = _load_cohort_series("novice", tail=tail)
    expert = _load_cohort_series("expert", tail=tail)
    
    # Figure 5A: 2D Attractor
    plot_attractor_2d(
        novice, expert, 
        save_path=Path(pu.PLOT_DIR) / "Fig5A_Attractor2D.png"
    )
    
    # Figure 5B: 3D Landscape
    plot_attractor_landscape_3d(
        novice, expert, 
        save_path=Path(pu.PLOT_DIR) / "Fig5B_Attractor3D.png"
    )
    try:
        rel = os.path.relpath(str(pu.PLOT_DIR), start=os.getcwd())
    except Exception:
        rel = str(pu.PLOT_DIR)
    logging.info("Saved attractor plots to %s", rel)

if __name__ == "__main__":
    generate_plots()
