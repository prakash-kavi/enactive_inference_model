"""
attractors.py

Attractor visualizations using PCA projections.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

from utils.config import STATES, THOUGHTSEEDS, NETWORKS
from viz.plotting_utils import (
    STATE_COLORS,
    STATE_SHORT_NAMES,
    save_figure,
    set_plot_style,
)

def _prepare_data(tail_data: dict) -> Dict:
    """Prepare data dict in format expected by plot functions."""
    ts_history = tail_data.get('thoughtseed_activations_history', [])
    activations = np.array(ts_history) if ts_history else np.zeros((0, len(THOUGHTSEEDS)))
    state_history = tail_data.get('state_history', [])

    net_history = tail_data.get('network_activations_history', [])
    if net_history:
        net_acts = np.array(
            [[row.get(net, 0.0) for net in NETWORKS] for row in net_history],
            dtype=float,
        )
    else:
        net_acts = np.zeros((0, len(NETWORKS)), dtype=float)
    
    return {
        "activations": activations,
        "network_activations": net_acts,
        "states": state_history,
    }


def _fit_pca(X: np.ndarray) -> Tuple[Optional[PCA], np.ndarray]:
    if X.size == 0 or X.shape[0] < 2:
        return None, np.zeros(2, dtype=float)
    n_components = min(2, X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X)
    var_ratio = pca.explained_variance_ratio_
    if var_ratio.size < 2:
        var_ratio = np.pad(var_ratio, (0, 2 - var_ratio.size))
    return pca, var_ratio[:2]


def _project_pca(X: np.ndarray, pca: Optional[PCA]) -> np.ndarray:
    if X.size == 0 or pca is None:
        return np.zeros((0, 2), dtype=float)
    proj = pca.transform(X)
    if proj.shape[1] == 1:
        proj = np.column_stack([proj, np.zeros(proj.shape[0], dtype=float)])
    return proj


def _state_centroids_pca(
    projected: np.ndarray,
    states: List[str],
    pca: Optional[PCA],
) -> Dict[str, np.ndarray]:
    centroids: Dict[str, np.ndarray] = {}
    if pca is None:
        return centroids
    for state in STATES:
        idx = [i for i, st in enumerate(states) if st == state]
        if idx:
            centroids[state] = projected[idx].mean(axis=0)
    return centroids


def _plot_pca_pair(
    axes: tuple[plt.Axes, plt.Axes],
    novice: Dict[str, np.ndarray],
    expert: Dict[str, np.ndarray],
    feature_key: str,
    row_title: str,
) -> None:
    all_acts = np.concatenate([novice[feature_key], expert[feature_key]], axis=0)
    pca, var_ratio = _fit_pca(all_acts)
    nov_proj = _project_pca(novice[feature_key], pca)
    exp_proj = _project_pca(expert[feature_key], pca)

    axis_titles = (
        f"PC1 ({var_ratio[0]*100:.1f}%)",
        f"PC2 ({var_ratio[1]*100:.1f}%)",
    )

    for ax, cohort_data, proj, title in zip(
        axes, (novice, expert), (nov_proj, exp_proj), ("Novice", "Expert")
    ):
        states = cohort_data["states"]

        x = proj[:, 0] if proj.size else np.zeros(0, dtype=float)
        y = proj[:, 1] if proj.size else np.zeros(0, dtype=float)

        if len(x) > 1:
            max_points = 1200
            step = max(1, len(x) // max_points)
            x_s = x[::step]
            y_s = y[::step]

            points = np.column_stack([x_s, y_s])
            if len(points) > 1:
                segments = np.stack([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, colors="#1B4F72")
                lc.set_linewidth(1.0)
                lc.set_alpha(0.4)
                ax.add_collection(lc)


        centroids = _state_centroids_pca(proj, states, pca)
        for state, centre in centroids.items():
            ax.text(
                centre[0],
                centre[1],
                STATE_SHORT_NAMES.get(state, state),
                color=STATE_COLORS.get(state, "#000000"),
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2),
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(axis_titles[0], fontweight="bold")
        ax.grid(False)
        ax.margins(0.05)
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel(axis_titles[1], fontweight="bold")
    axes[0].text(
        -0.18,
        0.5,
        row_title,
        transform=axes[0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )


def plot_attractor_pca(
    novice_data: dict,
    expert_data: dict,
    save_path: str,
    show: bool = False,
):
    """Fig4: Stacked PCA trajectories (Thoughtseeds + Networks)."""
    novice = _prepare_data(novice_data)
    expert = _prepare_data(expert_data)

    set_plot_style()

    fig = plt.figure(figsize=(11.5, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.12)

    ax_ts_left = fig.add_subplot(gs[0, 0])
    ax_ts_right = fig.add_subplot(gs[0, 1], sharex=ax_ts_left, sharey=ax_ts_left)
    ax_net_left = fig.add_subplot(gs[1, 0])
    ax_net_right = fig.add_subplot(gs[1, 1], sharex=ax_net_left, sharey=ax_net_left)

    _plot_pca_pair(
        (ax_ts_left, ax_ts_right),
        novice,
        expert,
        "activations",
        "L2 Thoughtseeds",
    )
    _plot_pca_pair(
        (ax_net_left, ax_net_right),
        novice,
        expert,
        "network_activations",
        "L1 Networks",
    )

    # Ensure bottom row tick labels (PC1) remain visible.
    ax_net_left.tick_params(axis="x", labelbottom=True)
    ax_net_right.tick_params(axis="x", labelbottom=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.12)

    fig.suptitle("PCA Trajectories Across the Hierarchy", fontsize=16, fontweight="bold")

    save_figure(fig, Path(save_path), "Fig4 PCA")
    if show:
        plt.show()
    else:
        plt.close(fig)
