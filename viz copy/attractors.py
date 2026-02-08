"""
lean_attractors.py

Attractor visualizations using PCA projections.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from utils.config import STATES, THOUGHTSEEDS, NETWORKS
from viz.plotting_utils import (
    STATE_COLORS,
    STATE_SHORT_NAMES,
    save_figure,
    set_plot_style,
)

# Use constants
STATE_SHORT = STATE_SHORT_NAMES

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
    
    activation_means = tail_data.get('thoughtseed_means_per_state', {})
    activation_means_dict = {}
    for state, ts_array in activation_means.items():
        activation_means_dict[state] = {ts: float(ts_array[i]) for i, ts in enumerate(THOUGHTSEEDS)}

    network_means_dict = {}
    if net_acts.size and state_history:
        for state in STATES:
            idx = [i for i, st in enumerate(state_history) if st == state]
            if idx:
                means = net_acts[idx].mean(axis=0)
                network_means_dict[state] = {
                    net: float(means[j]) for j, net in enumerate(NETWORKS)
                }
    
    return {
        "activations": activations,
        "network_activations": net_acts,
        "free_energy": np.array(tail_data.get('free_energy_history', [])),
        "states": state_history,
        "activation_means": activation_means_dict,
        "network_means": network_means_dict,
    }


def _pca_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.size == 0:
        return np.zeros(0, dtype=float), np.zeros((2, 0), dtype=float), np.zeros(2, dtype=float)
    mean = X.mean(axis=0)
    Xc = X - mean
    if Xc.shape[0] < 2:
        comps = np.eye(Xc.shape[1])[:2]
        return mean, comps, np.array([1.0, 0.0], dtype=float)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    comps = vt[:2]
    var = (s ** 2) / max(1, (Xc.shape[0] - 1))
    total = var.sum()
    var_ratio = var / total if total > 0 else np.zeros_like(var)
    return mean, comps, var_ratio[:2]


def _pca_project(X: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.zeros((0, 2), dtype=float)
    return (X - mean) @ comps.T


def _state_centroids_pca(
    projected: np.ndarray,
    states: List[str],
    means_by_state: Dict[str, Dict[str, float]] | None,
    feature_names: List[str],
    mean: np.ndarray,
    comps: np.ndarray,
) -> Dict[str, np.ndarray]:
    centroids: Dict[str, np.ndarray] = {}
    for state in STATES:
        idx = [i for i, st in enumerate(states) if st == state]
        if idx:
            centroids[state] = projected[idx].mean(axis=0)
        elif means_by_state and state in means_by_state:
            vec = np.array([float(means_by_state[state].get(f, 0.5)) for f in feature_names])
            centroids[state] = (vec - mean) @ comps.T
    return centroids


def plot_attractor_2d(novice_data: dict, expert_data: dict, save_path: str):
    """Figure 5A: 2D PCA Attractor Landscape (Thoughtseeds)."""
    novice = _prepare_data(novice_data)
    expert = _prepare_data(expert_data)

    set_plot_style()

    all_acts = np.concatenate([novice["activations"], expert["activations"]], axis=0)
    mean, comps, var_ratio = _pca_fit(all_acts)
    nov_proj = _pca_project(novice["activations"], mean, comps)
    exp_proj = _pca_project(expert["activations"], mean, comps)

    axis_titles = (
        f"PC1 ({var_ratio[0]*100:.1f}%)",
        f"PC2 ({var_ratio[1]*100:.1f}%)",
    )

    nov_fe = novice["free_energy"]
    exp_fe = expert["free_energy"]
    
    all_fe = np.concatenate([nov_fe, exp_fe])
    fe_min, fe_max = all_fe.min(), all_fe.max()
    fe_range = fe_max - fe_min + 1e-10

    cmap = LinearSegmentedColormap.from_list(
        "fe_gradient", ["#0072B2", "#009E73", "#D55E00"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, cohort_data, proj, title in zip(
        axes, (novice, expert), (nov_proj, exp_proj), ("Novice", "Expert")
    ):
        fe = cohort_data["free_energy"]
        states = cohort_data["states"]
        means = cohort_data["activation_means"]

        x = proj[:, 0] if proj.size else np.zeros(0, dtype=float)
        y = proj[:, 1] if proj.size else np.zeros(0, dtype=float)

        if fe.size:
            fe_norm = (fe - fe_min) / fe_range
        else:
            fe_norm = np.zeros_like(x)

        for i in range(len(x) - 1):
            ax.plot(
                x[i : i + 2],
                y[i : i + 2],
                color=cmap(fe_norm[i]),
                linewidth=1.2,
                alpha=0.7,
            )

        centroids = _state_centroids_pca(
            proj, states, means, THOUGHTSEEDS, mean, comps
        )
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
        ax.set_xlabel(axis_titles[0], fontweight="bold")
        ax.grid(False)

    axes[0].set_ylabel(axis_titles[1], fontweight="bold")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.32, top=0.9, wspace=0.12)

    cbar_ax = fig.add_axes([0.12, 0.1, 0.76, 0.03])
    sm = mpl.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalized Free Energy", fontweight="bold")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    for tick in cbar.ax.get_xticklabels():
        tick.set_fontweight("bold")

    fig.suptitle("Thoughtseed Attractor Trajectories (PCA)", fontsize=16, fontweight="bold")

    save_figure(fig, Path(save_path), "Fig5A Thoughtseed PCA")
    plt.close(fig)


def plot_attractor_3d(novice_data: dict, expert_data: dict, save_path: str):
    """Figure 5B: 2D PCA Attractor Landscape (Networks)."""
    novice = _prepare_data(novice_data)
    expert = _prepare_data(expert_data)

    set_plot_style()

    all_acts = np.concatenate([novice["network_activations"], expert["network_activations"]], axis=0)
    mean, comps, var_ratio = _pca_fit(all_acts)
    nov_proj = _pca_project(novice["network_activations"], mean, comps)
    exp_proj = _pca_project(expert["network_activations"], mean, comps)

    axis_titles = (
        f"PC1 ({var_ratio[0]*100:.1f}%)",
        f"PC2 ({var_ratio[1]*100:.1f}%)",
    )

    nov_fe = novice["free_energy"]
    exp_fe = expert["free_energy"]
    all_fe = np.concatenate([nov_fe, exp_fe])
    fe_min, fe_max = all_fe.min(), all_fe.max()
    fe_range = fe_max - fe_min + 1e-10

    cmap = LinearSegmentedColormap.from_list(
        "fe_gradient", ["#0072B2", "#009E73", "#D55E00"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, cohort_data, proj, title in zip(
        axes, (novice, expert), (nov_proj, exp_proj), ("Novice", "Expert")
    ):
        fe = cohort_data["free_energy"]
        states = cohort_data["states"]
        means = cohort_data["network_means"]

        x = proj[:, 0] if proj.size else np.zeros(0, dtype=float)
        y = proj[:, 1] if proj.size else np.zeros(0, dtype=float)

        if fe.size:
            fe_norm = (fe - fe_min) / fe_range
        else:
            fe_norm = np.zeros_like(x)

        for i in range(len(x) - 1):
            ax.plot(
                x[i : i + 2],
                y[i : i + 2],
                color=cmap(fe_norm[i]),
                linewidth=1.2,
                alpha=0.7,
            )

        centroids = _state_centroids_pca(
            proj, states, means, NETWORKS, mean, comps
        )
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
        ax.set_xlabel(axis_titles[0], fontweight="bold")
        ax.grid(False)

    axes[0].set_ylabel(axis_titles[1], fontweight="bold")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.32, top=0.9, wspace=0.12)

    cbar_ax = fig.add_axes([0.12, 0.1, 0.76, 0.03])
    sm = mpl.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalized Free Energy", fontweight="bold")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    for tick in cbar.ax.get_xticklabels():
        tick.set_fontweight("bold")

    fig.suptitle("Network Attractor Trajectories (PCA)", fontsize=16, fontweight="bold")

    save_figure(fig, Path(save_path), "Fig5B Network PCA")
    plt.close(fig)
