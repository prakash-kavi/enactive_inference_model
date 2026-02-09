"""
attractors.py

Attractor visualizations using PCA projections.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

# Allow running as a script: `python viz/attractors.py`
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
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


def _fit_pca(X: np.ndarray) -> tuple[PCA | None, np.ndarray]:
    if X.size == 0 or X.shape[0] < 2:
        return None, np.zeros(2, dtype=float)
    n_components = min(2, X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X)
    var_ratio = pca.explained_variance_ratio_
    if var_ratio.size < 2:
        var_ratio = np.pad(var_ratio, (0, 2 - var_ratio.size))
    return pca, var_ratio[:2]


def _project_pca(X: np.ndarray, pca: PCA | None) -> np.ndarray:
    if X.size == 0 or pca is None:
        return np.zeros((0, 2), dtype=float)
    proj = pca.transform(X)
    if proj.shape[1] == 1:
        proj = np.column_stack([proj, np.zeros(proj.shape[0], dtype=float)])
    return proj


def _state_centroids_pca(
    projected: np.ndarray,
    states: List[str],
    means_by_state: Dict[str, Dict[str, float]] | None,
    feature_names: List[str],
    pca: PCA | None,
) -> Dict[str, np.ndarray]:
    centroids: Dict[str, np.ndarray] = {}
    if pca is None:
        return centroids
    mean = pca.mean_
    comps = pca.components_
    if comps.shape[0] == 1:
        comps = np.vstack([comps, np.zeros_like(comps)])
    for state in STATES:
        idx = [i for i, st in enumerate(states) if st == state]
        if idx:
            centroids[state] = projected[idx].mean(axis=0)
        elif means_by_state and state in means_by_state:
            vec = np.array([float(means_by_state[state].get(f, 0.5)) for f in feature_names])
            centroids[state] = (vec - mean) @ comps.T
    return centroids


def _plot_pca_pair(
    axes: tuple[plt.Axes, plt.Axes],
    novice: Dict[str, np.ndarray],
    expert: Dict[str, np.ndarray],
    feature_key: str,
    means_key: str,
    feature_names: List[str],
    row_title: str,
    cmap: LinearSegmentedColormap,
    norm: mpl.colors.Normalize,
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
        fe = cohort_data["free_energy"]
        states = cohort_data["states"]
        means = cohort_data[means_key]

        x = proj[:, 0] if proj.size else np.zeros(0, dtype=float)
        y = proj[:, 1] if proj.size else np.zeros(0, dtype=float)

        if fe.size:
            n = min(len(x), len(fe))
            x = x[:n]
            y = y[:n]
            fe_vals = np.asarray(fe[:n], dtype=float)
        else:
            fe_vals = np.zeros_like(x, dtype=float)

        if len(x) > 1:
            max_points = 1200
            step = max(1, len(x) // max_points)
            x_s = x[::step]
            y_s = y[::step]
            fe_s = fe_vals[::step]

            points = np.column_stack([x_s, y_s])
            if len(points) > 1:
                segments = np.stack([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=cmap, norm=norm)
                lc.set_array(fe_s[:-1])
                lc.set_linewidth(1.0)
                lc.set_alpha(0.35)
                ax.add_collection(lc)


        centroids = _state_centroids_pca(
            proj, states, means, feature_names, pca
        )
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
    """Figure 5: Stacked PCA trajectories (Thoughtseeds + Networks)."""
    novice = _prepare_data(novice_data)
    expert = _prepare_data(expert_data)

    set_plot_style()

    cmap = LinearSegmentedColormap.from_list(
        "fe_gradient", ["#0072B2", "#009E73", "#D55E00"]
    )
    fe_all = np.concatenate(
        [
            novice["free_energy"],
            expert["free_energy"],
        ]
    )
    if fe_all.size:
        fe_min, fe_max = float(fe_all.min()), float(fe_all.max())
        if abs(fe_max - fe_min) < 1e-9:
            fe_max = fe_min + 1e-6
    else:
        fe_min, fe_max = 0.0, 1.0
    norm = mpl.colors.PowerNorm(gamma=0.2, vmin=fe_min, vmax=fe_max)

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.15)

    ax_ts_left = fig.add_subplot(gs[0, 0])
    ax_ts_right = fig.add_subplot(gs[0, 1], sharex=ax_ts_left, sharey=ax_ts_left)
    ax_net_left = fig.add_subplot(gs[1, 0])
    ax_net_right = fig.add_subplot(gs[1, 1], sharex=ax_net_left, sharey=ax_net_left)

    _plot_pca_pair(
        (ax_ts_left, ax_ts_right),
        novice,
        expert,
        "activations",
        "activation_means",
        THOUGHTSEEDS,
        "L2 Thoughtseeds",
        cmap,
        norm,
    )
    _plot_pca_pair(
        (ax_net_left, ax_net_right),
        novice,
        expert,
        "network_activations",
        "network_means",
        NETWORKS,
        "L1 Networks",
        cmap,
        norm,
    )

    cbar_ax = fig.add_axes([0.12, 0.06, 0.76, 0.02])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Normalized Free Energy", fontweight="bold")
    mid = (fe_min + fe_max) / 2.0
    cbar.set_ticks([fe_min, mid, fe_max])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    for tick in cbar.ax.get_xticklabels():
        tick.set_fontweight("bold")

    fig.suptitle("PCA Trajectories Across the Hierarchy", fontsize=16, fontweight="bold")

    save_figure(fig, Path(save_path), "Fig4 PCA")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _load_results(data_dir: Path, cohort: str, seed: int) -> dict:
    path = data_dir / f"training_results_{cohort}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse
    import json

    from utils.analysis_utils import (
        TAIL_STEPS,
        get_tail_window,
        compute_network_profiles,
        compute_thoughtseed_means,
        compute_tail_statistics,
    )

    parser = argparse.ArgumentParser(description="Generate Fig4 PCA trajectories.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used in results filenames.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="Directory containing training_results_*.json files.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "plots" / "Fig4_PCA_Trajectories.png"),
        help="Output path for Fig4 PCA plot.",
    )
    parser.add_argument(
        "--tail-steps",
        type=int,
        default=TAIL_STEPS,
        help="Tail window length for trajectories.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    expert_results = _load_results(data_dir, "expert", args.seed)
    novice_results = _load_results(data_dir, "novice", args.seed)

    expert_tail = get_tail_window(expert_results, args.tail_steps)
    novice_tail = get_tail_window(novice_results, args.tail_steps)

    expert_network_profiles = compute_network_profiles(expert_results, STATES, NETWORKS, args.tail_steps)
    novice_network_profiles = compute_network_profiles(novice_results, STATES, NETWORKS, args.tail_steps)
    expert_ts_means = compute_thoughtseed_means(expert_results, STATES, THOUGHTSEEDS, args.tail_steps)
    novice_ts_means = compute_thoughtseed_means(novice_results, STATES, THOUGHTSEEDS, args.tail_steps)
    expert_tail_stats = compute_tail_statistics(expert_results, STATES, args.tail_steps)
    novice_tail_stats = compute_tail_statistics(novice_results, STATES, args.tail_steps)

    expert_tail["network_profiles_mean"] = expert_network_profiles
    novice_tail["network_profiles_mean"] = novice_network_profiles
    expert_tail["thoughtseed_means_per_state"] = expert_ts_means
    novice_tail["thoughtseed_means_per_state"] = novice_ts_means
    expert_tail.update(expert_tail_stats)
    novice_tail.update(novice_tail_stats)

    plot_attractor_pca(novice_tail, expert_tail, args.plot_path, show=args.show)
