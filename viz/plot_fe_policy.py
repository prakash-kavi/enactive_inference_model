"""
plot_fe_policy.py

Additional free-energy plots aligned with opacity and transition narratives:
1) FE vs Opacity scatter (Novice/Expert).
2) FE at transitions (cycle transitions; Novice/Expert).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from utils.meditation_config import STATES
from .plotting_utils import (
    DATA_DIR,
    PLOT_DIR,
    STATE_COLORS,
    STATE_DISPLAY_NAMES,
    STATE_SHORT_NAMES,
    TAIL_STEPS,
    get_tail_stats,
    load_json_data_from,
    save_figure,
    set_plot_style,
)


def _resolve_data_dir() -> Path:
    candidates = [
        Path(DATA_DIR) / "simulation",
        Path(DATA_DIR) / "training" / "convergence_plots_data",
        Path(DATA_DIR),
    ]
    for candidate in candidates:
        if (candidate / "thoughtseed_params_novice.json").exists():
            return candidate
    return Path(DATA_DIR)


def _extract_series(stats_data: dict, tail: int | None) -> dict:
    tail_stats = get_tail_stats(stats_data, tail=tail)
    fe = np.asarray(tail_stats.get("free_energy_history", []), dtype=float)
    meta = np.asarray(tail_stats.get("meta_awareness_history", []), dtype=float)
    states = tail_stats.get("state_history", [])
    n = min(len(fe), len(meta), len(states))
    return {
        "free_energy": fe[:n],
        "meta": meta[:n],
        "states": states[:n],
    }


def plot_fe_vs_opacity(data_dir: Path, save_path: Path, tail: int | None = TAIL_STEPS) -> None:
    set_plot_style()

    series = {}
    for cohort in ("novice", "expert"):
        _, _, stats = load_json_data_from(data_dir, cohort)
        series[cohort] = _extract_series(stats, tail)

    all_fe = np.concatenate([series["novice"]["free_energy"], series["expert"]["free_energy"]])
    if all_fe.size == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, cohort in zip(axes, ("novice", "expert")):
        fe = series[cohort]["free_energy"]
        meta = series[cohort]["meta"]
        states = series[cohort]["states"]
        opacity = np.clip(1.0 - meta, 0.0, 1.0)

        for state in STATES:
            idx = [i for i, s in enumerate(states) if s == state]
            if not idx:
                continue
            ax.scatter(
                opacity[idx],
                fe[idx],
                s=12,
                alpha=0.65,
                color=STATE_COLORS.get(state, "#666666"),
                label=STATE_SHORT_NAMES.get(state, state),
            )

        ax.set_title(cohort.title(), fontweight="bold")
        ax.set_xlabel("Opacity (1 − Meta-awareness)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Free Energy")
    # De-duplicate legend labels
    handles, labels = axes[0].get_legend_handles_labels()
    dedup = {}
    for h, l in zip(handles, labels):
        dedup[l] = h
    axes[0].legend(dedup.values(), dedup.keys(), loc="upper right", frameon=True)

    fig.suptitle("Free Energy vs Opacity", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, save_path, "FE vs Opacity")
    plt.close(fig)


def _collect_transition_fe(stats_data: dict, include_all: bool = False) -> Tuple[List[str], List[List[float]]]:
    patterns = stats_data.get("state_transition_patterns", [])
    if not patterns:
        return [], []

    cycle = [
        ("breath_focus", "mind_wandering"),
        ("mind_wandering", "meta_awareness"),
        ("meta_awareness", "redirect_attention"),
        ("redirect_attention", "breath_focus"),
    ]
    cycle_labels = [f"{STATE_SHORT_NAMES[a]}→{STATE_SHORT_NAMES[b]}" for a, b in cycle]
    cycle_set = set(cycle_labels)

    buckets: Dict[str, List[float]] = {}
    for row in patterns:
        frm = row.get("from")
        to = row.get("to")
        fe = row.get("free_energy")
        if frm is None or to is None or fe is None:
            continue
        label = f"{STATE_SHORT_NAMES.get(frm, frm)}→{STATE_SHORT_NAMES.get(to, to)}"
        buckets.setdefault(label, []).append(float(fe))

    labels = []
    series = []
    for label in cycle_labels:
        if label in buckets:
            labels.append(label)
            series.append(buckets[label])

    if include_all:
        others = [k for k in buckets.keys() if k not in cycle_set]
        for label in sorted(others):
            labels.append(label)
            series.append(buckets[label])

    return labels, series


def plot_transition_fe(data_dir: Path, save_path: Path, include_all: bool = False) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, cohort in zip(axes, ("novice", "expert")):
        _, _, stats = load_json_data_from(data_dir, cohort)
        labels, series = _collect_transition_fe(stats, include_all=include_all)
        if not series:
            ax.set_title(f"{cohort.title()} (no transitions)")
            continue

        box = ax.boxplot(series, patch_artist=True, labels=labels, showfliers=False)
        for patch, label in zip(box["boxes"], labels):
            frm = label.split("→")[0]
            color = STATE_COLORS.get(next((s for s in STATES if STATE_SHORT_NAMES[s] == frm), None), "#bbbbbb")
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_title(cohort.title(), fontweight="bold")
        ax.set_xlabel("Transition")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Free Energy at Transition")
    fig.suptitle("Free Energy by Transition Type (Cycle)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, save_path, "FE Transition")
    plt.close(fig)


def generate_plots(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    tail: int | None = TAIL_STEPS,
) -> None:
    data_dir = Path(data_dir) if data_dir else _resolve_data_dir()
    out_dir = Path(output_dir) if output_dir else Path(PLOT_DIR) / "simulation"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_fe_vs_opacity(data_dir, out_dir / "Fig5C_FE_Opacity.png", tail=tail)
    plot_transition_fe(data_dir, out_dir / "Fig5D_FE_Transitions.png", include_all=False)


if __name__ == "__main__":
    generate_plots()
