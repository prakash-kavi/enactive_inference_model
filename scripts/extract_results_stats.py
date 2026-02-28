"""
Extract statistics from training JSON files for manuscript update.
Computes dwell times, transition matrix, network profiles, thoughtseed stats, PCA.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

# Add project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import STATES, NETWORKS, THOUGHTSEEDS
from viz.analysis_utils import (
    get_tail_window,
    compute_tail_statistics,
    compute_network_profiles,
    compute_thoughtseed_means,
    TAIL_STEPS,
)


def main():
    data_dir = ROOT / "data"
    expert_path = data_dir / "training_results_expert_seed42.json"
    novice_path = data_dir / "training_results_novice_seed42.json"

    if not expert_path.exists() or not novice_path.exists():
        print("JSON files not found")
        return

    expert = json.load(open(expert_path))
    novice = json.load(open(novice_path))

    expert_tail = get_tail_window(expert, TAIL_STEPS)
    novice_tail = get_tail_window(novice, TAIL_STEPS)

    exp_stats = compute_tail_statistics(expert, STATES, TAIL_STEPS)
    nov_stats = compute_tail_statistics(novice, STATES, TAIL_STEPS)

    exp_profiles = compute_network_profiles(expert, STATES, NETWORKS, TAIL_STEPS)
    nov_profiles = compute_network_profiles(novice, STATES, NETWORKS, TAIL_STEPS)

    exp_ts = compute_thoughtseed_means(expert, STATES, THOUGHTSEEDS, TAIL_STEPS)
    nov_ts = compute_thoughtseed_means(novice, STATES, THOUGHTSEEDS, TAIL_STEPS)

    # --- Dwell times ---
    print("=" * 60)
    print("DWELL TIMES (mean steps)")
    print("=" * 60)
    for s in STATES:
        exp_d = exp_stats["dwell_times"].get(s, 0)
        nov_d = nov_stats["dwell_times"].get(s, 0)
        print(f"  {s:20s}: Expert {exp_d:6.1f}  Novice {nov_d:6.1f}")

    # Dwell std (run-length std)
    def dwell_std(tail_data):
        seq = tail_data["state_history"]
        dwells = {st: [] for st in STATES}
        if seq:
            cur, cnt = seq[0], 1
            for s in seq[1:]:
                if s == cur:
                    cnt += 1
                else:
                    dwells[cur].append(cnt)
                    cur, cnt = s, 1
            dwells[cur].append(cnt)
        return {st: float(np.std(v)) if v else 0.0 for st, v in dwells.items()}

    exp_dwell_std = dwell_std(expert_tail)
    nov_dwell_std = dwell_std(novice_tail)
    print("\nDwell std (steps):")
    for s in ["breath_focus", "mind_wandering", "meta_awareness"]:
        print(f"  {s}: Expert {exp_dwell_std[s]:.1f}, Novice {nov_dwell_std[s]:.1f}")

    # --- Transition matrix ---
    print("\n" + "=" * 60)
    print("TRANSITION MATRIX (tail-window event counts)")
    print("=" * 60)
    for phenotype, stats in [("Expert", exp_stats), ("Novice", nov_stats)]:
        print(f"\n{phenotype}:")
        tm = stats["transition_matrix"]
        for fs in STATES:
            row = tm.get(fs, {})
            total = sum(row.values())
            if total > 0:
                probs = {ts: row[ts] / total for ts in STATES}
                non_self = [(ts, probs[ts]) for ts in STATES if ts != fs and probs[ts] > 0.01]
                if non_self:
                    print(f"  {fs} -> " + ", ".join(f"{t}:{p:.2f}" for t, p in non_self))

    # Key transitions for manuscript
    print("\nKey transitions (for manuscript):")
    def get_trans(tm, f, t):
        row = tm.get(f, {})
        tot = sum(row.values())
        return row.get(t, 0) / tot if tot > 0 else 0.0
    print(f"  MW->MA: Expert {get_trans(exp_stats['transition_matrix'], 'mind_wandering', 'meta_awareness'):.2f}, Novice {get_trans(nov_stats['transition_matrix'], 'mind_wandering', 'meta_awareness'):.2f}")
    print(f"  MA->RA: Expert {get_trans(exp_stats['transition_matrix'], 'meta_awareness', 'redirect_attention'):.2f}, Novice {get_trans(nov_stats['transition_matrix'], 'meta_awareness', 'redirect_attention'):.2f}")
    print(f"  MA->MW: Expert {get_trans(exp_stats['transition_matrix'], 'meta_awareness', 'mind_wandering'):.2f}, Novice {get_trans(nov_stats['transition_matrix'], 'meta_awareness', 'mind_wandering'):.2f}")
    print(f"  RA->MW: Expert {get_trans(exp_stats['transition_matrix'], 'redirect_attention', 'mind_wandering'):.2f}, Novice {get_trans(nov_stats['transition_matrix'], 'redirect_attention', 'mind_wandering'):.2f}")

    # --- Network profiles (per state) ---
    print("\n" + "=" * 60)
    print("NETWORK PROFILES (mean activation per state)")
    print("=" * 60)
    for state in STATES:
        print(f"\n{state}:")
        for net in NETWORKS:
            exp_v = exp_profiles.get(state, {}).get(net, 0)
            nov_v = nov_profiles.get(state, {}).get(net, 0)
            print(f"  {net}: Expert {exp_v:.3f}  Novice {nov_v:.3f}")

    # --- Tail-window global network stats (for Fig 4 text) ---
    print("\n" + "=" * 60)
    print("TAIL-WINDOW GLOBAL STATS (for Fig 4)")
    print("=" * 60)

    def global_net_stats(tail_data, name):
        net_hist = tail_data.get("network_activations_history", [])
        if not net_hist:
            return
        arr = np.array([[r.get(n, 0) for n in NETWORKS] for r in net_hist])
        means = arr.mean(axis=0)
        stds = arr.std(axis=0, ddof=0)
        print(f"\n{name} L1 networks:")
        for i, n in enumerate(NETWORKS):
            print(f"  {n}: mean={means[i]:.3f}, std={stds[i]:.3f}")
        if len(NETWORKS) >= 4:  # DAN=2, FPN=3
            dan_fpn = np.corrcoef(arr[:, 2], arr[:, 3])[0, 1] if arr.shape[0] > 1 else 0
            print(f"  DAN-FPN correlation: {dan_fpn:.2f}")
        print(f"  DMN range: [{arr[:, 0].min():.2f}, {arr[:, 0].max():.2f}]")

    global_net_stats(expert_tail, "Expert")
    global_net_stats(novice_tail, "Novice")

    # --- Thoughtseed stats ---
    def ts_stats(tail_data, name):
        ts_hist = tail_data.get("thoughtseed_activations_history", [])
        if not ts_hist:
            return
        arr = np.array(ts_hist)
        print(f"\n{name} L2 thoughtseeds (tail mean +/- std):")
        for i, ts in enumerate(THOUGHTSEEDS):
            m, s = arr[:, i].mean(), arr[:, i].std(ddof=0)
            print(f"  {ts}: {m:.2f} +/- {s:.2f}")

    ts_stats(expert_tail, "Expert")
    ts_stats(novice_tail, "Novice")

    # --- PCA ---
    print("\n" + "=" * 60)
    print("PCA (L2 thoughtseeds, L1 networks)")
    print("=" * 60)

    def do_pca(tail_data, key, n_feat, label):
        if key == "thoughtseed":
            hist = tail_data.get("thoughtseed_activations_history", [])
            X = np.array(hist) if hist else np.zeros((0, n_feat))
        else:
            hist = tail_data.get("network_activations_history", [])
            X = np.array([[r.get(n, 0) for n in NETWORKS] for r in hist]) if hist else np.zeros((0, n_feat))
        if X.shape[0] < 2:
            return
        pca = PCA(n_components=2)
        pca.fit(X)
        var = pca.explained_variance_ratio_ * 100
        proj = pca.transform(X)
        std1, std2 = proj[:, 0].std(ddof=0), proj[:, 1].std(ddof=0)
        print(f"\n{label} {key}: PC1={var[0]:.1f}%, PC2={var[1]:.1f}%; PC1_std={std1:.3f}, PC2_std={std2:.3f}")

    do_pca(expert_tail, "thoughtseed", len(THOUGHTSEEDS), "Expert")
    do_pca(novice_tail, "thoughtseed", len(THOUGHTSEEDS), "Novice")
    do_pca(expert_tail, "network", len(NETWORKS), "Expert")
    do_pca(novice_tail, "network", len(NETWORKS), "Novice")

    # Pooled PCA (as in manuscript - both phenotypes together)
    print("\n--- Pooled PCA (expert+novice) ---")
    exp_ts_arr = np.array(expert_tail.get("thoughtseed_activations_history", []))
    nov_ts_arr = np.array(novice_tail.get("thoughtseed_activations_history", []))
    if exp_ts_arr.size > 0 and nov_ts_arr.size > 0:
        pooled_ts = np.vstack([exp_ts_arr, nov_ts_arr])
        pca = PCA(n_components=2)
        pca.fit(pooled_ts)
        var = pca.explained_variance_ratio_ * 100
        print(f"L2 thoughtseeds pooled: PC1={var[0]:.1f}%, PC2={var[1]:.1f}%")
        exp_proj = pca.transform(exp_ts_arr)
        nov_proj = pca.transform(nov_ts_arr)
        print(f"  Expert PC1/PC2 std: {exp_proj[:, 0].std():.3f}, {exp_proj[:, 1].std():.3f}")
        print(f"  Novice PC1/PC2 std: {nov_proj[:, 0].std():.3f}, {nov_proj[:, 1].std():.3f}")

    exp_net = np.array([[r.get(n, 0) for n in NETWORKS] for r in expert_tail.get("network_activations_history", [])])
    nov_net = np.array([[r.get(n, 0) for n in NETWORKS] for r in novice_tail.get("network_activations_history", [])])
    if exp_net.size > 0 and nov_net.size > 0:
        pooled_net = np.vstack([exp_net, nov_net])
        pca = PCA(n_components=2)
        pca.fit(pooled_net)
        var = pca.explained_variance_ratio_ * 100
        print(f"L1 networks pooled: PC1={var[0]:.1f}%, PC2={var[1]:.1f}%")
        exp_proj = pca.transform(exp_net)
        nov_proj = pca.transform(nov_net)
        print(f"  Expert PC1/PC2 std: {exp_proj[:, 0].std():.3f}, {exp_proj[:, 1].std():.3f}")
        print(f"  Novice PC1/PC2 std: {nov_proj[:, 0].std():.3f}, {nov_proj[:, 1].std():.3f}")

    # Free energy
    print("\n" + "=" * 60)
    print("FREE ENERGY")
    print("=" * 60)
    for name, data in [("Expert", expert), ("Novice", novice)]:
        fe = data.get("free_energy_history", [])
        if len(fe) >= 100:
            init = np.mean(fe[:100])
            fin = np.mean(fe[-100:])
            print(f"  {name}: initial={init:.5f}, final={fin:.5f}")


if __name__ == "__main__":
    main()
