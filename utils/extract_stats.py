"""CLI utility: print dwell/transition/network stats from saved JSON results.

Delegates dwell and transition computation to viz.analysis_utils (single source of truth).
"""

import json
import numpy as np
from pathlib import Path

from utils.config import STATES, NETWORKS, THOUGHTSEEDS, PLOT_STEPS
from viz.analysis_utils import get_dwell_times, get_transition_matrix, compute_network_profiles


def run_stats():
    for role in ['expert', 'novice']:
        print(f'\n================= {role.upper()} =================')
        path = Path(f'data/training_results_{role}_seed42.json')
        if not path.exists():
            print(f"File not found: {path}. Run 'python run_enactive_inference.py run' first.")
            continue
        with open(path, 'r') as f:
            data = json.load(f)

        # 1. Dwell times (delegated to analysis_utils)
        tail = {'state_history': data.get('state_history', [])[-PLOT_STEPS:]}
        dwells = get_dwell_times(tail, STATES)
        print('-- Mean Dwell Times (timesteps) --')
        for s in STATES:
            vals = dwells.get(s, [])
            print(f'{s:22}: {np.mean(vals):.1f}' if vals else f'{s:22}: —')

        # 2. Transition matrix (delegated to analysis_utils)
        trans = get_transition_matrix(tail, STATES)
        print('\n-- Transition Probabilities --')
        for fs in STATES:
            for ts in STATES:
                prob = trans.get(fs, {}).get(ts, 0.0)
                if prob > 0.01:
                    print(f'  {fs} -> {ts}: {prob:.2f}')

        # 3. Network activation profiles (delegated to analysis_utils)
        profiles = compute_network_profiles(data, STATES, NETWORKS, PLOT_STEPS)
        print('\n-- L1 Network Activations (plot window mean) --')
        for s in STATES:
            p = profiles.get(s, {})
            vals = '  '.join(f'{n}={p.get(n, 0):.2f}' for n in NETWORKS)
            print(f'  {s:22}: {vals}')


if __name__ == "__main__":
    run_stats()

