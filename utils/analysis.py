"""
analysis.py

Unified analysis utility for the Vipassana Meditation Model.
Provides verification, comparison, and steady-state analysis tools.
"""

import sys
import os
import argparse
import json
import numpy as np
import math
from pathlib import Path
from statistics import mean, median
from collections import defaultdict

# Add parent dir to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config.meditation_config as mc
from viz import plotting_utils
from viz.plotting_utils import load_json_data, STATE_DISPLAY_NAMES

# --- 1. Verify
NETWORKS = ['DMN', 'VAN', 'DAN', 'FPN']

def run_verification(cohort=None):
    cohorts = [cohort] if cohort else ["novice", "expert"]
    for c in cohorts:
        _calculate_stats(c)

def _calculate_stats(cohort):
    print(f"--- Processing {cohort} ---")
    ts_data, ai_data, stats_data = load_json_data(cohort)
    
    if 'state_history' not in stats_data:
        print(f"Error: No state_history found for {cohort}")
        return

    state_history = stats_data['state_history']
    fe_history = stats_data.get('free_energy_history', [])
    net_history = stats_data.get('network_activations_history', [])

    min_len = min(len(state_history), len(fe_history), len(net_history))
    state_history = state_history[:min_len]
    fe_history = fe_history[:min_len]
    net_history = net_history[:min_len]

    print(f"Total steps: {min_len}")
    tails = [("Full", 0), ("Tail (last 200)", -200)]
    
    for label, start_idx in tails:
        if abs(start_idx) > min_len: start_idx = 0
            
        s_hist = state_history[start_idx:]
        f_hist = fe_history[start_idx:]
        n_hist = net_history[start_idx:]
        
        print(f"\n--- {label} Stats ---")

        # Free Energy & Network Activations
        state_fe = {s: [] for s in STATE_DISPLAY_NAMES}
        state_net = {s: {n: [] for n in NETWORKS} for s in STATE_DISPLAY_NAMES}

        for i, state in enumerate(s_hist):
            if state in state_fe:
                state_fe[state].append(f_hist[i])
                for net in NETWORKS:
                    state_net[state][net].append(n_hist[i].get(net, 0.0))

        print("[Free Energy & Network Activations]")
        for state, name in STATE_DISPLAY_NAMES.items():
            fe_vals = state_fe[state]
            avg_fe = np.mean(fe_vals) if fe_vals else 0.0
            net_avgs = [f"{net}: {np.mean(state_net[state][net]):.2f}" if state_net[state][net] else f"{net}: 0.00" for net in NETWORKS]
            print(f"  {name}: FE={avg_fe:.2f} | {', '.join(net_avgs)}")

        # Dwell Times
        dwell_times = {s: [] for s in STATE_DISPLAY_NAMES}
        if s_hist:
            current_state = s_hist[0]
            current_duration = 0
            for state in s_hist:
                if state == current_state:
                    current_duration += 1
                else:
                    if current_state in dwell_times:
                        dwell_times[current_state].append(current_duration)
                    current_state = state
                    current_duration = 1
            if current_state in dwell_times:
                dwell_times[current_state].append(current_duration)

        print("[Dwell Times]")
        for state, name in STATE_DISPLAY_NAMES.items():
            durations = dwell_times[state]
            avg_dur = np.mean(durations) if durations else 0.0
            print(f"  {name}: {avg_dur:.1f} steps (n={len(durations)})")

        # Meta-Awareness
        ma_hist = stats_data.get('meta_awareness_history', [])
        ma_hist = ma_hist[start_idx:] if start_idx != 0 else ma_hist[:min_len]
        print(f"\n[Additional Diagnostics]\n  Meta-Awareness: Mean={np.mean(ma_hist) if ma_hist else 0.0:.2f}, SD={np.std(ma_hist) if ma_hist else 0.0:.2f}")

        # DAN-FPN Correlation
        dan_vals = [n.get('DAN', 0.0) for n in n_hist]
        fpn_vals = [n.get('FPN', 0.0) for n in n_hist]
        corr = np.corrcoef(dan_vals, fpn_vals)[0, 1] if len(dan_vals) > 1 else 0.0
        print(f"  DAN-FPN Correlation: {corr:.2f}")

        # Attractor Verification
        print("\n[Attractor Dynamics Verification]")
        act_hist = stats_data.get('activations_history', [])
        act_hist = act_hist[start_idx:] if start_idx != 0 else act_hist[:min_len]
        
        if act_hist:
            # breath_focus=0, pain_discomfort=1, pending_tasks=2
            bf_vals = [step[0] for step in act_hist]
            pt_vals = [step[2] for step in act_hist]
            
            print(f"  Pending Tasks > 0.8: {sum(1 for v in pt_vals if v > 0.8)} / {len(pt_vals)}")
            print(f"  Breath Focus Mean: {np.mean(bf_vals):.2f}")


# --- 2. Compare Transition Stats Logic ---
def run_comparison():
    base = Path(__file__).resolve().parents[1]
    _print_comparison('Novice', _summarize_trans(_load_json(base/'data'/'transition_stats_novice.json')))
    _print_comparison('Expert', _summarize_trans(_load_json(base/'data'/'transition_stats_expert.json')))

def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

def _summarize_trans(data):
    pats = data['state_transition_patterns']
    times = data['transition_timestamps']
    
    dmn_vals = [p['network_acts']['DMN'] for p in pats]
    bf_vals = [p['thoughtseed_activations']['attend_breath'] for p in pats]
    dan_vals = [p['network_acts']['DAN'] for p in pats]
    fpn_vals = [p['network_acts']['FPN'] for p in pats]

    runs = {}
    recovery = []
    for i, p in enumerate(pats):
        dur = times[0] if i == 0 else times[i] - times[i-1]
        runs.setdefault(p['from'], []).append(dur)
        if p['from'] == 'mind_wandering' and p['to'] == 'breath_focus':
            recovery.append(dur)

    corr = 0.0
    if len(dan_vals) > 1:
        m1, m2 = mean(dan_vals), mean(fpn_vals)
        num = sum((a-m1)*(b-m2) for a,b in zip(dan_vals, fpn_vals))
        den = math.sqrt(sum((a-m1)**2 for a in dan_vals)*sum((b-m2)**2 for b in fpn_vals))
        corr = num/den if den>0 else 0.0

    return {
        'n': len(pats),
        'dmn_mean': mean(dmn_vals) if dmn_vals else None,
        'bf_mean': mean(bf_vals) if bf_vals else None,
        'runs': runs,
        'recovery_count': len(recovery),
        'recovery_mean': mean(recovery) if recovery else None,
        'dan_mean': mean(dan_vals) if dan_vals else None,
        'corr': corr
    }

def _print_comparison(name, s):
    print(f"--- {name} Transition Stats ---")
    print(f"n_patterns: {s['n']}")
    print(f"DMN mean: {s['dmn_mean']:.4f}")
    print(f"attend_breath mean: {s['bf_mean']:.4f}")
    print(f"recovery (MW->BF): count={s['recovery_count']} mean={s['recovery_mean']}")
    print(f"DAN-FPN corr: {s['corr']:.4f}\n")


# --- 3. Steady State Analysis Logic ---
def run_steady_state(total_steps=2000, window=200):
    print("--- Steady State Analysis ---")
    base = Path(__file__).resolve().parents[1]
    _analyze_last_steps(base/'data'/'transition_stats_novice.json', total_steps, window)
    _analyze_last_steps(base/'data'/'transition_stats_expert.json', total_steps, window)

def _analyze_last_steps(filename, total_steps, window):
    with open(filename, 'r') as f: data = json.load(f)
    timestamps = data['transition_timestamps']
    patterns = data['state_transition_patterns']
    
    state_history = []
    last_time = 0
    for i, pattern in enumerate(patterns):
        state = pattern[0]
        if i < len(timestamps):
            duration = timestamps[i] - last_time
            state_history.extend([state] * int(duration))
            last_time = timestamps[i]
            
    if len(state_history) < total_steps:
        last_state = patterns[-1][1] if patterns else "breath_control"
        state_history.extend([last_state] * (total_steps - len(state_history)))
        
    relevant = state_history[-window:]
    counts = {s: relevant.count(s) for s in set(relevant)}
    
    print(f"\n{filename.name} (Last {window} steps):")
    total = len(relevant)
    for state, count in counts.items():
        print(f"  {state:<20}: {count:<5} ({count/total:.2%})")

# --- 4. Params Analysis Logic ---
def run_params():
    print("--- Config Parameter Analysis ---")
    vals = defaultdict(list)
    for k, v in mc.DEFAULTS.items():
        vals[(str(type(v)), repr(v))].append(k)
    
    dups = {k: ks for k, ks in vals.items() if len(ks) > 1}
    print(f'Exact-duplicate groups count: {len(dups)}')
    for (t, val), ks in dups.items():
        print(f"{ks} => {val}")

    states = mc.NETWORK_PROFILES['state_expected_profiles']
    print('\nState profile linear fits (a, b, resid):')
    for state, vals in states.items():
        novice = np.array(list(vals['novice'].values()))
        expert = np.array(list(vals['expert'].values()))
        A = np.vstack([novice, np.ones_like(novice)]).T
        a, b = np.linalg.lstsq(A, expert, rcond=None)[0]
        resid = np.linalg.norm(expert - (a * novice + b))
        print(f"{state}: a={a:.3f}, b={b:.3f}, resid={resid:.4f}")

if __name__ == "__main__":
    print("=== Vipassana Simulation Analysis ===\n")
    
    # 1. Verification (Novice & Expert)
    print(">>> 1. VERIFICATION STATS")
    run_verification()
    print("\n")

    # 2. Comparison
    print(">>> 2. TRANSITION COMPARISON")
    run_comparison()
    print("\n")

    # 3. Steady State
    print(">>> 3. STEADY STATE CONVERGENCE")
    run_steady_state()
    print("\n")

    # 4. Parameters
    print(">>> 4. PARAMETER CHECKS")
    run_params()
    print("\n")
