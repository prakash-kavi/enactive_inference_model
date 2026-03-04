import json
import numpy as np
from itertools import groupby

def run_stats():
    for role in ['expert', 'novice']:
        print(f'\n================= {role.upper()} =================')
        try:
            with open(f'data/training_results_{role}_seed42.json', 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"File not found for {role}. Make sure the data folder has the json files.")
            continue
        
        tail_steps = 2000
        state_hist = data.get('state_history', [])[-tail_steps:]
        
        # Safe extraction for networks
        net_hist = data.get('network_activations_history', [])[-tail_steps:]
        net_arrays = []
        for n in net_hist:
            if isinstance(n, dict):
                net_arrays.append([n.get('DMN',0), n.get('VAN',0), n.get('DAN',0), n.get('FPN',0)])
            else:
                net_arrays.append(n)
        net_hist = np.array(net_arrays)
        
        # Safe extraction for thoughtseeds
        ts_hist = data.get('thoughtseed_activations_history', [])[-tail_steps:]
        ts_arrays = []
        for t in ts_hist:
            if isinstance(t, dict):
                ts_arrays.append([t.get('attend_breath',0), t.get('pain_discomfort',0), t.get('pending_tasks',0), t.get('aha_moment',0), t.get('equanimity',0)])
            else:
                ts_arrays.append(t)
        ts_hist = np.array(ts_arrays)
        
        states = ['breath_focus', 'mind_wandering', 'meta_awareness', 'redirect_attention']
        
        # 1. Calculate Mean Dwell Times
        dwells = {s: [] for s in states}
        for k, g in groupby(state_hist):
            if k in dwells:
                dwells[k].append(sum(1 for _ in g))
                
        print('-- Mean Dwell Times --')
        for s in states:
            val = np.mean(dwells[s]) if dwells[s] else 0.0
            print(f'{s:18}: {val:.1f}')
        
        # 2. Calculate Mean Transitions
        print('\n-- Transitions (Count/Total Exits) --')
        transitions = {s: {s2: 0 for s2 in states} for s in states}
        exits = {s: 0 for s in states}
        for i in range(len(state_hist)-1):
            curr = state_hist[i]
            nxt = state_hist[i+1]
            if curr != nxt and curr in states and nxt in states:
                transitions[curr][nxt] += 1
                exits[curr] += 1
                
        for s in states:
            for s2 in states:
                if s != s2:
                    prob = (transitions[s][s2] / exits[s]) if exits[s] > 0 else 0.0
                    if prob > 0:
                        print(f'{s} -> {s2}: {prob:.2f}')
        
        # 3. Calculate Mean Network Activations (L1)
        print('\n-- L1 Network Activations --')
        for s in states:
            idx = [i for i, state in enumerate(state_hist) if state == s]
            if idx and len(idx) > 0:
                means = np.mean(net_hist[idx], axis=0)
                try:
                    print(f'{s:18}: DMN={means[0]:.2f}, VAN={means[1]:.2f}, DAN={means[2]:.2f}, FPN={means[3]:.2f}')
                except IndexError:
                    pass
        
if __name__ == "__main__":
    run_stats()
