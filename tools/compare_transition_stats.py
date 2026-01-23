import json
import math
from statistics import mean, median
from pathlib import Path

def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def summarize(data):
    pats = data['state_transition_patterns']
    times = data['transition_timestamps']
    n = len(pats)

    dmn_vals = [p['network_acts']['DMN'] for p in pats]
    bf_vals = [p['thoughtseed_activations']['breath_focus'] for p in pats]
    dan_vals = [p['network_acts']['DAN'] for p in pats]
    fpn_vals = [p['network_acts']['FPN'] for p in pats]

    # run-lengths: duration before each transition assigned to the 'from' state
    runs = {}
    recovery = []
    for i, p in enumerate(pats):
        if i == 0:
            dur = times[0]
        else:
            dur = times[i] - times[i-1]
        runs.setdefault(p['from'], []).append(dur)
        if p['from'] == 'mind_wandering' and p['to'] == 'breath_control':
            recovery.append(dur)

    # DAN-FPN correlation
    corr = None
    if len(dan_vals) > 1:
        m1 = mean(dan_vals)
        m2 = mean(fpn_vals)
        num = sum((a-m1)*(b-m2) for a,b in zip(dan_vals, fpn_vals))
        den = math.sqrt(sum((a-m1)**2 for a in dan_vals)*sum((b-m2)**2 for b in fpn_vals))
        corr = num/den if den>0 else 0.0

    return {
        'n_patterns': n,
        'dmn_mean': mean(dmn_vals) if dmn_vals else None,
        'dmn_sd': (mean([(x-mean(dmn_vals))**2 for x in dmn_vals])**0.5) if dmn_vals else None,
        'bf_mean': mean(bf_vals) if bf_vals else None,
        'bf_sd': (mean([(x-mean(bf_vals))**2 for x in bf_vals])**0.5) if bf_vals else None,
        'runs_by_state': {s: {'count': len(v), 'mean': mean(v), 'median': median(v)} for s,v in runs.items()},
        'recovery_count': len(recovery),
        'recovery_mean': mean(recovery) if recovery else None,
        'dan_mean': mean(dan_vals) if dan_vals else None,
        'fpn_mean': mean(fpn_vals) if fpn_vals else None,
        'dan_fpn_corr': corr
    }

def print_summary(name, s):
    print(f"--- {name} ---")
    print(f"n_patterns: {s['n_patterns']}")
    print(f"DMN mean: {s['dmn_mean']:.4f}  (sd {s['dmn_sd']:.4f})")
    print(f"breath_focus mean: {s['bf_mean']:.4f}  (sd {s['bf_sd']:.4f})")
    for st, info in s['runs_by_state'].items():
        print(f"runs {st}: count={info['count']} mean={info['mean']:.2f} median={info['median']}")
    print(f"recovery (mind_wandering->breath_control): count={s['recovery_count']} mean={s['recovery_mean']}")
    print(f"DAN mean: {s['dan_mean']:.4f}  FPN mean: {s['fpn_mean']:.4f}  corr: {s['dan_fpn_corr']}")
    print()

def main():
    base = Path(__file__).resolve().parents[1]
    nov = load(base/'data'/'transition_stats_novice.json')
    exp = load(base/'data'/'transition_stats_expert.json')
    s_n = summarize(nov)
    s_e = summarize(exp)
    print_summary('Novice', s_n)
    print_summary('Expert', s_e)

if __name__ == '__main__':
    main()
