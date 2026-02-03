import json
import numpy as np
from pathlib import Path

OUTPUT_BASE = Path('data/ablation')
SEED = 42
TIMESTEPS = 10000

ABLATION_CONFIGS = {
    'baseline': {'enable_forward_model': True, 'enable_l3_weighting': True, 'enable_forward_actions': True, 'description': 'Full Phase 4 system'},
    'no_forward_model': {'enable_forward_model': False, 'enable_l3_weighting': True, 'enable_forward_actions': False, 'description': 'No forward dynamics model'},
    'no_l3_weighting': {'enable_forward_model': True, 'enable_l3_weighting': False, 'enable_forward_actions': True, 'description': 'Fixed action loss weight (no L3 modulation)'},
    'no_forward_actions': {'enable_forward_model': True, 'enable_l3_weighting': True, 'enable_forward_actions': False, 'description': 'Forward model trains but actions not forward-informed'},
}

results = []
for c in ABLATION_CONFIGS:
    for l in ["expert", "novice"]:
        p = OUTPUT_BASE / c / l
        s = json.load(open(p / f"transition_stats_{l}.json"))
        d = json.load(open(p / f"active_inference_params_{l}.json"))
        t = s["state_transition_patterns"]
        ts = s["transition_timestamps"]
        
        # Compute dwell times
        dw = {}
        for i in range(len(t)):
            st = t[i]["from"]
            if st not in dw: dw[st] = []
            nt = ts[i+1] if i+1 < len(ts) else TIMESTEPS
            dw[st].append(nt - ts[i])
        
        # Compute transition matrix
        tm = {}
        for tr in t:
            f, to = tr["from"], tr["to"]
            if f not in tm: tm[f] = {}
            if to not in tm[f]: tm[f][to] = 0
            tm[f][to] += 1
        for f in tm:
            tot = sum(tm[f].values())
            for to in tm[f]: tm[f][to] /= tot
        
        results.append({
            "config": c,
            "level": l,
            "description": ABLATION_CONFIGS[c]["description"],
            "avg_dwell_mw": float(np.mean(dw.get("mind_wandering", [0]))),
            "avg_dwell_bf": float(np.mean(dw.get("breath_focus", [0]))),
            "final_fe": t[-1]["free_energy"] if t else None,
            "mw_to_ma_rate": tm.get("mind_wandering", {}).get("meta_awareness", 0),
            "action_error_mw": d.get("average_action_pred_error_by_state", {}).get("mind_wandering"),
            "action_error_bf": d.get("average_action_pred_error_by_state", {}).get("breath_focus")
        })

# Generate report
report_path = OUTPUT_BASE / 'ablation_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Phase 4 Ablation Study Results\n\n")
    f.write(f"**Seed**: {SEED}\n")
    f.write(f"**Timesteps**: {TIMESTEPS}\n\n")
    
    for level in ['expert', 'novice']:
        f.write(f"\n## {level.capitalize()} Results\n\n")
        level_results = [r for r in results if r['level'] == level]
        
        f.write("| Configuration | MW Dwell | BF Dwell | MW->MA Rate | Action Error (MW) |\n")
        f.write("|---------------|----------|----------|-------------|------------------|\n")
        
        baseline = next(r for r in level_results if r['config'] == 'baseline')
        f.write(f"| **Baseline** (full) | {baseline['avg_dwell_mw']:.1f} | "
               f"{baseline['avg_dwell_bf']:.1f} | {baseline['mw_to_ma_rate']:.3f} | "
               f"{baseline.get('action_error_mw', 'N/A')} |\n")
        
        for result in level_results:
            if result['config'] == 'baseline': continue
            mw_delta = result['avg_dwell_mw'] - baseline['avg_dwell_mw']
            mw_delta_str = f"{result['avg_dwell_mw']:.1f} ({mw_delta:+.1f})"
            ma_rate_delta = result['mw_to_ma_rate'] - baseline['mw_to_ma_rate']
            ma_rate_str = f"{result['mw_to_ma_rate']:.3f} ({ma_rate_delta:+.3f})"
            action_err = result.get('action_error_mw', 'N/A')
            if action_err != 'N/A':
                action_err = f"{action_err:.5f}"
            
            f.write(f"| {result['config'].replace('_', ' ').title()} | "
                   f"{mw_delta_str} | {result['avg_dwell_bf']:.1f} | "
                   f"{ma_rate_str} | {action_err} |\n")
        
        f.write("\n### Interpretation\n\n")
        f.write("- **MW Dwell**: Lower is better (faster escape from mind wandering)\n")
        f.write("- **MW->MA Rate**: Higher is better (more awareness transitions)\n")
        f.write("- **Positive deltas** = component removal hurt performance\n")
        f.write("- **Negative deltas** = component removal helped (rare, suggests overfitting)\n\n")

print(f"Report generated: {report_path}")
