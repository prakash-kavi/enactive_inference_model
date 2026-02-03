"""
Phase 4 Ablation Study
Systematically disable components to measure their contributions.
"""
import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Layer2AttentionalModel, PracticeTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

ABLATION_CONFIGS = {
    'baseline': {
        'enable_forward_model': True,
        'enable_l3_weighting': True,
        'enable_forward_actions': True,
        'description': 'Full Phase 4 system'
    },
    'no_forward_model': {
        'enable_forward_model': False,
        'enable_l3_weighting': True,
        'enable_forward_actions': False,  # Can't use forward actions without model
        'description': 'No forward dynamics model'
    },
    'no_l3_weighting': {
        'enable_forward_model': True,
        'enable_l3_weighting': False,
        'enable_forward_actions': True,
        'description': 'Fixed action loss weight (no L3 modulation)'
    },
    'no_forward_actions': {
        'enable_forward_model': True,
        'enable_l3_weighting': True,
        'enable_forward_actions': False,
        'description': 'Forward model trains but actions not forward-informed'
    },
}

SEED = 42
TIMESTEPS = 10000
OUTPUT_BASE = Path('data/ablation')

def run_single_ablation(config_name, config, level):
    """Run training with specific ablation configuration."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Configuration: {config_name}")
    logging.info(f"Level: {level}")
    logging.info(f"Description: {config['description']}")
    logging.info(f"{'='*60}")
    
    # Create output directory
    output_dir = OUTPUT_BASE / config_name / level
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build agent with ablation flags
    agent = Layer2AttentionalModel(
        experience_level=level,
        timesteps_per_cycle=TIMESTEPS,
        enable_forward_model=config['enable_forward_model'],
        enable_forward_actions=config['enable_forward_actions']
    )
    
    # Build trainer with ablation flags
    trainer = PracticeTrainer(
        agent=agent,
        enable_l3_weighting=config['enable_l3_weighting']
    )
    
    # Train
    trainer.train(
        enable_learning=True,
        seed=SEED,
        save_outputs=False  # We'll export manually
    )
    
    # Export results to our custom directory
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.logger.save_results(agent, str(output_dir))
    
    # Load key metrics from both JSON files
    params_file = output_dir / f'active_inference_params_{level}.json'
    stats_file = output_dir / f'transition_stats_{level}.json'
    
    with open(params_file) as f:
        params_data = json.load(f)
    with open(stats_file) as f:
        stats_data = json.load(f)
    
    # Compute state stats from transition patterns
    transitions = stats_data['state_transition_patterns']
    timestamps = stats_data['transition_timestamps']
    
    # Calculate dwell times
    state_dwells = {}
    for i in range(len(transitions)):
        state = transitions[i]['from']
        if state not in state_dwells:
            state_dwells[state] = []
        
        # Find next transition from same state
        next_time = timestamps[i+1] if i+1 < len(timestamps) else TIMESTEPS
        prev_time = timestamps[i]
        dwell = next_time - prev_time
        state_dwells[state].append(dwell)
    
    # Compute transition matrix
    transition_matrix = {}
    for trans in transitions:
        from_state = trans['from']
        to_state = trans['to']
        if from_state not in transition_matrix:
            transition_matrix[from_state] = {}
        if to_state not in transition_matrix[from_state]:
            transition_matrix[from_state][to_state] = 0
        transition_matrix[from_state][to_state] += 1
    
    # Normalize to probabilities
    for from_state in transition_matrix:
        total = sum(transition_matrix[from_state].values())
        for to_state in transition_matrix[from_state]:
            transition_matrix[from_state][to_state] /= total
    
    # Extract metrics
    import numpy as np
    metrics = {
        'config': config_name,
        'level': level,
        'description': config['description'],
        'avg_dwell_mw': float(np.mean(state_dwells.get('mind_wandering', [0]))),
        'avg_dwell_bf': float(np.mean(state_dwells.get('breath_focus', [0]))),
        'final_fe': transitions[-1]['free_energy'] if transitions else None,
        'mw_to_ma_rate': transition_matrix.get('mind_wandering', {}).get('meta_awareness', 0),
    }
    
    # Add Phase 4 metrics if available
    if config['enable_forward_model']:
        action_errors = params_data.get('average_action_pred_error_by_state', {})
        if action_errors:
            metrics['action_error_mw'] = action_errors.get('mind_wandering', None)
            metrics['action_error_bf'] = action_errors.get('breath_focus', None)
    
    return metrics

def generate_comparison_report(all_results):
    """Generate markdown report comparing all ablations."""
    report_path = OUTPUT_BASE / 'ablation_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 4 Ablation Study Results\n\n")
        f.write(f"**Seed**: {SEED}\n")
        f.write(f"**Timesteps**: {TIMESTEPS}\n\n")
        
        for level in ['expert', 'novice']:
            f.write(f"\n## {level.capitalize()} Results\n\n")
            
            # Filter results for this level
            level_results = [r for r in all_results if r['level'] == level]
            
            # Table header
            f.write("| Configuration | MW Dwell | BF Dwell | MW→MA Rate | Action Error (MW) |\n")
            f.write("|---------------|----------|----------|------------|------------------|\n")
            
            # Baseline first
            baseline = next(r for r in level_results if r['config'] == 'baseline')
            f.write(f"| **Baseline** (full) | {baseline['avg_dwell_mw']:.1f} | "
                   f"{baseline['avg_dwell_bf']:.1f} | {baseline['mw_to_ma_rate']:.3f} | "
                   f"{baseline.get('action_error_mw', 'N/A')} |\n")
            
            # Ablations
            for result in level_results:
                if result['config'] == 'baseline':
                    continue
                
                # Compute deltas vs baseline
                mw_delta = result['avg_dwell_mw'] - baseline['avg_dwell_mw']
                mw_delta_str = f"{result['avg_dwell_mw']:.1f} ({mw_delta:+.1f})"
                
                ma_rate_delta = result['mw_to_ma_rate'] - baseline['mw_to_ma_rate']
                ma_rate_str = f"{result['mw_to_ma_rate']:.3f} ({ma_rate_delta:+.3f})"
                
                action_err = result.get('action_error_mw', 'N/A')
                
                f.write(f"| {result['config'].replace('_', ' ').title()} | "
                       f"{mw_delta_str} | {result['avg_dwell_bf']:.1f} | "
                       f"{ma_rate_str} | {action_err} |\n")
            
            f.write("\n### Interpretation\n\n")
            f.write("- **MW Dwell**: Lower is better (faster escape from mind wandering)\n")
            f.write("- **MW→MA Rate**: Higher is better (more awareness transitions)\n")
            f.write("- **Positive deltas** = component removal hurt performance\n")
            f.write("- **Negative deltas** = component removal helped (rare, suggests overfitting)\n\n")
    
    logging.info(f"\n✅ Report saved to: {report_path}")
    return report_path

def main():
    """Run all ablation studies."""
    logging.info("="*60)
    logging.info("PHASE 4 ABLATION STUDY")
    logging.info("="*60)
    logging.info(f"Configurations: {len(ABLATION_CONFIGS)}")
    logging.info(f"Levels: expert, novice")
    logging.info(f"Total runs: {len(ABLATION_CONFIGS) * 2}")
    logging.info(f"Est. time: ~{len(ABLATION_CONFIGS) * 2 * 3} minutes")
    logging.info("="*60)
    
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Run all configurations for both levels
    for config_name, config in ABLATION_CONFIGS.items():
        for level in ['expert', 'novice']:
            try:
                metrics = run_single_ablation(config_name, config, level)
                all_results.append(metrics)
                logging.info(f"✅ Completed: {config_name} / {level}")
            except Exception as e:
                logging.error(f"❌ Failed: {config_name} / {level}: {e}")
                continue
    
    # Generate comparison report
    if all_results:
        report_path = generate_comparison_report(all_results)
        logging.info("\n" + "="*60)
        logging.info("ABLATION STUDY COMPLETE")
        logging.info(f"Results: {OUTPUT_BASE}")
        logging.info(f"Report: {report_path}")
        logging.info("="*60)
    else:
        logging.error("No results collected - all ablations failed")

if __name__ == '__main__':
    main()
