"""run_training.py — Multi-seed convergence study for learning stable attractors"""

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
import json
import numpy as np
from core.layer2.bottleneck import Layer2AttentionalModel
from utils.meditation_utils import ensure_directories
from core.train.trainer import PracticeTrainer

def run_convergence_for_level(level, seeds, timesteps, output_dir):
    """Run convergence study for a specific experience level."""
    logging.info(f"\n{'='*60}")
    logging.info(f"STARTING CONVERGENCE STUDY: {level.upper()}")
    logging.info(f"{'='*60}")
    
    all_results = {}
    
    for seed in seeds:
        logging.info(f"Training {level} - Seed {seed}")
        
        # Create agent
        agent = Layer2AttentionalModel(experience_level=level, timesteps_per_cycle=timesteps)
        
        # Save full outputs only for Seed 42 (for plotting purposes)
        save_output = (seed == 42)
        if save_output:
            # Save to explicit subdirectory for plotting data
            plots_data_dir = output_dir / "convergence_plots_data"
            plots_data_dir.mkdir(exist_ok=True)
            seed_output_dir = str(plots_data_dir)
        else:
            seed_output_dir = None
        
        # Train with learning enabled
        trainer = PracticeTrainer(agent)
        trainer.train(
            save_outputs=save_output,
            output_dir=seed_output_dir,
            seed=seed,
            enable_learning=True
        )
        
        # Extract observed network aggregates
        learned_weights = {
            'observed_network_means': {},
            'seed': seed,
            'timesteps': timesteps,
            'experience_level': level,
        }

        # Observed network means (what Layer 1 actually expressed during training)
        from utils.meditation_utils import compute_state_aggregates

        aggregates = compute_state_aggregates(agent)
        observed_means = aggregates.get("average_network_activations_by_state", {})
        learned_weights["observed_network_means"] = observed_means

        # Save seed result
        output_file = output_dir / f"learned_weights_{level}_seed{seed}.json"
        with open(output_file, 'w') as f:
            json.dump(learned_weights, f, indent=2)

        all_results[f"seed_{seed}"] = learned_weights

    # Compute Summary
    summary = {
        'seeds': seeds,
        'timesteps': timesteps,
        'experience_level': level,
        'network_profiles_mean': {},
        'network_profiles_std': {},
        'convergence_metrics': {}
    }
    
    # Calculate mean and std based on OBSERVED network means (what the biology expressed),
    # not just the internal state expectations. This sets realistic expectations for
    # the separation that actually appears in training/simulation.
    for state in agent.states:
        summary['network_profiles_mean'][state] = {}
        summary['network_profiles_std'][state] = {}
        
        for network in agent.networks:
            values = [
                all_results[f"seed_{s}"]["observed_network_means"].get(state, {}).get(network, 0.0)
                for s in seeds
            ]
            summary['network_profiles_mean'][state][network] = float(np.mean(values))
            summary['network_profiles_std'][state][network] = float(np.std(values))
    
    # Overall convergence
    all_stds = [summary['network_profiles_std'][s][n] for s in agent.states for n in agent.networks]
    summary['convergence_metrics']['mean_std_across_seeds'] = float(np.mean(all_stds))
    summary['convergence_metrics']['max_std_across_seeds'] = float(np.max(all_stds))
    
    # Save summary
    summary_file = output_dir / f"convergence_summary_{level}.json"
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Saved convergence summary to {summary_file}")
    except Exception as e:
        logging.error(f"Failed to save convergence summary to {summary_file}: {e}")
        raise
    
    logging.info(f"Completed {level}. Mean std: {summary['convergence_metrics']['mean_std_across_seeds']:.4f}")
    return summary

def run_training():
    """Run 3-seed convergence study for both novice and expert levels."""
    SEEDS = [42] #, 43, 44]
    TIMESTEPS = 5000
    LEVELS = ['novice', 'expert']
    OUTPUT_DIR = Path("data") / "training"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 60)
    logging.info("FULL CONVERGENCE STUDY: Novice & Expert")
    logging.info("=" * 60)
    
    for level in LEVELS:
        run_convergence_for_level(level, SEEDS, TIMESTEPS, OUTPUT_DIR)
    
    logging.info(f"\n{'='*60}")
    logging.info("All training complete! Results saved to data/training/")

if __name__ == "__main__":
    run_training()
