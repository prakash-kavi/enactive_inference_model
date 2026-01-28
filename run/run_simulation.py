"""run_simulation.py — Run simulations using recalibrated attractors from training"""

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
import json
import numpy as np
from core.layer2_gnw_bottleneck import GNWBottleneck
from utils.meditation_utils import ensure_directories
from core.meditation_trainer import Trainer
from core.layer1_brain_networks import MeditationGenerativeProcess

def load_trained_attractors(level: str, training_dir: str = "data/training") -> dict:
    """Load mean state-network attractors from training convergence summary."""
    summary_path = Path(training_dir) / f"convergence_summary_{level}.json"
    
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {summary_path}\n"
            f"Please run 'python run_training.py' first to generate trained attractors."
        )
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    return summary['network_profiles_mean']

def validate_attractors(agent: GNWBottleneck, attractors: dict):
    """Validate that trained attractors contain all required states and networks."""
    missing_states = [state for state in agent.states if state not in attractors]
    if missing_states:
        raise ValueError(f"Missing attractors for states: {missing_states}")
    for state in agent.states:
        missing_networks = [net for net in agent.networks if net not in attractors[state]]
        if missing_networks:
            raise ValueError(f"Missing networks in attractor for state {state}: {missing_networks}")

def run_simulation():
    """Run simulations using recalibrated attractors from training."""
    seed = 42
    T = 3000
    out_dir = "data/simulation"
    training_dir = "data/training"

    logging.info("=" * 60)
    logging.info("SIMULATION: Using Recalibrated Attractors")
    logging.info("=" * 60)
    ensure_directories()
    
    # Load trained attractors
    logging.info("Loading trained attractors from %s...", training_dir)
    try:
        novice_attractors = load_trained_attractors('novice', training_dir)
        expert_attractors = load_trained_attractors('expert', training_dir)
        logging.info("✓ Loaded trained attractors for both levels")
    except FileNotFoundError as e:
        logging.error(str(e))
        return

    # Run Novice Simulation
    logging.info("\n--- Running Novice Simulation ---")
    agent_novice = GNWBottleneck(experience_level='novice', timesteps_per_cycle=T)
    validate_attractors(agent_novice, novice_attractors)
    
    # Initialize Layer 1 with learned attractors (so generative process uses trained attractors)
    process_novice = MeditationGenerativeProcess(
        experience_level='novice',
        seed=seed,
        learned_attractors=novice_attractors
    )
    
    logging.info("Novice seed: %d", seed)
    Trainer(agent_novice, generative_process=process_novice).train(
        save_outputs=True, 
        output_dir=out_dir, 
        seed=seed,
        enable_learning=False  # Fixed attractors - no learning during simulation
    )

    # Run Expert Simulation
    logging.info("\n--- Running Expert Simulation ---")
    agent_expert = GNWBottleneck(experience_level='expert', timesteps_per_cycle=T)
    validate_attractors(agent_expert, expert_attractors)
    
    # Initialize Layer 1 with learned attractors (so generative process uses trained attractors)
    process_expert = MeditationGenerativeProcess(
        experience_level='expert',
        seed=seed,
        learned_attractors=expert_attractors
    )
    
    logging.info("Expert seed: %d", seed)
    Trainer(agent_expert, generative_process=process_expert).train(
        save_outputs=True, 
        output_dir=out_dir, 
        seed=seed,
        enable_learning=False  # Fixed attractors - no learning during simulation
    )
    
    logging.info("\n" + "=" * 60)
    logging.info("Simulations complete! Results saved to data/simulation/")
    logging.info("=" * 60)

if __name__ == "__main__":
    run_simulation()
