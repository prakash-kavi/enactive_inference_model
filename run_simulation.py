"""run_simulation.py — Run simulations using recalibrated attractors from training"""

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
import json
import numpy as np
from pathlib import Path
from meditation_model import ActInfAgent
from meditation_utils import ensure_directories
from meditation_trainer import Trainer

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

def initialize_agent_with_attractors(agent: ActInfAgent, attractors: dict):
    """Initialize agent's state-network expectations with trained attractors."""
    # Validate that all states are present in attractors
    missing_states = [state for state in agent.states if state not in attractors]
    if missing_states:
        raise ValueError(f"Missing attractors for states: {missing_states}")
    
    # Override the default state-network expectations with trained values
    for state in agent.states:
        if state in attractors:
            # Validate network keys match
            missing_networks = [net for net in agent.networks if net not in attractors[state]]
            if missing_networks:
                raise ValueError(f"Missing networks in attractor for state {state}: {missing_networks}")
            agent.learned_network_profiles["state_network_expectations"][state] = attractors[state].copy()
    
    # Rebuild state expectation vectors
    agent._state_expect_vectors = {
        state: agent._build_state_expect_vector(state) for state in agent.states
    }
    
    logging.info(f"Initialized {agent.experience_level} agent with trained attractors")

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
    agent_novice = ActInfAgent(experience_level='novice', timesteps_per_cycle=T)
    initialize_agent_with_attractors(agent_novice, novice_attractors)
    logging.info("Novice seed: %d", seed)
    Trainer(agent_novice).train(
        save_outputs=True, 
        output_dir=out_dir, 
        seed=seed,
        enable_learning=False  # Fixed attractors - no learning during simulation
    )

    # Run Expert Simulation
    logging.info("\n--- Running Expert Simulation ---")
    agent_expert = ActInfAgent(experience_level='expert', timesteps_per_cycle=T)
    initialize_agent_with_attractors(agent_expert, expert_attractors)
    logging.info("Expert seed: %d", seed)
    Trainer(agent_expert).train(
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
