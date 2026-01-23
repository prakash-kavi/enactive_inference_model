"""run_simulation.py — initialize and train novice/expert models"""

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
import numpy as np
from meditation_model import ActInfAgent
from meditation_utils import ensure_directories
from meditation_trainer import Trainer
from config.meditation_config import DEFAULTS

def run_simulation():
    seed = 42
    T = 2000
    out_dir = None

    logging.info("Starting simulation...")
    ensure_directories()
    logging.info("--- Training Novice Agent ---")
    # Use same seed for both agents
    novice_seed = seed

    learner_novice = ActInfAgent(experience_level='novice', timesteps_per_cycle=T)
    logging.info("Novice seed: %d", novice_seed)
    Trainer(learner_novice).train(save_outputs=True, output_dir=out_dir, seed=novice_seed)

    logging.info("--- Training Expert Agent ---")
    expert_seed = seed
    learner_expert = ActInfAgent(experience_level='expert', timesteps_per_cycle=T)
    logging.info("Expert seed: %d", expert_seed)
    Trainer(learner_expert).train(save_outputs=True, output_dir=out_dir, seed=expert_seed)
    
    logging.info("Active Inference training completed for both novice and expert agents.")

if __name__ == "__main__":
    run_simulation()
