
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.layer2_gnw_bottleneck import GNWBottleneck
from core.meditation_trainer import Trainer

def verify():
    print("Initializing Differentiable Engine (v2 - Contrastive)...")
    # 200 steps
    agent = GNWBottleneck(experience_level='expert', timesteps_per_cycle=200)
    trainer = Trainer(agent)
    
    print("Starting Training (BPTT)...")
    try:
        trainer.train(save_outputs=False, seed=42, enable_learning=True)
    except Exception as e:
        print(f"CRITICAL ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # Check VFE
    vfe = agent.free_energy_history
    if len(vfe) == 0:
        print("FAIL: No VFE history.")
        return
        
    # Check DMN-DAN Gap in Breath Focus
    # Filter observations for breath_focus
    states = agent.state_history
    nets = agent.network_activations_history
    
    dmn_vals = []
    dan_vals = []
    
    for s, n in zip(states, nets):
        if s == 'breath_focus':
            dmn_vals.append(n['DMN'])
            dan_vals.append(n['DAN'])
            
    if dmn_vals:
        avg_dmn = np.mean(dmn_vals)
        avg_dan = np.mean(dan_vals)
        gap = avg_dan - avg_dmn
        print(f"Breath Focus - Avg DMN: {avg_dmn:.4f}, Avg DAN: {avg_dan:.4f}, Gap: {gap:.4f}")
        
        # Check Learned Priors (The crucial part)
        print("\nChecking Learned Priors (state_expect_params)...")
        if 'breath_focus' in agent.state_expect_params:
            prior = agent.state_expect_params['breath_focus'].detach().cpu().numpy()
            # [DMN, VAN, DAN, FPN]
            p_dmn = prior[0]
            p_dan = prior[2]
            p_gap = p_dan - p_dmn
            print(f"Learned Prior - DMN: {p_dmn:.4f}, DAN: {p_dan:.4f}, Gap: {p_gap:.4f}")
            
            if p_gap > 0.15:
                print("PASS: Learned Prior shows healthy contrast.")
            else:
                print("WARN: Learned Prior contrast is weak.")

if __name__ == "__main__":
    verify()
