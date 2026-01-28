
import sys
import os
import torch
import numpy as np
import logging

# Add parent directory to path
sys.path.append(os.getcwd())

from core.layer1_brain_networks import MeditationGenerativeProcess
from config.meditation_config import NETWORKS, STATES, NETWORK_PROFILES

def test_layer1_dynamics():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Layer1Debug")
    
    print("="*60)
    print("LAYER 1 ISOLATION TEST")
    print("="*60)
    
    # Run Comparison
    for level in ['novice', 'expert']:
        print("\n" + "="*40)
        print(f"TESTING LEVEL: {level.upper()}")
        print("="*40)
        
        process = MeditationGenerativeProcess(experience_level=level, seed=42)
        device = process.x.device
        
        for state in STATES:
            mu, theta = process.get_dynamics(state, device)
            diag = torch.diag(theta).cpu().numpy()
            
            # Print Key Diagnostics
            if state == 'breath_focus':
                print(f"State: {state.upper()}")
                print(f"  Target DMN: {mu[NETWORKS.index('DMN')].item():.2f}")
                print(f"  Diagonal Stiffness (Resistance): {diag}")
                
            elif state == 'mind_wandering':
                 print(f"State: {state.upper()}")
                 # Check Sticky (DMN-VAN Synergy should be negative)
                 idx_dmn, idx_van = NETWORKS.index('DMN'), NETWORKS.index('VAN')
                 synergy = theta[idx_dmn, idx_van].item()
                 print(f"  DMN-VAN Synergy (Sticky): {synergy:.2f}")

        # Stability Run
        print("\n  Running Stability Check (Breath Focus)...")
        process.reset('breath_focus')
        
        # Collect variance stats
        dmn_vals = []
        for t in range(200):
            res, _ = process.update({'noise_reduction': 1.0, 'vfe_accum_val': 0.0})
            dmn_vals.append(res['DMN'].item())
            
        dmn_std = np.std(dmn_vals)
        dmn_mean = np.mean(dmn_vals)
        print(f"  DMN Mean: {dmn_mean:.4f}, Std: {dmn_std:.4f}")
        if level == 'novice':
             print(f"  (Expect higher Std/Wobbly, Mean > Target)")
        else:
             print(f"  (Expect lower Std/Tonic, Mean ~ Target)")

if __name__ == "__main__":
    test_layer1_dynamics()
