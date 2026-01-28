
import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from core.layer1_brain_networks import MeditationGenerativeProcess
from config.meditation_config import DEFAULTS

def test_dwell_times():
    print("="*60)
    print("DWELL TIME DIAGNOSTIC")
    print("="*60)
    
    dt = DEFAULTS['DEFAULT_DT']
    print(f"Global Time Step (dt): {dt} seconds")
    print("-" * 60)

    for level in ['novice', 'expert']:
        print(f"\n--- Checking Level: {level.upper()} ---")
        # Initialize
        process = MeditationGenerativeProcess(experience_level=level, seed=42)
        
        # Check Configs (Seconds)
        print(f"Configured Ranges (Seconds):")
        for state, (min_s, max_s) in process.dwell_configs_sec.items():
            print(f"  {state:<20}: {min_s}s - {max_s}s")
            
        print("\nSampling Distribution (1000 samples per state)...")
        print(f"{'State':<20} | {'Steps (Mean)':<12} | {'Seconds (Mean)':<15} | {'Min (s)':<8} | {'Max (s)':<8}")
        print("-" * 80)
        
        for state in process.dwell_configs_sec.keys():
            process.current_state = state
            samples = []
            for _ in range(1000):
                s = process._sample_dwell() # Returns steps
                samples.append(s)
            
            steps_mean = np.mean(samples)
            sec_mean = steps_mean * dt
            sec_min = np.min(samples) * dt
            sec_max = np.max(samples) * dt
            
            print(f"{state:<20} | {steps_mean:<12.1f} | {sec_mean:<15.2f} | {sec_min:<8.2f} | {sec_max:<8.2f}")

if __name__ == "__main__":
    test_dwell_times()
