
from core.layer1_brain_networks import StateMachine
from config.meditation_config import DEFAULTS
import numpy as np

def test_dwells():
    dt = DEFAULTS['DEFAULT_DT'] # 0.2
    print("="*60)
    print("DWELL TIME VERIFICATION")
    print("="*60)
    print(f"DT (Time Step): {dt}s")
    
    # Test Novice
    sm = StateMachine('novice', dt, seed=42)
    print(f"Novice Config Ranges (sec): {sm.dwell_ranges_sec}")
    
    # Run transitions
    print("\nSimulating Transitions...")
    for i in range(5):
        start_state = sm.current_state
        steps = 0
        while True:
            # Force transition check effectively by incrementing internal timer
            # We just simulate the loop
            new_state = sm.check_transition(vfe_trigger=False)
            steps += 1
            if new_state != start_state:
                duration_s = steps * dt
                print(f"State: {start_state:<18} | Steps: {steps:<5} | Duration: {duration_s:.2f}s | Bounds: {sm.dwell_ranges_sec[start_state]}")
                break
                
    # Test Expert
    print("-" * 60)
    sm_exp = StateMachine('expert', dt, seed=42)
    print(f"Expert Config Ranges (sec): {sm_exp.dwell_ranges_sec}")
    
    for i in range(5):
        start_state = sm_exp.current_state
        steps = 0
        while True:
            new_state = sm_exp.check_transition(vfe_trigger=False)
            steps += 1
            if new_state != start_state:
                duration_s = steps * dt
                print(f"State: {start_state:<18} | Steps: {steps:<5} | Duration: {duration_s:.2f}s | Bounds: {sm_exp.dwell_ranges_sec[start_state]}")
                break

if __name__ == "__main__":
    test_dwells()
