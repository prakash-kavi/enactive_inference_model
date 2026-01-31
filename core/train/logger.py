"""Simulation Logger: Decoupled history tracking for the meditation model."""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any
from utils.meditation_utils import compute_state_aggregates, build_transition_stats, _save_json_outputs

class SimulationLogger:
    """Handles logging of simulation steps, transitions, and metrics."""

    def __init__(self, experience_level: str):
        self.experience_level = experience_level
        self.reset()

    def reset(self):
        """Initialize empty history containers."""
        # Time Series
        self.state_history: List[str] = []
        self.activations_history: List[np.ndarray] = []
        self.meta_awareness_history: List[float] = []
        self.network_activations_history: List[Dict[str, float]] = []
        self.free_energy_history: List[float] = []
        self.prediction_error_history: List[float] = []
        self.precision_history: List[float] = []
        self.neural_efficiency_history: List[float] = []
        self.stability_indicators: List[float] = []
        self.dominant_ts_history: List[str] = []
        self.efe_history: List[float] = []
        self.transition_drive_history: List[float] = []
        self.recon_loss_history: List[float] = []
        self.kl_div_history: List[float] = []

        # Diagnostic Counters
        self.van_spike_detections: int = 0
        self.expert_mind_wandering_detections: int = 0

    def record_step(self, 
                    current_state: str,
                    activations: np.ndarray,
                    meta_awareness: float,
                    network_acts: Dict[str, float],
                    free_energy: float,
                    recon_loss: float,
                    kl_div: float,
                    transition_drive: float,
                    precision: float,
                    efe: float,
                    dominant_ts: str,
                    neural_efficiency: Optional[float],
                    stability_indicator: float,
                    is_expert_mw: bool = False,
                    is_van_spike: bool = False):
        """Log a single simulation timestep."""
        self.state_history.append(current_state)
        self.activations_history.append(activations)
        self.meta_awareness_history.append(meta_awareness)
        self.network_activations_history.append(network_acts)
        
        self.free_energy_history.append(free_energy)
        self.prediction_error_history.append(recon_loss) # recon_loss is used as approx for pred error
        self.recon_loss_history.append(recon_loss)
        self.kl_div_history.append(kl_div)
        
        self.transition_drive_history.append(transition_drive)
        self.precision_history.append(precision)
        self.efe_history.append(efe)
        self.dominant_ts_history.append(dominant_ts)
        
        if neural_efficiency is not None:
            self.neural_efficiency_history.append(neural_efficiency)
        
        self.stability_indicators.append(stability_indicator)

        if is_expert_mw:
            self.expert_mind_wandering_detections += 1
        
        if is_van_spike:
            self.van_spike_detections += 1

    def save_results(self, agent: Any, state_transition_patterns: List, transition_timestamps: List, output_dir: str = None):
        """Save simulation results to disk using shared utilities."""
        out_dir = output_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        os.makedirs(out_dir, exist_ok=True)

        # Compute Aggregates (Passing self as the history container)
        aggregates = compute_state_aggregates(agent, history=self) 
        
        # Build Transition Stats
        transition_stats = build_transition_stats(
            agent, 
            state_transition_patterns, 
            transition_timestamps, 
            aggregates
        )
        
        out_path = os.path.join(out_dir, f"transition_stats_{self.experience_level}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(transition_stats, f, indent=2)

        # Save JSON Outputs (Passing self as history container)
        _save_json_outputs(agent, history=self, output_dir=out_dir, aggregates=aggregates)
