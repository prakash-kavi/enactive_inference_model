"""Lightweight dwell-time verifier for the Layer 1 state machine."""

import argparse
import numpy as np

from core.layer1.state_machine import StateMachine
from core.layer1.layer1_config import L1_GENERATIVE_COSTS
from utils.meditation_config import DEFAULTS, STATES


def _summarize(values, dt):
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    secs = arr * dt
    return {
        "count": int(arr.size),
        "mean_steps": float(arr.mean()),
        "std_steps": float(arr.std(ddof=0)),
        "min_steps": float(arr.min()),
        "max_steps": float(arr.max()),
        "p10_steps": float(np.percentile(arr, 10)),
        "p50_steps": float(np.percentile(arr, 50)),
        "p90_steps": float(np.percentile(arr, 90)),
        "mean_sec": float(secs.mean()),
        "std_sec": float(secs.std(ddof=0)),
        "min_sec": float(secs.min()),
        "max_sec": float(secs.max()),
    }


def run(level, steps, seed, transition_drive):
    dt = DEFAULTS["DEFAULT_DT"]
    costs = L1_GENERATIVE_COSTS[level]
    sm = StateMachine(level, dt, seed, generative_costs=costs)

    dwell_steps_by_state = {s: [] for s in STATES}
    limit_steps_by_state = {s: [] for s in STATES}
    ratio_by_state = {s: [] for s in STATES}

    current_state = sm.current_state
    current_dwell = 0
    current_limit = sm.current_max_steps

    for _ in range(steps):
        prev_state = sm.current_state
        prev_limit = sm.current_max_steps
        sm.check_transition(transition_drive=transition_drive)
        current_dwell += 1

        if sm.current_state != prev_state:
            dwell_steps_by_state[prev_state].append(current_dwell)
            limit_steps_by_state[prev_state].append(prev_limit)
            ratio_by_state[prev_state].append(current_dwell / max(1, prev_limit))
            current_state = sm.current_state
            current_dwell = 0
            current_limit = sm.current_max_steps

    print(f"Level: {level} | Steps: {steps} | Seed: {seed} | dt: {dt}")
    print(f"Transition drive: {transition_drive}\n")

    for state in STATES:
        dwell_summary = _summarize(dwell_steps_by_state[state], dt)
        limit_summary = _summarize(limit_steps_by_state[state], dt)
        ratio_summary = _summarize(ratio_by_state[state], 1.0)

        print(f"State: {state}")
        print(f"  Actual dwell: {dwell_summary}")
        print(f"  Sampled limit: {limit_summary}")
        print(f"  Actual/limit ratio: {ratio_summary}\n")


def main():
    parser = argparse.ArgumentParser(description="Verify dwell time dispersion for L1 state machine.")
    parser.add_argument("--level", default="expert", choices=["novice", "expert"])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transition-drive", type=float, default=0.0)
    args = parser.parse_args()

    run(
        level=args.level,
        steps=args.steps,
        seed=args.seed,
        transition_drive=args.transition_drive,
    )


if __name__ == "__main__":
    main()
