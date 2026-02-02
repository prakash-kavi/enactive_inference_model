"""Post-run Phase-2 checks for latent VI logging and stability."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REQUIRED_VI_FIELDS = [
    "latent_reconstruction_history",
    "latent_prior_kl_history",
    "latent_sensory_consistency_history",
    "latent_temporal_consistency_history",
    "latent_vfe_total_history",
]


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) * (v - m) for v in values) / len(values))


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * q
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return sorted_vals[low]
    w = idx - low
    return sorted_vals[low] * (1.0 - w) + sorted_vals[high] * w


def _load_profile(data_dir: Path, profile: str) -> Tuple[dict, dict]:
    thoughtseed_path = data_dir / f"thoughtseed_params_{profile}.json"
    active_inf_path = data_dir / f"active_inference_params_{profile}.json"
    with thoughtseed_path.open("r", encoding="utf-8") as f:
        thoughtseed_data = json.load(f)
    with active_inf_path.open("r", encoding="utf-8") as f:
        active_inf_data = json.load(f)
    return thoughtseed_data, active_inf_data


def _tail(values: List, tail: int) -> List:
    if tail <= 0 or tail >= len(values):
        return values
    return values[-tail:]


def analyze_profile(
    data_dir: Path,
    profile: str,
    tail: int,
    max_step_delta: float,
    max_drift: float,
    max_kl_recon_ratio: float,
    max_clip_rate: float,
) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    failures: List[str] = []

    thoughtseed_data, active_inf_data = _load_profile(data_dir, profile)
    time_series = thoughtseed_data.get("time_series", {})

    for field in REQUIRED_VI_FIELDS:
        if field not in time_series:
            failures.append(f"{profile}: missing `{field}` in time_series")

    state_history = time_series.get("state_history", [])
    history_len = len(state_history)
    if history_len == 0:
        failures.append(f"{profile}: empty state history")
        return warnings, failures

    length_fields = [
        "activations_history",
        "recon_loss_history",
        "kl_div_history",
        "free_energy_history",
    ] + REQUIRED_VI_FIELDS
    for field in length_fields:
        if field in time_series and len(time_series[field]) != history_len:
            failures.append(
                f"{profile}: `{field}` length {len(time_series[field])} != state_history length {history_len}"
            )

    activations = time_series.get("activations_history", [])
    if not activations:
        failures.append(f"{profile}: empty activations_history")
        return warnings, failures

    activations_tail = _tail(activations, tail)
    recon_tail = [float(x) for x in _tail(time_series.get("recon_loss_history", []), tail)]
    kl_tail = [float(x) for x in _tail(time_series.get("kl_div_history", []), tail)]

    flat_tail = [float(v) for row in activations_tail for v in row]
    step_deltas: List[float] = []
    for t in range(1, len(activations_tail)):
        prev_row = activations_tail[t - 1]
        cur_row = activations_tail[t]
        for j in range(len(cur_row)):
            step_deltas.append(abs(float(cur_row[j]) - float(prev_row[j])))
    mean_step_delta = mean(step_deltas)
    if mean_step_delta > max_step_delta:
        warnings.append(f"{profile}: mean step delta {mean_step_delta:.4f} > {max_step_delta:.4f}")

    window = min(50, len(activations_tail) // 2)
    if window > 0:
        early = activations_tail[:window]
        late = activations_tail[-window:]
        drifts: List[float] = []
        for j in range(len(activations_tail[0])):
            early_mean = mean([float(row[j]) for row in early])
            late_mean = mean([float(row[j]) for row in late])
            drifts.append(abs(late_mean - early_mean))
        drift_mean = mean(drifts)
        if drift_mean > max_drift:
            warnings.append(f"{profile}: mean latent drift {drift_mean:.4f} > {max_drift:.4f}")
    else:
        drift_mean = 0.0

    low_clip = sum(1 for v in flat_tail if v <= 0.001) / max(1, len(flat_tail))
    high_clip = sum(1 for v in flat_tail if v >= 0.999) / max(1, len(flat_tail))
    if low_clip > max_clip_rate:
        warnings.append(f"{profile}: low clip rate {low_clip:.4%} > {max_clip_rate:.4%}")
    if high_clip > max_clip_rate:
        warnings.append(f"{profile}: high clip rate {high_clip:.4%} > {max_clip_rate:.4%}")

    if recon_tail and kl_tail and len(recon_tail) == len(kl_tail):
        ratio = mean([kl_tail[i] / (recon_tail[i] + 1e-12) for i in range(len(recon_tail))])
    else:
        ratio = 0.0
        failures.append(f"{profile}: missing recon/KL histories for ratio check")
    if ratio > max_kl_recon_ratio:
        warnings.append(f"{profile}: KL/recon ratio {ratio:.2f} > {max_kl_recon_ratio:.2f}")

    print(f"\n[{profile}]")
    print(f"- steps: {history_len}, tail: {len(activations_tail)}")
    print(f"- latent step delta mean: {mean_step_delta:.4f}")
    print(f"- latent drift mean: {drift_mean:.4f}")
    print(f"- clip rates (<=0.001, >=0.999): {low_clip:.2%}, {high_clip:.2%}")
    print(f"- recon tail mean/p95: {mean(recon_tail):.6f} / {quantile(recon_tail, 0.95):.6f}")
    print(f"- kl tail mean/p95: {mean(kl_tail):.6f} / {quantile(kl_tail, 0.95):.6f}")
    print(f"- KL/recon tail ratio mean: {ratio:.2f}")

    for field in REQUIRED_VI_FIELDS:
        if field in time_series:
            vals = [float(x) for x in _tail(time_series[field], tail)]
            print(f"- {field}: mean={mean(vals):.6f} std={std(vals):.6f} p95={quantile(vals, 0.95):.6f}")

    vi_param_keys = [
        "l2_vi_steps",
        "l2_vi_lr",
        "l2_vi_obs_weight",
        "l2_vi_prior_weight",
        "l2_vi_sensory_weight",
        "l2_vi_temporal_weight",
        "l2_vi_grad_clip",
    ]
    available_params = {k: active_inf_data.get(k) for k in vi_param_keys}
    print(f"- vi params: {available_params}")

    return warnings, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase-2 latent VI outputs from JSON logs.")
    parser.add_argument("--data-dir", default="data/simulation", help="Directory with simulation JSON outputs")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["novice", "expert"],
        help="Profiles to check, e.g. novice expert",
    )
    parser.add_argument("--tail", type=int, default=200, help="Tail window for stability checks")
    parser.add_argument("--max-step-delta", type=float, default=0.02, help="Warning threshold")
    parser.add_argument("--max-drift", type=float, default=0.35, help="Warning threshold")
    parser.add_argument("--max-kl-recon-ratio", type=float, default=100.0, help="Warning threshold")
    parser.add_argument("--max-clip-rate", type=float, default=0.01, help="Warning threshold")
    parser.add_argument("--strict", action="store_true", help="Return non-zero on warnings")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_warnings: List[str] = []
    all_failures: List[str] = []

    for profile in args.profiles:
        try:
            warnings, failures = analyze_profile(
                data_dir=data_dir,
                profile=profile,
                tail=args.tail,
                max_step_delta=args.max_step_delta,
                max_drift=args.max_drift,
                max_kl_recon_ratio=args.max_kl_recon_ratio,
                max_clip_rate=args.max_clip_rate,
            )
            all_warnings.extend(warnings)
            all_failures.extend(failures)
        except FileNotFoundError as exc:
            all_failures.append(f"{profile}: missing file {exc.filename}")

    if all_warnings:
        print("\nWarnings:")
        for msg in all_warnings:
            print(f"- {msg}")
    if all_failures:
        print("\nFailures:")
        for msg in all_failures:
            print(f"- {msg}")

    if all_failures:
        return 1
    if args.strict and all_warnings:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
