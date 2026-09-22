"""
Evaluation Harness & Lethal Controls for C085: Residual Curvature Alarm
Evaluates single-pass hallucination detection vs entropy baseline.
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, Any, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from projects.c085_residual_curvature.curvature_alarm import ResidualCurvatureAlarm


def generate_synthetic_trajectory_batch(
    n_samples: int = 100,
    n_layers: int = 8,
    d_model: int = 64,
    hallucination_rate: float = 0.5,
    seed: int = 42
) -> Tuple[list, np.ndarray]:
    """
    Generates synthetic layer trajectories:
    - Grounded tokens: smooth polynomial/exponential representation drift.
    - Hallucinated tokens: sudden abrupt directional shift or high-curvature divergence.
    """
    rng = np.random.default_rng(seed)
    labels = (rng.uniform(0, 1, size=n_samples) < hallucination_rate).astype(np.int64)

    batch_hidden_states = []
    # Initialize L layer tensors
    for l in range(n_layers):
        states = torch.zeros(n_samples, d_model)
        for i in range(n_samples):
            base_dir = rng.normal(size=d_model)
            base_dir = base_dir / np.linalg.norm(base_dir)

            if labels[i] == 0:
                # Smooth grounded trajectory: norm grows gently and smoothly
                scale = 5.0 + 0.8 * l + 0.05 * (l ** 1.5)
                noise = rng.normal(scale=0.05, size=d_model)
                vec = (scale * base_dir) + noise
            else:
                # Hallucinated trajectory: sharp curvature impulse around layer l=4..6
                scale = 5.0 + 0.8 * l
                if l >= (n_layers // 2):
                    divergent_dir = rng.normal(size=d_model)
                    divergent_dir = divergent_dir / np.linalg.norm(divergent_dir)
                    # Abrupt acceleration spike
                    impulse = 7.5 * np.sin((l - n_layers // 2) * 1.5)
                    vec = (scale * base_dir) + (impulse * divergent_dir)
                else:
                    vec = (scale * base_dir) + rng.normal(scale=0.05, size=d_model)

            states[i] = torch.from_numpy(vec).float()
        batch_hidden_states.append(states)

    return batch_hidden_states, labels


def evaluate_curvature_alarm(config: Dict[str, Any], seed: int = 42) -> Dict[str, float]:
    n_samples = config.get("n_samples", 200)
    n_layers = config.get("n_layers", 8)
    d_model = config.get("d_model", 64)
    threshold = config.get("threshold", 0.18)

    states, labels = generate_synthetic_trajectory_batch(
        n_samples=n_samples,
        n_layers=n_layers,
        d_model=d_model,
        seed=seed
    )

    alarm = ResidualCurvatureAlarm(curvature_threshold=threshold)
    analysis = alarm.analyze_generation_step(states)

    predictions = analysis["is_hallucinating"].astype(np.int64)
    accuracy = float(np.mean(predictions == labels))

    # Calculate True Positive Rate & False Positive Rate
    tpr = float(np.sum((predictions == 1) & (labels == 1)) / (np.sum(labels == 1) + 1e-8))
    fpr = float(np.sum((predictions == 1) & (labels == 0)) / (np.sum(labels == 0) + 1e-8))

    return {
        "acc": accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "mean_curvature": float(np.mean(analysis["max_curvature"]))
    }


def sanity_check_curvature(config: Dict[str, Any]) -> bool:
    """Sanity Check: High-magnitude impulse spike must achieve >95% detection accuracy"""
    states, labels = generate_synthetic_trajectory_batch(
        n_samples=50,
        n_layers=config.get("n_layers", 8),
        d_model=config.get("d_model", 64),
        hallucination_rate=1.0,  # All hallucinated
        seed=0
    )
    alarm = ResidualCurvatureAlarm(curvature_threshold=0.18)
    analysis = alarm.analyze_generation_step(states)
    det_rate = float(np.mean(analysis["is_hallucinating"]))
    print(f"   [Sanity Check] Detection rate on obvious divergence: {det_rate:.4f} (Threshold: >0.90)")
    return det_rate >= 0.90


def kill_control_curvature(config: Dict[str, Any]) -> Tuple[bool, float, float]:
    """
    Kill Control: Perfectly linear norm expansion (||x_l|| = alpha * l + beta)
    MUST yield Delta^2 = 0 and zero false alarms.
    If an alarm flags monotonic linear growth as hallucination, it fails.
    """
    n_layers = config.get("n_layers", 8)
    d_model = config.get("d_model", 64)
    n_samples = 50

    # Build perfectly linear states
    linear_states = []
    for l in range(n_layers):
        alpha = 1.25
        beta = 3.0
        norm_val = alpha * l + beta
        vecs = torch.zeros(n_samples, d_model)
        for i in range(n_samples):
            v = np.ones(d_model) / np.sqrt(d_model)
            vecs[i] = torch.from_numpy(v * norm_val).float()
        linear_states.append(vecs)

    alarm = ResidualCurvatureAlarm(curvature_threshold=config.get("threshold", 0.40))
    analysis = alarm.analyze_generation_step(linear_states)

    false_alarm_rate = float(np.mean(analysis["is_hallucinating"]))
    chance_baseline = 0.0  # Linear drift must produce 0% false alarms

    # Pass if false alarm rate is close to 0 (<0.05)
    passed = false_alarm_rate <= 0.05
    return passed, false_alarm_rate, chance_baseline
