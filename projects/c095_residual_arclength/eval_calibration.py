"""
Evaluation & Lethal Controls for C095: Residual Arc-Length Calibration
Tests resistance to sycophancy and prompt-induced overconfidence.
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, Any, Tuple, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from projects.c095_residual_arclength.arclength_metric import ResidualArcLengthMetric


def simulate_calibration_dataset(
    n_samples: int = 150,
    n_layers: int = 8,
    d_model: int = 64,
    threshold: float = 1.35,
    scramble_layers: bool = False,
    seed: int = 42
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    metric = ResidualArcLengthMetric(deliberation_threshold=threshold)

    true_deliberation_labels = []
    predicted_deliberation = []
    tortuosities = []

    for i in range(n_samples):
        # 50% genuine deliberation, 50% sycophantic / superficial overconfidence
        is_deliberating = (rng.uniform() < 0.5)
        true_deliberation_labels.append(1 if is_deliberating else 0)

        start_vec = torch.randn(d_model)
        start_vec = start_vec / torch.norm(start_vec)

        end_dir = torch.randn(d_model)
        end_dir = end_dir / torch.norm(end_dir)

        states = [start_vec]
        current = start_vec.clone()

        for l in range(1, n_layers):
            progress = l / (n_layers - 1)
            if is_deliberating:
                # Deliberative trajectory: traverses curved sub-manifolds (high tortuosity)
                tangent = torch.randn(d_model)
                tangent = tangent / torch.norm(tangent)
                # Adds intermediate curved exploration
                delta = 0.8 * (end_dir * 0.4 + tangent * 0.6)
            else:
                # Sycophantic / superficial shortcut: direct straight-line interpolation
                delta = 0.8 * end_dir + (torch.randn(d_model) * 0.02)

            current = current + delta
            states.append(current)

        if scramble_layers:
            # Kill control: scramble layer order
            perm = rng.permutation(len(states))
            states = [states[p] for p in perm]

        res = metric.compute_trajectory_metrics(states)
        tortuosities.append(res["tortuosity"])
        predicted_deliberation.append(1 if res["is_deliberating"] else 0)

    true_deliberation_labels = np.array(true_deliberation_labels)
    predicted_deliberation = np.array(predicted_deliberation)

    accuracy = float(np.mean(true_deliberation_labels == predicted_deliberation))
    tpr = float(np.sum((true_deliberation_labels == 1) & (predicted_deliberation == 1)) / (np.sum(true_deliberation_labels == 1) + 1e-8))
    fpr = float(np.sum((true_deliberation_labels == 0) & (predicted_deliberation == 1)) / (np.sum(true_deliberation_labels == 0) + 1e-8))

    return {
        "acc": accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "mean_tortuosity": float(np.mean(tortuosities))
    }


def evaluate_arclength_calibration(config: Dict[str, Any], seed: int = 42) -> Dict[str, float]:
    return simulate_calibration_dataset(
        n_samples=config.get("n_samples", 150),
        n_layers=config.get("n_layers", 8),
        threshold=config.get("threshold", 1.35),
        scramble_layers=False,
        seed=seed
    )


def sanity_check_arclength(config: Dict[str, Any]) -> bool:
    """Sanity Check: On clear contrastive trajectories, accuracy must exceed 85%"""
    res = simulate_calibration_dataset(n_samples=60, threshold=1.30, scramble_layers=False, seed=0)
    acc = res["acc"]
    print(f"   [Sanity Check] Calibration Accuracy: {acc:.4f} (Threshold: >0.85)")
    return acc >= 0.85


def kill_control_arclength(config: Dict[str, Any]) -> Tuple[bool, float, float]:
    """
    Kill Control: Scrambling layer ordering destroys path trajectory geometry.
    Discrimination must collapse to chance level (0.50).
    """
    res = simulate_calibration_dataset(n_samples=80, threshold=1.35, scramble_layers=True, seed=42)
    acc = res["acc"]
    chance_baseline = 0.50
    # Pass if scrambled accuracy collapses to within +-0.15 of chance
    passed = abs(acc - chance_baseline) < 0.15
    return passed, acc, chance_baseline
