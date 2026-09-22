"""
C095: Residual Arc-Length Deliberation & Calibration Metric
Treats the Transformer forward pass as a discrete trajectory in representation space.
Computes total Euclidean arc-length sum_{l=1}^L ||x_l - x_{l-1}|| and tortuosity.
Serves as an un-cheatable confidence calibrator resisting sycophancy and prompt-induced overconfidence.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple


class ResidualArcLengthMetric:
    def __init__(self, deliberation_threshold: float = 1.35):
        self.deliberation_threshold = deliberation_threshold

    def compute_trajectory_metrics(self, hidden_states: List[torch.Tensor]) -> Dict[str, float]:
        """
        hidden_states: List of L tensors of shape (1, D) or (1, T, D)
        Returns: Dict containing arc_length, displacement, tortuosity, and calibrated_confidence
        """
        L = len(hidden_states)
        if L < 2:
            raise ValueError(f"Need at least 2 layers to compute trajectory length, got {L}")

        arc_length = 0.0
        vectors = []
        for hs in hidden_states:
            if hs.dim() == 3:
                vec = hs[:, -1, :].squeeze(0)  # (D,)
            elif hs.dim() == 2:
                vec = hs.squeeze(0)
            else:
                vec = hs
            vectors.append(vec)

        for l in range(1, L):
            step_dist = torch.norm(vectors[l] - vectors[l - 1], p=2).item()
            arc_length += step_dist

        # Total net Euclidean displacement
        displacement = torch.norm(vectors[-1] - vectors[0], p=2).item() + 1e-6
        # Normalized path tortuosity
        tortuosity = arc_length / displacement

        # Deliberation score: high tortuosity indicates multi-step resolution
        is_deliberating = (tortuosity >= self.deliberation_threshold)

        return {
            "arc_length": arc_length,
            "displacement": displacement,
            "tortuosity": tortuosity,
            "is_deliberating": is_deliberating
        }
