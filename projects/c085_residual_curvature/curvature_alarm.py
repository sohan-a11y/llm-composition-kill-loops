"""
C085: Residual Curvature Hallucination Alarm (Delta^2 ||x_l||)
Computes the 1D discrete second derivative of layer norm / hidden state norms
across transformer depth during generation.
Acts as a zero-overhead, single-pass test-time hallucination and uncertainty alarm.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple


class ResidualCurvatureAlarm:
    def __init__(self, curvature_threshold: float = 0.18):
        self.curvature_threshold = curvature_threshold

    def compute_layer_norms(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        hidden_states: List of L tensors of shape (B, T, D) or (B, D)
        Returns: Tensor of shape (B, L) representing L2 norm per layer
        """
        norms = []
        for hs in hidden_states:
            if hs.dim() == 3:
                # Norm at the last token position
                norm = torch.norm(hs[:, -1, :], p=2, dim=-1)  # (B,)
            elif hs.dim() == 2:
                norm = torch.norm(hs, p=2, dim=-1)  # (B,)
            else:
                raise ValueError(f"Unsupported hidden state shape: {hs.shape}")
            norms.append(norm)
        return torch.stack(norms, dim=1)  # (B, L)

    def compute_curvature(self, layer_norms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        layer_norms: (B, L)
        Returns:
            delta2: (B, L-2) discrete second derivative
            curvature_score: (B,) scalar alarm score in [0, 1]
        """
        B, L = layer_norms.shape
        if L < 3:
            raise ValueError(f"Need at least 3 layers to compute discrete second derivative, got {L}")

        # Discrete second derivative: x_l - 2*x_{l-1} + x_{l-2}
        delta2 = layer_norms[:, 2:] - 2 * layer_norms[:, 1:-1] + layer_norms[:, :-2]  # (B, L-2)

        # Max absolute curvature normalized by average norm
        avg_norm = torch.mean(layer_norms, dim=1, keepdim=True) + 1e-6
        normalized_delta2 = torch.abs(delta2) / avg_norm  # (B, L-2)

        max_curvature, _ = torch.max(normalized_delta2, dim=1)  # (B,)
        # Sigmoid scaling to [0, 1] confidence alarm
        alarm_score = torch.sigmoid((max_curvature - self.curvature_threshold) * 15.0)
        return delta2, alarm_score

    def analyze_generation_step(
        self,
        hidden_states: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyzes a single forward pass and returns detailed diagnostic metrics."""
        layer_norms = self.compute_layer_norms(hidden_states)
        delta2, alarm_score = self.compute_curvature(layer_norms)

        avg_norm = torch.mean(layer_norms, dim=1, keepdim=True) + 1e-6
        normalized_delta2 = torch.abs(delta2) / avg_norm
        max_curv, argmax_layer = torch.max(normalized_delta2, dim=-1)

        return {
            "alarm_score": alarm_score.detach().cpu().numpy(),
            "max_curvature": max_curv.detach().cpu().numpy(),
            "inflection_layer": (argmax_layer + 1).detach().cpu().numpy(),
            "is_hallucinating": (max_curv > self.curvature_threshold).detach().cpu().numpy()
        }
