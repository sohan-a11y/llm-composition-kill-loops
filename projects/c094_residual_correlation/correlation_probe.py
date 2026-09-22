"""
C094: Cross-Layer Residual Correlation Probe
Tracks layer-wise delta update profiles across consecutive autoregressive steps.
Calculates dynamic Pearson correlation rho_t = Corr(Delta x^{(t)}, Delta x^{(t-1)}).
Predicts reasoning collapse and errors 1-2 steps prior to token emission.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class CrossLayerResidualCorrelationProbe:
    def __init__(self, correlation_threshold: float = 0.50):
        self.correlation_threshold = correlation_threshold
        self.step_history: List[np.ndarray] = []

    def compute_layer_delta_profile(self, hidden_states: List[torch.Tensor]) -> np.ndarray:
        """
        hidden_states: List of L tensors of shape (1, D) or (1, T, D)
        Returns: 1D numpy array of length L-1 with L2 norms of Delta x_l
        """
        deltas = []
        for l in range(1, len(hidden_states)):
            h_prev = hidden_states[l - 1]
            h_curr = hidden_states[l]
            if h_curr.dim() == 3:
                h_prev = h_prev[:, -1, :]
                h_curr = h_curr[:, -1, :]
            delta = torch.norm(h_curr - h_prev, p=2, dim=-1).item()
            deltas.append(delta)
        return np.array(deltas, dtype=np.float32)

    def record_step(self, hidden_states: List[torch.Tensor]) -> Tuple[float, bool]:
        """
        Records the current generation step.
        Returns:
            rho: Pearson correlation with the immediately preceding step
            alarm: True if rho < threshold (derailment detected)
        """
        profile = self.compute_layer_delta_profile(hidden_states)
        self.step_history.append(profile)

        if len(self.step_history) < 2:
            return 1.0, False  # Initial step has no predecessor

        v_curr = self.step_history[-1]
        v_prev = self.step_history[-2]

        # Compute Pearson correlation
        std_curr = np.std(v_curr)
        std_prev = np.std(v_prev)

        if std_curr < 1e-8 or std_prev < 1e-8:
            rho = 1.0
        else:
            rho = float(np.corrcoef(v_curr, v_prev)[0, 1])

        alarm = (rho < self.correlation_threshold)
        return rho, alarm

    def reset(self):
        self.step_history.clear()
