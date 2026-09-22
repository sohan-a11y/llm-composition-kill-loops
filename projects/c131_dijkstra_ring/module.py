"""
Self-Stabilizing Dijkstra-Ring Virtual Token Memory (C131)
==========================================================

Mathematical Foundation:
Edsger W. Dijkstra (1974) introduced self-stabilizing distributed systems: systems guaranteed
to converge to a legitimate configuration from ANY arbitrary initial state (including corrupted
registers, deadlocks, or Byzantine multi-token states) in bounded transitions without external intervention.

In neural models with recurrent scratchpads or persistent working memory slots, transient faults
(quantization errors, SRAM soft-errors, adversarial spikes) can push hidden activations into
pathological failure modes (permanent hallucination loops).

We structure K working memory slots into a Dijkstra Token Ring with discrete states s_i in Z_M (M >= K):
1. Node 0 (Leader Privilege Rule):
   Node 0 has privilege iff s_0 == s_{K-1}
   State transition: s_0 <- (s_{K-1} + 1) mod M
2. Node i (Worker Privilege Rule, for i in {1, ..., K-1}):
   Node i has privilege iff s_i != s_{i-1}
   State transition: s_i <- s_{i-1}

Dijkstra's Self-Stabilization Theorem:
Regardless of the initial state s in Z_M^K, the ring self-stabilizes to a legitimate state
having EXACTLY ONE PRIVILEGE (P = 1) circulating perpetually in at most K(K-1)/2 steps.

In this module:
Memory slots receive write updates gated by the active privilege vector priv in {0, 1}^K.
Even if transient hardware corruption randomizes all state registers, the Dijkstra ring
self-stabilizes back to mutual exclusion (P=1) in <= K steps, preserving robust memory routing.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class SelfStabilizingDijkstraRing(nn.Module):
    """
    Dijkstra Self-Stabilizing Token Ring for working memory slots.
    """
    def __init__(self, K: int = 4, M: int = 5, d_model: int = 32):
        super().__init__()
        assert M >= K, "Dijkstra ring requires M >= K discrete states"
        self.K = K
        self.M = M
        self.d_model = d_model

        self.slot_embeddings = nn.Parameter(torch.randn(K, d_model) * 0.02)
        self.write_proj = nn.Linear(d_model * 2, d_model)

    def step_discrete_dijkstra(
        self,
        s: torch.Tensor,
        control_mode: str = "normal"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes one Dijkstra ring transition step.
        Args:
            s: (batch, K) integer state array in {0..M-1}.
            control_mode: 'normal', 'kill_deadlock', or 'kill_scramble'.
        Returns:
            s_next: (batch, K) updated states.
            priv: (batch, K) boolean privilege mask.
        """
        batch_size = s.shape[0]
        s_next = s.clone()
        priv = torch.zeros((batch_size, self.K), dtype=torch.bool, device=s.device)

        if control_mode == "kill_deadlock":
            # Lethal control: Anti-Dijkstra rule. Freeze leader token when s_0 == s_{K-1},
            # permanently causing 0 privileges (total deadlock)
            return s, priv

        elif control_mode == "kill_scramble":
            # Lethal control: Scramble ring topology randomly at each step
            perm = torch.randperm(self.K, device=s.device)
            s_shuffled = s[:, perm]
            return s_shuffled, priv

        # Normal Dijkstra Rules:
        # 1. Leader (node 0) has privilege iff s_0 == s_{K-1}
        priv[:, 0] = (s[:, 0] == s[:, self.K - 1])
        p0_mask = priv[:, 0]
        s_next[p0_mask, 0] = (s[p0_mask, self.K - 1] + 1) % self.M

        # 2. Worker nodes (i > 0) have privilege iff s_i != s_{i-1}
        for i in range(1, self.K):
            priv[:, i] = (s[:, i] != s[:, i - 1])
            pi_mask = priv[:, i]
            s_next[pi_mask, i] = s[pi_mask, i - 1]

        return s_next, priv

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        s_init: Optional[torch.Tensor] = None,
        control_mode: str = "normal"
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Routes an input token sequence through the self-stabilizing memory ring.
        Args:
            x_seq: (batch, seq_len, d_model)
            s_init: Optional initial state (batch, K). If None, initializes to legitimate state [0..0].
            control_mode: 'normal', 'kill_deadlock', or 'kill_scramble'.
        Returns:
            H: (batch, K, d_model) final memory bank.
            priv_history: List of (batch, K) privilege masks per step.
            final_s: (batch, K) final state array.
        """
        batch_size, seq_len, _ = x_seq.shape
        device = x_seq.device

        if s_init is None:
            # Legitimate initial state: s = [0, ..., 0] has exactly 1 privilege at node 0
            s = torch.zeros((batch_size, self.K), dtype=torch.long, device=device)
        else:
            s = s_init.clone().to(device)

        H = self.slot_embeddings.unsqueeze(0).expand(batch_size, -1, -1).clone()

        priv_history = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            s_next, priv = self.step_discrete_dijkstra(s, control_mode=control_mode)
            priv_history.append(priv)
            s = s_next

            # Memory update gated by active privileges
            priv_weights = priv.float().unsqueeze(-1)
            H_in = torch.cat([H, x_t.unsqueeze(1).expand(-1, self.K, -1)], dim=-1)
            dH = self.write_proj(H_in)
            H = H + priv_weights * dH

        return H, priv_history, s


class DijkstraMemoryClassifier(nn.Module):
    """
    Sequence Classifier with Self-Stabilizing Dijkstra Working Memory.
    """
    def __init__(
        self,
        num_tokens: int = 20,
        d_model: int = 32,
        num_classes: int = 5,
        K: int = 4,
        M: int = 5
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.ring = SelfStabilizingDijkstraRing(K=K, M=M, d_model=d_model)
        self.K = K
        self.head = nn.Linear(d_model * K, num_classes)

    def forward(
        self,
        token_seq: torch.Tensor,
        s_init: Optional[torch.Tensor] = None,
        control_mode: str = "normal"
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        x_seq = self.embedding(token_seq)
        H, priv_history, final_s = self.ring.forward_sequence(
            x_seq, s_init=s_init, control_mode=control_mode
        )
        logits = self.head(H.view(H.shape[0], -1))
        return logits, priv_history, final_s
