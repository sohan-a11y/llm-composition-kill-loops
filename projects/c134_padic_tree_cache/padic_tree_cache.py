r"""
Non-Archimedean p-adic Ultrametric Tree Cache (C134)
====================================================
Mathematical Formulation:
In language and hierarchical reasoning (syntactic parse trees, call stacks, ASTs),
tokens are organized hierarchically. In Euclidean space \mathbb{R}^d, embedding trees
causes severe metric distortion (Sarkar's theorem).

In the non-Archimedean field of p-adic numbers \mathbb{Q}_p, the p-adic valuation v_p(x)
induces an ultrametric distance:
    d_p(x, y) = |x - y|_p = p^{-v_p(x - y)}
satisfying the Strong Triangle Inequality:
    d_p(x, z) \le \max(d_p(x, y), d_p(y, z))

In this space, every triangle is isosceles, and distances directly reflect lowest
common ancestor depth in the tree. By assigning KV-cache tokens p-adic coordinates
z(t) \in \mathbb{Z}_p, cross-attention logits are gated by:
    A_{i,j} = \frac{q_i k_j^T}{\sqrt{d}} - \lambda d_p(z_i, z_j)
Sibling tokens sharing ancestors at depth h have d_p = p^{-h} \to 0, preserving full
attention flux, while cross-branch distractor subtrees have d_p = 1.0, strictly
suppressing syntactic crosstalk without requiring quadratic tree traversal.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def p_adic_valuation(n: int, p: int = 2) -> int:
    """Computes v_p(n), highest power of prime p dividing integer n."""
    if n == 0:
        return 20  # Numerical infinity cutoff for identical positions
    v = 0
    n = abs(n)
    while n % p == 0:
        v += 1
        n = n // p
    return v


def p_adic_distance(a: int, b: int, p: int = 2) -> float:
    """Computes the p-adic ultrametric distance |a - b|_p."""
    diff = abs(a - b)
    if diff == 0:
        return 0.0
    return float(p ** (-p_adic_valuation(diff, p=p)))


class PAdicUltrametricKernel(nn.Module):
    """
    Computes pairwise p-adic distance matrices for given coordinate vectors.
    Supports:
        - "padic": exact p-adic ultrametric distance d_p(x, y)
        - "archimedean": standard Euclidean/linear distance |x - y| (lethal control)
    """
    def __init__(self, p: int = 2, max_coord: int = 16):
        super().__init__()
        self.p = p
        self.max_coord = max_coord

    def compute_distance_matrix(self, q_coords: List[int], k_coords: List[int], mode: str = "padic") -> torch.Tensor:
        n_q = len(q_coords)
        n_k = len(k_coords)
        dist_mat = torch.zeros(n_q, n_k)
        
        for i, q in enumerate(q_coords):
            for j, k in enumerate(k_coords):
                if mode == "padic":
                    dist_mat[i, j] = p_adic_distance(q, k, p=self.p)
                else: # "archimedean"
                    dist_mat[i, j] = abs(q - k) / float(self.max_coord)
                    
        return dist_mat


class UltrametricTreeAttention(nn.Module):
    """
    Cross-attention layer modulated by p-adic ultrametric tree distances.
    """
    def __init__(
        self,
        d_model: int = 32,
        penalty: float = 10.0,
        p: int = 2,
        mode: str = "padic"
    ):
        super().__init__()
        self.d_model = d_model
        self.penalty = penalty
        self.p = p
        self.mode = mode

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.kernel = PAdicUltrametricKernel(p=p)

        nn.init.eye_(self.q_proj.weight)
        nn.init.eye_(self.k_proj.weight)
        nn.init.eye_(self.v_proj.weight)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_coords: List[int],
        k_coords: List[int],
        override_mode: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (B, N_q, D)
        k: (B, N_k, D)
        v: (B, N_k, D)
        Returns:
            out: (B, N_q, D)
            attn_weights: (B, N_q, N_k)
        """
        B, N_q, D = q.shape
        B, N_k, _ = k.shape

        active_mode = override_mode if override_mode is not None else self.mode
        
        q_p = self.q_proj(q)
        k_p = self.k_proj(k)
        v_p = self.v_proj(v)

        # Base dot-product attention scores
        scores = torch.bmm(q_p, k_p.transpose(1, 2)) / math.sqrt(D)

        # Compute and subtract ultrametric tree distance penalty
        dist_mat = self.kernel.compute_distance_matrix(q_coords, k_coords, mode=active_mode).to(q.device)
        scores = scores - self.penalty * dist_mat.unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.bmm(attn_weights, v_p)
        return out, attn_weights


class PAdicTreeCacheModel(nn.Module):
    """
    End-to-end model for Hierarchical Tree-KV Cache Retrieval.
    """
    def __init__(
        self,
        d_model: int = 32,
        n_classes: int = 5,
        penalty: float = 10.0,
        p: int = 2,
        mode: str = "padic"
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.attention = UltrametricTreeAttention(d_model=d_model, penalty=penalty, p=p, mode=mode)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_coords: List[int],
        k_coords: List[int],
        override_mode: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, weights = self.attention(q, kv, kv, q_coords, k_coords, override_mode=override_mode)
        logits = self.classifier(out.squeeze(1))
        return logits, weights
