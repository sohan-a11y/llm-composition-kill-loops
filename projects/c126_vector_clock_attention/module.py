"""
Vector-Clock Asynchronous Causal Attention (C126)
================================================

Mathematical Foundation:
In distributed systems, Lamport vector clocks provide a sound and complete representation
of the 'happens-before' strict partial order (Fidge-Mattern causality). In multi-stream,
branching, or speculative LLM inference (e.g. Tree-of-Thoughts, multi-agent debate, concurrent
speculative decoding), linear serial token ordering imposes artificial synchronization barriers.

Vector-Clock Causal Attention assigns each token x_i in stream s_i a vector clock V(x_i) in N^B,
where B is the number of concurrent asynchronous streams.
1. Local generation event on stream s_i:
   V(x_i)[s_i] = V(prev_{s_i})[s_i] + 1
2. Asynchronous cross-stream message receive event from stream m:
   V(x_i) = max(V(prev_{s_i}), V(x_m)) + e_{s_i}

The Vector-Clock Causal Attention Mask M in {0, -inf}^{N x N} is defined by the partial order:
   x_j <= x_i  iff  forall b in [B]: V(x_j)[b] <= V(x_i)[b]
   M_{i,j} = 0.0 if x_j <= x_i else -inf

This guarantees:
1. Complete causal safety: Tokens can never attend to uncommitted future states.
2. Concurrency isolation: Concurrent branches (incomparable vector clocks x_i || x_j)
   are mutually isolated (M_{i,j} = M_{j,i} = -inf), eliminating race conditions without global locks.
3. Instantaneous causal merge: Upon receiving an asynchronous message, all causal predecessors
   become visible in O(1) mask evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VectorClockCausalMask(nn.Module):
    """
    Computes the Vector-Clock Happens-Before Causal Mask.
    Given V of shape (batch, N, B), returns mask of shape (batch, 1, N, N).
    """
    def __init__(self, mask_value: float = -1e9):
        super().__init__()
        self.mask_value = mask_value

    def forward(
        self,
        V: torch.Tensor,
        control_mode: str = "normal"
    ) -> torch.Tensor:
        """
        Args:
            V: (batch, N, B) integer or float vector clocks.
            control_mode: 'normal', 'kill_distractor_channel', or 'kill_scrambled'
        Returns:
            mask: (batch, 1, N, N) float tensor with 0.0 for causally allowed pairs, mask_value elsewhere.
        """
        B_sz, N, B_dim = V.shape
        device = V.device

        if control_mode == "kill_scrambled":
            # Scramble clock assignments randomly across token positions
            perm = torch.randperm(N, device=device)
            V_used = V[:, perm, :]
        elif control_mode == "kill_distractor_channel":
            # Lethal control: deliberately violate vector-clock isolation by allowing query
            # to attend only to concurrent distractor streams (indices 1, 3) and masking out
            # true causal predecessors
            mask = torch.full((B_sz, 1, N, N), self.mask_value, device=device)
            # Query attends only to distractors (indices 1 and 3)
            mask[:, 0, -1, 1] = 0.0
            mask[:, 0, -1, 3] = 0.0
            # Self-attention for other tokens
            for idx in range(N):
                mask[:, 0, idx, idx] = 0.0
            return mask
        else:
            V_used = V

        # Vectorized happens-before test:
        # V_j has shape (batch, 1, N, B_dim) -> key
        # V_i has shape (batch, N, 1, B_dim) -> query
        V_j = V_used.unsqueeze(1)
        V_i = V_used.unsqueeze(2)

        # Token j happened before or is token i iff V(x_j)[b] <= V(x_i)[b] for all streams b
        causal_pred = (V_j <= V_i).all(dim=-1) # (batch, N, N)

        mask = torch.where(causal_pred, 0.0, self.mask_value)
        return mask.unsqueeze(1) # (batch, 1, N, N)


class VectorClockAttention(nn.Module):
    """
    Multi-Head Attention parameterized with Vector-Clock Causal Masking.
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.vc_masker = VectorClockCausalMask()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        V: torch.Tensor,
        control_mode: str = "normal"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, N, d_model) input embeddings.
            V: (batch, N, B) vector clock timestamps.
            control_mode: 'normal', 'kill_distractor_channel', or 'kill_scrambled'
        Returns:
            out: (batch, N, d_model) output activations.
            attn_weights: (batch, n_heads, N, N) attention probabilities.
        """
        B_sz, N, D = x.shape

        mask = self.vc_masker(V, control_mode=control_mode)

        Q = self.q_proj(x).view(B_sz, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B_sz, N, self.n_heads, self.d_k).transpose(1, 2)
        Val = self.v_proj(x).view(B_sz, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, Val)
        out = out.transpose(1, 2).contiguous().view(B_sz, N, D)
        return self.out_proj(out), attn_weights


class VectorClockTransformerBlock(nn.Module):
    """
    Single Transformer Layer combining VectorClockAttention with FeedForward Network.
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 2

        self.attn = VectorClockAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        V: torch.Tensor,
        control_mode: str = "normal"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(h_norm, V, control_mode=control_mode)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


class VectorClockReasoningModel(nn.Module):
    """
    Complete Asynchronous Multi-Stream Reasoning Transformer using Vector Clocks.
    """
    def __init__(
        self,
        num_tokens: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.layers = nn.ModuleList([
            VectorClockTransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        token_ids: torch.Tensor,
        V: torch.Tensor,
        control_mode: str = "normal"
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, N)
            V: (batch, N, B)
            control_mode: 'normal', 'kill_distractor_channel', or 'kill_scrambled'
        Returns:
            logits: (batch, num_classes) for the final query token.
        """
        h = self.embedding(token_ids)
        for layer in self.layers:
            h, _ = layer(h, V, control_mode=control_mode)
        h = self.norm(h)
        logits = self.head(h[:, -1, :])
        return logits
