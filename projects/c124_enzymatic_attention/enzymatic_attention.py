"""
C124: Michaelis-Menten Enzymatic Substrate Attention
====================================================
Replaces unconstrained Softmax attention with Michaelis-Menten catalytic
substrate kinetics. Each key token acts as a saturable enzyme binding site:

    v_{i,j} = (V_max * S_{i,j}) / (K_m + S_{i,j})

where S_{i,j} = exp(clamp(q_i^T k_j / sqrt(d), -20, 20)) represents raw substrate affinity,
V_max is the catalytic saturation ceiling, and K_m is the Michaelis affinity constant.

Attention weights are normalized over enzymatic flux:
    alpha_{i,j} = v_{i,j} / sum_m v_{i,m}

This bounds the maximum attention mass that dominant attention sinks (e.g. <BOS>,
frequent delimiters) can monopolize, preventing semantic token starvation in
long-context and distractor-heavy composition tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EnzymaticMultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4, km=1.0, vmax=1.0, is_kill_control=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.is_kill_control = is_kill_control

        # Projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

        # Enzymatic parameters (per-head learnable or fixed)
        self.km = nn.Parameter(torch.full((1, n_heads, 1, 1), float(km)))
        self.vmax = nn.Parameter(torch.full((1, n_heads, 1, 1), float(vmax)))

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Linear projections & split into heads
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Dot-product raw affinities
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        if self.is_kill_control:
            # Lethal Kill-Control: Invert Michaelis kinetics (K_m -> inf, V_max -> 0)
            # Flattening all velocity differentials to uniform chance baseline
            attn_weights = torch.ones_like(scores) / seq_len
        else:
            # Substrate affinity S: unbounded positive affinity via softplus
            s = F.softplus(scores) + 1e-4

            # Michaelis-Menten Catalytic Rate Equation:
            # v = (V_max * S) / (K_m + S)
            km_clamped = F.softplus(self.km) + 1e-4
            vmax_clamped = F.softplus(self.vmax) + 1e-4
            v_flux = (vmax_clamped * s) / (km_clamped + s) + 1e-6

            if mask is not None:
                v_flux = v_flux.masked_fill(mask == 0, 0.0)

            # Conserve probability over enzymatic flux
            flux_sum = torch.sum(v_flux, dim=-1, keepdim=True) + 1e-8
            attn_weights = v_flux / flux_sum

        # Context output
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_out(context), attn_weights


class EnzymaticTransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff=128, km=1.0, vmax=1.0, is_kill_control=False):
        super().__init__()
        self.attn = EnzymaticMultiHeadAttention(d_model, n_heads, km=km, vmax=vmax, is_kill_control=is_kill_control)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attn(self.ln1(x), mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights
