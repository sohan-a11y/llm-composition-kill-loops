"""
C029: Static Single Assignment (SSA) Residual Transformer
Partitions the residual stream into L orthogonal virtual register banks.
Layer l reads all channels (past + present), but writes STRICTLY to slice l.
Intermediate feature values are preserved without destructive overwriting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SSABlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, layer_idx: int, total_layers: int, ssa_mode: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.ssa_mode = ssa_mode

        self.slice_dim = d_model // total_layers
        self.slice_start = layer_idx * self.slice_dim
        self.slice_end = (layer_idx + 1) * self.slice_dim

        self.n1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)

        if self.ssa_mode:
            # Output of block projects directly into the dedicated register slice
            self.ff = nn.Sequential(
                nn.Linear(d_model, 4 * self.slice_dim),
                nn.GELU(),
                nn.Linear(4 * self.slice_dim, self.slice_dim)
            )
            # Projection for attention delta into register slice
            self.attn_proj = nn.Linear(d_model, self.slice_dim)
        else:
            # Standard additive transformer block
            self.ff = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, scramble_slice: bool = False) -> torch.Tensor:
        h = self.n1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)

        if self.ssa_mode:
            a_proj = self.attn_proj(a)
            h2 = self.n2(x)
            ff_out = self.ff(h2)
            delta = a_proj + ff_out  # (B, T, slice_dim)

            # Write-once update into dedicated register bank
            out = x.clone()
            if scramble_slice:
                # Kill-control arm: write into a randomly chosen incorrect layer's slice
                target_start = ((self.layer_idx + 1) % self.total_layers) * self.slice_dim
                target_end = target_start + self.slice_dim
                out[:, :, target_start:target_end] = delta
            else:
                out[:, :, self.slice_start:self.slice_end] = delta
            return out
        else:
            # Standard additive residual stream
            x = x + a
            return x + self.ff(self.n2(x))


class SSATransformerLM(nn.Module):
    def __init__(
        self,
        vocab: int,
        d_model: int = 64,
        layers: int = 4,
        heads: int = 4,
        maxlen: int = 64,
        ssa_mode: bool = True
    ):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.layers = layers
        self.ssa_mode = ssa_mode

        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(maxlen, d_model)
        self.blocks = nn.ModuleList([
            SSABlock(d_model, heads, l, layers, ssa_mode=ssa_mode)
            for l in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx: torch.Tensor, scramble_control: bool = False) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)[None]
        x = self.tok(idx) + self.pos(pos)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=idx.device), 1)

        for b in self.blocks:
            x = b(x, mask, scramble_slice=scramble_control)

        x = self.norm(x)
        return self.head(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
