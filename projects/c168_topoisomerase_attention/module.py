"""
Project C168: DNA Topoisomerase-II Strand-Passage Attention Unknotting.

Mathematical Formulation:
In DNA topology and polymer physics (Wang 2002, Bates & Maxwell 2005),
Type-II topoisomerases manage topological supercoiling, catenanes, and knots.
A knot or self-intersection in a sequence reasoning trajectory z_0, ..., z_{L-1}
occurs when non-adjacent steps (|i - j| >= 2) cross in Euclidean latent space:
    C_{i, j} = Theta(delta_knot - ||z_i - z_j||_2) * I(|i - j| >= 2)

In standard attention, when the query z_{L-1} becomes entangled with an intermediate
knot attractor z_k (k in [1, L-3]), dot-product attention gets trapped in the crossing,
retrieving distractor features instead of the root premise z_0.

Topoisomerase-II Attention resolves topological entrapment via a strand-passage cut-and-rejoin
cleavage operation:
    S_{i, j} = (Q_i K_j^T) / sqrt(d_k)
    S~_{i, j} = S_{i, j} - lambda_strand * C_{i, j}
    A_{i, j} = softmax(S~_{i, j})

This cleaves the self-intersecting knot, unlinking the entangled loop and allowing
attention to freely reach the root premise.

Lethal Kill Controls:
1. Anti-Topoisomerase / Topo-Poison (Etoposide/Teniposide analogue):
   Covalently traps the cleavage complex into a permanent knot lock:
   S~_{i, j} = S_{i, j} + lambda_strand * C_{i, j}
   Forces attention 100% into the distractor knot, collapsing accuracy to chance (0.2000).
2. Scrambled Strand-Breaker:
   Indiscriminately cleaves arbitrary strands in the trajectory regardless of crossing
   topology, collapsing logical continuity to chance (0.2000).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopoisomeraseAttentionLayer(nn.Module):
    def __init__(self, d_model=32, n_heads=2, delta_knot=1.5, lambda_strand=40.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.delta_knot = delta_knot
        self.lambda_strand = lambda_strand
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def detect_crossings(self, x):
        """
        Detects non-adjacent topological crossings between query (L-1) and intermediate loop steps.
        Root premise is at index 0. Intermediate steps are 1 .. L-3.
        Crossing occurs if ||x_{L-1} - x_j||_2 < delta_knot for j in [1, L-3].
        
        Args:
            x: [B, L, D]
        Returns:
            crossings: [B, L-1] binary tensor indicating detected crossings
        """
        B, L, D = x.shape
        q_pos = x[:, -1:, :] # [B, 1, D]
        k_pos = x[:, :-1, :] # [B, L-1, D]
        dist = torch.norm(q_pos - k_pos, p=2, dim=-1) # [B, L-1]
        
        crossings = torch.zeros(B, L - 1, device=x.device)
        if L > 3:
            # Check intermediate non-adjacent positions 1 .. L-3
            crossings[:, 1:-1] = (dist[:, 1:-1] < self.delta_knot).float()
        return crossings

    def forward(self, x, mode="topoisomerase"):
        """
        Forward pass with configurable unknotting mode.
        
        Args:
            x: [B, L, D]
            mode: 'topoisomerase', 'standard', 'anti_topo', or 'scrambled_topo'
        Returns:
            context: [B, D]
        """
        B, L, D = x.shape
        H = self.n_heads
        d_k = self.head_dim
        
        q = x[:, -1:, :]
        k = x[:, :-1, :]
        v = x[:, :-1, :]
        
        Q = self.q_proj(q).view(B, 1, H, d_k).transpose(1, 2)
        K = self.k_proj(k).view(B, L-1, H, d_k).transpose(1, 2)
        V = self.v_proj(v).view(B, L-1, H, d_k).transpose(1, 2)
        
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k) # [B, H, 1, L-1]
        
        if mode == "standard":
            attn = F.softmax(logits, dim=-1)
            ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, D)
            return self.out_proj(ctx)
            
        crossings = self.detect_crossings(x) # [B, L-1]
        cross_gate = crossings.unsqueeze(1).unsqueeze(2) # [B, 1, 1, L-1]
        
        if mode == "topoisomerase":
            # Type-II Strand Passage: cleave through topological knot
            gated_logits = logits - cross_gate * self.lambda_strand
        elif mode == "anti_topo":
            # Lethal Control 1: Topo-Poison locks the knot crossing
            gated_logits = logits + cross_gate * self.lambda_strand
        elif mode == "scrambled_topo":
            # Lethal Control 2: Random strand cleavage across arbitrary keys
            rand_gate = (torch.rand(B, 1, 1, L - 1, device=x.device) > 0.5).float()
            gated_logits = logits - rand_gate * self.lambda_strand
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        attn = F.softmax(gated_logits, dim=-1)
        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, D)
        return self.out_proj(ctx)

class TopoisomeraseUnknottingClassifier(nn.Module):
    def __init__(self, d_model=32, n_heads=2, n_classes=5, delta_knot=1.5, lambda_strand=40.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_classes = n_classes
        
        self.attn = TopoisomeraseAttentionLayer(
            d_model=d_model, n_heads=n_heads, delta_knot=delta_knot, lambda_strand=lambda_strand
        )
        self.classifier = nn.Linear(d_model, n_classes, bias=False)
        
    def forward(self, x, mode="topoisomerase"):
        ctx = self.attn(x, mode=mode)
        return self.classifier(ctx)
