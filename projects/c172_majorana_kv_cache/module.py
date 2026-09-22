"""
Project C172: Majorana Zero-Mode Non-Abelian Anyon Braiding KV Cache (MZM-KV Cache).

Mathematical Formulation:
In topological quantum condensed matter physics (Kitaev 2001, Ivanov 2001, Nayak et al. 2008),
a set of 2N Majorana fermion zero modes gamma_1, ..., gamma_{2N} satisfy:
    {gamma_i, gamma_j} = 2 delta_{i, j} I,   gamma_i^dagger = gamma_i
Non-local complex Dirac fermion creation/annihilation operators are defined as:
    c_k = 0.5 * (gamma_{2k-1} + i * gamma_{2k})
with occupation number n_k = c_k^dagger c_k in {0, 1} and topological parity:
    P_k = (-1)^{n_k} = i * gamma_{2k-1} gamma_{2k} in {-1, +1}

Exchanging (braiding) adjacent Majorana modes applies the non-Abelian unitary braid operator:
    tau_i = exp(pi/4 * gamma_i gamma_{i+1}) = 1/sqrt(2) * (I + gamma_i gamma_{i+1})

In Transformer KV Caches:
Standard KV caches store key representations locally in Euclidean space R^d, making them
vulnerable to local adversarial distractor perturbations and long-context drift.
Majorana KV Cache pairs tokens into non-local topological qubits and tracks discrete
braid sector windings theta_braid in Z_4. The non-local parity correlator is:
    P_{ij} = cos((theta_i - theta_j) * pi / 2)
The attention logits are topologically modulated by:
    S_{ij} = (Q_i K_j^T) / sqrt(d_k)
    S~_{ij} = S_{ij} - lambda_topological * (1.0 - P_{ij})
    A_{ij} = softmax(S~_{ij})

Keys with matching topological braid sectors receive zero penalty (P_{ij} = 1.0),
while adversarial distractor keys with orthogonal or mismatched braid sectors are exponentially
quenched (P_{ij} <= 0 -> S~ receives heavy topological suppression).

Lethal Kill Controls:
1. Quasi-Particle Poisoning (Fermion Parity Flip):
   Physically models environmental quasi-particle poisoning flipping Majorana parity:
   S~_{ij} = S_{ij} - lambda_topological * (1.0 + P_{ij})
   Inverts the topological sector, destroying target needle retrieval and collapsing
   accuracy strictly to random chance (0.2000).
2. Abelian Phase Scrambling:
   Destroys non-Abelian topological coherence by randomizing braid sector phases.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MajoranaKVAttentionLayer(nn.Module):
    def __init__(self, d_model=32, n_heads=2, lambda_topological=30.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.lambda_topological = lambda_topological
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def compute_braid_phase(self, braid_indices):
        """
        Computes non-local Majorana fermion parity correlation.
        braid_indices: [B, L] discrete braid winding sector in {0, 1, 2, 3} (mod 4)
        Query is at position L-1. Preceding keys are at 0 .. L-2.
        
        Returns:
            braid_parity: [B, L-1] continuous topological correlation in [-1.0, 1.0]
        """
        B, L = braid_indices.shape
        q_braid = braid_indices[:, -1:] # [B, 1]
        k_braid = braid_indices[:, :-1] # [B, L-1]
        
        diff = (k_braid - q_braid) % 4
        braid_parity = torch.cos(diff.float() * (math.pi / 2.0))
        return braid_parity

    def forward(self, x, braid_indices, mode="majorana"):
        """
        Forward pass with configurable topological braid gating.
        
        Args:
            x: [B, L, D]
            braid_indices: [B, L]
            mode: 'majorana', 'standard', 'poisoned_parity', 'abelian_scramble'
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
            
        braid_parity = self.compute_braid_phase(braid_indices) # [B, L-1]
        gate = braid_parity.unsqueeze(1).unsqueeze(2) # [B, 1, 1, L-1]
        
        if mode == "majorana":
            # Majorana Non-Abelian Topological Gating:
            # Preserves matching parity sector, quenches orthogonal sectors
            penalty = (1.0 - gate) * self.lambda_topological
            gated_logits = logits - penalty
        elif mode == "poisoned_parity":
            # Lethal Control 1: Quasi-Particle Poisoning (inverts fermion parity)
            penalty = (1.0 + gate) * self.lambda_topological
            gated_logits = logits - penalty
        elif mode == "abelian_scramble":
            # Lethal Control 2: Scrambled Braid Sectors
            rand_gate = torch.randn_like(gate)
            penalty = (1.0 - rand_gate) * self.lambda_topological
            gated_logits = logits - penalty
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        attn = F.softmax(gated_logits, dim=-1)
        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, D)
        return self.out_proj(ctx)

class MajoranaKVClassifier(nn.Module):
    def __init__(self, d_model=32, n_heads=2, n_classes=5, lambda_topological=30.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_classes = n_classes
        
        self.attn = MajoranaKVAttentionLayer(
            d_model=d_model, n_heads=n_heads, lambda_topological=lambda_topological
        )
        self.classifier = nn.Linear(d_model, n_classes, bias=False)
        
    def forward(self, x, braid_indices, mode="majorana"):
        ctx = self.attn(x, braid_indices, mode=mode)
        return self.classifier(ctx)
