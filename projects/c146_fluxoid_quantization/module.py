"""
Project C146: Superconducting Fluxoid Quantization KV Cache Gating
===================================================================

Mathematical Theory:
-------------------
In Type-II superconductors, the macroscopic Ginzburg-Landau order parameter
psi(r) = |psi(r)| * exp(i * theta(r)) describes the collective condensate of Cooper pairs.
Around any closed loop C containing supercurrent j_s, the single-valuedness of psi
mandates that the phase gradient satisfies the topological fluxoid quantization condition:

    oint_C nabla theta * dl = 2 * pi * n,   n in Z

where n is an integer vortex winding number and Phi_0 = h / (2e) is the magnetic flux quantum.

In Transformer Attention:
-------------------------
In standard dot-product attention, continuous floating-point errors, semantic drift,
and adversarial distractor tokens cause gradual degradation and phase decoherence over long contexts.
Fluxoid Attention equips each token with a topological phase coordinate theta in [-pi, pi].
Attention gating is governed by a Josephson-like pinning potential:

    T(i, j) = exp( - min(|delta_theta - delta_theta_quantized|, 2*pi - |...|)^2 / (2 * sigma_vortex^2) )

where delta_theta_quantized = (2 * pi / N_flux) * round( (delta_theta + pi) / (2 * pi) * N_flux ) - pi.

Tokens satisfying the integer fluxoid condition (n in Z) experience lossless superconducting
transmission (T = 1.0), whereas fractional phase-slip distractors are exponentially quenched,
preventing context pollution.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FluxoidAttention(nn.Module):
    """
    Multi-Head Attention module gated by Superconducting Fluxoid Phase Quantization.
    """
    def __init__(self, d_model=32, n_heads=2, n_flux=4, sigma_vortex=0.15):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_flux = n_flux
        self.sigma_vortex = sigma_vortex
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, q, k, v, phases_q, phases_k, mode="fluxoid"):
        """
        Args:
            q: [B, N_q, D]
            k: [B, N_k, D]
            v: [B, N_k, D]
            phases_q: [B, N_q] or [B, 1, N_q] in [-pi, pi]
            phases_k: [B, N_k] or [B, 1, N_k] in [-pi, pi]
            mode: 'fluxoid', 'standard', 'fractional_slip', 'scrambled_phase'
        """
        B, N_q, D = q.shape
        _, N_k, _ = k.shape
        H = self.n_heads
        d_k = self.head_dim
        
        Q = self.q_proj(q).view(B, N_q, H, d_k).transpose(1, 2) # [B, H, N_q, d_k]
        K = self.k_proj(k).view(B, N_k, H, d_k).transpose(1, 2) # [B, H, N_k, d_k]
        V = self.v_proj(v).view(B, N_k, H, d_k).transpose(1, 2) # [B, H, N_k, d_k]
        
        base_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k) # [B, H, N_q, N_k]
        
        if mode == "standard":
            w = F.softmax(base_logits, dim=-1)
            ctx = torch.matmul(w, V).transpose(1, 2).contiguous().view(B, N_q, D)
            return self.out_proj(ctx), w
            
        theta_q = phases_q.view(B, 1, N_q, 1) # [B, 1, N_q, 1]
        theta_k = phases_k.view(B, 1, 1, N_k) # [B, 1, 1, N_k]
        
        if mode == "scrambled_phase":
            # Lethal Control 2: destroy Cooper-pair coherence by randomizing key phases
            theta_k = torch.rand_like(theta_k) * 2 * math.pi - math.pi
            
        # Angular phase difference delta_theta in [-pi, pi]
        d_theta = theta_q - theta_k # [B, 1, N_q, N_k]
        d_theta = torch.remainder(d_theta + math.pi, 2 * math.pi) - math.pi
        
        # Discrete vortex quantization to nearest integer fluxoid state
        phi = (d_theta + math.pi) / (2 * math.pi) * self.n_flux
        phi_quantized = torch.round(phi) % self.n_flux
        d_theta_quantized = (phi_quantized / self.n_flux) * 2 * math.pi - math.pi
        
        # Shortest arc on circle S^1
        phase_dev = torch.abs(d_theta - d_theta_quantized)
        phase_dev = torch.min(phase_dev, 2 * math.pi - phase_dev)
        
        if mode == "fractional_slip":
            # Lethal Control 1: Anti-vortex mode - transmit ONLY on fractional phase slip boundaries
            half_step = math.pi / self.n_flux
            phase_dev_slip = torch.abs(phase_dev - half_step)
            T = torch.exp(- (phase_dev_slip ** 2) / (2 * (self.sigma_vortex ** 2)))
        else:
            # Mode "fluxoid": Josephson vortex pinning potential
            T = torch.exp(- (phase_dev ** 2) / (2 * (self.sigma_vortex ** 2)))
            
        # Gate attention logits with topological fluxoid transmission
        gated_logits = base_logits + torch.log(T + 1e-6)
        w = F.softmax(gated_logits, dim=-1)
        ctx = torch.matmul(w, V).transpose(1, 2).contiguous().view(B, N_q, D)
        return self.out_proj(ctx), w

class FluxoidHaystackClassifier(nn.Module):
    """
    End-to-End Classifier evaluating topological passkey retrieval in an adversarial distractor haystack.
    """
    def __init__(self, d_model=32, n_heads=2, n_classes=5, n_flux=4, sigma_vortex=0.15):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.attn = FluxoidAttention(d_model=d_model, n_heads=n_heads, n_flux=n_flux, sigma_vortex=sigma_vortex)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes)
        )
        
    def forward(self, x, phases, mode="fluxoid"):
        """
        x: [B, L, D]
        phases: [B, L] in [-pi, pi]
        Last token (L-1) is the query probe attending to tokens 0..L-2.
        """
        q = x[:, -1:, :]
        k = x[:, :-1, :]
        v = x[:, :-1, :]
        
        pq = phases[:, -1:]
        pk = phases[:, :-1]
        
        ctx, _ = self.attn(q, k, v, pq, pk, mode=mode)
        logits = self.classifier(ctx.squeeze(1))
        return logits
