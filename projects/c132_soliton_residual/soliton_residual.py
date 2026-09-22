r"""
Soliton Wave-Packet Dispersion-Balanced Residual Stream (C132)
==============================================================
Mathematical Physics Formulation:
In deep unnormalized residual networks (x_{l+1} = x_l + f_l(x_l)), non-linear
activations act like steepening terms (u * \nabla u) in the Korteweg-de Vries (KdV)
equation:
    \partial_t u + u \nabla u + \beta \nabla^3 u + \nu \nabla^4 u = 0
Without dispersion (\beta = 0), non-linear steepening induces shockwave divergence.
Coupling the discrete residual update with third-order spatial dispersion
(-\beta \nabla_d^3 x) and fourth-order biharmonic dissipation (-\nu \nabla_d^4 x)
across hidden channels preserves solitary wave-packet (soliton) stability
across 16+ layers without LayerNorm.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteKdVOperator(nn.Module):
    """
    Discrete 1D KdV Dispersion + Biharmonic Dissipation along channel dimension d:
        \nabla^3 x[d] \approx 0.5*x[d+2] - x[d+1] + x[d-1] - 0.5*x[d-2]  (3rd derivative dispersion)
        \nabla^4 x[d] \approx x[d+2] - 4*x[d+1] + 6*x[d] - 4*x[d-1] + x[d-2] (4th derivative dissipation)
    Periodic boundary padding along hidden channel dimension.
    """
    def __init__(self, d_model: int, beta: float = 0.1, nu: float = 0.03):
        super().__init__()
        self.d_model = d_model
        self.beta = beta
        self.nu = nu

        # 5-point finite difference stencils
        stencil_3 = torch.tensor([0.5, -1.0, 0.0, 1.0, -0.5], dtype=torch.float32)
        stencil_4 = torch.tensor([1.0, -4.0, 6.0, -4.0, 1.0], dtype=torch.float32)
        self.register_buffer("kernel_3", stencil_3.view(1, 1, 5))
        self.register_buffer("kernel_4", stencil_4.view(1, 1, 5))

    def forward(self, x: torch.Tensor, mode: str = "soliton") -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.shape
        x_flat = x.view(B * T, 1, D)
        x_padded = F.pad(x_flat, (2, 2), mode="circular")
        
        nabla3 = F.conv1d(x_padded, self.kernel_3, padding=0).view(B, T, D)
        nabla4 = F.conv1d(x_padded, self.kernel_4, padding=0).view(B, T, D)
        
        if mode == "soliton":
            # Balanced dispersion and dissipation
            return -self.beta * nabla3 - self.nu * nabla4
        elif mode == "anti_dispersion":
            # Lethal control: invert dispersion sign and eliminate dissipation
            return self.beta * nabla3
        else: # "standard"
            return torch.zeros_like(x)


class SolitonResidualBlock(nn.Module):
    """
    Unnormalized residual block with discrete KdV dispersion balancing.
    """
    def __init__(
        self,
        d_model: int,
        beta: float = 0.1,
        nu: float = 0.03,
        mode: str = "soliton"
    ):
        super().__init__()
        self.d_model = d_model
        self.mode = mode

        # Unnormalized MLP (NO LayerNorm)
        self.w1 = nn.Linear(d_model, d_model, bias=False)
        self.w2 = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.GELU()

        self.kdv = DiscreteKdVOperator(d_model, beta=beta, nu=nu)

        nn.init.xavier_normal_(self.w1.weight, gain=0.5)
        nn.init.xavier_normal_(self.w2.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_x = 0.2 * self.w2(self.act(self.w1(x)))
        kdv_term = self.kdv(x, mode=self.mode)
        return x + f_x + kdv_term


class DeepSolitonNetwork(nn.Module):
    """
    Deep unnormalized neural network stack without LayerNorm.
    """
    def __init__(
        self,
        d_model: int = 32,
        n_layers: int = 16,
        n_classes: int = 5,
        beta: float = 0.1,
        nu: float = 0.03,
        mode: str = "soliton"
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.mode = mode

        self.layers = nn.ModuleList([
            SolitonResidualBlock(d_model=d_model, beta=beta, nu=nu, mode=mode)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        # Classify from first token representation
        logits = self.head(h[:, 0, :])
        return logits
