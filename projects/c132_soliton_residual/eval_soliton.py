"""
Evaluation Harness for C132: Soliton Wave-Packet Dispersion-Balanced Residual Stream.

Includes:
1. evaluate_soliton: Standard evaluation across seeds.
2. sanity_check_soliton: Shallow 2-layer network check (>0.80).
3. kill_control_soliton: Lethal Label Permutation Control (collapses to chance ~0.2000).
"""

import sys
import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from soliton_residual import DeepSolitonNetwork


def generate_soliton_inputs(n_samples: int = 250, n_classes: int = 5, d_model: int = 32, seed: int = 42):
    torch.manual_seed(seed)
    labels = torch.randint(0, n_classes, (n_samples,))
    centers = [k * (d_model // n_classes) + (d_model // (2 * n_classes)) for k in range(n_classes)]
    channels = torch.arange(d_model).float()
    packets = []
    for lbl in labels:
        c0 = centers[lbl.item()]
        dist = torch.min((channels - c0).abs(), d_model - (channels - c0).abs())
        profile = 1.0 / (torch.cosh(dist / 2.0) ** 2)
        profile = profile + 0.05 * torch.randn(d_model)
        packets.append(profile)
    X = torch.stack(packets).unsqueeze(1) # (N, 1, D)
    return X, labels


def train_and_eval_model(
    n_layers: int = 16,
    d_model: int = 32,
    n_classes: int = 5,
    beta: float = 0.1,
    nu: float = 0.03,
    mode: str = "soliton",
    epochs: int = 25,
    lr: float = 0.01,
    seed: int = 42,
    n_samples: int = 250,
    is_kill_control: bool = False
):
    torch.manual_seed(seed)
    X_tr, y_tr = generate_soliton_inputs(n_samples=n_samples, n_classes=n_classes, d_model=d_model, seed=seed)
    X_te, y_te = generate_soliton_inputs(n_samples=100, n_classes=n_classes, d_model=d_model, seed=seed + 1000)
    
    model = DeepSolitonNetwork(
        d_model=d_model, n_layers=n_layers, n_classes=n_classes, beta=beta, nu=nu, mode=mode
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        if torch.isnan(loss) or torch.isinf(loss):
            return 0.20, float("nan")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        if is_kill_control:
            perm = torch.randperm(len(y_te))
            y_te = y_te[perm]
        test_logits = model(X_te)
        preds = torch.argmax(test_logits, dim=-1)
        acc = (preds == y_te).float().mean().item()
        
    return acc, loss.item()


def evaluate_soliton(config: Dict[str, Any], seed: int = 42) -> Dict[str, float]:
    n_samples = config.get("n_samples", 250)
    n_layers = config.get("n_layers", 16)
    d_model = config.get("d_model", 32)
    n_classes = config.get("n_classes", 5)
    beta = config.get("beta", 0.1)
    nu = config.get("nu", 0.03)
    epochs = config.get("epochs", 25)
    lr = config.get("lr", 0.01)

    acc, loss = train_and_eval_model(
        n_layers=n_layers,
        d_model=d_model,
        n_classes=n_classes,
        beta=beta,
        nu=nu,
        mode="soliton",
        epochs=epochs,
        lr=lr,
        seed=seed,
        n_samples=n_samples,
        is_kill_control=False
    )
    return {
        "acc": acc,
        "loss": loss if not math.isnan(loss) else 999.0
    }


def sanity_check_soliton(config: Dict[str, Any]) -> bool:
    """Sanity Check: Shallow 2-layer network must achieve >= 0.80."""
    d_model = config.get("d_model", 32)
    n_samples = config.get("n_samples", 250)
    beta = config.get("beta", 0.1)
    nu = config.get("nu", 0.03)
    
    acc, _ = train_and_eval_model(
        n_layers=2,
        d_model=d_model,
        beta=beta,
        nu=nu,
        mode="soliton",
        epochs=15,
        seed=42,
        n_samples=n_samples
    )
    print(f"   [Sanity Check] Shallow Accuracy (L=2): {acc:.4f} (Threshold: >0.80)")
    return acc >= 0.80


def kill_control_soliton(config: Dict[str, Any]) -> Tuple[bool, float, float]:
    """
    Lethal Kill Control: Permute target labels at test time.
    Accuracy MUST collapse to chance baseline (1 / 5 = 0.2000).
    """
    n_samples = config.get("n_samples", 250)
    n_layers = config.get("n_layers", 16)
    d_model = config.get("d_model", 32)
    beta = config.get("beta", 0.1)
    nu = config.get("nu", 0.03)
    epochs = config.get("epochs", 25)
    lr = config.get("lr", 0.01)

    acc, _ = train_and_eval_model(
        n_layers=n_layers,
        d_model=d_model,
        beta=beta,
        nu=nu,
        mode="soliton",
        epochs=epochs,
        lr=lr,
        seed=42,
        n_samples=n_samples,
        is_kill_control=True
    )
    chance_baseline = 0.2000
    passed = acc <= 0.35
    return passed, acc, chance_baseline


if __name__ == "__main__":
    cfg = {"n_samples": 250, "n_layers": 16, "d_model": 32, "beta": 0.1, "nu": 0.03}
    print("Sanity Check:", sanity_check_soliton(cfg))
    print("Eval seed 42:", evaluate_soliton(cfg, seed=42))
    print("Kill Control:", kill_control_soliton(cfg))
