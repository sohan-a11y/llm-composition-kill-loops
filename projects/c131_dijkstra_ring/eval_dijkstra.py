"""
Meta-Loop Integration Interface for C131: Self-Stabilizing Dijkstra-Ring Virtual Token Memory.

Provides standard callable hooks for engine.continuous_meta_loop:
1. evaluate_dijkstra(config, seed) -> Dict[str, float]
2. sanity_check_dijkstra(config) -> bool
3. kill_control_dijkstra(config) -> Tuple[bool, float, float]
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple

from projects.c131_dijkstra_ring.module import DijkstraMemoryClassifier
from projects.c131_dijkstra_ring.evaluate import generate_task_data



def evaluate_dijkstra(config: Dict[str, Any], seed: int = 42) -> Dict[str, float]:
    d_model = config.get("d_model", 32)
    K = config.get("K", 4)
    M = config.get("M", 5)
    num_classes = config.get("num_classes", 5)
    epochs = config.get("epochs", 80)
    batch_size = config.get("batch_size", 64)
    seq_len = config.get("seq_len", 6)
    lr = config.get("lr", 0.01)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DijkstraMemoryClassifier(
        num_tokens=20,
        d_model=d_model,
        num_classes=num_classes,
        K=K,
        M=M
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        tokens, targets = generate_task_data(batch_size=batch_size, num_classes=num_classes, seq_len=seq_len)
        if epoch % 2 == 0:
            s_init = torch.randint(0, M, (batch_size, K))
        else:
            s_init = None

        optimizer.zero_grad()
        logits, _, _ = model(tokens, s_init=s_init, control_mode="normal")
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        tokens, targets = generate_task_data(batch_size=300, num_classes=num_classes, seq_len=seq_len)
        s_corrupt = torch.randint(0, M, (300, K))
        logits, priv_hist, _ = model(tokens, s_init=s_corrupt, control_mode="normal")
        acc = (logits.argmax(dim=-1) == targets).float().mean().item()
        final_priv = priv_hist[-1].sum(dim=-1)
        mutex_conv = (final_priv == 1).float().mean().item()

    return {"acc": acc, "mutex_conv": mutex_conv}


def sanity_check_dijkstra(config: Dict[str, Any]) -> bool:
    """Sanity Check: Clean execution from legitimate state must achieve >= 0.85."""
    d_model = config.get("d_model", 32)
    K = config.get("K", 4)
    M = config.get("M", 5)
    num_classes = config.get("num_classes", 5)
    seq_len = config.get("seq_len", 6)

    torch.manual_seed(999)
    model = DijkstraMemoryClassifier(
        num_tokens=20,
        d_model=d_model,
        num_classes=num_classes,
        K=K,
        M=M
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(60):
        tokens, targets = generate_task_data(batch_size=64, num_classes=num_classes, seq_len=seq_len)
        optimizer.zero_grad()
        logits, _, _ = model(tokens, s_init=None, control_mode="normal")
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        tokens, targets = generate_task_data(batch_size=200, num_classes=num_classes, seq_len=seq_len)
        logits, _, _ = model(tokens, s_init=None, control_mode="normal")
        sanity_acc = (logits.argmax(dim=-1) == targets).float().mean().item()

    print(f"   [Sanity Check] Clean Execution Accuracy: {sanity_acc:.4f} (Threshold: >= 0.85)")
    return sanity_acc >= 0.85


def kill_control_dijkstra(config: Dict[str, Any]) -> Tuple[bool, float, float]:
    """
    Lethal Negative Control: Freeze leader privilege (Anti-Dijkstra deadlock).
    Accuracy MUST collapse to chance baseline (1/5 = 0.2000).
    """
    d_model = config.get("d_model", 32)
    K = config.get("K", 4)
    M = config.get("M", 5)
    num_classes = config.get("num_classes", 5)
    seq_len = config.get("seq_len", 6)

    torch.manual_seed(777)
    model = DijkstraMemoryClassifier(
        num_tokens=20,
        d_model=d_model,
        num_classes=num_classes,
        K=K,
        M=M
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(60):
        tokens, targets = generate_task_data(batch_size=64, num_classes=num_classes, seq_len=seq_len)
        if epoch % 2 == 0:
            s_init = torch.randint(0, M, (64, K))
        else:
            s_init = None
        optimizer.zero_grad()
        logits, _, _ = model(tokens, s_init=s_init, control_mode="normal")
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        tokens, targets = generate_task_data(batch_size=400, num_classes=num_classes, seq_len=seq_len)
        s_corrupt = torch.randint(0, M, (400, K))
        d_logits, _, _ = model(tokens, s_init=s_corrupt, control_mode="kill_deadlock")
        acc = (d_logits.argmax(dim=-1) == targets).float().mean().item()

    chance_baseline = 0.2000
    passed = acc <= 0.35
    print(f"   [Kill Control] Deadlock Mode Acc: {acc:.4f} (Chance: {chance_baseline:.4f})")
    return passed, acc, chance_baseline


if __name__ == "__main__":
    cfg = {"d_model": 32, "K": 4, "M": 5, "num_classes": 5, "epochs": 80, "seq_len": 6}
    print("Sanity Check:", sanity_check_dijkstra(cfg))
    print("Eval seed 42:", evaluate_dijkstra(cfg, seed=42))
    print("Kill Control:", kill_control_dijkstra(cfg))
