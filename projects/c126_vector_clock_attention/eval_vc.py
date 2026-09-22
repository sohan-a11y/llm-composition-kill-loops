"""
Meta-Loop Integration Interface for C126: Vector-Clock Asynchronous Causal Attention.

Provides standard callable hooks for engine.continuous_meta_loop:
1. evaluate_vc(config, seed) -> Dict[str, float]
2. sanity_check_vc(config) -> bool
3. kill_control_vc(config) -> Tuple[bool, float, float]
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from module import VectorClockReasoningModel
from evaluate import generate_multistream_data, generate_serial_sanity_data


def evaluate_vc(config: Dict[str, Any], seed: int = 42) -> Dict[str, float]:
    d_model = config.get("d_model", 32)
    n_heads = config.get("n_heads", 2)
    num_layers = config.get("num_layers", 2)
    num_classes = config.get("num_classes", 5)
    epochs = config.get("epochs", 80)
    batch_size = config.get("batch_size", 64)
    lr = config.get("lr", 0.01)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = VectorClockReasoningModel(
        num_tokens=20,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        num_classes=num_classes
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        tokens, V, targets = generate_multistream_data(batch_size=batch_size, num_classes=num_classes)
        optimizer.zero_grad()
        logits = model(tokens, V, control_mode="normal")
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        eval_tokens, eval_V, eval_targets = generate_multistream_data(batch_size=300, num_classes=num_classes)
        logits = model(eval_tokens, eval_V, control_mode="normal")
        acc = (logits.argmax(dim=-1) == eval_targets).float().mean().item()

    return {"acc": acc}


def sanity_check_vc(config: Dict[str, Any]) -> bool:
    """Sanity Check: Serial execution (B=1) with standard causal ordering must achieve >= 0.85."""
    d_model = config.get("d_model", 32)
    n_heads = config.get("n_heads", 2)
    num_layers = config.get("num_layers", 2)
    num_classes = config.get("num_classes", 5)

    torch.manual_seed(999)
    model = VectorClockReasoningModel(
        num_tokens=20,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        num_classes=num_classes
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(80):
        tokens, V, targets = generate_multistream_data(batch_size=64, num_classes=num_classes)
        optimizer.zero_grad()
        logits = model(tokens, V, control_mode="normal")
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        s_tokens, s_V, s_targets = generate_serial_sanity_data(batch_size=200, num_classes=num_classes)
        s_logits = model(s_tokens, s_V, control_mode="normal")
        sanity_acc = (s_logits.argmax(dim=-1) == s_targets).float().mean().item()

    print(f"   [Sanity Check] Serial B=1 Accuracy: {sanity_acc:.4f} (Threshold: >= 0.85)")
    return sanity_acc >= 0.85


def kill_control_vc(config: Dict[str, Any]) -> Tuple[bool, float, float]:
    """
    Lethal Negative Control: Violate vector-clock concurrency isolation (distractor channel leakage).
    Accuracy MUST collapse to chance baseline (1/5 = 0.2000).
    """
    d_model = config.get("d_model", 32)
    n_heads = config.get("n_heads", 2)
    num_layers = config.get("num_layers", 2)
    num_classes = config.get("num_classes", 5)

    torch.manual_seed(777)
    model = VectorClockReasoningModel(
        num_tokens=20,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        num_classes=num_classes
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(80):
        tokens, V, targets = generate_multistream_data(batch_size=64, num_classes=num_classes)
        optimizer.zero_grad()
        logits = model(tokens, V, control_mode="normal")
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        tokens, V, targets = generate_multistream_data(batch_size=400, num_classes=num_classes)
        d_logits = model(tokens, V, control_mode="kill_distractor_channel")
        acc = (d_logits.argmax(dim=-1) == targets).float().mean().item()

    chance_baseline = 0.2000
    passed = acc <= 0.35
    print(f"   [Kill Control] Distractor Leakage Acc: {acc:.4f} (Chance: {chance_baseline:.4f})")
    return passed, acc, chance_baseline


if __name__ == "__main__":
    cfg = {"d_model": 32, "n_heads": 2, "num_layers": 2, "num_classes": 5, "epochs": 80}
    print("Sanity Check:", sanity_check_vc(cfg))
    print("Eval seed 42:", evaluate_vc(cfg, seed=42))
    print("Kill Control:", kill_control_vc(cfg))
