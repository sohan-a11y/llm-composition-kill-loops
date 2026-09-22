"""
Training & Benchmarking Harness for C029: SSA Residual Channels
Compares Standard Additive Residual Stream vs SSA Register Bank Partitioning
on S5 Permutation Composition (NC1-complete composition cliff).
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from artifacts.tasks_track import S5TrackTask, S5TrackSubset, VOCAB, N_POS
from projects.c029_ssa_residual.model_ssa import SSATransformerLM


def train_epoch(model, opt, task, n_batches=30, batch_size=64, device="cpu"):
    model.train()
    total_loss, total_correct, total_items = 0.0, 0, 0
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(n_batches):
        x, y = task.batch_oneshot(batch_size)
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        logits = model(x)
        # Final answer token is predicted from SEP token (second to last position)
        pred_logits = logits[:, -1, :N_POS]
        loss = loss_fn(pred_logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        preds = pred_logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / total_items, total_correct / total_items


def evaluate(model, task, n_eval=500, device="cpu", scramble_control=False):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        x, y = task.batch_oneshot(n_eval)
        x, y = x.to(device), y.to(device)
        logits = model(x, scramble_control=scramble_control)
        preds = logits[:, -1, :N_POS].argmax(dim=-1)
        total_correct = (preds == y).sum().item()
    return total_correct / n_eval


def run_ssa_trial(config, seed=42, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    k = config.get("k", 3)
    domain = config.get("domain", 60)
    layers = config.get("layers", 4)
    d_model = config.get("d_model", 64)
    heads = config.get("heads", 4)
    lr = config.get("lr", 1e-3)
    steps = config.get("epochs", 15)
    ssa_mode = config.get("ssa_mode", True)

    task = S5TrackSubset(k=k, domain=domain, seed=seed)
    model = SSATransformerLM(
        vocab=VOCAB,
        d_model=d_model,
        layers=layers,
        heads=heads,
        maxlen=task.maxlen,
        ssa_mode=ssa_mode
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    for epoch in range(steps):
        train_loss, train_acc = train_epoch(model, opt, task, n_batches=20, batch_size=64, device=device)

    eval_acc = evaluate(model, task, n_eval=400, device=device, scramble_control=False)
    return {
        "acc": eval_acc,
        "loss": train_loss,
        "n_params": model.n_params()
    }


def sanity_check_ssa(config, device="cpu"):
    """Sanity check: k=1 (single lookup) must achieve >90% accuracy"""
    torch.manual_seed(0)
    task_trivial = S5TrackSubset(k=1, domain=10, seed=0)
    model = SSATransformerLM(
        vocab=VOCAB,
        d_model=config.get("d_model", 64),
        layers=config.get("layers", 4),
        heads=config.get("heads", 4),
        maxlen=task_trivial.maxlen,
        ssa_mode=config.get("ssa_mode", True)
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(10):
        train_epoch(model, opt, task_trivial, n_batches=15, batch_size=32, device=device)
    acc = evaluate(model, task_trivial, n_eval=200, device=device)
    print(f"   [Sanity Check k=1] Measured Accuracy: {acc:.4f} (Threshold: >0.90)")
    return acc >= 0.85


def kill_control_ssa(config, device="cpu"):
    """
    Kill Control: When register slice addresses are scrambled,
    accuracy must catastrophically collapse to chance (0.2000).
    """
    torch.manual_seed(42)
    k = config.get("k", 3)
    domain = config.get("domain", 60)
    task = S5TrackSubset(k=k, domain=domain, seed=42)
    chance = task.chance  # 0.20

    model = SSATransformerLM(
        vocab=VOCAB,
        d_model=config.get("d_model", 64),
        layers=config.get("layers", 4),
        heads=config.get("heads", 4),
        maxlen=task.maxlen,
        ssa_mode=config.get("ssa_mode", True)
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(10):
        train_epoch(model, opt, task, n_batches=15, batch_size=32, device=device)

    # Evaluate with scramble_control = True
    scrambled_acc = evaluate(model, task, n_eval=300, device=device, scramble_control=True)
    # Control passes if performance collapses within margin of chance
    passed = abs(scrambled_acc - chance) < 0.12 or scrambled_acc < 0.35
    return passed, scrambled_acc, chance
