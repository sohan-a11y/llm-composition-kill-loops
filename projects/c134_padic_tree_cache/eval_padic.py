"""
Evaluation Harness for C134: Non-Archimedean p-adic Ultrametric Tree Cache.

Includes:
1. evaluate_padic: Multi-seed evaluation with randomized distractor subtrees.
2. sanity_check_padic: Clean retrieval without distractors (Threshold: >0.85).
3. kill_control_padic: Lethal Negative Control Arm (Permuted ground-truth labels, collapses to chance ~0.2000).
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from padic_tree_cache import PAdicTreeCacheModel

# Global class prototypes for consistent evaluation
_GLOBAL_PROTOTYPES = None
PROTOS_SEED = 42

def get_prototypes(n_classes: int = 5, d_model: int = 32) -> torch.Tensor:
    global _GLOBAL_PROTOTYPES
    if _GLOBAL_PROTOTYPES is None:
        torch.manual_seed(PROTOS_SEED)
        p = torch.randn(n_classes, d_model)
        _GLOBAL_PROTOTYPES = p / p.norm(dim=-1, keepdim=True)
    return _GLOBAL_PROTOTYPES


# Standard Tree Hierarchy Layout:
# Query is in Subtree 0 (address 0)
# Keys:
# Index 0: True sibling in Subtree 0 (address 4; diff = 4, p-adic dist = 0.25)
# Index 1..6: Distractor subtrees (addresses 1, 2, 3, 5, 6, 7; p-adic dist >= 0.50)
QUERY_COORD = [0]
KEY_COORDS = [4, 1, 2, 3, 5, 6, 7]


def generate_tree_dataset(
    n_samples: int = 300,
    d_model: int = 32,
    n_classes: int = 5,
    seed: int = 42,
    n_distractors: int = 6,
    random_distractors: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    prototypes = get_prototypes(n_classes=n_classes, d_model=d_model)
    
    labels = torch.randint(0, n_classes, (n_samples,))
    q_vecs = torch.randn(n_samples, 1, d_model) * 0.1
    
    n_keys = 1 + n_distractors
    kv_seqs = torch.randn(n_samples, n_keys, d_model) * 0.1
    
    for b in range(n_samples):
        target_class = labels[b].item()
        # Key 0 holds the true target prototype
        kv_seqs[b, 0] = prototypes[target_class] + 0.02 * torch.randn(d_model)
        # Remaining keys are distractors
        for k_idx in range(1, n_keys):
            if random_distractors:
                distr_class = torch.randint(0, n_classes, (1,)).item()
            else:
                distr_class = (target_class + k_idx) % n_classes
            kv_seqs[b, k_idx] = prototypes[distr_class] + 0.02 * torch.randn(d_model)
            
    return q_vecs, kv_seqs, labels


def train_and_eval(
    model: nn.Module,
    tr_loader: DataLoader,
    q_te: torch.Tensor,
    kv_te: torch.Tensor,
    y_te: torch.Tensor,
    epochs: int = 25,
    lr: float = 0.01,
    override_mode: str = None
) -> Tuple[float, float]:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    k_coords = KEY_COORDS[:kv_te.shape[1]]

    for epoch in range(epochs):
        model.train()
        for b_q, b_kv, b_y in tr_loader:
            optimizer.zero_grad()
            logits, _ = model(b_q, b_kv, QUERY_COORD, k_coords, override_mode=override_mode)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits, attn_w = model(q_te, kv_te, QUERY_COORD, k_coords, override_mode=override_mode)
        preds = test_logits.argmax(dim=-1)
        acc = (preds == y_te).float().mean().item()
        target_attn_mass = attn_w[:, 0, 0].mean().item()
        
    return acc, target_attn_mass


def evaluate_padic(config: Dict[str, Any], seed: int = 42) -> Dict[str, float]:
    n_samples = config.get("n_samples", 300)
    d_model = config.get("d_model", 32)
    n_classes = config.get("n_classes", 5)
    penalty = config.get("penalty", 10.0)
    p = config.get("p", 2)
    epochs = config.get("epochs", 25)
    lr = config.get("lr", 0.01)

    torch.manual_seed(seed)
    q_tr, kv_tr, y_tr = generate_tree_dataset(n_samples=n_samples, d_model=d_model, n_classes=n_classes, seed=seed)
    q_te, kv_te, y_te = generate_tree_dataset(n_samples=100, d_model=d_model, n_classes=n_classes, seed=seed + 1000)

    tr_loader = DataLoader(TensorDataset(q_tr, kv_tr, y_tr), batch_size=32, shuffle=True)
    model = PAdicTreeCacheModel(d_model=d_model, n_classes=n_classes, penalty=penalty, p=p, mode="padic")
    
    acc, target_attn = train_and_eval(model, tr_loader, q_te, kv_te, y_te, epochs=epochs, lr=lr)
    return {
        "acc": acc,
        "target_attn_mass": target_attn
    }


def sanity_check_padic(config: Dict[str, Any]) -> bool:
    """Sanity Check: Clean retrieval with 0 distractors must achieve >0.85 accuracy."""
    d_model = config.get("d_model", 32)
    n_classes = config.get("n_classes", 5)
    
    q_tr, kv_tr, y_tr = generate_tree_dataset(n_samples=250, d_model=d_model, n_classes=n_classes, seed=999, n_distractors=0)
    q_te, kv_te, y_te = generate_tree_dataset(n_samples=80, d_model=d_model, n_classes=n_classes, seed=998, n_distractors=0)

    tr_loader = DataLoader(TensorDataset(q_tr, kv_tr, y_tr), batch_size=32, shuffle=True)
    model = PAdicTreeCacheModel(d_model=d_model, n_classes=n_classes, penalty=5.0, p=2, mode="padic")
    
    acc, _ = train_and_eval(model, tr_loader, q_te, kv_te, y_te, epochs=15)
    print(f"   [Sanity Check] Clean Accuracy (0 distractors): {acc:.4f} (Threshold: >0.85)")
    return acc >= 0.85


def kill_control_padic(config: Dict[str, Any]) -> Tuple[bool, float, float]:
    """
    Lethal Negative Control: Permute target labels at test time.
    Accuracy MUST collapse to chance baseline (1 / 5 = 0.2000).
    """
    n_samples = config.get("n_samples", 300)
    d_model = config.get("d_model", 32)
    n_classes = config.get("n_classes", 5)
    penalty = config.get("penalty", 10.0)
    p = config.get("p", 2)
    epochs = config.get("epochs", 25)

    q_tr, kv_tr, y_tr = generate_tree_dataset(n_samples=n_samples, d_model=d_model, n_classes=n_classes, seed=777)
    q_te, kv_te, y_te = generate_tree_dataset(n_samples=100, d_model=d_model, n_classes=n_classes, seed=888)

    # Permute ground truth labels
    perm = torch.randperm(len(y_te))
    y_te_shuffled = y_te[perm]

    tr_loader = DataLoader(TensorDataset(q_tr, kv_tr, y_tr), batch_size=32, shuffle=True)
    model = PAdicTreeCacheModel(d_model=d_model, n_classes=n_classes, penalty=penalty, p=p, mode="padic")
    
    acc, _ = train_and_eval(model, tr_loader, q_te, kv_te, y_te_shuffled, epochs=epochs)
    chance_baseline = 0.2000
    passed = acc <= 0.35
    return passed, acc, chance_baseline


if __name__ == "__main__":
    cfg = {"n_samples": 300, "d_model": 32, "n_classes": 5, "penalty": 10.0, "p": 2}
    print("Sanity Check:", sanity_check_padic(cfg))
    print("Eval seed 42:", evaluate_padic(cfg, seed=42))
    print("Kill Control:", kill_control_padic(cfg))
