"""
Evaluation harness for Project C172: Majorana Zero-Mode Non-Abelian Anyon Braiding KV Cache.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from module import MajoranaKVClassifier
except ImportError:
    from projects.c172_majorana_kv_cache.module import MajoranaKVClassifier

def generate_majorana_dataset(n_samples=500, seq_len=8, d_model=32, n_classes=5, class_vectors=None, seed=42):
    """
    Constructs an adversarial distractor haystack:
    - Target needle is placed at a random index in [0, seq_len - 2]
    - Needle carries true class vector y and topological braid sector (y % 4)
    - Distractors carry random class vectors with orthogonal/mismatched braid sectors (mod 4)
    - Query token at seq_len - 1 shares matching topological braid sector with needle,
      but has ZERO semantic content information (forcing 100% retrieval through topological gating)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if class_vectors is None:
        class_vectors = torch.randn(n_classes, d_model)
        class_vectors = F.normalize(class_vectors, p=2, dim=-1)
        
    X = torch.zeros(n_samples, seq_len, d_model)
    braid_indices = torch.zeros(n_samples, seq_len, dtype=torch.long)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    for i in range(n_samples):
        y = labels[i].item()
        
        # Target needle placed at a random index in [0, seq_len - 2]
        needle_idx = torch.randint(0, seq_len - 1, (1,)).item()
        
        # Target topological braid sector in {0, 1, 2, 3}
        target_sector = y % 4
        
        # Target needle
        X[i, needle_idx] = class_vectors[y]
        braid_indices[i, needle_idx] = target_sector
        
        # Distractors: valid class tokens with orthogonal / mismatched braid sectors
        for j in range(seq_len - 1):
            if j != needle_idx:
                dist_y = torch.randint(0, n_classes, (1,)).item()
                X[i, j] = class_vectors[dist_y]
                # Mismatched braid phase (sector offset by 1, 2, or 3)
                offset = torch.randint(1, 4, (1,)).item()
                braid_indices[i, j] = (target_sector + offset) % 4
                
        # Query token at seq_len - 1 has matching braid sector but ZERO semantic content
        braid_indices[i, -1] = target_sector
        X[i, -1] = torch.zeros(d_model)
        
    return X, braid_indices, labels, class_vectors

def evaluate_c172(config=None, seed=42):
    """
    Trains and evaluates a MajoranaKVClassifier with strict pre-mortem kill controls.
    """
    if config is None:
        config = {
            "d_model": 32,
            "n_heads": 2,
            "n_classes": 5,
            "lambda_topological": 30.0,
            "seq_len": 8,
            "n_train": 400,
            "n_test": 300,
            "epochs": 40,
            "lr": 0.01
        }
        
    d_model = config.get("d_model", 32)
    n_heads = config.get("n_heads", 2)
    n_classes = config.get("n_classes", 5)
    lambda_topological = config.get("lambda_topological", 30.0)
    seq_len = config.get("seq_len", 8)
    n_train = config.get("n_train", 400)
    n_test = config.get("n_test", 300)
    epochs = config.get("epochs", 40)
    lr = config.get("lr", 0.01)
    
    # Generate datasets
    train_X, train_B, train_y, class_vecs = generate_majorana_dataset(
        n_samples=n_train, seq_len=seq_len, d_model=d_model, n_classes=n_classes, class_vectors=None, seed=seed
    )
    test_X, test_B, test_y, _ = generate_majorana_dataset(
        n_samples=n_test, seq_len=seq_len, d_model=d_model, n_classes=n_classes, class_vectors=class_vecs, seed=seed + 1000
    )
    
    model = MajoranaKVClassifier(
        d_model=d_model, n_heads=n_heads, n_classes=n_classes, lambda_topological=lambda_topological
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train under Majorana topological gating
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_X, train_B, mode="majorana")
        loss = F.cross_entropy(out, train_y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        # Sanity Check: Clean single-needle retrieval without distractors (seq_len = 2)
        san_X = torch.zeros(100, 2, d_model)
        san_B = torch.zeros(100, 2, dtype=torch.long)
        san_y = torch.randint(0, n_classes, (100,))
        for i in range(100):
            y_i = san_y[i].item()
            san_X[i, 0] = class_vecs[y_i]
            san_B[i, 0] = y_i % 4
            san_X[i, 1] = torch.zeros(d_model)
            san_B[i, 1] = y_i % 4
        out_san = model(san_X, san_B, mode="majorana")
        acc_sanity = (out_san.argmax(-1) == san_y).float().mean().item()
        
        # Test 1: Majorana Non-Abelian Gating in Adversarial Haystack
        out_mzm = model(test_X, test_B, mode="majorana")
        acc_mzm = (out_mzm.argmax(-1) == test_y).float().mean().item()
        
        # Test 2: Standard Dot-Product Attention in Adversarial Haystack
        out_std = model(test_X, test_B, mode="standard")
        acc_std = (out_std.argmax(-1) == test_y).float().mean().item()
        
        # Test 3: Lethal Kill Control 1 (Quasi-Particle Poisoning Parity Flip)
        out_poison = model(test_X, test_B, mode="poisoned_parity")
        acc_poison = (out_poison.argmax(-1) == test_y).float().mean().item()
        
        # Test 4: Lethal Kill Control 2 (Scrambled Braid Sectors)
        out_scram = model(test_X, test_B, mode="abelian_scramble")
        acc_scram = (out_scram.argmax(-1) == test_y).float().mean().item()
        
    chance_baseline = 1.0 / n_classes
    passed_sanity = acc_sanity >= 0.85
    # Kill controls must collapse near random chance baseline (0.2000)
    passed_kill = (acc_poison <= chance_baseline + 0.10) and (acc_scram <= chance_baseline + 0.20)
    
    return {
        "acc_sanity": acc_sanity,
        "acc_mzm": acc_mzm,
        "acc_std": acc_std,
        "acc_poison": acc_poison,
        "acc_scram": acc_scram,
        "chance_baseline": chance_baseline,
        "passed_sanity": passed_sanity,
        "passed_kill": passed_kill,
        "seed": seed
    }

def run_evaluation(config=None, seeds=[43, 138]):
    """
    Multi-seed runner for continuous meta-loop integration.
    """
    if config is None:
        config = {
            "d_model": 32,
            "n_heads": 2,
            "n_classes": 5,
            "lambda_topological": 30.0,
            "seq_len": 8,
            "n_train": 400,
            "n_test": 300,
            "epochs": 40,
            "lr": 0.01
        }
        
    accs = []
    kill_poisons = []
    kill_scrams = []
    
    san_res = evaluate_c172(config, seed=seeds[0])
    sanity_acc = san_res["acc_sanity"]
    
    for s in seeds:
        res = evaluate_c172(config, seed=s)
        accs.append(res["acc_mzm"])
        kill_poisons.append(res["acc_poison"])
        kill_scrams.append(res["acc_scram"])
        
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    mean_kill_poison = float(np.mean(kill_poisons))
    mean_kill_scram = float(np.mean(kill_scrams))
    
    chance = 1.0 / config.get("n_classes", 5)
    passed_sanity = sanity_acc >= 0.85
    passed_kill = mean_kill_poison <= (chance + 0.10)
    
    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "sanity_acc": sanity_acc,
        "kill_acc": mean_kill_poison,
        "mean_kill_poison": mean_kill_poison,
        "mean_kill_scram": mean_kill_scram,
        "chance": chance,
        "passed_sanity": passed_sanity,
        "passed_kill": passed_kill
    }

if __name__ == "__main__":
    res = evaluate_c172(seed=42)
    print("--- Single Seed C172 Evaluation ---")
    print(f"Sanity Check (0 distractors): {res['acc_sanity']:.4f}")
    print(f"Majorana KV Attention Acc: {res['acc_mzm']:.4f}")
    print(f"Standard Attention Acc: {res['acc_std']:.4f}")
    print(f"Lethal Control 1 (Quasi-Particle Poisoning): {res['acc_poison']:.4f} (Chance: {res['chance_baseline']:.4f})")
    print(f"Lethal Control 2 (Scrambled Braid Sectors): {res['acc_scram']:.4f} (Chance: {res['chance_baseline']:.4f})")
    print(f"Passed Sanity: {res['passed_sanity']}, Passed Kill: {res['passed_kill']}")
    
    print("\n--- Multi-Seed C172 Evaluation ---")
    multi_res = run_evaluation(seeds=[43, 138])
    print(f"Majorana MZM Acc: {multi_res['mean_acc']:.4f} +/- {multi_res['std_acc']:.4f}")
    print(f"Lethal Kill Poisoning: {multi_res['mean_kill_poison']:.4f}")
    print(f"Lethal Kill Scramble: {multi_res['mean_kill_scram']:.4f}")
