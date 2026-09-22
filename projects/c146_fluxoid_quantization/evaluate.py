"""
Evaluation harness for Project C146: Superconducting Fluxoid Quantization KV Cache Gating.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from module import FluxoidHaystackClassifier
except ImportError:
    from projects.c146_fluxoid_quantization.module import FluxoidHaystackClassifier

def generate_haystack_dataset(n_samples=500, seq_len=10, d_model=32, n_classes=5, n_flux=4, class_vectors=None, seed=42):
    """
    Constructs an adversarial distractor haystack:
    - Target needle is placed at a random index in [0, seq_len - 2]
    - Distractors have IDENTICAL class vector magnitude and distribution (adversarial haystack)
    - Target needle has exact integer vortex winding phase matching the query
    - Distractors have fractional phase slips (offset by pi / n_flux)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if class_vectors is None:
        class_vectors = torch.randn(n_classes, d_model)
        class_vectors = F.normalize(class_vectors, p=2, dim=-1)
    
    X = torch.zeros(n_samples, seq_len, d_model)
    phases = torch.zeros(n_samples, seq_len)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    for i in range(n_samples):
        y = labels[i].item()
        needle_idx = torch.randint(0, seq_len - 1, (1,)).item()
        
        # Target integer fluxoid winding state: k_flux in {0, 1, ..., n_flux - 1}
        k_flux = y % n_flux
        vortex_phase = (k_flux / n_flux) * 2 * math.pi - math.pi
        
        # Target needle
        X[i, needle_idx] = class_vectors[y]
        phases[i, needle_idx] = vortex_phase
        
        # Distractors: valid class tokens with fractional phase slips
        for j in range(seq_len - 1):
            if j != needle_idx:
                dist_y = torch.randint(0, n_classes, (1,)).item()
                X[i, j] = class_vectors[dist_y]
                # Fractional phase slip midway between integer vortex states
                slip_offset = math.pi / n_flux
                phases[i, j] = vortex_phase + slip_offset + (torch.randn(1).item() * 0.02)
                
        # Query token at seq_len - 1 shares matching integer vortex phase but has ZERO content information
        phases[i, -1] = vortex_phase
        X[i, -1] = torch.zeros(d_model)
        
    return X, phases, labels, class_vectors

def evaluate_c146(config=None, seed=42):
    """
    Trains and evaluates a FluxoidHaystackClassifier with strict pre-mortem kill controls.
    """
    if config is None:
        config = {
            "d_model": 32,
            "n_heads": 2,
            "n_classes": 5,
            "n_flux": 4,
            "sigma_vortex": 0.15,
            "seq_len": 10,
            "n_train": 400,
            "n_test": 300,
            "epochs": 40,
            "lr": 0.01,
            "batch_size": 64
        }
        
    d_model = config.get("d_model", 32)
    n_heads = config.get("n_heads", 2)
    n_classes = config.get("n_classes", 5)
    n_flux = config.get("n_flux", 4)
    sigma_vortex = config.get("sigma_vortex", 0.15)
    seq_len = config.get("seq_len", 10)
    n_train = config.get("n_train", 400)
    n_test = config.get("n_test", 300)
    epochs = config.get("epochs", 40)
    lr = config.get("lr", 0.01)
    
    # Generate training and test sets
    train_X, train_P, train_y, class_vecs = generate_haystack_dataset(
        n_samples=n_train, seq_len=seq_len, d_model=d_model, n_classes=n_classes, n_flux=n_flux, class_vectors=None, seed=seed
    )
    test_X, test_P, test_y, _ = generate_haystack_dataset(
        n_samples=n_test, seq_len=seq_len, d_model=d_model, n_classes=n_classes, n_flux=n_flux, class_vectors=class_vecs, seed=seed + 1000
    )
    
    model = FluxoidHaystackClassifier(
        d_model=d_model, n_heads=n_heads, n_classes=n_classes, n_flux=n_flux, sigma_vortex=sigma_vortex
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train in fluxoid mode
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_X, train_P, mode="fluxoid")
        loss = F.cross_entropy(logits, train_y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        # Sanity Check: Clean needle without distractors (seq_len = 2)
        san_X = torch.zeros(100, 2, d_model)
        san_P = torch.zeros(100, 2)
        san_y = torch.randint(0, n_classes, (100,))
        for i in range(100):
            y_i = san_y[i].item()
            k_f = y_i % n_flux
            v_p = (k_f / n_flux) * 2 * math.pi - math.pi
            san_X[i, 0] = class_vecs[y_i]
            san_P[i, 0] = v_p
            san_X[i, 1] = torch.zeros(d_model)
            san_P[i, 1] = v_p
        out_san = model(san_X, san_P, mode="fluxoid")
        acc_sanity = (out_san.argmax(-1) == san_y).float().mean().item()
        
        # Fluxoid Quantized Attention in Haystack
        out_flux = model(test_X, test_P, mode="fluxoid")
        acc_fluxoid = (out_flux.argmax(-1) == test_y).float().mean().item()
        
        # Standard Dot-Product Attention in Haystack
        out_std = model(test_X, test_P, mode="standard")
        acc_standard = (out_std.argmax(-1) == test_y).float().mean().item()
        
        # Lethal Kill Control 1: Fractional Phase Slip Inversion Arm
        out_slip = model(test_X, test_P, mode="fractional_slip")
        acc_kill_slip = (out_slip.argmax(-1) == test_y).float().mean().item()
        
        # Lethal Kill Control 2: Scrambled Phase Arm
        out_scramble = model(test_X, test_P, mode="scrambled_phase")
        acc_kill_scramble = (out_scramble.argmax(-1) == test_y).float().mean().item()
        
    chance_baseline = 1.0 / n_classes
    passed_sanity = acc_sanity >= 0.85
    # Kill controls must collapse near chance baseline
    passed_kill = (acc_kill_slip <= chance_baseline + 0.12) and (acc_kill_scramble <= chance_baseline + 0.12)
    
    return {
        "acc_sanity": acc_sanity,
        "acc_fluxoid": acc_fluxoid,
        "acc_standard": acc_standard,
        "acc_kill_slip": acc_kill_slip,
        "acc_kill_scramble": acc_kill_scramble,
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
            "n_flux": 4,
            "sigma_vortex": 0.15,
            "seq_len": 10,
            "n_train": 400,
            "n_test": 300,
            "epochs": 40,
            "lr": 0.01
        }
        
    accs = []
    kill_slips = []
    kill_scrambles = []
    
    # Run sanity on seed 0
    san_res = evaluate_c146(config, seed=seeds[0])
    sanity_acc = san_res["acc_sanity"]
    
    for s in seeds:
        res = evaluate_c146(config, seed=s)
        accs.append(res["acc_fluxoid"])
        kill_slips.append(res["acc_kill_slip"])
        kill_scrambles.append(res["acc_kill_scramble"])
        
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    mean_kill_slip = float(np.mean(kill_slips))
    mean_kill_scramble = float(np.mean(kill_scrambles))
    
    chance = 1.0 / config.get("n_classes", 5)
    passed_sanity = sanity_acc >= 0.85
    passed_kill = mean_kill_slip <= (chance + 0.10)
    
    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "sanity_acc": sanity_acc,
        "kill_acc": mean_kill_slip,
        "mean_kill_slip": mean_kill_slip,
        "mean_kill_scramble": mean_kill_scramble,
        "chance": chance,
        "passed_sanity": passed_sanity,
        "passed_kill": passed_kill
    }

if __name__ == "__main__":
    res = evaluate_c146(seed=42)
    print("--- Single Seed C146 Evaluation ---")
    print(f"Sanity Check (0 distractors): {res['acc_sanity']:.4f}")
    print(f"Fluxoid Quantized (Haystack): {res['acc_fluxoid']:.4f}")
    print(f"Standard Attention (Haystack): {res['acc_standard']:.4f}")
    print(f"Lethal Control 1 (Fractional Slip): {res['acc_kill_slip']:.4f} (Chance: {res['chance_baseline']:.4f})")
    print(f"Lethal Control 2 (Scrambled Phase): {res['acc_kill_scramble']:.4f} (Chance: {res['chance_baseline']:.4f})")
    print(f"Passed Sanity: {res['passed_sanity']}, Passed Kill: {res['passed_kill']}")
    
    print("\n--- Multi-Seed C146 Evaluation ---")
    multi_res = run_evaluation(seeds=[43, 138])
    print(f"Fluxoid Acc: {multi_res['mean_acc']:.4f} +/- {multi_res['std_acc']:.4f}")
    print(f"Lethal Kill Slip: {multi_res['mean_kill_slip']:.4f}")
    print(f"Lethal Kill Scramble: {multi_res['mean_kill_scramble']:.4f}")
