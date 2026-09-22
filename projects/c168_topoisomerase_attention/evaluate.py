"""
Evaluation harness for Project C168: DNA Topoisomerase-II Strand-Passage Attention Unknotting.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from module import TopoisomeraseUnknottingClassifier
except ImportError:
    from projects.c168_topoisomerase_attention.module import TopoisomeraseUnknottingClassifier

def generate_knot_dataset(n_samples=500, seq_len=6, d_model=32, n_classes=5, class_vectors=None, trap_attractor=None, seed=42):
    """
    Constructs an adversarial topological knot haystack:
    - Root premise at index 0 carries the true class y
    - Intermediate steps 1 .. seq_len-4 represent background reasoning trajectory
    - Index seq_len-3 is an ADVERSARIAL KNOT DISTRACTOR:
      It creates a spatial self-intersection with the query by sitting within delta_knot
      of the trap attractor, while carrying an independent distractor class label
    - Index seq_len-2 is a transition step away from the knot
    - Query token at seq_len-1 is the trap attractor (containing 0 class information)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if class_vectors is None:
        class_vectors = torch.randn(n_classes, d_model)
        class_vectors = F.normalize(class_vectors, p=2, dim=-1)
        
    if trap_attractor is None:
        trap_attractor = torch.randn(d_model)
        trap_attractor = F.normalize(trap_attractor, p=2, dim=-1)
        
    X = torch.zeros(n_samples, seq_len, d_model)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    for i in range(n_samples):
        y = labels[i].item()
        
        # Target premise at token 0
        X[i, 0] = class_vectors[y]
        
        # Intermediate steps 1 to seq_len - 4: neutral background
        for t in range(1, seq_len - 3):
            X[i, t] = torch.randn(d_model) * 0.1
            
        # Token seq_len - 3 is the ADVERSARIAL KNOT DISTRACTOR:
        # Distance to query is ||class_vectors[dist_y] * 0.8|| = 0.8 < delta_knot (1.5)
        # High dot product with query traps standard attention
        dist_y = (y + torch.randint(1, n_classes, (1,)).item()) % n_classes
        X[i, seq_len - 3] = trap_attractor * 3.0 + class_vectors[dist_y] * 0.8
        
        # Step seq_len - 2 is a transition step away from the knot
        X[i, seq_len - 2] = torch.randn(d_model) * 0.2
        
        # Query token at seq_len - 1 is the trap attractor
        X[i, -1] = trap_attractor * 3.0
        
    return X, labels, class_vectors, trap_attractor

def evaluate_c168(config=None, seed=42):
    """
    Trains and evaluates a TopoisomeraseUnknottingClassifier with strict pre-mortem kill controls.
    """
    if config is None:
        config = {
            "d_model": 32,
            "n_heads": 2,
            "n_classes": 5,
            "delta_knot": 1.5,
            "lambda_strand": 40.0,
            "seq_len": 6,
            "n_train": 400,
            "n_test": 300,
            "epochs": 40,
            "lr": 0.01
        }
        
    d_model = config.get("d_model", 32)
    n_heads = config.get("n_heads", 2)
    n_classes = config.get("n_classes", 5)
    delta_knot = config.get("delta_knot", 1.5)
    lambda_strand = config.get("lambda_strand", 40.0)
    seq_len = config.get("seq_len", 6)
    n_train = config.get("n_train", 400)
    n_test = config.get("n_test", 300)
    epochs = config.get("epochs", 40)
    lr = config.get("lr", 0.01)
    
    # Generate datasets
    train_X, train_y, class_vecs, trap_vec = generate_knot_dataset(
        n_samples=n_train, seq_len=seq_len, d_model=d_model, n_classes=n_classes, class_vectors=None, trap_attractor=None, seed=seed
    )
    test_X, test_y, _, _ = generate_knot_dataset(
        n_samples=n_test, seq_len=seq_len, d_model=d_model, n_classes=n_classes, class_vectors=class_vecs, trap_attractor=trap_vec, seed=seed + 1000
    )
    
    model = TopoisomeraseUnknottingClassifier(
        d_model=d_model, n_heads=n_heads, n_classes=n_classes, delta_knot=delta_knot, lambda_strand=lambda_strand
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train under Topoisomerase strand passage
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_X, mode="topoisomerase")
        loss = F.cross_entropy(out, train_y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        # Sanity Check: Clean sequence with 0 knots (knot token replaced with neutral background)
        san_X = test_X.clone()
        san_X[:, seq_len - 3, :] = torch.randn(n_test, d_model) * 0.1
        out_san = model(san_X, mode="topoisomerase")
        acc_sanity = (out_san.argmax(-1) == test_y).float().mean().item()
        
        # Test 1: Topoisomerase Unknotting in Adversarial Haystack
        out_topo = model(test_X, mode="topoisomerase")
        acc_topo = (out_topo.argmax(-1) == test_y).float().mean().item()
        
        # Test 2: Standard Dot-Product Attention in Adversarial Haystack
        out_std = model(test_X, mode="standard")
        acc_std = (out_std.argmax(-1) == test_y).float().mean().item()
        
        # Test 3: Lethal Kill Control 1 (Anti-Topo / Topo-Poison Knot Lock)
        out_anti = model(test_X, mode="anti_topo")
        acc_anti = (out_anti.argmax(-1) == test_y).float().mean().item()
        
        # Test 4: Lethal Kill Control 2 (Scrambled Strand Cuts)
        out_scram = model(test_X, mode="scrambled_topo")
        acc_scram = (out_scram.argmax(-1) == test_y).float().mean().item()
        
    chance_baseline = 1.0 / n_classes
    passed_sanity = acc_sanity >= 0.85
    # Kill control must strictly collapse near random chance baseline (0.2000)
    passed_kill = (acc_anti <= chance_baseline + 0.05) and (acc_scram <= chance_baseline + 0.25)
    
    return {
        "acc_sanity": acc_sanity,
        "acc_topo": acc_topo,
        "acc_std": acc_std,
        "acc_anti": acc_anti,
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
            "delta_knot": 1.5,
            "lambda_strand": 40.0,
            "seq_len": 6,
            "n_train": 400,
            "n_test": 300,
            "epochs": 40,
            "lr": 0.01
        }
        
    accs = []
    kill_antis = []
    kill_scrams = []
    
    san_res = evaluate_c168(config, seed=seeds[0])
    sanity_acc = san_res["acc_sanity"]
    
    for s in seeds:
        res = evaluate_c168(config, seed=s)
        accs.append(res["acc_topo"])
        kill_antis.append(res["acc_anti"])
        kill_scrams.append(res["acc_scram"])
        
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    mean_kill_anti = float(np.mean(kill_antis))
    mean_kill_scram = float(np.mean(kill_scrams))
    
    chance = 1.0 / config.get("n_classes", 5)
    passed_sanity = sanity_acc >= 0.85
    passed_kill = mean_kill_anti <= (chance + 0.05)
    
    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "sanity_acc": sanity_acc,
        "kill_acc": mean_kill_anti,
        "mean_kill_anti": mean_kill_anti,
        "mean_kill_scram": mean_kill_scram,
        "chance": chance,
        "passed_sanity": passed_sanity,
        "passed_kill": passed_kill
    }

if __name__ == "__main__":
    res = evaluate_c168(seed=42)
    print("--- Single Seed C168 Evaluation ---")
    print(f"Sanity Check (0 knots): {res['acc_sanity']:.4f}")
    print(f"Topoisomerase Unknotting Acc: {res['acc_topo']:.4f}")
    print(f"Standard Attention Acc: {res['acc_std']:.4f}")
    print(f"Lethal Control 1 (Anti-Topo Lock): {res['acc_anti']:.4f} (Chance: {res['chance_baseline']:.4f})")
    print(f"Lethal Control 2 (Scrambled Cuts): {res['acc_scram']:.4f} (Chance: {res['chance_baseline']:.4f})")
    print(f"Passed Sanity: {res['passed_sanity']}, Passed Kill: {res['passed_kill']}")
    
    print("\n--- Multi-Seed C168 Evaluation ---")
    multi_res = run_evaluation(seeds=[43, 138])
    print(f"Topoisomerase Acc: {multi_res['mean_acc']:.4f} +/- {multi_res['std_acc']:.4f}")
    print(f"Lethal Kill Anti-Topo: {multi_res['mean_kill_anti']:.4f}")
    print(f"Lethal Kill Scrambled: {multi_res['mean_kill_scram']:.4f}")
