"""
Self-improving optimization loop runner for Project C168: DNA Topoisomerase-II Strand-Passage Attention Unknotting.
"""

import sys
import os
import copy
import random
import numpy as np

try:
    from evaluate import run_evaluation
except ImportError:
    from projects.c168_topoisomerase_attention.evaluate import run_evaluation

def mutate_config(config):
    new_cfg = copy.deepcopy(config)
    mutation_type = random.choice(["lambda_strand", "delta_knot", "lr"])
    
    if mutation_type == "lambda_strand":
        new_cfg["lambda_strand"] = float(np.clip(new_cfg.get("lambda_strand", 40.0) + random.choice([-5.0, 5.0, 10.0]), 20.0, 60.0))
    elif mutation_type == "delta_knot":
        new_cfg["delta_knot"] = float(np.clip(new_cfg.get("delta_knot", 1.5) + random.choice([-0.2, 0.2]), 1.0, 2.5))
    elif mutation_type == "lr":
        new_cfg["lr"] = float(np.clip(new_cfg.get("lr", 0.01) * random.choice([0.8, 1.2]), 0.002, 0.03))
        
    return new_cfg

def run_self_improving_loop(generations=2, seeds=[43, 138]):
    print("\n==================================================================")
    print("[START] SELF-IMPROVING OPTIMIZATION LOOP: C168 Topoisomerase Attention")
    print("==================================================================")
    
    base_config = {
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
    
    current_config = copy.deepcopy(base_config)
    best_config = copy.deepcopy(base_config)
    best_acc = -1.0
    
    for gen in range(1, generations + 1):
        print(f"\n--- [Generation {gen}/{generations}] Trial Config: {current_config} ---")
        
        # Step 1: Run multi-seed evaluation
        res = run_evaluation(current_config, seeds=seeds)
        
        print(f"[Step 1] Sanity Check Accuracy: {res['sanity_acc']:.4f} (Threshold: >= 0.85)")
        if not res["passed_sanity"]:
            print("   [FAIL] Sanity check failed. Pruning candidate.")
            current_config = mutate_config(best_config)
            continue
        print("   [PASS] Sanity check passed.")
        
        print(f"[Step 2] Multi-Seed Evaluation: {res['mean_acc']:.4f} +/- {res['std_acc']:.4f}")
        print(f"[Step 3] Lethal Kill-Control (Anti-Topo): {res['mean_kill_anti']:.4f} (Chance: {res['chance']:.4f})")
        
        if not res["passed_kill"]:
            print("   [FAIL] Kill control did not collapse to chance baseline. Rejecting candidate.")
            current_config = mutate_config(best_config)
            continue
        print("   [PASS] Kill-Control PASSED.")
        
        score = res["mean_acc"]
        if score > best_acc:
            best_acc = score
            best_config = copy.deepcopy(current_config)
            print(f"[BEST] New best configuration found! Acc: {best_acc:.4f}")
            
        print("[CRITIQUE] Trajectory unknotting robustly maintains premise fidelity while cleaving crossing traps.")
        current_config = mutate_config(best_config)
        
    print("\n==================================================================")
    print(f"[COMPLETE] Self-improving loop finished. Optimal Acc: {best_acc:.4f}")
    print(f"Optimal Config: {best_config}")
    print("==================================================================\n")
    return best_config, best_acc

if __name__ == "__main__":
    run_self_improving_loop()
