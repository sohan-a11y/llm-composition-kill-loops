#!/usr/bin/env python3
"""
Master Project Runner for C146: Superconducting Fluxoid Quantization KV Cache Gating.
"""

import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
for p in [current_dir, parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate import run_evaluation, evaluate_c146

def run_project():
    print("=" * 65)
    print("PROJECT C146: SUPERCONDUCTING FLUXOID QUANTIZATION KV CACHE GATING")
    print("=" * 65)
    
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
    
    print("\n[Phase 1] Executing Multi-Seed Evaluation across seeds [43, 138]...")
    res = run_evaluation(config, seeds=[43, 138])
    
    print(f"  Sanity Check (0 distractors)   : {res['sanity_acc']:.4f} (Threshold: >0.85)")
    print(f"  Fluxoid Quantized (Haystack)   : {res['mean_acc']:.4f} +/- {res['std_acc']:.4f}")
    print(f"  Lethal Kill Arm 1 (Slip)       : {res['mean_kill_slip']:.4f} (Chance: {res['chance']:.4f})")
    print(f"  Lethal Kill Arm 2 (Scramble)   : {res['mean_kill_scramble']:.4f} (Chance: {res['chance']:.4f})")
    print(f"  Sanity Gate Passed             : {res['passed_sanity']}")
    print(f"  Kill-Control Gate Passed       : {res['passed_kill']}")
    
    # Save metrics summary
    logs_dir = os.path.join(current_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    summary_path = os.path.join(logs_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[Phase 2] Metrics successfully saved to: {summary_path}")
    
    # Assertions
    assert res["passed_sanity"], f"Sanity check failed: {res['sanity_acc']:.4f} < 0.85"
    assert res["passed_kill"], f"Kill control failed: {res['mean_kill_slip']:.4f} > {res['chance'] + 0.10:.4f}"
    assert res["mean_acc"] >= 0.85, f"Fluxoid mode accuracy too low: {res['mean_acc']:.4f} < 0.85"
    
    print("\n[SUCCESS] ALL EMPIRICAL GATES SATISFIED! PROJECT C146 VERIFIED.")
    print("=" * 65)

if __name__ == "__main__":
    run_project()
