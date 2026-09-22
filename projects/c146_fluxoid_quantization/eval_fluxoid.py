#!/usr/bin/env python3
"""
Meta-Loop Integration Interface & CLI Evaluator for C146: Superconducting Fluxoid Quantization KV Cache Gating.

Provides standard callable hooks for engine.continuous_meta_loop:
1. evaluate_fluxoid(config, seed) -> Dict[str, float]
2. sanity_check_fluxoid(config) -> bool
3. kill_control_fluxoid(config) -> Tuple[bool, float, float]
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
for p in [current_dir, parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate import evaluate_c146, run_evaluation

def evaluate_fluxoid(config: Dict[str, Any], seed: int = 42) -> Dict[str, float]:
    """Evaluates Fluxoid Attention in adversarial haystack."""
    res = evaluate_c146(config, seed=seed)
    return {
        "acc": res["acc_fluxoid"],
        "acc_standard": res["acc_standard"],
        "acc_kill_slip": res["acc_kill_slip"],
        "acc_kill_scramble": res["acc_kill_scramble"]
    }

def sanity_check_fluxoid(config: Dict[str, Any]) -> bool:
    """Sanity Check: Clean single-needle retrieval without distractors must achieve >= 0.85."""
    res = evaluate_c146(config, seed=42)
    acc = res["acc_sanity"]
    passed = acc >= 0.85
    print(f"   [Sanity Check] Clean Haystack Accuracy: {acc:.4f} (Threshold: >= 0.85)")
    return passed

def kill_control_fluxoid(config: Dict[str, Any]) -> Tuple[bool, float, float]:
    """Kill Control: Inverted Fractional Phase Slip Arm must collapse to chance baseline (0.2000)."""
    res = evaluate_c146(config, seed=42)
    measured_kill = res["acc_kill_slip"]
    chance_level = res["chance_baseline"]
    passed = measured_kill <= (chance_level + 0.10)
    print(f"   [Kill Control] Fractional Phase Slip Acc: {measured_kill:.4f} (Chance: {chance_level:.4f})")
    return passed, measured_kill, chance_level

def main():
    parser = argparse.ArgumentParser(description="Evaluate C146 Superconducting Fluxoid Quantization KV Cache Gating")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--d_model", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--n_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--n_flux", type=int, default=4, help="Discrete fluxoid vortex quantum count")
    parser.add_argument("--n_classes", type=int, default=5, help="Number of class labels")
    parser.add_argument("--seq_len", type=int, default=10, help="Total sequence length (haystack)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--multi_seed", action="store_true", help="Run multi-seed evaluation across seeds [43, 138]")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    config = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_classes": args.n_classes,
        "n_flux": args.n_flux,
        "sigma_vortex": 0.15,
        "seq_len": args.seq_len,
        "n_train": 400,
        "n_test": 300,
        "epochs": args.epochs,
        "lr": 0.01
    }

    if args.multi_seed:
        print(f"Running Multi-Seed Evaluation for C146 across seeds [43, 138]...")
        res = run_evaluation(config, seeds=[43, 138])
        print("=" * 60)
        print(f"[C146] Fluxoid Haystack Accuracy: {res['mean_acc']:.4f} +/- {res['std_acc']:.4f}")
        print(f"[C146] Sanity Check (0 distractors): {res['sanity_acc']:.4f}")
        print(f"[C146] Lethal Control 1 (Fractional Slip): {res['mean_kill_slip']:.4f} (Chance: {res['chance']:.4f})")
        print(f"[C146] Lethal Control 2 (Scrambled Phase): {res['mean_kill_scramble']:.4f} (Chance: {res['chance']:.4f})")
        print(f"[C146] Gate Check: Sanity={res['passed_sanity']}, Kill={res['passed_kill']}")
        print("=" * 60)
    else:
        print(f"Running Single-Seed Evaluation for C146 (seed={args.seed})...")
        res = evaluate_c146(config, seed=args.seed)
        print("=" * 60)
        print(f"[C146] Fluxoid Haystack Accuracy: {res['acc_fluxoid']:.4f}")
        print(f"[C146] Standard Attention (Haystack): {res['acc_standard']:.4f}")
        print(f"[C146] Sanity Check (0 distractors): {res['acc_sanity']:.4f}")
        print(f"[C146] Lethal Control 1 (Fractional Slip): {res['acc_kill_slip']:.4f} (Chance: {res['chance_baseline']:.4f})")
        print(f"[C146] Lethal Control 2 (Scrambled Phase): {res['acc_kill_scramble']:.4f} (Chance: {res['chance_baseline']:.4f})")
        print(f"[C146] Gate Check: Sanity={res['passed_sanity']}, Kill={res['passed_kill']}")
        print("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
