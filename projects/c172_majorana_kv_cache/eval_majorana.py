#!/usr/bin/env python3
"""
Meta-Loop Integration Interface & CLI Evaluator for C172: Majorana Zero-Mode Non-Abelian Anyon Braiding KV Cache.

Provides standard callable hooks for engine.continuous_meta_loop:
1. evaluate_majorana(config, seed) -> Dict[str, float]
2. sanity_check_majorana(config) -> bool
3. kill_control_majorana(config) -> Tuple[bool, float, float]
"""

import os
import sys
import argparse
import json
import torch
from typing import Dict, Any, Tuple

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
for p in [current_dir, parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from projects.c172_majorana_kv_cache.evaluate import evaluate_c172, run_evaluation
except ImportError:
    from evaluate import evaluate_c172, run_evaluation

def evaluate_majorana(config: Dict[str, Any], seed: int = 42) -> Dict[str, float]:
    """Evaluates Majorana KV Attention in adversarial haystack."""
    res = evaluate_c172(config, seed=seed)
    return {
        "acc": res["acc_mzm"],
        "acc_std": res["acc_std"],
        "acc_poison": res["acc_poison"],
        "acc_scram": res["acc_scram"]
    }

def sanity_check_majorana(config: Dict[str, Any]) -> bool:
    """Sanity Check: Clean single-needle retrieval without distractors must achieve >= 0.85."""
    res = evaluate_c172(config, seed=42)
    acc = res["acc_sanity"]
    passed = acc >= 0.85
    print(f"   [Sanity Check] Clean Haystack Accuracy: {acc:.4f} (Threshold: >= 0.85)")
    return passed

def kill_control_majorana(config: Dict[str, Any]) -> Tuple[bool, float, float]:
    """Kill Control: Quasi-Particle Poisoning Arm must collapse to chance baseline (0.2000)."""
    res = evaluate_c172(config, seed=42)
    measured_kill = res["acc_poison"]
    chance_level = res["chance_baseline"]
    passed = measured_kill <= (chance_level + 0.10)
    print(f"   [Kill Control] Quasi-Particle Poisoning Acc: {measured_kill:.4f} (Chance: {chance_level:.4f})")
    return passed, measured_kill, chance_level

def main():
    parser = argparse.ArgumentParser(description="Evaluate C172 Majorana Zero-Mode KV Cache Attention")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--d_model", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--n_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--n_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--lambda_topological", type=float, default=30.0, help="Topological braid penalty scale")
    parser.add_argument("--seq_len", type=int, default=8, help="Sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--multi_seed", action="store_true", help="Run multi-seed evaluation across seeds [43, 138]")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    config = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_classes": args.n_classes,
        "lambda_topological": args.lambda_topological,
        "seq_len": args.seq_len,
        "n_train": 400,
        "n_test": 300,
        "epochs": args.epochs,
        "lr": 0.01
    }

    if args.multi_seed:
        print("Running Multi-Seed Evaluation for C172 across seeds [43, 138]...")
        res = run_evaluation(config, seeds=[43, 138])
        print("=" * 60)
        print(f"[C172] Majorana MZM Attention Acc: {res['mean_acc']:.4f} +/- {res['std_acc']:.4f}")
        print(f"[C172] Sanity Check (0 distractors): {res['sanity_acc']:.4f}")
        print(f"[C172] Lethal Control 1 (Poisoning): {res['mean_kill_poison']:.4f} (Chance: {res['chance']:.4f})")
        print(f"[C172] Lethal Control 2 (Scramble): {res['mean_kill_scram']:.4f} (Chance: {res['chance']:.4f})")
        print(f"[C172] Gate Check: Sanity={res['passed_sanity']}, Kill={res['passed_kill']}")
        print("=" * 60)
    else:
        print(f"Running Single-Seed Evaluation for C172 (seed={args.seed})...")
        res = evaluate_c172(config, seed=args.seed)
        print("=" * 60)
        print(f"[C172] Majorana MZM Attention Acc: {res['acc_mzm']:.4f}")
        print(f"[C172] Standard Attention: {res['acc_std']:.4f}")
        print(f"[C172] Sanity Check (0 distractors): {res['acc_sanity']:.4f}")
        print(f"[C172] Lethal Control 1 (Poisoning): {res['acc_poison']:.4f} (Chance: {res['chance_baseline']:.4f})")
        print(f"[C172] Lethal Control 2 (Scramble): {res['acc_scram']:.4f} (Chance: {res['chance_baseline']:.4f})")
        print(f"[C172] Gate Check: Sanity={res['passed_sanity']}, Kill={res['passed_kill']}")
        print("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
