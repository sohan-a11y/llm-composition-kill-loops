#!/usr/bin/env python3
"""
Meta-Loop Integration Interface & CLI Evaluator for C168: DNA Topoisomerase-II Strand-Passage Attention Unknotting.

Provides standard callable hooks for engine.continuous_meta_loop:
1. evaluate_topo(config, seed) -> Dict[str, float]
2. sanity_check_topo(config) -> bool
3. kill_control_topo(config) -> Tuple[bool, float, float]
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
    from projects.c168_topoisomerase_attention.evaluate import evaluate_c168, run_evaluation
except ImportError:
    from evaluate import evaluate_c168, run_evaluation

def evaluate_topo(config: Dict[str, Any], seed: int = 42) -> Dict[str, float]:
    """Evaluates Topoisomerase Unknotting Attention in adversarial knot haystack."""
    res = evaluate_c168(config, seed=seed)
    return {
        "acc": res["acc_topo"],
        "acc_std": res["acc_std"],
        "acc_anti": res["acc_anti"],
        "acc_scram": res["acc_scram"]
    }

def sanity_check_topo(config: Dict[str, Any]) -> bool:
    """Sanity Check: Clean sequence with 0 knots must achieve >= 0.85."""
    res = evaluate_c168(config, seed=42)
    acc = res["acc_sanity"]
    passed = acc >= 0.85
    print(f"   [Sanity Check] Clean Haystack Accuracy: {acc:.4f} (Threshold: >= 0.85)")
    return passed

def kill_control_topo(config: Dict[str, Any]) -> Tuple[bool, float, float]:
    """Kill Control: Anti-Topoisomerase Knot Lock Arm must collapse to chance baseline (0.2000)."""
    res = evaluate_c168(config, seed=42)
    measured_kill = res["acc_anti"]
    chance_level = res["chance_baseline"]
    passed = measured_kill <= (chance_level + 0.10)
    print(f"   [Kill Control] Anti-Topo Lock Acc: {measured_kill:.4f} (Chance: {chance_level:.4f})")
    return passed, measured_kill, chance_level

def main():
    parser = argparse.ArgumentParser(description="Evaluate C168 DNA Topoisomerase-II Strand-Passage Attention Unknotting")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--d_model", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--n_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--n_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--delta_knot", type=float, default=1.5, help="Topological crossing distance threshold")
    parser.add_argument("--lambda_strand", type=float, default=40.0, help="Strand-passage cleavage penalty")
    parser.add_argument("--seq_len", type=int, default=6, help="Sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--multi_seed", action="store_true", help="Run multi-seed evaluation across seeds [43, 138]")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    config = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_classes": args.n_classes,
        "delta_knot": args.delta_knot,
        "lambda_strand": args.lambda_strand,
        "seq_len": args.seq_len,
        "n_train": 400,
        "n_test": 300,
        "epochs": args.epochs,
        "lr": 0.01
    }

    if args.multi_seed:
        print("Running Multi-Seed Evaluation for C168 across seeds [43, 138]...")
        res = run_evaluation(config, seeds=[43, 138])
        print("=" * 60)
        print(f"[C168] Topoisomerase Unknotting Acc: {res['mean_acc']:.4f} +/- {res['std_acc']:.4f}")
        print(f"[C168] Sanity Check (0 knots): {res['sanity_acc']:.4f}")
        print(f"[C168] Lethal Control 1 (Anti-Topo Lock): {res['mean_kill_anti']:.4f} (Chance: {res['chance']:.4f})")
        print(f"[C168] Lethal Control 2 (Scrambled Cuts): {res['mean_kill_scram']:.4f} (Chance: {res['chance']:.4f})")
        print(f"[C168] Gate Check: Sanity={res['passed_sanity']}, Kill={res['passed_kill']}")
        print("=" * 60)
    else:
        print(f"Running Single-Seed Evaluation for C168 (seed={args.seed})...")
        res = evaluate_c168(config, seed=args.seed)
        print("=" * 60)
        print(f"[C168] Topoisomerase Unknotting Acc: {res['acc_topo']:.4f}")
        print(f"[C168] Standard Attention: {res['acc_std']:.4f}")
        print(f"[C168] Sanity Check (0 knots): {res['acc_sanity']:.4f}")
        print(f"[C168] Lethal Control 1 (Anti-Topo Lock): {res['acc_anti']:.4f} (Chance: {res['chance_baseline']:.4f})")
        print(f"[C168] Lethal Control 2 (Scrambled Cuts): {res['acc_scram']:.4f} (Chance: {res['chance_baseline']:.4f})")
        print(f"[C168] Gate Check: Sanity={res['passed_sanity']}, Kill={res['passed_kill']}")
        print("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
