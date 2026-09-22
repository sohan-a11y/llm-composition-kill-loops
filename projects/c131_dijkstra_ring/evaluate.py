"""
Evaluation Harness for C131: Self-Stabilizing Dijkstra-Ring Virtual Token Memory
==============================================================================
Evaluates:
1. Sanity Check: Clean execution from legitimate state (P=1 initial) achieves >0.85 accuracy.
2. Multi-Seed Fault-Tolerant Self-Stabilization: Random corrupted states converge to P=1 in <= 5 steps
   and achieve >0.90 accuracy across seeds [42, 137].
3. Lethal Control 1: Deadlock Anti-Dijkstra mode (destroys leader rule, collapses to chance 0.2000).
4. Lethal Control 2: Topology Scramble mode (destroys spatial invariant, collapses to chance 0.2000).
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

try:
    from module import DijkstraMemoryClassifier
except ImportError:
    from projects.c131_dijkstra_ring.module import DijkstraMemoryClassifier


Tuple_Data = Tuple[torch.Tensor, torch.Tensor]


def generate_task_data(
    batch_size: int = 64,
    num_classes: int = 5,
    seq_len: int = 6,
    device: str = "cpu"
) -> Tuple_Data:
    """
    State tracking sequence:
    Token 0 carries the target class payload c in {0..num_classes-1}.
    Tokens 1..seq_len-1 are neutral circulation step tokens (ID 10).
    The Dijkstra ring must route the payload through circulating privileges.
    """
    targets = torch.randint(0, num_classes, (batch_size,), device=device)
    tokens = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    tokens[:, 0] = targets
    tokens[:, 1:] = 10
    return tokens, targets


def train_dijkstra_model(
    seed: int,
    epochs: int = 100,
    batch_size: int = 64,
    seq_len: int = 6,
    num_classes: int = 5,
    d_model: int = 32,
    K: int = 4,
    M: int = 5,
    device: str = "cpu"
) -> DijkstraMemoryClassifier:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DijkstraMemoryClassifier(
        num_tokens=20,
        d_model=d_model,
        num_classes=num_classes,
        K=K,
        M=M
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        tokens, targets = generate_task_data(batch_size=batch_size, num_classes=num_classes, seq_len=seq_len, device=device)
        # Train with random initial state corruptions so model is robust to arbitrary states
        if epoch % 2 == 0:
            s_init = torch.randint(0, M, (batch_size, K), device=device)
        else:
            s_init = None

        optimizer.zero_grad()
        logits, _, _ = model(tokens, s_init=s_init, control_mode="normal")
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    return model


def evaluate_single_seed(
    seed: int,
    K: int = 4,
    M: int = 5,
    seq_len: int = 6,
    device: str = "cpu"
) -> Dict[str, float]:
    model = train_dijkstra_model(seed=seed, K=K, M=M, seq_len=seq_len, device=device)
    model.eval()

    with torch.no_grad():
        tokens, targets = generate_task_data(batch_size=500, seq_len=seq_len, device=device)

        # 1. Sanity Check (Clean Execution, legitimate state)
        s_logits, _, _ = model(tokens, s_init=None, control_mode="normal")
        sanity_acc = (s_logits.argmax(dim=-1) == targets).float().mean().item()

        # 2. Corrupted State Initialization (Fault Injection):
        s_corrupt = torch.randint(0, M, (500, K), device=device)
        c_logits, c_priv_hist, _ = model(tokens, s_init=s_corrupt, control_mode="normal")
        fault_acc = (c_logits.argmax(dim=-1) == targets).float().mean().item()

        # Measure Mutex Convergence rate: proportion of samples with exactly 1 privilege at step 5
        final_priv_count = c_priv_hist[-1].sum(dim=-1)
        mutex_convergence = (final_priv_count == 1).float().mean().item()

        # 3. Lethal Control 1: Deadlock Mode (Anti-Dijkstra rule)
        k1_logits, _, _ = model(tokens, s_init=s_corrupt, control_mode="kill_deadlock")
        deadlock_acc = (k1_logits.argmax(dim=-1) == targets).float().mean().item()

        # 4. Lethal Control 2: Scramble Mode (Destroys spatial invariants)
        k2_logits, _, _ = model(tokens, s_init=s_corrupt, control_mode="kill_scramble")
        scramble_acc = (k2_logits.argmax(dim=-1) == targets).float().mean().item()

    return {
        "seed": seed,
        "sanity_acc": sanity_acc,
        "fault_acc": fault_acc,
        "mutex_convergence": mutex_convergence,
        "deadlock_acc": deadlock_acc,
        "scramble_acc": scramble_acc
    }


def run_full_evaluation(seeds: List[int] = [42, 137], output_dir: str = "logs") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    results = []

    print("=" * 70)
    print("C131: Self-Stabilizing Dijkstra-Ring Virtual Token Memory Evaluation")
    print("=" * 70)

    for seed in seeds:
        print(f"\n--- Evaluating Seed {seed} ---")
        metrics = evaluate_single_seed(seed=seed)
        print(f"  Sanity Check (Clean Execution):       Acc = {metrics['sanity_acc']:.4f}")
        print(f"  Corrupted Fault-Injection:             Acc = {metrics['fault_acc']:.4f}, Mutex Conv = {metrics['mutex_convergence']:.4f}")
        print(f"  Lethal Control 1 (Deadlock Mode):      Acc = {metrics['deadlock_acc']:.4f} (Chance: 0.2000)")
        print(f"  Lethal Control 2 (Scramble Mode):      Acc = {metrics['scramble_acc']:.4f} (Chance: 0.2000)")
        results.append(metrics)

    avg_sanity = float(np.mean([r["sanity_acc"] for r in results]))
    avg_fault = float(np.mean([r["fault_acc"] for r in results]))
    std_fault = float(np.std([r["fault_acc"] for r in results]))
    avg_mutex_conv = float(np.mean([r["mutex_convergence"] for r in results]))
    avg_deadlock = float(np.mean([r["deadlock_acc"] for r in results]))
    avg_scramble = float(np.mean([r["scramble_acc"] for r in results]))

    passed_sanity = avg_sanity >= 0.85
    passed_fault = avg_fault >= 0.90
    passed_mutex_conv = avg_mutex_conv >= 0.95
    passed_kill_deadlock = avg_deadlock <= 0.30
    passed_kill_scramble = avg_scramble <= 0.30

    overall_pass = (
        passed_sanity and
        passed_fault and
        passed_mutex_conv and
        passed_kill_deadlock and
        passed_kill_scramble
    )

    summary = {
        "project": "C131_dijkstra_ring",
        "seeds": seeds,
        "avg_sanity_acc": avg_sanity,
        "avg_fault_acc": avg_fault,
        "std_fault_acc": std_fault,
        "avg_mutex_convergence": avg_mutex_conv,
        "avg_deadlock_acc": avg_deadlock,
        "avg_scramble_acc": avg_scramble,
        "chance_baseline": 0.2000,
        "passed_sanity": passed_sanity,
        "passed_fault_tolerance": passed_fault,
        "passed_mutex_convergence": passed_mutex_conv,
        "passed_kill_controls": passed_kill_deadlock and passed_kill_scramble,
        "overall_pass": overall_pass,
        "seed_details": results
    }

    log_path = os.path.join(output_dir, "metrics_summary.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("EMPIRICAL VALIDATION SUMMARY:")
    print(f"  Sanity Check (Clean Execution):       {avg_sanity:.4f}  (Threshold >= 0.85: {'PASS' if passed_sanity else 'FAIL'})")
    print(f"  Fault Robustness (Corrupted Init):    {avg_fault:.4f} +/- {std_fault:.4f}  (Threshold >= 0.90: {'PASS' if passed_fault else 'FAIL'})")
    print(f"  Mutex Convergence (P=1 Rate):         {avg_mutex_conv:.4f}  (Threshold >= 0.95: {'PASS' if passed_mutex_conv else 'FAIL'})")
    print(f"  Kill Control 1 (Deadlock):            {avg_deadlock:.4f}  (Chance: 0.2000: {'COLLAPSED' if passed_kill_deadlock else 'FAIL'})")
    print(f"  Kill Control 2 (Scramble):            {avg_scramble:.4f}  (Chance: 0.2000: {'COLLAPSED' if passed_kill_scramble else 'FAIL'})")
    print(f"  OVERALL RESULT:                       {'PASSED ALL GATES' if overall_pass else 'FAILED'}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137])
    parser.add_argument("--output_dir", type=str, default="logs")
    args = parser.parse_args()

    run_full_evaluation(seeds=args.seeds, output_dir=args.output_dir)
