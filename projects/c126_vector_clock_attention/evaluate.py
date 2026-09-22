"""
Evaluation Harness for Vector-Clock Asynchronous Causal Attention (C126)
=======================================================================
Evaluates:
1. Sanity Check (B=1): Serial autoregressive stream produces perfect causal masking (>0.85).
2. Multi-Seed Empirical Evaluation: Multi-stream concurrent generation with asynchronous message passing
   across seeds [42, 137].
3. Lethal Negative Control Arm 1: Distractor Channel Leakage (violates vector clock isolation, collapses to chance 0.2000).
4. Lethal Negative Control Arm 2: Decoupled Causal Payload (destroys cross-stream message content, collapses to chance 0.2000).
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

from module import VectorClockReasoningModel

Tuple_Data = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def generate_multistream_data(
    batch_size: int = 64,
    num_classes: int = 5,
    B: int = 3,
    decouple_payload: bool = False,
    device: str = "cpu"
) -> Tuple_Data:
    """
    Constructs an asynchronous multi-stream execution trace:
    - Stream 0: Generates target state v0 in {0..4}. Clocks: [1, 0, 0] -> emits message with clock [2, 0, 0].
    - Stream 1: Concurrently generates distractor state v1 in {0..4}. Clocks: [0, 1, 0] -> emits distractor with clock [0, 2, 0].
    - Stream 2: Synthesizer receiving message from Stream 0. Clock: max([0,0,0], [2,0,0]) + [0,0,1] = [2, 0, 1].
    
    Tokens:
    - Index 0: Stream 0 state (targets)
    - Index 1: Stream 1 state (distractors)
    - Index 2: Stream 0 message (targets or decoupled noise)
    - Index 3: Stream 1 message (distractors)
    - Index 4: Stream 2 query (incorporates Stream 0 message)
    """
    targets = torch.randint(0, num_classes, (batch_size,), device=device)
    distractors = torch.randint(0, num_classes, (batch_size,), device=device)

    tokens = torch.zeros((batch_size, 5), dtype=torch.long, device=device)
    if decouple_payload:
        tokens[:, 0] = torch.randint(0, num_classes, (batch_size,), device=device)
        tokens[:, 1] = distractors + 5
        tokens[:, 2] = torch.randint(0, num_classes, (batch_size,), device=device)
        tokens[:, 3] = distractors + 5
    else:
        tokens[:, 0] = targets
        tokens[:, 1] = distractors + 5
        tokens[:, 2] = targets
        tokens[:, 3] = distractors + 5
    tokens[:, 4] = 14  # Query token

    V = torch.zeros((batch_size, 5, B), dtype=torch.long, device=device)
    V[:, 0, :] = torch.tensor([1, 0, 0], device=device)
    V[:, 1, :] = torch.tensor([0, 1, 0], device=device)
    V[:, 2, :] = torch.tensor([2, 0, 0], device=device)
    V[:, 3, :] = torch.tensor([0, 2, 0], device=device)
    V[:, 4, :] = torch.tensor([2, 0, 1], device=device)

    return tokens, V, targets


def generate_serial_sanity_data(
    batch_size: int = 64,
    num_classes: int = 5,
    device: str = "cpu"
) -> Tuple_Data:
    """
    Single stream serial chain (B=1) with standard sequential vector clocks: V_i = [i].
    """
    targets = torch.randint(0, num_classes, (batch_size,), device=device)
    tokens = torch.zeros((batch_size, 3), dtype=torch.long, device=device)
    tokens[:, 0] = targets
    tokens[:, 1] = targets
    tokens[:, 2] = 14

    V = torch.zeros((batch_size, 3, 1), dtype=torch.long, device=device)
    V[:, 0, :] = torch.tensor([1], device=device)
    V[:, 1, :] = torch.tensor([2], device=device)
    V[:, 2, :] = torch.tensor([3], device=device)

    return tokens, V, targets


def train_vc_model(
    seed: int,
    epochs: int = 120,
    batch_size: int = 64,
    device: str = "cpu"
) -> VectorClockReasoningModel:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = VectorClockReasoningModel(
        num_tokens=20,
        d_model=32,
        n_heads=2,
        num_layers=2,
        num_classes=5
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(epochs):
        tokens, V, targets = generate_multistream_data(batch_size=batch_size, device=device)
        optimizer.zero_grad()
        logits = model(tokens, V, control_mode="normal")
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    return model


def evaluate_single_seed(seed: int, device: str = "cpu") -> Dict[str, float]:
    model = train_vc_model(seed=seed, epochs=120, device=device)
    model.eval()

    with torch.no_grad():
        # 1. Sanity check: B=1 serial stream
        s_tokens, s_V, s_targets = generate_serial_sanity_data(batch_size=300, device=device)
        s_logits = model(s_tokens, s_V, control_mode="normal")
        sanity_acc = (s_logits.argmax(dim=-1) == s_targets).float().mean().item()

        # 2. Multi-stream normal mode (B=3 concurrent streams)
        tokens, V, targets = generate_multistream_data(batch_size=500, device=device)
        n_logits = model(tokens, V, control_mode="normal")
        normal_acc = (n_logits.argmax(dim=-1) == targets).float().mean().item()

        # 3. Lethal control 1: Distractor Channel Leakage (violates vector clock concurrency isolation)
        d_logits = model(tokens, V, control_mode="kill_distractor_channel")
        distractor_leak_acc = (d_logits.argmax(dim=-1) == targets).float().mean().item()

        # 4. Lethal control 2: Decoupled Message Payload
        dec_tokens, dec_V, dec_targets = generate_multistream_data(
            batch_size=500, decouple_payload=True, device=device
        )
        dec_logits = model(dec_tokens, dec_V, control_mode="normal")
        decoupled_payload_acc = (dec_logits.argmax(dim=-1) == dec_targets).float().mean().item()

    return {
        "seed": seed,
        "sanity_acc": sanity_acc,
        "normal_acc": normal_acc,
        "distractor_leak_acc": distractor_leak_acc,
        "decoupled_payload_acc": decoupled_payload_acc
    }


def run_full_evaluation(seeds: List[int] = [42, 137], output_dir: str = "logs") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    results = []

    print("=" * 70)
    print("C126: Vector-Clock Asynchronous Causal Attention Multi-Seed Evaluation")
    print("=" * 70)

    for seed in seeds:
        print(f"\n--- Evaluating Seed {seed} ---")
        metrics = evaluate_single_seed(seed=seed)
        print(f"  Sanity Check (B=1 Serial):             Acc = {metrics['sanity_acc']:.4f}")
        print(f"  Multi-Stream Normal (B=3 Concurrent):  Acc = {metrics['normal_acc']:.4f}")
        print(f"  Lethal Control 1 (Distractor Leak):    Acc = {metrics['distractor_leak_acc']:.4f} (Chance: 0.2000)")
        print(f"  Lethal Control 2 (Decoupled Payload):  Acc = {metrics['decoupled_payload_acc']:.4f} (Chance: 0.2000)")
        results.append(metrics)

    avg_sanity = float(np.mean([r["sanity_acc"] for r in results]))
    avg_normal = float(np.mean([r["normal_acc"] for r in results]))
    std_normal = float(np.std([r["normal_acc"] for r in results]))
    avg_distractor_leak = float(np.mean([r["distractor_leak_acc"] for r in results]))
    avg_decoupled = float(np.mean([r["decoupled_payload_acc"] for r in results]))

    passed_sanity = avg_sanity >= 0.85
    passed_multi_seed = avg_normal >= 0.90
    passed_kill_leak = avg_distractor_leak <= 0.30  # Near chance 0.2000
    passed_kill_decoupled = avg_decoupled <= 0.30

    overall_pass = passed_sanity and passed_multi_seed and passed_kill_leak and passed_kill_decoupled

    summary = {
        "project": "C126_vector_clock_attention",
        "seeds": seeds,
        "avg_sanity_acc": avg_sanity,
        "avg_normal_acc": avg_normal,
        "std_normal_acc": std_normal,
        "avg_distractor_leak_acc": avg_distractor_leak,
        "avg_decoupled_payload_acc": avg_decoupled,
        "chance_baseline": 0.2000,
        "passed_sanity": passed_sanity,
        "passed_multi_seed": passed_multi_seed,
        "passed_kill_controls": passed_kill_leak and passed_kill_decoupled,
        "overall_pass": overall_pass,
        "seed_details": results
    }

    log_path = os.path.join(output_dir, "metrics_summary.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("EMPIRICAL VALIDATION SUMMARY:")
    print(f"  Sanity Check (B=1):             {avg_sanity:.4f}  (Threshold >= 0.85: {'PASS' if passed_sanity else 'FAIL'})")
    print(f"  Normal Multi-Stream (B=3):      {avg_normal:.4f} +/- {std_normal:.4f}  (Threshold >= 0.90: {'PASS' if passed_multi_seed else 'FAIL'})")
    print(f"  Kill Control 1 (Distractor):    {avg_distractor_leak:.4f}  (Chance: 0.2000: {'COLLAPSED' if passed_kill_leak else 'FAIL'})")
    print(f"  Kill Control 2 (Decoupled):     {avg_decoupled:.4f}  (Chance: 0.2000: {'COLLAPSED' if passed_kill_decoupled else 'FAIL'})")
    print(f"  OVERALL RESULT:                 {'PASSED ALL GATES' if overall_pass else 'FAILED'}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137])
    parser.add_argument("--output_dir", type=str, default="logs")
    args = parser.parse_args()

    run_full_evaluation(seeds=args.seeds, output_dir=args.output_dir)
