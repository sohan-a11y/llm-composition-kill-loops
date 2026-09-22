"""
Main entry point for C124: Michaelis-Menten Enzymatic Substrate Attention
Runs autonomous self-improving loop across generations.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from engine.self_improving_loop import SelfImprovingLoop
from projects.c124_enzymatic_attention.eval_enzymatic import (
    evaluate_enzymatic,
    sanity_check_enzymatic,
    kill_control_enzymatic
)


def mutate_enzymatic_config(current_config, best_config, generation, **kwargs):
    """Adaptive search: calibrate Michaelis-Menten constant Km and catalytic velocity Vmax."""
    new_cfg = dict(best_config)
    km_candidates = [0.8, 1.0, 1.2, 1.5]
    vmax_candidates = [1.0, 1.2, 1.5]
    new_cfg["km"] = km_candidates[generation % len(km_candidates)]
    new_cfg["vmax"] = vmax_candidates[generation % len(vmax_candidates)]
    return new_cfg


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    initial_config = {
        "n_samples": 250,
        "n_distractors": 12,
        "n_classes": 5,
        "km": 1.0,
        "vmax": 1.0
    }

    loop = SelfImprovingLoop(
        experiment_name="c124_enzymatic_attention",
        initial_config=initial_config,
        train_and_eval_fn=evaluate_enzymatic,
        sanity_check_fn=sanity_check_enzymatic,
        kill_control_fn=kill_control_enzymatic,
        mutate_config_fn=mutate_enzymatic_config,
        log_dir="logs/c124_enzymatic",
        primary_metric="acc",
        higher_is_better=True,
        seeds=[42, 137],
        max_generations=2
    )

    results = loop.run_cycle()
    print("\nFinal Results Table:\n")
    print(results["summary_table"])


if __name__ == "__main__":
    main()
