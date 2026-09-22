"""
Main entry point for C134: Non-Archimedean p-adic Ultrametric Tree Cache.
Runs autonomous self-improving loop across parameter generations.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from engine.self_improving_loop import SelfImprovingLoop
from projects.c134_padic_tree_cache.eval_padic import (
    evaluate_padic,
    sanity_check_padic,
    kill_control_padic
)


def mutate_padic_config(current_config, best_config, generation, **kwargs):
    """Adaptive parameter search: calibrate ultrametric penalty lambda and prime p."""
    new_cfg = dict(best_config)
    penalty_candidates = [8.0, 10.0, 12.0, 15.0]
    new_cfg["penalty"] = penalty_candidates[generation % len(penalty_candidates)]
    return new_cfg


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    initial_config = {
        "n_samples": 300,
        "d_model": 32,
        "n_classes": 5,
        "penalty": 10.0,
        "p": 2,
        "epochs": 25,
        "lr": 0.01
    }

    loop = SelfImprovingLoop(
        experiment_name="c134_padic_tree_cache",
        initial_config=initial_config,
        train_and_eval_fn=evaluate_padic,
        sanity_check_fn=sanity_check_padic,
        kill_control_fn=kill_control_padic,
        mutate_config_fn=mutate_padic_config,
        log_dir="logs/c134_padic",
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
