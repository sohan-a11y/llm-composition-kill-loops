"""
Main entry point for C132: Soliton Wave-Packet Dispersion-Balanced Residual Stream.
Runs autonomous self-improving loop across parameter generations.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from engine.self_improving_loop import SelfImprovingLoop
from projects.c132_soliton_residual.eval_soliton import (
    evaluate_soliton,
    sanity_check_soliton,
    kill_control_soliton
)


def mutate_soliton_config(current_config, best_config, generation, **kwargs):
    """Adaptive parameter search: calibrate dispersion beta and dissipation nu."""
    new_cfg = dict(best_config)
    beta_candidates = [0.08, 0.10, 0.12, 0.15]
    nu_candidates = [0.02, 0.03, 0.04]
    new_cfg["beta"] = beta_candidates[generation % len(beta_candidates)]
    new_cfg["nu"] = nu_candidates[generation % len(nu_candidates)]
    return new_cfg


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    initial_config = {
        "n_samples": 250,
        "n_layers": 16,
        "d_model": 32,
        "n_classes": 5,
        "beta": 0.10,
        "nu": 0.03,
        "epochs": 25,
        "lr": 0.01
    }

    loop = SelfImprovingLoop(
        experiment_name="c132_soliton_residual",
        initial_config=initial_config,
        train_and_eval_fn=evaluate_soliton,
        sanity_check_fn=sanity_check_soliton,
        kill_control_fn=kill_control_soliton,
        mutate_config_fn=mutate_soliton_config,
        log_dir="logs/c132_soliton",
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
