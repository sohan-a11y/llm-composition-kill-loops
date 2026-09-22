"""
Main entry point for C029: SSA Residual Channels
Runs autonomous self-improving loop across generations.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from engine.self_improving_loop import SelfImprovingLoop
from projects.c029_ssa_residual.train_ssa import run_ssa_trial, sanity_check_ssa, kill_control_ssa


def mutate_ssa_config(current_config, best_config, generation):
    """Mutate hyperparameters toward better compositional retention."""
    new_cfg = dict(best_config)
    # Adaptive search: adjust learning rate and hidden dimension
    lr_choices = [5e-4, 1e-3, 2e-3]
    d_model_choices = [64, 96, 128]

    new_cfg["lr"] = lr_choices[generation % len(lr_choices)]
    if generation >= 2:
        new_cfg["d_model"] = d_model_choices[min(generation - 1, len(d_model_choices) - 1)]
    return new_cfg


def main():
    initial_config = {
        "k": 3,
        "domain": 50,
        "layers": 4,
        "d_model": 64,
        "heads": 4,
        "lr": 1e-3,
        "epochs": 12,
        "ssa_mode": True
    }

    loop = SelfImprovingLoop(
        experiment_name="c029_ssa_residual_channels",
        initial_config=initial_config,
        train_and_eval_fn=run_ssa_trial,
        sanity_check_fn=sanity_check_ssa,
        kill_control_fn=kill_control_ssa,
        mutate_config_fn=mutate_ssa_config,
        log_dir="logs/c029_ssa",
        primary_metric="acc",
        higher_is_better=True,
        seeds=[42, 137],
        max_generations=3
    )

    results = loop.run_cycle()
    print("\nFinal Results Table:\n")
    print(results["summary_table"])


if __name__ == "__main__":
    main()
