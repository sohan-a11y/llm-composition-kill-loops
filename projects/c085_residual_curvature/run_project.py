"""
Main entry point for C085: Residual Curvature Hallucination Alarm
Runs autonomous self-improving loop across generations.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from engine.self_improving_loop import SelfImprovingLoop
from projects.c085_residual_curvature.eval_curvature import (
    evaluate_curvature_alarm,
    sanity_check_curvature,
    kill_control_curvature
)


def mutate_curvature_config(current_config, best_config, generation):
    """Adaptive search: calibrate alarm threshold for optimal TPR/FPR trade-off."""
    new_cfg = dict(best_config)
    thresholds = [0.15, 0.18, 0.22]
    new_cfg["threshold"] = thresholds[generation % len(thresholds)]
    return new_cfg


def main():
    initial_config = {
        "n_samples": 200,
        "n_layers": 8,
        "d_model": 64,
        "threshold": 0.18
    }

    loop = SelfImprovingLoop(
        experiment_name="c085_residual_curvature_alarm",
        initial_config=initial_config,
        train_and_eval_fn=evaluate_curvature_alarm,
        sanity_check_fn=sanity_check_curvature,
        kill_control_fn=kill_control_curvature,
        mutate_config_fn=mutate_curvature_config,
        log_dir="logs/c085_curvature",
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
