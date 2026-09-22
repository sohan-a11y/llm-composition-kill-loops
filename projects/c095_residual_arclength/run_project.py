"""
Main entry point for C095: Residual Arc-Length Deliberation Metric
Runs autonomous self-improving loop across generations.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from engine.self_improving_loop import SelfImprovingLoop
from projects.c095_residual_arclength.eval_calibration import (
    evaluate_arclength_calibration,
    sanity_check_arclength,
    kill_control_arclength
)


def mutate_arclength_config(current_config, best_config, generation):
    new_cfg = dict(best_config)
    thresholds = [1.25, 1.35, 1.45]
    new_cfg["threshold"] = thresholds[generation % len(thresholds)]
    return new_cfg


def main():
    initial_config = {
        "n_samples": 150,
        "n_layers": 8,
        "threshold": 1.35
    }

    loop = SelfImprovingLoop(
        experiment_name="c095_residual_arclength_deliberation",
        initial_config=initial_config,
        train_and_eval_fn=evaluate_arclength_calibration,
        sanity_check_fn=sanity_check_arclength,
        kill_control_fn=kill_control_arclength,
        mutate_config_fn=mutate_arclength_config,
        log_dir="logs/c095_arclength",
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
