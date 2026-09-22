"""
Main entry point for C094: Cross-Layer Residual Correlation Probe
Runs autonomous self-improving loop across generations.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from engine.self_improving_loop import SelfImprovingLoop
from projects.c094_residual_correlation.eval_correlation import (
    evaluate_residual_correlation,
    sanity_check_correlation,
    kill_control_correlation
)


def mutate_correlation_config(current_config, best_config, generation):
    new_cfg = dict(best_config)
    thresholds = [0.40, 0.50, 0.60]
    new_cfg["threshold"] = thresholds[generation % len(thresholds)]
    return new_cfg


def main():
    initial_config = {
        "n_sequences": 120,
        "seq_len": 6,
        "n_layers": 8,
        "threshold": 0.50
    }

    loop = SelfImprovingLoop(
        experiment_name="c094_residual_correlation_probe",
        initial_config=initial_config,
        train_and_eval_fn=evaluate_residual_correlation,
        sanity_check_fn=sanity_check_correlation,
        kill_control_fn=kill_control_correlation,
        mutate_config_fn=mutate_correlation_config,
        log_dir="logs/c094_correlation",
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
