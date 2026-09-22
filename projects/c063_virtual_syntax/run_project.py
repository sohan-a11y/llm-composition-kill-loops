"""
Main entry point for C063: Virtual Syntax Skipping
Runs autonomous self-improving loop across generations.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from engine.self_improving_loop import SelfImprovingLoop
from projects.c063_virtual_syntax.eval_grammar_tax import (
    evaluate_virtual_syntax,
    sanity_check_syntax,
    kill_control_syntax
)


def mutate_syntax_config(current_config, best_config, generation):
    new_cfg = dict(best_config)
    noise_levels = [0.25, 0.35, 0.45]
    new_cfg["noise_level"] = noise_levels[generation % len(noise_levels)]
    return new_cfg


def main():
    initial_config = {
        "n_samples": 150,
        "noise_level": 0.35
    }

    loop = SelfImprovingLoop(
        experiment_name="c063_virtual_syntax_skipping",
        initial_config=initial_config,
        train_and_eval_fn=evaluate_virtual_syntax,
        sanity_check_fn=sanity_check_syntax,
        kill_control_fn=kill_control_syntax,
        mutate_config_fn=mutate_syntax_config,
        log_dir="logs/c063_syntax",
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
