"""
Autonomous Self-Improving Experiment Loop Harness
Executes continuous iterative cycles of:
1. Hypothesis formulation & parameter mutation
2. Multi-seed training & evaluation on consumer GPU / CPU
3. Mandatory Sanity Check verification (trivial case must pass)
4. Mandatory Lethal Kill-Control verification (control must drop to chance)
5. Automated Error-Attribution & mechanistic critique
6. Adaptive Pareto-optimal parameter selection for subsequent generations
"""

import os
import sys
import copy
import time
import random
from typing import Dict, Any, Callable, List, Tuple
import numpy as np

from engine.metrics_logger import MetricsLogger, ExperimentTrialResult


class SelfImprovingLoop:
    def __init__(
        self,
        experiment_name: str,
        initial_config: Dict[str, Any],
        train_and_eval_fn: Callable[[Dict[str, Any], int], Dict[str, float]],
        sanity_check_fn: Callable[[Dict[str, Any]], bool],
        kill_control_fn: Callable[[Dict[str, Any]], Tuple[bool, float, float]],
        mutate_config_fn: Callable[[Dict[str, Any], Dict[str, Any], int], Dict[str, Any]],
        log_dir: str = "logs/self_improving",
        primary_metric: str = "acc",
        higher_is_better: bool = True,
        seeds: List[int] = [42, 137, 2024],
        max_generations: int = 5
    ):
        self.experiment_name = experiment_name
        self.current_config = copy.deepcopy(initial_config)
        self.train_and_eval_fn = train_and_eval_fn
        self.sanity_check_fn = sanity_check_fn
        self.kill_control_fn = kill_control_fn
        self.mutate_config_fn = mutate_config_fn
        self.logger = MetricsLogger(log_dir, experiment_name)
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better
        self.seeds = seeds
        self.max_generations = max_generations
        self.history: List[Dict[str, Any]] = []

    def run_cycle(self) -> Dict[str, Any]:
        sys.stdout.reconfigure(encoding='utf-8')
        print(f"\n==================================================================")
        print(f"[START] SELF-IMPROVING OPTIMIZATION LOOP: {self.experiment_name}")
        print(f"==================================================================")

        best_score = -float('inf') if self.higher_is_better else float('inf')
        best_generation_config = copy.deepcopy(self.current_config)

        for gen in range(1, self.max_generations + 1):
            trial_id = f"{self.experiment_name}_gen_{gen}"
            print(f"\n--- [Generation {gen}/{self.max_generations}] Trial ID: {trial_id} ---")
            print(f"Active Config: {self.current_config}")

            # 1. Mandatory Sanity Check (Trivial harness check must pass)
            print("[Step 1] Running Sanity Check on trivial baseline...")
            sanity_pass = self.sanity_check_fn(self.current_config)
            if not sanity_pass:
                print("[FAIL] SANITY CHECK FAILED! Harness is broken or trivial case collapsed.")
                self.logger.log_trial(
                    trial_id=trial_id,
                    config=self.current_config,
                    metrics={"error": -1.0},
                    sanity_passed=False,
                    kill_control_passed=False,
                    notes="Sanity check failed: trivial harness check did not hit passing threshold."
                )
                break
            print("[PASS] Sanity check passed.")

            # 2. Multi-Seed Training & Evaluation
            print(f"[Step 2] Running Multi-Seed Evaluation across seeds {self.seeds}...")
            seed_metrics: List[Dict[str, float]] = []
            for seed in self.seeds:
                m = self.train_and_eval_fn(self.current_config, seed)
                seed_metrics.append(m)

            # Aggregate metrics across seeds
            agg_metrics: Dict[str, float] = {}
            for k in seed_metrics[0].keys():
                vals = [sm[k] for sm in seed_metrics if k in sm]
                agg_metrics[f"{k}_mean"] = float(np.mean(vals))
                agg_metrics[f"{k}_std"] = float(np.std(vals))
            
            mean_primary = agg_metrics.get(f"{self.primary_metric}_mean", 0.0)
            std_primary = agg_metrics.get(f"{self.primary_metric}_std", 0.0)
            print(f"[METRICS] Across {len(self.seeds)} seeds: {self.primary_metric} = {mean_primary:.4f} +/- {std_primary:.4f}")

            # 3. Mandatory Lethal Kill-Control Verification
            print("[Step 3] Executing Lethal Kill-Control Arm...")
            kill_passed, control_acc, chance_baseline = self.kill_control_fn(self.current_config)
            if kill_passed:
                print(f"[PASS] Kill-Control PASSED: Control arm dropped to chance baseline ({control_acc:.4f} vs chance {chance_baseline:.4f})")
            else:
                print(f"[FAIL] Kill-Control FAILED: Control arm did not collapse to chance ({control_acc:.4f} vs chance {chance_baseline:.4f})! Mechanism is cheating/leaking.")

            # 4. Error Attribution & Critique
            critique = self._attribute_error(agg_metrics, kill_passed)
            print(f"[CRITIQUE] Mechanistic Critique: {critique}")

            # 5. Log Trial
            trial_res = self.logger.log_trial(
                trial_id=trial_id,
                config=copy.deepcopy(self.current_config),
                metrics=agg_metrics,
                sanity_passed=sanity_pass,
                kill_control_passed=kill_passed,
                notes=critique
            )

            # 6. Check for Improvement & Update Best Config
            is_better = (mean_primary > best_score) if self.higher_is_better else (mean_primary < best_score)
            if is_better and sanity_pass and kill_passed:
                best_score = mean_primary
                best_generation_config = copy.deepcopy(self.current_config)
                print(f"[BEST] NEW BEST CONFIGURATION FOUND at Gen {gen}! ({self.primary_metric}={best_score:.4f})")

            # 7. Adaptive Mutation for Next Generation
            if gen < self.max_generations:
                print("[Step 4] Mutating parameters for next generation...")
                self.current_config = self.mutate_config_fn(
                    current_config=self.current_config,
                    best_config=best_generation_config,
                    generation=gen
                )

        print("\n==================================================================")
        print(f"[COMPLETE] SELF-IMPROVING LOOP COMPLETE. Best {self.primary_metric}: {best_score:.4f}")
        print(f"Optimal Configuration: {best_generation_config}")
        print("==================================================================")
        return {
            "best_score": best_score,
            "best_config": best_generation_config,
            "summary_table": self.logger.summary_table()
        }

    def _attribute_error(self, metrics: Dict[str, float], kill_passed: bool) -> str:
        if not kill_passed:
            return "LETHAL COLLAPSE: Negative control arm failed to drop to chance level; information leakage suspected."
        loss_val = metrics.get("loss_mean", 0.0)
        acc_val = metrics.get(f"{self.primary_metric}_mean", 0.0)
        if acc_val < 0.3:
            return f"Low composition accuracy ({acc_val:.4f}). Representation may suffer from severe crosstalk or insufficient capacity."
        elif acc_val < 0.8:
            return f"Moderate composition ({acc_val:.4f}). Intermediate states partially preserved; tuning channel allocation or learning rate recommended."
        else:
            return f"Strong compositional retention ({acc_val:.4f}). Robust generalization across seeds with verified kill control."
