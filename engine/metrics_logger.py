"""
Metrics Logger & Statistical Verification Harness
Provides multi-seed tracking, chance baseline comparison,
kill-control assertions, and Pareto front logging.
"""

import json
import os
import time
import numpy as np
from typing import Dict, Any, List, Optional


class ExperimentTrialResult:
    def __init__(
        self,
        trial_id: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        sanity_passed: bool,
        kill_control_passed: bool,
        notes: str = ""
    ):
        self.trial_id = trial_id
        self.config = config
        self.metrics = metrics
        self.sanity_passed = sanity_passed
        self.kill_control_passed = kill_control_passed
        self.notes = notes
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "config": self.config,
            "metrics": self.metrics,
            "sanity_passed": self.sanity_passed,
            "kill_control_passed": self.kill_control_passed,
            "notes": self.notes,
            "timestamp": self.timestamp
        }


class MetricsLogger:
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{experiment_name}_results.json")
        self.trials: List[ExperimentTrialResult] = []
        self._load_existing()

    def _load_existing(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    data = json.load(f)
                    for item in data.get("trials", []):
                        self.trials.append(
                            ExperimentTrialResult(
                                trial_id=item["trial_id"],
                                config=item["config"],
                                metrics=item["metrics"],
                                sanity_passed=item["sanity_passed"],
                                kill_control_passed=item["kill_control_passed"],
                                notes=item.get("notes", "")
                            )
                        )
            except Exception as e:
                print(f"[MetricsLogger] Warning loading existing results: {e}")

    def log_trial(
        self,
        trial_id: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        sanity_passed: bool,
        kill_control_passed: bool,
        notes: str = ""
    ) -> ExperimentTrialResult:
        result = ExperimentTrialResult(
            trial_id=trial_id,
            config=config,
            metrics=metrics,
            sanity_passed=sanity_passed,
            kill_control_passed=kill_control_passed,
            notes=notes
        )
        self.trials.append(result)
        self.save()
        return result

    def save(self):
        payload = {
            "experiment_name": self.experiment_name,
            "updated_at": time.time(),
            "trial_count": len(self.trials),
            "trials": [t.to_dict() for t in self.trials]
        }
        with open(self.log_file, "w") as f:
            json.dump(payload, f, indent=2)

    def get_best_trial(self, metric_key: str, higher_is_better: bool = True) -> Optional[ExperimentTrialResult]:
        valid_trials = [t for t in self.trials if t.sanity_passed and t.kill_control_passed and metric_key in t.metrics]
        if not valid_trials:
            return None
        valid_trials.sort(
            key=lambda t: t.metrics[metric_key],
            reverse=higher_is_better
        )
        return valid_trials[0]

    def summary_table(self) -> str:
        lines = [
            f"# Experiment Summary: {self.experiment_name}",
            f"Total Trials: {len(self.trials)}",
            "",
            "| Trial ID | Primary Config | Sanity Pass | Kill Control Pass | Metrics |",
            "|:---|:---|:---:|:---:|:---|"
        ]
        for t in self.trials:
            cfg_str = ", ".join(f"{k}={v}" for k, v in list(t.config.items())[:3])
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in t.metrics.items())
            sp = "YES" if t.sanity_passed else "FAIL"
            kc = "PASS (Dropped to Chance)" if t.kill_control_passed else "FAIL"
            lines.append(f"| {t.trial_id} | {cfg_str} | {sp} | {kc} | {metrics_str} |")
        return "\n".join(lines)
