"""
Continuous Meta-Loop Auto-Runner
Runs autonomous self-improving cycles across all flagship projects,
scaling difficulty, exploring hyperparameter frontiers, evaluating lethal controls,
and accumulating Pareto-optimal configurations.
"""

import sys
import os
import time
import json
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.self_improving_loop import SelfImprovingLoop
from projects.c029_ssa_residual.train_ssa import run_ssa_trial, sanity_check_ssa, kill_control_ssa
from projects.c085_residual_curvature.eval_curvature import evaluate_curvature_alarm, sanity_check_curvature, kill_control_curvature
from projects.c063_virtual_syntax.eval_grammar_tax import evaluate_virtual_syntax, sanity_check_syntax, kill_control_syntax
from projects.c094_residual_correlation.eval_correlation import evaluate_residual_correlation, sanity_check_correlation, kill_control_correlation
from projects.c095_residual_arclength.eval_calibration import evaluate_arclength_calibration, sanity_check_arclength, kill_control_arclength
from projects.c124_enzymatic_attention.eval_enzymatic import evaluate_enzymatic, sanity_check_enzymatic, kill_control_enzymatic
from projects.c132_soliton_residual.eval_soliton import evaluate_soliton, sanity_check_soliton, kill_control_soliton
from projects.c134_padic_tree_cache.eval_padic import evaluate_padic, sanity_check_padic, kill_control_padic
from projects.c126_vector_clock_attention.eval_vc import evaluate_vc, sanity_check_vc, kill_control_vc


def run_continuous_meta_loop(max_cycles: int = 10):
    sys.stdout.reconfigure(encoding='utf-8')
    log_dir = "logs/continuous_meta_loop"
    os.makedirs(log_dir, exist_ok=True)
    summary_file = os.path.join(log_dir, "meta_loop_summary.json")

    print("\n==================================================================")
    print(f"[START] CONTINUOUS META-LOOP AUTO-RUNNER (Max Cycles: {max_cycles})")
    print("==================================================================")

    master_history = []

    for cycle in range(1, max_cycles + 1):
        print(f"\n>>>>>>>>>>>>>>>>>>>> [CYCLE {cycle}/{max_cycles}] <<<<<<<<<<<<<<<<<<<<")
        cycle_start = time.time()
        cycle_results = {}

        # -------------------------------------------------------------
        # 1. Project C085: Residual Curvature Alarm (Scaling depths & thresholds)
        # -------------------------------------------------------------
        try:
            curv_threshold = 0.15 + (cycle * 0.02)
            curv_layers = 6 + (cycle * 2)
            print(f"\n[Running C085] Layers: {curv_layers}, Threshold: {curv_threshold:.2f}")
            loop_c085 = SelfImprovingLoop(
                experiment_name=f"c085_cycle_{cycle}",
                initial_config={"n_samples": 150, "n_layers": curv_layers, "d_model": 64, "threshold": curv_threshold},
                train_and_eval_fn=evaluate_curvature_alarm,
                sanity_check_fn=sanity_check_curvature,
                kill_control_fn=kill_control_curvature,
                mutate_config_fn=lambda current_config, best_config, generation: {**best_config, "threshold": best_config["threshold"] + 0.02 * (generation % 2 == 0)},
                log_dir=os.path.join(log_dir, "c085"),
                primary_metric="acc",
                seeds=[42 + cycle, 137 + cycle],
                max_generations=2
            )
            res_c085 = loop_c085.run_cycle()
            cycle_results["c085"] = {"best_acc": res_c085["best_score"], "config": res_c085["best_config"]}
        except Exception as e:
            print(f"[ERROR in C085]: {e}")
            traceback.print_exc()

        # -------------------------------------------------------------
        # 2. Project C063: Virtual Syntax Skipping (Scaling noise & complexity)
        # -------------------------------------------------------------
        try:
            noise_val = 0.20 + (cycle * 0.05)
            print(f"\n[Running C063] Noise Level: {noise_val:.2f}")
            loop_c063 = SelfImprovingLoop(
                experiment_name=f"c063_cycle_{cycle}",
                initial_config={"n_samples": 120, "noise_level": noise_val},
                train_and_eval_fn=evaluate_virtual_syntax,
                sanity_check_fn=sanity_check_syntax,
                kill_control_fn=kill_control_syntax,
                mutate_config_fn=lambda current_config, best_config, generation: {**best_config, "noise_level": min(0.60, best_config["noise_level"] + 0.05)},
                log_dir=os.path.join(log_dir, "c063"),
                primary_metric="acc",
                seeds=[42 + cycle, 137 + cycle],
                max_generations=2
            )
            res_c063 = loop_c063.run_cycle()
            cycle_results["c063"] = {"best_acc": res_c063["best_score"], "config": res_c063["best_config"]}
        except Exception as e:
            print(f"[ERROR in C063]: {e}")
            traceback.print_exc()

        # -------------------------------------------------------------
        # 3. Project C094: Residual Correlation Probe (Scaling sequence lengths)
        # -------------------------------------------------------------
        try:
            seq_len = 5 + cycle
            print(f"\n[Running C094] Sequence Length: {seq_len}, Layers: 8")
            loop_c094 = SelfImprovingLoop(
                experiment_name=f"c094_cycle_{cycle}",
                initial_config={"n_sequences": 100, "seq_len": seq_len, "n_layers": 8, "threshold": 0.50},
                train_and_eval_fn=evaluate_residual_correlation,
                sanity_check_fn=sanity_check_correlation,
                kill_control_fn=kill_control_correlation,
                mutate_config_fn=lambda current_config, best_config, generation: {**best_config, "threshold": 0.45 if generation % 2 == 0 else 0.55},
                log_dir=os.path.join(log_dir, "c094"),
                primary_metric="acc",
                seeds=[42 + cycle, 137 + cycle],
                max_generations=2
            )
            res_c094 = loop_c094.run_cycle()
            cycle_results["c094"] = {"best_acc": res_c094["best_score"], "config": res_c094["best_config"]}
        except Exception as e:
            print(f"[ERROR in C094]: {e}")
            traceback.print_exc()

        # -------------------------------------------------------------
        # 4. Project C095: Residual Arc-Length Deliberation (Scaling thresholds)
        # -------------------------------------------------------------
        try:
            arclength_thresh = 1.25 + (cycle * 0.05)
            print(f"\n[Running C095] Deliberation Threshold: {arclength_thresh:.2f}")
            loop_c095 = SelfImprovingLoop(
                experiment_name=f"c095_cycle_{cycle}",
                initial_config={"n_samples": 120, "n_layers": 8, "threshold": arclength_thresh},
                train_and_eval_fn=evaluate_arclength_calibration,
                sanity_check_fn=sanity_check_arclength,
                kill_control_fn=kill_control_arclength,
                mutate_config_fn=lambda current_config, best_config, generation: {**best_config, "threshold": best_config["threshold"] + 0.05},
                log_dir=os.path.join(log_dir, "c095"),
                primary_metric="acc",
                seeds=[42 + cycle, 137 + cycle],
                max_generations=2
            )
            res_c095 = loop_c095.run_cycle()
            cycle_results["c095"] = {"best_acc": res_c095["best_score"], "config": res_c095["best_config"]}
        except Exception as e:
            print(f"[ERROR in C095]: {e}")
            traceback.print_exc()

        # -------------------------------------------------------------
        # 5. Project C029: SSA Residual Channels (Scaling composition parameters)
        # -------------------------------------------------------------
        try:
            k_val = 2 if cycle % 2 == 1 else 3
            print(f"\n[Running C029] S5 Permutation Composition k={k_val}, domain=40")
            loop_c029 = SelfImprovingLoop(
                experiment_name=f"c029_cycle_{cycle}",
                initial_config={"k": k_val, "domain": 40, "layers": 4, "d_model": 64, "heads": 4, "lr": 1e-3, "epochs": 8, "ssa_mode": True},
                train_and_eval_fn=run_ssa_trial,
                sanity_check_fn=sanity_check_ssa,
                kill_control_fn=kill_control_ssa,
                mutate_config_fn=lambda current_config, best_config, generation: {**best_config, "lr": 2e-3 if generation % 2 == 0 else 1e-3},
                log_dir=os.path.join(log_dir, "c029"),
                primary_metric="acc",
                seeds=[42 + cycle],
                max_generations=2
            )
            res_c029 = loop_c029.run_cycle()
            cycle_results["c029"] = {"best_acc": res_c029["best_score"], "config": res_c029["best_config"]}
        except Exception as e:
            print(f"[ERROR in C029]: {e}")
            traceback.print_exc()

        # -------------------------------------------------------------
        # 6. Project C124: Enzymatic Substrate Attention (Scaling sinks & Km)
        # -------------------------------------------------------------
        try:
            sinks_count = 6 + (cycle * 2)
            print(f"\n[Running C124] Attention Sinks: {sinks_count}, Km: 1.0")
            loop_c124 = SelfImprovingLoop(
                experiment_name=f"c124_cycle_{cycle}",
                initial_config={"n_samples": 250, "n_distractors": sinks_count, "n_classes": 5, "km": 1.0, "vmax": 1.0},
                train_and_eval_fn=evaluate_enzymatic,
                sanity_check_fn=sanity_check_enzymatic,
                kill_control_fn=kill_control_enzymatic,
                mutate_config_fn=lambda current_config, best_config, generation, **kwargs: {**best_config, "km": best_config["km"] + 0.2 * (generation % 2 == 0)},
                log_dir=os.path.join(log_dir, "c124"),
                primary_metric="acc",
                seeds=[42 + cycle, 137 + cycle],
                max_generations=2
            )
            res_c124 = loop_c124.run_cycle()
            cycle_results["c124"] = {"best_acc": res_c124["best_score"], "config": res_c124["best_config"]}
        except Exception as e:
            print(f"[ERROR in C124]: {e}")
            traceback.print_exc()

        # -------------------------------------------------------------
        # 7. Project C132: Soliton Residual Stream (Scaling depth & beta)
        # -------------------------------------------------------------
        try:
            layers_count = 12 + (cycle * 2)
            beta_val = 0.08 + (cycle * 0.01)
            print(f"\n[Running C132] Layers: {layers_count} (unnormalized), beta: {beta_val:.2f}, nu: 0.03")
            loop_c132 = SelfImprovingLoop(
                experiment_name=f"c132_cycle_{cycle}",
                initial_config={"n_samples": 250, "n_layers": layers_count, "d_model": 32, "n_classes": 5, "beta": beta_val, "nu": 0.03, "epochs": 20, "lr": 0.01},
                train_and_eval_fn=evaluate_soliton,
                sanity_check_fn=sanity_check_soliton,
                kill_control_fn=kill_control_soliton,
                mutate_config_fn=lambda current_config, best_config, generation, **kwargs: {**best_config, "beta": best_config["beta"] + 0.02 * (generation % 2 == 0)},
                log_dir=os.path.join(log_dir, "c132"),
                primary_metric="acc",
                seeds=[42 + cycle, 137 + cycle],
                max_generations=2
            )
            res_c132 = loop_c132.run_cycle()
            cycle_results["c132"] = {"best_acc": res_c132["best_score"], "config": res_c132["best_config"]}
        except Exception as e:
            print(f"[ERROR in C132]: {e}")
            traceback.print_exc()

        # -------------------------------------------------------------
        # 8. Project C134: p-adic Ultrametric Tree Cache (Scaling penalty & samples)
        # -------------------------------------------------------------
        try:
            pen_val = 8.0 + (cycle * 2.0)
            print(f"\n[Running C134] Tree KV-Cache: penalty: {pen_val:.1f}, p: 2")
            loop_c134 = SelfImprovingLoop(
                experiment_name=f"c134_cycle_{cycle}",
                initial_config={"n_samples": 300, "d_model": 32, "n_classes": 5, "penalty": pen_val, "p": 2, "epochs": 20, "lr": 0.01},
                train_and_eval_fn=evaluate_padic,
                sanity_check_fn=sanity_check_padic,
                kill_control_fn=kill_control_padic,
                mutate_config_fn=lambda current_config, best_config, generation, **kwargs: {**best_config, "penalty": best_config["penalty"] + 2.0 * (generation % 2 == 0)},
                log_dir=os.path.join(log_dir, "c134"),
                primary_metric="acc",
                seeds=[42 + cycle, 137 + cycle],
                max_generations=2
            )
            res_c134 = loop_c134.run_cycle()
            cycle_results["c134"] = {"best_acc": res_c134["best_score"], "config": res_c134["best_config"]}
        except Exception as e:
            print(f"[ERROR in C134]: {e}")
            traceback.print_exc()

        # -------------------------------------------------------------
        # 9. Project C126: Vector-Clock Asynchronous Causal Attention
        # -------------------------------------------------------------
        try:
            epochs_count = 60 + (cycle * 10)
            print(f"\n[Running C126] Multi-Stream Vector-Clock Attention (epochs: {epochs_count})")
            loop_c126 = SelfImprovingLoop(
                experiment_name=f"c126_cycle_{cycle}",
                initial_config={"d_model": 32, "n_heads": 2, "num_layers": 2, "num_classes": 5, "epochs": epochs_count, "batch_size": 64, "lr": 0.01},
                train_and_eval_fn=evaluate_vc,
                sanity_check_fn=sanity_check_vc,
                kill_control_fn=kill_control_vc,
                mutate_config_fn=lambda current_config, best_config, generation, **kwargs: {**best_config, "epochs": best_config["epochs"] + 10 * (generation % 2 == 0)},
                log_dir=os.path.join(log_dir, "c126"),
                primary_metric="acc",
                seeds=[42 + cycle, 137 + cycle],
                max_generations=2
            )
            res_c126 = loop_c126.run_cycle()
            cycle_results["c126"] = {"best_acc": res_c126["best_score"], "config": res_c126["best_config"]}
        except Exception as e:
            print(f"[ERROR in C126]: {e}")
            traceback.print_exc()

        cycle_elapsed = time.time() - cycle_start

        master_history.append({
            "cycle": cycle,
            "duration_sec": cycle_elapsed,
            "results": cycle_results
        })

        # Save cumulative meta-loop log
        with open(summary_file, "w") as f:
            json.dump({"cycles_completed": cycle, "history": master_history}, f, indent=2)

        print(f"\n>>> CYCLE {cycle} COMPLETED in {cycle_elapsed:.2f}s. Results logged to {summary_file} <<<")

    print("\n==================================================================")
    print(f"[DONE] CONTINUOUS META-LOOP FINISHED ALL {max_cycles} CYCLES.")
    print("==================================================================")


if __name__ == "__main__":
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_continuous_meta_loop(max_cycles=cycles)
