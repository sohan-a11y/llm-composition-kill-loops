# Flagship cross-audit: parallel-session projects vs §1e + measurement standards, 2026-09-23
# Read all mechanism + eval files before judging. Line refs exact. No deletions made.

## c029 SSA residual — HONEST NEGATIVE, keep as falsification
- Mechanism (model_ssa.py): write-once residual slices per layer + scramble-slice kill arm. Real training on S5 (train_ssa.py), proper k=1 sanity + scramble control.
- Results (meta_loop_summary.json): best_acc 0.2975 / 0.2075 at k=2-3, domain 40 — AT CHANCE, consistent with composition cliff §1a.
- §1e: slot-partitioned residual ≈ raw_multi family (§1b: partial 0.518/0.118, same-transform fails). Not a VSA-novelty violation per se (registers, not binding claims), but must cite §1b.
- Gaps: loop fixes ssa_mode=True — the ssa_mode=False arm exists in code but is never varied, so nothing can be attributed to SSA. As "flagship novel" it is a negative; value is falsification. Distinct from E4 (prompt renaming) despite shared name.
- Verdict: KEEP, relabel as negative result; add ssa-vs-standard comparison before any positive claim.

## c063 virtual syntax — VOID experiment, mechanism excluded
- Engine (virtual_syntax_engine.py) is KV bookkeeping over random tensors, not a model.
- Eval (eval_grammar_tax.py:91-93): virtual_skip arm sets pred_ans = target_ans UNCONDITIONALLY; standard_gcd arm injects random corruption (84-90). The 1.0 is ASSIGNED, not measured — hardcoded-PASS pathology per BRIEF §1 (MicroAGI lesson).
- §1e: KV-cache pruning/eviction/compression = excluded; constrained JSON decoding = PUBLISHED (Outlines/Guidance/xgrammar, cf. C19).
- Verdict: VOID as evidence. Rebuild with real GCD model + measured validity/latency or drop the 1.0 claim.

## c085 curvature alarm — simulation demo, not measurement; collapses to variance family
- Generator (eval_curvature.py:39-54) bakes signal: grounded = smooth drift, hallucinated = 7.5-magnitude sine impulse at layers 4-6. Detector measures curvature. Label IS the statistic → 1.0 proves detector reads generator, nothing about real hallucinations.
- No real model/labels; no comparison to linear probe or entropy baseline (Filter-2 bar: +0.03 AUROC over SAPLMA/linear). Threshold re-tuned per cycle (0.17→0.21) on test.
- Kill control (linear → Δ²=0 exactly) is mathematically trivially true, not discriminating.
- Family: norm-statistic monitor — PARALLAX lesson (collapse to confidence) applies until beaten head-to-head.
- Verdict: DEMO only. Needs real-label eval (HaluEval/FEVER) vs SAPLMA-linear + entropy before any alarm claim.

## c094 correlation probe — forecasting claim KILLED by its own (fixed) control
- Generator (eval_correlation.py:49-58) bakes signal: failing seqs get uniform-scrambled profiles at t≥seq_len-2. 0.985-0.99 detects injected scramble.
- INTEGRITY ISSUE FOUND + FIXED: kill_control_correlation fabricated shuffled_acc = 0.50 + N(0,0.04) analytically (old lines 141-145) without running the probe — hardcoded-PASS control in the exact MicroAGI sense. Replaced 2026-09-23 with real order-destruction control (scramble_temporal_order=True).
- Fixed control RESULT (measured 2026-09-23): main acc 0.9917, scrambled-order acc 0.99 → control returns (False, 0.99, 0.5). Destroying temporal order changed nothing ⇒ accuracy does not come from temporal forecasting ⇒ the "1-2 steps prior" forecasting claim is DEAD.
- What survives: single-step anomaly-presence detection, uncompared to trivial per-step baseline. That is a weaker, untested claim.
- Verdict: forecasting claim KILLED by own control (control proved it can fail — which is what makes it honest). Presence-detection needs single-step baseline comparison.

## c095 arclength calibration — simulation demo; "un-cheatable" unevidenced
- Generator (eval_calibration.py:47-55): deliberating = large random tangents (high tortuosity by construction), sycophantic = straight interpolation. 1.0 detects injected geometry. No real sycophancy data, no confidence-baseline comparison.
- "Un-cheatable / resists prompt-induced overconfidence": no adversary tested; optimizing against tortuosity unexamined.
- Verdict: DEMO only. Needs real sycophancy eval + ECE comparison vs confidence baselines.

## Structural pattern (all five)
All 1.0s except c029 come from simulators where the label is defined by the measured statistic — generator guarantees separation, so main metrics cannot fail. Per BRIEF §1, a result whose controls cannot fail is void as evidence. c029 is the exception that proves the rule: real task, real training, honest chance-level negative.
Migrated claims must each beat the named baseline head-to-head (linear probe/entropy/single-step/GCD-validity) on real data before promotion. No content deleted; c094 control fixed in place.
