# Batch2 web-verification 4/4: E31–E40 (E36 already §1e-FAIL), 2026-09-23
# ≥3 phrasings × arXiv/GitHub/HF/PyPI/HN/r/LocalLLaMA+r/MachineLearning per idea.
# NOT FOUND = not found in returned results, does not prove absence.

## E31 prompt-diff reviewer — PARTIALLY EXPLORED (wrapper)
381 prompt-regression repos adjacent (agentprdiff 14★, prompt-replay, prompt-ci); arXiv prompt-regression+diff = 0; PyPI prompt-diff 404 (promptfoo adjacent, evals not diff).
Gap: diff(old,new)→affected-capabilities→cheap impact score without full re-eval + estimator calibration. Orchestration, no new mechanism.

## E32 adapter-capability matrix — PARTIALLY EXPLORED (wrapper)
Leaderboards published (Open LLM LB analyses, lora-speedrun 148★ frozen-task). LoRA-Factory 0★ has registry+lineage. arXiv registry-gate = 0; HF []; PyPI 404; HN 0.
Gap: gated-publish policy + adapter×capability matrix schema + base-pin/seed reproducibility. Policy/orchestration over harness infra.

## E33 determinism badge — PARTIALLY EXPLORED (mechanism published, packaging gap; wrapper)
Mechanism PUBLISHED: CoRun 2608.14376, LLM-42 2601.17768 (microsoft/llm-42 29★), MarginGate 2605.30218, TBIK, repro 2511.17826; 156 GH repos (detllm 52★, mlx-deterministic, LockStep).
Gap: CI badge packaging (1-vs-N invariance test → model-card badge) + standard protocol (prompts, shapes, TP sizes, tolerance). Trust signal, zero modeling novelty; must cite mechanism priors.

## E34 RAG freshness monitor — PUBLISHED, discard
aws-samples/sample-bedrock-knowledge-base-drift-detection (2026-06-27, verbatim) + driftguard-llm + embedding-drift monitors (0★, demo-grade) + evidently 7933★ generic. arXiv exact-combo 0 but code is exact. Novelty at most metric/threshold.

## E35 cost-anomaly detector — PARTIALLY EXPLORED
Monitoring published (Bedrock token alarms 8★, ethoscompute, Langfuse 34944★ tracing). arXiv exact 0; HN 0.
Gap: explicit per-request >3σ tokens/cost → hold-for-review policy (window, per-model/per-user normalization, workflow). No artifact documents the rule.

## E37 cgroup-per-request truncation — NOT FOUND (cleanest of batch)
GH exact 0×2; arXiv only host-level (Predictable Serving 2508.20274 MPS/cgroup-I/O guards); nearest linux-kernel-inference-fastpath 3★ (hints, no per-request migration/truncation); HN 0.
Gap: transfer-per-request + truncate-instead-of-OOM-kill. Clean within fetched sources (Reddit/PyPI-search gaps noted).

## E38 homoglyph gate — PUBLISHED, discard
2508.14070 §6 (NFKC + homoglyph map + mixed-script flag + zero-width strip) + ACL-Findings25 Lies-Characters-Tell (pcoopercoder repo) + unicode-shield + prompt-canon 0.1.0 PyPI (verbatim tagline). Glitch-token distinction CONFIRMED distinct (2404.09894/2408.04905/GlitchMiner = vocab anomalies, no confusables). Gate itself published.

## E39 red-team seed bank — PARTIALLY EXPLORED
Datasets abundant (RedBench 2601.03699 37dsets/29k; hh-rlhf 1.92k likes; in-the-wild 15k; AttaQ; MHJ; PromptBank). Seeds-as-starting-prompts (AutoRed 2510.08329) + PyRIT datasets/seeds + redteam-foundry versioned packs (harness feature) adjacent.
Gap: citable DATASET artifact with seed set + per-prompt provenance (source/commit/lineage/target/ASR) + SemVer + content hash + de-dup manifest. Mechanics (HF SHAs/DVC) exist; schema packaging missing.

## E40 token futures — PUBLISHED, discard
Xing 2603.21690 (SIT design) + Touchmark live forward market (Aug 2026, exact product) + IFX/SHFE/CME compute futures + OpenAI Batch/Reserved bilateral primitives. Core pre-empted; batch-window forward at most product variant.

## Tally (this file): NOT FOUND 1 (E37) / PARTIAL 5 (E31,E32,E33,E35,E39) / PUBLISHED 3 (E34,E38,E40)
## Cumulative batch2 web (E1–E40 excl. E9,E36 §1e-fails): NOT FOUND 10 (E1,E4,E10,E15-weak,E20,E23-narrow,E25-lint,E27-lint,E30,E37) / PARTIAL 21 / PUBLISHED 7 (E5,E8,E14,E17,E34,E38,E40).
