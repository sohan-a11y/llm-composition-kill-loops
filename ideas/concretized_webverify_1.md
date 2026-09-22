# Concretized web-verification 1/7: F1–F5 + G1–G5, 2026-09-23
# ≥3 phrasings × arXiv/GitHub/HF/PyPI/HN/Reddit per claim (Exa 429s → direct APIs; Reddit 403, PyPI search challenged — documented per agent).

## F1 per-block KV CRC + recompute — PARTIALLY EXPLORED
Neighbor: Yamamoto 2604.17249 (SECrypt26, scheduling-time checksum for shared prefix blocks, bit-flip threat). GhostServe/DéjàVu own replication/erasure, not CRC+recompute.
Gap: per-block CRC checked during long decode + recompute-only-corrupt-block + overhead analysis.

## F2 static dead-prompt ablation — PARTIALLY EXPLORED
Neighbors: SGP-CoT repo (ACL26, 0★) + Thought-Anchors 2506.19143 (143★/42 forks, LLM-resampling by construction) + SPOT/BRIDGE + PartPrompt/CodePromptZip/AGORA (inference-free scorer, closest).
Gap: purely static zero-LLM-call removal with cost proof.

## F3 verifier-allocated compute — PARTIALLY EXPLORED
Neighbors: TALE 2412.18547, BudgetThinker, BG-MCTS 2602.09574 (all self-allocated) + 2504.01005 solve-vs-verify split (verifier doesn't set per-step budget).
Gap: external verifier dictates per-step budget, generator zero self-allocation + iso-budget proof.

## F4 noise-asymmetric RAG — PARTIALLY EXPLORED
Neighbor: RAFT 2403.10131 (distractor-injection finetune, gorilla repo) + Passage-Injection + BAR-RAG (Self-RAG is interleave, distinct).
Gap: explicit train-noisier-than-infer asymmetry + shuffled-order operator + clean-inference eval. Must cite RAFT.

## F5 verify-after-generate RAG — PARTIALLY EXPLORED
Neighbors: GENREAD 2209.10063 (293★, generates evidence not draft-to-falsify) + LLatrieval/RealRoute (retrieve→verify direction) + GRG.
Gap: answer-draft-first with retrieval-only-to-falsify as DEFAULT order. Must cite GENREAD + LLatrieval.

## G1 committed-output training — NOT FOUND (within reached sources)
Only hit unrelated: 2608.07762 eval-judge commit-reveal. Hash-conditioned tool-output training + reveal-at-inference: 0 everywhere reached. PyPI/Reddit/Exa gaps noted — no novelty claim without reproducible full-text+code search.

## G2 span-WAL — PARTIALLY EXPLORED / WEAK (pre-registration confirmed)
Atomix 2602.14849 (tool-action WAL, ARIES/Sagas) owns the family; HN "Log is the Agent" discusses same. Gap: span granularity + truncate-cost proof only.

## G3 role-swap eval protocol — PARTIALLY EXPLORED
Swap-as-mechanism published (Feng-Steinhardt 2310.17191, Oh-Demberg 2606.08644, Dai 2409.05448, NER-replacement EMNLP23). Gap: formal train-clean→test-swapped-roles metric as held-out eval (vs patching/augmentation uses).

## G4 prefix-train/causal-infer — PUBLISHED (pre-registration confirmed)
PrefixLM literature + hf_prefixlm_converter (nomic-ai/gpt4all-mpt-2 + forks) + Tree Training 2511.00413 / 2606.01143 / BICACHE KV-correctness. Both halves exist; needs new correctness proof or cost bound to revive.

## G5 learned inline threshold — NOT FOUND (threshold-hash prior refuted)
ToolGate contracts, ATR routing, fixed 0.8/top-k thresholds adjacent; no threshold-hash paper surfaced anywhere. Exact learned-threshold + optimality-proof combo: 0 hits.

## Tally: NOT FOUND 2 (G1, G5) / PARTIAL 7 / PUBLISHED 1 (G4).
