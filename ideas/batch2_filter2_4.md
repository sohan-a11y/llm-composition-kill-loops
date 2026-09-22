# Batch2 Filter 2 (4/4): E24/E26/E28/E29/E31/E32/E33/E35/E39, 2026-09-23
# Batch2 Filter 2 now COMPLETE (10 NOT FOUND + 21 PARTIAL all screened).

## NEEDS-KILL-TEST (2, both capped)
### E26 emoji-variant guard (micro-tool cap, not paper)
Attacks published; guard mechanism (NFKC + VS/strip + demojize) untested. Wrong-layer risk: wins are semantic, not variant-driven.
Kill (1d, 100 attack + 100 benign emoji prompts): ASR/FPR with vs without normalization. Should-fails: pure-text jailbreaks must show ~0 delta; variant-free semantic payload must NOT be stopped.
GO iff ASR -≥30pp absolute, FPR +≤5pp, variant-free payload still blocked. Else KILL. Cap: gist, never paper.

### E31 prompt-diff reviewer (wrapper, correlation-or-kill)
diff→impact without full re-eval; power inherited from judge + rubric.
Kill (20 prompt pairs: 10 paraphrase / 10 breaking; scorer vs full promptfoo ground truth): whitespace/paraphrase must score low; delete-schema/invert-instruction must score high (else no discrimination → kill).
GO iff Spearman >0.7 vs true delta AND AUC >0.85 AND cost <10% AND beats edit-distance by >0.2. Else KILL.

## WEAK (7)
- E24 fence closer: deterministic regex+stack post-processing; LatentMD/Format-Tax dominate. GO iff natural mismatch >2% AND regex <80% while guard >95% (100% on synthetic corruptions for both).
- E28 currency gate: normalization destroys locale signal (anti-feature risk). GO iff symbol-swap alone swings ≥10% AND normalization recovers ≥80% without locale-format regressions.
- E29 span re-decode: LanguageTool + local resample; SOTA mismatch rate ~0. GO iff natural rate >1% AND fixes >80% with <2% fluency regressions.
- E32 capability matrix: leaderboard clone + publish checklist; governance, not research. GO iff ≥2 replicable off-diagonal tradeoffs p<0.05 + gated base-pin compliance gap.
- E33 determinism badge: trust-signal; batch-invariance ≠ determinism (CUDA/MoE/FP16/TP flapping); no adopter. GO iff hardware-stable pass/fail AND 3+ external maintainers commit in writing.
- E35 cost gate: 3σ on heavy-tailed multimodal traffic = alert fatigue + latency-breaking holds; one-sprint vendor feature. GO iff recall ≥90% blowups AND false-hold ≤2% AND <5ms overhead.
- E39 seed bank: 20+ sets exist; manifest format ≠ curation value; rots without maintenance. GO iff <40% incumbent overlap AND ≥5/10 testers prefer AND verifier catches planted dupes.

## Batch2 Filter-2 totals
NEEDS-KILL-TEST: E4, E6-lean, E13, E20, E22, E26-cap, E31 (7) / WEAK: all others / PUBLISHED-discards: 9.
Remaining: auto-65 screening + T4 kill-runs (GPU-gated).
