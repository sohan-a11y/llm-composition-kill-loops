# Auto-65 audit + concretization sample, 2026-09-23
# Honest finding first: the auto-65 are NOT screenable ideas. They are 6 recycled
# template suffixes ("swap train/infer placement", "add explicit kill control", ...)
# crossed with strategy tags — no precise claim, no mechanism, no falsifiable content.
# Worse, the G1000/G1001 blocks appear TWICE verbatim: the generator re-seeded at
# 1000 every invocation (seed=1000+it bug, fixed this iteration: base now derives
# from pool size + dedupe on IDs). Counting these as "65 raw candidates" would be
# count-padding, which the brief forbids. They are marked SEEDS, not candidates.
# Effective screened pool remains 150 (110 + 40). The 65 stubs await concretization.

## Concretization spec (what a stub needs to become screenable)
1. Precise one-sentence claim with mechanism + domain + comparison target.
2. Named §1e exclusion check with verdict.
3. Cheapest kill-test sketch with should-fail control.
Stubs failing (1) are not admitted to Filter 1.

## Sample: 5 stubs concretized + §1e-screened (pipeline proof)
### F1 (ecc + bit-exact control) → "Per-block CRC over KV cache with recompute-on-mismatch in long generation"
§1e: KV-cache family adjacent, but error-DETECTION ≠ eviction/compression. BORDERLINE, advances to web leg with KV-integrity neighbors (LLMFI checksums etc.).

### F2 (compilers + kill control) → "Dead-prompt elimination: statically remove prompt spans whose ablation never changes output, with must-drop control"
§1e-adjacent: B-C4 dead-CoT/step-pruning is PUBLISHED (SGP-CoT, Thought-Anchors). PARTIAL at best — survives only as static-analysis (no-LLM-call) variant with cost proof. Flagged likely-WEAK for Filter 2.

### F3 (control-plane inversion) → "Verifier-allocated compute: external verifier sets per-step budget, generator executes with zero self-allocation"
§1e-adjacent: budget-conditioned generation (excluded) + E13 class-scheduling. BORDERLINE — advances only with proof external allocation beats self-allocation at iso-budget.

### F4 (anomaly + train/infer swap) → "Noise-asymmetric RAG training: inject retrieval noise (shuffled/distractor docs) in training, infer clean"
§1e: no hit (robustness training ≠ listed bullets). PASS-leaning to web leg (neighbors: Self-RAG/FLARE interleave, noisy-embedding work must be checked).

### F5 (db + train/infer swap) → "Verify-after-generate RAG: draft answer first, retrieve only to verify/falsify draft claims as default order"
§1e: no hit. PARTIAL — GENREAD-style generate-then-read exists as named neighbor for web leg.

## Tally: 65 stubs triaged (0 admitted unscreened) / 5 concretized (1 PASS-leaning F4, 2 BORDERLINE F1/F3, 2 PARTIAL-leaning F2-flagged/F5) / generator seed bug fixed.
## Remaining: concretize-then-screen queue for the other 60 stubs (template per spec above) + T4 kill-runs.
