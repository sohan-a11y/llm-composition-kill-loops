# Auto-concretize batch 2: 10 stubs → precise claims + §1e screen, 2026-09-23
# Each: one-sentence mechanism + domain + comparison target, then exclusion verdict.
# None web-verified yet — §1e only. Filter-2 + web legs pending.

### G1 (crypto + swap) → "Committed-output training: train on hashed tool outputs, infer on revealed outputs"
§1e: no hit (commit-reveal is B-CY1 NOT-FOUND territory, distinct: training regime vs planning protocol). PASS-leaning to web leg (must check hash-conditioned training literature).

### G2 (db + swap) → "Write-ahead inference: log planned tokens before emitting, truncate span on verifier reject"
§1e-adjacent: B-D5 WAL-agents are PUBLISHED (Atomix/Cordon). PARTIAL at best — survives only as token-span (not tool-action) WAL with cheaper-truncate proof. Flagged likely-WEAK for Filter 2.

### G3 (anomaly + swap) → "Role-swap eval: train clean, test with swapped entity roles to measure binding reliance"
§1e: no hit (robustness eval, not listed bullets). PASS-leaning to web leg (must check counterfactual/role-swap eval literature).

### G4 (invert + swap) → "Prefix-train/causal-infer with shared weights"
§1e-adjacent: A1 PrefixLM/UniLM is PUBLISHED (mask-switch same weights). Flagged likely-PUBLISHED — web leg must find distinguishing detail (e.g., KV-correctness construction) or discard.

### G5 (compilers + bound cost) → "Learned inline threshold: inline tool outputs under N tokens else reference-by-hash, N tuned to latency budget"
§1e-adjacent: B-C5 threshold-hash inlining is PUBLISHED (Haystack/ctxzip). PARTIAL at best — survives only as learned-threshold (vs fixed) with optimality proof. Flagged likely-WEAK.

### G6 (ecc + swap) → "Odd/even interleave training: train on interleaved token order, infer sequential"
§1e: no hit. BORDERLINE on substance (order-scrambling usually destroys LM priors — Filter-2 likely WEAK), but no exclusion bullet covers it. Advances on technicality with low expectations.

### G7 (control + bound cost) → "Fixed FLOP ceiling with hard early-stop, explicitly non-adaptive"
§1e-adjacent: budget-conditioned generation (excluded) is adaptive/conditioned; fixed-cap is the degenerate constant case. BORDERLINE — advances only to test whether non-adaptivity beats adaptive schedulers (unlikely; Filter-2 skepticism pre-registered).

### G8 (bio-phys + swap) → "Circadian training phases: alternate clean and corrupted batches on a fixed period"
§1e-adjacent: spaced-repetition ordering (excluded) is review-scheduling by difficulty; periodic corruption is a different axis. BORDERLINE — advances with note; Filter-2 must beat random-mix baseline (probably collapses to augmentation).

### G9 (tooling + bound cost) → "Cost-capped evals: fixed $ budget per run with significance early-stop"
§1e-adjacent: eval harnesses (excluded) own the infra; fixed-budget early-stop is a stopping rule. BORDERLINE — advances as methods-adjacent statistics (sequential testing) with pre-registered skepticism.

### G10 (os + invert plane) → "Checkpointed interruptible decoding: external signal preempts at token boundaries, resumes from KV checkpoint"
§1e: no direct hit. PARTIAL — must cite B-O4 audit (PARTIALLY, abort+restart neighbor) and prove checkpoint/resume beats abort+prefix-restart. Distinct claim: resume-where-left-off vs restart.

## Tally: 10 concretized (2 PASS-leaning G1/G3, 3 PARTIAL-leaning G2/G5/G10, 5 BORDERLINE G4-flagged/G6/G7/G8/G9).
## Cumulative concretized: 15/65 stubs. Remaining queue: 50.
