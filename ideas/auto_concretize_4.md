# Auto-concretize batch 4: 10 stubs → precise claims + §1e screen, 2026-09-23
# §1e only; web + Filter-2 legs pending.

### I1 (econ + kill control) → "VCG-priced model routing: route queries by truthful cost-bids"
§1e: no direct hit. BORDERLINE, pre-flagged likely-WEAK — inherits B-EC1's self-play flaw (operator invents bids, then runs truthful auction on itself). Advances only with external-bidder design.

### I2 (os + kill control) → "Priority-inheritance decoding: low-priority request inherits priority while blocking a high-priority prefix share"
§1e: no hit (scheduling mechanism, distinct from budget-conditioned generation). PASS-leaning to web leg (neighbors: vLLM scheduler, RadixAttention prefix sharing, priority-inversion literature).

### I3 (signal + bound cost) → "Matched prefetch predictor for next-layer KV"
§1e-adjacent: B-O1 activation offload+prefetch is PUBLISHED (FlexGen/DynamicInfer). PARTIAL at best — survives only as correlation-based predictor variant with prefetch-accuracy proof. Flagged likely-narrow.

### I4 (bio-phys + kill control) → "Apoptotic inference pruning: kill lowest-activation neurons per forward pass with regeneration"
§1e: no direct hit (pruning literature is training-time/structured). BORDERLINE — Filter-2 must beat magnitude-pruning + activation-sparsity baselines and justify per-pass regeneration cost.

### I5 (invert + kill control) → "Fixed explore-then-exploit temperature schedule (high early, greedy late)"
§1e-adjacent: B-P1 anneal+reheat (PARTIAL). PARTIAL-leaning — must cite anneal work; delta is fixed-schedule (no reheat trigger) vs adaptive. Narrow.

### I6 (compilers + bit-exact) → "Constant-folded refusal DFA over prompt prefixes"
§1e: no hit. PARTIAL-leaning (neighbors: constrained decoding, safety classifiers for web leg; distinction is precomputed refusal automaton vs runtime judge).

### I7 (ecc + kill control) → "Syndrome-checked CoT: parity over reasoning-step hashes verified by external script"
§1e-adjacent: B-E1 parity step (PARTIAL) + E15 CRC-args. PARTIAL-leaning — must cite both; delta is hash-level syndrome vs step-level parity. Narrow; self-attestation flaw (model emits its own hashes) pre-registered for Filter-2.

### I8 (crypto + bound cost) → "Macaroon-scoped tool tokens per request"
§1e: no hit. PARTIAL-leaning, narrow (neighbors: SatGate macaroons per E30 audit; distinction is per-request attenuation vs budget caps).

### I9 (anomaly + bound cost) → "Truncate-vs-abort quality study for oversized requests"
§1e: no hit. BORDERLINE — methods note, thin; advances only with pre-registered degradation curves on real tasks, else kill as eval trivia.

### I10 (db + bit-exact) → "Exact-duplicate batch dedup pre-training with membership ledger"
§1e-adjacent: decontamination/dedup area (D10-adjacent, heavily published: Lee et al. deduplicating training data and successors). Flagged likely-PUBLISHED — web leg must find distinguishing scope (online batch-level vs corpus-level) or discard.

## Tally: 10 concretized (1 PASS-leaning I2, 5 PARTIAL-leaning I3/I5/I6/I7/I8, 4 BORDERLINE incl. 2 likely-dead I1/I10).
## Cumulative concretized: 35/65 stubs. Remaining queue: 30.
