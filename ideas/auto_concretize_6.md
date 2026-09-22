# Auto-concretize batch 6: 10 stubs → precise claims + §1e screen, 2026-09-23
# §1e only; web + Filter-2 legs pending.

### K1 (econ + bit-exact) → "Exact-cost routing tables: precomputed $/token lookup per model, no estimator"
§1e: no hit. BORDERLINE (lookup table, thin; must cite routing/cost literature for web leg). Low traction pre-registered.

### K2 (os + bit-exact) → "Schedule-logged deterministic replay (record scheduling decisions, not just seeds)"
§1e-adjacent: D1 replay-logging (PARTIAL). PARTIAL-leaning — must cite D1; delta is schedule-log vs seed-log. Narrow.

### K3 (signal + bit-exact) → "TPDF-dithered sampling: add dither noise to logits pre-top-p"
§1e-adjacent: E11 dithers weights (different domain); entropy-temp family owns sampling behavior. BORDERLINE — advances with note; Filter-2 must beat temp sweep (probably collapses to temperature).

### K4 (bio + bit-exact) → "Overnight LoRA replay of day's failures during idle (sleep consolidation for adapters)"
§1e-adjacent: continual learning / writing-into-weights (excluded family: Lin/Goyal/Gupta line). Likely FAIL — finetune-on-failures family. Marked BORDERLINE pending proof it differs from standard continual-finetune baselines.

### K5 (invert + swap) → "Tied-swap probe: decode from input embeddings, encode with output embeddings"
§1e: no hit. BORDERLINE (probe, likely null result; advances as falsification-grade micro-probe with pre-registered expectation of null).

### K6 (compilers + swap) → "Decompile-then-edit: decompile model output to IR, edit IR, re-render"
§1e: no hit. PASS-leaning to web leg (neighbors: code editing/transpilation literature; distinction is IR-mediated edit vs direct regeneration).

### K7 (ecc + swap) → "Inverted parity framing: CoT as parity, answer as data"
§1e-adjacent: B-E1 parity step (PARTIAL, WEAK-leaning). Pre-flagged likely-WEAK — framing swap of an already-weak idea. BORDERLINE on technicality.

### K8 (crypto + swap) → "Small-prover/large-verifier asymmetry (direction swap of A12)"
§1e-adjacent: A12 verifier-heavy/generator-light is PUBLISHED (spec-decoding/Best-of-N/GenRM). Likely PUBLISHED — direction swap lives in the same literature. Marked BORDERLINE pending distinguishing detail or discard.

### K9 (anomaly + kill control) → "Pre-deploy glitch-token probe with quarantine list"
§1e-adjacent: glitch-token detection/repair (excluded: Fishing for Magikarp). Likely FAIL — family-owned. Marked BORDERLINE pending proof of delta vs Magikarp-style probing.

### K10 (db + swap) → "Reverse retrieval: index the queries, stream documents as the query side"
§1e: no hit. BORDERLINE (inversion relabel risk; advances with note; Filter-2 must beat standard bi-encoder at iso-compute or die as notation swap).

## Tally: 10 concretized (1 PASS-leaning K6, 1 PARTIAL-leaning K2, 8 BORDERLINE incl. 4 likely-dead K4/K7/K8/K9).
## Cumulative concretized: 55/65 stubs. Remaining queue: 10.
