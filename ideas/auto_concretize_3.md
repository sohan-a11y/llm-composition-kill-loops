# Auto-concretize batch 3: 10 stubs → precise claims + §1e screen, 2026-09-23
# §1e only; web + Filter-2 legs pending. Kush verdicts pre-registered where adjacent.

### H1 (econ + bound cost) → "Off-peak batch slot auction with reserve pricing for queue scheduling"
§1e: no direct hit. PARTIAL-leaning — must cite E40 token-futures audit (PUBLISHED: Xing + Touchmark) and prove batch-window specificity. Flagged likely-narrow.

### H2 (os + bound cost) → "Hard KV budget per sequence with oldest-eviction (FIFO), explicitly unscored"
§1e-adjacent: KV-cache eviction (excluded). BORDERLINE — survives only if unscored FIFO beats scored eviction (implausible; Filter-2 skepticism pre-registered, likely WEAK).

### H3 (signal + kill control) → "Spectral-flatness gate: reject generations with narrowband logit-spectrum peaks"
§1e-adjacent: FFT/spectral repetition detection (SpecRA excluded) owns detection; B-S5 notch owns intervention. Likely FAIL — detection-side relabel. Marked BORDERLINE only pending proof it differs from SpecRA features.

### H4 (bio-phys + bound cost) → "Metabolic neuron quotas: per-neuron firing quota per sequence, silenced after exhaustion"
§1e: no direct hit (MoD is depth-routing, distinct level). BORDERLINE — advances with note; Filter-2 must beat activation-sparsity baselines and justify quota over top-k.

### H5 (invert + bit-exact) → "Deterministic nucleus: fixed top-p mass with seeded RNG logged per request for replay"
§1e-adjacent: D1 replay-logging (PARTIAL) + E33 determinism (wrapper). PARTIAL-leaning — survives as determinism-plus-replay primitive; must cite both.

### H6 (compilers + swap) → "Compile-then-prompt: lower task to bytecode-like IR, prompt with IR instead of NL"
§1e: no hit. PASS-leaning to web leg (neighbors: PAL/PoT program-aided prompting must be checked; distinction is IR-in-prompt vs code-exec).

### H7 (ecc + bound cost) → "Single-parity KV checkpoint per N steps enabling single-segment rollback"
§1e-adjacent: KV-cache rollback/backtracking (excluded). Likely FAIL — rollback family. Marked BORDERLINE pending proof parity-checkpoint differs from logged rollback techniques.

### H8 (crypto + bit-exact) → "HMAC-signed prompt templates verified at serving to detect injection"
§1e: no hit. PARTIAL-leaning (neighbors: authenticated-prompts work per B-CY1 audit; must cite and distinguish template-signing from plan-commit).

### H9 (anomaly + bit-exact) → "Byte-exact quote-or-abstain: spans verified by hash, abstain on mismatch"
§1e-adjacent: Copy-as-Decode (excluded) + C17 span-copier (PARTIAL). BORDERLINE — survives only as abstain-policy (vs copy mechanism) with hallucination-rate proof.

### H10 (tooling + kill control) → "Mutation-tested evals: inject pipeline faults, require detection before green CI"
§1e-adjacent: eval harnesses (excluded) own the infra. BORDERLINE — advances as testing-methodology with pre-registered skepticism (mutation coverage ≠ capability signal).

## Tally: 10 concretized (1 PASS-leaning H6, 3 PARTIAL-leaning H1/H5/H8, 6 BORDERLINE incl. 2 likely-FAIL H3/H7).
## Cumulative concretized: 25/65 stubs. Remaining queue: 40.
