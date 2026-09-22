# Auto-concretize batch 5: 10 stubs → precise claims + §1e screen, 2026-09-23
# §1e only; web + Filter-2 legs pending.

### J1 (tooling + swap) → "Test-first data selection: evaluate on held-out before training, select only mispredicted slices"
§1e-adjacent: train-before-test evaluation (lm-eval-harmony family, excluded-eval-adjacent). BORDERLINE — advances only as data-selection (vs evaluation-integrity) with contamination controls pre-registered.

### J2 (econ + swap) → "Split-budget RAG accounting: charge retrieval vs generation to separate budgets with swap audit"
§1e: no hit. BORDERLINE (accounting infra, thin; must cite E30/SatGate for web leg). Pre-registered low traction.

### J3 (os + swap) → "Mid-decode device migration at token boundaries (GPU↔CPU)"
§1e-adjacent: offload (FlexGen published) owns placement; migration is its dynamic form. PARTIAL-leaning, narrow — survives only with migration-vs-offload latency proof.

### J4 (signal + swap) → "Spectral-domain drift detection with token-domain action"
§1e-adjacent: SpecRA/SpecDetect own spectral detection (excluded/published). BORDERLINE — detection side relabel unless action loop (detect→intervene) is the contribution; pre-register as likely-WEAK.

### J5 (bio + swap) → "Offline consolidation: rewrite memory index between sessions (sleep phase)"
§1e: no hit (continual-learning exclusion is weights-side; this is retrieval-index-side). PASS-leaning to web leg (neighbors: MemGPT-style memory, RAG index refresh work).

### J6 (invert + bound cost) → "Single-token reasoning bottleneck per N steps"
Artifact-adjacent: screen_experiments bottleneck5_nosup scored 0.19775 (chance) in our own artifacts. Pre-flagged likely-WEAK — advances only to confirm the artifact's negative at a second scale. BORDERLINE.

### J7 (compilers + kill control) → "Range analysis on numeric tool args before exec (bounds-check)"
§1e: no hit. BORDERLINE (engineering guard; must cite E15/schema-validation area; traction lint-tier).

### J8 (ecc + bit-exact) → "Triple-emit answers with mismatch-abstain (repetition code)"
§1e-adjacent: self-consistency / test-time voting family (excluded-adjacent). Likely FAIL — N=3 vote relabel. Marked BORDERLINE pending proof abstain-policy beats majority (doubtful; Filter-2 skepticism pre-registered).

### J9 (crypto + kill control) → "Nonce challenge-response tool authentication"
§1e: no hit. PARTIAL-leaning (neighbors: AgentJWT per B-CY1 audit for web leg; distinction is per-call nonce vs intent-bound token).

### J10 (db + kill control) → "LSM-style vector compaction for memory stores"
§1e-adjacent: B-D4 LSM prefix cache (NOT FOUND, WEAK on compaction latency). PARTIAL-leaning — must cite B-D4 audit; delta is vector-merge vs prefix-blocks. Narrow; latency objection carries over.

## Tally: 10 concretized (1 PASS-leaning J5, 3 PARTIAL-leaning J3/J9/J10, 6 BORDERLINE J1/J2/J4/J6/J7/J8 incl. 2 likely-dead J6-artifact-negative/J8-vote-relabel).
## Cumulative concretized: 45/65 stubs. Remaining queue: 20.
