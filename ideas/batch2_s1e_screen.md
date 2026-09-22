# Batch2 §1e exclusion cross-check (Filter-1 leg 1 of 2), 2026-09-23
# Checked each E-idea against BRIEF §1e exclusion bullets (verbatim + near-variant).
# PASS = no §1e hit, advances to web-verification leg (3 phrasings × sources, NOT DONE).
# BORDERLINE = adjacent bullet, survives only with named distinction + kill control.
# FAIL = excluded, discarded. "Not found" never means proven absent.

- E1 unembedding-norm gating: PASS (logit-lens exclusion is depth-ensembling; per-token norm gain is distinct actuation)
- E2 causal-mask dropout in training: PASS
- E3 prompt-last training: PASS
- E4 SSA-form prompts: PASS
- E5 loop-unroll CoT template: PASS (prompting template, not a CoT-limits claim)
- E6 covering-index RAG: PASS
- E7 two-phase commit tools: PASS §1e, NOTE web-leg likely PUBLISHED (Atomix/Cordon family per batch-1 findings)
- E8 nice-level preemptible reasoning: PASS
- E9 OOM-killer KV by value score: FAIL — §1e KV-cache pruning/eviction/compression (score-eviction = H2O-class, no load-bearing delta)
- E10 matched-filter retrieval: PASS
- E11 TPDF dithering pre-quant: PASS §1e (dynamic per-token precision exclusion is distinct mechanism; classic DSP, web-leg must check PTQ literature)
- E12 length-prior gravity comp: PASS
- E13 gain-scheduled temp by prompt class: BORDERLINE — adjacent §1e entropy-adaptive temperature; survives only if class signal beats entropy signal head-to-head with shuffled-class control
- E14 batch-dimension parity row: PASS
- E15 CRC over tool args only: PASS (narrow exact domain; web-leg must check structured-generation CRC mentions)
- E16 threshold-signed releases: PASS (process control, near-zero research traction — flag)
- E17 blind-token PII inference: PASS
- E18 all-pay auction for beam slots: BORDERLINE — adjacent §1e attention-head auction/budget allocation; distinct level (beams vs heads) but auction framing shared; survives only with proof auction beats fixed-beam at equal FLOPs
- E19 escalation insurance: PASS (weak traction — flag)
- E20 refractory-period heads: PASS (note MoD/early-exit adjacency is scheduling vs skipping — distinct)
- E21 bigram fusion fast-path: BORDERLINE — adjacent §1e dynamic/adaptive tokenizers + speculative/MTP family; survives only vs tokenizer-retrain baseline
- E22 hysteresis decoding on entropy: BORDERLINE — adjacent §1e entropy-adaptive temperature (same signal, hysteresis actuation); survives only if hysteresis beats plain entropy-temp with shuffled-history control
- E23 citation-order constraint: PASS
- E24 code-fence closer constraint: PASS
- E25 list-index alignment check: PASS
- E26 emoji normalizer: PASS (engineering)
- E27 tz-aware resolver: PASS (engineering)
- E28 currency-locale map: PASS (engineering)
- E29 plural-agreement span re-decode: BORDERLINE — adjacent §1e self-revision decoding; narrow span vs full revision loop; survives only vs prompt-baseline with false-positive rate reported
- E30 token-budget ledger: PASS (accounting infra, distinct from budget-conditioned generation)
- E31 prompt-diff reviewer: PASS
- E32 adapter-capability matrix: PASS
- E33 determinism badge CI: PASS
- E34 retrieval-freshness monitor: PASS
- E35 cost-anomaly detector: PASS
- E36 logit-lens readout for verifier: FAIL — §1e logit-lens depth self-ensembling (single-lens vs ensemble is incremental; author's own nuance note concedes the overlap)
- E37 cgroup limit per request: PASS
- E38 homoglyph normalizer: PASS-leaning, NOTE §1e glitch-token adjacency is distinct class (confusable-unicode vs glitch tokens); web-leg must confirm
- E39 red-team seed bank: PASS (dataset, not harness)
- E40 token futures: PASS (weak traction — flag)

## Tally
- PASS (incl. leaning): 32
- BORDERLINE: 6 (E13, E18, E21, E22, E29, E38-note)
- FAIL: 2 (E9, E36)
- Web-verification leg (3 phrasings × arXiv/GitHub/HF/PyPI/HN/Reddit): NOT DONE — required before any novelty claim.
