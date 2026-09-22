# Batch2: 40 new raw candidates (110 -> 150), tagged, novelty UNVERIFIED
# Strategies varied; must pass Filter1 (3 phrasings) + Filter2 + §1e cross-check before promotion.

E1 [invert] Unembedding-norm gating: scale logits by output-embedding norm per token at decode only
E2 [invert] Train with causal mask dropout (randomly reveal future 5%) then test strict causal
E3 [invert] Prompt-last training: completion before prompt, infer prompt-first
E4 [xdomain-compilers] SSA-form prompts: single-assignment rename of entities before reasoning
E5 [xdomain-compilers] Loop-unroll CoT: unroll k=2 into k=1 steps via template, no model change
E6 [xdomain-db] Covering-index RAG: index stores answer-projection, avoids doc fetch on hit
E7 [xdomain-db] Two-phase commit tool exec: prepare all tools, commit iff all prepare-ok
E8 [xdomain-os] Nice-level reasoning: low-priority requests preemptible at token boundaries
E9 [xdomain-os] OOM-killer for KV: kill lowest-value prefix under pressure with score
E10 [xdomain-signal] Matched-filter retrieval: correlate query embedding as template over doc stream
E11 [xdomain-signal] Dithering before quantization: add TPDF noise to weights pre-quant
E12 [xdomain-control] Feedforward Gravity comp for length bias: subtract length prior before sampling
E13 [xdomain-control] Gain-scheduled temp: schedule temp by prompt class, not entropy
E14 [xdomain-ecc] Interleaved parity across batch: parity row over batch dimension for fault tolerance
E15 [xdomain-ecc] CRC over tool args only (not CoT): cheap, exact domain
E16 [xdomain-crypto] Threshold-sign model releases: 2-of-3 maintainers sign weight hash
E17 [xdomain-crypto] Blind-token inference: mask PII spans with commitments, reveal after
E18 [xdomain-econ] All-pay auction for beam slots: beams bid compute, losers pay cost (prunes weak)
E19 [xdomain-econ] Escalation insurance: pay small premium (extra call) to cap worst-case latency
E20 [xdomain-bio] Refractory period heads: head forced idle N tokens after firing hard
E21 [xdomain-bio] Myelination fast-path: frequent bigrams get fused token shortcut at inference
E22 [xdomain-phys] Hysteresis decoding: different temp for rising vs falling entropy (prevents flicker)
E23 [anomaly] Citation-order drift: enforce citation order = retrieval rank order as constraint
E24 [anomaly] Code-fence leak fix: constrained closer that must match opener type exactly
E25 [anomaly] List-index shift: detect off-by-one in enumerated reasoning via alignment check
E26 [anomaly] Emoji-escape guard: normalize emoji variants before tokenization for safety evals
E27 [anomaly] Timezone-naive bug trap: force tz-aware resolver, reject naive datetimes
E28 [anomaly] Currency-symbol confusion gate: map $/€/¥ by locale before arithmetic
E29 [anomaly] Plural-agreement stabilizer: re-decode determiner-noun spans on mismatch
E30 [tooling] Token-budget ledger: per-request token accounting with refund on abort
E31 [tooling] Prompt-diff reviewer: semantic diff of prompt versions with impact estimate
E32 [tooling] Adapter-capability matrix: eval each LoRA on frozen suite before registry publish
E33 [tooling] Inference-determinism badge: batch-invariance test in CI, badge on pass
E34 [tooling] Retrieval-freshness monitor: embedding drift alert when corpus shifts
E35 [tooling] Cost-anomaly detector: flag requests >3sigma tokens/cost for review
E36 [invert] Logit-lens early readout as feature for verifier (not ensemble - excluded nuance: single-lens feature)
E37 [xdomain-os] Cgroup memory limit per request with graceful truncation
E38 [anomaly] Homoglyph attack gate: confusable-unicode normalizer before safety check
E39 [tooling] Red-team seed bank: versioned adversarial prompts with provenance
E40 [xdomain-econ] Token futures: reserve future capacity at fixed price for batch jobs
