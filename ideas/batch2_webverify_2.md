# Batch2 web-verification 2/5: E11–E20, 2026-09-23
# ≥3 phrasings × arXiv/GitHub/HF/PyPI/HN/r/LocalLLaMA+r/MachineLearning per idea.
# NOT FOUND = not found in returned results, does not prove absence.

## E11 TPDF dithering pre-quant — PARTIALLY EXPLORED (narrow)
Neighbors: 1802.02271 (uniform dither + lattice quant, ResNet/AlexNet pre-LLM); 2004.04729 dithered backprop (backward sparsity, not PTQ). ArXiv TPDF+dither+quant = 0; GH only audio dither + optical MNIST; HF []; PyPI dithering pkg image-only; HN 1 unrelated hit; Reddit 403.
Gap: TPDF specifically + GPTQ/AWQ-era LLM weight-only/low-bit eval. Do not claim dithering-for-quant as novel.

## E12 length-prior gravity comp — PARTIALLY EXPLORED
Problem crowded (2002.02492 consistency; 2205.00659 label-smoothing length bias; LBR 2607.04270; D3 2406.14900; CASTILLO dataset). Exact "length prior subtraction pre-sampling" = 0 (arXiv exact 0, GH 0, HN 0, HF [], PyPI 404).
Gap: explicit prior estimation + pre-sampling subtraction vs post-hoc norm/offset/masking.

## E13 gain-scheduled temp by prompt class — PARTIALLY EXPLORED (narrow survival)
Neighbors: AdapT 2309.02772 (token difficulty); ATS 2409.19817 (per-token calibration); DecoRTL 2507.02226 (token syntactic class — closest, will be cited); TAMPO 2602.11779 (RL meta-policy); Top-H 2509.02510 (entropy-based — the signal E13 avoids). gain-scheduled+prompt = 0 everywhere.
Survives only head-to-head vs AdapT/ATS/Top-H + ablated taxonomy. Never "first adaptive temperature".

## E14 batch parity row — PUBLISHED, discard
ParM 1905.00863 (SOSP19, coding group/parity batch) + Thesys-lab/parity-models (49 stars). Parity-across-batch for fault tolerance is prior art; exact-XOR variant at most incremental.

## E15 CRC over tool args only — NOT FOUND (weak novelty flag)
Schema/allowlist validation = published practice (r/LocalLLaMA 1r288w3 etc.); file-checksum tools unrelated. CRC32-over-canonicalized-args as transmission check, CoT excluded — no exact hit. Novelty = trivial CRC+JSON combo; flag as weak.

## E16 threshold-signed releases — PARTIALLY EXPLORED
Single-signer weight signing PUBLISHED: sigstore/model-transparency (246 stars, 2023-08-23) + model-signing PyPI + 2505.22778 + HF Cosign docs. Gap: m-of-n maintainer quorum as release gate — not found. Only quorum is novel, not signing.

## E17 blind-token PII inference — PUBLISHED, discard
PrivacyRestore 2406.01394 + HaS 2309.03057 + PromptGraph 2607.10709 + cloakpipe (39 stars)/Preserve/privalyse/CloakLLM/PrivAiTe + preserve-pii/masked-ai PyPI + ai4privacy HF. Mask→generate→restore crowded; commit/reveal = relabel of AES/HMAC vaulting.

## E18 all-pay beam auction — PARTIALLY EXPLORED
Neighbors: heads-as-game 2602.00861 (head level, not beams); mechanism-design-for-LLMs 2310.10826; ad-auctions RAG; beam literature (best-first/diverse/trie, no auction). GH/HF/PyPI/HN/Reddit: no beam+all-pay.
Gap: formal losing-beams-pay rule at candidate level. Risk: reads as top-k relabel unless bidding currency does work beyond score pruning.

## E19 escalation insurance — PARTIALLY EXPLORED (relabel risk)
Halves published: FrugalGPT 2305.05176 cascades (average-cost) + Tail-at-Scale hedged/tied requests (p99 cap). No "escalation insurance" formalism found.
Gap: premium-vs-p99 pricing analysis for LLM cheap-call hedge. Without it, collapses to hedged-requests + FrugalGPT.

## E20 refractory-period heads — NOT FOUND (cleanest of batch)
SNN refractory (RPLIF 2509.17769) + spiking transformers (neuron-level) + AttentionDrop/DropHead/dormant-heads (not firing-triggered timers) all published at other levels. Head forced-idle-N-tokens-after-hard-fire with recovery dynamics: zero hits everywhere searched.
Gap requires: hard-fire detector + N-token mask + recovery spec, ablated vs DropHead/dormant baselines.

## Tally (this file): NOT FOUND 2 (E15-weak, E20) / PARTIAL 6 (E11, E12, E13, E16, E18, E19) / PUBLISHED 2 (E14, E17)
## Cumulative batch2 web: NOT FOUND 5 (E1,E4,E10,E15,E20) / PARTIAL 10 / PUBLISHED 4 (E5,E8,E14,E17). Remaining: E21–E40.
