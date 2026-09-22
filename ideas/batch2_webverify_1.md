# Batch2 web-verification 1/5: E1–E8 + E10 (E9 already §1e-FAIL), 2026-09-23
# ≥3 phrasings × arXiv/GitHub/HF/PyPI/HN/r/LocalLLaMA+r/MachineLearning per idea.
# NOT FOUND = not found in returned results, does not prove absence.

## E1 unembedding-norm gating — NOT FOUND
Phrasings: unembedding norm gating decode-only; output embedding norm scaling inference; unembedding matrix norm bias token frequency; github unembedding norm scaling.
Sources: HF models API [] ; HN Algolia nbHits 0; Reddit search 403 blocked; PyPI search JS-challenge, no package.
Neighbors (distinct): Logit Lens (decode w/ W_U, no scaling); Tuned Lens 2303.08112 (per-layer affine, not per-token norm); OEC 2601.02031 (training-time centering); Cho COLING25 frequency-encoding analysis; 2606.07502 EmbedFilter; 2212.09686 unigram-bias init.
Gap: norm↔frequency correlation established, active decode-only intervention untested.

## E2 causal-mask future-dropout — PARTIALLY EXPLORED
Phrasings: causal mask dropout reveal future; future leakage dropout pretraining; non-causal train causal test; github future tokens.
Neighbor: Forgetful Causal Masking 2210.13432 (masks PAST, eta 0.15, PaLM gains) — opposite direction. TLM 2310.18738 (past-only). Future-token prediction 2410.18160 (objective change, not holes). Future Lens CoNLL23 (analysis).
Gap: stochastic FUTURE leak p~0.05 train / strict test — not found.

## E3 prompt-last training — PARTIALLY EXPLORED
Phrasings: completion-before-prompt reverse ordering; answer-before-question training; response-first instruction tuning; reverse prompt completion order.
Neighbors: Reverse Training 2403.13799 (within-sequence reversal + [REV], forward test); SAPT Findings-ACL24 (chunk permute); LEDOM/TRLM (full R2L, symmetric eval); Response Tuning Findings-EMNLP25 (response-only train); LookAhead 2503.19041 (m=6 preview); preemptive-answer CoT 2405.20902 (prompting observation).
Gap: full [completion][prompt] train / [prompt][completion] test asymmetry — not found.

## E4 SSA-form prompts — NOT FOUND
Phrasings: SSA form prompts LLM; arxiv SSA prompt single assignment; github SSA prompting compiler; variable renaming coreference CoT; HN/Reddit SSA prompt; PyPI ssa-prompt.
Neighbors (distinct): GE-Reasoning 2311.09762 (typed variables, no SSA discipline/phi); predicate renaming 2510.25517 (opposite); SOAP@PLDI24 (analysis→prompt); CoSm 2401.09074 (compiler-trace, no SSA).
Gap: single-assignment versioning + phi-merge discipline — not found.

## E5 loop-unroll CoT template — PUBLISHED, discard
Least-to-Most 2205.10625; DecomP 2210.02406 (allenai/DecomP); Successive Prompting EMNLP22 (dDua repo); Self-Ask pattern; GenDec 2402.11166. k=2→k=1 prompting with no weight change is prior art; "loop-unroll" is relabel.

## E6 covering-index RAG — PARTIALLY EXPLORED
Phrasings: covering index RAG answer projection; arxiv covering index RAG cache; RAG cache answer skip retrieval; HF answer caching; github covering-index RAG; reddit covering index RAG.
Neighbors: FAISS/Chroma semantic cache (query→text, no covering contract); HF Discuss 180273 (asks the question, no answer); RAGCache 2404.12457 (KV reuse, still retrieves IDs); contextual-summary cache 2505.11271 (closest, no HIT/MISS contract); IndexRAG/SiReRAG (precompute, still fetch-then-generate).
Gap: (key, answer-projection + provenance) + HIT-skips-fetch contract + invalidation bounds — not found.

## E7 two-phase-commit tools — PARTIALLY EXPLORED (load-bearing overlap flagged)
Phrasings: 2PC tool execution agents; transactional agent atomic commit; Atomix Cordon; github 2PC MCP; PyPI/HF transactional agents; HN/Reddit transactional agents; TCC prepare-confirm-cancel.
Neighbors: Atomix 2602.14849 (frontier commit+compensation; explicitly rejects pure tool-side 2PC as undeployable); Cordon 2606.17573 (Prepare/Validate/Commit-or-Abort semantic protocol); saga-agent/pherix PyPI; TCC theory (Seata/Oracle).
Gap (narrow): standard prepare() interface + coordinator recovery on irreversible tools where Atomix says impossible. Without (a)-(c), textbook restatement.

## E8 nice-level preemptible reasoning — PUBLISHED, discard
FastServe 2305.05920/NSDI26 (per-token preemption); vLLM PRIORITY scheduler + #6077/#40004; ProServe 2512.12928 ("nice values in Linux" quote); HiveMind 2604.17111 Table 2; IBM Token Slice 2026-04-07. Nice→priority + token-boundary preemption is prior art.

## E10 matched-filter retrieval — NOT FOUND
Phrasings: matched filter retrieval query template; convolution query embedding doc stream; matched filter IR text embeddings; HF/PyPI matched filter retrieval.
Neighbors (must-differentiate): ConvKNRM/DRMM CNN-over-interaction-matrix; ColBERT MaxSim; sliding-window passage scoring; cosine ANN. None frames query as time-reversed template with detection/peak-picking theory; embedding noise is not AWGN (plausibility risk — flag for Filter 2).
Gap: matched-filter SNR-optimality transfer claim — not found.

## Tally (this file): NOT FOUND 3 (E1, E4, E10) / PARTIAL 4 (E2, E3, E6, E7) / PUBLISHED 2 (E5, E8)
## Remaining web batches: E11–E20, E21–E30, E31–E40.
