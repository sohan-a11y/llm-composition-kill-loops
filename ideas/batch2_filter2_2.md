# Batch2 Filter 2 (2/4): E2/E3/E6/E7/E11/E12/E13, 2026-09-23
# Harsh screen: unpublished-and-probably-fails labeled WEAK.

## NEEDS-KILL-TEST (2)
### E6 covering-index RAG (leaning WEAK)
90% semantic cache (GPTCache/Redis). Survives only on provenance-invalidation economics.
Kill (1-2d, MiniLM+FAISS+Llama-3-8B, NQ 5k + Hotpot 1k + 200 edits; 40% repeat / 30% paraphrase / 15% entity-swap / 15% time-sensitive): standard RAG vs GPTCache+TTL vs covering(index key, answer-projection, provenance). Should-fail: entity-swapped + post-mutation stale queries MUST MISS (HIT = hallucination machine = instant KILL).
GO iff hit>20% at precision≥95% + invalidation recall≥95% (zero stale) + p95 ≥2x better than RAGCache-KV baseline.

### E13 class-scheduled temperature
Coarser than DecoRTL token-class, static vs AdapT/ATS/Top-H dynamic, hand-table vs TAMPO learned. Only product loophole.
Kill (2-3d, Llama-3-8B, GSM8K/HumanEval/MTBench splits, frozen 4-way classifier): fixed T vs entropy-adaptive Top-H vs class table (val-swept). Should-fails: random-class + permuted-T must not-win/hurt.
GO iff class beats BOTH fixed and entropy by ≥3pts + better ECE held-out. If ≈ entropy or only beats fixed: KILL.

## WEAK (5)
- E2 future-dropout: anti-dropout (adds info = cheat incentive, cf. FCM masks past). Kill: nanoGPT 60M causal vs 5%-leak, strict-causal eval; leak-enabled-eval control must show PPL delta >1.0 (proves cheat channel). GO iff causal-PPL(B) ≤ causal-PPL(A)+0.02. Expected kill (0.2-1.0 worse).
- E3 prompt-last: train-backwards-test-forwards; reversal-curse disproves (needs forward order or bidir augment). Kill: 500 fictitious pairs, [Q][A] vs [A][Q] to 100% train fit; reversed-eval control must ≈100%. GO iff forward-EM(B) ≥ 0.9×EM(A). Expected 0-5% vs 95%.
- E7 2PC tools: trapped by Atomix (tool mod required = concede; else impossible on irreversible tools; saga owns fallback). Kill: 5 irreversible tools + crash injection 50 runs, zero custom adapters. GO iff 100% exactly-once zero orphans. Expected kill (orphan/double-apply); escrow re-scope = Cordon/saga clone.
- E11 TPDF dither: non-subtractive TPDF raises variance LSB²/12→~LSB²/4; GPTQ minimizes (w-q)ᵀH(w-q), noise adds Tr(H)σ² by construction; LLM weights heavy-tailed (uniform-in-bin violated). Kill: TinyLlama/Phi-2 RTN-4b, base vs RPDF vs TPDF×2 + 3-LSB must-explode control. GO iff TPDF beats base >3% PPL 3/3 seeds AND beats RPDF.
- E12 length-prior subtraction: = length-penalty rebrand; prior non-stationary, non-decomposable per-token, double-counts learned p(L). Kill: CNN/DM + GSM8K, prior-fit vs length_penalty sweep 0.6-1.4 + shuffled-prior must-≈-null control. GO iff beats best penalty >2 ROUGE-L length-controlled, no drift. Else KILL.

## Tally: NEEDS-KILL-TEST 2 / WEAK 5. Remaining Filter 2: E16,E18,E19,E21,E22,E24,E26,E28,E29,E31,E32,E33,E35,E37,E39 (15).
