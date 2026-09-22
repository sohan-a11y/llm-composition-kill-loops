# Batch2 Filter 2 (3/4): E16/E18/E19/E21/E22, 2026-09-23
# Harsh screen: framing/process-only ideas labeled as such.

## NEEDS-KILL-TEST (1)
### E22 hysteresis decoding
Single-valued T=f(H) published (EDT 2403.14541, KLD, AdapT, EAD, Selective Sampling, entropix); path-dependence untested. EMA-smoothed EDT is the cheap killer (likely captures 95%).
Kill (2-3d, Llama-3.2-1B/Mistral-AWQ, GSM8K/HumanEval/Wiki-gen): fixed T vs EDT vs EMA-EDT vs hysteresis (T_up/T_down, gap 0.15-0.2). Should-fail: shuffled-history (gain must die; else mean-T shift relabel).
GO iff beats EDT AND EMA on quality at matched diversity (≥2pp, p<0.05) + flicker ↓>30% + shuffle destroys gain. Else KILL (EMA ablation at most).

## WEAK (4)
- E16 quorum releases: PROCESS-ONLY. Collapses to sigstore + HF Cosign (single-signer published); delta is release-engineering policy, zero modeling content. Ship as eng note, not research.
- E18 beam auction: FRAMING-ONLY. Beam search already pays sunk compute + prunes top-k (= all-pay by construction); beams lack private values/incentives (heads-as-game is head-level). Kill: logprob top-k vs bid=scorer+length-penalty at iso-FLOPs + random-bids-must-crash + uniform-bids≡top-k controls. GO iff >+3pp iso-FLOPs p<0.05.
- E19 escalation insurance: BILLING + FRAMING. Premium = E[escalation]+margin by definition; FrugalGPT (avg-cost) + Tail-at-Scale hedges (p99) already own both halves. Kill: FrugalGPT-threshold vs always-hedge on real latency dist + always-duplicate-2-3x-cost + always-cheap-blows-p99 sensitivity controls. GO iff p99 -30% at <15% overhead beating naive hedge.
- E21 bigram fusion: collapses to degraded 2411.18553 (which adds hypernetwork; mean-pool is its removed ablation). Kill: fuse top-100 frequent vs 100 random-rare bigrams, mean-pool embeddings. GO iff >15% shorter AND PPL +<5% AND acc -<2pp AND frequent beats rare >3 PPL. Drop myelin story regardless.

## Tally: NEEDS-KILL-TEST 1 / WEAK 4. Remaining Filter 2: E24,E26,E28,E29,E31,E32,E33,E35,E37,E39 (10).
