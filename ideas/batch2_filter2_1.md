# Batch2 Filter 2 (1/2): 10 NOT FOUNDs → plausibility, 2026-09-23
# Harsh screen: unpublished-and-probably-fails labeled WEAK, not smuggled in.

## NEEDS-KILL-TEST (2)
### E4 SSA-form prompts
Adjacent wins prove structure helps (RwG, GraphPRM +9%, Ge BFS +12.8pp, SOAP recall +25pp) but rename-alone HARMS (-8.1pp, RAnA). Must prove phi > triplets.
Kill (2d, T4+API, 250 ProofWriter/CLUTRR/bAbI + Hotpot-subset with reassignment): CoT vs rename-only (should-fail #1, must ≈ baseline) vs triplet-variables vs SSA+phi + shuffled-phi (should-fail #2). GO iff SSA beats best rename/triplet by ≥5pp (McNemar p<0.05) with token overhead reported. Else KILL (jargon or privacy-harm).

### E20 refractory-period heads
Conditional cooldown ≠ random DropHead in principle; threshold/recovery ad-hoc, must prove timing matters.
Kill (2-3d, GPT-2-small/TinyStories, matched idle budget): vanilla vs DropHead vs refractory (fire>τ → mask N tokens, linear recovery); metrics val ppl + firing overlap + repeat rate. Should-fails: matched DropHead ≥ refractory → kill; shuffled-timing ties true → timing claim dead → kill.
GO iff refractory beats vanilla AND matched DropHead (>noise, ≥1% ppl or diversity at fixed ppl) AND shuffle destroys gain.

## WEAK — kill, do not run (8)
- E1 norm gating: collapses to temp/unigram relabel (OEC Prop.3 razor: centering changes nothing; Tuned Lens needs learned map). Kill: sweep α vs temp sweep vs log-unigram subtraction + shuffled-norm (must be null). GO iff beats both >0.5 PPL + >2pp with D null.
- E10 matched filter: AWGN assumption false for embeddings; = uniform-weight MaxSim with length bias (ColBERT/ConvKNRM gains came from learned kernels/distillation). Kill: fixed-encoder BEIR (FiQA/SciFact/NFCorpus) cosine vs mean vs MaxSim vs matched + non-reversed/phase-scrambled control. GO iff beats MaxSim ≥+0.02 on 2/3 AND (D-E) ≥0.03.
- E15 CRC32 args: json+CRC textbook; schema dominates; adversarial recompute bypass 100% by construction. GO iff catches >5% real corruptions missed by schema in explicitly non-adversarial model.
- E23 rank-order FSM: output-equivalent to generate+re-sort; rank ≠ reasoning order. GO iff FSM > re-sort on citation-F1 no fluency drop AND task rewards rank-order.
- E25 index check: regex lint, unpublishable. GO iff rate >2-3% AND fixing flips >1% accuracy.
- E27 tz trap: USE_TZ/ruff-DTZ folklore; baseline fires 20/20 injected bugs by construction. GO iff >15pp incremental real-bug recall causing test failure.
- E30 refund ledger: Stripe logic, savings come from aborter not ledger (refund=0 and abort_rate=0 controls must show zero delta). GO iff >20% waste cut vs cap with learned aborter.
- E37 cgroup/request: fights PagedAttention/prefix sharing; migration overhead swamps. GO iff <5% p99 hit at 100RPS + >95% prefix-hit + prevents OOM where baseline OOMs.

## Tally: NEEDS-KILL-TEST 2 (E4, E20) / WEAK 8. Remaining Filter 2: 21 PARTIALs.
