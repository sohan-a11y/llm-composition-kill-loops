# Batch2 web-verification 3/5: E21–E30, 2026-09-23
# ≥3 phrasings × arXiv/GitHub/HF/PyPI/HN/r/LocalLLaMA+r/MachineLearning per idea.
# NOT FOUND = not found in returned results, does not prove absence.

## E21 bigram fusion fast-path — PARTIALLY EXPLORED
Bio framing not found. Core published: dynamic tokenization 2411.18553 (ACL25, 13 stars, batch merge top-10 pairs, >20% shorter); Fusion Token ICLR-subm (bi→10-grams); MultiTok LZW 2410.21548; ADAT/AdaptBPE/FLEXITOKENS; MTP family (2404.19737 etc.).
Gap: frequency-triggered inference shortcut vs tokenizer-retrain baseline at matched compression. Myelin story is not a contribution.

## E22 hysteresis decoding — PARTIALLY EXPLORED
Exact hysteresis (T_up != T_down on dH/dt) not found. Entropy-adaptive T published: EDT 2403.14541 (single-valued T=f(H)); KLD dynamic T; AdapT AAAI24; EAD 2502.06833 (model-switch anti-oscillation, closest anti-flicker); Selective Sampling 2510.01218; entropix/qr-sampler (stateful, not asymmetric).
Gap survives only with shuffled-history control + flicker metric.

## E23 citation-order constraint — NOT FOUND (narrow)
Input-order work (Cuconasu EMNLP25, Stable-RAG ACL26, OKH-RAG) is prompt order, not output-citation order. Citation-content constraints (ReClaim 2407.01796, FullCite, CiteGuard, VeriCite) constrain what, not order.
Gap: FSM mask allowed_next_cite > last_cite_rank + drift metric + post-hoc-resort baseline. Narrow; cite both families.

## E24 fence-matched closer — PARTIALLY EXPLORED (narrow guard)
LatentMD 2609.06993 (benchmark, 38% boundary-broken, no fix) + Format-Tax 2604.03616 (generic constrained decoding) adjacent. GH code-fence-constrained = 0; HN 0.
Gap: fence-type-aware closer (``` vs ~~~, info-string/length) — not found. Engineering guard, not research.

## E25 list-index alignment check — NOT FOUND (narrow, ~zero traction)
arXiv/GH/HF/HN all zero-relevant. Assertion-style micro-guard; novel-as-guard at best, no research artifact.

## E26 emoji variant normalization guard — PARTIALLY EXPLORED (narrow guard)
Attack side published (Emoji Attack 2411.01077, emoji-jailbreak 2601.00936, Smiley-Hostile 2509.11141). Defense "normalize VS15/VS16/ZWJ/modifiers pre-tokenize-and-judge": GH 0, arXiv 0, HN 0; PyPI generic emoji lib only.
Gap: variant-normalization eval guard — not found. Attacks motivate it.

## E27 tz-aware reject-naive trap — NOT FOUND (lint-tier, not research)
All surfaces zero for LLM-specific trap; python-dateutil exists as utility. Decades-old SE guidance (USE_TZ/zoneinfo); "NOT FOUND" = no LLM-novel traction, not new practice.

## E28 currency-locale pre-arithmetic gate — PARTIALLY EXPLORED
Exact gate not found (arXiv/GitHub/HF/HN zero). Neighbor: socio-cultural math localization 2508.14913 (offline dataset construction, not inference gate). Symbolic-Arithmetic 2410.15580 ruled out (no locale).
Gap: locale-conditional $/€/¥ normalization + arithmetic-accuracy ablation ($→USD/MXN/ARS, ¥→JPY/CNY).

## E29 span-scoped agreement re-decode — PARTIALLY EXPLORED
Neighbors full-output/step-level: Self-Refine 2303.17651, SSR 2511.10621, Self-Refine-IT 2405.00402; homophone/gender analyses (no fix); CIDER remasking is channel-decoding, not NLP.
Gap: mismatch-triggered determiner–noun span re-decode + agreement metric vs full-revision baselines.

## E30 token-budget refund ledger — NOT FOUND
All token-budget arXiv hits generation-side (TALE 2412.18547, BG-MCTS 2602.09574, length-control 2406.10278/2308.12030/2412.14656/2508.13805). Serving neighbors lack refund: SatGate (11 stars, caps+attribution, no abort credit), Forge (VRAM budget only). vLLM refund/usage-accounting: 0.
Gap: per-request accounting + abort refund. Risk: LiteLLM/OpenRouter billing + Reddit not exhaustively covered.

## Tally (this file): NOT FOUND 4 (E23-narrow, E25-lint, E27-lint, E30) / PARTIAL 6
## Cumulative batch2 web: NOT FOUND 9 / PARTIAL 16 / PUBLISHED 4. Remaining: E31–E40.
