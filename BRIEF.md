# PROJECT BRIEF — LLM ENGINEERING RESEARCH: FIND UN-PUBLISHED, PLAUSIBLE, TESTABLE IDEAS

**Read this entire file before doing anything else. Then read `artifacts/` before writing new code —
several mechanisms below were already built and measured; do not rebuild them.**

---

## 0. WHO YOU ARE WORKING FOR AND WHAT THEY WANT

A solo AI/ML researcher, one consumer GPU class machine (Colab T4 16GB, Kaggle ~30GB VRAM, plus CPU).
No team, no institutional compute. They have spent roughly a week on this specific sub-goal and are
frustrated: every idea investigated so far turned out to already exist in the published literature —
sometimes under a completely different name — which means building it would waste more time.

**Their standing instructions, stated repeatedly and non-negotiably:**

1. **Exhaustive verification before any claim of novelty.** Search arXiv (including current-year and
   prior-year preprints), GitHub, Hugging Face, PyPI, Hacker News, Reddit (r/LocalLLaMA, r/MachineLearning)
   using AT LEAST THREE different phrasings per idea, because the same mechanism is routinely published
   under an unrelated name. Report exact IDs/dates/repo names/star counts. Never say "novel," "unexplored,"
   or "first" unless the verification trail actually supports it.
2. **"Not found" is not "will work."** An idea absent from the literature can mean two very different
   things: (a) genuinely nobody has thought of it, or (b) someone tried it, it failed quietly, and
   negative results rarely get published. Every surviving idea must be screened for PLAUSIBILITY, not
   just novelty. If the closest related literature suggests the mechanism should collapse to something
   already known (e.g., "X turns out to just be confidence/entropy wearing a different name"), say so
   explicitly and do not present the idea as strong.
3. **Systematic enumeration and empirical elimination.** When a space is large, do not declare it "too
   big to search." Generate many candidates, screen cheaply, eliminate on measured failure, promote
   survivors. This is the working methodology throughout this project (see `artifacts/` — the composition-
   cliff screens, the role-tagged-latents controls, are all built this way).
4. **State plainly what you cannot verify.** If you cannot confirm absence after real search effort, say
   "not found after N phrasings, which does not prove absence" — never imply certainty you don't have.
5. **If you have an objection to an instruction, say so first, then proceed anyway** unless told otherwise.
6. **If you hit a hard limit (tool budget, context, time) mid-task, say so explicitly and stop** — do not
   silently truncate the deliverable and present it as complete.
7. **Do not ask the user to re-confirm things they've already told you.** Work from this brief.
8. **Minimize wasted compute/tokens.** Don't re-run experiments that are already in `artifacts/` with
   results already recorded. Don't re-derive things already established below.

**The end goal is not a paper.** It's a real, working artifact — code, a tool, a measured result — that
would get genuine attention/adoption from LLM engineers, AI practitioners, or labs, built by one person on
one consumer GPU. Fame/traction is an explicit stated goal, not a side effect. But the user has also been
told, and has accepted, that traction is not something methodology guarantees — a correct, rigorous,
replicated result is worth producing even if it doesn't go viral, and is strictly better than another week
of false-novelty claims.

---

## 1. FULL PROJECT HISTORY (READ THIS — DO NOT RE-EXPLORE THESE PATHS)

The project began as "Active Session": a continuously-running LLM that reads the internet, writes what it
learns into its own weights, verifies before committing, and grows capacity instead of overwriting old
knowledge. `artifacts/ACTIVE_SESSION_SPEC.md` and `artifacts/ActiveSession_SMF_T4.ipynb` document this
phase, including real measured numbers: ~85% lesson-commit rate, +0.18 held-out generalization, and a
caught bug where a "frozen" control arm was silently still writing (drift 14.88 instead of 0.0) — the run
was correctly discarded once caught. This established the "control must be provably inert or the
experiment is void" standard used throughout everything since.

An early benchmark notebook (`artifacts/MicroAGI_v2_honest_benchmark.ipynb`) is included as a cautionary
example: a prior "nine pillars of AGI" notebook had hardcoded `PASS` strings and no-op loss functions (a
TTT loss that mathematically reduces to driving its own output to zero, etc.) that made every test trivially
pass regardless of whether anything worked. It was rebuilt with real controls and thresholds. **Lesson:
every claimed result in this project must have a control that can fail, and the failure must be checked,
not assumed away.**

### 1a. The core measured finding: composition requires re-entry into the token stream, not mere presence in hidden state

This is the project's one fully-validated, multiply-replicated result. Full code in `artifacts/composition_cliff_FINAL.ipynb`, `artifacts/tasks_track.py`, `artifacts/model.py`, `artifacts/screen_core.py`, results in `artifacts/measured_results.json` and `artifacts/all_measured_results.json`.

**Setup:** tiny transformers (2 layers, 64-dim) trained on two synthetic composition tasks:
- S5 state tracking: track a value in {0..4} through *k* permutations of the symmetric group S5 (non-solvable
  group; NC1-complete word problem).
- Affine chain mod 17: track a value in Z_17 through *k* steps of x -> (a*x+b) mod 17 (different algebra:
  abelian ring, arithmetic composition instead of lookup composition).

**Finding 1 — the composition cliff.** One-shot (non-CoT) accuracy is near-perfect below a domain-size
threshold and collapses to chance above it — a sharp phase transition, not a gradual decline. S5: cliff
between domain 80 (0.9862) and domain 100 (0.2032). Affine: cliff between domain 20 (0.9882) and domain 40
(0.0542).

**Finding 2 — nothing "more of the same" fixes it.** A falsification battery of 15 interventions (width up
to 14x params, depth to 6 layers, recurrent looping x2/x4, RoPE, no positional encoding, 20,000 training
steps, low LR) all stayed at chance. This is the strongest-tested part of the whole project — it was
specifically designed to break the headline claim and did not.

**Finding 3 — only re-entry into the token stream fixes it, at 1/14th the parameters.** Chain-of-thought
(autoregressive, model generates its own intermediate tokens, evaluated by scoring only the final answer)
reaches 1.0000 with far fewer parameters than the largest failed one-shot config.

**Finding 4 (important, do not re-derive) — recurrent depth failure was a training-budget artifact, not
evidence the mechanism is impossible.** Kohli et al., "Loop, Think & Generalize: Implicit Reasoning in
Recurrent-Depth Transformers" (arXiv:2604.07822, Apr 2026) show compositional generalization from looping
is a grokking phenomenon requiring thousands of epochs (not steps) and zero-initialized recurrent blocks.
This project's own retest of looping with zero-init and 12,000 steps (see cell 9/10 of
`role_tagged_latents_FINAL.ipynb`) STILL failed at chance (0.2055) at this tiny model scale — so the
grokking explanation did not reproduce here either. **Current honest status: recurrent depth compositional
generalization is unresolved at this scale, in either direction. Do not assume it works OR that it's dead.**

### 1b. The role-tagged-latents result (the strongest and most-verified finding in the project)

Full code and diagnosis in `artifacts/role_tagged_latents_FINAL.ipynb`, `artifacts/role_binding_ALL.ipynb`,
`artifacts/bind_experiments/`.

**What was tested:** at k=4 (3 superposed intermediate values), oracle (ground-truth) intermediates are
injected into the residual stream between layer 1 and layer 2 of a tiny transformer via different
mechanisms, with NO extra tokens emitted:
- `baseline`: nothing injected -> chance (0.2025 S5, 0.0620 Affine)
- `sum_noroles`: 1 memory slot = plain sum of the 3 values, no role tags -> 0.6305 / 0.0910
- `raw_multi`: 3 separate slots, raw values, no role tags -> 0.5180 / 0.1180
- `hrr` (Holographic Reduced Representation: circular-convolution bind + FFT-based unbind, Plate 1995,
  role vectors FIXED/frozen) -> **1.0000 / 1.0000**, matching token-level CoT exactly
- `cot`: token-level chain of thought (upper bound) -> 1.0000 / 1.0000

Replicated 3 seeds both tasks. Multi-seed table is in the notebook.

**CRITICAL — the classical "variable binding" explanation for WHY this works was tested and KILLED by
controls. Do not re-propose classical binding theory (Smolensky TPR, Plate HRR "binding," Shastri-Ajjanagadde
temporal synchrony, VSA/HDC) as an explanation without accounting for this:**
- Shuffling which role unbinds which value (so each slot receives the WRONG filler): still 1.0000. Binding-
  by-role is NOT what the model is using.
- Bind-only, never unbound: 1.0000. Unbinding is not needed.
- Round-trip per slot with NO superposition at all (zero crosstalk, so there is nothing for roles to
  disambiguate): still 1.0000. Superposition is not needed.
- **What actually matters, isolated by controls:** ANY distinct, invertible, per-slot transform works
  (random orthogonal matrix per slot: 1.0000; random dimension-permutation per slot: 1.0000; circular
  convolution per slot: 1.0000). The SAME transform applied to every slot fails completely (0.2050 S5 —
  chance). Raw values with no transform: 0.4745-0.4560 (partial). Norms are closely matched across winning
  and failing arms (7.9-8.8), ruling out scale as the explanation.

**Honest current statement of the mechanism:** each injected value must carry a role-DISTINCT signature
(any distinct invertible transform suffices); positional/slot separation alone is insufficient; the
classical bind/superpose/unbind story is not what's happening. WHY a distinct-per-slot transform helps
while raw values don't, and why the SAME transform is actively worse than NO transform, is not understood
and is itself an open, interesting question.

**What this does NOT show, and must not be overclaimed:** the model is GIVEN the correct intermediates
(oracle injection) — it does not compute them itself. This tests whether a model can USE a value it has,
not whether it can produce one. This is the single biggest open gap in the whole project.

### 1c. The metacognitive-input kill test (in progress, inconclusive, do not re-run naively)

`artifacts/metacog_kill_test/exp.py`. Idea: train a model to consume its OWN per-token sampled-answer
confidence (probability of the sampled token + entropy) as an input feature at a "verdict" position, judging
whether its own sampled answer was right or wrong. Control arms: true signal vs. shuffled signal (across
batch) vs. zero signal — if shuffled/zero match true, the model already has this information internally and
the channel adds nothing.

**Status: the first run was invalid** — task was too easy (64 questions, memorized to 100% answer accuracy,
so there were zero wrong answers for the verdict head to ever detect; the experiment cannot measure anything
under those conditions). A fix was designed (2048 questions to keep the model genuinely uncertain; train the
verdict head only in stage 2 so answer accuracy can't re-saturate) but was not confirmed to have executed
before the session ended. **If you pick this up: verify NQ=2048 and stage-2-only loss are actually in the
file before trusting any output, and confirm `sampled_answer_acc` is meaningfully below 1.0 in the eval
before reading `verdict_acc` as informative.**

### 1d. V-RLS / Directional Precision Memory — a SEPARATE, PARKED project

A full-matrix recursive-least-squares alternative to linear attention (chunked Woodbury updates + a
closed-form prefix-Cholesky intra-chunk lemma), scoped to a 350.7M-param model, 65k-token context, single
24GB GPU. **Verification found this space is CROWDED as of 2026**: "Variational Linear Attention" (VLA,
arXiv:2605.11196, May 2026) explicitly connects linear attention to classical RLS with a Sherman-Morrison
exact inverse-covariance update and a stability theory. MesaNet/Mesalayer (von Oswald et al. 2024-2025)
already solve the underlying least-squares problem exactly or via conjugate gradient. "Preconditioned
DeltaNet" (arXiv:2604.21100, Apr 2026) covers the same "test-time regression" framing. **This project is
parked as too contested; do not resume it unless a genuinely different sub-angle is found and re-verified
fresh.**

### 1e. Ideas explicitly explored and REJECTED as already-published — DO NOT PROPOSE THESE OR NEAR-VARIANTS

This list is the accumulated exclusion set from ~30 candidates already run through novelty verification.
Treat every item below, and close paraphrases of it, as PUBLISHED. If a new candidate idea turns out on
inspection to reduce to one of these, discard it immediately and generate a different one — do not present
it.

- Continual learning / writing knowledge into weights / sparse memory finetuning / always-on self-learning
  models (cf. Lin et al. Meta, arXiv:2510.15103; Goyal et al., arXiv:2604.05248; Gupta et al., arXiv:2605.03229)
- Chain-of-thought vs one-shot composition limits, intermediate-step supervision (Merrill & Sabharwal
  arXiv:2310.07923, Li/Liu/Zhou/Ma arXiv:2402.12875, Bachmann & Nagarajan arXiv:2403.06963, NExT
  arXiv:2404.14662)
- Variable binding / tensor product representations / holographic reduced representations / vector symbolic
  architectures / role-tagged latent injection AS A NOVEL PROPOSAL (the specific empirical result in 1b
  above IS this project's own work and is fine to build on/write up, but the general mechanism area is
  heavily populated — see Feng & Steinhardt arXiv:2310.17191, TP-Transformer arXiv:1910.06611, LARS-VSA
  arXiv:2405.14436, "Attention as Binding" arXiv:2512.14709)
- Execution-trace supervision for code models, debugging MCP servers, trace stores (mcp-debugger,
  debugger-mcp, python-debugger-mcp, mcp-agent-trace all exist and are functional; NExT already does
  execution-trace-as-training-signal)
- Typed/calibrated decision endpoints, "TypeSafe AI"/"Jev" clones (many immature clones appeared within days
  of the Sept 2026 launch)
- Latent communication between multi-agent LLMs (LatentMAS, Interlat, etc.)
- Soft context compression / soft-prompt RAG (xRAG, ICAE, AutoCompressors, Cartridges, gist tokens, COCOM,
  PISCO, OSCAR — note: Cartridges-at-scale, arXiv:2606.04557, shows accuracy DROPS as more compressed
  documents are added, which the role-tagged-latents same-transform-fails result in 1b may explain
  mechanistically — this connection is a legitimate WRITE-UP angle, not a new build)
- Recurrent-depth / looped transformers / latent reasoning as a NOVEL proposal (COCONUT arXiv:2412.06769,
  Huginn arXiv:2502.05171, Ouro arXiv:2510.25741) — see 1a Finding 4 for this project's own inconclusive
  retest
- Linear attention variants, DeltaNet, recursive least squares state, fast weight programmers (Schlag/
  Irie/Schmidhuber ICML 2021, arXiv:2102.11174 — already mainstream), modern Hopfield networks (Ramsauer
  et al. arXiv:2008.02217 — already mainstream, Hopfield won 2024 Nobel Prize)
- Feeding the model's own uncertainty/confidence back as a trained input channel — PARTIALLY explored by
  this project itself (1c), inconclusive; nearest prior art arXiv:2605.25459 (probing shows models may
  already latently encode next-step uncertainty)
- Copy-reference / copy-emit decoding to cut output tokens (Copy-as-Decode, arXiv:2604.18170, Apr 2026)
- Adaptive per-token compute / early exit (CALM, NeurIPS 2022, arXiv:2207.07061; many 2025-26 follow-ups)
- Mixture-of-Depths (arXiv:2404.02258)
- Learned/adaptive samplers (arXiv:2603.09065, DiffSampling arXiv:2502.14037, min-p ICLR 2025)
- Entropy-adaptive temperature (arXiv:2502.06833, arXiv:2603.03310, arXiv:2601.01714)
- Dynamic/adaptive tokenizers (arXiv:2411.18553, TokenAdapt arXiv:2505.09738, arXiv:2512.03989)
- Self-revision decoding (Stream of Revision arXiv:2602.01187, NAVIRA arXiv:2606.06031)
- KV cache pruning/eviction/compression/cross-request reuse (SP-KV arXiv:2605.14037, SAGE-KV arXiv:2503.08879,
  AttentionPredictor arXiv:2502.04077, CacheBlend, EPIC, PromptCache, vLLM/TensorRT-LLM native features)
- Dynamic/per-token quantization precision (MoBiQuant arXiv:2602.20191, RAMP arXiv:2603.17891, MoQAE ACL 2025)
- Token-budget-conditioned generation (Budgeted Attention Allocation arXiv:2605.05697, SelfBudgeter
  arXiv:2505.11274 — reports 61% length compression on math reasoning with maintained accuracy)
- Token-level loss reweighting (RHO-1/Selective LM NeurIPS 2024, PriFT arXiv:2606.09396)
- Speculative decoding, prompt-lookup decoding (Saxena's PLD, in HF transformers + vLLM already)
- Diffusion language models as a general category
- LLM evaluation harnesses and agent/tool-use benchmarks (lm-eval-harness, inspect_ai, promptfoo, DeepEval,
  Ragas, TruLens, MCP-Bench, tool-eval-bench, ToolFailBench)
- Steering-vector / activation-steering methods and closed-loop control of them (STU-PID arXiv:2506.18831
  already does PID-controlled steering)
- FFT/spectral repetition detection (SpecRA, Nougat arXiv:2308.13418)
- Layer permutation/reordering at inference ("Transformer Layers as Painters" arXiv:2407.09298, PoLar
  arXiv:2606.06574, LayerShuffle arXiv:2407.04513)
- KV-cache rollback/backtracking on degeneration (arXiv:2604.18567, arXiv:2608.14653)
- LoRA adapter ensembles/voting at decode (ELREA arXiv:2502.00089, MoLoRA arXiv:2603.15965)
- Marginalizing over tokenizations at inference (Cao & Rimell 2021; arXiv:2510.20208)
- Attention-head "auction"/budget allocation (arXiv:2605.05697, BudgetFormer arXiv:2604.22583)
- Spaced-repetition data ordering (SRT arXiv:2608.17530, LFR arXiv:2409.06131)
- Test-time input augmentation/paraphrase self-consistency (arXiv:2608.09351)
- MC-dropout / weight-dropout ensembles for calibration (Gal & Ghahramani 2016 lineage)
- Glitch-token detection/repair (Fishing for Magikarp, arXiv:2405.05417, EMNLP 2024)
- Attention-entropy rescaling for long context (Ms-PoE arXiv:2403.04797, YaRN)
- Logit-lens depth self-ensembling (arXiv:2506.01951, Tuned Lens)
- Filler/pause tokens at inference with no training (arXiv:2509.24884, Korea University, Sept 2025)
- Gaussian weight-perturbation semantic entropy for hallucination detection — PARTIALLY explored; closest
  prior art arXiv:2502.03799 (Qualcomm AI Research + UC Santa Barbara, ICLR 2026) already does an
  activation/parameter-noise Bayesian proxy; genuine gap is explicit Gaussian noise on BASE WEIGHT MATRICES
  specifically, untested — flagged as weak (small effect size in closest prior work: AUROC 75.70 -> 77.21)
- Speculative-decoding draft/target rejection as a hallucination-detection diagnostic — PARTIALLY explored;
  closest prior art (arXiv:2609.05274, arXiv:2604.14682) found the entropy-acceptance correlation is
  "consistently negative but weak" (rho in [-0.20,-0.15]), meaning this idea's most likely outcome is that
  it collapses to target-model confidence and fails its own kill control. **Flagged by the user's own
  review as probably not worth building — the evidence already available suggests failure is the likely
  outcome, not just an untested possibility.** Do not present this as a strong candidate without new
  evidence changing that assessment.

---

## 2. THE CURRENT STANDING TASK

Generate a genuinely large pool of candidate original LLM-engineering ideas (aim for enough raw candidates
that ~21 can survive BOTH filters below — based on the prior batch's hit rate of roughly 2 survivors per 21
fully-verified candidates, this likely means generating on the order of 100+ raw candidates, not 30).

Every candidate must pass TWO independent filters, and both must be reported, not just the first:

**FILTER 1 — NOVELTY.** Search arXiv, GitHub, HuggingFace, PyPI, Hacker News, Reddit (r/LocalLLaMA,
r/MachineLearning) with at least 3 distinct phrasings per idea. Verdict: PUBLISHED (discard) /
PARTIALLY EXPLORED (state the exact remaining gap) / NOT FOUND (state plainly this means "not found after
these searches," not proven absent). Cross-check every survivor against Section 1e above before reporting
it — many close variants of already-excluded ideas will recur under new names; catch this yourself before
presenting anything.

**FILTER 2 — PLAUSIBILITY.** For every idea that passes Filter 1, actively look for reasons it would fail —
does the closest adjacent literature suggest the mechanism reduces to something already known or already
shown weak? State this explicitly, the way the user's own review caught the speculative-decoding idea
collapsing to "target confidence in disguise" using evidence that was already in the report but under-
weighted. An idea should not be called a strong candidate just because it's unpublished — unpublished-and-
probably-fails is common, and must be labeled as such, not silently included as if it were unpublished-and-
promising.

For every idea that survives BOTH filters, give: the precise one-sentence claim; the single most uncertain
assumption; the cheapest decisive 1-3 day T4/Kaggle experiment with an explicit designed-to-KILL control and
a stated go/no-go threshold; the traction case (who adopts it, why, what the demo looks like); the most
likely failure mode even if it's not disqualifying.

**Generation strategies to use (vary them, don't rely on one):**
(a) Invert foundational assumptions across every layer (tokenization, embedding, attention, layer ordering,
training curriculum, sampling, serving, memory, numerical precision, the relationship between one model and
one set of weights, etc.) — go well beyond the obvious ones already tried.
(b) Cross-domain transfer from compilers, databases, operating systems, signal processing, control theory,
error-correcting codes, cryptography, auction theory/economics, biology, physics — ask what the direct LLM
analog of an established mechanism in that field would be.
(c) Observed-anomaly mining — real, under-explained phenomena practitioners report (search current
r/LocalLLaMA, Hacker News, GitHub issues on llama.cpp/vLLM/transformers/SGLang for open, unexplained
complaints) and ask what mechanism would exploit or fix the anomaly.
(d) Tooling gaps — things engineers explicitly say they wish existed where the missing piece is a mechanism,
not a UI.

**Final deliverable format:** a ranked table of all survivors (both filters passed), each with the full
write-up specified above, ordered by (novelty confidence) x (plausibility) x (feasibility on this hardware)
x (traction potential). State the true count if it's not exactly 21 — do not pad to hit a round number. State
explicitly the total raw candidate count generated vs. how many survived each filter, so the yield rate is
visible.

---

## 3. STANDING TECHNICAL CONVENTIONS FOR ANY NEW EXPERIMENT CODE

- PyTorch, small models, CPU-testable-then-GPU-scalable. See `artifacts/bind_experiments/core.py` and
  `artifacts/composition_cliff_FINAL.ipynb` for the established house style: synthetic task classes with
  `.sample()`/`.trace()`/`.batch()`/`.batch_cot()`, a shared `fit()`/`evaluate()` harness, explicit chance-
  level baselines printed alongside every result, multi-seed tables before trusting any single number.
- Every experiment needs a sanity check that the harness itself works (e.g., k=1 / trivial-difficulty case
  should hit ~100%) BEFORE trusting a negative result on the real task — this project was burned once by a
  broken-harness false negative (see the k=1 sanity checks throughout `composition_cliff_FINAL.ipynb`).
- Every experiment needs at least one control specifically designed so that IF the hypothesis is wrong, the
  control will show it — not a control that happens to also confirm the hypothesis. The role-tagged-latents
  work is the model example: the "same transform for every slot" control was the one that actually
  discriminated between competing explanations, and initially a shuffle-test was written incorrectly (it
  shuffled the same roles object used for both bind and unbind, so the permutation canceled and the test was
  a silent no-op) — CHECK that a "should fail" control can actually fail before trusting that it passed.
- Report exact numbers, not vibes. State chance-level baseline next to every accuracy number.

---

## 4. WHAT TO DO FIRST

1. Read `artifacts/` in full — do not skip the notebooks assuming the summaries above are complete; the
   notebooks contain the exact code and full numeric tables.
2. Confirm you understand Section 1e (the exclusion list) well enough to recognize a near-variant when you
   generate one.
3. Begin Section 2 (the standing task). Report progress in stages: (i) raw candidate list with generation
   strategy tagged per idea, (ii) Filter 1 results for all of them, (iii) Filter 2 results for Filter-1
   survivors, (iv) final ranked write-up.
4. If you hit a tool/context/time limit before finishing, stop and report exactly how far you got and what
   remains — per the user's standing instruction in Section 0.7.
