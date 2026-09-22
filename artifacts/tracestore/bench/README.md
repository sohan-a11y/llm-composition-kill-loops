# Fix-rate benchmark

**Does an execution trace improve an LLM's ability to repair broken code?**

This is the experiment that can prove `tracestore` worthless. It should be run before anyone believes the tool works.

## Run it

```bash
export ANTHROPIC_API_KEY=sk-...
python3 run_bench.py --model claude-sonnet-4-6 --k 4
```

~144 API calls at k=4 (18 bugs x 2 arms x 4 attempts). Verify the pipeline first with no API cost:

```bash
python3 run_bench.py --mock
```

## Design

| choice | reason |
|---|---|
| **Paired** | Every bug runs under both arms with an identical prompt. The trace block is the only difference. |
| **k attempts at temp 0.7** | One sample per cell is dominated by sampling noise. Reported as fix rate and pass@k. |
| **Two bug categories** | `trace_revealing` (wrong intermediate value) vs `trace_neutral` (visible from source, or on a path the failing call never executes). |
| **Arm order shuffled** | Any API drift over the run cannot land systematically on one arm. |
| **Subprocess verification** | A fix counts only if the real tests pass in a clean interpreter with a timeout. No self-report, no judge model. |
| **Sign test** | Exact two-sided test over discordant pairs. Prints a warning when p > 0.05. |

## The internal control

`trace_neutral` exists to catch a broken experiment. A trace cannot explain a bug on a path it never executed:

```python
def safe_div(a, b):
    """Return a/b, or None when b is zero."""
    return a / b          # traced on (6, 3) — the b=0 path never runs
```

**If a real run shows equal gains on both categories, something is leaking** — prompt-length effects, the trace acting as generic "try harder" signal, or a formatting artifact. The gain should concentrate in `trace_revealing`.

For contrast, here is a `trace_revealing` trace:

```python
def group_sums(rows):
    out = []
    total = 0                     # rows=[('a', [1, 2]), ('b', [10])]
    for key, vals in rows:        # out=[]
        for v in vals:            # total=0, out=[('a', 3)], out=[('a', 3), ('b', 13)]
            total += v            # [5x, first 3] key='a', vals=[1, 2], total=1
        out.append((key, total))
    return out                    # [2x, first 1] total=13
```

`('b', 13)` instead of `('b', 10)` — the accumulator was never reset. Visible in one line.

## Suite integrity

```bash
python3 verify_suite.py
```

All 18 bugs verified: every buggy version fails its test, every reference fix passes. If this does not hold, the benchmark measures nothing.

## Reading the result

- **Gain concentrated in `trace_revealing`, p < 0.05** — the tool works, for the reason claimed.
- **Equal gains in both categories** — confounded. Suspect prompt length or a generic salience effect. Add a control arm with an *irrelevant* trace before believing anything.
- **No gain anywhere** — the tool does not help this model on this class of bug. That is a real finding and should be published as one.

## Limits

- 18 bugs, all small single functions. Real repairs span files and involve failing imports, fixtures, and state. This is a lower bound on difficulty and an upper bound on how clean the signal will look.
- Bugs were written for this benchmark, so they are not drawn from a natural distribution.
- k=4 gives wide confidence intervals. For a publishable number use k >= 10 and multiple seeds.
- Single model per run. Effects may not transfer across model families.
