"""
Fix-rate benchmark: does an execution trace improve an LLM's ability to repair
broken code?

Design, and the reasons for each choice:

  PAIRED. Every bug is attempted under both arms with an otherwise IDENTICAL
  prompt. The only difference is the presence of the trace block. Anything else
  differing between arms would confound the result.

  MULTIPLE SAMPLES. k attempts per (bug, arm) at non-zero temperature, scored as
  fix rate and pass@k. One sample per cell would be dominated by sampling noise.

  TWO BUG CATEGORIES. trace_revealing vs trace_neutral. If traces help equally on
  both, the experiment is broken -- the trace cannot explain a bug whose failing
  path it never executed. The neutral category is the internal control.

  ORDER RANDOMISED. Arm order is shuffled per attempt so any drift in the API
  over the run does not land systematically on one arm.

  VERIFIED IN A SUBPROCESS. A fix counts only if the real tests pass in a clean
  interpreter with a timeout. No self-report, no judge model.

Run:
    ANTHROPIC_API_KEY=sk-... python3 run_bench.py --model claude-sonnet-4-6 --k 4
    python3 run_bench.py --mock          # verify the pipeline with no API calls
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bugs import BUGS
from tracestore.tracer import Tracer

API_URL = "https://api.anthropic.com/v1/messages"

PROMPT = """The function below has a bug. Its test is failing.

```python
{source}
```

Failing test:
```python
{test}
```
{trace_block}
Rewrite the function so the test passes. Reply with ONLY the corrected function
in a single ```python code block. No explanation."""

TRACE_BLOCK = """
Here is an execution trace of the buggy function on the failing input. Each line
is annotated with the values variables ACTUALLY held when that line ran:

```python
{trace}
```
"""


# ------------------------------------------------------------------ tracing
def capture_trace(bug: Dict[str, Any]) -> str:
    """Run the buggy function on its failing input and render the trace."""
    fn_name, args, kwargs = bug["call"]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(bug["buggy"]); path = f.name
    try:
        ns: Dict[str, Any] = {"__file__": path}
        code = compile(bug["buggy"], path, "exec")
        exec(code, ns)
        fn = ns[fn_name]
        tr = Tracer(path, per_line_cap=3)
        try:
            with tr:
                fn(*args, **kwargs)
        except BaseException as e:            # noqa: BLE001
            tr.exception = f"{type(e).__name__}: {e}"
        return tr.render(bug["buggy"].strip(), first_lineno=1)
    finally:
        os.unlink(path)


# ------------------------------------------------------------------ scoring
def _test_names(test_src: str) -> List[str]:
    return [l.split("(")[0].replace("def ", "").strip()
            for l in test_src.splitlines() if l.startswith("def test")]


def verify(candidate: str, test_src: str, timeout: int = 10) -> bool:
    """A fix counts only if the real tests pass in a clean subprocess."""
    prog = (candidate + "\n" + test_src + "\n" +
            "\n".join(f"{n}()" for n in _test_names(test_src)) +
            "\nprint('ALLPASS')\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
        return "ALLPASS" in r.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except OSError: pass


def extract_code(text: str) -> str:
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m[0] if m else text).strip()


# -------------------------------------------------------------------- model
def anthropic_call(model: str, prompt: str, temperature: float,
                   max_tokens: int = 1024, retries: int = 3) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    body = json.dumps({"model": model, "max_tokens": max_tokens,
                       "temperature": temperature,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "content-type": "application/json", "x-api-key": key,
        "anthropic-version": "2023-06-01"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            return "".join(b.get("text", "") for b in data.get("content", []))
        except Exception as e:                # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"API failed after {retries} attempts: {last}")


def mock_call(model: str, prompt: str, temperature: float, **kw) -> str:
    """Pipeline verification with no API. Deliberately imperfect, and slightly
    better when a trace is present, so the whole scoring path is exercised."""
    has_trace = "execution trace" in prompt
    m = re.search(r"```python\n(.*?)```", prompt, re.S)
    src = m.group(1) if m else ""
    name = re.search(r"def (\w+)", src)
    name = name.group(1) if name else "f"
    for b in BUGS:
        if f"def {name}(" in b["buggy"]:
            p = 0.75 if has_trace else 0.35
            if random.random() < p:
                return f"```python\n{b['fixed'].strip()}\n```"
            return f"```python\n{b['buggy'].strip()}\n```"
    return "```python\npass\n```"


# ---------------------------------------------------------------- benchmark
def run(call_model: Callable, model: str, k: int, temperature: float,
        bug_filter: Optional[str], out_path: str, verbose: bool = True):
    bugs = [b for b in BUGS if not bug_filter or b["cat"] == bug_filter]
    traces = {b["id"]: capture_trace(b) for b in bugs}
    results: List[Dict[str, Any]] = []

    for bi, b in enumerate(bugs, 1):
        for attempt in range(k):
            arms = ["with_trace", "without_trace"]
            random.shuffle(arms)                      # no systematic order effect
            for arm in arms:
                tb = TRACE_BLOCK.format(trace=traces[b["id"]]) if arm == "with_trace" else ""
                prompt = PROMPT.format(source=b["buggy"].strip(),
                                       test=b["test"].strip(), trace_block=tb)
                t0 = time.time()
                try:
                    reply = call_model(model, prompt, temperature)
                    err = None
                except Exception as e:                # noqa: BLE001
                    reply, err = "", str(e)[:200]
                passed = verify(extract_code(reply), b["test"]) if reply else False
                results.append(dict(bug=b["id"], cat=b["cat"], arm=arm,
                                    attempt=attempt, passed=passed, error=err,
                                    prompt_chars=len(prompt),
                                    sec=round(time.time() - t0, 2)))
                json.dump(results, open(out_path, "w"), indent=1)
        if verbose:
            done = [r for r in results if r["bug"] == b["id"]]
            wt = sum(r["passed"] for r in done if r["arm"] == "with_trace")
            wo = sum(r["passed"] for r in done if r["arm"] == "without_trace")
            print(f"  [{bi}/{len(bugs)}] {b['id']:<26} {b['cat']:<16} "
                  f"trace {wt}/{k}   no-trace {wo}/{k}", flush=True)
    return results


def report(results: List[Dict[str, Any]]):
    def rate(rs): return (sum(r["passed"] for r in rs) / len(rs)) if rs else float("nan")
    print("\n" + "=" * 68)
    print("FIX RATE")
    print("=" * 68)
    print(f"{'category':<20}{'with trace':>14}{'without':>12}{'delta':>12}{'n/arm':>8}")
    print("-" * 68)
    for cat in ["trace_revealing", "trace_neutral"]:
        rs = [r for r in results if r["cat"] == cat]
        if not rs: continue
        wt = [r for r in rs if r["arm"] == "with_trace"]
        wo = [r for r in rs if r["arm"] == "without_trace"]
        print(f"{cat:<20}{rate(wt):>13.1%}{rate(wo):>12.1%}"
              f"{rate(wt)-rate(wo):>+12.1%}{len(wt):>8}")
    wt = [r for r in results if r["arm"] == "with_trace"]
    wo = [r for r in results if r["arm"] == "without_trace"]
    print("-" * 68)
    print(f"{'OVERALL':<20}{rate(wt):>13.1%}{rate(wo):>12.1%}"
          f"{rate(wt)-rate(wo):>+12.1%}{len(wt):>8}")

    # per-bug pass@k, and the sign test over bugs
    print("\nPER-BUG (pass@k)")
    print(f"{'bug':<28}{'cat':<18}{'trace':>8}{'no-trace':>10}")
    print("-" * 68)
    wins = losses = ties = 0
    for bug in sorted({r["bug"] for r in results}):
        rs = [r for r in results if r["bug"] == bug]
        a = any(r["passed"] for r in rs if r["arm"] == "with_trace")
        c = any(r["passed"] for r in rs if r["arm"] == "without_trace")
        if a and not c: wins += 1
        elif c and not a: losses += 1
        else: ties += 1
        print(f"{bug:<28}{rs[0]['cat']:<18}{('yes' if a else 'no'):>8}"
              f"{('yes' if c else 'no'):>10}")
    print("-" * 68)
    print(f"pass@k: trace-only wins {wins}, no-trace-only wins {losses}, ties {ties}")
    n = wins + losses
    if n:
        # exact two-sided sign test
        from math import comb
        p = sum(comb(n, i) for i in range(min(wins, losses) + 1)) / (2 ** n) * 2
        print(f"sign test over discordant pairs: n={n}, two-sided p={min(p,1.0):.4f}")
        if min(p, 1.0) > 0.05:
            print("  NOT significant at 0.05 — do not claim an effect from this run.")
    errs = [r for r in results if r.get("error")]
    if errs:
        print(f"\n{len(errs)} API errors (counted as failures): "
              f"{errs[0]['error'][:120]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--k", type=int, default=4, help="attempts per bug per arm")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--cat", default=None,
                    choices=["trace_revealing", "trace_neutral"])
    ap.add_argument("--mock", action="store_true", help="no API; verify pipeline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench_results.json")
    a = ap.parse_args()

    random.seed(a.seed)
    fn = mock_call if a.mock else anthropic_call
    if a.mock:
        print("MOCK MODE — no API calls. Verifies the pipeline, not the tool.\n")
    print(f"model={a.model}  k={a.k}  temp={a.temperature}  "
          f"bugs={len([b for b in BUGS if not a.cat or b['cat']==a.cat])}\n")
    res = run(fn, a.model, a.k, a.temperature, a.cat, a.out)
    report(res)
    print(f"\nwrote {a.out}")
