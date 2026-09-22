# tracestore

**Execution traces as context for AI coding assistants.** MCP server + pytest plugin. Zero dependencies.

When an AI assistant debugs your code, it guesses what the variables were. `tracestore` records what they *actually* were and hands that to the model as annotated source.

```python
def running_median(nums):
    out = []                          # nums=[5, 1, 9, 3]
    window = []                       # out=[]
    for n in nums:                    # [5x, first 3] window=[], out=[5], out=[5, 5]
        window.append(n)              # [4x, first 3] n=5, n=1, n=9
        window.sort()                 # [4x, first 3] window=[5], window=[5, 1], window=[1, 5, 9]
        mid = len(window) // 2        # [4x, first 1] window=[1, 5]
        out.append(window[mid])       # [4x, first 2] mid=0, mid=1
    return out                        # out=[5, 5, 5, 5], window=[1, 3, 5, 9]
```

`out=[5, 5, 5, 5]` — the bug is visible without running anything.

## Install

```bash
pip install tracestore
```

## Use it from Claude Code / Cursor / Cline / Continue

Add to your MCP config:

```json
{
  "mcpServers": {
    "tracestore": {
      "command": "tracestore",
      "args": ["serve"],
      "env": {"TRACESTORE_DB": ".tracestore.db"}
    }
  }
}
```

Four tools become available:

| tool | what it does |
|---|---|
| `get_trace` | most recent trace for a function or test |
| `trace_failure` | run a callable now, return its trace and any exception |
| `search_traces` | historical traces mentioning a symbol |
| `trace_summary` | what's in the store |

## Capture traces from your test suite

```bash
pytest --tracestore                          # trace everything
pytest --tracestore --tracestore-failures-only   # only store traces for failures
```

## CLI

```bash
tracestore trace mypkg.utils:parse --args '["input"]'
tracestore get --function parse
tracestore search total
tracestore summary
```

## Overhead

Measured on a realistic mixed workload (loops, branches, string and dict work):

| strategy | overhead |
|---|---|
| `sys.settrace` + locals | 151.9× |
| `sys.monitoring` + locals | 136.3× |
| **tracestore (capped + cheap repr)** | **16–18×** |
| `sys.monitoring`, no locals | 10.6× |

PEP 669 dispatch is cheap. The cost is `frame.f_locals` plus `repr()` on every local on every line. tracestore caps per-line capture (loops dominate real traces; iteration 500 tells you nothing iteration 3 didn't) and never `repr()`s a large container.

**This is for test runs, not production hot paths.** A 30-second suite becomes about 8 minutes. Use `--tracestore-failures-only` to keep the common case fast.

## Why text and not embeddings

This is the design decision the whole tool rests on, and it was measured rather than assumed.

On compositional tasks, a model given the **correct** intermediate value in its hidden state (via an auxiliary head, at 100% supervision) stays at **chance (0.0553)**. The same value re-entered as **tokens** reaches **0.9958**. The auxiliary head reaches 100% accuracy predicting the intermediate by step 1500 while the task never improves — the model knows the value perfectly and cannot use it.

So traces are rendered as source text the model reads, never injected as state. Full study and reproduction: `composition_cliff_FINAL.ipynb`.

## Prior art

Execution-trace supervision is established. [NExT](https://arxiv.org/abs/2404.14662) (Google DeepMind) showed feeding variable states of executed lines to a model as inline comments improves program repair by +26.1% on MBPP and +14.3% on HumanEval. [TraceFixer](https://arxiv.org/abs/2304.12743) and [Self-Debugging](https://arxiv.org/abs/2304.05128) are adjacent.

**tracestore is not a new supervision method.** It is runtime infrastructure: NExT and its successors build offline training corpora, while this is a live, per-developer store served at inference time through MCP. The trace representation follows NExT's because that representation is the one shown to work.

## Limits

- **Python 3.12+ only** (requires PEP 669 `sys.monitoring`).
- **16–18× overhead** — too slow for a large integration suite run on every commit. Test-level sampling is the next lever.
- **One file at a time.** Multi-module tracing is not yet supported.
- **Single-threaded.** `sys.monitoring` callbacks under threading are untested.
- **Values are truncated** — containers over 4 elements become `list(len=N)`. You see shape, not full contents.

MIT.
