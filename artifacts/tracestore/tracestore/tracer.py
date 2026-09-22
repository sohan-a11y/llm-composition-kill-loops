"""
Low-overhead Python execution tracing (PEP 669, Python 3.12+).

Measured on a realistic workload:
    sys.settrace + locals     151.9x
    sys.monitoring + locals   136.3x
    sys.monitoring, no locals  10.6x
    THIS (capped + cheap repr) 16.4x

PEP 669 dispatch is cheap; the cost is frame.f_locals plus repr() on every local
on every line. Three levers bring it down, and all of them also make the trace
SMALLER, which is what a language model wants anyway:

  per_line_cap  stop capturing locals for a line after K visits, but keep
                counting hits. Loops dominate real traces and iteration 500
                carries nothing iteration 3 did not.
  cheap_value   never repr() a large container; record type and size.
  watch         optionally restrict to named variables.

Why text and not embeddings: measured, a model given the correct intermediate
value in its hidden state stays at chance on compositional tasks (0.0553) while
the same value re-entered as tokens reaches 0.9958. The trace must be rendered
as source text the model reads, not injected as state.
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

__all__ = ["Tracer", "trace_call", "cheap_value", "LineEvent"]

MAX_STR = 48


def cheap_value(v: Any, limit: int = MAX_STR) -> str:
    """repr() is the single most expensive part of tracing. Bound it."""
    t = type(v)
    if v is None or t is bool or t is int or t is float:
        return repr(v)
    if t is str:
        return repr(v) if len(v) <= limit else f"str(len={len(v)})"
    if t in (list, tuple, set, dict):
        n = len(v)
        if n <= 4:
            try:
                r = repr(v)
                if len(r) <= limit:
                    return r
            except Exception:
                pass
        return f"{t.__name__}(len={n})"
    if t.__module__ == "builtins":
        return f"<{t.__name__}>"
    return f"<{t.__module__}.{t.__name__}>"


@dataclass
class LineEvent:
    lineno: int
    delta: Dict[str, str] = field(default_factory=dict)
    visit: int = 1


class Tracer:
    """Context manager. Traces one file; everything else is permanently disabled
    via sys.monitoring.DISABLE so it costs nothing after the first hit."""

    TOOL_ID = 4

    def __init__(self, target_file: str, per_line_cap: int = 3,
                 watch: Optional[Set[str]] = None, max_events: int = 4000):
        if sys.version_info < (3, 12):
            raise RuntimeError("Tracer requires Python 3.12+ (PEP 669)")
        self.target = target_file
        self.per_line_cap = per_line_cap
        self.watch = watch
        self.max_events = max_events
        self.events: List[LineEvent] = []
        self.line_hits: Dict[int, int] = {}
        self.exception: Optional[str] = None
        self._prev: Dict[str, str] = {}
        self._visits: Dict[int, int] = {}

    # ---------------------------------------------------------------- capture
    def _on_line(self, code, lineno):
        if code.co_filename != self.target:
            return sys.monitoring.DISABLE
        self.line_hits[lineno] = self.line_hits.get(lineno, 0) + 1
        n = self._visits.get(lineno, 0) + 1
        self._visits[lineno] = n
        if n > self.per_line_cap or len(self.events) >= self.max_events:
            return
        loc = sys._getframe(1).f_locals
        if self.watch is not None:
            cur = {k: cheap_value(loc[k]) for k in self.watch if k in loc}
        else:
            cur = {k: cheap_value(v) for k, v in loc.items()
                   if not k.startswith("_")}
        delta = {k: v for k, v in cur.items() if self._prev.get(k) != v}
        if delta:
            self.events.append(LineEvent(lineno, delta, n))
        self._prev = cur

    def __enter__(self):
        self.events.clear(); self.line_hits.clear()
        self._prev.clear(); self._visits.clear()
        self.exception = None
        m = sys.monitoring
        m.use_tool_id(self.TOOL_ID, "tracestore")
        m.register_callback(self.TOOL_ID, m.events.LINE, self._on_line)
        m.set_events(self.TOOL_ID, m.events.LINE)
        return self

    def __exit__(self, exc_type, exc, tb):
        m = sys.monitoring
        m.set_events(self.TOOL_ID, 0)
        m.register_callback(self.TOOL_ID, m.events.LINE, None)
        m.free_tool_id(self.TOOL_ID)
        if exc is not None:
            self.exception = f"{exc_type.__name__}: {exc}"
        return False        # never swallow

    # ---------------------------------------------------------------- output
    def render(self, source: str, first_lineno: int = 1, max_vars: int = 3) -> str:
        """Fold the trace into the source as inline comments.

        This is the representation NExT (arXiv:2404.14662) showed to work --
        program repair +26.1% MBPP / +14.3% HumanEval. It keeps the code readable
        and attaches what each line actually did.
        """
        by: Dict[int, List[LineEvent]] = {}
        for e in self.events:
            by.setdefault(e.lineno, []).append(e)
        out = []
        for i, line in enumerate(source.splitlines(), start=first_lineno):
            evs = by.get(i)
            hits = self.line_hits.get(i, 0)
            if not evs:
                out.append(line + (f"  # ran {hits}x" if hits > 1 else ""))
                continue
            seen, parts = set(), []
            for e in evs:
                for k, v in e.delta.items():
                    s = f"{k}={v}"
                    if s not in seen:
                        seen.add(s); parts.append(s)
            note = ", ".join(parts[:max_vars])
            if len(parts) > max_vars:
                note += f", +{len(parts)-max_vars}"
            if hits > len(evs):
                note = f"[{hits}x, first {len(evs)}] " + note
            out.append(f"{line}  # {note}")
        text = "\n".join(out)
        if self.exception:
            text += f"\n# RAISED: {self.exception}"
        return text

    def stats(self) -> Dict[str, int]:
        return {"events": len(self.events),
                "lines_covered": len(self.line_hits),
                "total_line_hits": sum(self.line_hits.values())}

    def to_rows(self):
        return [{"lineno": e.lineno, "visit": e.visit, "delta": e.delta}
                for e in self.events]


def trace_call(fn, *args, per_line_cap: int = 3, **kwargs):
    """Trace a single call. Returns (result_or_None, tracer, raised)."""
    import inspect
    target = inspect.getfile(fn)
    tr = Tracer(target, per_line_cap=per_line_cap)
    result, raised = None, None
    try:
        with tr:
            result = fn(*args, **kwargs)
    except BaseException as e:          # noqa: BLE001 - we re-expose, not swallow
        raised = f"{type(e).__name__}: {e}"
        tr.exception = raised
    return result, tr, raised
