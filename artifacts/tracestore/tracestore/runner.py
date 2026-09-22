"""Resolve and execute a target under tracing."""
import importlib, inspect, json
from typing import Any, Dict, List
from .tracer import Tracer

def resolve(target: str):
    """'pkg.mod:func' -> callable"""
    if ":" not in target:
        raise ValueError("target must be 'module:function'")
    mod_name, fn_name = target.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = mod
    for part in fn_name.split("."):
        fn = getattr(fn, part)
    if not callable(fn):
        raise TypeError(f"{target} is not callable")
    return fn

def trace_target(target: str, args: List[Any], kwargs: Dict[str, Any],
                 per_line_cap: int = 3) -> Dict[str, Any]:
    fn = resolve(target)
    file = inspect.getfile(fn)
    src_lines, first = inspect.getsourcelines(fn)
    tr = Tracer(file, per_line_cap=per_line_cap)
    exc = None
    try:
        with tr:
            fn(*args, **kwargs)
    except BaseException as e:            # noqa: BLE001
        exc = f"{type(e).__name__}: {e}"
        tr.exception = exc
    return {"file": file, "function": getattr(fn, "__qualname__", target),
            "first_line": first, "rendered": tr.render("".join(src_lines), first),
            "rows": tr.to_rows(), "stats": tr.stats(), "exception": exc}
