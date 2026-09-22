"""
MCP server exposing the trace store to any MCP client (Claude Code, Cursor,
Cline, Continue, ...).

Implemented directly against the MCP wire protocol -- JSON-RPC 2.0 over stdio --
with no dependencies. A tool a developer installs should not drag a tree of
packages into their environment.

Tools exposed:
    get_trace      most recent execution trace for a function or test
    trace_failure  run a test/callable NOW and return its trace, incl. exception
    search_traces  historical traces mentioning a symbol
    trace_summary  what is in the store

Protocol: https://modelcontextprotocol.io  (2024-11-05 revision)
"""

import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from .store import TraceStore
from .runner import trace_target

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "tracestore", "version": "0.1.0"}

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_trace",
        "description": (
            "Return the most recent execution trace for a function or test, as "
            "source annotated with the ACTUAL variable values each line produced. "
            "Use this before reasoning about why code behaves a certain way -- it "
            "replaces guessing at intermediate state with observing it."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "function": {"type": "string", "description": "Function name"},
                "test": {"type": "string", "description": "Test name"},
                "file": {"type": "string", "description": "Path fragment"},
            },
        },
    },
    {
        "name": "trace_failure",
        "description": (
            "Execute a Python callable or test NOW under tracing and return the "
            "annotated source plus any exception. Use when no stored trace exists "
            "or the code changed since the last run."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string",
                           "description": "module:function, e.g. mypkg.utils:parse"},
                "args_json": {"type": "string",
                              "description": "JSON list of positional args"},
                "kwargs_json": {"type": "string",
                                "description": "JSON object of keyword args"},
            },
            "required": ["target"],
        },
    },
    {
        "name": "search_traces",
        "description": "Search stored traces for a symbol (function, variable, or text).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "trace_summary",
        "description": "Summarise the trace store: counts, files, recent failures.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class Server:
    def __init__(self, db_path: str):
        self.store = TraceStore(db_path)

    # ------------------------------------------------------------- tool impls
    def _render_hits(self, hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return ("No stored trace found. Run the test suite with the pytest "
                    "plugin (`pytest --tracestore`) or call trace_failure to "
                    "capture one now.")
        out = []
        for h in hits:
            head = f"# {h['file']}"
            if h["function"]: head += f" :: {h['function']}"
            if h["test"]:     head += f"  (test: {h['test']})"
            if h["commit"]:   head += f"  @{h['commit']}"
            if h["passed"] is False: head += "  [FAILED]"
            out.append(head + "\n" + h["rendered"])
        return "\n\n".join(out)

    def call_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name == "get_trace":
            hits = self.store.latest_for(
                function=args.get("function"), test=args.get("test"),
                file=args.get("file"), limit=1)
            return self._render_hits(hits)

        if name == "trace_failure":
            pos = json.loads(args.get("args_json") or "[]")
            kw = json.loads(args.get("kwargs_json") or "{}")
            res = trace_target(args["target"], pos, kw)
            tid = self.store.add(
                file=res["file"], rendered=res["rendered"], rows=res["rows"],
                stats=res["stats"], function=res["function"],
                passed=res["exception"] is None, exception=res["exception"],
                first_line=res["first_line"])
            head = f"# {res['file']} :: {res['function']}  (trace #{tid})"
            if res["exception"]:
                head += f"\n# RAISED {res['exception']}"
            return head + "\n" + res["rendered"]

        if name == "search_traces":
            hits = self.store.search(args["symbol"], int(args.get("limit", 5)))
            return self._render_hits(hits)

        if name == "trace_summary":
            s = self.store.summary()
            fails = self.store.failures(5)
            txt = (f"traces={s['traces']} files={s['files']} "
                   f"functions={s['functions']} failures={s['failures']}\n"
                   f"db: {s['db']}")
            if fails:
                txt += "\n\nrecent failures:\n" + "\n".join(
                    f"  {f['function'] or f['test']}: {f['exception']}" for f in fails)
            return txt

        raise ValueError(f"unknown tool: {name}")

    # ----------------------------------------------------------------- JSON-RPC
    def handle(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        mid, method = msg.get("id"), msg.get("method")

        if method == "initialize":
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO}}

        if method in ("notifications/initialized", "initialized"):
            return None                      # notification: no reply

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}

        if method == "tools/call":
            p = msg.get("params", {})
            try:
                text = self.call_tool(p.get("name"), p.get("arguments") or {})
                return {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": text}],
                    "isError": False}}
            except Exception as e:
                return {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": f"{type(e).__name__}: {e}\n"
                                         f"{traceback.format_exc(limit=3)}"}],
                    "isError": True}}

        if method == "ping":
            return {"jsonrpc": "2.0", "id": mid, "result": {}}

        if mid is None:
            return None                      # unknown notification: ignore
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"method not found: {method}"}}

    def serve(self, stdin=None, stdout=None):
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            reply = self.handle(msg)
            if reply is not None:
                stdout.write(json.dumps(reply) + "\n")
                stdout.flush()


def main():
    db = os.environ.get("TRACESTORE_DB", ".tracestore.db")
    Server(db).serve()


if __name__ == "__main__":
    main()
