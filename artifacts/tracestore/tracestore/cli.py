"""tracestore CLI."""
import argparse, json, os, sys
from .store import TraceStore
from .runner import trace_target

def main(argv=None):
    p = argparse.ArgumentParser(prog="tracestore",
        description="Execution-trace store for AI coding assistants")
    p.add_argument("--db", default=os.environ.get("TRACESTORE_DB", ".tracestore.db"))
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("serve", help="Run the MCP server on stdio")
    sp = sub.add_parser("trace", help="Trace a callable now")
    sp.add_argument("target", help="module:function")
    sp.add_argument("--args", default="[]"); sp.add_argument("--kwargs", default="{}")

    sp = sub.add_parser("get", help="Show the latest trace")
    sp.add_argument("--function"); sp.add_argument("--test"); sp.add_argument("--file")

    sp = sub.add_parser("search", help="Search traces"); sp.add_argument("symbol")
    sub.add_parser("summary", help="Store summary")

    a = p.parse_args(argv)

    if a.cmd == "serve":
        from .server import Server
        Server(a.db).serve(); return 0

    store = TraceStore(a.db)
    if a.cmd == "trace":
        sys.path.insert(0, os.getcwd())
        r = trace_target(a.target, json.loads(a.args), json.loads(a.kwargs))
        store.add(file=r["file"], rendered=r["rendered"], rows=r["rows"],
                  stats=r["stats"], function=r["function"],
                  passed=r["exception"] is None, exception=r["exception"],
                  first_line=r["first_line"])
        print(r["rendered"])
        if r["exception"]: print(f"\nRAISED: {r['exception']}")
    elif a.cmd == "get":
        for h in store.latest_for(function=a.function, test=a.test, file=a.file):
            print(f"# {h['file']} :: {h['function']}\n{h['rendered']}")
    elif a.cmd == "search":
        for h in store.search(a.symbol):
            print(f"# {h['function'] or h['test']}\n{h['rendered']}\n")
    elif a.cmd == "summary":
        print(json.dumps(store.summary(), indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())
