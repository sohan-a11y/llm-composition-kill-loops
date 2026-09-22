"""
pytest plugin: capture execution traces automatically during test runs.

Usage:   pytest --tracestore
         pytest --tracestore --tracestore-failures-only

Overhead is ~16x on traced modules, so this is meant for test runs, not
production. A 30s suite becomes ~8 minutes; run it in CI or overnight and you
get a trace store keyed to your codebase.

--tracestore-failures-only re-runs only failing tests under tracing, which keeps
the common case fast and captures traces exactly where they are worth having.
"""
import os
from .tracer import Tracer
from .store import TraceStore

def pytest_addoption(parser):
    g = parser.getgroup("tracestore")
    g.addoption("--tracestore", action="store_true", default=False,
                help="Capture execution traces into the trace store")
    g.addoption("--tracestore-db", action="store", default=".tracestore.db",
                help="Path to the trace store database")
    g.addoption("--tracestore-failures-only", action="store_true", default=False,
                help="Only store traces for tests that fail")
    g.addoption("--tracestore-target", action="store", default=None,
                help="Only trace this file (default: the test file itself)")

def pytest_configure(config):
    if config.getoption("--tracestore"):
        config._tracestore = TraceStore(config.getoption("--tracestore-db"))
        config.pluginmanager.register(TraceCollector(config), "tracestore-collector")

class TraceCollector:
    def __init__(self, config):
        self.config = config
        self.store = config._tracestore
        self.failures_only = config.getoption("--tracestore-failures-only")
        self.target_override = config.getoption("--tracestore-target")
        self._pending = {}

    def pytest_runtest_call(self, item):
        target = self.target_override or str(item.fspath)
        tr = Tracer(target)
        self._pending[item.nodeid] = (tr, target)
        tr.__enter__()
        try:
            item.runtest()
        finally:
            tr.__exit__(None, None, None)
        return True      # we ran it

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return
        got = self._pending.pop(report.nodeid, None)
        if got is None:
            return
        tr, target = got
        passed = report.passed
        if self.failures_only and passed:
            return
        try:
            with open(target, "r", encoding="utf-8") as f:
                source = f.read()
        except OSError:
            return
        self.store.add(file=target, rendered=tr.render(source, 1),
                       rows=tr.to_rows(), stats=tr.stats(),
                       test=report.nodeid, passed=passed,
                       exception=(str(report.longrepr)[:2000] if not passed else None))

    def pytest_sessionfinish(self):
        s = self.store.summary()
        print(f"\n[tracestore] {s['traces']} traces across {s['files']} files "
              f"-> {s['db']}")
