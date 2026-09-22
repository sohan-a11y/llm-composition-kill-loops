"""
Persistent trace store. SQLite, no server, no dependencies.

Keyed by (file, function, test, commit) so a trace can be retrieved for the code
as it was when it ran. Stores the rendered text -- the thing actually handed to a
model -- alongside the structured rows, because the render is what matters and
re-rendering from rows would drift as the source changes.
"""

import json
import os
import sqlite3
import subprocess
import time
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file        TEXT NOT NULL,
    function    TEXT,
    test        TEXT,
    commit_sha  TEXT,
    ts          REAL NOT NULL,
    passed      INTEGER,
    exception   TEXT,
    first_line  INTEGER,
    rendered    TEXT NOT NULL,
    rows_json   TEXT NOT NULL,
    stats_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_file     ON traces(file);
CREATE INDEX IF NOT EXISTS ix_function ON traces(function);
CREATE INDEX IF NOT EXISTS ix_test     ON traces(test);
CREATE INDEX IF NOT EXISTS ix_ts       ON traces(ts DESC);
"""


def git_sha(cwd: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=cwd, capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None
    except Exception:
        return None


class TraceStore:
    def __init__(self, path: str = ".tracestore.db"):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ write
    def add(self, *, file: str, rendered: str, rows: List[Dict[str, Any]],
            stats: Dict[str, int], function: Optional[str] = None,
            test: Optional[str] = None, passed: Optional[bool] = None,
            exception: Optional[str] = None, first_line: int = 1,
            commit_sha: Optional[str] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO traces (file,function,test,commit_sha,ts,passed,exception,"
            "first_line,rendered,rows_json,stats_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (file, function, test, commit_sha or git_sha(os.path.dirname(file) or None),
             time.time(), None if passed is None else int(passed), exception,
             first_line, rendered, json.dumps(rows), json.dumps(stats)))
        self.conn.commit()
        return cur.lastrowid

    # ------------------------------------------------------------------- read
    def _fmt(self, r: sqlite3.Row, include_rows=False) -> Dict[str, Any]:
        d = {"id": r["id"], "file": r["file"], "function": r["function"],
             "test": r["test"], "commit": r["commit_sha"], "ts": r["ts"],
             "passed": None if r["passed"] is None else bool(r["passed"]),
             "exception": r["exception"], "rendered": r["rendered"],
             "stats": json.loads(r["stats_json"])}
        if include_rows:
            d["rows"] = json.loads(r["rows_json"])
        return d

    def latest_for(self, *, function: Optional[str] = None,
                   test: Optional[str] = None, file: Optional[str] = None,
                   limit: int = 1) -> List[Dict[str, Any]]:
        where, args = [], []
        if function: where.append("function = ?"); args.append(function)
        if test:     where.append("test = ?");     args.append(test)
        if file:     where.append("file LIKE ?");  args.append(f"%{file}%")
        sql = "SELECT * FROM traces"
        if where: sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        return [self._fmt(r) for r in self.conn.execute(sql, args)]

    def search(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find traces mentioning a symbol in the function name or the trace text."""
        rows = self.conn.execute(
            "SELECT * FROM traces WHERE function LIKE ? OR test LIKE ? "
            "OR rendered LIKE ? ORDER BY ts DESC LIMIT ?",
            (f"%{symbol}%", f"%{symbol}%", f"%{symbol}%", limit))
        return [self._fmt(r) for r in rows]

    def failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM traces WHERE passed = 0 OR exception IS NOT NULL "
            "ORDER BY ts DESC LIMIT ?", (limit,))
        return [self._fmt(r) for r in rows]

    def summary(self) -> Dict[str, Any]:
        c = self.conn.execute(
            "SELECT COUNT(*) n, COUNT(DISTINCT file) files, "
            "COUNT(DISTINCT function) funcs, SUM(passed=0) fails FROM traces"
        ).fetchone()
        return {"traces": c["n"], "files": c["files"],
                "functions": c["funcs"], "failures": c["fails"] or 0,
                "db": os.path.abspath(self.path)}

    def close(self):
        self.conn.close()
