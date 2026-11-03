"""Run SQL against the benchmark SQLite files and compare result sets.

Two things matter here and nowhere else in the repo:

1.  Execution must never hang. BIRD contains gold queries that take seconds and
    model output that can cartesian-join a database into oblivion, so every
    query runs under a wall-clock deadline enforced by SQLite's progress
    handler (checked every `_PROGRESS_OPS` virtual-machine instructions).
2.  Comparison must match the official BIRD metric. BIRD's evaluator compares
    `set(predicted_rows) == set(gold_rows)`, so that is the default here.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

_PROGRESS_OPS = 1000
_FLOAT_TOL = 1e-6

# Anything that writes is rejected before it reaches SQLite. The connection is
# also opened read-only, so this is belt and braces -- but a clear error beats
# a confusing "attempt to write a readonly database".
_WRITE_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|"
    r"pragma|vacuum|reindex|begin|commit|rollback)\b",
    re.IGNORECASE,
)


class ExecStatus(str, Enum):
    OK = "ok"
    TIMEOUT = "timeout"
    ERROR = "error"
    REJECTED = "rejected"
    EMPTY = "empty"  # no SQL could be parsed out of the model's answer


@dataclass(slots=True)
class ExecResult:
    status: ExecStatus
    rows: tuple[tuple[Any, ...], ...] = ()
    error: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status is ExecStatus.OK


class ComparisonMode(str, Enum):
    BIRD = "bird"          # set equality, the official BIRD metric
    MULTISET = "multiset"  # row counts matter, order does not
    ORDERED = "ordered"    # order matters too


def is_read_only(sql: str) -> bool:
    """True when `sql` is a single SELECT/WITH statement with no write verbs."""
    stripped = _strip_sql_comments(sql).strip().rstrip(";").strip()
    if not stripped:
        return False
    # Reject stacked statements: "SELECT 1; DROP TABLE t".
    if ";" in stripped:
        return False
    return _WRITE_RE.search(stripped) is None


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return sql


def execute_sql(
    db_path: str | Path,
    sql: str,
    timeout: float = 30.0,
    max_rows: int | None = 100_000,
) -> ExecResult:
    """Execute `sql` read-only against `db_path` under a wall-clock deadline."""
    if not sql or not sql.strip():
        return ExecResult(ExecStatus.EMPTY, error="no SQL to execute")
    if not is_read_only(sql):
        return ExecResult(ExecStatus.REJECTED, error="non-SELECT or stacked statement")

    db_path = Path(db_path)
    if not db_path.exists():
        return ExecResult(ExecStatus.ERROR, error=f"database not found: {db_path}")

    started = time.perf_counter()
    deadline = started + timeout
    conn: sqlite3.Connection | None = None
    try:
        uri = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=timeout, check_same_thread=False)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, _PROGRESS_OPS)
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows) if max_rows else cur.fetchall()
        return ExecResult(
            ExecStatus.OK,
            rows=tuple(tuple(r) for r in rows),
            seconds=time.perf_counter() - started,
        )
    except sqlite3.OperationalError as exc:
        elapsed = time.perf_counter() - started
        # The progress handler aborts with "interrupted"; everything else is a
        # genuine SQL error (bad column, syntax, ...).
        if "interrupt" in str(exc).lower() or elapsed >= timeout:
            return ExecResult(ExecStatus.TIMEOUT, error="execution timed out", seconds=elapsed)
        return ExecResult(ExecStatus.ERROR, error=str(exc), seconds=elapsed)
    except Exception as exc:  # noqa: BLE001 - sqlite raises a wide family
        return ExecResult(
            ExecStatus.ERROR, error=f"{type(exc).__name__}: {exc}",
            seconds=time.perf_counter() - started,
        )
    finally:
        if conn is not None:
            conn.close()


def _norm_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isnan(f):
            return "nan"
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        # Round so 0.1+0.2 and 0.3 compare equal.
        return round(f / _FLOAT_TOL) * _FLOAT_TOL
    if isinstance(v, bytes):
        return v
    return str(v).strip()


def _norm_rows(rows: Iterable[Sequence[Any]]) -> list[tuple[Any, ...]]:
    return [tuple(_norm_value(v) for v in row) for row in rows]


def has_order_by(sql: str) -> bool:
    return re.search(r"\border\s+by\b", _strip_sql_comments(sql), re.IGNORECASE) is not None


def compare_results(
    pred_rows: Iterable[Sequence[Any]],
    gold_rows: Iterable[Sequence[Any]],
    mode: ComparisonMode | str = ComparisonMode.BIRD,
) -> bool:
    """Compare two result sets under the chosen semantics."""
    mode = ComparisonMode(mode)
    pred = _norm_rows(pred_rows)
    gold = _norm_rows(gold_rows)

    if mode is ComparisonMode.ORDERED:
        return pred == gold
    if mode is ComparisonMode.MULTISET:
        return Counter(pred) == Counter(gold)
    return set(pred) == set(gold)


@dataclass(slots=True)
class Comparison:
    correct: bool
    pred: ExecResult
    gold: ExecResult


def execution_match(
    db_path: str | Path,
    pred_sql: str,
    gold_sql: str,
    timeout: float = 30.0,
    mode: ComparisonMode | str = ComparisonMode.BIRD,
    order_sensitive_when_gold_orders: bool = False,
) -> Comparison:
    """Run both queries and say whether the predicted one reproduces the gold result."""
    gold = execute_sql(db_path, gold_sql, timeout=timeout)
    pred = execute_sql(db_path, pred_sql, timeout=timeout)
    if not (gold.ok and pred.ok):
        return Comparison(False, pred, gold)

    effective = mode
    if order_sensitive_when_gold_orders and has_order_by(gold_sql):
        effective = ComparisonMode.ORDERED
    return Comparison(compare_results(pred.rows, gold.rows, effective), pred, gold)


# --- hard isolation ----------------------------------------------------------
# The progress handler covers runaway queries, but a malformed query can also
# make SQLite allocate until the process dies. During GRPO rollouts -- thousands
# of untrusted queries per step -- run them in a child process instead.

def _exec_in_child(args: tuple[str, str, float]) -> ExecResult:
    db_path, sql, timeout = args
    return execute_sql(db_path, sql, timeout=timeout)


def execute_sql_isolated(
    db_path: str | Path,
    sql: str,
    timeout: float = 30.0,
    pool: ProcessPoolExecutor | None = None,
) -> ExecResult:
    """Same as `execute_sql`, but in a child process so a crash cannot take us down."""
    owns_pool = pool is None
    pool = pool or ProcessPoolExecutor(max_workers=1)
    try:
        future = pool.submit(_exec_in_child, (str(db_path), sql, timeout))
        try:
            return future.result(timeout=timeout + 5.0)
        except FutureTimeout:
            future.cancel()
            return ExecResult(ExecStatus.TIMEOUT, error="child process timed out")
        except Exception as exc:  # noqa: BLE001 - child died
            return ExecResult(ExecStatus.ERROR, error=f"child failed: {exc}")
    finally:
        if owns_pool:
            pool.shutdown(wait=False, cancel_futures=True)
