"""Execution accuracy and the breakdowns the results table needs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Iterable

from text2sql.sql_exec import ExecStatus

DIFFICULTY_ORDER = ("simple", "moderate", "challenging", "unknown")


@dataclass(slots=True)
class Prediction:
    """One scored prediction, as written to `results/*/predictions.jsonl`."""

    question_id: str
    db_id: str
    question: str
    gold_sql: str
    pred_sql: str
    correct: bool
    difficulty: str = "unknown"
    source: str = "bird"
    pred_status: str = ExecStatus.OK.value
    gold_status: str = ExecStatus.OK.value
    error: str = ""
    raw_output: str = ""
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


def summarise(predictions: Iterable[Prediction]) -> dict[str, Any]:
    """Headline EX plus per-difficulty, per-source and failure-mode breakdowns."""
    preds = list(predictions)
    total = len(preds)
    correct = sum(p.correct for p in preds)

    by_difficulty: dict[str, dict[str, Any]] = {}
    buckets: dict[str, list[Prediction]] = defaultdict(list)
    for p in preds:
        buckets[p.difficulty or "unknown"].append(p)
    for name in DIFFICULTY_ORDER:
        if name in buckets:
            rows = buckets[name]
            by_difficulty[name] = {
                "n": len(rows),
                "correct": sum(r.correct for r in rows),
                "ex": _pct(sum(r.correct for r in rows), len(rows)),
            }

    by_source: dict[str, dict[str, Any]] = {}
    source_buckets: dict[str, list[Prediction]] = defaultdict(list)
    for p in preds:
        source_buckets[p.source].append(p)
    for name, rows in sorted(source_buckets.items()):
        by_source[name] = {
            "n": len(rows),
            "correct": sum(r.correct for r in rows),
            "ex": _pct(sum(r.correct for r in rows), len(rows)),
        }

    executable = sum(1 for p in preds if p.pred_status == ExecStatus.OK.value)
    non_empty = sum(1 for p in preds if p.pred_sql.strip())
    latencies = [p.latency_ms for p in preds if p.latency_ms is not None]

    summary: dict[str, Any] = {
        "n": total,
        "correct": correct,
        "ex": _pct(correct, total),
        "valid_sql_rate": _pct(executable, total),
        "parsed_rate": _pct(non_empty, total),
        "by_difficulty": by_difficulty,
        "by_source": by_source,
        "pred_status_counts": dict(Counter(p.pred_status for p in preds)),
        "gold_status_counts": dict(Counter(p.gold_status for p in preds)),
    }
    if latencies:
        summary["latency_ms"] = {
            "mean": round(mean(latencies), 1),
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
        }
    in_tok = [p.input_tokens for p in preds if p.input_tokens is not None]
    out_tok = [p.output_tokens for p in preds if p.output_tokens is not None]
    if in_tok or out_tok:
        summary["tokens"] = {
            "input_total": sum(in_tok),
            "output_total": sum(out_tok),
            "input_mean": round(mean(in_tok), 1) if in_tok else None,
            "output_mean": round(mean(out_tok), 1) if out_tok else None,
        }
    return summary


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile; `q` in [0, 100]."""
    if not values:
        raise ValueError("percentile of an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (q / 100.0) * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def majority_vote(
    candidates: list[tuple[str, tuple[tuple[Any, ...], ...] | None]],
) -> str:
    """Self-consistency: pick the query whose *result set* is most common.

    Voting on result sets rather than query strings is the point -- two correct
    queries rarely look alike but always return the same rows.
    """
    groups: dict[Any, list[str]] = defaultdict(list)
    for sql, rows in candidates:
        if rows is None:
            continue
        key = frozenset(rows)
        groups[key].append(sql)
    if not groups:
        return candidates[0][0] if candidates else ""
    best = max(groups.values(), key=len)
    # Shortest query in the winning group: least likely to carry dead weight.
    return min(best, key=len)
