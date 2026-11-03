#!/usr/bin/env python3
"""Sort the failures into categories and draw the chart for the write-up.

    python scripts/error_analysis.py --run results/sft-grpo --sample 100

Categories follow the plan: schema linking (wrong table or column), joins,
aggregation, filter values, ordering/limits, plus the mechanical failures
(nothing parsed, query did not run, timed out). The point is to be able to say
*which* 40% is wrong, not just that 40% is.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from text2sql.data import read_jsonl

CATEGORIES = [
    "no_sql_parsed",
    "sql_error_unknown_column",
    "sql_error_unknown_table",
    "sql_error_syntax",
    "sql_error_other",
    "timeout",
    "wrong_schema_linking",
    "wrong_join",
    "wrong_aggregation",
    "wrong_filter_value",
    "wrong_ordering_or_limit",
    "wrong_projection",
    "wrong_other",
    "gold_unexecutable",
]

_AGGS = ("count", "sum", "avg", "min", "max")


def _ident_set(sql: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql)}


def _count(sql: str, pattern: str) -> int:
    return len(re.findall(pattern, sql, re.IGNORECASE))


def _literals(sql: str) -> set[str]:
    return {m.lower() for m in re.findall(r"'([^']*)'", sql)}


def categorise(row: dict[str, Any]) -> str:
    """One label per failing prediction."""
    if row.get("gold_status") != "ok":
        return "gold_unexecutable"

    pred_sql = (row.get("pred_sql") or "").strip()
    if not pred_sql:
        return "no_sql_parsed"

    status = row.get("pred_status")
    error = (row.get("error") or "").lower()
    if status == "timeout":
        return "timeout"
    if status in {"error", "rejected"}:
        if "no such column" in error or "ambiguous column" in error:
            return "sql_error_unknown_column"
        if "no such table" in error:
            return "sql_error_unknown_table"
        if "syntax error" in error or "incomplete input" in error:
            return "sql_error_syntax"
        return "sql_error_other"

    # Ran fine, wrong rows. Diff the two queries structurally.
    gold_sql = row.get("gold_sql") or ""
    gold_tokens, pred_tokens = _ident_set(gold_sql), _ident_set(pred_sql)

    gold_joins = _count(gold_sql, r"\bjoin\b")
    pred_joins = _count(pred_sql, r"\bjoin\b")
    if gold_joins != pred_joins:
        return "wrong_join"

    if gold_tokens - pred_tokens and len(gold_tokens - pred_tokens) > 1:
        return "wrong_schema_linking"

    gold_aggs = Counter(a for a in _AGGS if _count(gold_sql, rf"\b{a}\s*\("))
    pred_aggs = Counter(a for a in _AGGS if _count(pred_sql, rf"\b{a}\s*\("))
    if gold_aggs != pred_aggs or _count(gold_sql, r"\bgroup\s+by\b") != _count(pred_sql, r"\bgroup\s+by\b"):
        return "wrong_aggregation"

    if _literals(gold_sql) != _literals(pred_sql):
        return "wrong_filter_value"

    if (_count(gold_sql, r"\border\s+by\b") != _count(pred_sql, r"\border\s+by\b")
            or _count(gold_sql, r"\blimit\b") != _count(pred_sql, r"\blimit\b")):
        return "wrong_ordering_or_limit"

    gold_select = re.split(r"\bfrom\b", gold_sql, maxsplit=1, flags=re.IGNORECASE)[0]
    pred_select = re.split(r"\bfrom\b", pred_sql, maxsplit=1, flags=re.IGNORECASE)[0]
    if gold_select.count(",") != pred_select.count(","):
        return "wrong_projection"

    return "wrong_other"


def write_chart(counts: Counter, path: Path, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not installed; no chart written "
              '(pip install -e ".[viz]")')
        return False

    items = [(name, counts[name]) for name in CATEGORIES if counts.get(name)]
    items.sort(key=lambda kv: kv[1])
    labels = [name.replace("_", " ") for name, _ in items]
    values = [count for _, count in items]

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(items) + 1.6))
    ax.barh(labels, values, color="#4C72B0")
    ax.set_xlabel("failing questions")
    ax.set_title(title)
    for y, value in enumerate(values):
        ax.text(value, y, f" {value}", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--sample", type=int, default=100,
                        help="how many failures to categorise (0 = all)")
    parser.add_argument("--examples-per-category", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    pred_path = args.run / "predictions.jsonl"
    if not pred_path.exists():
        print(f"[fail] {pred_path} not found; run scripts/evaluate.py first")
        return 1

    rows = list(read_jsonl(pred_path))
    failures = [r for r in rows if not r.get("correct")]
    print(f"[load ] {len(rows)} predictions, {len(failures)} failures")
    if not failures:
        print("[done ] nothing to analyse")
        return 0

    rng = random.Random(args.seed)
    if args.sample and len(failures) > args.sample:
        failures = rng.sample(failures, args.sample)
        print(f"[samp ] categorising a random {args.sample}")

    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = {}
    for row in failures:
        label = categorise(row)
        counts[label] += 1
        bucket = examples.setdefault(label, [])
        if len(bucket) < args.examples_per_category:
            bucket.append({
                "question_id": row.get("question_id", ""),
                "db_id": row.get("db_id", ""),
                "question": row.get("question", ""),
                "gold_sql": row.get("gold_sql", ""),
                "pred_sql": row.get("pred_sql", ""),
                "error": row.get("error", ""),
            })

    total = sum(counts.values())
    report = {
        "run": args.run.name,
        "n_predictions": len(rows),
        "n_failures_total": sum(1 for r in rows if not r.get("correct")),
        "n_categorised": total,
        "counts": {name: counts[name] for name in CATEGORIES if counts.get(name)},
        "share_pct": {
            name: round(100 * counts[name] / total, 1)
            for name in CATEGORIES if counts.get(name)
        },
        "examples": examples,
    }
    (args.run / "error_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["| Failure mode | n | share of failures |", "| --- | ---: | ---: |"]
    for name, count in counts.most_common():
        lines.append(f"| {name.replace('_', ' ')} | {count} | {100*count/total:.1f}% |")
    table = "\n".join(lines)
    (args.run / "error_analysis.md").write_text(table + "\n", encoding="utf-8")

    chart = args.run / "error_analysis.png"
    write_chart(counts, chart, f"{args.run.name}: failure modes (n={total})")

    print()
    print(table)
    print()
    print(f"[write] {args.run / 'error_analysis.json'}")
    print(f"[write] {args.run / 'error_analysis.md'}")
    if chart.exists():
        print(f"[write] {chart}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
