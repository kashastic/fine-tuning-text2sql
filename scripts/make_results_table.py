#!/usr/bin/env python3
"""Collect every results/*/summary.json into the README's results table.

    python scripts/make_results_table.py --results-dir results --out results/RESULTS.md

Columns match the plan's table: EX overall and by difficulty, Spider EX where a
Spider run exists under the same label, p95 latency and cost per 1,000 queries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

HEADERS = [
    "Run", "Model", "Method", "BIRD dev EX (%)",
    "Simple / Moderate / Challenging (%)", "Spider dev EX (%)",
    "p95 latency (ms)", "Cost per 1k queries (EUR)",
]


def method_label(summary: dict[str, Any]) -> str:
    meta = summary.get("meta") or {}
    parts: list[str] = []
    backend = summary.get("backend")
    if backend in ("anthropic", "openai-api"):
        parts.append("Prompted")
    elif meta.get("lora") or "sft" in summary.get("run", "").lower():
        parts.append("Fine-tuned")
    else:
        parts.append("Zero-shot")
    if summary.get("self_consistency"):
        parts.append("self-consistency")
    return " + ".join(parts)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/RESULTS.md"))
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for path in sorted(args.results_dir.glob("*/summary.json")):
        data = json.loads(path.read_text())
        data.setdefault("run", path.parent.name)
        meta_path = path.parent / "generation_meta.json"
        if meta_path.exists():
            data["meta"] = json.loads(meta_path.read_text())
        summaries.append(data)

    if not summaries:
        print(f"[fail] no summary.json under {args.results_dir}; run scripts/evaluate.py first")
        return 1

    # A Spider run named "<run>-spider" fills the Spider column of "<run>".
    spider_ex = {
        s["run"].removesuffix("-spider"): s.get("ex")
        for s in summaries
        if s.get("dataset", "").startswith("spider")
    }

    rows: list[list[str]] = []
    for summary in summaries:
        if summary.get("dataset", "").startswith("spider"):
            continue
        diff = summary.get("by_difficulty", {})
        by_diff = " / ".join(
            fmt(diff.get(name, {}).get("ex")) for name in ("simple", "moderate", "challenging")
        )
        latency = (summary.get("latency_ms") or {}).get("p95")
        cost = (summary.get("cost_per_1k_queries") or {}).get("eur")
        rows.append([
            summary["run"],
            summary.get("model") or "--",
            method_label(summary),
            fmt(summary.get("ex")),
            by_diff,
            fmt(spider_ex.get(summary["run"])),
            fmt(latency, 0),
            fmt(cost, 3),
        ])

    rows.sort(key=lambda r: (r[3] == "--", -float(r[3]) if r[3] != "--" else 0))

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(HEADERS)]
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    table = "\n".join(
        [line(HEADERS), "| " + " | ".join("-" * w for w in widths) + " |"]
        + [line(r) for r in rows]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "# Results\n\nExecution accuracy on BIRD dev, scored with the official "
        "set-comparison metric.\n\n" + table + "\n", encoding="utf-8"
    )
    print(table)
    print(f"\n[write] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
