#!/usr/bin/env python3
"""Score a generations file: execution accuracy, breakdowns, cost.

    python scripts/evaluate.py --run results/qwen3-4b-zeroshot
    python scripts/evaluate.py --run results/sft-grpo-sc8 --self-consistency
    python scripts/evaluate.py --run results/api-baseline --gpu a100-40gb

Writes `predictions.jsonl` (one scored row per question, with the raw output
kept for error analysis) and `summary.json` next to the generations.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import _bootstrap  # noqa: F401
from text2sql.data import read_jsonl
from text2sql.metrics import Prediction, majority_vote, summarise
from text2sql.postprocess import extract_sql
from text2sql.pricing import api_cost_per_1k, self_hosted_cost_per_1k
from text2sql.sql_exec import ComparisonMode, ExecStatus, compare_results, execute_sql

# Backends billed per token rather than per GPU hour. `openai` is excluded:
# it points at a locally served model, whose cost is GPU time.
TOKEN_BILLED = {"anthropic", "openai-api"}


def score_row(row: dict[str, Any], timeout: float, mode: ComparisonMode,
              self_consistency: bool) -> Prediction:
    db_path = row["db_path"]
    gold_sql = row["gold_sql"]
    samples = row.get("samples") or []

    gold = execute_sql(db_path, gold_sql, timeout=timeout)

    candidates: list[tuple[str, tuple | None]] = []
    first_pred = None
    for sample in samples:
        sql = extract_sql(sample.get("text", ""))
        result = execute_sql(db_path, sql, timeout=timeout) if sql else None
        if first_pred is None:
            first_pred = (sql, result, sample)
        candidates.append((sql, result.rows if (result and result.ok) else None))

    if self_consistency and len(candidates) > 1:
        chosen_sql = majority_vote(candidates)
        chosen = execute_sql(db_path, chosen_sql, timeout=timeout) if chosen_sql else None
        sample = samples[0]
    else:
        chosen_sql, chosen, sample = first_pred if first_pred else ("", None, {})

    correct = bool(
        gold.ok and chosen is not None and chosen.ok
        and compare_results(chosen.rows, gold.rows, mode)
    )
    latencies = [s.get("latency_ms") for s in samples if s.get("latency_ms") is not None]
    in_tok = [s.get("input_tokens") for s in samples if s.get("input_tokens") is not None]
    out_tok = [s.get("output_tokens") for s in samples if s.get("output_tokens") is not None]

    return Prediction(
        question_id=str(row.get("question_id", "")),
        db_id=row.get("db_id", ""),
        question=row.get("question", ""),
        gold_sql=gold_sql,
        pred_sql=chosen_sql,
        correct=correct,
        difficulty=row.get("difficulty", "unknown"),
        source=row.get("source", "bird"),
        pred_status=(chosen.status.value if chosen is not None else ExecStatus.EMPTY.value),
        gold_status=gold.status.value,
        error=(chosen.error if chosen is not None else "no SQL parsed") or gold.error,
        raw_output=sample.get("text", "") if sample else "",
        latency_ms=(sum(latencies) if latencies else None),
        input_tokens=(sum(in_tok) if in_tok else None),
        output_tokens=(sum(out_tok) if out_tok else None),
        meta={"n_samples": len(samples), "self_consistency": self_consistency},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path, help="results/<run-name>")
    parser.add_argument("--generations", type=Path, help="defaults to <run>/generations.jsonl")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--comparison", choices=[m.value for m in ComparisonMode],
                        default=ComparisonMode.BIRD.value)
    parser.add_argument("--self-consistency", action="store_true",
                        help="majority-vote over result sets when the run has n>1 samples")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu", default=None, help="price self-hosted cost at this GPU's hourly rate")
    parser.add_argument("--gpu-hourly-usd", type=float, default=None)
    args = parser.parse_args()

    gen_path = args.generations or (args.run / "generations.jsonl")
    if not gen_path.exists():
        print(f"[fail] {gen_path} not found; run scripts/generate.py first")
        return 1

    rows = list(read_jsonl(gen_path))
    print(f"[eval] scoring {len(rows)} questions from {gen_path}")
    mode = ComparisonMode(args.comparison)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        predictions = list(
            pool.map(lambda r: score_row(r, args.timeout, mode, args.self_consistency), rows)
        )

    summary = summarise(predictions)
    meta_path = args.run / "generation_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    summary["run"] = meta.get("run_name", args.run.name)
    summary["model"] = meta.get("model")
    summary["backend"] = meta.get("backend")
    summary["dataset"] = f"{meta.get('dataset', '?')}/{meta.get('split', '?')}"
    summary["self_consistency"] = args.self_consistency
    summary["comparison"] = mode.value

    # Cost: tokens for an API run, GPU seconds for a self-hosted one.
    # Every hosted backend is token-billed, not just the Anthropic one --
    # gating this on a single backend name left the cost column, which is one
    # of the three the project is actually comparing on, permanently empty.
    tokens = summary.get("tokens") or {}
    if meta.get("backend") in TOKEN_BILLED and tokens.get("input_mean"):
        try:
            summary["cost_per_1k_queries"] = api_cost_per_1k(
                meta["model"], tokens["input_mean"], tokens.get("output_mean") or 0.0
            )
        except KeyError as exc:
            summary["cost_per_1k_queries"] = {"error": str(exc)}
    elif meta.get("questions_per_second") and (args.gpu or args.gpu_hourly_usd):
        summary["cost_per_1k_queries"] = self_hosted_cost_per_1k(
            meta["questions_per_second"], gpu=args.gpu or "a100-40gb",
            hourly_usd=args.gpu_hourly_usd,
        )

    gold_failures = sum(1 for p in predictions if p.gold_status != ExecStatus.OK.value)
    if gold_failures:
        summary["gold_unexecutable"] = gold_failures
        print(f"[warn] {gold_failures} gold queries did not execute; those count as wrong")

    pred_path = args.run / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as fh:
        for p in predictions:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"  EX            {summary['ex']:.2f}%  ({summary['correct']}/{summary['n']})")
    print(f"  valid SQL     {summary['valid_sql_rate']:.2f}%")
    for name, stats in summary["by_difficulty"].items():
        print(f"  {name:<13} {stats['ex']:.2f}%  (n={stats['n']})")
    if "latency_ms" in summary:
        lat = summary["latency_ms"]
        print(f"  latency       p50 {lat['p50']:.0f} ms / p95 {lat['p95']:.0f} ms")
    if "cost_per_1k_queries" in summary:
        print(f"  cost / 1k     {summary['cost_per_1k_queries']}")
    print()
    print(f"[write] {pred_path}")
    print(f"[write] {args.run / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
