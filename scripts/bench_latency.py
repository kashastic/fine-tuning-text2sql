#!/usr/bin/env python3
"""Measure real serving latency and throughput against a running endpoint.

    # terminal 1
    make serve MODEL=outputs/qwen3-4b-text2sql-merged
    # terminal 2
    python scripts/bench_latency.py --base-url http://localhost:8000/v1 \
        --model qwen3-4b-text2sql --concurrency 1 4 16 --requests 64 --gpu a100-40gb

Offline batched generation cannot give these numbers -- it hides queueing, and
p95 under load is the figure that belongs in the results table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401
from text2sql.backends import OpenAICompatBackend
from text2sql.data import load_dataset
from text2sql.metrics import percentile
from text2sql.pricing import self_hosted_cost_per_1k
from text2sql.prompts import build_prompt


def run_level(backend: OpenAICompatBackend, prompts: list, concurrency: int) -> dict:
    latencies: list[float] = []
    out_tokens: list[int] = []
    errors = 0

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for generations in pool.map(lambda m: backend._one(m, 1), prompts):  # noqa: SLF001
            gen = generations[0]
            if gen.error:
                errors += 1
                continue
            if gen.latency_ms is not None:
                latencies.append(gen.latency_ms)
            if gen.output_tokens:
                out_tokens.append(gen.output_tokens)
    wall = time.perf_counter() - started

    if not latencies:
        return {"concurrency": concurrency, "errors": errors, "note": "every request failed"}
    return {
        "concurrency": concurrency,
        "requests": len(prompts),
        "errors": errors,
        "wall_seconds": round(wall, 2),
        "queries_per_second": round(len(prompts) / wall, 3),
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "p99": round(percentile(latencies, 99), 1),
            "max": round(max(latencies), 1),
        },
        "mean_output_tokens": round(sum(out_tokens) / len(out_tokens), 1) if out_tokens else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="bird")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-tables", type=int, default=8)
    parser.add_argument("--sample-values", type=int, default=3)
    parser.add_argument("--gpu", default="a100-40gb")
    parser.add_argument("--gpu-hourly-usd", type=float, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/latency.json"))
    args = parser.parse_args()

    examples = load_dataset(args.dataset, data_root=args.data_root, split=args.split)
    examples = [e for e in examples if Path(e.db_path).exists()][: args.requests]
    if not examples:
        print("[fail] no examples with a database on disk")
        return 1
    prompts = [
        build_prompt(e, max_tables=args.max_tables, n_sample_values=args.sample_values)
        for e in examples
    ]

    backend = OpenAICompatBackend(
        model=args.model, base_url=args.base_url,
        max_new_tokens=args.max_new_tokens, max_workers=max(args.concurrency),
    )

    if args.warmup:
        print(f"[warm ] {args.warmup} requests")
        run_level(backend, prompts[: args.warmup], concurrency=min(args.warmup, 4))

    levels = []
    for concurrency in args.concurrency:
        print(f"[bench] concurrency={concurrency}, {len(prompts)} requests")
        level = run_level(backend, prompts, concurrency)
        if "queries_per_second" in level:
            level["cost_per_1k_queries"] = self_hosted_cost_per_1k(
                level["queries_per_second"], gpu=args.gpu, hourly_usd=args.gpu_hourly_usd
            )
            lat = level["latency_ms"]
            print(f"        p50 {lat['p50']:.0f} ms  p95 {lat['p95']:.0f} ms  "
                  f"{level['queries_per_second']:.2f} q/s  "
                  f"EUR {level['cost_per_1k_queries']['eur']:.3f} / 1k")
        levels.append(level)

    report = {
        "model": args.model,
        "base_url": args.base_url,
        "gpu": args.gpu,
        "gpu_hourly_usd": args.gpu_hourly_usd,
        "max_new_tokens": args.max_new_tokens,
        "levels": levels,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[write] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
