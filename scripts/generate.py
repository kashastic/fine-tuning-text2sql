#!/usr/bin/env python3
"""Generate SQL for a split and write raw model output to results/<run>/.

Generation and scoring are separate steps on purpose: generating on a rented
GPU costs money, scoring is free and gets re-run every time the comparison
logic changes. So this writes raw text, and `scripts/evaluate.py` turns it
into numbers.

    # zero-shot baseline on a rented GPU
    python scripts/generate.py --backend vllm --model Qwen/Qwen3-4B \
        --dataset bird --split dev --run-name qwen3-4b-zeroshot

    # the large prompted baseline, same prompts
    python scripts/generate.py --backend anthropic --model claude-opus-5-5 \
        --dataset bird --split dev --run-name api-baseline --limit 200

    # a fine-tuned adapter, 8 samples for self-consistency
    python scripts/generate.py --backend vllm --model Qwen/Qwen3-4B \
        --lora outputs/sft-best --dataset bird --split dev -n 8 \
        --run-name sft-grpo-sc8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from text2sql.backends import build_backend
from text2sql.config import dump_config, load_config
from text2sql.data import load_dataset
from text2sql.prompts import build_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--backend", choices=["vllm", "openai", "anthropic", "hf"], default="vllm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", help="adapter directory (vllm / hf backends)")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--dataset", default="bird")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--style", choices=["sql", "reasoning"], default="sql")
    parser.add_argument("--max-tables", type=int, default=8)
    parser.add_argument("--max-columns", type=int, default=0)
    parser.add_argument("--sample-values", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("-n", "--num-samples", type=int, default=1,
                        help="samples per question; >1 enables self-consistency at eval time")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--effort", default="low", help="anthropic backend only")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    run_dir = Path(args.results_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    examples = load_dataset(args.dataset, data_root=args.data_root, split=args.split)
    if args.limit:
        examples = examples[: args.limit]
    missing = [e for e in examples if not Path(e.db_path).exists()]
    if missing:
        print(f"[warn] {len(missing)} examples reference a missing database; dropping them")
        examples = [e for e in examples if Path(e.db_path).exists()]
    if not examples:
        print("[fail] no examples to run; check --data-root and run download_data.py --verify-only")
        return 1

    print(f"[prep] building {len(examples)} prompts (style={args.style}, max_tables={args.max_tables})")
    prompts = [
        build_prompt(e, style=args.style, max_tables=args.max_tables,
                     max_columns=args.max_columns,
                     n_sample_values=args.sample_values)
        for e in examples
    ]

    backend_kwargs: dict = {"model": args.model, "max_new_tokens": args.max_new_tokens}
    if args.backend == "vllm":
        backend_kwargs |= {
            "max_model_len": args.max_model_len,
            "temperature": args.temperature,
            "lora_path": args.lora,
        }
    elif args.backend == "openai":
        backend_kwargs |= {
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_workers": args.max_workers,
        }
    elif args.backend == "anthropic":
        backend_kwargs |= {"effort": args.effort, "max_workers": args.max_workers}
    else:
        backend_kwargs |= {"temperature": args.temperature, "adapter_path": args.lora}

    print(f"[gen ] backend={args.backend} model={args.model} n={args.num_samples}")
    backend = build_backend(args.backend, **backend_kwargs)
    started = time.perf_counter()
    outputs = backend.generate(prompts, n=args.num_samples)
    wall = time.perf_counter() - started
    print(f"[gen ] done in {wall:.1f}s ({len(examples) / wall:.2f} questions/s)")

    path = run_dir / "generations.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for example, generations in zip(examples, outputs, strict=True):
            fh.write(json.dumps({
                "question_id": example.question_id,
                "db_id": example.db_id,
                "db_path": example.db_path,
                "question": example.question,
                "evidence": example.evidence,
                "gold_sql": example.gold_sql,
                "difficulty": example.difficulty,
                "source": example.source,
                "samples": [
                    {
                        "text": g.text,
                        "input_tokens": g.input_tokens,
                        "output_tokens": g.output_tokens,
                        "latency_ms": g.latency_ms,
                        "error": g.error,
                    }
                    for g in generations
                ],
            }, ensure_ascii=False) + "\n")

    meta = {
        "run_name": args.run_name,
        "backend": args.backend,
        "model": args.model,
        "lora": args.lora,
        "dataset": args.dataset,
        "split": args.split,
        "style": args.style,
        "max_tables": args.max_tables,
        "max_columns": args.max_columns,
        "sample_values": args.sample_values,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "effort": args.effort if args.backend == "anthropic" else None,
        "n_examples": len(examples),
        "wall_seconds": round(wall, 2),
        "questions_per_second": round(len(examples) / wall, 4),
        "config": dict(cfg),
    }
    (run_dir / "generation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    dump_config(meta, run_dir / "run.yaml")
    print(f"[write] {path}")
    print(f"[next ] python scripts/evaluate.py --run {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
