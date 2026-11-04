#!/usr/bin/env python3
"""Turn BIRD/Spider into the chat-format training set the trainers consume.

Writes JSONL with one `{"messages": [...], "prompt": [...], "db_path": ...,
"gold_sql": ...}` row per example. The extra columns are not decoration: GRPO's
reward function reads `db_path` and `gold_sql` straight off the dataset.

    python scripts/build_dataset.py --config configs/data.yaml
    python scripts/build_dataset.py --datasets bird --splits train --limit 200
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from text2sql.config import load_config
from text2sql.data import Example, load_dataset, write_jsonl
from text2sql.prompts import to_sft_record
from text2sql.schema import database_schema
from text2sql.sql_exec import ExecStatus, execute_sql


def gold_executes(example: Example, timeout: float) -> tuple[bool, str]:
    result = execute_sql(example.db_path, example.gold_sql, timeout=timeout)
    if result.status is ExecStatus.OK:
        return True, ""
    return False, f"{result.status.value}: {result.error}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--style", choices=["sql", "reasoning"], default=None)
    parser.add_argument("--max-tables", type=int, default=None)
    parser.add_argument("--sample-values", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None, help="held out from train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--drop-unexecutable-gold", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    datasets = args.datasets or cfg.get_path("data.datasets", ["bird", "spider"])
    splits = args.splits or cfg.get_path("data.splits", ["train", "dev"])
    data_root = Path(args.data_root or cfg.get_path("data.root", "data/raw"))
    out_dir = Path(args.out_dir or cfg.get_path("data.out_dir", "data/processed"))
    style = args.style or cfg.get_path("data.style", "sql")
    max_tables = args.max_tables if args.max_tables is not None else cfg.get_path("data.max_tables", 8)
    sample_values = args.sample_values if args.sample_values is not None else cfg.get_path("data.sample_values", 3)
    val_size = args.val_size if args.val_size is not None else cfg.get_path("data.val_size", 300)
    limit = args.limit if args.limit is not None else cfg.get_path("data.limit")
    drop_bad_gold = (
        args.drop_unexecutable_gold
        if args.drop_unexecutable_gold is not None
        else cfg.get_path("data.drop_unexecutable_gold", True)
    )
    seed = args.seed if args.seed is not None else cfg.get_path("data.seed", 13)

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    incomplete: list[str] = []
    for split in splits:
        rows: list[dict] = []
        dropped = {"missing_db": 0, "bad_gold": 0, "schema_error": 0}
        for name in datasets:
            try:
                examples = load_dataset(name, data_root=data_root, split=split)
            except FileNotFoundError as exc:
                print(f"[warn] {name}/{split}: {exc}; skipping")
                continue
            if limit:
                examples = examples[:limit]
            print(f"[load] {name}/{split}: {len(examples)} examples")

            for example in examples:
                if not Path(example.db_path).exists():
                    dropped["missing_db"] += 1
                    continue
                if drop_bad_gold and split == "train":
                    ok, why = gold_executes(example, timeout=30.0)
                    if not ok:
                        dropped["bad_gold"] += 1
                        continue
                try:
                    database_schema(example.db_path)  # warm the cache, catch corrupt files
                    rows.append(
                        to_sft_record(
                            example,
                            style=style,
                            max_tables=max_tables,
                            n_sample_values=sample_values,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    dropped["schema_error"] += 1
                    print(f"[warn] {example.db_id}/{example.question_id}: {type(exc).__name__}: {exc}")

        if not rows:
            print(f"[warn] {split}: nothing written")
            continue

        if split == "train" and val_size:
            rng.shuffle(rows)
            val, train = rows[:val_size], rows[val_size:]
            write_jsonl(out_dir / "train.jsonl", train)
            write_jsonl(out_dir / "val.jsonl", val)
            print(f"[write] {out_dir/'train.jsonl'}: {len(train)} rows")
            print(f"[write] {out_dir/'val.jsonl'}: {len(val)} rows")
        else:
            write_jsonl(out_dir / f"{split}.jsonl", rows)
            print(f"[write] {out_dir / f'{split}.jsonl'}: {len(rows)} rows")
        print(f"[drop ] {split}: {dropped}")
        if split == "dev" and any(dropped.values()):
            # A dev row lost here is a question the final numbers silently
            # never answer, so EX stops being comparable with published
            # BIRD results. Loud failure beats a quietly shortened eval set.
            print(f"[fail ] {split}: {sum(dropped.values())} rows dropped from an "
                  f"evaluation split; fix the cause rather than scoring a subset")
            incomplete.append(split)

    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
