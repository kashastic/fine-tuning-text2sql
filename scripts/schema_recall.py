#!/usr/bin/env python3
"""Does the trimmed prompt still contain the columns the gold query needs?

This is the measurement behind M4, M11 and M16: schema trimming is the biggest
lever on prompt length, so on training cost, but trimming away a column the
answer needs turns a hard question into an impossible one -- and nothing
errors. It has been re-derived by hand three times, so it lives here now.

    python scripts/schema_recall.py --dataset bird --split dev
    python scripts/schema_recall.py --max-columns 20 --list 40
    python scripts/schema_recall.py --json out.json

It reports against the **selected column set**, not against a substring search
of the rendered prompt. The old proxy (assumption A9) asked whether the gold's
column names appeared anywhere in the prompt text, which over-counts: a short
name like `id` or `type` matches inside an unrelated one. Here a gold column
counts as present only when that exact column of that exact table was
selected, so the number is a recall figure rather than an estimate of one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from text2sql.data import load_dataset
from text2sql.schema import (
    database_schema,
    select_relevant_columns,
    select_relevant_tables,
)

try:  # postdates the baseline this script is sometimes run against
    from text2sql.schema import value_columns_for
except ImportError:  # pragma: no cover
    def value_columns_for(table_name, value_hits):  # type: ignore[misc]
        return set()

# Identifiers as the benchmarks actually write them: BIRD quotes names with
# spaces and parentheses in backticks (`Free Meal Count (K-12)`), SQLite also
# accepts [brackets] and "double quotes", and plain names need no quoting.
_IDENT = re.compile(
    r"`([^`]+)`"          # `Free Meal Count (K-12)`
    r'|\[([^\]]+)\]'      # [Order Details]
    r'|"([^"]+)"'         # "District Code"
    r"|([A-Za-z_][A-Za-z0-9_]*)"
)

# Bare words in SQL are not all column references, and the overlap with real
# column names is not hypothetical: `count`, `order`, `date`, `year`, `name`,
# `max` and `value` are all column names somewhere in BIRD. Counting the
# `COUNT` in `SELECT COUNT(*)` as a reference to a column called `count`
# inflates the loss figure, so keywords and functions come out first.
_SQL_WORDS = {
    "select", "distinct", "from", "where", "group", "by", "having", "order",
    "asc", "desc", "limit", "offset", "join", "inner", "outer", "left",
    "right", "full", "cross", "on", "using", "as", "and", "or", "not", "in",
    "is", "null", "like", "glob", "between", "exists", "case", "when",
    "then", "else", "end", "union", "all", "intersect", "except", "with",
    "recursive", "cast", "collate", "nocase", "escape", "values", "insert",
    "into", "update", "set", "delete", "create", "table", "view", "index",
    "count", "sum", "avg", "min", "max", "total", "abs", "round", "length",
    "substr", "substring", "replace", "trim", "ltrim", "rtrim", "upper",
    "lower", "coalesce", "ifnull", "nullif", "iif", "strftime", "date",
    "datetime", "julianday", "time", "typeof", "instr", "printf", "format",
    "group_concat", "row_number", "rank", "dense_rank", "over", "partition",
    "cast_as", "real", "integer", "text", "numeric", "blob", "div", "mod",
    "true", "false",
}


def gold_identifiers(sql: str) -> set[str]:
    """Every name the gold query could be referring to, lowercased.

    Quoted names are kept whatever they say -- a column really can be called
    `Order` or `Date`, and the quotes are what prove the author meant the
    column. Only *bare* words are filtered against the keyword list.
    """
    found: set[str] = set()
    for match in _IDENT.finditer(sql):
        quoted = match.group(1) or match.group(2) or match.group(3)
        if quoted is not None:
            found.add(quoted.strip().lower())
            continue
        bare = match.group(4).strip().lower()
        if bare not in _SQL_WORDS:
            found.add(bare)
    return found


def audit_example(example, max_tables: int, max_columns: int,
                  link_values: bool = True) -> dict:
    """What the prompt would show for one question, and what it would lose.

    Only tables the gold query actually names are audited. Scanning every
    table in the database would count a loss whenever *some* table happened
    to share a column name with one the gold used -- `name` and `id` would
    then register against half the schema.
    """
    schema = database_schema(example.db_path)
    value_hits: set[str] = frozenset()
    if link_values:
        from text2sql.values import columns_matching_values

        value_hits = columns_matching_values(
            example.db_path, example.question, example.evidence
        )
    table_kwargs = {"value_hits": value_hits} if value_hits else {}
    chosen = select_relevant_tables(
        schema, example.question, example.evidence, max_tables=max_tables,
        **table_kwargs,
    )
    wanted = gold_identifiers(example.gold_sql)
    referenced = {n for n in schema.tables if n.lower() in wanted}

    shown_total = 0
    column_total = 0
    missing_columns: list[str] = []
    for name, table in schema.tables.items():
        column_total += len(table.columns)
        if name in chosen:
            # Only pass value_columns when there are some: the baseline
            # revision this script is also run against has no such parameter,
            # and a comparison script that cannot run on the thing being
            # compared to is useless.
            extra = {}
            if value_hits:
                extra["value_columns"] = value_columns_for(name, value_hits)
            kept = select_relevant_columns(
                table, example.question, example.evidence,
                max_columns=max_columns, **extra,
            )
            shown_total += len(kept)
        else:
            kept = []
        if name not in referenced:
            continue
        needed = [c for c in table.columns if c.name.lower() in wanted]
        kept_names = {c.name for c in kept}
        # A needed column in a table the prompt dropped entirely still counts
        # as lost; `missing_tables` says which losses have that cause.
        missing_columns.extend(
            f"{name}.{c.name}" for c in needed if c.name not in kept_names
        )

    missing_tables = sorted(referenced - set(chosen))
    return {
        "question_id": example.question_id,
        "db_id": example.db_id,
        "question": example.question,
        "evidence": example.evidence,
        "gold_sql": example.gold_sql,
        "missing_columns": sorted(set(missing_columns)),
        "missing_tables": missing_tables,
        "columns_shown": shown_total,
        "columns_total": column_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="bird")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--data-root", default="data/raw", type=Path)
    parser.add_argument("--max-tables", type=int, default=8)
    parser.add_argument("--max-columns", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tokenizer", default=None,
                        help="measure real prompt lengths, e.g. Qwen/Qwen3-4B")
    parser.add_argument("--no-value-linking", action="store_true",
                        help="name-based selection only, for the before/after")
    parser.add_argument("--list", type=int, default=0,
                        help="print this many failing questions in full")
    parser.add_argument("--json", type=Path, default=None,
                        help="write every per-question audit here")
    args = parser.parse_args()

    examples = load_dataset(args.dataset, data_root=args.data_root, split=args.split)
    if args.limit:
        examples = examples[: args.limit]

    audits = []
    for example in examples:
        try:
            audits.append(audit_example(
                example, args.max_tables, args.max_columns,
                link_values=not args.no_value_linking))
        except Exception as exc:  # noqa: BLE001 - a broken database is a result
            print(f"[warn] {example.db_id}/{example.question_id}: "
                  f"{type(exc).__name__}: {exc}")

    n = len(audits)
    lost_column = [a for a in audits if a["missing_columns"]]
    lost_table = [a for a in audits if a["missing_tables"]]
    shown = sum(a["columns_shown"] for a in audits)
    total = sum(a["columns_total"] for a in audits)

    print(f"\n{args.dataset}/{args.split} at max_tables={args.max_tables} "
          f"max_columns={args.max_columns}, "
          f"value_linking={'off' if args.no_value_linking else 'on'}, "
          f"{n} questions")
    print(f"  missing a needed column : {len(lost_column):5d}  ({len(lost_column)/n:.2%})")
    print(f"  missing a needed table  : {len(lost_table):5d}  ({len(lost_table)/n:.2%})")
    print(f"  columns shown           : {shown/total:.2%} of {total}")

    if args.tokenizer:
        # Prompt length is the other half of the trade: trimming exists to
        # fit a context window, so a recall gain that pushes prompts past
        # the trainer's limit is not a gain. Measured with the real
        # tokenizer, over the same prompts the model would see.
        from transformers import AutoTokenizer

        from text2sql.prompts import build_prompt
        from text2sql.schema import schema_for_example

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        lengths, with_gold = [], []
        for example in examples:
            kwargs = {}
            if args.no_value_linking:
                # Absent on the baseline revision, where there is nothing to
                # switch off in the first place.
                kwargs["link_values"] = False
            text = schema_for_example(
                example.db_path, example.question, example.evidence,
                max_tables=args.max_tables, n_sample_values=3,
                max_columns=args.max_columns, **kwargs,
            )
            messages = build_prompt(example, style="sql", schema_text=text)
            prompt = "\n".join(m["content"] for m in messages)
            n_prompt = len(tok(prompt).input_ids)
            lengths.append(n_prompt)
            with_gold.append(n_prompt + len(tok(example.gold_sql).input_ids))
        lengths.sort()
        def pct(xs, q):
            return xs[min(len(xs) - 1, int(q * len(xs)))]
        print(f"  prompt tokens           : p50 {pct(lengths,.5)}  p90 {pct(lengths,.9)}"
              f"  p99 {pct(lengths,.99)}  max {lengths[-1]}")
        print(f"  over 3072 / 4096        : "
              f"{sum(x>3072 for x in lengths)/len(lengths):.2%} / "
              f"{sum(x>4096 for x in lengths)/len(lengths):.2%}")
        print(f"  longest prompt + gold   : {max(with_gold)}")

    by_db = Counter(a["db_id"] for a in lost_column)
    if by_db:
        print("  worst databases:")
        for db, count in by_db.most_common(8):
            print(f"    {count:4d}  {db}")

    for audit in lost_column[: args.list]:
        print(f"\n--- {audit['db_id']}/{audit['question_id']}")
        print(f"  Q: {audit['question']}")
        if audit["evidence"]:
            print(f"  E: {audit['evidence']}")
        print(f"  missing: {', '.join(audit['missing_columns'])}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(audits, indent=2), encoding="utf-8")
        print(f"\n[write] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
