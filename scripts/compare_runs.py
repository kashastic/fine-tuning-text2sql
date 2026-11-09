#!/usr/bin/env python3
"""Is the difference between two runs real, or is it the backend's own noise?

Two EX numbers from this project's prompted baseline cannot be compared by
subtracting them: the API refuses greedy decoding, so it disagrees with itself
on about 10% of BIRD dev given byte-identical prompts (M12). The honest
comparison is **paired** -- same question, both runs -- which is what this
does.

    python scripts/compare_runs.py results/api-baseline-v2 results/api-baseline-v3

`wrong->right` and `right->wrong` are the discordant pairs. Exact McNemar is a
two-sided binomial test on them against p=0.5: "given that an answer flipped,
was it as likely to flip either way?" Concordant pairs carry no information
about the difference and are excluded by construction, which is the whole
point -- 1,283 questions both runs got right say nothing about which prompt
recipe is better.

To attribute the flips, pass prompt hashes from each side (the second half of
M11's method, which previously lived in a shell history):

    # in a worktree at the old revision
    python scripts/compare_runs.py --hash-prompts /tmp/before.json
    # back on the new one
    python scripts/compare_runs.py --hash-prompts /tmp/after.json
    python scripts/compare_runs.py results/A results/B \
        --prompts-before /tmp/before.json --prompts-after /tmp/after.json

Questions whose prompt is byte-identical across the two revisions are a
control group measured at the same moment as the treatment: whatever they
flip by is noise, because nothing about them changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import _bootstrap  # noqa: F401


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b wrong->right, c right->wrong.

    The exact test rather than the chi-square approximation: with the counts
    this project sees (tens, not thousands) and no continuity correction to
    argue about, `math.comb` is both simpler and right. Implemented here to
    keep the script dependency-free -- it is one line of combinatorics.
    """
    n = b + c
    if n == 0:
        return 1.0
    # P(X <= min) + P(X >= max) under Binomial(n, 0.5), i.e. everything at
    # least as lopsided as what was observed, in either direction.
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def load_correct(run_dir: Path) -> dict[str, bool]:
    """question_id -> correct, from a scored run's predictions.jsonl."""
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        print(f"[fail] {path} not found; run scripts/evaluate.py on {run_dir} first")
        raise SystemExit(1)
    out: dict[str, bool] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            # db_id is in the key because Spider question ids restart per
            # split file, and a silent id collision would pair up two
            # different questions and report the result as a flip.
            out[f"{row['db_id']}/{row['question_id']}"] = bool(row["correct"])
    return out


def hash_prompts(dataset: str, split: str, data_root: str, max_tables: int,
                 max_columns: int, sample_values: int, out: Path) -> int:
    """Write question -> sha256 of the prompt this checkout builds for it."""
    from text2sql.data import load_dataset
    from text2sql.prompts import build_prompt

    examples = load_dataset(dataset, data_root=data_root, split=split)
    hashes: dict[str, str] = {}
    for example in examples:
        try:
            messages = build_prompt(
                example, style="sql", max_tables=max_tables,
                max_columns=max_columns, n_sample_values=sample_values,
            )
        except Exception as exc:  # noqa: BLE001 - a broken database is a result
            print(f"[warn] {example.db_id}/{example.question_id}: {exc}")
            continue
        text = "\n".join(m["content"] for m in messages)
        key = f"{example.db_id}/{example.question_id}"
        hashes[key] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(f"[write] {out}  ({len(hashes)} prompts)")
    return 0


def report(label: str, keys: list[str], a: dict[str, bool],
           b: dict[str, bool]) -> dict:
    both = sum(1 for k in keys if a[k] and b[k])
    neither = sum(1 for k in keys if not a[k] and not b[k])
    gained = [k for k in keys if not a[k] and b[k]]
    lost = [k for k in keys if a[k] and not b[k]]
    n = len(keys)
    p = exact_mcnemar(len(gained), len(lost))
    churn = (len(gained) + len(lost)) / n if n else 0.0
    print(f"\n{label}  (n={n})")
    print(f"  A correct               : {both + len(lost):5d}  ({(both+len(lost))/n:.2%})" if n else "")
    print(f"  B correct               : {both + len(gained):5d}  ({(both+len(gained))/n:.2%})" if n else "")
    print(f"  both right / both wrong : {both:5d} / {neither}")
    print(f"  wrong -> right          : {len(gained):5d}")
    print(f"  right -> wrong          : {len(lost):5d}")
    print(f"  net                     : {len(gained) - len(lost):+5d} questions")
    print(f"  churn (either direction) : {churn:.2%}")
    print(f"  exact McNemar p          : {p:.4f}")
    return {
        "label": label, "n": n, "a_correct": both + len(lost),
        "b_correct": both + len(gained), "gained": len(gained),
        "lost": len(lost), "net": len(gained) - len(lost),
        "churn": round(churn, 6), "p_value": round(p, 6),
        "gained_keys": sorted(gained), "lost_keys": sorted(lost),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="*", type=Path,
                        help="two scored run directories: before, after")
    parser.add_argument("--hash-prompts", type=Path, default=None,
                        help="write prompt hashes for this checkout and exit")
    parser.add_argument("--prompts-before", type=Path, default=None)
    parser.add_argument("--prompts-after", type=Path, default=None)
    parser.add_argument("--dataset", default="bird")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--max-tables", type=int, default=8)
    parser.add_argument("--max-columns", type=int, default=20)
    parser.add_argument("--sample-values", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.hash_prompts:
        return hash_prompts(args.dataset, args.split, args.data_root,
                            args.max_tables, args.max_columns,
                            args.sample_values, args.hash_prompts)

    if len(args.runs) != 2:
        print("[fail] need exactly two run directories (before, after)")
        return 1

    before, after = args.runs
    a, b = load_correct(before), load_correct(after)
    shared = sorted(set(a) & set(b))
    if not shared:
        print("[fail] the two runs share no question ids")
        return 1
    if len(shared) != len(a) or len(shared) != len(b):
        # Not fatal, but it changes what the number means, so say it.
        print(f"[warn] pairing on {len(shared)} shared questions "
              f"({len(a)} in {before.name}, {len(b)} in {after.name})")

    print(f"A = {before}\nB = {after}")
    results = {"a": str(before), "b": str(after),
               "overall": report("overall", shared, a, b)}

    if args.prompts_before and args.prompts_after:
        h_before = json.loads(args.prompts_before.read_text(encoding="utf-8"))
        h_after = json.loads(args.prompts_after.read_text(encoding="utf-8"))
        known = [k for k in shared if k in h_before and k in h_after]
        if len(known) != len(shared):
            print(f"\n[warn] {len(shared) - len(known)} questions have no hash "
                  f"on one side; excluded from the split")
        same = [k for k in known if h_before[k] == h_after[k]]
        diff = [k for k in known if h_before[k] != h_after[k]]
        # The control arm first: it is the one that says what the noise is.
        results["prompt_identical"] = report(
            "prompt byte-identical (control: nothing changed here)", same, a, b)
        results["prompt_changed"] = report(
            "prompt changed (treatment)", diff, a, b)
        print("\nRead the control arm as this backend's noise on this split; a "
              "\ntreatment effect has to clear it to mean anything.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n[write] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
