"""Load BIRD and Spider into one `Example` shape.

Both benchmarks ship a JSON list of questions plus a directory of SQLite files.
They disagree on nearly every key name, so normalise once here and let the rest
of the repo work with `Example`.

Expected layout under `--data-root` (what `scripts/download_data.py` produces):

    data/raw/bird/train/train.json
    data/raw/bird/train/train_databases/<db_id>/<db_id>.sqlite
    data/raw/bird/dev/dev.json
    data/raw/bird/dev/dev_databases/<db_id>/<db_id>.sqlite
    data/raw/spider/train_spider.json
    data/raw/spider/dev.json
    data/raw/spider/database/<db_id>/<db_id>.sqlite
    data/raw/spider/test.json
    data/raw/spider/test_database/<db_id>/<db_id>.sqlite
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

Difficulty = Literal["simple", "moderate", "challenging", "unknown"]


@dataclass(slots=True)
class Example:
    """One natural-language question paired with its gold SQL."""

    question_id: str
    db_id: str
    question: str
    gold_sql: str
    db_path: str
    source: str = "bird"
    split: str = "train"
    evidence: str = ""
    difficulty: Difficulty = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Example":
        known = {f for f in cls.__slots__}
        return cls(**{k: v for k, v in d.items() if k in known})


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not hold a JSON list")
    return data


def _normalise_difficulty(raw: Any) -> Difficulty:
    text = str(raw or "").strip().lower()
    if text in {"simple", "easy"}:
        return "simple"
    if text in {"moderate", "medium"}:
        return "moderate"
    if text in {"challenging", "hard", "extra", "extra hard"}:
        return "challenging"
    return "unknown"


def load_bird(
    root: str | Path,
    split: str = "train",
    questions_file: str | None = None,
    databases_dir: str | None = None,
) -> list[Example]:
    """Load a BIRD split. `root` is the directory holding that split's files."""
    root = Path(root)
    qpath = Path(questions_file) if questions_file else root / f"{split}.json"
    if not qpath.exists():  # some mirrors drop the split prefix
        alt = root / ("dev.json" if split == "dev" else "train.json")
        qpath = alt if alt.exists() else qpath
    dbdir = Path(databases_dir) if databases_dir else root / f"{split}_databases"
    if not dbdir.exists():
        for candidate in (root / "databases", root / "database"):
            if candidate.exists():
                dbdir = candidate
                break

    examples: list[Example] = []
    for i, row in enumerate(_read_json(qpath)):
        db_id = row["db_id"]
        gold = row.get("SQL") or row.get("sql") or row.get("query") or ""
        examples.append(
            Example(
                question_id=str(row.get("question_id", i)),
                db_id=db_id,
                question=row["question"].strip(),
                gold_sql=gold.strip(),
                db_path=str(dbdir / db_id / f"{db_id}.sqlite"),
                source="bird",
                split=split,
                evidence=(row.get("evidence") or "").strip(),
                difficulty=_normalise_difficulty(row.get("difficulty")),
            )
        )
    return examples


def load_spider(
    root: str | Path,
    split: str = "train",
    questions_file: str | None = None,
    databases_dir: str | None = None,
) -> list[Example]:
    """Load a Spider split. Spider keeps every database under one `database/` dir."""
    root = Path(root)
    if questions_file:
        qpath = Path(questions_file)
    else:
        candidates = (
            [root / "train_spider.json", root / "train.json"]
            if split == "train"
            else [root / f"{split}.json"]
        )
        qpath = next((c for c in candidates if c.exists()), candidates[0])
    # Spider keeps train and dev databases in `database/` but the test ones in
    # `test_database/`. Defaulting every split to `database/` would make the
    # test split load cleanly and then report all 2147 databases missing.
    if databases_dir:
        dbdir = Path(databases_dir)
    elif split == "test" and (root / "test_database").exists():
        dbdir = root / "test_database"
    else:
        dbdir = root / "database"

    examples: list[Example] = []
    for i, row in enumerate(_read_json(qpath)):
        db_id = row["db_id"]
        gold = row.get("query") or row.get("SQL") or ""
        examples.append(
            Example(
                question_id=f"spider-{split}-{i}",
                db_id=db_id,
                question=row["question"].strip(),
                gold_sql=gold.strip(),
                db_path=str(dbdir / db_id / f"{db_id}.sqlite"),
                source="spider",
                split=split,
                evidence="",  # Spider has no evidence field
                difficulty="unknown",
            )
        )
    return examples


def load_dataset(
    name: str,
    data_root: str | Path = "data/raw",
    split: str = "train",
    **kwargs: Any,
) -> list[Example]:
    """`load_dataset("bird", split="dev")` -> examples from data/raw/bird/dev."""
    data_root = Path(data_root)
    name = name.lower()
    if name == "bird":
        return load_bird(data_root / "bird" / split, split=split, **kwargs)
    if name == "spider":
        return load_spider(data_root / "spider", split=split, **kwargs)
    raise ValueError(f"unknown dataset: {name!r} (expected 'bird' or 'spider')")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def missing_databases(examples: list[Example]) -> list[str]:
    """Database files referenced by `examples` that are not on disk."""
    return sorted({e.db_path for e in examples if not Path(e.db_path).exists()})
