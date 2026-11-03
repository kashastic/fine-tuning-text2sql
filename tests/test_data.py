from __future__ import annotations

from pathlib import Path

import pytest

from text2sql.data import Example, load_dataset, missing_databases, read_jsonl, write_jsonl


def test_bird_loader(data_root: Path):
    examples = load_dataset("bird", data_root=data_root, split="dev")
    assert len(examples) == 4
    first = examples[0]
    assert first.db_id == "shop"
    assert first.source == "bird"
    assert first.difficulty == "moderate"
    assert first.evidence
    assert Path(first.db_path).exists()


def test_spider_loader_has_no_evidence(data_root: Path):
    examples = load_dataset("spider", data_root=data_root, split="train")
    assert examples[0].source == "spider"
    assert examples[0].evidence == ""
    assert examples[0].gold_sql.lower().startswith("select count")


def test_difficulty_normalisation(data_root: Path):
    seen = {e.difficulty for e in load_dataset("bird", data_root=data_root, split="dev")}
    assert seen <= {"simple", "moderate", "challenging", "unknown"}


def test_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("wikisql")


def test_missing_databases_is_reported(data_root: Path):
    examples = load_dataset("bird", data_root=data_root, split="dev")
    examples.append(Example("x", "ghost", "?", "SELECT 1", "/nope/ghost.sqlite"))
    assert missing_databases(examples) == ["/nope/ghost.sqlite"]


def test_jsonl_roundtrip(tmp_path: Path):
    rows = [{"a": 1, "b": "ü"}, {"a": 2, "b": "x"}]
    path = tmp_path / "nested" / "rows.jsonl"
    write_jsonl(path, rows)
    assert list(read_jsonl(path)) == rows
