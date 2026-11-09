"""End-to-end on the fixture benchmark: build a dataset, fake a run, score it.

This is the test that would catch a break between the pieces -- a renamed
column in the dataset that the reward function reads, or a scorer that stops
understanding what the generator wrote.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / name), *args],
        capture_output=True, text=True, cwd=REPO, check=False,
    )


def test_build_dataset_then_score_a_run(tmp_path: Path, data_root: Path):
    out_dir = tmp_path / "processed"
    built = run_script(
        "build_dataset.py",
        "--datasets", "bird", "--splits", "dev",
        "--data-root", str(data_root), "--out-dir", str(out_dir),
        "--max-tables", "8", "--sample-values", "2",
    )
    assert built.returncode == 0, built.stderr
    rows = [json.loads(line) for line in (out_dir / "dev.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert {"messages", "prompt", "db_path", "gold_sql"} <= set(rows[0])

    # Fake a generation run: two right, one unparseable, one whose gold is broken.
    run_dir = tmp_path / "results" / "fake"
    run_dir.mkdir(parents=True)
    answers = [
        rows[0]["gold_sql"],                                    # correct
        "SELECT COUNT(*) FROM customer WHERE city = 'Zurich'",  # correct
        "",                                                     # nothing parseable
        "SELECT name FROM customer",                            # gold is broken
    ]
    with (run_dir / "generations.jsonl").open("w") as fh:
        for row, answer in zip(rows, answers, strict=True):
            fh.write(json.dumps({
                "question_id": row["question_id"], "db_id": row["db_id"],
                "db_path": row["db_path"], "question": "q",
                "gold_sql": row["gold_sql"], "difficulty": row["difficulty"],
                "source": row["source"],
                "samples": [{"text": f"```sql\n{answer}\n```" if answer else "sorry",
                             "latency_ms": 120.0, "input_tokens": 900, "output_tokens": 40}],
            }) + "\n")

    scored = run_script("evaluate.py", "--run", str(run_dir))
    assert scored.returncode == 0, scored.stderr

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n"] == 4
    assert summary["correct"] == 2
    assert summary["ex"] == 50.0
    # The fixture's fourth question has deliberately broken gold SQL.
    assert summary["gold_unexecutable"] == 1
    assert summary["by_difficulty"]["moderate"]["ex"] == 100.0

    predictions = [json.loads(l) for l in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert predictions[2]["pred_status"] == "empty"

    analysed = run_script("error_analysis.py", "--run", str(run_dir), "--sample", "0")
    assert analysed.returncode == 0, analysed.stderr
    report = json.loads((run_dir / "error_analysis.json").read_text())
    assert report["n_categorised"] == 2
    # A broken gold query is its own category, never blamed on the model.
    assert report["counts"] == {"no_sql_parsed": 1, "gold_unexecutable": 1}

    table = run_script("make_results_table.py", "--results-dir", str(run_dir.parent),
                       "--out", str(tmp_path / "RESULTS.md"))
    assert table.returncode == 0, table.stderr
    assert "BIRD dev EX" in (tmp_path / "RESULTS.md").read_text()


def test_self_consistency_beats_a_single_bad_sample(tmp_path: Path, data_root: Path):
    from text2sql.data import load_dataset

    example = load_dataset("bird", data_root=data_root, split="dev")[1]
    run_dir = tmp_path / "sc"
    run_dir.mkdir(parents=True)
    wrong = "SELECT COUNT(*) FROM customer"
    with (run_dir / "generations.jsonl").open("w") as fh:
        fh.write(json.dumps({
            "question_id": example.question_id, "db_id": example.db_id,
            "db_path": example.db_path, "question": example.question,
            "gold_sql": example.gold_sql, "difficulty": example.difficulty,
            "source": "bird",
            "samples": [
                {"text": f"```sql\n{wrong}\n```"},                 # first sample is wrong
                {"text": f"```sql\n{example.gold_sql}\n```"},
                {"text": "```sql\nSELECT COUNT(*) FROM customer WHERE city='Zurich'\n```"},
            ],
        }) + "\n")

    greedy = run_script("evaluate.py", "--run", str(run_dir))
    assert greedy.returncode == 0, greedy.stderr
    assert json.loads((run_dir / "summary.json").read_text())["correct"] == 0

    voted = run_script("evaluate.py", "--run", str(run_dir), "--self-consistency")
    assert voted.returncode == 0, voted.stderr
    assert json.loads((run_dir / "summary.json").read_text())["correct"] == 1


@pytest.mark.parametrize("script", [
    "download_data.py", "build_dataset.py", "generate.py", "evaluate.py",
    "train_sft.py", "train_grpo.py", "merge_lora.py", "error_analysis.py",
    "bench_latency.py", "make_results_table.py", "schema_recall.py",
    "compare_runs.py",
])
def test_every_script_has_working_help(script: str):
    result = run_script(script, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_download_rejects_an_html_page_posing_as_an_archive(tmp_path: Path):
    """A download host that answers with a web page must fail by saying so.

    Drive-style hosts answer an unconfirmed large-file download with an HTML
    interstitial, and a login wall or rate limit looks the same. The file that
    lands is a plausible size, so without a magic-byte check the only symptom
    is `BadZipFile` -- after the download has apparently succeeded.
    """
    fake = tmp_path / "spider.zip"
    fake.write_text(
        "<!DOCTYPE html><html><body>Sign in to continue</body></html>",
        encoding="utf-8",
    )
    result = run_script(
        "download_data.py",
        "--dataset", "spider",
        "--data-root", str(tmp_path / "raw"),
        "--archive", str(fake),
    )
    assert result.returncode != 0
    out = result.stdout + result.stderr
    assert "not a zip" in out, out
    assert "Sign in to continue" in out, "the guard should quote what arrived"
    assert "BadZipFile" not in out, "it should not get as far as unzipping"
