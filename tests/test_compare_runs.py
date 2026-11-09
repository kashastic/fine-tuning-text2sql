"""The paired comparison behind every "is this difference real?" claim.

These numbers decide whether a measured EX change gets published as a gain or
as noise (M12's +/-1.5 point band), so the arithmetic is worth pinning down --
especially the pairing, where a silent id collision would invent flips.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from compare_runs import exact_mcnemar, load_correct  # noqa: E402


def test_exact_mcnemar_matches_a_two_sided_binomial():
    # Hand-checkable cases. 10 flips all one way is 2 * 0.5**10.
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar(10, 0) == 2 * 0.5**10
    # A single discordant pair cannot be evidence of anything, and the
    # doubled one-sided tail would exceed 1 without the cap.
    assert exact_mcnemar(1, 0) == 1.0
    # Symmetric in its arguments: which run is "before" cannot change whether
    # the difference is significant.
    assert exact_mcnemar(30, 12) == exact_mcnemar(12, 30)


def test_reproduces_the_published_m11_comparison():
    """The two committed nano runs, whose figures are quoted in M11.

    If this drifts, either the comparison changed or a results directory did,
    and docs/MEASUREMENTS.md is then quoting something that no longer exists.
    """
    before = REPO / "results" / "api-baseline"
    after = REPO / "results" / "api-baseline-v2"
    if not (before / "predictions.jsonl").exists():
        import pytest
        pytest.skip("scored baseline runs not present in this checkout")
    a, b = load_correct(before), load_correct(after)
    shared = set(a) & set(b)
    assert len(shared) == 1534
    gained = sum(1 for k in shared if not a[k] and b[k])
    lost = sum(1 for k in shared if a[k] and not b[k])
    assert (gained, lost) == (76, 70)
    assert round(exact_mcnemar(gained, lost), 4) == 0.6792


def _write_run(path: Path, rows: list[tuple[str, str, bool]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for db_id, qid, correct in rows:
            fh.write(json.dumps({
                "db_id": db_id, "question_id": qid, "correct": correct,
            }) + "\n")


def test_pairing_keys_on_db_and_question_id(tmp_path: Path):
    """Two databases reusing question id "0" are two questions, not one.

    Spider's splits restart ids per file, so keying on question_id alone would
    pair unrelated questions and report the mismatch as a flip.
    """
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_run(a, [("alpha", "0", True), ("beta", "0", False)])
    _write_run(b, [("alpha", "0", True), ("beta", "0", False)])
    left, right = load_correct(a), load_correct(b)
    assert set(left) == {"alpha/0", "beta/0"}
    assert sum(1 for k in left if left[k] != right[k]) == 0


def test_cli_reports_net_and_p_value(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    # Six flips one way, none the other: p = 2 * 0.5**6 = 0.03125.
    rows_a = [("db", str(i), False) for i in range(6)] + [("db", "x", True)]
    rows_b = [("db", str(i), True) for i in range(6)] + [("db", "x", True)]
    _write_run(a, rows_a)
    _write_run(b, rows_b)
    out = tmp_path / "cmp.json"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "compare_runs.py"),
         str(a), str(b), "--json", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text(encoding="utf-8"))["overall"]
    assert (result["gained"], result["lost"], result["net"]) == (6, 0, 6)
    assert round(result["p_value"], 5) == 0.03125
