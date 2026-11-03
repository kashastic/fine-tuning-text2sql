from __future__ import annotations

import pytest

from text2sql.postprocess import extract_reasoning, extract_sql, normalise_sql


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("```SQL\nSELECT 1\n```", "SELECT 1"),
        ("```\nSELECT 1\n```", "SELECT 1"),
        ("SELECT 1", "SELECT 1"),
        ("Here you go:\n```sql\nSELECT 1;\n```\nThis query returns one.", "SELECT 1"),
        ("no sql at all here", ""),
        ("", ""),
    ],
)
def test_extraction(raw, expected):
    assert extract_sql(raw) == expected


def test_last_sql_block_wins():
    raw = "First attempt:\n```sql\nSELECT 1\n```\nActually:\n```sql\nSELECT 2\n```"
    assert extract_sql(raw) == "SELECT 2"


def test_unterminated_fence_is_recovered():
    # Hitting max_tokens mid-answer should still yield the query.
    assert extract_sql("```sql\nSELECT name FROM customer") == "SELECT name FROM customer"


def test_with_clause_is_found_without_a_fence():
    raw = "I'd write:\nWITH t AS (SELECT 1 AS x) SELECT x FROM t"
    assert extract_sql(raw).startswith("WITH t AS")


def test_trailing_prose_is_dropped():
    raw = "```sql\nSELECT name\nFROM customer\n```"
    assert extract_sql(raw) == "SELECT name\nFROM customer"


def test_only_the_first_statement_is_kept():
    assert normalise_sql("SELECT 1; SELECT 2") == "SELECT 1"


def test_sql_comments_survive():
    assert "-- keep me" in normalise_sql("SELECT 1 -- keep me")


def test_reasoning_is_separable():
    raw = "We need a join on customer_id.\n\n```sql\nSELECT 1\n```"
    assert extract_reasoning(raw) == "We need a join on customer_id."
    assert extract_sql(raw) == "SELECT 1"
