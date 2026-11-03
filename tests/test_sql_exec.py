from __future__ import annotations

from text2sql.sql_exec import (
    ComparisonMode,
    ExecStatus,
    compare_results,
    execute_sql,
    execution_match,
    has_order_by,
    is_read_only,
)


def test_select_runs(shop_db):
    result = execute_sql(shop_db, "SELECT name FROM customer ORDER BY id")
    assert result.ok
    assert result.rows == (("Ada",), ("Linus",), ("Grace",))


def test_writes_are_rejected(shop_db):
    for sql in ("DROP TABLE customer", "DELETE FROM customer", "UPDATE customer SET name='x'",
                "INSERT INTO customer VALUES (9,'x','y','z')", "PRAGMA table_info(customer)"):
        assert execute_sql(shop_db, sql).status is ExecStatus.REJECTED, sql


def test_stacked_statements_are_rejected(shop_db):
    assert not is_read_only("SELECT 1; DROP TABLE customer")
    assert execute_sql(shop_db, "SELECT 1; DROP TABLE customer").status is ExecStatus.REJECTED


def test_a_write_hidden_in_a_comment_does_not_block_a_select(shop_db):
    # Comments are stripped before the write check, so this is a plain SELECT.
    result = execute_sql(shop_db, "SELECT name FROM customer -- DROP TABLE customer")
    assert result.ok


def test_bad_column_is_an_error_not_a_crash(shop_db):
    result = execute_sql(shop_db, "SELECT nope FROM customer")
    assert result.status is ExecStatus.ERROR
    assert "nope" in result.error


def test_empty_sql(shop_db):
    assert execute_sql(shop_db, "   ").status is ExecStatus.EMPTY


def test_missing_database_is_reported():
    assert execute_sql("/nope/missing.sqlite", "SELECT 1").status is ExecStatus.ERROR


def test_runaway_query_times_out(shop_db):
    # A recursive CTE with no bound: without the progress handler this hangs.
    sql = ("WITH RECURSIVE forever(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM forever) "
           "SELECT COUNT(*) FROM forever")
    result = execute_sql(shop_db, sql, timeout=1.0)
    assert result.status is ExecStatus.TIMEOUT
    assert result.seconds < 10


def test_comparison_modes():
    a = [(1, "x"), (2, "y")]
    b = [(2, "y"), (1, "x")]
    assert compare_results(a, b, ComparisonMode.BIRD)
    assert compare_results(a, b, ComparisonMode.MULTISET)
    assert not compare_results(a, b, ComparisonMode.ORDERED)


def test_bird_mode_ignores_duplicate_rows():
    # Set semantics is the official BIRD metric, duplicates and all.
    assert compare_results([(1,), (1,)], [(1,)], ComparisonMode.BIRD)
    assert not compare_results([(1,), (1,)], [(1,)], ComparisonMode.MULTISET)


def test_float_tolerance():
    assert compare_results([(0.1 + 0.2,)], [(0.3,)])


def test_int_and_float_compare_equal():
    assert compare_results([(80,)], [(80.0,)])


def test_execution_match_on_equivalent_queries(shop_db):
    gold = ("SELECT T1.name FROM customer AS T1 JOIN orders AS T2 ON T2.customer_id = T1.id "
            "GROUP BY T1.id ORDER BY SUM(T2.total) DESC LIMIT 1")
    pred = ("SELECT c.name FROM orders o JOIN customer c ON c.id = o.customer_id "
            "GROUP BY c.id ORDER BY SUM(o.total) DESC LIMIT 1")
    assert execution_match(shop_db, pred, gold).correct


def test_execution_match_is_false_when_gold_is_broken(shop_db):
    outcome = execution_match(shop_db, "SELECT 1", "SELECT broken FROM customer")
    assert not outcome.correct
    assert outcome.gold.status is ExecStatus.ERROR


def test_order_sensitivity_is_opt_in(shop_db):
    gold = "SELECT name FROM customer ORDER BY id"
    pred = "SELECT name FROM customer ORDER BY id DESC"
    assert has_order_by(gold)
    assert execution_match(shop_db, pred, gold).correct
    assert not execution_match(
        shop_db, pred, gold, order_sensitive_when_gold_orders=True
    ).correct
