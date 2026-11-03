from __future__ import annotations

import pytest

from text2sql.rewards import DEFAULT_WEIGHTS, RewardWeights, execution_reward, format_reward, score_one

GOLD = ("SELECT T1.name FROM customer AS T1 JOIN orders AS T2 ON T2.customer_id = T1.id "
        "GROUP BY T1.id ORDER BY SUM(T2.total) DESC LIMIT 1")


def fence(sql: str) -> str:
    return f"```sql\n{sql}\n```"


def test_reward_ladder(shop_db):
    db = str(shop_db)
    correct = score_one(fence(GOLD), db, GOLD)
    equivalent = score_one(
        fence("SELECT c.name FROM orders o JOIN customer c ON c.id = o.customer_id "
              "GROUP BY c.id ORDER BY SUM(o.total) DESC LIMIT 1"), db, GOLD)
    runs_but_wrong = score_one(fence("SELECT name FROM customer"), db, GOLD)
    does_not_run = score_one(fence("SELECT nope FROM customer"), db, GOLD)
    no_sql = score_one("I cannot answer that.", db, GOLD)

    assert correct == pytest.approx(1.15)
    assert equivalent == pytest.approx(1.15)
    assert runs_but_wrong == pytest.approx(0.15)
    assert does_not_run == pytest.approx(0.05)
    assert no_sql == 0.0
    # Being right must dominate: valid-but-wrong SQL earns under a sixth of it.
    assert runs_but_wrong < correct / 6


def test_reward_handles_chat_shaped_completions(shop_db):
    completion = [{"role": "assistant", "content": fence(GOLD)}]
    assert score_one(completion, str(shop_db), GOLD) == pytest.approx(1.15)


def test_unexecutable_gold_does_not_punish_the_model(shop_db):
    # Our own broken gold query must not teach the model that correct SQL is bad.
    reward = score_one(fence("SELECT name FROM customer"), str(shop_db), "SELECT broken FROM x")
    assert reward == pytest.approx(DEFAULT_WEIGHTS.formatted + DEFAULT_WEIGHTS.executable)


def test_timeout_is_not_fatal(shop_db):
    runaway = ("WITH RECURSIVE forever(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM forever) "
               "SELECT COUNT(*) FROM forever")
    weights = RewardWeights(timeout=1.0)
    assert score_one(fence(runaway), str(shop_db), GOLD, weights) == weights.formatted


def test_batch_interface_matches_trl(shop_db):
    db = str(shop_db)
    completions = [fence(GOLD), fence("SELECT name FROM customer"), "nothing"]
    rewards = execution_reward(
        prompts=[None] * 3, completions=completions,
        db_path=[db] * 3, gold_sql=[GOLD] * 3,
    )
    assert rewards == pytest.approx([1.15, 0.15, 0.0])


def test_batch_requires_the_dataset_columns():
    try:
        execution_reward(completions=["x"])
    except ValueError as exc:
        assert "db_path" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError naming the missing columns")


def test_format_reward():
    assert format_reward(completions=[fence("SELECT 1"), "no sql"]) == [1.0, 0.0]
