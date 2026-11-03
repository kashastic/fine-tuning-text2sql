"""GRPO reward functions.

The reward is the whole point of the RL stage: a query is right when running it
reproduces the gold result, and nothing else is worth much. Partial credit is
deliberately small -- large format or syntax bonuses teach the model to emit
safe, valid, wrong SQL, which is the classic way this stage goes nowhere.

Default weights:
    1.00  result set matches gold
    0.10  query runs at all
    0.05  a ```sql block was present and parseable

TRL's `GRPOTrainer` calls reward functions as
`fn(prompts, completions, **dataset_columns) -> list[float]`, so each public
function here takes that shape. `db_path` and `gold_sql` come straight from the
dataset columns written by `text2sql.prompts.to_sft_record`.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

from text2sql.postprocess import extract_sql
from text2sql.sql_exec import ComparisonMode, ExecStatus, compare_results, execute_sql


@dataclass(slots=True)
class RewardWeights:
    correct: float = 1.0
    executable: float = 0.10
    formatted: float = 0.05
    comparison: ComparisonMode = ComparisonMode.BIRD
    timeout: float = 15.0
    max_workers: int = 8


DEFAULT_WEIGHTS = RewardWeights()


def _completion_text(completion: Any) -> str:
    """TRL hands back either a string or a list of chat messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion or "")


def score_one(
    completion: Any,
    db_path: str,
    gold_sql: str,
    weights: RewardWeights = DEFAULT_WEIGHTS,
    gold_rows: tuple[tuple[Any, ...], ...] | None = None,
) -> float:
    """Reward for a single rollout."""
    text = _completion_text(completion)
    pred_sql = extract_sql(text)
    if not pred_sql:
        return 0.0

    reward = weights.formatted
    pred = execute_sql(db_path, pred_sql, timeout=weights.timeout)
    if pred.status is not ExecStatus.OK:
        return reward

    reward += weights.executable

    if gold_rows is None:
        gold = execute_sql(db_path, gold_sql, timeout=weights.timeout)
        if gold.status is not ExecStatus.OK:
            # Can't grade this example; don't punish the model for our data.
            return reward
        gold_rows = gold.rows

    if compare_results(pred.rows, gold_rows, weights.comparison):
        reward += weights.correct
    return reward


def execution_reward(
    prompts: Sequence[Any] | None = None,
    completions: Sequence[Any] | None = None,
    *,
    db_path: Sequence[str] | None = None,
    gold_sql: Sequence[str] | None = None,
    weights: RewardWeights = DEFAULT_WEIGHTS,
    **kwargs: Any,
) -> list[float]:
    """The reward function to hand to `GRPOTrainer(reward_funcs=[...])`."""
    del prompts, kwargs
    completions = list(completions or [])
    if not completions:
        return []
    if db_path is None or gold_sql is None:
        raise ValueError(
            "execution_reward needs `db_path` and `gold_sql` dataset columns; "
            "build the dataset with scripts/build_dataset.py"
        )

    jobs = list(zip(completions, db_path, gold_sql, strict=True))
    workers = min(weights.max_workers, max(1, len(jobs)), (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(lambda job: score_one(job[0], job[1], job[2], weights), jobs)
        )


def format_reward(
    prompts: Sequence[Any] | None = None,
    completions: Sequence[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Small separate signal for "produced one parseable ```sql block"."""
    del prompts, kwargs
    out = []
    for completion in completions or []:
        text = _completion_text(completion)
        out.append(1.0 if extract_sql(text) else 0.0)
    return out


# TRL reads `__name__` for the metric label in its logs.
execution_reward.__name__ = "execution_reward"
format_reward.__name__ = "format_reward"
