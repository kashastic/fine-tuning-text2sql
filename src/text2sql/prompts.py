"""One prompt format, used by every model in the comparison.

The fairness rule from the plan: the API baseline gets the same schema text,
the same evidence and the same instructions as the fine-tuned model. That only
holds if a single function builds the prompt, so this is it.

Two output styles:
    "sql"        -- answer is a single ```sql block (what SFT variant A trains)
    "reasoning"  -- a few lines of reasoning, then the ```sql block (variant B)
"""

from __future__ import annotations

from typing import Literal

from text2sql.data import Example
from text2sql.schema import schema_for_example

OutputStyle = Literal["sql", "reasoning"]

SYSTEM_SQL = (
    "You are an expert data analyst who writes SQLite queries.\n"
    "Given a database schema and a question, reply with one SQLite query that "
    "answers it.\n"
    "Rules:\n"
    "- Use only the tables and columns in the schema.\n"
    "- Return exactly the columns the question asks for, nothing extra.\n"
    "- Reply with a single ```sql code block and no other text."
)

SYSTEM_REASONING = (
    "You are an expert data analyst who writes SQLite queries.\n"
    "Given a database schema and a question, work out which tables, joins and "
    "filters are needed, then write one SQLite query.\n"
    "Rules:\n"
    "- Use only the tables and columns in the schema.\n"
    "- Return exactly the columns the question asks for, nothing extra.\n"
    "- Keep the reasoning under five short lines, then give a single ```sql "
    "code block as the last thing in your reply."
)


def system_prompt(style: OutputStyle = "sql") -> str:
    return SYSTEM_REASONING if style == "reasoning" else SYSTEM_SQL


def user_prompt(schema_text: str, question: str, evidence: str = "") -> str:
    parts = ["Database schema:", "", schema_text, ""]
    if evidence.strip():
        parts += [f"Evidence: {evidence.strip()}", ""]
    parts += [f"Question: {question.strip()}"]
    return "\n".join(parts)


def build_prompt(
    example: Example,
    style: OutputStyle = "sql",
    max_tables: int = 8,
    n_sample_values: int = 0,
    schema_text: str | None = None,
) -> list[dict[str, str]]:
    """Chat messages for one example. `schema_text` overrides schema lookup."""
    if schema_text is None:
        schema_text = schema_for_example(
            example.db_path,
            example.question,
            example.evidence,
            max_tables=max_tables,
            n_sample_values=n_sample_values,
        )
    return [
        {"role": "system", "content": system_prompt(style)},
        {"role": "user", "content": user_prompt(schema_text, example.question, example.evidence)},
    ]


def target_completion(example: Example, reasoning: str = "") -> str:
    """The assistant turn SFT trains on."""
    sql = example.gold_sql.strip().rstrip(";")
    block = f"```sql\n{sql}\n```"
    if reasoning.strip():
        return f"{reasoning.strip()}\n\n{block}"
    return block


def to_sft_record(
    example: Example,
    style: OutputStyle = "sql",
    reasoning: str = "",
    max_tables: int = 8,
    n_sample_values: int = 0,
    schema_text: str | None = None,
) -> dict[str, object]:
    """A chat-format training row: `{"messages": [...]}` plus metadata.

    TRL's SFTTrainer and GRPOTrainer both accept this shape and apply the
    model's own chat template, so nothing here is Qwen-specific.
    """
    messages = build_prompt(
        example, style=style, max_tables=max_tables,
        n_sample_values=n_sample_values, schema_text=schema_text,
    )
    return {
        "messages": messages + [{"role": "assistant", "content": target_completion(example, reasoning)}],
        "prompt": messages,
        "question_id": example.question_id,
        "db_id": example.db_id,
        "db_path": example.db_path,
        "gold_sql": example.gold_sql,
        "source": example.source,
        "difficulty": example.difficulty,
    }
