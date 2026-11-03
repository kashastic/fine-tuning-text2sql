"""Get a runnable query out of whatever the model actually said.

Every model drifts from the requested format sooner or later -- fenced blocks
without a language tag, a trailing explanation, `SELECT` with no fence at all,
repeated statements. Extraction has to be forgiving here, because the
alternative is scoring a formatting slip as a wrong answer.
"""

from __future__ import annotations

import re

_FENCE_SQL = re.compile(r"```[ \t]*sql[ \t]*\r?\n(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE)
_FENCE_ANY = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\r?\n(.*?)(?:```|\Z)", re.DOTALL)
_BARE_SELECT = re.compile(r"\b(WITH|SELECT)\b.*", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Return the SQL in `text`, or an empty string when there is none."""
    if not text:
        return ""

    # Prefer the last ```sql block: reasoning styles sometimes show a draft first.
    blocks = _FENCE_SQL.findall(text)
    if not blocks:
        blocks = [b for b in _FENCE_ANY.findall(text) if _BARE_SELECT.search(b)]
    candidate = blocks[-1] if blocks else ""

    if not candidate:
        match = _BARE_SELECT.search(text)
        candidate = match.group(0) if match else ""

    return normalise_sql(candidate)


def normalise_sql(sql: str) -> str:
    """Collapse whitespace, drop trailing prose and keep one statement."""
    if not sql:
        return ""
    sql = sql.strip()
    sql = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", sql)
    sql = re.sub(r"```\s*$", "", sql).strip()

    # Cut at the first statement boundary; anything after it is a second query
    # or an explanation the model appended.
    if ";" in sql:
        sql = sql.split(";", 1)[0]

    # Drop a trailing explanation that starts on its own line in prose.
    lines = sql.splitlines()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if kept and stripped and _looks_like_prose(stripped):
            break
        kept.append(line)
    sql = "\n".join(kept)

    sql = re.sub(r"[ \t]+", " ", sql)
    sql = re.sub(r"\n{2,}", "\n", sql)
    return sql.strip().rstrip(";").strip()


_PROSE_STARTS = (
    "this query", "the query", "note:", "explanation", "here", "we ", "i ",
    "that returns", "it returns",
)


def _looks_like_prose(line: str) -> bool:
    low = line.lower()
    if low.startswith("--"):
        return False
    return low.startswith(_PROSE_STARTS)


def extract_reasoning(text: str) -> str:
    """Whatever the model said before its SQL block (may be empty)."""
    if not text:
        return ""
    match = _FENCE_SQL.search(text) or _FENCE_ANY.search(text)
    return text[: match.start()].strip() if match else ""
