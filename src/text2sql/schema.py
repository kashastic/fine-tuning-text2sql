"""Build the schema text that goes into the prompt.

A BIRD database can have 80 tables; the whole DDL does not fit in a 4B model's
context alongside the question, and padding every example to the longest schema
wastes most of the training compute. So:

  * `database_schema` reads the real CREATE TABLE statements out of
    `sqlite_master` (no hand-rolled DDL -- the benchmark's own text is what the
    gold SQL was written against),
  * `select_relevant_tables` keeps the tables the question plausibly needs plus
    their foreign-key neighbours,
  * `format_schema` renders them, optionally with a few sample values per
    column, which is what lifts accuracy on value-matching questions.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_STOPWORDS = {
    "a", "all", "an", "and", "any", "are", "as", "at", "be", "been", "between",
    "both", "but", "by", "can", "did", "do", "does", "each", "every", "for",
    "from", "give", "had", "has", "have", "how", "in", "is", "it", "its",
    "list", "many", "me", "most", "much", "name", "names", "of", "on", "one",
    "only", "or", "please", "show", "that", "the", "their", "them", "there",
    "these", "they", "this", "those", "to", "total", "under", "was", "were",
    "what", "when", "where", "which", "who", "whose", "with", "number",
}


@dataclass(slots=True)
class Column:
    name: str
    type: str
    notnull: bool = False
    pk: bool = False


@dataclass(slots=True)
class Table:
    name: str
    ddl: str
    columns: list[Column] = field(default_factory=list)
    foreign_keys: list[tuple[str, str, str]] = field(default_factory=list)  # (col, ref_table, ref_col)


@dataclass(slots=True)
class DatabaseSchema:
    db_id: str
    tables: dict[str, Table] = field(default_factory=dict)

    @property
    def table_names(self) -> list[str]:
        return list(self.tables)


def _quote(name: str) -> str:
    """Quote an identifier, always.

    Quoting only when a name looks unusual is the bug that drops BIRD's
    `financial` database: its main table is called `order`, which is a
    reserved word, so `PRAGMA table_info(order)` is a syntax error even
    though the name is plain. There is no cheap test for "is this a
    keyword" -- SQLite has ~140 of them -- and quoting a name that did not
    need it is harmless, so quote everything and escape embedded quotes.
    """
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


@lru_cache(maxsize=256)
def database_schema(db_path: str) -> DatabaseSchema:
    """Read tables, columns and foreign keys out of a SQLite file. Cached."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(db_path)

    schema = DatabaseSchema(db_id=path.stem)
    uri = f"file:{path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for name, ddl in rows:
            table = Table(name=name, ddl=(ddl or "").strip())
            for cid, col_name, col_type, notnull, _default, pk in conn.execute(
                f"PRAGMA table_info({_quote(name)})"
            ):
                del cid, _default
                table.columns.append(
                    Column(name=col_name, type=(col_type or "").upper(),
                           notnull=bool(notnull), pk=bool(pk))
                )
            for fk in conn.execute(f"PRAGMA foreign_key_list({_quote(name)})"):
                # (id, seq, ref_table, from_col, to_col, on_update, on_delete, match)
                table.foreign_keys.append((fk[3], fk[2], fk[4]))
            schema.tables[name] = table
    return schema


def sample_values(db_path: str, table: str, column: str, limit: int = 3) -> list[str]:
    """A few distinct non-null values, for value-matching questions.

    The column is checked against the schema rather than left to fail at
    query time. SQLite resolves a double-quoted name it does not recognise
    as a *string literal* instead of raising, so `SELECT DISTINCT "nope"`
    cheerfully returns the word "nope" once per row -- which would land in
    the prompt as a sample value the database does not contain.
    """
    schema = database_schema(db_path)
    known = schema.tables.get(table)
    if known is None or column not in {c.name for c in known.columns}:
        return []
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            rows = conn.execute(
                f"SELECT DISTINCT {_quote(column)} FROM {_quote(table)} "
                f"WHERE {_quote(column)} IS NOT NULL LIMIT {int(limit)}"
            ).fetchall()
    except sqlite3.Error:
        return []
    out = []
    for (value,) in rows:
        text = str(value)
        out.append(text[:40] + "..." if len(text) > 40 else text)
    return out


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _name_tokens(name: str) -> set[str]:
    # account_number -> {account, number}; CustomerID -> {customer, id}
    parts = re.split(r"[_\s\-]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name))
    return {p.lower() for p in parts if p}


def _table_score(table: Table, question_tokens: set[str]) -> float:
    score = 0.0
    if _name_tokens(table.name) & question_tokens:
        score += 2.0
    for col in table.columns:
        if _name_tokens(col.name) & question_tokens:
            score += 1.0
    return score


def select_relevant_tables(
    schema: DatabaseSchema,
    question: str,
    evidence: str = "",
    max_tables: int = 8,
    include_fk_neighbours: bool = True,
) -> list[str]:
    """Pick the tables worth putting in the prompt, largest schemas first.

    Lexical overlap with the question and evidence, then one hop along foreign
    keys so joins stay expressible. Returns every table when the database is
    already small enough.
    """
    if max_tables <= 0 or len(schema.tables) <= max_tables:
        return schema.table_names

    question_tokens = _tokens(f"{question} {evidence}")
    scored = sorted(
        ((_table_score(t, question_tokens), name) for name, t in schema.tables.items()),
        key=lambda pair: (-pair[0], pair[1]),
    )
    kept = [name for score, name in scored if score > 0][:max_tables]
    if not kept:  # nothing matched -- fall back to the biggest tables by column count
        kept = sorted(schema.tables, key=lambda n: -len(schema.tables[n].columns))[:max_tables]

    if include_fk_neighbours:
        neighbours: list[str] = []
        for name in kept:
            for _col, ref_table, _ref_col in schema.tables[name].foreign_keys:
                if ref_table in schema.tables and ref_table not in kept:
                    neighbours.append(ref_table)
        for name in neighbours:
            if len(kept) >= max_tables:
                break
            if name not in kept:
                kept.append(name)

    # Keep the database's own table order so the prompt is stable across runs.
    order = {name: i for i, name in enumerate(schema.table_names)}
    return sorted(set(kept), key=lambda n: order[n])


def format_schema(
    schema: DatabaseSchema,
    tables: list[str] | None = None,
    db_path: str | None = None,
    n_sample_values: int = 0,
    use_raw_ddl: bool = True,
) -> str:
    """Render the schema as the prompt sees it."""
    names = tables if tables is not None else schema.table_names
    blocks: list[str] = []
    for name in names:
        table = schema.tables.get(name)
        if table is None:
            continue
        if use_raw_ddl and table.ddl:
            block = table.ddl.rstrip().rstrip(";") + ";"
        else:
            cols = ",\n".join(
                f"  {_quote(c.name)} {c.type}".rstrip() + (" PRIMARY KEY" if c.pk else "")
                for c in table.columns
            )
            block = f"CREATE TABLE {_quote(name)} (\n{cols}\n);"
        if n_sample_values > 0 and db_path:
            lines = []
            for col in table.columns:
                values = sample_values(db_path, name, col.name, n_sample_values)
                if values:
                    lines.append(f"-- {name}.{col.name}: " + ", ".join(values))
            if lines:
                block += "\n" + "\n".join(lines)
        blocks.append(block)
    return "\n\n".join(blocks)


def schema_for_example(
    db_path: str,
    question: str,
    evidence: str = "",
    max_tables: int = 8,
    n_sample_values: int = 0,
) -> str:
    """The one call the prompt builder needs: trimmed, rendered schema text."""
    schema = database_schema(db_path)
    tables = select_relevant_tables(schema, question, evidence, max_tables=max_tables)
    return format_schema(
        schema, tables, db_path=db_path, n_sample_values=n_sample_values
    )
