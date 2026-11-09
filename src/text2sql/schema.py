"""Build the schema text that goes into the prompt.

A BIRD database can have 80 tables; the whole DDL does not fit in a 4B model's
context alongside the question, and padding every example to the longest schema
wastes most of the training compute. So:

  * `database_schema` reads the real CREATE TABLE statements out of
    `sqlite_master` (no hand-rolled DDL -- the benchmark's own text is what the
    gold SQL was written against),
  * `select_relevant_tables` keeps the tables the question plausibly needs plus
    their foreign-key neighbours,
  * `select_relevant_columns` trims *within* a table, which is where BIRD's
    real cost is: `european_football_2.Match` has 115 columns, and keeping
    eight such tables whole puts a third of BIRD dev over a 3k prompt budget,
  * `format_schema` renders them, optionally with a few sample values per
    column, which is what lifts accuracy on value-matching questions.

Keys are never dropped. A trimmed table is marked in the rendered DDL with
the number of columns elided, so the model is told the schema is partial
rather than being left to infer that a column does not exist.
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
    # Columns of *this* table that other tables point at. A join needs both
    # ends, and only one end is a foreign key: `legalities.uuid REFERENCES
    # cards.uuid` makes `uuid` a key column of `cards` too, though `cards`
    # declares no foreign key and its primary key is `id`. Without this,
    # `cards.uuid` is scored as an ordinary column, loses the budget race,
    # and every question joining `cards` to `legalities` or `rulings` gets a
    # prompt in which the join cannot be written -- 59 of BIRD dev's
    # remaining recall losses were this one column.
    inbound_keys: set[str] = field(default_factory=set)


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

    # Second pass: a foreign key names a column in *another* table, so the
    # inbound side can only be filled in once every table has been read.
    for table in schema.tables.values():
        for _col, ref_table, ref_col in table.foreign_keys:
            target = schema.tables.get(ref_table)
            if target is None:
                continue
            if ref_col:
                target.inbound_keys.add(ref_col)
            else:
                # `REFERENCES parent` with no column list means the parent's
                # primary key; PRAGMA reports that target as NULL.
                target.inbound_keys.update(c.name for c in target.columns if c.pk)
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


def value_columns_for(table_name: str, value_hits: set[str]) -> set[str]:
    """The `{"table.column"}` hits that belong to one table, as column names."""
    prefix = f"{table_name}."
    return {h[len(prefix):] for h in value_hits if h.startswith(prefix)}


def _table_score(
    table: Table,
    question_tokens: set[str],
    value_columns: set[str] = frozenset(),
) -> float:
    """How likely is this table to be needed? Higher is better.

    Three signals, weighted by how much each is worth in practice:

    * the question **quotes this table's contents** (value linking) -- the
      strongest signal there is, because the question is naming data that
      only exists here. "Lucas di Grassi's Q1 result" contains no word
      resembling `drivers`, and no name-based score can reach that table.
    * a name overlap, exactly or by the same 4-character prefix rule the
      column tier uses. Exact-only scoring is why "race No. 901" missed the
      `races` table: `race != races` and the whole table was dropped, which
      is strictly worse than dropping a column.
    * a column-name overlap, worth less than a table-name one.
    """
    score = 0.0
    table_tokens = _name_tokens(table.name)
    if table_tokens & question_tokens:
        score += 2.0
    elif any(_tokens_match(t, q) for t in table_tokens for q in question_tokens):
        score += 1.0
    for col in table.columns:
        col_tokens = _name_tokens(col.name)
        if col_tokens & question_tokens:
            score += 1.0
        elif any(_tokens_match(t, q) for t in col_tokens for q in question_tokens):
            score += 0.5
        if col.name in value_columns:
            score += 3.0
    return score


def _tokens_match(a: str, b: str) -> bool:
    """Do two name tokens refer to the same thing?

    Exact equality misses the most common shape in BIRD's questions: the
    question inflects the word the column does not. "schools that were
    *opened* after 1991" needs `OpenDate`, whose tokens are {open, date}, and
    `opened != open` throws the column away. A shared prefix of four
    characters with similar length catches the inflections (open/opened,
    enroll/enrollment, test/testing) without collapsing genuinely different
    words, which a 3-character prefix would.
    """
    if a == b:
        return True
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 4:
        return False
    shared = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        shared += 1
    return shared >= 4


def _match_score(name: str, question_tokens: set[str]) -> int:
    """How strongly does a column name overlap the question? 0 means not at all."""
    score = 0
    for token in _name_tokens(name):
        if token in question_tokens:
            score += 2  # exact token, the stronger signal
        elif any(_tokens_match(token, q) for q in question_tokens):
            score += 1
    return score


def _phrase_hits(name: str, haystack: str) -> bool:
    """Does the question or evidence name this column outright?

    BIRD's evidence usually quotes columns verbatim, parentheses and all --
    "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment
    (K-12)`" -- and those names tokenise badly, so match the whole string.
    """
    return len(name) > 2 and name.lower() in haystack


def select_relevant_columns(
    table: Table,
    question: str,
    evidence: str = "",
    max_columns: int = 0,
    value_columns: set[str] = frozenset(),
) -> list[Column]:
    """Pick the columns worth showing for one table, in schema order.

    Three tiers, because measuring which needed columns got dropped showed
    two distinct causes (see docs/MEASUREMENTS.md, M4):

    1. **Keys** -- primary, outbound foreign, and *inbound* foreign (columns
       other tables point at) -- are never dropped. Without them a join stops
       being expressible, and an unanswerable prompt teaches GRPO nothing.
       The inbound half matters as much as the outbound: a join has two ends
       and only one of them declares the constraint.
    2. **Columns the question names -- by name or by content.** A column the
       question spells out verbatim is never dropped, and neither is one whose
       *values* the question quotes (`value_columns`, from
       `text2sql.values`). The second is the larger class: "the school run by
       Kacey Gibson" needs `AdmFName1`, and nothing about that question
       resembles that column's name.
       Verbatim naming alone These were the single largest recoverable loss: 54 of 165
       missing columns were *explicitly named in the question* and still
       thrown away, because the old code filled matches in schema order and
       ran out of budget. A name the user typed is the strongest signal
       available, so it outranks the budget rather than competing with it.
    3. **Everything else competes on match strength**, strongest first, and
       schema order only breaks ties. Filling in schema order was the other
       half of the bug: a weak match near the top of the table displaced a
       strong one further down.

    The budget is a floor rather than a ceiling: it stretches to fit tiers 1
    and 2, since a table whose keys and named columns alone exceed
    `max_columns` is better rendered wide than rendered useless.
    """
    if max_columns <= 0 or len(table.columns) <= max_columns:
        return list(table.columns)

    haystack = f"{question} {evidence}".lower()
    question_tokens = _tokens(f"{question} {evidence}")
    key_names = {col for col, _ref_table, _ref_col in table.foreign_keys}
    key_names |= table.inbound_keys

    required: set[str] = set()
    scored: list[tuple[int, int, str]] = []
    for position, col in enumerate(table.columns):
        if (col.pk or col.name in key_names
                or _phrase_hits(col.name, haystack)
                or col.name in value_columns):
            required.add(col.name)
            continue
        score = _match_score(col.name, question_tokens)
        # Negative score sorts strongest-first; position breaks ties in
        # schema order, so the choice stays deterministic across runs.
        scored.append((-score, position, col.name))

    kept = set(required)
    budget = max(max_columns, len(kept))
    for neg_score, _position, name in sorted(scored):
        if len(kept) >= budget:
            break
        del neg_score
        kept.add(name)

    return [col for col in table.columns if col.name in kept]


def select_relevant_tables(
    schema: DatabaseSchema,
    question: str,
    evidence: str = "",
    max_tables: int = 8,
    include_fk_neighbours: bool = True,
    value_hits: set[str] = frozenset(),
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
        ((_table_score(t, question_tokens, value_columns_for(name, value_hits)), name)
         for name, t in schema.tables.items()),
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


def render_table(table: Table, columns: list[Column]) -> str:
    """A CREATE TABLE for a subset of columns, with the foreign keys kept.

    Used only when columns were dropped; an untrimmed table keeps the
    benchmark's own DDL text, which is what the gold SQL was written
    against.
    """
    kept = {col.name for col in columns}
    lines = [
        f"  {_quote(col.name)} {col.type}".rstrip() + (" PRIMARY KEY" if col.pk else "")
        for col in columns
    ]
    for col, ref_table, ref_col in table.foreign_keys:
        if col not in kept:
            continue
        # PRAGMA foreign_key_list reports a NULL target for `REFERENCES
        # parent` with no column list, which means the parent's primary key.
        # Re-emit it in the same implicit form rather than inventing a name.
        target = f"{_quote(ref_table)} ({_quote(ref_col)})" if ref_col else _quote(ref_table)
        lines.append(f"  FOREIGN KEY ({_quote(col)}) REFERENCES {target}")
    elided = len(table.columns) - len(columns)
    body = ",\n".join(lines)
    block = f"CREATE TABLE {_quote(table.name)} (\n{body}\n);"
    if elided:
        block += f"\n-- {elided} further column(s) of {table.name} not shown"
    return block


def format_schema(
    schema: DatabaseSchema,
    tables: list[str] | None = None,
    db_path: str | None = None,
    n_sample_values: int = 0,
    use_raw_ddl: bool = True,
    question: str = "",
    evidence: str = "",
    max_columns: int = 0,
    value_hits: set[str] = frozenset(),
) -> str:
    """Render the schema as the prompt sees it."""
    names = tables if tables is not None else schema.table_names
    blocks: list[str] = []
    for name in names:
        table = schema.tables.get(name)
        if table is None:
            continue
        columns = select_relevant_columns(
            table, question, evidence, max_columns=max_columns,
            value_columns=value_columns_for(name, value_hits),
        )
        trimmed = len(columns) < len(table.columns)
        if use_raw_ddl and table.ddl and not trimmed:
            block = table.ddl.rstrip().rstrip(";") + ";"
        else:
            block = render_table(table, columns)
        if n_sample_values > 0 and db_path:
            lines = []
            for col in columns:
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
    max_columns: int = 0,
    link_values: bool = True,
) -> str:
    """The one call the prompt builder needs: trimmed, rendered schema text.

    `link_values` turns on value linking, which asks the database which
    columns actually contain the literals the question spells out. It costs
    one indexing pass per database, cached on disk; see `text2sql.values`.
    Pass False to get name-based selection only -- that is the comparison
    the measurement in docs/MEASUREMENTS.md (M17) is against, and the escape
    hatch if a database cannot be read.
    """
    schema = database_schema(db_path)
    value_hits: set[str] = frozenset()
    if link_values:
        # Imported here, not at module scope: text2sql.values reads schemas,
        # so a top-level import would be circular.
        from .values import columns_matching_values

        try:
            value_hits = columns_matching_values(db_path, question, evidence)
        except Exception:  # noqa: BLE001 - prompts must still build
            value_hits = frozenset()
    tables = select_relevant_tables(
        schema, question, evidence, max_tables=max_tables, value_hits=value_hits
    )
    return format_schema(
        schema,
        tables,
        db_path=db_path,
        n_sample_values=n_sample_values,
        question=question,
        evidence=evidence,
        max_columns=max_columns,
        value_hits=value_hits,
    )
