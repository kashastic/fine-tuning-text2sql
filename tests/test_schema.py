from __future__ import annotations

import sqlite3

from text2sql.schema import (
    database_schema,
    format_schema,
    sample_values,
    schema_for_example,
    select_relevant_columns,
    select_relevant_tables,
)


def test_reads_tables_columns_and_foreign_keys(shop_db):
    schema = database_schema(str(shop_db))
    assert set(schema.table_names) == {"customer", "product", "orders"}
    assert [c.name for c in schema.tables["customer"].columns][:2] == ["id", "name"]
    assert schema.tables["customer"].columns[0].pk
    fks = schema.tables["orders"].foreign_keys
    assert ("customer_id", "customer", "id") in fks


def test_sample_values(shop_db):
    values = sample_values(str(shop_db), "customer", "city", limit=3)
    assert "Zurich" in values


def test_sample_values_on_a_bad_column_is_empty_not_an_error(shop_db):
    assert sample_values(str(shop_db), "customer", "nope") == []


def test_small_schemas_are_kept_whole(shop_db):
    schema = database_schema(str(shop_db))
    assert select_relevant_tables(schema, "anything", max_tables=8) == schema.table_names


def test_trimming_keeps_the_question_s_tables_and_their_fk_neighbours(shop_db):
    schema = database_schema(str(shop_db))
    kept = select_relevant_tables(schema, "how many orders per customer?", max_tables=2)
    assert len(kept) <= 2
    assert "orders" in kept


def test_trimming_falls_back_when_nothing_matches(shop_db):
    schema = database_schema(str(shop_db))
    kept = select_relevant_tables(schema, "zzzz qqqq", max_tables=1)
    assert len(kept) == 1


def test_format_schema_emits_ddl_and_sample_comments(shop_db):
    schema = database_schema(str(shop_db))
    text = format_schema(schema, ["customer"], db_path=str(shop_db), n_sample_values=2)
    assert "CREATE TABLE customer" in text
    assert "-- customer.city:" in text


def test_schema_for_example_is_deterministic(shop_db):
    first = schema_for_example(str(shop_db), "which customers ordered a keyboard?")
    second = schema_for_example(str(shop_db), "which customers ordered a keyboard?")
    assert first == second
    assert "CREATE TABLE" in first


def test_reserved_word_table_and_column_names_are_readable(tmp_path):
    """BIRD's `financial` database has a table called `order`.

    An unquoted `PRAGMA table_info(order)` is a syntax error, which used to
    drop all 106 `financial` questions from BIRD dev at dataset-build time.
    """
    db = tmp_path / "reserved.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE "order" (order_id INTEGER PRIMARY KEY, "group" TEXT, "select" REAL)')
        conn.execute('INSERT INTO "order" VALUES (1, \'retail\', 9.5)')

    schema = database_schema(str(db))
    assert "order" in schema.tables
    assert [c.name for c in schema.tables["order"].columns] == ["order_id", "group", "select"]
    assert sample_values(str(db), "order", "group") == ["retail"]

    ddl = format_schema(schema, tables=["order"], db_path=str(db), n_sample_values=1)
    assert "order" in ddl


def test_sample_values_rejects_an_unknown_column_instead_of_echoing_it(tmp_path):
    """SQLite resolves an unknown double-quoted name as a string literal.

    So `SELECT DISTINCT "nope" FROM customer` returns "nope" per row rather
    than raising, which would put a value the database does not hold into
    the prompt.
    """
    db = tmp_path / "dqs.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE customer (city TEXT)")
        conn.execute("INSERT INTO customer VALUES ('Zurich')")

    assert sample_values(str(db), "customer", "nope") == []
    assert sample_values(str(db), "nosuchtable", "city") == []
    assert sample_values(str(db), "customer", "city") == ["Zurich"]


def _wide_db(tmp_path):
    db = tmp_path / "wide.sqlite"
    extras = ", ".join(f"filler_{i} TEXT" for i in range(40))
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE team (team_id INTEGER PRIMARY KEY, team_name TEXT)")
        conn.execute(
            "CREATE TABLE match (match_id INTEGER PRIMARY KEY, team_id INTEGER, "
            f"home_goals INTEGER, {extras}, "
            "FOREIGN KEY (team_id) REFERENCES team (team_id))"
        )
    return db


def test_column_trimming_keeps_keys_and_question_columns(tmp_path):
    schema = database_schema(str(_wide_db(tmp_path)))
    kept = select_relevant_columns(
        schema.tables["match"], "how many home goals", max_columns=6
    )
    names = [c.name for c in kept]
    assert len(names) == 6
    assert "match_id" in names, "primary key dropped"
    assert "team_id" in names, "foreign key dropped -- the join stops being expressible"
    assert "home_goals" in names, "column named in the question dropped"


def test_column_trimming_matches_a_name_quoted_in_the_evidence(tmp_path):
    """BIRD evidence quotes columns verbatim, and those names tokenise badly."""
    db = tmp_path / "frpm.sqlite"
    extras = ", ".join(f'"col {i}" TEXT' for i in range(30))
    with sqlite3.connect(db) as conn:
        conn.execute(
            'CREATE TABLE frpm (id INTEGER PRIMARY KEY, '
            '"Free Meal Count (K-12)" REAL, "Enrollment (K-12)" REAL, ' + extras + ")"
        )
    schema = database_schema(str(db))
    kept = select_relevant_columns(
        schema.tables["frpm"],
        "What is the highest eligible free rate?",
        evidence="free rate = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
        max_columns=5,
    )
    names = [c.name for c in kept]
    assert "Free Meal Count (K-12)" in names
    assert "Enrollment (K-12)" in names


def test_untrimmed_tables_keep_the_benchmarks_own_ddl(tmp_path):
    schema = database_schema(str(_wide_db(tmp_path)))
    whole = format_schema(schema, tables=["team"], max_columns=20)
    assert "CREATE TABLE team" in whole
    assert "not shown" not in whole


def test_a_trimmed_table_says_so_and_keeps_its_foreign_key(tmp_path):
    schema = database_schema(str(_wide_db(tmp_path)))
    text = format_schema(
        schema, tables=["match"], question="home goals", max_columns=5
    )
    assert "further column(s) of match not shown" in text
    assert "FOREIGN KEY" in text
    assert "filler_39" not in text


def test_zero_budget_keeps_every_column(tmp_path):
    schema = database_schema(str(_wide_db(tmp_path)))
    kept = select_relevant_columns(schema.tables["match"], "anything", max_columns=0)
    assert len(kept) == len(schema.tables["match"].columns)


def test_implicit_foreign_key_target_renders_without_a_column_list(tmp_path):
    """`REFERENCES parent` with no column list reports a NULL target column."""
    db = tmp_path / "implicit.sqlite"
    extras = ", ".join(f"filler_{i} TEXT" for i in range(30))
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE account (account_id INTEGER PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE trans (trans_id INTEGER PRIMARY KEY, account_id INTEGER "
            f"REFERENCES account, {extras})"
        )
    schema = database_schema(str(db))
    text = format_schema(schema, tables=["trans"], question="account", max_columns=4)
    assert "FOREIGN KEY" in text
    assert "REFERENCES" in text
    assert "None" not in text
