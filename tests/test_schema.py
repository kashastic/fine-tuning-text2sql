from __future__ import annotations

from text2sql.schema import (
    database_schema,
    format_schema,
    sample_values,
    schema_for_example,
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
