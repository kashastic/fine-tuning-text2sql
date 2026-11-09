"""Value linking: does the question spell out a column's contents?"""

from __future__ import annotations

from pathlib import Path

from text2sql.values import (
    build_value_index,
    columns_matching_values,
    normalise,
    question_ngrams,
    value_index,
)


def test_normalise_meets_in_the_middle():
    """Values and question text get the same treatment, punctuation included."""
    assert normalise("K-12") == normalise("k 12") == "k 12"
    assert normalise("Continuation Schools") == normalise("continuation school")
    # "ss" endings are left alone, so these stay distinct words
    assert normalise("address") == "address"
    assert normalise("class") == "class"


def test_ngrams_come_back_longest_first_with_positions():
    grams = question_ngrams("los angeles county", max_words=3)
    # "angeles" singularises; "los" is too short for the rule to touch.
    assert grams[0] == ("los angele county", 0, 3)
    assert ("los angele", 0, 2) in grams
    assert ("county", 2, 1) in grams


def test_finds_the_column_whose_values_the_question_names(catalog_db):
    hits = columns_matching_values(str(catalog_db), "cards banned in gladiator format")
    assert "legalities.format" in hits
    assert "legalities.status" in hits  # 'Banned'


def test_a_longer_match_suppresses_a_shorter_coincidence(catalog_db):
    """"Black Lotus" is a card name; "Black" alone is a border colour.

    Without longest-match resolution the border column rides along on every
    question that happens to contain the word "black".
    """
    hits = columns_matching_values(str(catalog_db), "tell me about Black Lotus")
    assert "cards.name" in hits
    assert "cards.border" not in hits


def test_pure_numbers_and_stopword_values_are_not_indexed(catalog_db):
    index = build_value_index(str(catalog_db))
    assert "2015" not in index, "a bare year matches far too much"
    assert "none" not in index, "a placeholder value identifies no column"


def test_the_index_is_cached_on_disk_and_reused(catalog_db, tmp_path: Path):
    cache = tmp_path / "vcache"
    first = value_index(str(catalog_db), str(cache))
    files = list(cache.glob("*.json"))
    assert files, "the index should be written to the cache directory"

    # Corrupt the cache: a second call must read it rather than rebuild,
    # which is how we know the cache is actually on the hot path.
    value_index.cache_clear()
    files[0].write_text('{"zendikar": ["cards.flavor"]}', encoding="utf-8")
    second = value_index(str(catalog_db), str(cache))
    assert second == {"zendikar": ["cards.flavor"]}
    assert first != second
    value_index.cache_clear()


def test_an_unreadable_database_does_not_break_prompt_building(tmp_path: Path):
    """Value linking is an enhancement; it must never stop a prompt building."""
    missing = tmp_path / "nope.sqlite"
    assert columns_matching_values(str(missing), "anything") == set()
