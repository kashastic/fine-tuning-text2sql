"""Value linking: find the columns whose *contents* the question names.

Schema trimming scores columns on how much their **name** overlaps the
question, which cannot see the largest remaining class of failure. "the school
run by Kacey Gibson" needs `schools.AdmFName1`, and the question contains no
word resembling "AdmFName1"; "continuation schools" needs
`frpm.Educational Option Type`, whose *value* is "Continuation School". The
signal is in the data, not in the schema.

So: index each database's distinct text values once, then look the question's
word n-grams up in that index. A hit means the question spells a value that
this column actually contains, which is a far stronger signal than any name
overlap -- the question is quoting the database.

    cols = columns_matching_values(db_path, "schools run by Kacey Gibson")
    # -> {"schools.AdmFName1", "schools.AdmLName1"}

Three things keep this affordable, because BIRD's training databases are 30 GB
and the index is built per database rather than per question:

  * only TEXT-affinity columns, and only short values -- a column of forum
    posts cannot be a value the question names in an equality, and indexing
    it would dominate both time and size,
  * a cap on distinct values read per column, so one huge column cannot
    stall the build,
  * the result is cached on disk, keyed by the database's size and mtime, so
    a rebuild pays nothing.

The index deliberately stores whole values, not value tokens. Matching whole
values against question n-grams keeps precision high: "Continuation School"
identifies one column, whereas the token "school" appears in the contents of
half of `california_schools`.
"""

from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

from .schema import database_schema

# Values this common say nothing about which column is meant. Kept separate
# from the schema module's stopwords, which are tuned for *names*.
_VALUE_STOPWORDS = {
    "yes", "no", "true", "false", "none", "null", "n/a", "na", "other",
    "others", "unknown", "all", "and", "the", "not", "nan",
}

# Tuning. These are the knobs the measurement in docs/MEASUREMENTS.md (M17)
# was taken with; they trade index build time against recall.
MAX_DISTINCT_PER_COLUMN = 20_000
MAX_VALUE_CHARS = 48
MAX_NGRAM_WORDS = 6
MAX_COLUMNS_PER_VALUE = 8

CACHE_VERSION = 1
DEFAULT_CACHE_DIR = Path("data/cache/values")

_WORD = re.compile(r"[A-Za-z0-9]+")


def _stem(word: str) -> str:
    """Strip one trailing plural "s", so questions and values meet.

    This is a hash key, not display text, so it only has to be *consistent*:
    "angeles" and "angele" would both be fine as long as both sides agree.
    What it has to fix is real and common -- the question says "continuation
    schools" and the column contains "Continuation School", and without this
    those never meet. "ss" is left alone so "address" and "class" survive.
    """
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def normalise(text: str) -> str:
    """Lowercased, singularised words joined by single spaces.

    Applied identically to stored values and to question n-grams, so
    "K-12", "k 12" and "K 12" all meet in the middle and punctuation stops
    mattering on either side.
    """
    return " ".join(_stem(w.lower()) for w in _WORD.findall(text))


def _is_indexable(value: str) -> bool:
    if len(value) > MAX_VALUE_CHARS:
        return False
    norm = normalise(value)
    if not norm or norm in _VALUE_STOPWORDS:
        return False
    # Pure numbers match far too much: every year, count and id in every
    # question would light up unrelated columns.
    if norm.replace(" ", "").isdigit():
        return False
    # A single short token is usually a code or an abbreviation shared by
    # many columns; two characters is not evidence.
    return len(norm) >= 3


def _text_columns(db_path: str) -> list[tuple[str, str]]:
    """(table, column) pairs worth indexing: TEXT affinity, by declared type."""
    schema = database_schema(db_path)
    out: list[tuple[str, str]] = []
    for name, table in schema.tables.items():
        for col in table.columns:
            declared = (col.type or "").upper()
            # SQLite type affinity: anything containing CHAR, CLOB or TEXT is
            # TEXT. An empty declared type means BLOB affinity, but BIRD and
            # Spider both carry untyped text columns, so include those too.
            if not declared or any(k in declared for k in ("CHAR", "CLOB", "TEXT")):
                out.append((name, col.name))
    return out


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_value_index(db_path: str) -> dict[str, list[str]]:
    """normalised value -> ["table.column", ...]. Reads the database."""
    index: dict[str, set[str]] = {}
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        for table, column in _text_columns(db_path):
            try:
                rows = conn.execute(
                    f"SELECT DISTINCT {_quote(column)} FROM {_quote(table)} "
                    f"WHERE {_quote(column)} IS NOT NULL "
                    f"AND length({_quote(column)}) <= {MAX_VALUE_CHARS} "
                    f"LIMIT {MAX_DISTINCT_PER_COLUMN}"
                ).fetchall()
            except sqlite3.Error:
                # A corrupt or oddly-typed column is not worth failing a
                # whole dataset build over.
                continue
            key = f"{table}.{column}"
            for (value,) in rows:
                if value is None:
                    continue
                text = str(value)
                if not _is_indexable(text):
                    continue
                index.setdefault(normalise(text), set()).add(key)

    # A value found in many columns identifies none of them.
    return {
        value: sorted(cols)
        for value, cols in index.items()
        if len(cols) <= MAX_COLUMNS_PER_VALUE
    }


def _cache_path(db_path: str, cache_dir: Path) -> Path:
    stat = Path(db_path).stat()
    stem = Path(db_path).stem
    # Size and mtime together are enough to notice a database being replaced,
    # without hashing hundreds of megabytes on every call.
    token = f"{stat.st_size}-{int(stat.st_mtime)}-{CACHE_VERSION}"
    return cache_dir / f"{stem}.{token}.json"


@lru_cache(maxsize=64)
def value_index(db_path: str, cache_dir: str | None = None) -> dict[str, list[str]]:
    """The value index for one database, from disk cache when possible."""
    directory = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    path = _cache_path(db_path, directory)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass  # a truncated cache file is not a reason to fail

    index = build_value_index(db_path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a killed build cannot leave a half-written
        # cache that the next run would read as truth.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index), encoding="utf-8")
        tmp.replace(path)
        for stale in directory.glob(f"{Path(db_path).stem}.*.json"):
            if stale != path:
                stale.unlink(missing_ok=True)
    except OSError:
        pass  # an unwritable cache directory only costs speed
    return index


def question_ngrams(
    text: str, max_words: int = MAX_NGRAM_WORDS
) -> list[tuple[str, int, int]]:
    """Every normalised 1..n word window as (gram, start, size), longest first.

    Positions come back because the caller resolves overlaps: the longest
    match over a span of the question is the specific one, and shorter
    matches inside it are coincidences.
    """
    words = [_stem(w.lower()) for w in _WORD.findall(text)]
    grams: list[tuple[str, int, int]] = []
    for size in range(min(max_words, len(words)), 0, -1):
        for start in range(len(words) - size + 1):
            grams.append((" ".join(words[start : start + size]), start, size))
    return grams


def columns_matching_values(
    db_path: str,
    question: str,
    evidence: str = "",
    cache_dir: str | None = None,
) -> set[str]:
    """`{"table.column"}` whose contents the question or evidence spells out.

    Overlapping matches resolve longest-first, and a window whose words are
    already fully claimed by a longer match is skipped. Without that,
    "Los Angeles" matches `schools.City` *and* drags in `schools.AdmLName1`,
    because "Angeles" on its own is somebody's surname. The longer match is
    the one the question meant.
    """
    try:
        index = value_index(db_path, cache_dir)
    except (OSError, sqlite3.Error):
        return set()
    if not index:
        return set()
    hits: set[str] = set()
    claimed: set[int] = set()
    for gram, start, size in question_ngrams(f"{question} {evidence}"):
        keys = index.get(gram)
        if not keys:
            continue
        span = set(range(start, start + size))
        if span <= claimed:
            continue
        claimed |= span
        hits.update(keys)
    return hits
