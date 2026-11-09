"""A miniature BIRD-shaped benchmark on disk, so tests need no download.

Two tiny SQLite databases, a questions file in BIRD's format and a Spider one
in Spider's format. Everything in the repo that touches data can then be
exercised on a laptop with no GPU and no 40 GB of benchmark archives.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SHOP_DDL = """
CREATE TABLE customer (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT,
    signup_date TEXT
);
CREATE TABLE product (
    id INTEGER PRIMARY KEY,
    title TEXT,
    category TEXT,
    price REAL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    total REAL,
    FOREIGN KEY (customer_id) REFERENCES customer(id),
    FOREIGN KEY (product_id) REFERENCES product(id)
);
INSERT INTO customer VALUES
    (1, 'Ada',   'Zurich',   '2024-01-05'),
    (2, 'Linus', 'Helsinki', '2024-02-11'),
    (3, 'Grace', 'Zurich',   '2024-03-02');
INSERT INTO product VALUES
    (1, 'Keyboard', 'hardware', 80.0),
    (2, 'Mouse',    'hardware', 25.0),
    (3, 'Notebook', 'office',   4.5);
INSERT INTO orders VALUES
    (1, 1, 1, 1, 80.0),
    (2, 1, 3, 4, 18.0),
    (3, 2, 2, 2, 50.0),
    (4, 3, 3, 1, 4.5);
"""

LIBRARY_DDL = """
CREATE TABLE author (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE book (
    id INTEGER PRIMARY KEY, title TEXT, author_id INTEGER, year INTEGER,
    FOREIGN KEY (author_id) REFERENCES author(id)
);
INSERT INTO author VALUES (1, 'Lem', 'Poland'), (2, 'Borges', 'Argentina');
INSERT INTO book VALUES (1, 'Solaris', 1, 1961), (2, 'Ficciones', 2, 1944);
"""


# Shaped after BIRD's `card_games`, which is where the two sharpest schema
# bugs showed up: the join key is NOT the primary key (so protecting only
# primary and outbound foreign keys drops it), the table is wide enough for
# the column budget to bite, and `flavor` holds values a question would name
# without ever resembling the column's name.
CATALOG_DDL = """
CREATE TABLE cards (
    id INTEGER PRIMARY KEY,
    uuid TEXT,
    name TEXT,
    rarity TEXT,
    flavor TEXT,
    artist TEXT,
    layout TEXT,
    border TEXT,
    frame TEXT,
    watermark TEXT
);
CREATE TABLE legalities (
    id INTEGER PRIMARY KEY,
    uuid TEXT,
    format TEXT,
    status TEXT,
    FOREIGN KEY (uuid) REFERENCES cards(uuid)
);
INSERT INTO cards VALUES
    (1, 'aaa-111', 'Black Lotus', 'mythic', 'Zendikar', 'Chris Rahn',
     'normal', 'black', '2015', 'none'),
    (2, 'bbb-222', 'Shock', 'common', 'Dominaria', 'Jim Pavelec',
     'normal', 'black', '2015', 'none');
CREATE TABLE rulings (
    id INTEGER PRIMARY KEY,
    uuid TEXT,
    comment TEXT,
    FOREIGN KEY (uuid) REFERENCES cards(uuid)
);
INSERT INTO legalities VALUES
    (1, 'aaa-111', 'gladiator', 'Banned'),
    (2, 'bbb-222', 'commander', 'Legal');
INSERT INTO rulings VALUES
    (1, 'aaa-111', 'Taps for three mana'),
    (2, 'bbb-222', 'Deals two damage');
"""


@pytest.fixture(scope="session")
def catalog_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("catalog")
    return _build_db(root / "catalog" / "catalog.sqlite", CATALOG_DDL)


def _build_db(path: Path, ddl: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(ddl)
    conn.commit()
    conn.close()
    return path


@pytest.fixture(scope="session")
def shop_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dbs")
    return _build_db(root / "shop.sqlite", SHOP_DDL)


@pytest.fixture(scope="session")
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A `data/raw`-shaped tree holding a two-database BIRD and Spider split."""
    root = tmp_path_factory.mktemp("data_raw")

    for split in ("train", "dev"):
        split_dir = root / "bird" / split
        dbs = split_dir / f"{split}_databases"
        _build_db(dbs / "shop" / "shop.sqlite", SHOP_DDL)
        _build_db(dbs / "library" / "library.sqlite", LIBRARY_DDL)
        questions = [
            {
                "question_id": 0,
                "db_id": "shop",
                "question": "Which customer spent the most in total?",
                "evidence": "spent the most refers to MAX(SUM(orders.total))",
                "SQL": ("SELECT T1.name FROM customer AS T1 JOIN orders AS T2 "
                        "ON T2.customer_id = T1.id GROUP BY T1.id "
                        "ORDER BY SUM(T2.total) DESC LIMIT 1"),
                "difficulty": "moderate",
            },
            {
                "question_id": 1,
                "db_id": "shop",
                "question": "How many customers live in Zurich?",
                "evidence": "",
                "SQL": "SELECT COUNT(*) FROM customer WHERE city = 'Zurich'",
                "difficulty": "simple",
            },
            {
                "question_id": 2,
                "db_id": "library",
                "question": "List the titles of books written by Polish authors.",
                "evidence": "",
                "SQL": ("SELECT T1.title FROM book AS T1 JOIN author AS T2 "
                        "ON T1.author_id = T2.id WHERE T2.country = 'Poland'"),
                "difficulty": "challenging",
            },
            {
                "question_id": 3,
                "db_id": "shop",
                "question": "A question whose gold SQL is broken.",
                "evidence": "",
                "SQL": "SELECT nonexistent FROM customer",
                "difficulty": "simple",
            },
        ]
        (split_dir / f"{split}.json").write_text(json.dumps(questions), encoding="utf-8")

    spider = root / "spider"
    _build_db(spider / "database" / "shop" / "shop.sqlite", SHOP_DDL)
    spider_questions = [{
        "db_id": "shop",
        "question": "How many products are in the hardware category?",
        "query": "SELECT count(*) FROM product WHERE category = 'hardware'",
    }]
    (spider / "train_spider.json").write_text(json.dumps(spider_questions), encoding="utf-8")
    (spider / "dev.json").write_text(json.dumps(spider_questions), encoding="utf-8")
    # Spider's test databases live in `test_database/`, not `database/`, and
    # nothing but a separate directory here would catch a loader that sends
    # every split to the same place.
    _build_db(spider / "test_database" / "depot" / "depot.sqlite", SHOP_DDL)
    (spider / "test.json").write_text(json.dumps([{
        "db_id": "depot",
        "question": "How many products are in the hardware category?",
        "query": "SELECT count(*) FROM product WHERE category = 'hardware'",
    }]), encoding="utf-8")
    return root


@pytest.fixture
def shop_example(data_root: Path):
    from text2sql.data import load_dataset

    return load_dataset("bird", data_root=data_root, split="dev")[0]


# The tokenizer the real runs use. Only its files are needed (not the 4B of
# weights), and they are already in the HF cache on any machine that has run
# `make recall --tokenizer`. Nothing here downloads.
TOKENIZER_REPO = "Qwen/Qwen3-4B"


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A randomly-initialised Qwen3 small enough to train on a laptop CPU.

    The training scripts are the only part of this repo that a GPU-less
    machine cannot exercise, which is exactly why they were the only part
    never run: `--help` passing is not evidence that a config loads, that the
    LoRA target modules match the architecture, or that an adapter lands on
    disk. Those break in the first minute of a rented GPU, at $1.20/hour.

    Same architecture (`Qwen3ForCausalLM`) and the same tokenizer as the real
    model, so the chat template, the LoRA module names and the dataset
    plumbing are the real ones. Two layers and a 64-wide hidden state, so it
    is about 10M parameters instead of 4B. It is random, so it learns
    nothing -- this fixture is for wiring, never for quality.
    """
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    except Exception as exc:  # noqa: BLE001 - offline or uncached is a skip
        pytest.skip(f"{TOKENIZER_REPO} tokenizer not available locally: {exc}")

    config = transformers.Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
    )
    model = transformers.Qwen3ForCausalLM(config)
    path = tmp_path_factory.mktemp("tiny_qwen3")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path
