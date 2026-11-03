#!/usr/bin/env python3
"""FastAPI wrapper in front of a vLLM server: question in, SQL (and rows) out.

vLLM already exposes an OpenAI-compatible endpoint; what it does not do is the
part that makes this a text-to-SQL service -- read the database's schema, trim
it to the question, build the same prompt the model was trained on, pull the
SQL out of the answer, and optionally run it read-only.

    export T2S_MODEL=qwen3-4b-text2sql
    export T2S_VLLM_URL=http://localhost:8000/v1
    export T2S_DB_ROOT=data/raw/bird/dev/dev_databases
    uvicorn serve.app:app --host 0.0.0.0 --port 8080

    curl -s localhost:8080/sql -H 'content-type: application/json' -d '{
      "question": "How many schools in Alameda county are charter schools?",
      "db_id": "california_schools", "execute": true}' | jq
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.backends import OpenAICompatBackend  # noqa: E402
from text2sql.postprocess import extract_sql  # noqa: E402
from text2sql.prompts import build_prompt  # noqa: E402
from text2sql.data import Example  # noqa: E402
from text2sql.schema import database_schema, schema_for_example  # noqa: E402
from text2sql.sql_exec import ExecStatus, execute_sql  # noqa: E402

MODEL = os.getenv("T2S_MODEL", "qwen3-4b-text2sql")
VLLM_URL = os.getenv("T2S_VLLM_URL", "http://localhost:8000/v1")
DB_ROOT = Path(os.getenv("T2S_DB_ROOT", "data/raw/bird/dev/dev_databases"))
STYLE = os.getenv("T2S_STYLE", "sql")
MAX_TABLES = int(os.getenv("T2S_MAX_TABLES", "8"))
SAMPLE_VALUES = int(os.getenv("T2S_SAMPLE_VALUES", "3"))
MAX_NEW_TOKENS = int(os.getenv("T2S_MAX_NEW_TOKENS", "512"))
EXEC_TIMEOUT = float(os.getenv("T2S_EXEC_TIMEOUT", "15"))
MAX_ROWS = int(os.getenv("T2S_MAX_ROWS", "100"))

app = FastAPI(
    title="text2sql",
    version="0.1.0",
    description="Fine-tuned small model for text-to-SQL over SQLite databases.",
)
_backend: OpenAICompatBackend | None = None


def backend() -> OpenAICompatBackend:
    global _backend
    if _backend is None:
        _backend = OpenAICompatBackend(
            model=MODEL, base_url=VLLM_URL, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0
        )
    return _backend


def resolve_db(db_id: str | None, db_path: str | None) -> Path:
    if db_path:
        path = Path(db_path)
    elif db_id:
        # Guard against `../`: the id must name a directory directly under DB_ROOT.
        if "/" in db_id or "\\" in db_id or db_id.startswith("."):
            raise HTTPException(400, f"invalid db_id: {db_id!r}")
        path = DB_ROOT / db_id / f"{db_id}.sqlite"
    else:
        raise HTTPException(400, "give either db_id or db_path")
    if not path.exists():
        raise HTTPException(404, f"database not found: {path}")
    return path


class SQLRequest(BaseModel):
    question: str = Field(..., min_length=1)
    db_id: str | None = None
    db_path: str | None = None
    evidence: str = ""
    execute: bool = False
    max_rows: int = Field(default=MAX_ROWS, ge=1, le=1000)


class SQLResponse(BaseModel):
    sql: str
    db_id: str
    raw_output: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    executed: bool = False
    columns: list[str] | None = None
    rows: list[list[Any]] | None = None
    row_count: int | None = None
    exec_status: str | None = None
    exec_error: str | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL,
        "vllm_url": VLLM_URL,
        "db_root": str(DB_ROOT),
        "databases": sorted(p.name for p in DB_ROOT.glob("*") if p.is_dir())[:200]
        if DB_ROOT.exists() else [],
    }


@app.get("/schema/{db_id}")
def schema(db_id: str) -> dict[str, Any]:
    path = resolve_db(db_id, None)
    db = database_schema(str(path))
    return {
        "db_id": db.db_id,
        "tables": {
            name: {
                "columns": [{"name": c.name, "type": c.type, "pk": c.pk} for c in table.columns],
                "foreign_keys": [
                    {"column": col, "references": f"{ref_t}.{ref_c}"}
                    for col, ref_t, ref_c in table.foreign_keys
                ],
            }
            for name, table in db.tables.items()
        },
    }


@app.post("/sql", response_model=SQLResponse)
def generate_sql(request: SQLRequest) -> SQLResponse:
    path = resolve_db(request.db_id, request.db_path)
    example = Example(
        question_id="live",
        db_id=path.stem,
        question=request.question,
        gold_sql="",
        db_path=str(path),
        evidence=request.evidence,
    )

    started = time.perf_counter()
    schema_text = schema_for_example(
        str(path), request.question, request.evidence,
        max_tables=MAX_TABLES, n_sample_values=SAMPLE_VALUES,
    )
    prompt = build_prompt(example, style=STYLE, schema_text=schema_text)
    generations = backend().generate([prompt], n=1)[0]
    generation = generations[0]
    if generation.error:
        raise HTTPException(502, f"model backend failed: {generation.error}")

    sql = extract_sql(generation.text)
    response = SQLResponse(
        sql=sql,
        db_id=path.stem,
        raw_output=generation.text,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        prompt_tokens=generation.input_tokens,
        completion_tokens=generation.output_tokens,
    )
    if not sql:
        response.exec_status = ExecStatus.EMPTY.value
        response.exec_error = "no SQL found in the model output"
        return response

    if request.execute:
        result = execute_sql(path, sql, timeout=EXEC_TIMEOUT, max_rows=request.max_rows)
        response.executed = True
        response.exec_status = result.status.value
        response.exec_error = result.error or None
        if result.ok:
            response.rows = [list(r) for r in result.rows]
            response.row_count = len(result.rows)
    return response
