#!/usr/bin/env python3
"""Small demo: pick a database, ask a question, see the SQL and the rows.

    python serve/demo_gradio.py --api http://localhost:8080 --share

Talks to the FastAPI service in `serve/app.py`, so the demo exercises the same
path a caller would.
"""

from __future__ import annotations

import argparse

import gradio as gr
import httpx

EXAMPLES = [
    ["How many schools in Alameda county are charter schools?", "california_schools"],
    ["What is the highest eligible free rate for K-12 students in Alameda county?", "california_schools"],
]


def build_ui(api: str) -> gr.Blocks:
    client = httpx.Client(base_url=api, timeout=120.0)

    def databases() -> list[str]:
        try:
            return client.get("/health").json().get("databases", [])
        except Exception:  # noqa: BLE001 - the dropdown should not crash the UI
            return []

    def ask(question: str, db_id: str, evidence: str, execute: bool):
        if not question.strip():
            return "", "", "Ask a question first."
        try:
            response = client.post("/sql", json={
                "question": question, "db_id": db_id,
                "evidence": evidence, "execute": execute,
            })
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            return "", "", f"Request failed: {exc}"

        sql = body.get("sql") or "(the model produced no SQL)"
        note = f"{body.get('latency_ms', 0):.0f} ms"
        if body.get("exec_status") and body["exec_status"] != "ok":
            note += f" · execution: {body['exec_status']} {body.get('exec_error') or ''}"
        rows = body.get("rows")
        if not rows:
            return sql, "", note
        table = "| " + " | ".join(f"c{i}" for i in range(len(rows[0]))) + " |\n"
        table += "| " + " | ".join("---" for _ in rows[0]) + " |\n"
        for row in rows[:50]:
            table += "| " + " | ".join("" if v is None else str(v) for v in row) + " |\n"
        if body.get("row_count", 0) > 50:
            table += f"\n_{body['row_count']} rows, showing 50._"
        return sql, table, note

    with gr.Blocks(title="Text-to-SQL") as ui:
        gr.Markdown(
            "# Text-to-SQL\nA fine-tuned 4B model writing SQLite queries over "
            "the BIRD databases. Ask in English; the query runs read-only."
        )
        with gr.Row():
            question = gr.Textbox(label="Question", lines=2, scale=3)
            db = gr.Dropdown(choices=databases(), label="Database", scale=1)
        with gr.Row():
            evidence = gr.Textbox(label="Evidence (optional hint)", scale=3)
            execute = gr.Checkbox(label="Run the query", value=True, scale=1)
        button = gr.Button("Generate SQL", variant="primary")
        sql_out = gr.Code(label="SQL", language="sql")
        rows_out = gr.Markdown(label="Result")
        note_out = gr.Markdown()
        gr.Examples(examples=EXAMPLES, inputs=[question, db])

        button.click(ask, [question, db, evidence, execute], [sql_out, rows_out, note_out])
        question.submit(ask, [question, db, evidence, execute], [sql_out, rows_out, note_out])
    return ui


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_ui(args.api).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
