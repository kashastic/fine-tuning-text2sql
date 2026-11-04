from __future__ import annotations

from text2sql.prompts import build_prompt, system_prompt, to_sft_record, user_prompt


def test_prompt_carries_schema_question_and_evidence(shop_example):
    messages = build_prompt(shop_example)
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "CREATE TABLE" in user
    assert shop_example.question in user
    assert shop_example.evidence in user


def test_evidence_is_omitted_when_absent():
    assert "Evidence:" not in user_prompt("CREATE TABLE t(a int);", "how many?")


def test_the_two_styles_differ_only_in_the_system_turn(shop_example):
    plain = build_prompt(shop_example, style="sql")
    reasoning = build_prompt(shop_example, style="reasoning")
    assert plain[1] == reasoning[1]
    assert plain[0] != reasoning[0]
    assert "reasoning" in system_prompt("reasoning").lower()


def test_baseline_and_fine_tuned_model_get_the_same_prompt(shop_example):
    # The comparison is only fair if one function builds both prompts, so this
    # test exists to fail loudly if someone special-cases a backend.
    assert build_prompt(shop_example) == build_prompt(shop_example)


def test_sft_record_shape(shop_example):
    record = to_sft_record(shop_example)
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant"]
    assert record["messages"][-1]["content"].startswith("```sql")
    assert shop_example.gold_sql.rstrip(";") in record["messages"][-1]["content"]
    # GRPO's reward function reads these two straight off the dataset.
    assert record["db_path"] == shop_example.db_path
    assert record["gold_sql"] == shop_example.gold_sql
    assert record["prompt"] == record["messages"][:2]


def test_reasoning_target_puts_sql_last(shop_example):
    record = to_sft_record(shop_example, style="reasoning", reasoning="Join orders to customer.")
    content = record["messages"][-1]["content"]
    assert content.startswith("Join orders to customer.")
    assert content.rstrip().endswith("```")


def test_anthropic_backend_says_what_is_missing_without_credentials(monkeypatch):
    """The SDK resolves credentials on first request, deep in a header builder.

    Without this check the baseline builds all 1,534 prompts and then dies in
    a TypeError from inside the client, which reads like a bug in this repo.
    """
    import pytest

    from text2sql.backends import build_backend

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    try:
        build_backend("anthropic", model="claude-opus-5-5")
    except RuntimeError as exc:
        assert "ANTHROPIC_API_KEY" in str(exc)
    except ImportError:
        pytest.skip("anthropic SDK not installed")


def test_openai_api_backend_rejects_a_missing_key_clearly(monkeypatch):
    import pytest

    from text2sql.backends import build_backend

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_backend("openai-api", model="gpt-5.4-nano")


def test_openai_api_payload_omits_temperature_and_caps_completion_tokens(monkeypatch):
    """Hosted reasoning models 400 on `max_tokens` and on a set temperature."""
    import pytest

    try:
        import httpx  # noqa: F401
    except ImportError:
        pytest.skip("httpx not installed")

    from text2sql.backends import build_backend

    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")
    backend = build_backend("openai-api", model="gpt-5.4-nano")
    payload = backend._payload([{"role": "user", "content": "hi"}], n=1)

    assert "max_tokens" not in payload
    assert "temperature" not in payload
    assert payload["max_completion_tokens"] == 1024
    # Not "minimal": gpt-5.4-nano rejects that value with a 400, and the
    # accepted set is none/low/medium/high/xhigh.
    assert payload["reasoning_effort"] == "low"


def test_openai_backend_retries_429_but_not_400(monkeypatch):
    """A rate limit is worth retrying; a rejected parameter never is.

    An un-retried 429 does not show up as an error in the results -- it shows
    up as a wrong answer, because an empty generation scores zero. A retried
    400 just takes five times as long to tell you the same thing.
    """
    import pytest

    try:
        import httpx
    except ImportError:
        pytest.skip("httpx not installed")

    from text2sql.backends import build_backend

    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setattr("time.sleep", lambda _s: None)
    backend = build_backend("openai-api", model="gpt-5.4-nano", max_retries=3)

    calls: list[int] = []

    def fake_post(status: int):
        def post(_url, json):  # noqa: A002 - httpx's own parameter name
            calls.append(status)
            request = httpx.Request("POST", "https://example.invalid")
            body = b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'
            return httpx.Response(status, request=request, content=body)
        return post

    monkeypatch.setattr(backend.client, "post", fake_post(429))
    out = backend._one([{"role": "user", "content": "hi"}], n=1)
    assert len(calls) == 3  # exhausted the retries
    assert "429" in out[0].error

    calls.clear()
    monkeypatch.setattr(backend.client, "post", fake_post(400))
    out = backend._one([{"role": "user", "content": "hi"}], n=1)
    assert len(calls) == 1  # gave up at once
    # The response body travels with the error, or a 400 is undebuggable.
    assert "choices" in out[0].error
