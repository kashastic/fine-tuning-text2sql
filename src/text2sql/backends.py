"""Generation backends behind one interface.

Everything downstream -- evaluation, self-consistency, the latency bench --
only needs "turn these chat prompts into text", so each backend implements
exactly that. Heavy imports stay inside the constructors: a laptop running the
tests has neither vLLM nor the Anthropic SDK installed, and should not need
them.

    vllm      offline batched generation, the one to use for a 1,534-question
              evaluation sweep on a rented GPU
    openai    an already-running OpenAI-compatible server (vLLM's own, or the
              FastAPI wrapper in serve/) -- this is what the latency numbers
              must come from, since offline vLLM hides queueing
    anthropic the large prompted baseline
    hf        plain transformers; slow, but runs on a laptop for smoke tests
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

Messages = list[dict[str, str]]


@dataclass(slots=True)
class Generation:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    error: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    name: str
    model: str

    def generate(self, prompts: Sequence[Messages], n: int = 1) -> list[list[Generation]]:
        """One list of `n` generations per prompt, in the order given."""


# --------------------------------------------------------------------------- vLLM

class VLLMBackend:
    """Offline batched generation. Needs a GPU."""

    name = "vllm"

    def __init__(
        self,
        model: str,
        max_model_len: int = 8192,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
        lora_path: str | None = None,
        dtype: str = "auto",
        seed: int = 0,
    ) -> None:
        from vllm import LLM  # noqa: PLC0415 - optional heavy dependency

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.lora_path = lora_path
        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            seed=seed,
            enable_lora=lora_path is not None,
            max_lora_rank=64,
        )

    def generate(self, prompts: Sequence[Messages], n: int = 1) -> list[list[Generation]]:
        from vllm import SamplingParams  # noqa: PLC0415

        params = SamplingParams(
            n=n,
            temperature=self.temperature if n == 1 else max(self.temperature, 0.7),
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        kwargs: dict[str, Any] = {}
        if self.lora_path:
            from vllm.lora.request import LoRARequest  # noqa: PLC0415

            kwargs["lora_request"] = LoRARequest("adapter", 1, self.lora_path)

        started = time.perf_counter()
        outputs = self.llm.chat(list(prompts), params, **kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000

        results: list[list[Generation]] = []
        for out in outputs:
            prompt_tokens = len(out.prompt_token_ids or [])
            results.append(
                [
                    Generation(
                        text=choice.text,
                        input_tokens=prompt_tokens,
                        output_tokens=len(choice.token_ids or []),
                        # Batched: per-item latency is meaningless, so report
                        # the amortised figure and get real latency from the
                        # server backend instead.
                        latency_ms=elapsed_ms / max(len(outputs), 1),
                    )
                    for choice in out.outputs
                ]
            )
        return results


# ------------------------------------------------------------- OpenAI-compatible

class OpenAICompatBackend:
    """Talks to a running vLLM / FastAPI server. Use this for latency numbers."""

    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_workers: int = 16,
        max_retries: int = 5,
    ) -> None:
        import httpx  # noqa: PLC0415

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key or os.getenv('VLLM_API_KEY', 'EMPTY')}"},
        )

    def _payload(self, messages: Messages, n: int) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature if n == 1 else max(self.temperature, 0.7),
            "n": n,
        }

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST once, retrying only what retrying can fix.

        A hosted API rate-limits a 1,534-question sweep long before a local
        vLLM server would, and an un-retried 429 does not look like an error
        in the results -- it looks like a wrong answer, because an empty
        generation scores zero. So back off and try again on 429 and 5xx, and
        let a 400 through immediately: a rejected parameter is rejected every
        time, and retrying it just costs five times as long to find out.
        """
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status is None or status == 429 or status >= 500
                if not retryable or attempt == self.max_retries - 1:
                    raise
                last = exc
                time.sleep(min(2**attempt, 30) * (0.5 + random.random()))
        raise last or RuntimeError("retries exhausted")  # pragma: no cover

    def _one(self, messages: Messages, n: int) -> list[Generation]:
        payload = self._payload(messages, n)
        started = time.perf_counter()
        try:
            body = self._post_with_retry(payload)
        except Exception as exc:  # noqa: BLE001 - a dead server is a result too
            # Carry the response body: a 400 from a hosted API says exactly
            # which parameter it rejected, and losing that text turns a
            # one-line fix into a debugging session.
            detail = ""
            body = getattr(getattr(exc, "response", None), "text", "")
            if body:
                detail = f" -- {body.strip()[:400]}"
            return [Generation("", error=f"{type(exc).__name__}: {exc}{detail}")] * n
        latency_ms = (time.perf_counter() - started) * 1000
        usage = body.get("usage") or {}
        return [
            Generation(
                text=choice["message"]["content"] or "",
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                latency_ms=latency_ms,
            )
            for choice in body.get("choices", [])
        ] or [Generation("", error="empty response", latency_ms=latency_ms)]

    def generate(self, prompts: Sequence[Messages], n: int = 1) -> list[list[Generation]]:
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda m: self._one(m, n), prompts))


class OpenAIAPIBackend(OpenAICompatBackend):
    """A hosted OpenAI model as the prompted baseline.

    The wire format is the same chat/completions shape vLLM serves, but the
    hosted reasoning models differ in three ways that each return a 400:
    `max_tokens` is rejected in favour of `max_completion_tokens`, a
    non-default `temperature` is rejected outright, and `reasoning_effort`
    decides how many billed reasoning tokens arrive before the answer. That
    last one is the cost lever -- reasoning tokens bill as output, so a
    default-effort run on 1,534 questions can cost several times what the
    visible answers suggest.

    The accepted efforts vary by model: gpt-5.4-nano takes
    "none"/"low"/"medium"/"high"/"xhigh" and rejects the "minimal" that older
    GPT-5 models took. "low" is the default here -- measured at zero reasoning
    tokens on text-to-SQL prompts, so it costs what the visible answer costs,
    while leaving the model free to reason on a question that needs it.
    """

    name = "openai-api"

    def __init__(
        self,
        model: str = "gpt-5.4-nano",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        reasoning_effort: str | None = "low",
        max_new_tokens: int = 1024,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "no OpenAI credentials found. Set OPENAI_API_KEY (create a key at "
                "platform.openai.com). Only the prompted baseline needs one."
            )
        # The cap has to cover reasoning tokens as well as the SQL, and only
        # tokens actually produced are billed, so a roomy cap costs nothing
        # while a tight one truncates answers into wrong ones.
        super().__init__(
            model=model, base_url=base_url, api_key=key,
            max_new_tokens=max_new_tokens, **kwargs,
        )
        self.reasoning_effort = reasoning_effort
        self.seed = seed

    def _payload(self, messages: Messages, n: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # Hosted reasoning models reject `max_tokens`, and this budget has
            # to cover reasoning tokens as well as the SQL.
            "max_completion_tokens": self.max_new_tokens,
            "n": n,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.seed is not None:
            # Best-effort only, and the provider says so. This model refuses
            # temperature=0 ("only the default (1) is supported") and rejects
            # top_p, so greedy decoding is impossible and the baseline is
            # sampled whether we like it or not: two runs over identical
            # prompts disagreed on 10.1% of BIRD dev. A seed narrows that; it
            # does not close it. See docs/MEASUREMENTS.md, M12.
            payload["seed"] = self.seed
        return payload


# ---------------------------------------------------------------------- Anthropic

class AnthropicBackend:
    """The large prompted baseline.

    Same prompt, same schema, same evidence as the fine-tuned model -- an
    unfair baseline would sink the whole comparison, so the prompt is built by
    `text2sql.prompts` for both and never tweaked here.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        max_new_tokens: int = 4096,
        effort: str | None = "low",
        max_workers: int = 8,
        max_retries: int = 5,
        timeout: float = 300.0,
    ) -> None:
        import anthropic  # noqa: PLC0415

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.effort = effort
        self.max_workers = max_workers
        self.max_retries = max_retries
        self._anthropic = anthropic
        # Picks up ANTHROPIC_API_KEY, or an `ant auth login` profile. The SDK
        # only resolves credentials on the first request, which would surface
        # as a TypeError from deep inside its header builder after the prompts
        # are already built -- so fail here, before any work or spend.
        try:
            self.client = anthropic.Anthropic(timeout=timeout, max_retries=0)
            self.client.api_key or self.client.auth_token or self.client._auth_headers  # noqa: B018
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "no Anthropic credentials found. Set ANTHROPIC_API_KEY (create a key "
                "at console.anthropic.com -- API billing is separate from a Claude "
                "subscription), or run `ant auth login`. The API baseline is the only "
                "part of this project that needs one."
            ) from exc
        self._supports_output_config = effort is not None

    def _request(self, messages: Messages) -> Generation:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_new_tokens,
            "system": system,
            "messages": turns,
        }
        if self._supports_output_config:
            kwargs["output_config"] = {"effort": self.effort}

        started = time.perf_counter()
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(**kwargs)
                break
            except TypeError:
                # Installed SDK predates output_config: drop it and retry once.
                if "output_config" in kwargs:
                    kwargs.pop("output_config")
                    self._supports_output_config = False
                    continue
                raise
            except (self._anthropic.BadRequestError, self._anthropic.NotFoundError) as exc:
                return Generation("", error=f"{type(exc).__name__}: {exc}")
            except (
                self._anthropic.RateLimitError,
                self._anthropic.APIStatusError,
                self._anthropic.APIConnectionError,
            ) as exc:
                if attempt == self.max_retries - 1:
                    return Generation("", error=f"{type(exc).__name__}: {exc}")
                sleep = min(2**attempt, 30) * (0.5 + random.random())
                time.sleep(sleep)
        else:  # pragma: no cover - loop always breaks or returns
            return Generation("", error="retries exhausted")

        latency_ms = (time.perf_counter() - started) * 1000
        if getattr(response, "stop_reason", None) == "refusal":
            return Generation("", error="refusal", latency_ms=latency_ms)

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return Generation(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )

    def generate(self, prompts: Sequence[Messages], n: int = 1) -> list[list[Generation]]:
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        def run(messages: Messages) -> list[Generation]:
            return [self._request(messages) for _ in range(n)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(run, prompts))


# ------------------------------------------------------------------ transformers

class HFBackend:
    """Plain transformers. Slow; here so the pipeline is testable without a GPU."""

    name = "hf"

    def __init__(
        self,
        model: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        device: str | None = None,
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        batch_size: int = 4,
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.model_name = model
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        load_kwargs: dict[str, Any] = {"dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        self.net = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        if adapter_path:
            from peft import PeftModel  # noqa: PLC0415

            self.net = PeftModel.from_pretrained(self.net, adapter_path)
        if not load_in_4bit:
            self.net = self.net.to(self.device)
        self.net.eval()

    def generate(self, prompts: Sequence[Messages], n: int = 1) -> list[list[Generation]]:
        results: list[list[Generation]] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = list(prompts[start : start + self.batch_size])
            texts = [
                self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in batch
            ]
            encoded = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            started = time.perf_counter()
            with self.torch.inference_mode():
                output = self.net.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0 or n > 1,
                    temperature=max(self.temperature, 0.7) if n > 1 else (self.temperature or None),
                    num_return_sequences=n,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            latency_ms = (time.perf_counter() - started) * 1000 / max(len(batch), 1)
            prompt_len = encoded["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)
            for i in range(len(batch)):
                results.append(
                    [
                        Generation(
                            text=decoded[i * n + j],
                            input_tokens=int(encoded["attention_mask"][i].sum()),
                            output_tokens=None,
                            latency_ms=latency_ms,
                        )
                        for j in range(n)
                    ]
                )
        return results


def build_backend(kind: str, **kwargs: Any) -> Backend:
    kinds = {
        "vllm": VLLMBackend,
        "openai": OpenAICompatBackend,
        "openai-api": OpenAIAPIBackend,
        "anthropic": AnthropicBackend,
        "hf": HFBackend,
    }
    if kind not in kinds:
        raise ValueError(f"unknown backend {kind!r}; expected one of {sorted(kinds)}")
    return kinds[kind](**kwargs)  # type: ignore[return-value]
