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
    ) -> None:
        import httpx  # noqa: PLC0415

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_workers = max_workers
        self.client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key or os.getenv('VLLM_API_KEY', 'EMPTY')}"},
        )

    def _one(self, messages: Messages, n: int) -> list[Generation]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature if n == 1 else max(self.temperature, 0.7),
            "n": n,
        }
        started = time.perf_counter()
        try:
            response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001 - a dead server is a result too
            return [Generation("", error=f"{type(exc).__name__}: {exc}")] * n
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
        # Picks up ANTHROPIC_API_KEY, or an `ant auth login` profile.
        self.client = anthropic.Anthropic(timeout=timeout, max_retries=0)
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
        "anthropic": AnthropicBackend,
        "hf": HFBackend,
    }
    if kind not in kinds:
        raise ValueError(f"unknown backend {kind!r}; expected one of {sorted(kinds)}")
    return kinds[kind](**kwargs)  # type: ignore[return-value]
