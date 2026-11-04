"""Prices used for the cost-per-1,000-queries column of the results table.

Two very different cost models sit side by side in this project:

  * the API baseline is billed per token,
  * the self-hosted model is billed per GPU hour, so its per-query cost falls
    as throughput rises -- which is exactly the comparison the project is
    making, and the reason throughput is measured rather than guessed.

Prices are editable on purpose: override them from the config rather than
trusting the defaults below, which were current when the repo was written
(Anthropic list prices, USD per million tokens) and will drift.
"""

from __future__ import annotations

from dataclasses import dataclass

USD_PER_EUR = 1.08  # override in config; only used for the EUR columns


@dataclass(frozen=True, slots=True)
class TokenPrice:
    input_per_mtok: float
    output_per_mtok: float


# USD per million tokens. Checked 2026-11-04; verify before quoting a cost,
# since published prices move and a wrong one silently skews the comparison.
API_PRICES: dict[str, TokenPrice] = {
    "claude-opus-5-5": TokenPrice(4.00, 20.00),
    "claude-opus-5": TokenPrice(5.00, 25.00),
    "claude-sonnet-5-5": TokenPrice(2.00, 10.00),
    "claude-haiku-5-5": TokenPrice(0.10, 0.50),
    # Hosted OpenAI models. Reasoning tokens bill as output, so the output
    # price is the one that decides a run's cost -- see OpenAIAPIBackend's
    # reasoning_effort.
    "gpt-5.4-mini": TokenPrice(0.75, 4.50),
    "gpt-5.4-nano": TokenPrice(0.20, 1.25),
    "gpt-5-mini": TokenPrice(0.25, 2.00),    # deprecated by OpenAI
    "gpt-5-nano": TokenPrice(0.05, 0.40),    # deprecated by OpenAI
}

# USD per GPU hour, on-demand, rented. Override per provider.
GPU_HOURLY: dict[str, float] = {
    "rtx4090": 0.45,
    "l40s": 0.85,
    "a100-40gb": 1.20,
    "a100-80gb": 1.75,
    "h100": 2.60,
}


def api_cost_per_1k(
    model: str,
    mean_input_tokens: float,
    mean_output_tokens: float,
    usd_per_eur: float = USD_PER_EUR,
    prices: dict[str, TokenPrice] | None = None,
) -> dict[str, float]:
    """Cost of 1,000 queries against a token-billed API."""
    table = prices or API_PRICES
    price = table.get(model)
    if price is None:
        raise KeyError(
            f"no price for {model!r}; add it to configs or to text2sql.pricing.API_PRICES"
        )
    usd = 1000 * (
        mean_input_tokens / 1e6 * price.input_per_mtok
        + mean_output_tokens / 1e6 * price.output_per_mtok
    )
    return {"usd": round(usd, 4), "eur": round(usd / usd_per_eur, 4)}


def self_hosted_cost_per_1k(
    queries_per_second: float,
    gpu: str = "a100-40gb",
    hourly_usd: float | None = None,
    usd_per_eur: float = USD_PER_EUR,
) -> dict[str, float]:
    """Cost of 1,000 queries on a rented GPU at the measured throughput."""
    if queries_per_second <= 0:
        raise ValueError("queries_per_second must be positive")
    rate = hourly_usd if hourly_usd is not None else GPU_HOURLY.get(gpu)
    if rate is None:
        raise KeyError(f"no hourly price for GPU {gpu!r}; pass hourly_usd")
    hours = 1000 / queries_per_second / 3600
    usd = hours * rate
    return {"usd": round(usd, 4), "eur": round(usd / usd_per_eur, 4), "gpu_hours": round(hours, 5)}
