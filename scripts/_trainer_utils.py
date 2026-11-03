"""Helpers shared by the two training scripts."""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any


def filter_kwargs(cls: type, values: dict[str, Any], label: str = "") -> dict[str, Any]:
    """Keep only the keys `cls` actually accepts, and say which were dropped.

    TRL renames config fields between minor versions. Rather than pin an exact
    version, pass what the installed one understands and print the rest, so a
    config written for a newer TRL fails loudly in one line instead of a
    TypeError traceback.
    """
    if dataclasses.is_dataclass(cls):
        allowed = {f.name for f in dataclasses.fields(cls)}
    else:  # pragma: no cover
        allowed = set(inspect.signature(cls).parameters)
    kept = {k: v for k, v in values.items() if k in allowed}
    dropped = sorted(set(values) - set(kept))
    if dropped:
        print(f"[warn] {label or cls.__name__} does not accept: {', '.join(dropped)} "
              f"(installed version is older or the key is misspelled)")
    return kept


def describe_trainable(model: Any) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total else 0.0
    print(f"[model] trainable {trainable/1e6:.2f}M / {total/1e6:.0f}M params ({pct:.3f}%)")
