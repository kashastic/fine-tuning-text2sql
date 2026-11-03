"""Thin YAML config loading, with CLI overrides.

Every script takes `--config configs/<name>.yaml` plus `--set key=value`
overrides, so a run is reproducible from one file and a sweep does not need a
new file per variant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a declared dependency
    yaml = None  # type: ignore[assignment]


class Config(dict):
    """A dict with dotted access: `cfg.get_path("train.lr")`."""

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node: dict[str, Any] = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot set {path}: {part} is not a mapping")
        node[parts[-1]] = value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def _coerce(text: str) -> Any:
    """`"3"` -> 3, `"true"` -> True, `"[1,2]"` -> [1, 2], else the string."""
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML config, then apply `key.path=value` overrides in order."""
    data: dict[str, Any] = {}
    if path:
        if yaml is None:
            raise RuntimeError("pyyaml is required to read config files: pip install pyyaml")
        with Path(path).open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a mapping at the top level")
        data = loaded

    cfg = Config(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        cfg.set_path(key.strip(), _coerce(value))
    return cfg


def dump_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Write the resolved config next to a run's results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        if yaml is not None:
            yaml.safe_dump(dict(cfg), fh, sort_keys=False, default_flow_style=False)
        else:  # pragma: no cover
            json.dump(dict(cfg), fh, indent=2)
