"""Load and access experiment configuration from YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

@dataclass
class Config:
    """
    Thin wrapper around the parsed YAML config dict.

    Access nested values with dotted paths, e.g. cfg.get("data.raw_dir").
    """

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r") as f:
            data = yaml.safe_load(f)
        return cls(raw=data or {})

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_default_config() -> Config:
    """Convenience loader for configs/default.yaml relative to the repo root."""
    repo_root = Path(__file__).resolve().parents[2]
    return Config.load(repo_root / "configs" / "default.yaml")
