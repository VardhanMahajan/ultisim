"""Load the TOML config into attribute-access namespaces.

    cfg = load_config()            # configs/default.toml
    cfg.field.width                # 37.0
    cfg.players.top_speed.mean     # 7.5
"""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.toml"


def _ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config_dict(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "rb") as f:
        raw = tomllib.load(f)
    if overrides:
        raw = _merge(raw, overrides)
    return raw


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> SimpleNamespace:
    """Load config; `overrides` is a nested dict merged on top (for sensitivity runs)."""
    return _ns(load_config_dict(path, overrides))
