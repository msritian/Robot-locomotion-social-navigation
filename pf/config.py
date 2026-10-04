"""YAML config loading with deep-merge overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_WORLD = CONFIG_DIR / "world_default.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load the default world config, then an optional experiment YAML, then overrides."""
    with open(DEFAULT_WORLD) as f:
        cfg = yaml.safe_load(f)
    if path is not None:
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    return deep_merge(cfg, overrides or {})


def split_of(seed: int, cfg: dict) -> str:
    """Seeds >= test_seed_start (1000) are held-out test seeds."""
    return "test" if seed >= cfg["maps"]["test_seed_start"] else "train"
