"""Offline tests for matins.config.load_config (no network, no API keys)."""
from __future__ import annotations

from pathlib import Path

from matins.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_example_config() -> None:
    cfg = load_config(REPO_ROOT / "config.example.yaml")
    assert cfg.provider.name == "anthropic"
    assert cfg.generation.output_language == "bilingual"
    assert cfg.fast_kernel is not None
    assert cfg.fast_kernel.name == "fast"
    assert cfg.slow_kernel is not None
    assert cfg.slow_kernel.window_days == 75


def test_load_missing_config_falls_back_to_defaults() -> None:
    cfg = load_config(REPO_ROOT / "does_not_exist.yaml")
    assert cfg.provider.name == "anthropic"
    # Two default memory kernels are present when no file is found.
    assert len(cfg.memory_kernels) == 2
    assert cfg.fast_kernel is not None
    assert cfg.slow_kernel is not None
