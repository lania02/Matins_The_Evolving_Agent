"""Offline tests for matins.config (no network, no API keys)."""
from __future__ import annotations

import os
from pathlib import Path

from matins.config import load_config, load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_example_config() -> None:
    # The example is a user-editable template; assert structural invariants, not
    # specific provider/window values a user may legitimately customize.
    cfg = load_config(REPO_ROOT / "config.example.yaml")
    assert cfg.generation.n_slots >= 1
    assert cfg.generation.output_language in {"en", "zh", "bilingual"}
    assert cfg.fast_kernel is not None and cfg.fast_kernel.feeds == "generation"
    assert cfg.slow_kernel is not None and cfg.slow_kernel.feeds == "consolidation"


def test_load_missing_config_falls_back_to_defaults() -> None:
    cfg = load_config(REPO_ROOT / "does_not_exist.yaml")
    # These are code defaults, independent of any (editable) file on disk.
    assert cfg.provider.name == "anthropic"
    assert len(cfg.memory_kernels) == 2
    assert cfg.fast_kernel is not None
    assert cfg.slow_kernel is not None


def test_load_dotenv_populates_env(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MATINS_TEST_KEY", raising=False)
    monkeypatch.delenv("MATINS_TEST_KEY2", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        'MATINS_TEST_KEY=secret123\n# a comment\nexport MATINS_TEST_KEY2="q"\n',
        encoding="utf-8",
    )
    assert load_dotenv(env) == env
    assert os.environ["MATINS_TEST_KEY"] == "secret123"
    assert os.environ["MATINS_TEST_KEY2"] == "q"  # quotes stripped, export ignored


def test_dotenv_does_not_override_existing_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MATINS_TEST_KEY3", "real")
    env = tmp_path / ".env"
    env.write_text("MATINS_TEST_KEY3=fromfile\n", encoding="utf-8")
    load_dotenv(env)
    assert os.environ["MATINS_TEST_KEY3"] == "real"  # existing OS env wins


def test_default_kernels_keep_the_whole_window() -> None:
    # Regression: neither kernel may decimate the log by default. A slow stride>1 silently
    # dropped most batches before consolidation ever saw them (the log is the asset).
    cfg = load_config(REPO_ROOT / "does_not_exist.yaml")   # code defaults
    assert cfg.fast_kernel is not None and cfg.fast_kernel.stride == 1
    assert cfg.slow_kernel is not None and cfg.slow_kernel.stride == 1
