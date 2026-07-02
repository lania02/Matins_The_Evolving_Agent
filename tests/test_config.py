"""Offline tests for matins.config (no network, no API keys)."""
from __future__ import annotations

import os
from pathlib import Path

from matins.config import Config, DeepDiveCfg, ProviderCfg, load_config, load_dotenv

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


def _provider_pair() -> Config:
    return Config(
        provider=ProviderCfg(name="openai_compatible", model="main-model",
                             base_url="https://main/v1", api_key_env="MAIN_KEY"),
        deep_dive=DeepDiveCfg(model="", provider_name="",
                              base_url="https://apihub.agnes-ai.com/v1",
                              api_key_env="AGNES_API_KEY"),
    )


def test_dig_provider_inherits_main_when_override_absent() -> None:
    # Default: no deep_dive.provider_name -> the dig runs on the MAIN provider (only the
    # model may differ). This is the original behavior; it must be preserved so a user can
    # revert simply by blanking provider_name (base_url/api_key_env below are ignored).
    cfg = _provider_pair()                                  # provider_name == ""
    dp = cfg.dig_provider()
    assert dp.name == "openai_compatible"
    assert dp.base_url == "https://main/v1"                 # NOT the agnes base_url
    assert dp.api_key_env == "MAIN_KEY"
    assert dp.model == "main-model"                         # empty dig model -> provider.model


def test_dig_provider_uses_standalone_override_without_touching_main() -> None:
    # Setting provider_name routes ONLY the dig to a separate vendor (agnes); the main
    # provider that daily generation uses is left completely untouched.
    cfg = _provider_pair()
    cfg.deep_dive.provider_name = "openai_compatible"
    cfg.deep_dive.model = "agnes-2.0-flash"
    dp = cfg.dig_provider()
    assert dp.base_url == "https://apihub.agnes-ai.com/v1"
    assert dp.api_key_env == "AGNES_API_KEY"
    assert dp.model == "agnes-2.0-flash"
    assert cfg.provider.base_url == "https://main/v1"       # main provider unchanged
    assert cfg.provider.model == "main-model"


def test_provider_roster_resolves_active(tmp_path) -> None:
    # Pluggable providers: `providers:` + `active_provider:` selects one endpoint config;
    # switching vendors/models is a one-line edit.
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "active_provider: nvidia-llama\n"
        "providers:\n"
        "  nvidia-llama:\n"
        "    name: openai_compatible\n"
        "    model: meta/llama-3.3-70b-instruct\n"
        "    base_url: https://integrate.api.nvidia.com/v1\n"
        "    api_key_env: NVIDIA_API_KEY\n"
        "  alt:\n"
        "    name: openai_compatible\n"
        "    model: other-model\n"
        "    base_url: https://integrate.api.nvidia.com/v1\n"
        "    api_key_env: NVIDIA_API_KEY_ALT\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.provider.model == "meta/llama-3.3-70b-instruct"
    assert cfg.provider.api_key_env == "NVIDIA_API_KEY"

    # one-line switch
    cfg_file.write_text(cfg_file.read_text(encoding="utf-8").replace(
        "active_provider: nvidia-llama", "active_provider: alt"), encoding="utf-8")
    assert load_config(cfg_file).provider.model == "other-model"


def test_provider_roster_fails_loud_on_dangling_active(tmp_path) -> None:
    import pytest

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "active_provider: nope\n"
        "providers:\n"
        "  real:\n"
        "    name: openai_compatible\n"
        "    model: m\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nope"):
        load_config(cfg_file)

    cfg_file.write_text(
        "providers:\n"
        "  real:\n"
        "    name: openai_compatible\n"
        "    model: m\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="active_provider"):
        load_config(cfg_file)
