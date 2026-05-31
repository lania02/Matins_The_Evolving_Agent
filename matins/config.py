"""Configuration loading for Matins (model-agnostic core, DESIGN.md section 13).

All knobs live in a single YAML file. This module turns it into typed dataclasses
with sensible defaults, so the rest of the code never touches raw dict access.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ProviderCfg:
    name: str = "anthropic"          # anthropic | openai | openai_compatible
    model: str = "claude-opus-4-8"
    base_url: str | None = None      # set for local / OpenAI-compatible endpoints
    api_key_env: str = "MATINS_API_KEY"


@dataclass
class GenerationCfg:
    n_slots: int = 4
    temperature: float = 0.4         # 0..1 explore aggressiveness
    output_language: str = "bilingual"  # en | zh | bilingual


@dataclass
class NoveltyCfg:
    search_provider: str = "web"     # web | arxiv | none
    k: int = 5


@dataclass
class TelegramCfg:
    bot_token_env: str = "MATINS_TELEGRAM_TOKEN"
    chat_id: str = ""


@dataclass
class MessagingCfg:
    channel: str = "telegram"        # telegram | none | whatsapp_baileys | whatsapp_cloud
    telegram: TelegramCfg = field(default_factory=TelegramCfg)
    collect_delay_hours: int = 3


@dataclass
class MemoryKernelCfg:
    name: str
    window_days: int
    stride: int
    aggregator: str                  # llm_summarize_recent | llm_propose_skill_diff
    feeds: str                       # generation | consolidation


@dataclass
class ConsolidationCfg:
    cadence_days: int = 7
    hypothesis_occurrence_threshold: int = 3
    require_human_approval: bool = True


@dataclass
class RetrievalCfg:
    sources: list[str] = field(default_factory=list)
    dedup_against_days: int = 30


@dataclass
class DeepDiveCfg:
    model: str = "gemini-3.5-flash"      # stronger model for on-demand briefings; "" = provider.model
    web_search: str = "tavily"           # tavily | none
    web_api_key_env: str = "TAVILY_API_KEY"
    k_per_query: int = 5
    max_queries: int = 4


DEFAULT_KERNELS = [
    MemoryKernelCfg("fast", 7, 1, "llm_summarize_recent", "generation"),
    MemoryKernelCfg("slow", 75, 5, "llm_propose_skill_diff", "consolidation"),
]


@dataclass
class Config:
    provider: ProviderCfg = field(default_factory=ProviderCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    novelty: NoveltyCfg = field(default_factory=NoveltyCfg)
    messaging: MessagingCfg = field(default_factory=MessagingCfg)
    memory_kernels: list[MemoryKernelCfg] = field(default_factory=lambda: list(DEFAULT_KERNELS))
    consolidation: ConsolidationCfg = field(default_factory=ConsolidationCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    deep_dive: DeepDiveCfg = field(default_factory=DeepDiveCfg)
    interest_seed_file: str = "prompts/interest_seed.md"
    # Optional sandbox switch: redirect all MUTABLE state (db, favorites, deep-dive
    # mirrors) under root/<state_dir> instead of root itself, while prompts/skills/
    # interest_seed stay shared with the real project. Lets you test generation/dig
    # against a throwaway DB without ever touching the production data. None = off.
    state_dir: str | None = None
    # Runtime-resolved, not from YAML:
    root: Path = field(default_factory=lambda: Path("."))

    def api_key(self) -> str | None:
        """Resolve the LLM API key from the configured environment variable."""
        return os.environ.get(self.provider.api_key_env)

    def telegram_token(self) -> str | None:
        return os.environ.get(self.messaging.telegram.bot_token_env)

    @property
    def fast_kernel(self) -> MemoryKernelCfg | None:
        return next((k for k in self.memory_kernels if k.feeds == "generation"), None)

    @property
    def slow_kernel(self) -> MemoryKernelCfg | None:
        return next((k for k in self.memory_kernels if k.feeds == "consolidation"), None)

    def interest_seed_path(self) -> Path:
        return (self.root / self.interest_seed_file)

    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    def skills_dir(self) -> Path:
        return self.root / "skills"

    def _state_root(self) -> Path:
        """Root for mutable state; redirected to root/<state_dir> in sandbox mode."""
        return (self.root / self.state_dir) if self.state_dir else self.root

    def db_path(self) -> Path:
        return self._state_root() / "data" / "matins.db"

    def favorites_path(self) -> Path:
        return self._state_root() / "favorites.md"

    def deep_dives_dir(self) -> Path:
        return self._state_root() / "deep_dives"

    def deep_dive_web_key(self) -> str | None:
        return os.environ.get(self.deep_dive.web_api_key_env)

    def dig_model(self) -> str:
        """Model for on-demand deep dives (falls back to the main provider model)."""
        return self.deep_dive.model or self.provider.model


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file. Ignores blanks, # comments, and a
    leading `export `. Strips one layer of matching surrounding quotes."""
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def load_dotenv(*candidates: Path, override: bool = False) -> Path | None:
    """Load the first existing .env candidate into os.environ (zero-dependency).

    By default existing OS environment variables win (override=False), matching
    python-dotenv. Returns the file that was loaded, or None.
    """
    for cand in candidates:
        if cand and cand.exists():
            for key, val in _parse_env_file(cand).items():
                if override or key not in os.environ:
                    os.environ[key] = val
            return cand
    return None


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load config from YAML, falling back to defaults for any missing block.

    Unknown top-level blocks are ignored; missing blocks use dataclass defaults,
    so a minimal or empty config file still yields a working Config. A `.env` file
    next to the config (or in the cwd) is loaded first so that api_key_env /
    bot_token_env lookups can resolve from it.
    """
    p = Path(path)
    load_dotenv(Path(".env"), p.parent / ".env")
    data: dict = {}
    if p.exists():
        loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded

    msg = data.get("messaging") or {}
    tg = msg.get("telegram") or {}
    kernels_raw = data.get("memory_kernels") or []
    kernels = [MemoryKernelCfg(**k) for k in kernels_raw] if kernels_raw else list(DEFAULT_KERNELS)

    return Config(
        provider=ProviderCfg(**(data.get("provider") or {})),
        generation=GenerationCfg(**(data.get("generation") or {})),
        novelty=NoveltyCfg(**(data.get("novelty") or {})),
        messaging=MessagingCfg(
            channel=msg.get("channel", "telegram"),
            telegram=TelegramCfg(**tg),
            collect_delay_hours=msg.get("collect_delay_hours", 3),
        ),
        memory_kernels=kernels,
        consolidation=ConsolidationCfg(**(data.get("consolidation") or {})),
        retrieval=RetrievalCfg(**(data.get("retrieval") or {})),
        deep_dive=DeepDiveCfg(**(data.get("deep_dive") or {})),
        interest_seed_file=data.get("interest_seed_file", "prompts/interest_seed.md"),
        state_dir=data.get("state_dir"),
        root=(p.parent if p.parent != Path("") else Path(".")),
    )
