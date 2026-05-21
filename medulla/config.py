"""Medulla configuration — stored at ~/.medulla/config.toml, managed via `medulla use`."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONFIG_FILE = Path.home() / ".medulla" / "config.toml"


@dataclass
class BedrockConfig:
    model: str = "anthropic.claude-haiku-4-5-20251001"
    aws_profile: str = "default"
    aws_region: str = "us-east-1"


@dataclass
class AnthropicConfig:
    model: str = "claude-haiku-4-5-20251001"


@dataclass
class OllamaConfig:
    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"


@dataclass
class LLMConfig:
    active: str = "bedrock"       # "bedrock" | "anthropic" | "ollama"
    bedrock: BedrockConfig = field(default_factory=BedrockConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


@dataclass
class Config:
    medulla_dir: Path = field(default_factory=lambda: Path.home() / ".medulla")
    llm: LLMConfig = field(default_factory=LLMConfig)

    @property
    def db_path(self) -> Path:
        return self.medulla_dir / "medulla.db"

    @property
    def wiki_path(self) -> Path:
        return self.medulla_dir / "wiki"

    @property
    def config_path(self) -> Path:
        return self.medulla_dir / "config.toml"


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = _load_config()
    return _config


def _load_config() -> Config:
    cfg = Config()
    if not CONFIG_FILE.exists():
        return cfg
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # fallback
        except ImportError:
            return cfg
    try:
        data = tomllib.loads(CONFIG_FILE.read_text())
        llm_data = data.get("llm", {})
        cfg.llm.active = llm_data.get("active", cfg.llm.active)
        bd = llm_data.get("bedrock", {})
        cfg.llm.bedrock.model = bd.get("model", cfg.llm.bedrock.model)
        cfg.llm.bedrock.aws_profile = bd.get("aws_profile", cfg.llm.bedrock.aws_profile)
        cfg.llm.bedrock.aws_region = bd.get("aws_region", cfg.llm.bedrock.aws_region)
        an = llm_data.get("anthropic", {})
        cfg.llm.anthropic.model = an.get("model", cfg.llm.anthropic.model)
        ol = llm_data.get("ollama", {})
        cfg.llm.ollama.model = ol.get("model", cfg.llm.ollama.model)
        cfg.llm.ollama.host = ol.get("host", cfg.llm.ollama.host)
    except Exception:
        pass
    return cfg


def save_config(cfg: Config) -> None:
    import tomli_w
    cfg.medulla_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "llm": {
            "active": cfg.llm.active,
            "bedrock": {
                "model": cfg.llm.bedrock.model,
                "aws_profile": cfg.llm.bedrock.aws_profile,
                "aws_region": cfg.llm.bedrock.aws_region,
            },
            "anthropic": {
                "model": cfg.llm.anthropic.model,
            },
            "ollama": {
                "model": cfg.llm.ollama.model,
                "host": cfg.llm.ollama.host,
            },
        }
    }
    cfg.config_path.write_bytes(tomli_w.dumps(data).encode())


def set_active_provider(provider: str) -> None:
    """Switch the active LLM provider and persist to config.toml."""
    cfg = get_config()
    cfg.llm.active = provider
    save_config(cfg)
    global _config
    _config = cfg
