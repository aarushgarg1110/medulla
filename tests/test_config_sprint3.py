"""Tests for config — provider switching, TOML persistence."""
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def reset_config():
    import medulla.config as cfg
    cfg._config = None
    yield
    cfg._config = None


def test_default_config_has_all_providers():
    from medulla.config import get_config
    cfg = get_config()
    assert cfg.llm.active == "bedrock"
    assert cfg.llm.bedrock.model
    assert cfg.llm.anthropic.model
    assert cfg.llm.ollama.model
    assert cfg.llm.ollama.host


def test_save_and_reload_config(tmp_path, monkeypatch):
    import medulla.config as cfg_module
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", tmp_path / "config.toml")
    cfg_module._config = None

    from medulla.config import get_config, save_config, set_active_provider
    cfg = get_config()
    cfg.medulla_dir = tmp_path
    set_active_provider("anthropic")

    # Reload from disk
    cfg_module._config = None
    cfg2 = get_config()
    assert cfg2.llm.active == "anthropic"


def test_set_active_provider_updates_singleton(tmp_path, monkeypatch):
    import medulla.config as cfg_module
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", tmp_path / "config.toml")
    cfg_module._config = None
    from medulla.config import get_config, set_active_provider
    cfg = get_config()
    cfg.medulla_dir = tmp_path

    set_active_provider("ollama")
    assert get_config().llm.active == "ollama"


def test_config_with_missing_file_uses_defaults(tmp_path, monkeypatch):
    import medulla.config as cfg_module
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", tmp_path / "nonexistent.toml")
    cfg_module._config = None
    from medulla.config import get_config
    cfg = get_config()
    assert cfg.llm.active == "bedrock"
