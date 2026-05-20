"""Tests for medulla.config."""
from pathlib import Path

from medulla.config import Config, get_config


def test_config_default_db_path():
    c = Config()
    assert c.db_path == Path.home() / ".medulla" / "medulla.db"


def test_config_default_wiki_path():
    c = Config()
    assert c.wiki_path == Path.home() / ".medulla" / "wiki"


def test_config_custom_dir():
    c = Config(medulla_dir=Path("/tmp/custom"))
    assert c.db_path == Path("/tmp/custom/medulla.db")
    assert c.wiki_path == Path("/tmp/custom/wiki")


def test_get_config_returns_singleton():
    a = get_config()
    b = get_config()
    assert a is b


def test_get_config_is_config_instance():
    assert isinstance(get_config(), Config)
