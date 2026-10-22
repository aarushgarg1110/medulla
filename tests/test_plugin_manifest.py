"""The repo is also a Claude Code plugin: one install brings the MCP server (and, in later steps, the
skills and hooks). Claude Code drops a component whose path is missing without failing the
install, so a broken manifest would surface on a tester's machine as "the tools aren't there". These
tests check it in CI instead."""
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / ".claude-plugin"
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def _manifest(name: str) -> dict:
    return json.loads((PLUGIN / name).read_text())


def test_the_plugin_version_is_the_package_version():
    """0.2.0-rc1 in the manifest and the git tag is 0.2.0rc1 in PEP 440 spelling."""
    m = _manifest("plugin.json")
    assert m["name"] == "medulla"
    assert m["version"].replace("-", "") == PYPROJECT["project"]["version"]


def test_the_mcp_server_runs_medullas_entry_point_from_the_installed_plugin():
    srv = _manifest("plugin.json")["mcpServers"]["medulla"]
    args = srv["args"]
    assert srv["command"] == "uv"
    assert args[args.index("--project") + 1] == "${CLAUDE_PLUGIN_ROOT}"
    assert args[-2:] == ["medulla", "mcp"]
    assert PYPROJECT["project"]["scripts"]["medulla"] == "medulla.cli:main"


def test_every_component_path_the_manifest_names_exists():
    m = _manifest("plugin.json")
    for key in ("skills", "commands", "agents", "hooks", "outputStyles"):
        for p in ([m[key]] if isinstance(m.get(key), str) else m.get(key) or []):
            if isinstance(p, str):
                assert (ROOT / p).exists(), f"{key}: {p} is missing"


def test_the_marketplace_lists_this_repo_as_its_one_plugin():
    mk = _manifest("marketplace.json")
    assert [(p["name"], p["source"]) for p in mk["plugins"]] == [("medulla", "./")]

