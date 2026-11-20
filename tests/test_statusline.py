"""medulla's statusline: what it prints, what it records for autohandoff, and how setup installs it next
to a user's own statusline without changing what that statusline shows."""
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from medulla import statusline as sl
from medulla import statusline_setup as setup
from medulla.cli import app

runner = CliRunner()
INPUT = {"session_id": "s-1", "session_name": "fooPart2",
         "model": {"id": "claude-opus-5-5[1m]", "display_name": "Opus 5.5 (1M context)"},
         "context_window": {"context_window_size": 1_000_000, "used_percentage": 77.5},
         "rate_limits": {"five_hour": {"used_percentage": 23.0}}}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDULLA_DIR", str(tmp_path / "medulla"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("ANTIGRAVITY_CONFIG_DIR", str(tmp_path / "agy"))
    return tmp_path


def _plain(s: str) -> str:
    return re.sub(r"\033\[\d+m", "", s)


def _settings(home, data):
    sp = home / "claude" / "settings.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(data))
    return sp


def _main(argv, data, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(data))
    rc = sl.main(argv)
    return rc, capsys.readouterr().out


# ── what it shows and records ─────────────────────────────────────────────────

def test_render_is_model_session_then_context_with_tokens_and_usage():
    assert _plain(sl.render(INPUT)) == ("[Opus 5.5 (1M context)] | fooPart2\n"
                                         "Context ██████░░ 78% (775k/1M) | Usage █░░░░░░░ 23%")


def test_render_without_a_window_size_or_a_session_name():
    assert _plain(sl.render({"context_window": {"used_percentage": 5}})) == (
        "[Claude]\nContext ░░░░░░░░ 5% | Usage ░░░░░░░░ 0%")


def test_record_writes_the_window_size_autohandoff_reads(home):
    p = Path(sl.record(INPUT))
    assert p.parent == home / "medulla" / "state"
    assert json.loads(p.read_text()) == {"size": 1_000_000, "pct": 77.5, "name": "fooPart2",
                                         "model": "claude-opus-5-5[1m]"}
    assert sl.record({}) is None


def test_the_folder_rule_is_medulla_configs(monkeypatch, tmp_path):
    from medulla.config import _default_medulla_dir
    monkeypatch.delenv("MEDULLA_DIR", raising=False)
    assert Path(sl.medulla_dir()) == _default_medulla_dir()
    monkeypatch.setenv("MEDULLA_DIR", str(tmp_path))
    assert Path(sl.medulla_dir()) == _default_medulla_dir()


def test_tap_records_and_prints_nothing(home, monkeypatch, capsys):
    assert _main(["--tap"], json.dumps(INPUT), monkeypatch, capsys) == (0, "")
    assert (home / "medulla" / "state" / "s-1.ctx.json").exists()


def test_wrap_prints_the_users_output_from_the_same_input(home, monkeypatch, capsys):
    raw = json.dumps(INPUT)
    assert _main(["--wrap", "cat"], raw, monkeypatch, capsys) == (0, raw)
    assert (home / "medulla" / "state" / "s-1.ctx.json").exists()


def test_bad_input_or_an_unwritable_home_never_blanks_it(home, monkeypatch, capsys):
    def read_only(d):
        raise OSError("read-only file system")
    monkeypatch.setattr(sl, "record", read_only)
    rc, out = _main([], "not json", monkeypatch, capsys)
    assert rc == 0 and _plain(out).startswith("[Claude]")


# ── setup: install, wrap, undo ────────────────────────────────────────────────

def test_no_statusline_gets_medullas_and_the_launcher_runs_it(home):
    sp = _settings(home, {"model": "opus"})
    assert setup.plan()["mode"] == "install" and "no statusline" in setup.describe(setup.plan())
    cmd = setup.apply()["command"]
    s = json.loads(sp.read_text())
    assert s["model"] == "opus" and s["statusLine"] == {"type": "command", "command": cmd}
    r = subprocess.run(cmd, shell=True, input=json.dumps(INPUT), capture_output=True, text=True)
    assert "fooPart2" in r.stdout and (home / "medulla" / "state" / "s-1.ctx.json").exists()


def test_their_statusline_is_wrapped_shows_the_same_and_undo_restores_it(home):
    script = home / "mine.sh"
    script.write_text('cat > /dev/null\necho "my line"   # prints used_percentage in real life\n')
    original = f"sh {script}"
    sp = _settings(home, {"statusLine": {"type": "command", "command": original, "padding": 1}})
    p = setup.plan()
    assert p["mode"] == "wrap" and p["shows_context"] and "already shows" in setup.describe(p)
    cmd = setup.apply()["command"]
    r = subprocess.run(cmd, shell=True, input=json.dumps(INPUT), capture_output=True, text=True)
    assert r.stdout == "my line\n" and (home / "medulla" / "state" / "s-1.ctx.json").exists()
    assert json.loads(sp.read_text())["statusLine"]["padding"] == 1
    assert setup.plan()["mode"] == "present" and setup.apply()["mode"] == "present"
    assert "Restored" in setup.undo()
    assert json.loads(sp.read_text())["statusLine"] == {"type": "command", "command": original, "padding": 1}


def test_a_statusline_that_does_not_show_context_is_still_wrapped(home):
    plain = home / "plain.sh"
    plain.write_text("echo hi\n")
    _settings(home, {"statusLine": {"type": "command", "command": f"sh {plain}"}})
    p = setup.plan()
    assert p["mode"] == "wrap" and not p["shows_context"] and "already shows" not in setup.describe(p)


def test_undo_removes_an_installed_one_and_leaves_others_alone(home):
    sp = _settings(home, {})
    setup.apply()
    assert "Removed" in setup.undo() and "statusLine" not in json.loads(sp.read_text())
    _settings(home, {"statusLine": {"type": "command", "command": "echo theirs"}})
    assert "nothing to undo" in setup.undo()


def test_settings_are_backed_up_once_and_a_symlink_stays_a_link(home):
    real = home / "dotfiles-settings.json"
    real.write_text(json.dumps({"a": 1}))
    sp = home / "claude" / "settings.json"
    sp.parent.mkdir(parents=True)
    sp.symlink_to(real)
    setup.apply()
    setup.undo()
    setup.apply()
    assert sp.is_symlink() and "statusLine" in json.loads(real.read_text())
    assert json.loads((home / "claude" / "settings.json.medulla-backup").read_text()) == {"a": 1}


def test_the_launcher_follows_a_moved_install(home):
    assert setup.refresh_target() is False                 # never opted in: nothing is written
    setup.install_launcher()
    assert setup.refresh_target() is False                 # same install: unchanged
    assert setup.write_target("/old/venv/bin/python") is True
    assert setup.refresh_target() is True
    assert (home / "medulla" / "statusline.target").read_text().splitlines()[0] == sys.executable


def test_a_failed_refresh_never_stops_the_server(monkeypatch):
    from medulla.mcp import _refresh_statusline

    def broken():
        raise RuntimeError("disk full")
    monkeypatch.setattr(setup, "refresh_target", broken)
    _refresh_statusline()


# ── the CLI front door ────────────────────────────────────────────────────────

def test_cli_dry_run_changes_nothing_then_yes_applies_and_undo_reverts(home):
    sp = _settings(home, {})
    r = runner.invoke(app, ["statusline", "setup", "--dry-run"])
    assert r.exit_code == 0 and "no statusline" in r.output and json.loads(sp.read_text()) == {}
    r = runner.invoke(app, ["statusline", "setup", "--yes"])
    assert "Done" in r.output and "statusLine" in json.loads(sp.read_text())
    assert "already set up" in runner.invoke(app, ["statusline", "setup"]).output
    assert "Removed" in runner.invoke(app, ["statusline", "undo"]).output


def test_cli_asks_first_and_respects_a_no(home):
    sp = _settings(home, {})
    r = runner.invoke(app, ["statusline", "setup"], input="n\n")
    assert r.exit_code == 1 and json.loads(sp.read_text()) == {}


def test_cli_tap_installs_only_the_launcher_and_prints_its_line(home):
    sp = _settings(home, {})
    r = runner.invoke(app, ["statusline", "setup", "--tap"])
    assert str(home / "medulla" / "bin" / "statusline") in r.output and "--tap" in r.output
    assert (home / "medulla" / "bin" / "statusline").exists() and json.loads(sp.read_text()) == {}


# ── Antigravity / Gemini CLI support ──────────────────────────────────────────

def test_render_antigravity_payload_uses_conversation_title_and_quota():
    agy_input = {
        "product": "antigravity",
        "conversation_id": "c-1",
        "conversation_title": "introToNeon",
        "model": {"id": "gemini-4-neon-high", "display_name": "Gemini 4 Neon (High)"},
        "context_window": {"context_window_size": 1_048_576, "used_percentage": 20.0},
        "quota": {"gemini-5h": {"remaining_fraction": 0.96}, "gemini-weekly": {"remaining_fraction": 0.70}},
    }
    assert _plain(sl.render(agy_input)) == (
        "[Gemini 4 Neon (High)] | introToNeon\n"
        "Context █░░░░░░░ 20% (210k/1.05M) | Usage ░░░░░░░░ 4%"
    )
    # 3P model prefers 3p-5h; missing 5h falls back to weekly; empty/no-fraction quota -> 0%
    third_party = {"model": {"display_name": "Opus 5.5"}, "quota": {"3p-5h": {"remaining_fraction": 0.5}}}
    assert "Usage ████░░░░ 50%" in _plain(sl.render(third_party))
    weekly_only = {"product": "antigravity", "quota": {"gemini-weekly": {"remaining_fraction": 0.75}}}
    assert _plain(sl.render(weekly_only)).startswith("[Antigravity]\n") and "Usage ██░░░░░░ 25%" in _plain(sl.render(weekly_only))
    no_frac = {"product": "antigravity", "quota": {"gemini-5h": {"disabled": True}}}
    assert "Usage ░░░░░░░░ 0%" in _plain(sl.render(no_frac))


def test_record_antigravity_payload_captures_conversation_id_and_title(home):
    agy_input = {
        "conversation_id": "c-42",
        "conversation_title": "introToNeon",
        "model": {"id": "gemini-4-neon-high"},
        "context_window": {"context_window_size": 1_048_576, "used_percentage": 14.5},
    }
    p = Path(sl.record(agy_input))
    assert p == home / "medulla" / "state" / "c-42.ctx.json"
    assert json.loads(p.read_text()) == {"size": 1_048_576, "pct": 14.5, "name": "introToNeon",
                                         "model": "gemini-4-neon-high"}


def test_statusline_setup_target_gemini_reads_and_writes_agy_settings(home):
    sp = home / "agy" / "settings.json"
    sp.parent.mkdir(parents=True)
    sp.write_text(json.dumps({"model": "Gemini 4 Neon (High)"}))
    r = runner.invoke(app, ["statusline", "setup", "--target", "gemini", "--yes"])
    assert r.exit_code == 0 and "Done" in r.output
    assert "statusLine" in json.loads(sp.read_text())
    assert not (home / "claude" / "settings.json").exists()
    r = runner.invoke(app, ["statusline", "undo", "--target", "gemini"])
    assert r.exit_code == 0 and "Removed" in r.output and "statusLine" not in json.loads(sp.read_text())
