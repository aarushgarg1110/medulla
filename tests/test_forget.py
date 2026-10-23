"""`medulla forget` removes a session from the index and keeps it out of every later scan (#65).

The real index on 2026-10-23: 216 sessions, ~10 of them test/demo sessions (ahtest*, ahProbe, MedullaDemo,
medullaMCPTest, TestMedullaForDan); removing any took hand-written SQL. Transcripts are Claude Code's: a forget
leaves them on disk and records the id on a forgotten list instead, and deleting one is a separate, explicit
step. The wiki layer is authored and never touched by a forget.
"""
import os
import shutil
import time

import pytest

from medulla.episodic import forget, scanner
from medulla.episodic.resume import NotFound, resume
from medulla.semantic.store import upsert_wiki_page
from tests.conftest import claude_assistant, claude_user, make_claude_jsonl


class _MockEmbedProvider:
    dimension = 768
    model_name = "mock"
    def embed(self, texts):
        return [[0.1] * self.dimension for _ in texts]


@pytest.fixture(autouse=True)
def projects(tmp_path, monkeypatch):
    """A Claude projects dir with one project, whose memory/ folder no forget may touch."""
    proj = tmp_path / "projects" / "-proj-a"
    (proj / "memory").mkdir(parents=True)
    (proj / "memory" / "MEMORY.md").write_text("keep me")
    monkeypatch.setattr(scanner, "CLAUDE_PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(scanner, "KIRO_SESSIONS_DIR", tmp_path / "no_kiro")
    monkeypatch.setattr(scanner, "_get_embedding_provider", lambda: _MockEmbedProvider())
    return proj


def _age(*paths, s=600):
    t = time.time() - s
    for p in paths:
        os.utime(p, (t, t))


def write_session(proj, sid, title):
    """Two turns, one Bash call with its result, a subagent and a tool-results file: a row in every table."""
    use = {"type": "tool_use", "id": f"t-{sid}", "name": "Bash", "input": {"command": f"profile {title}"}}
    result = {"type": "tool_result", "tool_use_id": f"t-{sid}", "content": "42 rows"}
    lines = [{"type": "custom-title", "customTitle": title},
             claude_user(f"why is {title} slow", session_id=sid),
             {**claude_assistant(session_id=sid),
              "message": {"role": "assistant", "content": [{"type": "text", "text": f"{title} reads every row"}, use]}},
             {**claude_user("", session_id=sid), "message": {"role": "user", "content": [result]}},
             claude_assistant(session_id=sid)]
    path = proj / f"{sid}.jsonl"
    path.write_text(make_claude_jsonl(lines))
    agent = proj / sid / "subagents" / f"agent-{sid}x.jsonl"
    agent.parent.mkdir(parents=True)
    agent.write_text(make_claude_jsonl([claude_user(f"look into {title}", session_id=sid)]))
    (proj / sid / "tool-results").mkdir()
    (proj / sid / "tool-results" / "out.txt").write_text("42 rows")
    _age(path, agent)
    return path


def rows(db, sid, word):
    """Rows of `sid` in every table a session writes, and FTS hits for a word only it contains."""
    out = {t: db.execute(f"SELECT COUNT(*) FROM {t} WHERE {col} = ?", (sid,)).fetchone()[0]
           for t, col in [("sessions", "session_id"), ("session_chunks", "session_id"), ("vec_chunks", "session_id"),
                          ("tool_events", "session_id"), ("agent_sessions", "parent_session_id")]}
    for fts in ("sessions_fts", "session_chunks_fts", "tool_events_fts", "agent_sessions_fts"):
        out[fts] = db.execute(f"SELECT COUNT(*) FROM {fts} WHERE {fts} MATCH ?", (f'"{word}"',)).fetchone()[0]
    return out


def test_forget_removes_every_row_of_the_session(db, projects):
    """vec_chunks has no delete trigger, so it is deleted explicitly; the FTS tables follow their triggers."""
    write_session(projects, "sess-a", "alphaTask")
    write_session(projects, "sess-b", "betaTask")
    scanner.scan(db)
    assert all(rows(db, "sess-a", "alphatask").values())
    other = rows(db, "sess-b", "betatask")

    out = forget.forget(db, forget.find(db, "alphaTask"))

    assert not any(rows(db, "sess-a", "alphatask").values())
    assert rows(db, "sess-b", "betatask") == other
    assert out.startswith("Forgot alphaTask (sess-a, proj,") and "1 subagent" in out


def test_forget_keeps_wiki_pages_that_cite_the_session(db, projects, tmp_path):
    write_session(projects, "sess-a", "alphaTask")
    scanner.scan(db)
    upsert_wiki_page(db, "alpha-notes", "concept", "Alpha notes", "body", tmp_path / "alpha.md", session_id="sess-a")
    forget.forget(db, forget.find(db, "alphaTask"))
    assert db.execute("SELECT session_id FROM wiki_pages WHERE slug = 'alpha-notes'").fetchone()[0] == "sess-a"


def test_forget_by_title_or_id_prefix_and_unknown_is_a_clear_error(db, projects):
    write_session(projects, "sess-a1", "alphaTask")
    scanner.scan(db)
    assert forget.find(db, "ALPHATASK").session_id == "sess-a1"
    assert forget.find(db, "sess-a").session_id == "sess-a1"
    with pytest.raises(NotFound, match="Closest titles: alphaTask"):
        forget.find(db, "alphaTsk")
    assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_a_title_or_prefix_that_matches_several_sessions_forgets_nothing(db, projects):
    """Resume takes the newest of several matches; a forget must not guess. 8 titles in the real index belonged
    to 2-4 sessions each on 2026-10-23 (MedullaDemo x4)."""
    write_session(projects, "sess-d1", "demo")
    write_session(projects, "sess-d2", "demo")
    scanner.scan(db)
    with pytest.raises(forget.Ambiguous) as e:
        forget.find(db, "demo")
    assert "sess-d1" in str(e.value) and "sess-d2" in str(e.value) and "nothing was forgotten" in str(e.value)
    with pytest.raises(forget.Ambiguous):
        forget.find(db, "sess-d")
    assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2


def test_a_forgotten_session_stays_out_of_every_scan_resume_and_the_stale_note(db, projects):
    """The transcript stays on disk, and a session with no rows reads as stale forever (scanner.py:136)."""
    path = write_session(projects, "sess-a", "alphaTask")
    scanner.scan(db)
    forget.forget(db, forget.find(db, "alphaTask"))

    assert scanner.scan(db)["forgotten"] == 1
    assert scanner.scan(db, force=True)["forgotten"] == 1
    assert scanner.index_session(db, path) == "skipped_forgotten"
    assert scanner.stale_sessions(db) == []
    with pytest.raises(NotFound, match="was forgotten on .*medulla forget --undo sess-a"):
        resume(db, "alphaTask")
    assert not any(rows(db, "sess-a", "alphatask").values())


def test_a_scan_does_not_parse_a_forgotten_transcript(db, projects, monkeypatch):
    """It has no rows, so without the check every scan would parse it again."""
    write_session(projects, "sess-a", "alphaTask")
    scanner.scan(db)
    forget.forget(db, forget.find(db, "alphaTask"))
    monkeypatch.setattr(scanner, "parse_session", lambda p: pytest.fail(f"parsed {p.name}"))
    assert scanner.scan(db, force=True)["forgotten"] == 1


def test_a_copy_of_a_forgotten_transcript_under_another_name_stays_out(db, projects):
    """One real transcript has a hand-made copy (medullaSprint3Backup.jsonl) holding the same session id."""
    path = write_session(projects, "sess-a", "alphaTask")
    scanner.scan(db)
    forget.forget(db, forget.find(db, "alphaTask"))
    copy = projects / "alphaBackup.jsonl"
    copy.write_text(path.read_text())
    _age(copy)
    assert scanner.scan(db)["forgotten"] == 2
    assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_forgetting_twice_says_so_and_the_list_shows_it(db, projects):
    write_session(projects, "sess-a", "alphaTask")
    scanner.scan(db)
    assert forget.listing(db) == "No forgotten sessions."
    forget.forget(db, forget.find(db, "alphaTask"))
    again = forget.forget(db, forget.find(db, "alphaTask"))
    assert again.startswith("alphaTask (sess-a, proj,") and "was already forgotten on" in again
    assert "1 forgotten session:" in forget.listing(db) and "alphaTask" in forget.listing(db)


def test_undo_puts_a_forgotten_session_back_on_the_next_scan(db, projects):
    write_session(projects, "sess-a", "alphaTask")
    scanner.scan(db)
    forget.forget(db, forget.find(db, "alphaTask"))
    assert "the next scan indexes it again" in forget.undo(db, "alphaTask")
    scanner.scan(db)
    assert all(rows(db, "sess-a", "alphatask").values())
    with pytest.raises(NotFound, match="not on the forgotten list"):
        forget.undo(db, "alphaTask")


def test_deleting_the_transcript_removes_its_file_and_folder_and_nothing_else(db, projects):
    write_session(projects, "sess-a", "alphaTask")
    write_session(projects, "sess-b", "betaTask")
    scanner.scan(db)
    target = forget.find(db, "alphaTask")
    report = forget.forget(db, target)
    assert "sess-a.jsonl, " in report and "sess-a/ (2 files)" in report
    before = set(projects.parent.rglob("*"))

    out = forget.delete_transcript(db, target)

    gone = before - set(projects.parent.rglob("*"))
    assert gone == {p for p in before if p.name.startswith("sess-a") or "/sess-a/" in str(p)}
    assert (projects / "memory" / "MEMORY.md").read_text() == "keep me"
    assert f"Deleted {projects / 'sess-a.jsonl'}" in out
    assert "(transcript deleted)" in forget.listing(db)
    assert "transcript is not on disk" in forget.undo(db, "sess-a")


def test_a_transcript_written_in_the_last_minute_is_not_deleted(db, projects):
    """The session doing the forgetting may be the one being forgotten."""
    path = write_session(projects, "sess-a", "alphaTask")
    scanner.scan(db)
    target = forget.find(db, "alphaTask")
    forget.forget(db, target)
    os.utime(path, None)
    with pytest.raises(forget.LiveTranscript, match="last minute"):
        forget.delete_transcript(db, target)
    assert path.exists() and (projects / "sess-a").is_dir()


def test_only_a_claude_transcript_named_by_a_plain_id_is_ever_offered_for_deletion(db, projects):
    """A Kiro session, or an id that is not a plain file name, has no transcript medulla will delete."""
    write_session(projects, "sess-a", "alphaTask")
    scanner.scan(db)
    target = forget.find(db, "alphaTask")
    assert forget.transcript(forget.Target(**{**vars(target), "source": "kiro"})) == []
    assert forget.transcript(forget.Target(**{**vars(target), "session_id": "../-proj-a/sess-a"})) == []
    shutil.rmtree(projects / "sess-a")
    assert forget.transcript(target) == [projects / "sess-a.jsonl"]
    (projects / "sess-a.jsonl").unlink()
    assert forget.forget(db, target).endswith("Transcript: none on disk.")
    assert forget.delete_transcript(db, target) == "Transcript: none on disk."
