"""Tests for conversation-text sanitization in the parser."""
from medulla.episodic.parser import _sanitize_text, _extract_user_text, _extract_assistant_text


def test_strips_local_command_caveat():
    t = _sanitize_text("<local-command-caveat>Caveat: blah blah</local-command-caveat>")
    assert t == ""


def test_strips_command_wrappers():
    t = _sanitize_text("<command-name>/exit</command-name>\n"
                       "<command-message>exit</command-message>\n"
                       "<command-args></command-args>")
    assert t == ""


def test_strips_local_command_stdout():
    assert _sanitize_text("<local-command-stdout>Catch you later!</local-command-stdout>") == ""


def test_strips_system_reminder():
    t = _sanitize_text('<system-reminder>\nThe user named this session "x".\n</system-reminder>')
    assert t == ""


def test_strips_image_markers():
    assert _sanitize_text("look at this [Image #1] and [Image #2]") == "look at this  and"
    assert _sanitize_text("[Image: source: /Users/a/x.png]") == ""


def test_preserves_real_content_and_internal_newlines():
    src = "Here is the plan:\n\n1. step one\n2. step two"
    assert _sanitize_text(src) == src


def test_mixed_content_keeps_real_text():
    src = ("<system-reminder>meta</system-reminder>\n"
           "what were the logD outliers?\n"
           "<command-args></command-args>")
    assert _sanitize_text(src) == "what were the logD outliers?"


def test_empty_and_none_safe():
    assert _sanitize_text("") == ""
    assert _sanitize_text("   \n  ") == ""


def test_extract_user_text_drops_pure_noise_message():
    # a user turn that is only a slash command → empty after sanitize → dropped
    msg = {"role": "user", "content": "<command-name>/model</command-name>"}
    assert _extract_user_text(msg) == ""


def test_extract_user_text_sanitizes_list_content():
    msg = {"role": "user", "content": [
        {"type": "text", "text": "<local-command-caveat>c</local-command-caveat> real question here"}]}
    assert _extract_user_text(msg) == "real question here"


def test_extract_assistant_text_string_content_sanitized():
    msg = {"role": "assistant", "content": "<system-reminder>x</system-reminder> the answer"}
    assert _extract_assistant_text(msg) == "the answer"


def test_kiro_parser_sanitizes(tmp_path):
    """Kiro sessions run through the same sanitizer (no-op today, future-proof)."""
    import json
    from medulla.episodic.parser import parse_session
    rec = {"version": "v1", "kind": "Prompt", "data": {
        "meta": {"timestamp": 1_700_000_000},
        "content": [{"kind": "text",
                     "data": "<system-reminder>meta</system-reminder> real kiro question here"}]}}
    p = tmp_path / "k1.jsonl"
    p.write_text(json.dumps(rec))
    ps = parse_session(p)
    assert ps is not None and ps.source == "kiro"
    assert ps.messages == ["real kiro question here"]
    assert "system-reminder" not in ps.first_message
