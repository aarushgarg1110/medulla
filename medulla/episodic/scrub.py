"""Redact secrets from tool commands/output before they are stored.

Shell commands and their output routinely contain credentials — inline Postgres
passwords, AWS keys, bearer tokens, `PW=...` assignments. Backfilling tool_events
into ~/.medulla/medulla.db would persist those verbatim, so every harvested
command/output string is scrubbed first.

Note: `PW=$(aws secretsmanager ...)` is a command *substitution* — the secret is
fetched at runtime and is not in the text, so it is intentionally left intact.
Only literal secrets are redacted.
"""
from __future__ import annotations

import re

REDACTED = "<REDACTED>"

_PATTERNS: list[tuple[re.Pattern, str]] = [
    # postgres://user:PASSWORD@host  → redact the password
    (re.compile(r"(postgres(?:ql)?://[^:/\s]+:)([^@\s]+)(@)"), rf"\1{REDACTED}\3"),
    # generic scheme://user:PASSWORD@host (mysql, mongodb, redis, amqp, ...)
    (re.compile(r"([a-z][a-z0-9+.\-]*://[^:/\s]+:)([^@\s]+)(@)"), rf"\1{REDACTED}\3"),
    # AWS access key id
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<AWS_KEY>"),
    # aws secret access key = value
    (re.compile(r"(aws_secret_access_key\s*=\s*)(\S+)", re.I), rf"\1{REDACTED}"),
    # KEY=value where value is a LITERAL secret (not a $(...) substitution or $VAR)
    (re.compile(
        r"\b((?:PW|PASS|PASSWORD|PASSWD|TOKEN|SECRET|API_?KEY|ACCESS_?KEY)\s*=\s*)"
        r"(?![$'\"]?\$)([^\s'\"]+)", re.I), rf"\1{REDACTED}"),
    # Bearer / Authorization tokens
    (re.compile(r"(Bearer\s+)([A-Za-z0-9._\-]{8,})"), rf"\1{REDACTED}"),
    # PEM private key blocks
    (re.compile(r"(-----BEGIN [^-]+-----).*?(-----END [^-]+-----)", re.S), rf"\1{REDACTED}\2"),
]


def scrub_secrets(text: str) -> str:
    """Return text with literal secrets replaced by placeholders."""
    if not text:
        return text
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text
