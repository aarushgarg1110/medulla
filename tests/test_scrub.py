"""Tests for medulla.episodic.scrub — secret redaction before storage."""
from medulla.episodic.scrub import scrub_secrets, REDACTED


def test_postgres_url_password_redacted():
    s = scrub_secrets("ATTACH 'postgresql://ldodda:hunter2@acas-host:5432/acas'")
    assert "hunter2" not in s
    assert REDACTED in s
    assert "ldodda" in s and "acas-host" in s   # user/host kept


def test_generic_scheme_url_password_redacted():
    assert "s3cret" not in scrub_secrets("mysql://user:s3cret@db/schema")


def test_aws_access_key_redacted():
    assert "AKIAIOSFODNN7EXAMPLE" not in scrub_secrets("aws key AKIAIOSFODNN7EXAMPLE here")


def test_aws_secret_access_key_assignment():
    assert "abc123def" not in scrub_secrets("aws_secret_access_key=abc123def")


def test_literal_pw_assignment_redacted():
    assert "topsecret" not in scrub_secrets("PW=topsecret duckdb ...")


def test_command_substitution_left_intact():
    # $(...) fetches the secret at runtime — not a literal, keep it queryable
    cmd = 'PW=$(aws secretsmanager get-secret-value --secret-id ldodda_db/acas)'
    assert scrub_secrets(cmd) == cmd


def test_env_var_reference_left_intact():
    cmd = "PW=$DB_PASSWORD psql -h host"
    assert scrub_secrets(cmd) == cmd


def test_bearer_token_redacted():
    assert "abcdef12345678" not in scrub_secrets("Authorization: Bearer abcdef12345678")


def test_pem_block_redacted():
    s = scrub_secrets("-----BEGIN PRIVATE KEY-----\nMIIsecret\n-----END PRIVATE KEY-----")
    assert "MIIsecret" not in s
    assert "BEGIN PRIVATE KEY" in s   # markers kept


def test_clean_command_unchanged():
    t = "duckdb -c \"SELECT * FROM 'x.csv' LIMIT 2\""
    assert scrub_secrets(t) == t


def test_empty_string():
    assert scrub_secrets("") == ""
