"""Rule 6: secrets never reach the Process page."""

import pytest

from app.process.redact import MASK, redact, redact_obj

SECRET_CASES = [
    ("OPENAI_API_KEY=sk-proj-abcDEF1234567890xyz", "sk-proj-abcDEF1234567890xyz"),
    ("ANTHROPIC_API_KEY: sk-ant-api03-ABCDEFGHIJKLMNOP", "sk-ant-api03-ABCDEFGHIJKLMNOP"),
    ('{"api_key": "deepinfra-secret-value-123"}', "deepinfra-secret-value-123"),
    ("NEO4J_PASSWORD=hunter2hunter2", "hunter2hunter2"),
    ("password: 'my pass'", "my"),
    ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "eyJhbGciOiJIUzI1NiJ9.payload.sig"),
    ("headers={'Authorization': 'Basic dXNlcjpwYXNz'}", "dXNlcjpwYXNz"),
    ("connecting to neo4j://neo4j:s3cr3t@localhost:7687", "s3cr3t"),
    ("postgresql://admin:pa55word@db.example.com/x", "pa55word"),
    ("HF_TOKEN=hf_abcdefghijklmnopqrstuvwx", "hf_abcdefghijklmnopqrstuvwx"),
    ("token = ghp_abcdefghijklmnopqrstuvwxyz123456", "ghp_abcdefghijklmnopqrstuvwxyz123456"),
    ("aws AKIAABCDEFGHIJKLMNOP used", "AKIAABCDEFGHIJKLMNOP"),
    ("client_secret=abc123def456", "abc123def456"),
    ("log line with a bare key sk-1234567890abcdefgh in it", "sk-1234567890abcdefgh"),
]


@pytest.mark.parametrize("text,secret", SECRET_CASES)
def test_secrets_are_masked(text, secret):
    out = redact(text)
    assert secret not in out
    assert MASK in out


@pytest.mark.parametrize("text", [
    "max_tokens=1500",
    "TOKEN_CHUNK_LIMIT=400",
    "total_tokens: 12345",
    "Connected to Neo4j: bolt://localhost:7687 | database: graphrag",
    "P99 score -- Answer: 46.0 | Question: 30.0",
    "commit 6a42942 at 2026-10-14T15:56:53+07:00",
])
def test_ordinary_text_is_unchanged(text):
    assert redact(text) == text


def test_redact_obj_is_recursive():
    obj = {"a": ["x", {"b": "api_key=abcdef123456"}], "n": 3, "t": ("password=zzz9999",)}
    out = redact_obj(obj)
    assert "abcdef123456" not in str(out) and "zzz9999" not in str(out)
    assert out["n"] == 3


def test_every_process_endpoint_redacts(monkeypatch):
    """A secret planted in a log tail comes back masked through the router."""
    from fastapi.testclient import TestClient

    from app.main import app
    from app.process import logs

    monkeypatch.setattr(logs, "read_tail", lambda log_id, lines=200: {"text": "OPENAI_API_KEY=sk-live-abcdef1234567890"})
    body = TestClient(app).get("/api/process/logs/kg_logs/x.log").json()
    assert "sk-live-abcdef1234567890" not in body["text"]
