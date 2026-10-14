"""Rule 3 / registry integrity for the Process page: rationale text only
ever comes from an existing document, missing rationale is flagged (not
filled), and every pointer in the registry resolves."""

import dataclasses

from fastapi.testclient import TestClient

from app.main import app
from app.process import registry, sources
from app.process.decisions import parse_decisions
from app.process.registry import PARAMETERS, STATUSES, STEPS

client = TestClient(app)


def test_registry_holds_pointers_not_values():
    fields = {f.name for f in dataclasses.fields(registry.Param)}
    assert "value" not in fields
    assert "text" not in {f.name for f in dataclasses.fields(registry.Quote)}


def test_keys_unique_and_steps_known():
    keys = [p.key for p in PARAMETERS]
    assert len(keys) == len(set(keys))
    step_keys = {s.key for s in STEPS}
    assert {p.step for p in PARAMETERS} <= step_keys
    assert [s.number for s in STEPS] == list(range(1, len(STEPS) + 1))


def test_statuses_and_kinds_valid():
    for p in PARAMETERS:
        assert p.status in STATUSES, p.key
        assert p.kind in ("parameter", "result"), p.key
        if p.kind == "result":
            assert p.rationale is None, p.key


def test_every_quote_resolves_to_exactly_one_line():
    quotes = [q for p in PARAMETERS for q in (p.rationale, p.spec) if q]
    quotes += [q for s in STEPS for q in s.quotes]
    for q in quotes:
        out = sources.quote(q.file, q.needle)
        assert out["text"], q


def test_quoted_text_is_verbatim_from_the_source():
    """Each quote's text lines appear in the source file (modulo indentation
    and the comment marker) -- nothing paraphrased."""
    for p in PARAMETERS:
        for q in (p.rationale, p.spec):
            if not q:
                continue
            out = sources.quote(q.file, q.needle)
            src = sources.read_text(q.file)
            for line in out["text"].splitlines():
                assert line.strip() in src, (p.key, line)


def test_modules_and_used_by_exist():
    for s in STEPS:
        for m in s.modules:
            assert sources.repo_path(m).exists(), (s.key, m)
    for p in PARAMETERS:
        for f in p.used_by:
            assert sources.repo_path(f).exists(), (p.key, f)


def test_set_by_refers_to_a_parsed_decision():
    ids = {d["id"] for d in parse_decisions()}
    for p in PARAMETERS:
        if p.set_by:
            assert p.set_by in ids, p.key


def test_missing_rationale_is_flagged_not_filled():
    body = client.get("/api/process/parameters").json()
    rows = {r["key"]: r for r in body["parameters"]}
    expected = sorted(p.key for p in PARAMETERS if p.kind == "parameter" and p.rationale is None)
    assert sorted(body["undocumented"]) == expected
    for key in expected:
        assert rows[key]["rationale"] is None
        assert rows[key]["rationale_documented"] is False
    documented = [p for p in PARAMETERS if p.rationale is not None]
    for p in documented:
        assert rows[p.key]["rationale"]["file"] == p.rationale.file
        assert rows[p.key]["rationale_documented"] is True


def test_decisions_parsed_in_date_order_with_amendments():
    decisions = parse_decisions()
    assert decisions[0]["id"] == "decision"
    numbers = [d["number"] for d in decisions[1:]]
    assert numbers == sorted(numbers) and numbers[0] == 1
    dates = [d["date"] for d in decisions]
    assert dates == sorted(dates)
    text = sources.read_text("docs/DECISION_C_SCORING.md")
    for d in decisions[1:]:
        assert f"## Amendment {d['number']} ({d['date']}): {d['title']}" in text


def test_decision_parser_on_a_fake_document():
    doc = (
        "# Decision record: X\n\nDate written: 2026-01-01\n\n## Problem\n- p\n\n---\n\n"
        "## Amendment 1 (2026-01-02): first\n\n### Why\n- because\n\n### Record\n"
        "- DB update applied: 2026-01-03 at `abc1234`\n\n---\n\n## Outcome\nx\n"
    )
    out = parse_decisions(doc)
    assert [d["id"] for d in out] == ["decision", "amendment-1"]
    a1 = out[1]
    assert a1["why"]["text"] == "- because"
    assert a1["commits"] == ["abc1234"]
    assert a1["applied"] == ["DB update applied: 2026-01-03 at `abc1234`"]
