"""2026-10-07 follow-up: judge_resume_key()/JUDGE_RESUME_FIELDS gained
parse_version/max_tokens so a judge-v2 parse-error record becomes
re-judgeable once the parser (or an explicit max_tokens cap) changes,
without breaking judge-v1 resume (which has neither field at all)."""

from _judge_common import JUDGE_RESUME_FIELDS, judge_resume_key


def _record(**overrides):
    base = {
        "run_id": "r1", "config_hash": "h1", "question_id": 1, "judge_id": "primary",
        "judge_model": "gpt-4o-mini", "prompt_version": "judge-v2", "blinding_version": "blind-v2",
    }
    base.update(overrides)
    return base


def test_resume_key_includes_parse_version_and_max_tokens():
    assert "parse_version" in JUDGE_RESUME_FIELDS
    assert "max_tokens" in JUDGE_RESUME_FIELDS


def test_resume_key_differs_when_parse_version_changes():
    old = judge_resume_key(_record(parse_version="pv1"))
    new = judge_resume_key(_record(parse_version="pv2"))
    assert old != new


def test_resume_key_differs_when_max_tokens_changes():
    old = judge_resume_key(_record(max_tokens=None))
    new = judge_resume_key(_record(max_tokens=1024))
    assert old != new


def test_resume_key_same_when_parse_version_and_max_tokens_unchanged():
    a = judge_resume_key(_record(parse_version="pv1", max_tokens=None))
    b = judge_resume_key(_record(parse_version="pv1", max_tokens=None))
    assert a == b


def test_resume_key_for_v1_style_record_missing_both_fields_is_stable():
    """judge-v1 records never carry parse_version/max_tokens at all --
    both sides of any comparison get None uniformly, so this must never
    cause a spurious cache-miss for v1."""
    v1_record = _record(prompt_version="judge-v1", blinding_version="blind-v1")
    v1_record.pop("parse_version", None)
    v1_record.pop("max_tokens", None)
    a = judge_resume_key(v1_record)
    b = judge_resume_key(dict(v1_record))
    assert a == b
    assert a[JUDGE_RESUME_FIELDS.index("parse_version")] is None
    assert a[JUDGE_RESUME_FIELDS.index("max_tokens")] is None
