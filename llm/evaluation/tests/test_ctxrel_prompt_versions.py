"""ctxrel-v2 is added as a separate prompt version; ctxrel-v1 must stay
byte-identical (docs/DECISION_C_SCORING.md, Amendment 2)."""

import hashlib
import json

import pytest

import ctxrel_prompt_v2
import llm_judge_context_relevance_v1 as ctxrel

# sha256 of ctxrel-v1's messages for a fixed input, computed from the module
# as committed BEFORE ctxrel-v2 was added -- pins v1's prompt text.
V1_MESSAGES_SHA256 = "cfe5b1bbc511903cc0bcc7ca4d2a72a155dd103d975bccd3841465e6a1edaac7"


def _sha(messages):
    return hashlib.sha256(json.dumps(messages).encode()).hexdigest()


def test_v1_prompt_unchanged():
    assert ctxrel.PROMPT_VERSION == "ctxrel-v1"
    assert _sha(ctxrel.build_messages("T", "B", "a, b", "Q: x\nA: y")) == V1_MESSAGES_SHA256
    assert ctxrel.get_prompt_builder("ctxrel-v1") is ctxrel.build_messages


def test_v2_registered_and_distinct():
    assert ctxrel_prompt_v2.PROMPT_VERSION == "ctxrel-v2"
    assert ctxrel.get_prompt_builder("ctxrel-v2") is ctxrel_prompt_v2.build_messages
    v1 = ctxrel.build_messages("T", "B", "a, b", "Q: x\nA: y")
    v2 = ctxrel_prompt_v2.build_messages("T", "B", "a, b", "Q: x\nA: y")
    assert v1 != v2


def test_v2_same_inputs_and_output_contract_as_v1():
    msgs = ctxrel_prompt_v2.build_messages("My title", "My body", "python, flask", "Q: t\nA: a")
    user = msgs[1]["content"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert user.startswith("Question title: My title\nTags: python, flask\nQuestion body: My body\n")
    assert "Retrieved item:\nQ: t\nA: a" in user
    assert '"label": "RELEVANT" | "PARTIAL" | "IRRELEVANT"' in user
    assert "a workaround, the cause" in user
    assert "Sharing a technology, library, or keyword with the question is NOT enough on its own." in user
    parsed = ctxrel.parse_judgment('{"label": "PARTIAL", "reason": "x"}')
    assert parsed == {"label": "PARTIAL", "reason": "x", "parse_error": False}


def test_unknown_version_raises():
    with pytest.raises(ValueError):
        ctxrel.get_prompt_builder("ctxrel-v3")


def test_load_ctxrel_records_filters_by_prompt_version(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"question_id": 1, "answer_id": 2, "text_hash": "h", "label": "PARTIAL", "prompt_version": "ctxrel-v1"},
        {"question_id": 1, "answer_id": 2, "text_hash": "h", "label": "RELEVANT", "prompt_version": "ctxrel-v2"},
    ]))
    assert ctxrel.load_ctxrel_records([p], prompt_version="ctxrel-v1")[(1, 2, "h")]["label"] == "PARTIAL"
    assert ctxrel.load_ctxrel_records([p], prompt_version="ctxrel-v2")[(1, 2, "h")]["label"] == "RELEVANT"
