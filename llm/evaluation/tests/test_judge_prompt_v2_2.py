"""judge-v2.2 (Amendment 11): prompt, input preparation, parser, derived rules and runner -- fake client only."""

import hashlib
import json

import pytest

import judge_prompt_v2_1 as v21
import judge_prompt_v2_2 as v22
import llm_judge_hallucination_v2_2 as runner

CTX = [{"question_id": 11, "answer_id": 101, "chunk_text": "Q: parse dates\nA: use fromisoformat"},
       {"question_id": 22, "answer_id": 202, "chunk_text": "Q: other\nA: use strptime"},
       {"question_id": 11, "answer_id": 103, "chunk_text": "Q: parse dates\nA: strip the Z first"}]
ANSWER = ("Use datetime.fromisoformat(s) [SO-11]. The retrieved sources do not cover time zones. "
          "Alternatively call strptime [SO-22]. Also try parse_iso [SO-99].")


def _claim(verdict, sbc="YES", items=(), cis=None, severity=None):
    return {"claim": "c", "verdict": verdict, "severity": severity, "reference_conflict": False,
            "supported_by_context": sbc, "cited_items": list(items), "cited_item_supports": cis, "evidence": "e"}


def _raw(claims, attempted=True, approach="SAME", label="FAKTUAL"):
    return json.dumps({"answer_attempted": attempted, "completeness": "FULL", "approach_match": approach,
                       "claims": claims, "reasoning": "r", "label": label})


# -- v2.1 untouched ---------------------------------------------------------

def test_v2_1_prompt_and_messages_unchanged():
    assert v21.PROMPT_VERSION == "judge-v2.1" and v21.BLINDING_VERSION == "blind-v2"
    assert hashlib.sha256(v21.SYSTEM_PROMPT.encode()).hexdigest() == \
        "d86776bc6c4480b5e9218a066e06eaa7605db10b48a6ed98c6f7b7079831c63d"
    assert hashlib.sha256(repr(v21.build_messages("T", "B", "py", "R", "C [SO-1]")).encode()).hexdigest() == \
        "d6f37ce8764e5562c4f0687483f2b3edcb60b75ef019a71cb1103cf422dd4f7b"


def test_v2_2_prompt_extends_v2_1():
    assert (v22.PROMPT_VERSION, v22.BLINDING_VERSION, v22.REQUEST_TEMPERATURE) == ("judge-v2.2", "blind-v3", 0.0)
    p = v22.SYSTEM_PROMPT
    assert "Keep the qualifiers" in p and "A solution that differs from the reference is not an error" in p
    assert "## Step 2b - Retrieved context and citations" in p and "approach_match" in p
    assert p.count("## Output") == 1 and '"supported_by_context": "YES | NO | N/A"' in p
    assert "[SO-" not in p


# -- input: context block, citation map, blinding ---------------------------

def test_citation_map_and_cited_statements():
    prep = v22.prepare_judge_input(ANSWER, CTX, has_context=True)
    assert prep["citation_map"] == {"11": ["K1", "K3"], "22": ["K2"]}
    s = prep["cited_statements"]
    assert [x["cited_items"] for x in s] == [["K1", "K3"], ["K2"], []]
    assert [x["n_unmatched"] for x in s] == [0, 0, 1]
    assert prep["offered_items"] == ["K1", "K2", "K3"]
    assert '1. "Use datetime.fromisoformat(s)." -> [K1], [K3]' in prep["citations_block"]
    assert "(a cited source not among the context items)" in prep["citations_block"]
    assert "[K1]\nQ: parse dates" in prep["context_block"] and prep["n_context_items"] == 3


def test_messages_hide_so_ids_and_use_blind_v3():
    msgs = v22.build_messages("T", "B", "py", "R", ANSWER, CTX, has_context=True)
    user = msgs[1]["content"]
    assert "[SO-" not in user and "101" not in user and "SO-11" not in user
    assert "[catatan: informasi tidak tersedia] time zones" in user
    assert user.index("### RETRIEVED CONTEXT") < user.index("### CITATIONS") < user.index("### CANDIDATE ANSWER")


def test_condition_a_gets_explicit_no_context():
    prep = v22.prepare_judge_input("Use fromisoformat [SO-11].", None, has_context=False)
    assert prep["context_block"] == v22.NO_CONTEXT_TEXT and prep["citations_block"] == v22.NO_CITATIONS_TEXT
    assert prep["citation_map"] == {} and prep["n_context_items"] == 0


def test_context_items_cut_at_the_documented_limit():
    long = [{"question_id": 1, "chunk_text": "x" * 3000}]
    block = v22.prepare_judge_input("a", long, True)["context_block"]
    assert block.count("x") == v22.CONTEXT_ITEM_MAX_CHARS and block.endswith("[...truncated]")


# -- parser + derived rules ----------------------------------------------------

def test_parse_new_schema_and_derived_metrics():
    raw = _raw([_claim("SUPPORTED", "YES", ["K1"], "YES"), _claim("CONTRADICTED", "NO", ["K2"], "NO", "CORE"),
                _claim("UNVERIFIABLE", "NO"), _claim("SUPPORTED", "YES")], approach="DIFFERENT_VALID",
               label="HALUSINASI_PENUH")
    rec = v22.parse_judgment_v2_2(raw, has_context=True, offered_items=["K1", "K2", "K3"])
    assert rec["parse_error"] is False and rec["prompt_version"] == "judge-v2.2"
    assert rec["approach_match"] == "DIFFERENT_VALID"
    assert rec["faithfulness"] == 0.5             # 2 of 4 claims supported by context
    assert rec["citation_support"] == 0.5         # 1 of 2 cited claims
    assert rec["n_cited_claims"] == 2
    from judge_binary import derive_hallucinated
    assert derive_hallucinated(rec) is True       # >= 1 CONTRADICTED, independent of approach_match


def test_approach_match_never_changes_hallucinated():
    from judge_binary import derive_hallucinated
    for approach in v22.APPROACH_VALUES:
        rec = v22.parse_judgment_v2_2(_raw([_claim("UNVERIFIABLE", "NO")], approach=approach), True)
        assert derive_hallucinated(rec) is False


def test_condition_a_faithfulness_is_null_and_context_fields_na():
    rec = v22.parse_judgment_v2_2(_raw([_claim("SUPPORTED", "YES")]), has_context=False)
    assert rec["faithfulness"] is None and rec["citation_support"] is None
    assert rec["claims"][0]["supported_by_context"] == "N/A"
    assert "supported_by_context_without_context" in rec["normalized_fields"]


def test_no_citation_gives_null_citation_support_and_abstain_nulls():
    rec = v22.parse_judgment_v2_2(_raw([_claim("SUPPORTED", "YES")]), True)
    assert rec["citation_support"] is None and rec["faithfulness"] == 1.0
    ab = v22.parse_judgment_v2_2(_raw([], attempted=False, approach=None, label="ABSTAIN"), True)
    assert ab["approach_match"] is None and ab["faithfulness"] is None and ab["label"] == "ABSTAIN"


@pytest.mark.parametrize("claims, kw", [
    ([_claim("SUPPORTED", "N/A")], {}),                          # N/A with context
    ([_claim("SUPPORTED", None)], {}),                           # missing supported_by_context
    ([_claim("SUPPORTED", "YES", ["K1"], None)], {}),            # cited claim without cited_item_supports
    ([_claim("SUPPORTED", "YES")], {"approach": "MAYBE"}),       # invalid approach_match
    ([_claim("SUPPORTED", "YES")], {"approach": None}),          # attempted answer without approach_match
])
def test_parse_errors(claims, kw):
    assert v22.parse_judgment_v2_2(_raw(claims, **kw), True, ["K1"])["parse_error"] is True


def test_unoffered_cited_items_are_dropped_and_recorded():
    rec = v22.parse_judgment_v2_2(_raw([_claim("SUPPORTED", "YES", ["K9"], "YES")]), True, ["K1"])
    assert rec["claims"][0]["cited_items"] == [] and rec["claims"][0]["cited_item_supports"] is None
    assert "cited_item_not_offered" in rec["normalized_fields"]


# -- runner (fake client) -----------------------------------------------------

class _FakeConfig:
    model = "fake-model"
    base_url = None


def _item(condition="C", candidate=ANSWER, ctx=CTX):
    return {"run_id": f"{condition}-1", "condition": condition, "run_label": condition, "fusion_mode": None,
            "retrieval_version": None, "source_file": "/fake.jsonl", "question_id": 1, "title": "t", "tags": "",
            "candidate": candidate, "ground_truth_answer_html": "ref", "retrieved_context": ctx}


def _payload(raw, finish="stop"):
    return {"raw": raw, "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.1, "response_id": "x",
            "response_model": "fake-model", "response_created": 1, "system_fingerprint": None,
            "finish_reason": finish, "usage_full": {}, "request_params": {}}


def test_runner_record_has_versions_context_and_derived_fields(monkeypatch):
    sent = {}

    def fake_call(client, config, messages, temperature, log, max_tokens):
        sent.update(temperature=temperature, max_tokens=max_tokens, user=messages[1]["content"])
        return _payload(_raw([_claim("SUPPORTED", "YES", ["K1"], "YES"), _claim("FABRICATED", "NO", [], None, "MINOR")],
                             label="HALUSINASI_SEBAGIAN")), []

    monkeypatch.setattr(runner, "call_judge_llm_with_retry", fake_call)
    rec = runner.judge_one(_item(), "primary", None, _FakeConfig(), "body", "ref")
    assert (sent["temperature"], sent["max_tokens"]) == (0.0, 2000)
    assert (rec["prompt_version"], rec["judge_version"], rec["blinding_version"]) == ("judge-v2.2",) * 2 + ("blind-v3",)
    assert rec["temperature"] == 0.0 and rec["has_context"] is True and rec["n_context_items"] == 3
    assert rec["citation_map"] == {"11": ["K1", "K3"], "22": ["K2"]}
    assert rec["hallucinated"] is True and rec["severity_label"] == "SEBAGIAN"
    assert rec["faithfulness"] == 0.5 and rec["citation_support"] == 1.0 and rec["approach_match"] == "SAME"
    assert "[SO-" not in sent["user"]

    monkeypatch.setattr(runner, "call_judge_llm_with_retry", lambda *a: (_payload(_raw([_claim("SUPPORTED", "N/A")])), []))
    a = runner.judge_one(_item("A", "Use fromisoformat.", []), "primary", None, _FakeConfig(), "b", "r")
    assert a["has_context"] is False and a["faithfulness"] is None and a["parse_error"] is False

    monkeypatch.setattr(runner, "call_judge_llm_with_retry", lambda *a: (None, [{"error": "boom"}]))
    failed = runner.judge_one(_item(), "primary", None, _FakeConfig(), "b", "r")
    assert failed["call_failed"] is True and failed["faithfulness"] is None and failed["max_tokens"] == 2000
