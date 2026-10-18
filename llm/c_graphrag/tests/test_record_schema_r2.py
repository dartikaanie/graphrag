"""Record schema r2 (llm/record_schema.py) in Conditions A, C and D: run_id,
condition, record_schema_version, retrieval/generation/total latency at the
top level of every record; D's context items carry is_accepted. Fake LLM
client and fake retrieval -- no API call, no Neo4j, no FAISS. Condition B:
llm/b_rag/tests/test_b_record_schema_r2.py (needs the tiny KG corpus)."""

import json

import numpy as np
import pandas as pd
import pytest

import c_graphrag as cg
from llm.a_pure_llm import a_baseline_replication as a
from llm.d_lightrag import d_lightrag as d
from llm.manifest import write_manifest
from llm.record_schema import RECORD_SCHEMA_VERSION

RUN_ID = "11111111-2222-3333-4444-555555555555"


def _fake_llm(client, messages, model):
    return {"content": "Use x [SO-901].", "response_model": "fake", "request_params": {"model": model}}


def _sample():
    return pd.DataFrame([{"Id": 100, "Title": "How to x?", "Body": "<p>x</p>", "Tags": "<python>",
                          "AcceptedAnswerId": 101, "AcceptedAnswerBody": "do x", "n_tokens": 5,
                          "ViewCount": 3, "Score": 1}])


def _check_header(rec, condition, retrieval_is_none):
    assert rec["run_id"] == RUN_ID and rec["condition"] == condition
    assert rec["record_schema_version"] == RECORD_SCHEMA_VERSION == "r2"
    for key in ("generation_latency_sec", "total_latency_sec"):
        assert isinstance(rec[key], float) and rec[key] >= 0, key
    if retrieval_is_none:
        assert rec["retrieval_latency_sec"] is None
    else:
        assert isinstance(rec["retrieval_latency_sec"], float)
        assert rec["total_latency_sec"] >= rec["retrieval_latency_sec"]
    assert rec["total_latency_sec"] >= rec["generation_latency_sec"]
    # old fields stay
    for key in ("question_id", "llm_answer", "cosine_similarity", "config", "config_hash", "request_params"):
        assert key in rec, key


@pytest.fixture
def quiet(monkeypatch):
    for mod in (a, cg, d):
        monkeypatch.setattr(mod, "log", lambda *x, **k: None)
        monkeypatch.setattr(mod, "compute_similarity", lambda m, x, y: 0.5)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)


def test_condition_a_record_and_manifest(tmp_path, quiet):
    out = tmp_path / "a.jsonl"
    recs = a.process_sample(_sample(), None, _fake_llm, None, "fake", out, run_id=RUN_ID)
    _check_header(recs[0], "A", retrieval_is_none=True)
    assert json.loads(out.read_text().splitlines()[0])["run_id"] == RUN_ID
    m = json.loads(write_manifest(out, run_id=RUN_ID, run_label="A", config={}, started_at_utc="t0",
                                  finished_at_utc="t1", item_counts={}).read_text())
    assert m["run_id"] == RUN_ID and m["record_schema_version"] == "r2" and m["run_ids_in_output"] == [RUN_ID]


def test_run_id_defaults_to_a_fresh_uuid(tmp_path, quiet):
    recs = a.process_sample(_sample(), None, _fake_llm, None, "fake", tmp_path / "a.jsonl")
    import uuid

    uuid.UUID(recs[0]["run_id"])


def _c_candidates():
    return [{"via_question_id": 900, "via_question_title": "x?", "answer_id": 901, "answer_body": "do x",
             "answer_trust_score": 0.5, "is_accepted": True, "edge_weight": 1.0, "hop": 1,
             "rel_type": "HAS_ACCEPTED_ANSWER"}]


def test_condition_c_record(tmp_path, quiet, monkeypatch):
    monkeypatch.setattr(cg, "embed_query", lambda text, model: np.ones((1, 384), dtype="float32"))
    monkeypatch.setattr(cg, "anchor_via_vector_search", lambda *x, **k: [(900, 0.9)])
    monkeypatch.setattr(cg, "traverse_graph", lambda *x, **k: _c_candidates())
    monkeypatch.setattr(cg, "semantic_expansion", lambda *x, **k: [])
    out = tmp_path / "c.jsonl"
    recs, _ = cg.process_sample(_sample(), None, _fake_llm, None, None, "db", None, None, None, {}, {100: [101]},
                                5, 3, 3, 400, "fake", out, run_id=RUN_ID)
    _check_header(recs[0], "C", retrieval_is_none=False)
    assert recs[0]["retrieved_context"][0]["is_accepted"] is True


def test_condition_c_retrieval_only_record(tmp_path, quiet, monkeypatch):
    monkeypatch.setattr(cg, "embed_query", lambda text, model: np.ones((1, 384), dtype="float32"))
    monkeypatch.setattr(cg, "anchor_via_vector_search", lambda *x, **k: [(900, 0.9)])
    monkeypatch.setattr(cg, "traverse_graph", lambda *x, **k: _c_candidates())
    monkeypatch.setattr(cg, "semantic_expansion", lambda *x, **k: [])
    recs, _ = cg.process_sample(_sample(), None, None, None, None, "db", None, None, None, {}, {100: [101]},
                                5, 3, 3, 400, "fake", tmp_path / "c.jsonl", run_id=RUN_ID, retrieval_only=True)
    r = recs[0]
    assert (r["run_id"], r["condition"], r["generation_latency_sec"]) == (RUN_ID, "C", None)
    assert r["total_latency_sec"] >= r["retrieval_latency_sec"]


def test_condition_d_record_and_is_accepted(tmp_path, quiet, monkeypatch):
    monkeypatch.setattr(d, "embed_query", lambda text, model: np.ones((1, 384), dtype="float32"))
    monkeypatch.setattr(d, "anchor_via_vector_search", lambda *x, **k: [(900, 0.9)])
    low = [{"via_question_id": 900, "via_question_title": "x?", "answer_id": 901, "answer_body": "do x", "hop": 1,
            "rel_type": "HAS_ACCEPTED_ANSWER", "relevance_score": 0.9, "is_accepted": True}]
    high = [{"via_question_id": 902, "via_question_title": "y?", "answer_id": 903, "answer_body": "do y", "hop": 2,
             "rel_type": "EMBED_SIM", "relevance_score": 0.4, "is_accepted": False}]
    monkeypatch.setattr(d, "retrieve_low_level", lambda *x, **k: low)
    monkeypatch.setattr(d, "retrieve_high_level", lambda *x, **k: high)
    recs, _ = d.process_sample(_sample(), None, _fake_llm, None, None, "db", None, None, None, {}, {100: [101]},
                               5, 3, 3, 400, "fake", tmp_path / "d.jsonl", run_id=RUN_ID)
    _check_header(recs[0], "D", retrieval_is_none=False)
    assert [i["is_accepted"] for i in recs[0]["retrieved_context"]] == [True, False]


def test_d_queries_return_is_accepted_without_extra_query():
    import inspect

    for fn in (d.retrieve_low_level, d.retrieve_high_level):
        src = inspect.getsource(fn)
        assert "a.isAccepted AS is_accepted" in src and src.count("session.run(") == 1, fn.__name__
