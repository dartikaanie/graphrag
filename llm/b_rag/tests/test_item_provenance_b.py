"""Condition B item provenance (llm/common/item_provenance.py), on the tiny
offline KG corpus with the real MiniLM (no Neo4j, no LLM).

* Regression: retrieve_context_kg()'s items, minus the keys provenance added,
  equal -- ids, order, rounded scores, text hashes -- what the code BEFORE
  provenance returned (fixtures/b_retrieval_golden_pre_provenance.json,
  generated from the committed HEAD b_condition_b_rag.py by collect()).
* Every item is vector_only: hop None, path = the retrieved Answer only,
  anchor_id None, trust_weight None, and validates against the schema.
* The prompt B sends is unchanged by the extra fields."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from llm.b_rag import b_condition_b_rag as b
from llm.b_rag.kg_corpus import load_kg_answer_index
from llm.common import item_provenance as ip
from llm.common.leakage import LEGACY_CONTEXT
from llm.prompts import build_rag_messages

GOLDEN = Path(__file__).parent / "fixtures" / "b_retrieval_golden_pre_provenance.json"
QUESTIONS = {501: "How to git rebase conflict", 507: "How to docker volume mount", 512: "How to sql join null"}


def collect(b_module, kg, model, top_k=5):
    """Items of retrieve_context_kg() per query, in a comparable (JSON) form."""
    out = {}
    for qid, text in QUESTIONS.items():
        thread = kg.thread_answer_ids([qid]).get(qid, [])
        exclusion = LEGACY_CONTEXT.exclusion_for(qid, thread, thread[0] if thread else 0)
        items = b_module.retrieve_context_kg(model, kg, text, top_k, exclusion)
        out[str(qid)] = [{**{k: (round(v, 5) if k == "score" else v) for k, v in it.items() if k != "chunk_text"},
                          "chunk_sha": hashlib.sha256(it["chunk_text"].encode()).hexdigest()} for it in items]
    return out


def _kg(tiny_corpus, minilm):
    root, _, _ = tiny_corpus
    return load_kg_answer_index(root, 400, "all-MiniLM-L6-v2", log=lambda *a: None)


def test_retrieval_identical_to_pre_provenance_code(tiny_corpus, minilm):
    new = collect(b, _kg(tiny_corpus, minilm), minilm)
    legacy = {q: [{k: v for k, v in ip.legacy_view(it, drop_rank=True).items()
                   if k not in ("source_stage", "hop", "rel_type", "trust_weight")} for it in items]
              for q, items in new.items()}
    assert legacy == json.loads(GOLDEN.read_text())


def test_items_are_vector_only_and_validate(tiny_corpus, minilm):
    kg = _kg(tiny_corpus, minilm)
    items = b.retrieve_context_kg(minilm, kg, "How to git rebase conflict", 5,
                                  LEGACY_CONTEXT.exclusion_for(501, kg.thread_answer_ids([501])[501], 1001))
    assert len(items) == 5
    for rank, item in enumerate(items, start=1):
        assert ip.validate_item(item) == []
        assert item["rank"] == rank and item["source_stage"] == "vector_only"
        assert item["hop"] is None and item["anchor_id"] is None and item["trust_weight"] is None
        assert item["path"] == [{"node_type": "Answer", "node_id": item["answer_id"], "edge_type_to_next": None}]
        assert item["via_question_id"] == item["question_id"] and item["alt_paths"] == []
        assert "score" in item and item["is_accepted"] in (True, False)


def test_record_carries_provenance_version_and_the_prompt_is_unchanged(tiny_corpus, minilm, monkeypatch, tmp_path):
    kg = _kg(tiny_corpus, minilm)
    monkeypatch.setattr(b, "log", lambda *a, **k: None)
    monkeypatch.setattr(b.time, "sleep", lambda s: None)
    sample = pd.DataFrame([{"Id": 503, "Title": "How to git rebase conflict (case 503)?", "Body": "<p>?</p>",
                            "Tags": "<git>", "AcceptedAnswerId": 1013, "AcceptedAnswerBody": "x", "n_tokens": 5,
                            "ViewCount": 1, "Score": 1}])
    rec = b.process_sample(sample, None, lambda c, m, mo: {"content": "ok [SO-520]"}, minilm, kg, None, 5, "fake",
                           tmp_path / "o.jsonl", require_citation=True)[0]
    assert ip.validate_record(rec) == []
    bare = [{"question_id": i["question_id"], "chunk_text": i["chunk_text"]} for i in rec["retrieved_context"]]
    assert rec["prompt_messages"] == build_rag_messages("How to git rebase conflict (case 503)?", "<p>?</p>", "<git>",
                                                        bare, require_citation=True, require_grounding=False)
