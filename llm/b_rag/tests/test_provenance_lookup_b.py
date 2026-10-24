"""End to end for Condition B on a 12-question run over the tiny offline KG corpus (real MiniLM,
real process_sample, fake LLM that cites the first two retrieved sources): the run file is found by
run_id, every cited source is matched to its retrieved item, B says it has no KG path, and the
record's provenance validates. Also: the same run with provenance stripped (an old record)."""

import json

import pandas as pd

from llm.b_rag import b_condition_b_rag as b
from llm.b_rag.kg_corpus import load_kg_answer_index
from llm.common import item_provenance as ip
from llm.provenance_lookup import lookup_provenance, render_lookup

RUN_ID = "bbbbbbbb-2222-3333-4444-555555555555"


def _run(tiny_corpus, minilm, monkeypatch, tmp_path):
    root, _, _ = tiny_corpus
    kg = load_kg_answer_index(root, 400, "all-MiniLM-L6-v2", log=lambda *a: None)
    monkeypatch.setattr(b, "log", lambda *a, **k: None)
    monkeypatch.setattr(b.time, "sleep", lambda s: None)
    rows = [{"Id": 500 + q, "Title": f"How to {t} (case {q})?", "Body": "<p>?</p>", "Tags": "<x>",
             "AcceptedAnswerId": 1001 + 4 * q, "AcceptedAnswerBody": "x", "n_tokens": 5, "ViewCount": 1, "Score": 1}
            for q, t in enumerate(["git rebase conflict", "docker volume mount", "sql join null", "react state hook"] * 3)]
    sample = pd.DataFrame(rows)

    def fake_llm(client, messages, model):
        ids = [line[4:line.index("]")] for line in messages[-1]["content"].splitlines() if line.startswith("[SO-")]
        return {"content": "Answer based on " + " and ".join(f"[SO-{i}]" for i in ids[:2])}

    out = tmp_path / "condition_b_run.jsonl"
    recs = b.process_sample(sample, None, fake_llm, minilm, kg, None, 5, "fake", out, require_citation=True, run_id=RUN_ID)
    return out, recs


def test_lookup_end_to_end_on_12_questions(tiny_corpus, minilm, monkeypatch, tmp_path):
    out, recs = _run(tiny_corpus, minilm, monkeypatch, tmp_path)
    assert len(recs) == 12
    for rec in recs:
        assert ip.validate_record(rec) == []
        res = lookup_provenance(RUN_ID, rec["question_id"], search_dirs=[tmp_path])
        assert res["status"] == "ok" and res["condition"] == "B" and "no knowledge-graph path" in res["message"]
        assert len(res["cited"]) == len(rec["valid_cited_source_ids"]) >= 1
        retrieved_threads = {i["question_id"] for i in rec["retrieved_context"]}
        for entry in res["cited"]:
            assert entry["source_id"] in retrieved_threads and entry["items"]
            assert all(v["path_text"] == f"Answer {v['answer_id']}" for v in entry["items"])
        assert len(res["cited"][0]["items"]) + len(res["not_cited"]) <= 5 + len(res["cited"]) * 5
        assert "no knowledge-graph path" in render_lookup(res)


def test_the_same_run_without_provenance_gives_a_message(tiny_corpus, minilm, monkeypatch, tmp_path):
    out, recs = _run(tiny_corpus, minilm, monkeypatch, tmp_path)
    stripped = []
    for rec in recs:
        rec = {k: v for k, v in rec.items() if k != "provenance_version"}
        rec["retrieved_context"] = [{k: v for k, v in i.items() if k in
                                     ("answer_id", "question_id", "question_title", "chunk_text", "score", "is_accepted")}
                                    for i in rec["retrieved_context"]]
        stripped.append(rec)
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in stripped))
    res = lookup_provenance(RUN_ID, recs[0]["question_id"], results_file=out)
    assert res["status"] == "no_provenance" and res["cited"]
