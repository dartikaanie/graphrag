"""llm/provenance_lookup.py on hand-built run files (no Neo4j, no LLM): the paths shown for
cited sources, the not-cited list, and clear messages for A, B, old records, missing run/question."""

import json

import pytest

from llm.common import item_provenance as ip
from llm.provenance_lookup import find_run_file, lookup_provenance, main, path_text, render_lookup

RUN = "11111111-2222-3333-4444-555555555555"


def _item(stage, hop, path, rank, rel, accepted=False):
    return {"question_id": path[-2]["node_id"], "answer_id": path[-1]["node_id"], "chunk_text": "t", "hop": hop,
            "source_stage": stage, "rel_type": rel, "is_accepted": accepted, "trust_weight": None,
            **ip.provenance_fields(source_stage=stage, hop=hop, path=path, rank=rank, via_question_id=path[-2]["node_id"])}


def _c_items():
    return [
        _item("graph_traversal", 2, ip.two_hop_path(111, 222, 333, "TAG_COOCCUR", "HAS_ACCEPTED_ANSWER"), 1,
              "TAG_COOCCUR->HAS_ACCEPTED_ANSWER", True),
        _item("graph_traversal", 1, ip.one_hop_path(444, 555, "HAS_ANSWER"), 2, "HAS_ANSWER"),
        _item("semantic_expansion", "semantic", ip.semantic_path(444, 666, 777, "HAS_ANSWER"), 3, "HAS_ANSWER"),
    ]


def _write(tmp_path, records, name="condition_c_x.jsonl"):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r) + "\n" for r in records))
    return p


def _rec(cond, qid, answer, items, **extra):
    return {"question_id": qid, "run_id": RUN, "condition": cond, "llm_answer": answer, "retrieved_context": items, **extra}


def test_path_text_template():
    assert path_text(ip.two_hop_path(111, 222, 333, "TAG_COOCCUR", "HAS_ACCEPTED_ANSWER")) == \
        "Question 111 -(TAG_COOCCUR)-> Question 222 -(HAS_ACCEPTED_ANSWER)-> Answer 333"


def test_c_shows_the_full_path_of_cited_sources_and_lists_the_rest(tmp_path):
    f = _write(tmp_path, [_rec("C", 1, "Use X [SO-222] and also [SO-666].", _c_items(), provenance_version="v3")])
    res = lookup_provenance(RUN, 1, search_dirs=[tmp_path])
    assert res["status"] == "ok" and res["condition"] == "C" and res["results_file"] == str(f)
    assert [c["source_id"] for c in res["cited"]] == [222, 666]
    assert res["cited"][0]["items"][0]["path_text"] == \
        "Question 111 -(TAG_COOCCUR)-> Question 222 -(HAS_ACCEPTED_ANSWER)-> Answer 333"
    assert [v["source_id"] for v in res["not_cited"]] == [444]
    text = render_lookup(res)
    assert "Cited [SO-222]:" in text and "Question 111 -(TAG_COOCCUR)-> Question 222 -(HAS_ACCEPTED_ANSWER)-> Answer 333" in text
    assert "Question 444 -(SEMANTIC_SIM)-> Question 666 -(HAS_ANSWER)-> Answer 777" in text and "not a knowledge-graph edge" in text
    assert "Retrieved but not cited: SO-444 (answer 555, rank 2)" in text
    assert text.splitlines()[-1] == "Note: the path shows how the source was retrieved from the knowledge graph, not why the model used it."


def test_d_shows_its_question_answer_path(tmp_path):
    items = [_item("low_level", 1, ip.one_hop_path(10, 100, "HAS_ANSWER"), 1, "HAS_ANSWER"),
             _item("high_level", 2, ip.two_hop_path(10, 20, 200, "EMBED_SIM", "HAS_ANSWER"), 2, "EMBED_SIM")]
    _write(tmp_path, [_rec("D", 2, "see [SO-20]", items, provenance_version="v3")], "condition_d_x.jsonl")
    res = lookup_provenance(RUN, 2, search_dirs=[tmp_path])
    assert res["status"] == "ok"
    assert res["cited"][0]["items"][0]["path_text"] == "Question 10 -(EMBED_SIM)-> Question 20 -(HAS_ANSWER)-> Answer 200"
    assert "[high_level, rank 2]" in render_lookup(res) and "Retrieved but not cited: SO-10" in render_lookup(res)


def test_b_has_no_kg_path(tmp_path):
    items = [{"question_id": 90, "answer_id": 900, "chunk_text": "t", "score": 0.8, "is_accepted": None, "hop": None,
              "source_stage": "vector_only", "rel_type": None, "trust_weight": None,
              **ip.provenance_fields(source_stage="vector_only", hop=None, path=ip.vector_only_path(900), rank=1,
                                     via_question_id=90)}]
    _write(tmp_path, [_rec("B", 3, "ok [SO-90]", items, provenance_version="v3")], "condition_b_x.jsonl")
    res = lookup_provenance(RUN, 3, search_dirs=[tmp_path])
    assert res["status"] == "ok" and "no knowledge-graph path" in res["message"]
    assert "Answer 900   [vector search, rank 1; no knowledge-graph path]" in render_lookup(res)


def test_a_has_no_retrieval(tmp_path):
    _write(tmp_path, [{"question_id": 4, "run_id": RUN, "condition": "A", "llm_answer": "plain"}], "condition_a_x.jsonl")
    res = lookup_provenance(RUN, 4, search_dirs=[tmp_path])
    assert res["status"] == "no_retrieval" and "no retrieval" in render_lookup(res)


def test_old_records_without_provenance_give_a_message_not_a_crash(tmp_path):
    legacy = [{"question_id": 222, "answer_id": 333, "chunk_text": "t", "hop": 2, "rel_type": "TAG_COOCCUR->HAS_ANSWER",
               "source_stage": "graph_traversal", "trust_weight": 0.4, "combined_score": 0.3, "is_accepted": True}]
    _write(tmp_path, [_rec("C", 5, "x [SO-222]", legacy)])
    res = lookup_provenance(RUN, 5, search_dirs=[tmp_path])
    assert res["status"] == "no_provenance" and "before per-item provenance" in res["message"]
    text = render_lookup(res)
    assert "Answer 333 of Question 222" in text and "no path recorded" in text and "-(" not in text
    # an old record with no condition / run_id at all, found by file
    p = _write(tmp_path, [{"question_id": 6, "llm_answer": "x [SO-1]", "retrieved_context": [{"question_id": 1, "answer_id": 2}]}],
               "condition_c_old.jsonl")
    assert lookup_provenance(None, 6, results_file=p)["status"] == "no_provenance"
    # a record with no retrieved_context at all
    p2 = _write(tmp_path, [{"question_id": 7, "condition": "C", "llm_answer": "x"}], "condition_c_odd.jsonl")
    assert lookup_provenance(None, 7, results_file=p2)["status"] == "no_provenance"


def test_unknown_provenance_version_is_reported(tmp_path):
    _write(tmp_path, [_rec("C", 8, "x [SO-222]", _c_items(), provenance_version="v9")])
    assert lookup_provenance(RUN, 8, search_dirs=[tmp_path])["status"] == "no_provenance"


def test_v2_records_still_show_paths(tmp_path):
    items = _c_items()
    for it in items:                                             # v2 steps also carried edge weights
        for s in it["path"]:
            s["edge_weight_to_next"] = 0.5
    _write(tmp_path, [_rec("C", 9, "x [SO-222]", items, provenance_version="v2")])
    res = lookup_provenance(RUN, 9, search_dirs=[tmp_path])
    assert res["status"] == "ok" and res["cited"][0]["items"][0]["path_text"].startswith("Question 111 -(TAG_COOCCUR)->")


def test_citation_that_is_not_in_the_context_and_no_citation_at_all(tmp_path):
    _write(tmp_path, [_rec("C", 10, "see [SO-999] and [SO-444]", _c_items(), provenance_version="v3"),
                      _rec("C", 11, "no citation here", _c_items(), provenance_version="v3")])
    r10 = lookup_provenance(RUN, 10, search_dirs=[tmp_path])
    assert [c["source_id"] for c in r10["cited"]] == [444] and r10["cited_not_retrieved"] == [999]
    assert "Cited but not in the retrieved context: SO-999" in render_lookup(r10)
    r11 = lookup_provenance(RUN, 11, search_dirs=[tmp_path])
    assert r11["cited"] == [] and "cites no source" in render_lookup(r11) and len(r11["not_cited"]) == 3


def test_retrieval_only_record(tmp_path):
    rec = _rec("C", 12, None, _c_items(), provenance_version="v3")
    del rec["llm_answer"]
    _write(tmp_path, [rec])
    res = lookup_provenance(RUN, 12, search_dirs=[tmp_path])
    assert res["status"] == "retrieval_only" and len(res["not_cited"]) == 3


def test_missing_run_question_and_bad_arguments(tmp_path, capsys):
    _write(tmp_path, [_rec("C", 1, "x", _c_items(), provenance_version="v3")])
    assert lookup_provenance("nope", 1, search_dirs=[tmp_path])["status"] == "run_not_found"
    assert lookup_provenance(RUN, 424242, search_dirs=[tmp_path])["status"] == "question_not_found"
    assert lookup_provenance(None, 1)["status"] == "error"
    assert lookup_provenance(RUN, 1, results_file=tmp_path / "missing.jsonl")["status"] == "run_not_found"
    (tmp_path / "bad.jsonl").write_text("{not json\n")
    assert find_run_file(RUN, [tmp_path]).name == "condition_c_x.jsonl"          # a corrupt file is skipped
    assert render_lookup(lookup_provenance("nope", 1, search_dirs=[tmp_path])).count("Note:") == 1


def test_last_record_wins_for_a_resumed_question_and_cli(tmp_path, capsys):
    f = _write(tmp_path, [_rec("C", 1, "first [SO-222]", _c_items(), provenance_version="v3"),
                          _rec("C", 1, "second [SO-444]", _c_items(), provenance_version="v3")])
    res = lookup_provenance(RUN, 1, results_file=f)
    assert res["answer"] == "second [SO-444]" and res["n_records_for_question"] == 2
    assert main(["--run-id", RUN, "--question-id", "1", "--file", str(f)]) == 0
    assert "Cited [SO-444]:" in capsys.readouterr().out
    assert main(["--run-id", RUN, "--question-id", "1", "--file", str(f), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    assert main(["--run-id", "nope", "--question-id", "1", "--file", str(tmp_path / "x.jsonl")]) == 1
