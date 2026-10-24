"""analyze_retrieval_quality.py: the provenance breakdown reads hop / anchor_id
with .get -- a legacy record (no anchor_id/path) still loads and gets the same
Precision@k / MRR@k as before. Offline (tiny parquet in tmp)."""

import csv
import json

import duckdb

import analyze_retrieval_quality as arq
from llm.common import item_provenance as ip


def _questions_parquet(tmp_path):
    path = tmp_path / "q.parquet"
    con = duckdb.connect()
    con.execute(f"""COPY (SELECT * FROM (VALUES (11, '<python><list>', 'x'), (12, '<java>', 'x'),
                    (13, '<python>', 'x')) t(Id, Tags, match_source)) TO '{path}' (FORMAT PARQUET)""")
    return str(path)


LEGACY = [
    {"question_id": 11, "answer_id": 101, "hop": 1, "rel_type": "HAS_ANSWER", "source_stage": "graph_traversal"},
    {"question_id": 12, "answer_id": 102, "hop": 2, "rel_type": "TAG_COOCCUR->HAS_ANSWER", "source_stage": "graph_traversal"},
    {"question_id": 13, "answer_id": 103, "hop": "semantic", "rel_type": "HAS_ANSWER", "source_stage": "semantic_expansion"},
]


def _v2():
    paths = [ip.one_hop_path(900, 101, "HAS_ANSWER", 0.5), ip.two_hop_path(901, 12, 102, "TAG_COOCCUR", 0.4, "HAS_ANSWER", 0.7),
             ip.semantic_path(12, 13, 103, 0.6, "HAS_ANSWER", 0.2)]
    return [{**legacy, **ip.provenance_fields(source_stage=legacy["source_stage"], hop=legacy["hop"], path=path, rank=i,
                                              via_question_id=legacy["question_id"])}
            for i, (legacy, path) in enumerate(zip(LEGACY, paths), start=1)]


TAGS = {11: {"python", "list"}, 12: {"java"}, 13: {"python"}}


def test_breakdown_of_a_legacy_record_has_hops_and_no_anchors():
    b = arq.provenance_breakdown({"retrieved_context": LEGACY}, {"python"}, TAGS, 0.0)
    assert b["by_hop"] == {"1": [1, 1], "2": [1, 0], "semantic": [1, 1]} and b["by_anchor"] == {}


def test_breakdown_of_a_v2_record_adds_anchors():
    b = arq.provenance_breakdown({"retrieved_context": _v2()}, {"python"}, TAGS, 0.0)
    assert b["by_anchor"] == {"900": [1, 1], "901": [1, 0], "12": [1, 1]}


def test_legacy_record_has_unchanged_precision_and_loads_end_to_end(tmp_path):
    qp = _questions_parquet(tmp_path)
    for name, items, version in (("legacy", LEGACY, None), ("v2", _v2(), "v2")):
        src = tmp_path / f"condition_c_{name}.jsonl"
        rec = {"question_id": 1, "tags": "<python>", "retrieved_context": items, **({"provenance_version": version} if version else {})}
        src.write_text(json.dumps(rec) + "\n")
        res = arq.analyze_file(duckdb.connect(), src, qp, 0.0, 5)
        assert res["precision_mean"] == round(2 / 3, 4) and res["mrr_mean"] == 1.0     # same for both
        assert res["precision_by_hop"]["2"] == {"n_items": 1, "n_relevant": 0, "precision": 0.0}
        row = next(csv.DictReader(open(res["csv_path"])))
        assert row["provenance_version"] == (version or "")
        assert (row["precision_by_anchor"] != "") == (version == "v2")
        assert res["n_with_provenance"] == int(version == "v2")
    arq.print_summary_table([res])   # does not raise
