"""analyze_error_attribution.py reads the new provenance fields (hop, anchor_id)
with .get: records written before provenance v2 must still load and give the
same retrieval_status / attribution as before. Offline, tmp files only."""

import csv
import json

import analyze_error_attribution as attr
from llm.common import item_provenance as ip

LEGACY_ITEMS = [  # a pre-provenance C item: hop/rel_type/source_stage exist, path/anchor_id do not
    {"question_id": 11, "answer_id": 101, "chunk_text": "a", "hop": 1, "rel_type": "HAS_ANSWER",
     "source_stage": "graph_traversal", "trust_weight": 0.5, "combined_score": 0.4, "is_accepted": True},
    {"question_id": 12, "answer_id": 102, "chunk_text": "b", "hop": 2, "rel_type": "TAG_COOCCUR->HAS_ANSWER",
     "source_stage": "graph_traversal", "trust_weight": 0.3, "combined_score": 0.3, "is_accepted": False},
    {"question_id": 13, "answer_id": 103, "chunk_text": "c", "hop": "semantic", "rel_type": "HAS_ANSWER",
     "source_stage": "semantic_expansion", "trust_weight": 0.1, "combined_score": 0.2, "is_accepted": False},
]


def _v2_items():
    paths = [ip.one_hop_path(900, 101, "HAS_ANSWER"),
             ip.two_hop_path(901, 12, 102, "TAG_COOCCUR", "HAS_ANSWER"),
             ip.semantic_path(12, 13, 103, "HAS_ANSWER")]
    out = []
    for i, (legacy, path) in enumerate(zip(LEGACY_ITEMS, paths), start=1):
        out.append({**legacy, **ip.provenance_fields(source_stage=legacy["source_stage"], hop=legacy["hop"], path=path,
                                                     rank=i, via_question_id=legacy["question_id"])})
    return out


CTX = {"status": "ok", "n_items": 3, "sufficiency": "SEBAGIAN",
       "item_labels": [{"idx": 1, "answer_id": 101, "label": "RELEVAN"},
                       {"idx": 2, "answer_id": 102, "label": "TIDAK_RELEVAN"},
                       {"idx": 3, "answer_id": 103, "label": "RELEVAN"}]}


def test_summary_of_a_legacy_record_has_hops_but_no_anchors():
    s = attr.provenance_summary({"question_id": 1, "retrieved_context": LEGACY_ITEMS}, CTX)
    assert s["provenance_version"] is None and s["n_anchors"] is None and s["n_anchors_with_relevant"] is None
    assert (s["n_hop1"], s["n_hop2"], s["n_semantic"]) == (1, 1, 1)
    assert s["hop_label_counts"] == {"1": {"RELEVAN": 1}, "2": {"TIDAK_RELEVAN": 1}, "semantic": {"RELEVAN": 1}}


def test_summary_of_a_v2_record_counts_anchors_with_relevant_items():
    s = attr.provenance_summary({"question_id": 1, "provenance_version": "v2", "retrieved_context": _v2_items()}, CTX)
    assert s["provenance_version"] == "v2"
    assert s["n_anchors"] == 3                       # anchors 900, 901 and the semantic seed 12
    assert s["n_anchors_with_relevant"] == 2         # items 1 (anchor 900) and 3 (seed 12) are RELEVAN


def test_summary_tolerates_missing_record_and_missing_judge():
    assert attr.provenance_summary(None, None)["n_items"] == 0
    s = attr.provenance_summary({"retrieved_context": LEGACY_ITEMS}, None)
    assert s["hop_label_counts"] == {} and s["n_items"] == 3


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_legacy_and_v2_files_run_end_to_end_with_same_attribution(tmp_path):
    results = {}
    for name, items, version in (("legacy", LEGACY_ITEMS, None), ("v2", _v2_items(), "v2")):
        src = tmp_path / f"condition_c_{name}.jsonl"
        rec = {"question_id": 1, "retrieved_context": items, **({"provenance_version": version} if version else {})}
        _write(src, [rec])
        hall, ctx, ans = tmp_path / f"h_{name}.jsonl", tmp_path / f"c_{name}.jsonl", tmp_path / f"a_{name}.jsonl"
        _write(hall, [{"question_id": 1, "hallucination_label": "HALUSINASI_PENUH"}])
        _write(ctx, [{"question_id": 1, **CTX}])
        _write(ans, [{"question_id": 1, "answer_relevance_label": "MENJAWAB"}])
        res = attr.analyze_file(src, hall, ctx, ans, strict_hallucination=False)
        out_csv, out_md = tmp_path / f"{name}.csv", tmp_path / f"{name}.md"
        attr.write_csv([res], out_csv)
        attr.write_markdown_report([res], False, out_md)
        results[name] = (res, list(csv.DictReader(out_csv.open())), out_md.read_text())
    legacy, v2 = results["legacy"], results["v2"]
    assert legacy[0]["rows"][0]["attribution"] == v2[0]["rows"][0]["attribution"] == "PARTIAL_RETRIEVAL"
    assert legacy[1][0]["provenance_version"] == "" and legacy[1][0]["n_anchors"] == ""
    assert v2[1][0]["provenance_version"] == "v2" and v2[1][0]["n_anchors_with_relevant"] == "2"
    assert "Label relevansi item per hop" in legacy[2] and "Anchor yang menyumbang" not in legacy[2]
    assert "Anchor yang menyumbang" in v2[2]
