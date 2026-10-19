"""Pure-logic tests for reputation_stage1_recheck.py (no Neo4j/FAISS/API)."""

import json
from pathlib import Path

import reputation_stage1_recheck as rs
from llm import archive_paths

REPORT = archive_paths.resolve(Path(__file__).resolve().parents[1] / "results" / "stage1_report_ctxrel_v2_primary_n50_seed42.md")


def _m(pct, score):
    return {"pct_questions_with_relevant": pct, "mean_relevance_score": score}


def test_decision_include_when_within_tie_margins():
    base = {0.25: _m(48.0, 0.3284), 0.5: _m(48.0, 0.3114)}
    fix = {0.25: _m(46.0, 0.3100), 0.5: _m(50.0, 0.3200)}  # -2 pp and -0.0184: ties
    d = rs.apply_decision(base, fix)
    assert d["include_reputation"] and d["label"] == "SERTAKAN reputasi"


def test_decision_exclude_when_one_alpha_worse():
    base = {0.25: _m(48.0, 0.33), 0.5: _m(48.0, 0.31)}
    assert not rs.apply_decision(base, {0.25: _m(44.0, 0.33), 0.5: _m(48.0, 0.31)})["include_reputation"]
    assert not rs.apply_decision(base, {0.25: _m(48.0, 0.33), 0.5: _m(48.0, 0.28)})["include_reputation"]


def test_unjudged_items_dedups_and_skips_known():
    h = lambda t: "h:" + t  # noqa: E731
    rec = {"question_id": 1, "title": "T", "tags": "<a>",
           "retrieved_context": [{"answer_id": 10, "chunk_text": "x"}, {"answer_id": 11, "chunk_text": "y"}]}
    todo = rs.unjudged_items({0.25: [rec], 0.5: [rec]}, {(1, 10, "h:x")}, h)
    assert [(i["answer_id"], i["alpha_first_seen"]) for i in todo] == [(11, 0.25)]


def test_stage1_files_and_report_parse_match_committed_outputs():
    files = {a: rs.stage1_file_for_alpha(a) for a in rs.ALPHAS}
    assert len(set(files.values())) == 5
    assert all(len(rs.selected_ids(f)) == 50 for f in files.values())
    reported = rs.parse_stage1_report(REPORT)
    assert reported[0.5] == (48.0, 0.3114) and reported[1.0] == (34.0, 0.2288)


def test_stage1_file_lookup_requires_exactly_one(tmp_path):
    rec = {"question_id": 1, "retrieved_context": [], "config": {"alpha": 0.5}}
    for name in ("condition_c_x_n50_seed42_a_dev_v3_1.jsonl", "condition_c_x_n50_seed42_b_dev_v3_2.jsonl"):
        (tmp_path / name).write_text(json.dumps(rec) + "\n")
    try:
        rs.stage1_file_for_alpha(0.5, tmp_path)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for 2 matches")
