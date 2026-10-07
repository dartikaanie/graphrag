"""Step 4: stage-1 CLI runner -- tests for the pure/testable pieces
(plan/cost estimation, control-building, report assembly) per the
user's requirement: "Tests for the CLI runner (plan/confirm, control
failure stops the run, report contents)". Does NOT exercise any real
Neo4j/FAISS/LLM call -- those parts are covered by the Step 6 smoke
test already run manually against real data.
"""

import json

import pytest

from stage1_sweep import (
    build_control_items,
    build_plan,
    build_stage1_markdown,
    compute_alpha_metrics,
    format_plan,
    run_controls,
)


class _Row(dict):
    def __getattr__(self, item):
        return self[item]


def _sample_df(rows):
    import pandas as pd

    return pd.DataFrame(rows)


def test_build_control_items_positive_and_negative():
    rows = [
        {"Id": 1, "AcceptedAnswerId": 101, "Title": "python list comprehension",
         "AcceptedAnswerBody": "use [x for x in y]", "Tags": "<python><list>"},
        {"Id": 2, "AcceptedAnswerId": 202, "Title": "how to bake bread",
         "AcceptedAnswerBody": "knead the dough", "Tags": "<baking><bread>"},
    ]
    items = build_control_items(_sample_df(rows))
    assert len(items) == 2
    positive = next(i for i in items if i["control_type"] == "positive")
    negative = next(i for i in items if i["control_type"] == "negative")
    assert positive["expected_label"] == "RELEVANT"
    assert positive["answer_id"] == 101
    assert negative["expected_label"] == "IRRELEVANT"
    assert negative["answer_id"] == 202


def test_build_control_items_no_disjoint_tags_found_returns_only_positive():
    rows = [
        {"Id": 1, "AcceptedAnswerId": 101, "Title": "q1", "AcceptedAnswerBody": "a1", "Tags": "<python>"},
        {"Id": 2, "AcceptedAnswerId": 202, "Title": "q2", "AcceptedAnswerBody": "a2", "Tags": "<python><django>"},
    ]
    items = build_control_items(_sample_df(rows))
    assert len(items) == 1
    assert items[0]["control_type"] == "positive"


def test_build_control_items_empty_sample_returns_empty():
    assert build_control_items(_sample_df([])) == []


def test_run_controls_fails_fast_with_fewer_than_two_control_items(monkeypatch):
    rows = [{"Id": 1, "AcceptedAnswerId": 101, "Title": "q1", "AcceptedAnswerBody": "a1", "Tags": "<python>"}]
    result = run_controls(_sample_df(rows), judge_id="primary", questions_parquet="unused")
    assert result["passed"] is False
    assert result["results"] == []
    assert "error" in result


def test_run_controls_stops_run_when_judge_mismatches_expected(monkeypatch, tmp_path):
    rows = [
        {"Id": 1, "AcceptedAnswerId": 101, "Title": "python list comprehension",
         "AcceptedAnswerBody": "use [x for x in y]", "Tags": "<python><list>"},
        {"Id": 2, "AcceptedAnswerId": 202, "Title": "how to bake bread",
         "AcceptedAnswerBody": "knead the dough", "Tags": "<baking><bread>"},
    ]

    class _FakeConfig:
        model = "fake-model"

    def _fake_get_judge_client(judge_id):
        return object(), _FakeConfig()

    def _fake_call_judge_with_retry(client, config, messages):
        return {"raw": "irrelevant judge output"}, [{"error": None}]

    call_count = {"n": 0}

    def _fake_parse_judgment(raw):
        call_count["n"] += 1
        # Both calls return IRRELEVANT -- the positive control (expects
        # RELEVANT) will mismatch, so the overall control run must fail.
        return {"label": "IRRELEVANT", "parse_error": False}

    import stage1_sweep

    monkeypatch.setattr(stage1_sweep, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("judge_clients.get_judge_client", _fake_get_judge_client)
    import llm_judge_context_relevance_v1 as ctxrel

    monkeypatch.setattr(ctxrel, "call_judge_with_retry", _fake_call_judge_with_retry)
    monkeypatch.setattr(ctxrel, "parse_judgment", _fake_parse_judgment)

    result = run_controls(_sample_df(rows), judge_id="primary", questions_parquet="unused")
    assert result["passed"] is False
    assert any(not r["match"] for r in result["results"])


def test_build_plan_unknown_judge_id_marks_cost_unknown():
    plan = build_plan(unique_items=100, n_controls=2, judge_id="does-not-exist")
    assert plan["cost_unknown"] is True
    assert plan["est_cost_usd"] is None
    assert plan["items"] == 102


def test_build_plan_known_judge_uses_registry_defaults_when_no_history(tmp_path, monkeypatch):
    import stage1_sweep

    monkeypatch.setattr(stage1_sweep, "RESULTS_DIR", tmp_path / "results")  # empty -- falls back to registry defaults
    plan = build_plan(unique_items=100, n_controls=2, judge_id="primary", workers=4)
    assert plan["cost_unknown"] is False
    assert plan["token_source"] == "default estimate"
    assert plan["est_cost_usd"] > 0
    assert plan["est_time_sec"] > 0


def test_build_plan_uses_real_history_when_available(tmp_path, monkeypatch):
    import stage1_sweep

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr(stage1_sweep, "RESULTS_DIR", results_dir)
    from judge_clients import JUDGE_REGISTRY

    model = JUDGE_REGISTRY["primary"].model
    with open(results_dir / "ctxrel_v1_history_sample.jsonl", "w") as f:
        for _ in range(5):
            f.write(json.dumps({
                "judge_model": model, "prompt_tokens": 500, "completion_tokens": 50,
                "latency_s": 2.0, "call_failed": False,
            }) + "\n")

    plan = build_plan(unique_items=10, n_controls=2, judge_id="primary", workers=4)
    assert plan["token_source"] == "from history"
    assert plan["est_tokens_in"] == 500 * 12
    assert plan["est_tokens_out"] == 50 * 12


def test_format_plan_known_judge_includes_cost_and_time():
    plan = build_plan(unique_items=10, n_controls=2, judge_id="primary")
    text = format_plan(plan)
    assert "Estimated cost" in text
    assert "no API calls made yet" in text


def test_format_plan_unknown_judge_says_unknown():
    plan = build_plan(unique_items=10, n_controls=2, judge_id="nope")
    text = format_plan(plan)
    assert "unknown" in text


def test_compute_alpha_metrics_and_markdown_report(tmp_path):
    import llm_judge_context_relevance_v1 as ctxrel_mod

    retrieval_output = tmp_path / "retrieval_alpha_0.5.jsonl"
    record = {
        "question_id": 1,
        "retrieved_context": [
            {"answer_id": 101, "chunk_text": "relevant chunk", "sim": 0.8, "trust": 0.6,
             "is_accepted": True, "hop": 1, "rel_type": "HAS_ACCEPTED_ANSWER"},
            {"answer_id": 102, "chunk_text": "irrelevant chunk", "sim": 0.3, "trust": 0.1,
             "is_accepted": False, "hop": 2, "rel_type": "TAG_COOCCUR"},
        ],
    }
    with open(retrieval_output, "w") as f:
        f.write(json.dumps(record) + "\n")

    key_relevant = (1, 101, ctxrel_mod._text_hash("relevant chunk"))
    key_irrelevant = (1, 102, ctxrel_mod._text_hash("irrelevant chunk"))
    ctxrel_by_key = {
        key_relevant: {"label": "RELEVANT"},
        key_irrelevant: {"label": "IRRELEVANT"},
    }
    selected_by_alpha = {0.5: {1: {101, 102}}, 0.0: {1: {102, 103}}, 1.0: {1: {101}}}

    metrics = compute_alpha_metrics(str(retrieval_output), ctxrel_by_key, selected_by_alpha, this_alpha=0.5)
    assert metrics["n_questions"] == 1
    assert metrics["n_questions_with_relevant"] == 1
    assert metrics["pct_relevant"] == 50.0
    assert metrics["pct_irrelevant"] == 50.0
    assert metrics["pct_accepted_selected"] == 50.0
    assert metrics["jaccard_vs_alpha1"] == pytest.approx(0.5)  # {101,102} vs {101} -> 1/2
    assert metrics["jaccard_vs_alpha0"] == pytest.approx(1 / 3, abs=1e-4)  # {101,102} vs {102,103} -> 1/3
    assert metrics["by_hop"][1]["n_relevant"] == 1
    assert metrics["by_edge_type"]["HAS_ACCEPTED_ANSWER"]["n_relevant"] == 1

    from ctxrel_sweep import select_stage1_candidates

    selection = select_stage1_candidates({
        0.5: {"pct_questions_with_relevant": metrics["pct_questions_with_relevant"],
              "mean_relevance_score": metrics["mean_relevance_score"]},
        1.0: {"pct_questions_with_relevant": 10.0, "mean_relevance_score": 0.1},
    })
    markdown = build_stage1_markdown({0.5: metrics}, selection)
    assert "Stage 1 Report" in markdown
    assert "| 0.5 |" in markdown
    assert "HAS_ACCEPTED_ANSWER" in markdown
    assert "Selection helper output" in markdown
    assert "keep_current" in markdown
