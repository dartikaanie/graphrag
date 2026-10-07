"""Step 4: stage-1 CLI runner -- tests for the pure/testable pieces
(plan/cost estimation, control-building, report assembly) per the
user's requirement: "Tests for the CLI runner (plan/confirm, control
failure stops the run, report contents)". Does NOT exercise any real
Neo4j/FAISS/LLM call -- those parts are covered by the Step 6 smoke
test already run manually against real data.
"""

import json
from pathlib import Path

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


def _rows_two_disjoint():
    return [
        {"Id": 1, "AcceptedAnswerId": 101, "Title": "python list comprehension",
         "AcceptedAnswerBody": "use [x for x in y]", "Tags": "<python><list>"},
        {"Id": 2, "AcceptedAnswerId": 202, "Title": "how to bake bread",
         "AcceptedAnswerBody": "knead the dough", "Tags": "<baking><bread>"},
    ]


def _hard_pool():
    return _sample_df(_rows_two_disjoint() + [
        {"Id": 3, "AcceptedAnswerId": 303, "Title": "python dict merge",
         "AcceptedAnswerBody": "use {**a, **b}", "Tags": "<python><dictionary>"},
        {"Id": 4, "AcceptedAnswerId": 404, "Title": "sourdough starter",
         "AcceptedAnswerBody": "feed it daily", "Tags": "<baking>"},
    ])


def test_build_control_items_one_positive_and_one_easy_negative_per_question():
    items, info = build_control_items(_sample_df(_rows_two_disjoint()), control_sets=("positive", "easy_negative"))
    positives = [i for i in items if i["control_type"] == "positive"]
    negatives = [i for i in items if i["control_type"] == "easy_negative"]
    assert [(i["question_id"], i["answer_id"]) for i in positives] == [(1, 101), (2, 202)]
    assert [(i["question_id"], i["answer_id"]) for i in negatives] == [(1, 202), (2, 101)]
    assert all(i["expected_label"] == "RELEVANT" for i in positives)
    assert all(i["expected_label"] == "IRRELEVANT" for i in negatives)
    assert info["no_easy_negative"] == []
    # Formatted exactly like a retrieved context item; a negative shows the DONOR thread's title.
    assert positives[0]["chunk_text"] == "Q: python list comprehension\nA: use [x for x in y]"
    assert negatives[0]["chunk_text"] == "Q: how to bake bread\nA: knead the dough"
    assert negatives[0]["title"] == "python list comprehension"


def test_build_control_items_easy_negative_never_shares_a_tag_and_is_seeded():
    rows = [{"Id": i, "AcceptedAnswerId": 100 + i, "Title": f"q{i}", "AcceptedAnswerBody": f"a{i}",
             "Tags": f"<t{i % 3}><shared{i % 2}>"} for i in range(30)]
    a, _ = build_control_items(_sample_df(rows), seed=42, control_sets=("positive", "easy_negative"))
    b, _ = build_control_items(_sample_df(rows), seed=42, control_sets=("positive", "easy_negative"))
    assert a == b
    tags = {r["Id"]: r["Tags"] for r in rows}
    from stage1_sweep import _tagset
    for item in a:
        if item["control_type"] == "easy_negative":
            assert item["source_question_id"] != item["question_id"]
            assert not (_tagset(tags[item["source_question_id"]]) & _tagset(tags[item["question_id"]]))


def _unit(*xs):
    import numpy as np

    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def _vectors():
    # q1 is closest to q5, then q3; q2 is closest to q4.
    return {1: _unit(1, 0, 0), 2: _unit(0, 1, 0), 3: _unit(0.6, 0.8, 0), 4: _unit(0.1, 1, 0),
            5: _unit(0.9, 0.1, 0)}


def _hard_pool_with_python_twin():
    return _sample_df(_hard_pool().to_dict("records") + [
        {"Id": 5, "AcceptedAnswerId": 505, "Title": "python set union",
         "AcceptedAnswerBody": "use a | b", "Tags": "<python><set>"},
    ])


def test_hard_negative_is_most_similar_same_tag_donor_and_records_fields():
    items, info = build_control_items(_sample_df(_rows_two_disjoint()), hard_pool_df=_hard_pool_with_python_twin(),
                                      related_ids={}, question_vectors=_vectors(), control_sets=("hard_negative",))
    by_q = {i["question_id"]: i for i in items}
    assert by_q[1]["source_question_id"] == 5          # cos(q1,q5) > cos(q1,q3)
    assert by_q[1]["shared_tags"] == ["python"]
    assert by_q[1]["n_shared_tags"] == 1
    assert by_q[1]["n_eligible_donors"] == 2
    assert by_q[1]["donor_cosine_sim"] == pytest.approx(float(_vectors()[1] @ _vectors()[5]), abs=1e-6)
    assert by_q[1]["expected_label"] == "NOT_RELEVANT"
    assert by_q[1]["chunk_text"] == "Q: python set union\nA: use a | b"
    assert by_q[2]["source_question_id"] == 4
    assert info["no_hard_negative"] == []


def test_hard_negative_skips_is_related_to_neighbors_and_reports_no_donor():
    items, info = build_control_items(_sample_df(_rows_two_disjoint()), hard_pool_df=_hard_pool_with_python_twin(),
                                      related_ids={1: {5}, 2: {4}}, question_vectors=_vectors(),
                                      control_sets=("hard_negative",))
    assert [(i["question_id"], i["source_question_id"]) for i in items] == [(1, 3)]
    assert info["no_hard_negative"] == [2]


def test_hard_negative_ties_broken_by_question_id_ascending():
    vecs = _vectors()
    vecs[5] = vecs[3].copy()  # q3 and q5 now equally similar to q1
    items, _ = build_control_items(_sample_df(_rows_two_disjoint()), hard_pool_df=_hard_pool_with_python_twin(),
                                   related_ids={}, question_vectors=vecs, control_sets=("hard_negative",))
    assert next(i for i in items if i["question_id"] == 1)["source_question_id"] == 3


def test_hard_negative_missing_vectors_are_reported():
    vecs = _vectors()
    del vecs[5]   # an eligible donor without a vector is skipped
    del vecs[2]   # a dev question without a vector gets no hard negative
    items, info = build_control_items(_sample_df(_rows_two_disjoint()), hard_pool_df=_hard_pool_with_python_twin(),
                                      related_ids={}, question_vectors=vecs, control_sets=("hard_negative",))
    assert [(i["question_id"], i["source_question_id"]) for i in items] == [(1, 3)]
    assert info["hard_negative_missing_vector"] == [2]
    assert info["no_hard_negative"] == [2]
    assert info["n_eligible_donors_without_vector"] == 1


def test_hard_negative_is_deterministic_and_easy_negatives_unchanged():
    import numpy as np

    rows = [{"Id": i, "AcceptedAnswerId": 100 + i, "Title": f"q{i}", "AcceptedAnswerBody": f"a{i}",
             "Tags": f"<t{i % 3}><shared{i % 2}>"} for i in range(30)]
    rng = np.random.default_rng(0)
    vecs = {i: (lambda v: v / np.linalg.norm(v))(rng.normal(size=8).astype(np.float32)) for i in range(30)}
    easy_only, _ = build_control_items(_sample_df(rows), control_sets=("easy_negative",))
    a, _ = build_control_items(_sample_df(rows), hard_pool_df=_sample_df(rows), question_vectors=vecs, seed=1)
    b, _ = build_control_items(_sample_df(rows), hard_pool_df=_sample_df(rows), question_vectors=vecs, seed=99)
    hard_a = [i for i in a if i["control_type"] == "hard_negative"]
    assert hard_a == [i for i in b if i["control_type"] == "hard_negative"]  # seed plays no role
    assert easy_only == [i for i in build_control_items(_sample_df(rows), hard_pool_df=_sample_df(rows),
                                                        question_vectors=vecs)[0]
                         if i["control_type"] == "easy_negative"]
    from stage1_sweep import _tagset
    tags = {r["Id"]: r["Tags"] for r in rows}
    for item in hard_a:
        q = item["question_id"]
        eligible = [d for d in range(30) if d != q and _tagset(tags[d]) & _tagset(tags[q])]
        best = max(eligible, key=lambda d: (float(vecs[q] @ vecs[d]), -d))
        assert item["source_question_id"] == best


def test_hard_negatives_require_pool_and_vectors():
    with pytest.raises(ValueError):
        build_control_items(_sample_df(_rows_two_disjoint()), control_sets=("hard_negative",))
    with pytest.raises(ValueError):
        build_control_items(_sample_df(_rows_two_disjoint()), hard_pool_df=_hard_pool(),
                            control_sets=("hard_negative",))


def test_exclude_test_sample_drops_positions_0_to_383():
    import pandas as pd

    from stage1_sweep import exclude_test_sample

    df = pd.DataFrame({"Id": list(range(1000, 1500))})
    permuted = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    kept = exclude_test_sample(df, seed=42)
    assert set(kept["Id"]) == set(permuted.iloc[384:]["Id"])
    assert set(kept["Id"]).isdisjoint(set(permuted.iloc[:384]["Id"]))


def test_build_control_items_truncates_like_fuse_and_rank():
    import tiktoken

    long_body = "word " * 2000
    rows = [{"Id": 1, "AcceptedAnswerId": 101, "Title": "t", "AcceptedAnswerBody": long_body, "Tags": "<x>"}]
    item = build_control_items(_sample_df(rows), token_chunk_limit=400, control_sets=("positive",))[0][0]
    enc = tiktoken.get_encoding("cl100k_base")
    assert item["truncated"] is True
    assert item["n_tokens_full"] > 400
    assert len(enc.encode(item["chunk_text"])) <= 400
    assert item["chunk_text"] == enc.decode(enc.encode(f"Q: t\nA: {long_body}")[:400])


def test_build_control_items_no_disjoint_tags_found_reports_missing_easy_negatives():
    rows = [
        {"Id": 1, "AcceptedAnswerId": 101, "Title": "q1", "AcceptedAnswerBody": "a1", "Tags": "<python>"},
        {"Id": 2, "AcceptedAnswerId": 202, "Title": "q2", "AcceptedAnswerBody": "a2", "Tags": "<python><django>"},
    ]
    items, info = build_control_items(_sample_df(rows), control_sets=("positive", "easy_negative"))
    assert [i["control_type"] for i in items] == ["positive", "positive"]
    assert info["no_easy_negative"] == [1, 2]


def test_build_control_items_empty_sample_returns_empty():
    assert build_control_items(_sample_df([]), control_sets=("positive", "easy_negative"))[0] == []


def test_run_controls_fails_fast_when_a_requested_set_is_empty(monkeypatch):
    rows = [{"Id": 1, "AcceptedAnswerId": 101, "Title": "q1", "AcceptedAnswerBody": "a1", "Tags": "<python>"}]
    result = run_controls(_sample_df(rows), "ctxrel-v1", judge_ids=("primary",), questions_parquet="unused",
                          control_sets=("positive", "easy_negative"))
    assert result["passed"] is False
    assert result["results"] == []
    assert "error" in result


def test_run_controls_refuses_dirty_tree_without_calling_judge(monkeypatch, tmp_path):
    import stage1_sweep

    calls = _patch_judge(monkeypatch, tmp_path, lambda jid, m: "RELEVANT")
    monkeypatch.setattr(stage1_sweep, "git_state",
                        lambda exempt_paths=(): {"git_commit": "abc", "git_tree_clean": False,
                                                 "git_status_porcelain": [" M x.py"], "git_exempted": []})
    result = run_controls(_sample_df(_rows_two_disjoint()), "ctxrel-v2", questions_parquet="unused",
                          hard_pool_df=_hard_pool(), question_vectors=_vectors())
    assert result["passed"] is False and "not clean" in result["error"]
    assert calls == []


def test_git_state_ignores_control_outputs_only(monkeypatch):
    import subprocess

    import stage1_sweep

    def fake_run(cmd, **kw):
        out = "deadbeef\n" if "rev-parse" in cmd else porcelain
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    porcelain = "?? llm/c_graphrag/results/controls/ctxrel_v2_controls_x.jsonl\0"
    assert stage1_sweep.git_state()["git_tree_clean"] is True
    porcelain += " M llm/c_graphrag/stage1_sweep.py\0"
    state = stage1_sweep.git_state()
    assert state["git_tree_clean"] is False and state["git_commit"] == "deadbeef"
    assert len(state["git_status_porcelain"]) == 2


def test_run_controls_rejects_unknown_ctxrel_version():
    with pytest.raises(ValueError):
        run_controls(_sample_df(_rows_two_disjoint()), "ctxrel-v9", questions_parquet="unused")


def _patch_judge(monkeypatch, tmp_path, label_for):
    """label_for(judge_id, messages) -> label."""
    import llm_judge_context_relevance_v1 as ctxrel
    import stage1_sweep

    class _FakeConfig:
        def __init__(self, jid):
            self.model = f"fake-{jid}"

    seen_messages = []
    monkeypatch.setattr(stage1_sweep, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(stage1_sweep, "CONTROLS_DIR", tmp_path / "results" / "controls")
    monkeypatch.setattr("judge_clients.get_judge_client", lambda jid: (jid, _FakeConfig(jid)))
    monkeypatch.setattr(stage1_sweep, "_load_question_bodies_text",
                        lambda qp, qids: {q: f"body of {q}" for q in qids})
    monkeypatch.setattr(stage1_sweep, "git_state",
                        lambda exempt_paths=(): {"git_commit": "abc123", "git_tree_clean": True, "git_status_porcelain": [],
                                                 "git_exempted": []})

    def _fake_call(client, config, messages):
        seen_messages.append(messages)
        return {"raw": json.dumps({"label": label_for(client, messages), "reason": "r"})}, [{"error": None}]

    monkeypatch.setattr(ctxrel, "call_judge_with_retry", _fake_call)
    return seen_messages


def _title_match_label(jid, messages):
    user = messages[1]["content"]
    q_title = user.split("Question title: ")[1].split("\n")[0]
    item_title = user.split("Retrieved item:\nQ: ")[1].split("\n")[0]
    return "RELEVANT" if q_title == item_title else "IRRELEVANT"


def test_run_controls_fails_when_primary_mismatches(monkeypatch, tmp_path):
    _patch_judge(monkeypatch, tmp_path, lambda jid, m: "IRRELEVANT")
    result = run_controls(_sample_df(_rows_two_disjoint()), "ctxrel-v1", judge_ids=("primary", "secondary"),
                          questions_parquet="unused", hard_pool_df=_hard_pool(), question_vectors=_vectors())
    assert result["passed"] is False
    primary = result["summary"]["per_judge"]["primary"]
    assert primary["positive"]["pct"]["IRRELEVANT"] == 100.0
    assert primary["easy_negative"]["pass"] is True
    assert primary["hard_negative"]["pass"] is True


def test_run_controls_primary_decides_body_reaches_judge_and_version_recorded(monkeypatch, tmp_path):
    def label_for(jid, messages):
        return "PARTIAL" if jid == "secondary" else _title_match_label(jid, messages)

    seen = _patch_judge(monkeypatch, tmp_path, label_for)
    result = run_controls(_sample_df(_rows_two_disjoint()), "ctxrel-v2", judge_ids=("primary", "secondary"),
                          questions_parquet="unused", hard_pool_df=_hard_pool(), question_vectors=_vectors())
    assert result["passed"] is True
    assert result["summary"]["per_judge"]["secondary"]["pass"] is False
    assert len(result["results"]) == 12  # (2 pos + 2 easy + 2 hard) x 2 judges
    assert all("Question body: body of" in m[1]["content"] for m in seen)
    # v2 wording reached the judge
    assert all("Sharing a technology, library, or keyword" in m[1]["content"] for m in seen)
    out = Path(result["output_path"])
    assert out.parent == tmp_path / "results" / "controls"
    assert out.name.startswith("ctxrel_v2_controls_positive-easy_negative-hard_negative_")
    assert all(json.loads(l)["prompt_version"] == "ctxrel-v2" for l in out.read_text().splitlines())
    manifest = json.loads(Path(str(out) + ".manifest.json").read_text())
    assert manifest["prompt_version"] == "ctxrel-v2"
    assert manifest["config"]["ctxrel_version"] == "ctxrel-v2"
    summary_json = json.loads(Path(result["summary_path"]).read_text())
    assert summary_json["ctxrel_version"] == "ctxrel-v2"
    assert summary_json["git_commit"] == "abc123" and summary_json["git_tree_clean"] is True
    assert manifest["config"]["git_commit"] == "abc123"
    records = [json.loads(l) for l in out.read_text().splitlines()]
    assert all(r["git_commit"] == "abc123" and r["git_tree_clean"] is True for r in records)
    hard = [r for r in records if r["control_type"] == "hard_negative"]
    assert hard and all(r["donor_cosine_sim"] is not None and r["shared_tags"] for r in hard)


def test_run_controls_hard_negatives_only_reports_but_does_not_pass_or_fail(monkeypatch, tmp_path):
    _patch_judge(monkeypatch, tmp_path, lambda jid, m: "PARTIAL")
    result = run_controls(_sample_df(_rows_two_disjoint()), "ctxrel-v1", judge_ids=("primary", "secondary"),
                          questions_parquet="unused", hard_pool_df=_hard_pool(), question_vectors=_vectors(),
                          control_sets=("hard_negative",))
    assert result["passed"] is None
    assert result["summary"]["per_judge"]["primary"]["hard_negative"]["pass"] is True
    assert Path(result["output_path"]).name.startswith("ctxrel_v1_controls_hard_negative_2")


def _rs(ctype, labels, jid="primary"):
    return [{"judge_id": jid, "control_type": ctype, "label": lab} for lab in labels]


def test_summarize_controls_thresholds_match_amendment_2():
    from stage1_sweep import summarize_controls

    ok = (_rs("positive", ["RELEVANT"] * 9 + ["PARTIAL"]) + _rs("easy_negative", ["IRRELEVANT"] * 9 + [None])
          + _rs("hard_negative", ["PARTIAL"] * 4 + ["IRRELEVANT"] * 4 + ["RELEVANT"] * 2))
    out = summarize_controls(ok, ["primary"])
    assert out["passed"] is True
    assert out["per_judge"]["primary"]["hard_negative"]["pct_not_relevant"] == 80.0

    bad_pos = (_rs("positive", ["RELEVANT"] * 8 + ["PARTIAL"] * 2) + _rs("easy_negative", ["IRRELEVANT"] * 10)
               + _rs("hard_negative", ["PARTIAL"] * 10))
    out = summarize_controls(bad_pos, ["primary"])
    assert out["passed"] is False
    assert out["per_judge"]["primary"]["positive"]["pct"]["PARTIAL"] == 20.0

    bad_hard = (_rs("positive", ["RELEVANT"] * 10) + _rs("easy_negative", ["IRRELEVANT"] * 10)
                + _rs("hard_negative", ["PARTIAL"] * 7 + ["RELEVANT"] * 3))
    assert summarize_controls(bad_hard, ["primary"])["passed"] is False

    # Easy negatives need IRRELEVANT specifically (PARTIAL is not enough there).
    bad_easy = (_rs("positive", ["RELEVANT"] * 10) + _rs("easy_negative", ["PARTIAL"] * 2 + ["IRRELEVANT"] * 8)
                + _rs("hard_negative", ["PARTIAL"] * 10))
    assert summarize_controls(bad_easy, ["primary"])["passed"] is False


def test_summarize_controls_secondary_never_decides():
    from stage1_sweep import summarize_controls

    rs = (_rs("positive", ["RELEVANT"] * 10) + _rs("easy_negative", ["IRRELEVANT"] * 10)
          + _rs("hard_negative", ["PARTIAL"] * 10)
          + _rs("positive", ["PARTIAL"] * 10, "secondary") + _rs("easy_negative", ["PARTIAL"] * 10, "secondary")
          + _rs("hard_negative", ["RELEVANT"] * 10, "secondary"))
    out = summarize_controls(rs, ["primary", "secondary"], "primary")
    assert out["passed"] is True
    assert out["per_judge"]["secondary"]["pass"] is False


def test_cli_requires_ctxrel_version():
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "stage1_sweep.py"
    proc = subprocess.run([sys.executable, str(script), "--controls-only", "--questions-parquet", "x",
                           "--answers-parquet", "y"], capture_output=True, text=True)
    assert proc.returncode == 2
    assert "--ctxrel-version" in proc.stderr
    proc = subprocess.run([sys.executable, str(script), "--ctxrel-version", "ctxrel-v3", "--controls-only"],
                          capture_output=True, text=True)
    assert proc.returncode == 2


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
    markdown = build_stage1_markdown({0.5: metrics}, selection, ctxrel_version="ctxrel-v2")
    assert "Context-relevance judge version: ctxrel-v2" in markdown
    assert "Stage 1 Report" in markdown
    assert "| 0.5 |" in markdown
    assert "HAS_ACCEPTED_ANSWER" in markdown
    assert "Selection helper output" in markdown
    assert "keep_current" in markdown


# ---------------------------------------------------------------------
# Retrieval-only resume: per-question completeness
# ---------------------------------------------------------------------

def _rec(qid, config_hash="H", ctx=True):
    r = {"question_id": qid, "config_hash": config_hash}
    if ctx:
        r["retrieved_context"] = [{"answer_id": qid * 10, "chunk_text": "t"}]
    return json.dumps(r)


def test_scan_keeps_only_valid_unique_dev_records(tmp_path):
    from stage1_sweep import scan_retrieval_output

    p = tmp_path / "out.jsonl"
    p.write_text("\n".join([_rec(1), _rec(2, config_hash="OTHER"), _rec(99), _rec(3, ctx=False), _rec(1),
                            '{"question_id": 4, "config_ha']) + "\n")
    scan = scan_retrieval_output(p, [1, 2, 3, 4], "H")
    assert set(scan["valid"]) == {1}
    assert scan["duplicates"] == [1]
    assert len(scan["invalid_lines"]) == 4


def test_verify_complete_rejects_partial_and_accepts_complete(tmp_path):
    from stage1_sweep import IncompleteRetrievalOutput, verify_retrieval_output_complete

    p = tmp_path / "out.jsonl"
    p.write_text("")                                  # the 0-byte file a crash leaves behind
    with pytest.raises(IncompleteRetrievalOutput):
        verify_retrieval_output_complete(p, [1, 2], "H")
    p.write_text(_rec(1) + "\n")
    with pytest.raises(IncompleteRetrievalOutput):
        verify_retrieval_output_complete(p, [1, 2], "H")
    p.write_text(_rec(1) + "\n" + _rec(2) + "\n" + _rec(2) + "\n")
    with pytest.raises(IncompleteRetrievalOutput):     # duplicate
        verify_retrieval_output_complete(p, [1, 2], "H")
    p.write_text(_rec(2) + "\n" + _rec(1) + "\n")
    assert verify_retrieval_output_complete(p, [1, 2], "H") == 2


def test_prepare_resume_requires_same_commit_and_config(tmp_path):
    from stage1_sweep import _resume_sidecar, prepare_retrieval_resume

    p = tmp_path / "out.jsonl"
    p.write_text(_rec(1) + "\n" + '{"truncat')
    # no sidecar (e.g. the file left by the crashed runs) -> fresh start
    assert prepare_retrieval_resume(p, [1, 2], "H", "c1") == set()
    assert not p.exists() and _resume_sidecar(p).exists()

    p.write_text(_rec(1) + "\n" + _rec(7) + "\n" + '{"truncat')
    assert prepare_retrieval_resume(p, [1, 2], "H", "c1") == {1}
    assert p.read_text() == _rec(1) + "\n"            # truncated + foreign lines dropped

    assert prepare_retrieval_resume(p, [1, 2], "H", "c2") == set()   # different commit -> fresh
    assert not p.exists()
    p.write_text(_rec(1) + "\n")
    assert prepare_retrieval_resume(p, [1, 2], "H2", "c2") == set()  # different config -> fresh
    p.write_text(_rec(1) + "\n")
    assert prepare_retrieval_resume(p, [1, 2], "H", None) == set()   # unknown commit -> never resume


def _fake_retrieval_env(monkeypatch, tmp_path, crash_after=None):
    import c_graphrag as cg
    import stage1_sweep

    calls = []

    def fake_process_sample(sample_df, *a, already_done=None, config_hash=None, **k):
        output_path = a[15]  # process_sample(sample_df, ..., model, output_path, ...)
        todo = [int(q) for q in sample_df["Id"] if int(q) not in (already_done or set())]
        calls.append(todo)
        with open(output_path, "a") as f:
            for n, q in enumerate(todo):
                if crash_after is not None and n == crash_after:
                    f.write('{"question_id": %d, "trunc' % q)
                    raise RuntimeError("disk I/O error")
                f.write(_rec(q, config_hash=config_hash) + "\n")
        return [], {}

    monkeypatch.setattr(cg, "process_sample", fake_process_sample)
    monkeypatch.setattr(cg, "append_run_history", lambda rec: (calls.append(("history", rec)), tmp_path / "h.jsonl")[1])
    monkeypatch.setattr("llm.manifest.write_manifest", lambda *a, **k: calls.append(("manifest", k)))
    return calls


def _run_alpha(tmp_path, commit="c1"):
    import pandas as pd

    from stage1_sweep import run_retrieval_only_for_alpha

    df = pd.DataFrame({"Id": [11, 12, 13, 14]})
    return run_retrieval_only_for_alpha(df, 0.0, None, None, None, None, None, None, {}, None, None, "qp",
                                        output_dir=str(tmp_path), seed=42, n_sample=4, git_commit=commit)


def test_crashed_alpha_writes_no_history_then_resumes_to_complete(monkeypatch, tmp_path):
    calls = _fake_retrieval_env(monkeypatch, tmp_path, crash_after=2)
    with pytest.raises(RuntimeError):
        _run_alpha(tmp_path)
    assert not any(isinstance(c, tuple) for c in calls)   # no history/manifest for a partial file

    calls = _fake_retrieval_env(monkeypatch, tmp_path)
    info = _run_alpha(tmp_path)
    assert calls[0] == [13, 14]                           # only the missing questions re-run
    assert info["n_questions"] == 4 and info["n_resumed"] == 2
    history = [c[1] for c in calls if isinstance(c, tuple) and c[0] == "history"]
    assert history[0]["n_processed"] == 4 and history[0]["git_commit"] == "c1"


def test_incomplete_output_raises_before_history(monkeypatch, tmp_path):
    import c_graphrag as cg

    from stage1_sweep import IncompleteRetrievalOutput

    calls = _fake_retrieval_env(monkeypatch, tmp_path)
    monkeypatch.setattr(cg, "process_sample", lambda sample_df, *a, **k: ([], {}))  # silently writes nothing
    with pytest.raises(IncompleteRetrievalOutput):
        _run_alpha(tmp_path)
    assert not any(isinstance(c, tuple) for c in calls)


# ---------------------------------------------------------------------
# Clean-tree check: this run's own outputs are exempt, nothing else
# ---------------------------------------------------------------------

def _fake_git(monkeypatch, entries):
    import subprocess

    raw = "".join(f"{e}\0" for e in entries)

    def fake_run(cmd, **kw):
        out = "deadbeef\n" if "rev-parse" in cmd else raw
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_git_state_exempts_only_this_runs_outputs(monkeypatch):
    import stage1_sweep

    own = "llm/c_graphrag/results/condition_c_x_dev_v3_abc.jsonl"
    _fake_git(monkeypatch, [f"?? {own}", "?? logs/run_history.jsonl",
                            "?? llm/c_graphrag/results/controls/ctxrel_v2_controls_x.jsonl"])
    exempt = [stage1_sweep.REPO_ROOT / own, stage1_sweep.REPO_ROOT / "logs/run_history.jsonl"]
    assert stage1_sweep.git_state(exempt)["git_tree_clean"] is True
    assert stage1_sweep.git_state()["git_tree_clean"] is False          # without exemption: dirty

    _fake_git(monkeypatch, [f"?? {own}", "?? llm/c_graphrag/results/condition_c_OTHER_dev.jsonl",
                            " M llm/c_graphrag/stage1_sweep.py"])
    state = stage1_sweep.git_state(exempt)
    assert state["git_tree_clean"] is False
    assert state["git_exempted"] == [own]


def test_git_state_handles_spaces_and_renames(monkeypatch):
    import stage1_sweep

    _fake_git(monkeypatch, ["R  docs/new name.md", "docs/old name.md", "?? llm/c_graphrag/results/controls/a b.jsonl"])
    state = stage1_sweep.git_state()
    assert state["git_status_porcelain"] == ["R  docs/new name.md", "?? llm/c_graphrag/results/controls/a b.jsonl"]
    assert state["git_tree_clean"] is False
    _fake_git(monkeypatch, ["?? llm/c_graphrag/results/controls/a b.jsonl"])
    assert stage1_sweep.git_state()["git_tree_clean"] is True


def test_run_controls_passes_exemptions_to_clean_check(monkeypatch, tmp_path):
    import stage1_sweep

    _patch_judge(monkeypatch, tmp_path, _title_match_label)
    seen = {}

    def fake_git_state(exempt_paths=()):
        seen["exempt"] = list(exempt_paths)
        return {"git_commit": "abc", "git_tree_clean": True, "git_status_porcelain": [], "git_exempted": []}

    monkeypatch.setattr(stage1_sweep, "git_state", fake_git_state)
    run_controls(_sample_df(_rows_two_disjoint()), "ctxrel-v2", questions_parquet="unused",
                 hard_pool_df=_hard_pool(), question_vectors=_vectors(), exempt_paths=["x.jsonl"])
    assert seen["exempt"] == ["x.jsonl"]
