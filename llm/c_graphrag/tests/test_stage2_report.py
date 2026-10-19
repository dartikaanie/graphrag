"""stage2_report.py on synthetic fixtures (no API, no Neo4j)."""

import json
from pathlib import Path

import pytest

import stage2_report as s2

QIDS = list(range(1000, 1050))
# Stage-2 logic is tested with fixed candidates, independent of the STAGE2_ALPHAS decision
# (test_stage2_alphas_are_the_amendment9_top_two pins that separately).
ALPHAS = (0.5, 0.25)
AFTER = "2026-10-14T01:00:00+00:00"


@pytest.fixture(autouse=True)
def _graph_update_applied(monkeypatch):
    """TRUST_V3_APPLIED_AT is None until Amendment 6 is applied (the report then fails closed,
    see test_fails_closed_without_applied_at); the other tests need a filled-in value."""
    monkeypatch.setattr(s2, "TRUST_V3_APPLIED_AT", "2026-10-13T13:04:20+00:00")
    monkeypatch.setattr(s2, "stage1_dev_ids", lambda: QIDS)  # synthetic ids stand in for the frozen dev list


def _cfg(alpha, grounded, **over):
    return {"condition": "C", "c_retrieval_version": "v3", "sample_split": "dev", "alpha": alpha,
            "require_grounding": grounded, "expansion_require_answer": True, "expansion_overfetch": 20,
            "leakage": "dup+neardup0.95", **over}


def _run(tmp: Path, alpha, grounded, started=AFTER, qids=QIDS, **over) -> Path:
    path = tmp / f"run_a{alpha}_{'g' if grounded else 'p'}.jsonl"
    path.write_text("".join(json.dumps({"question_id": q, "config": _cfg(alpha, grounded, **over)}) + "\n"
                            for q in qids))
    path.with_name(path.name + ".manifest.json").write_text(json.dumps(
        {"started_at_utc": started, "graph": {"trustScoreVersion": "v3-logscore"}}))
    return path


def _judge(source: Path, qid: int, kind: str, version="judge-v2.1", judge_id="primary") -> dict:
    claims = {"hal": [{"verdict": "CONTRADICTED", "severity": "CORE"}],
              "ok": [{"verdict": "SUPPORTED"}, {"verdict": "UNVERIFIABLE"}], "abs": []}[kind]
    return {"judge_id": judge_id, "judge_version": version, "source_file": f"/Users/x/{source.name}",
            "question_id": qid, "answer_attempted": kind != "abs", "claims": claims,
            "call_failed": False, "parse_error": False}


def _setup(tmp, n_hal):
    """n_hal[(alpha, grounding)] hallucinated answers; 2 ABSTAIN per run; rest OK."""
    runs = [_run(tmp, a, g) for a in (0.5, 0.25) for g in (True, False)]
    recs = []
    for path in runs:
        cfg = json.loads(path.read_text().splitlines()[0])["config"]
        key = (cfg["alpha"], "on" if cfg["require_grounding"] else "off")
        for i, q in enumerate(QIDS):
            kind = "abs" if i < 2 else ("hal" if i < 2 + n_hal[key] else "ok")
            recs.append(_judge(path, q, kind))
            recs.append(_judge(path, q, "hal", judge_id="secondary"))  # must be ignored
    jf = tmp / "jv2_1.jsonl"
    jf.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return runs, jf


def test_pooled_counts_and_winner(tmp_path):
    runs, jf = _setup(tmp_path, {(0.5, "on"): 5, (0.5, "off"): 9, (0.25, "on"): 10, (0.25, "off"): 12})
    r = s2.load_runs(runs, alphas=ALPHAS)
    pooled, per_run = s2.stage2_metrics(s2.load_judgments([jf], r, "judge-v2.1"), alphas=ALPHAS)
    assert pooled[0.5]["hallucination_count_excl_abstain"] == 14 and pooled[0.5]["n_excl_abstain"] == 96
    assert pooled[0.5]["hallucination_count_incl_abstain"] == 18 and pooled[0.5]["n_incl_abstain"] == 100
    assert per_run["0.25|off"]["n_hallucinated"] == 12
    sel = s2.select_stage2_winner(pooled)
    assert sel["winner"] == 0.5 and not sel["keep_current"]
    assert s2.draft_official_params(sel, alphas=ALPHAS) == {
        "status": "selected", "c_retrieval_version": "v3", "alpha_primary": 0.5, "alpha_ablation": 0.25,
        "selection_record": {"path": s2.SELECTION_RECORD_PATH, "commit": None}, **s2.OFFICIAL_DEFAULTS}


def test_tie_goes_to_larger_alpha(tmp_path):
    runs, jf = _setup(tmp_path, {(0.5, "on"): 6, (0.5, "off"): 6, (0.25, "on"): 5, (0.25, "off"): 5})
    pooled, _ = s2.stage2_metrics(s2.load_judgments([jf], s2.load_runs(runs, alphas=ALPHAS), "judge-v2.1"), alphas=ALPHAS)
    assert s2.select_stage2_winner(pooled)["winner"] == 0.5  # 12 vs 10: within the 2-answer margin


@pytest.mark.parametrize("over, msg", [({"sample_split": "test"}, "sample_split"),
                                       ({"c_retrieval_version": "v2"}, "c_retrieval_version")])
def test_rejects_wrong_config(tmp_path, over, msg):
    runs = [_run(tmp_path, 0.5, True, **over)]
    with pytest.raises(s2.Stage2Error, match=msg):
        s2.load_runs(runs, alphas=ALPHAS)


def test_rejects_run_before_graph_update(tmp_path):
    runs = [_run(tmp_path, a, g, started="2026-10-12T00:00:00+00:00" if (a, g) == (0.5, True) else AFTER)
            for a in (0.5, 0.25) for g in (True, False)]
    with pytest.raises(s2.Stage2Error, match="Amendment 6"):
        s2.load_runs(runs, alphas=ALPHAS)


def test_fails_closed_without_applied_at(tmp_path, monkeypatch):
    monkeypatch.setattr(s2, "TRUST_V3_APPLIED_AT", None)  # overrides the autouse fixture
    runs = [_run(tmp_path, a, g) for a in (0.5, 0.25) for g in (True, False)]
    with pytest.raises(s2.Stage2Error, match="TRUST_V3_APPLIED_AT is not set"):
        s2.load_runs(runs, alphas=ALPHAS)


def test_rejects_missing_run_and_wrong_alpha(tmp_path):
    with pytest.raises(s2.Stage2Error, match="missing runs"):
        s2.load_runs([_run(tmp_path, 0.5, True)], alphas=ALPHAS)
    with pytest.raises(s2.Stage2Error, match="not a stage-2 candidate"):
        s2.load_runs([_run(tmp_path, 0.75, True)], alphas=ALPHAS)


def test_rejects_wrong_judge_version_and_unusable(tmp_path):
    runs, jf = _setup(tmp_path, {(a, g): 3 for a in (0.5, 0.25) for g in ("on", "off")})
    r = s2.load_runs(runs, alphas=ALPHAS)
    with pytest.raises(s2.Stage2Error, match="frozen"):
        s2.load_judgments([jf], r, "judge-v2.2")
    lines = jf.read_text().splitlines()
    first = json.loads(lines[0])
    first["parse_error"] = True
    jf.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
    with pytest.raises(s2.Stage2Error, match="unusable"):
        s2.load_judgments([jf], r, "judge-v2.1")


def test_rejects_unjudged_answers(tmp_path):
    runs, jf = _setup(tmp_path, {(a, g): 3 for a in (0.5, 0.25) for g in ("on", "off")})
    jf.write_text("\n".join(jf.read_text().splitlines()[2:]) + "\n")  # drop q1000 of the first run
    with pytest.raises(s2.Stage2Error, match="not judged"):
        s2.load_judgments([jf], s2.load_runs(runs, alphas=ALPHAS), "judge-v2.1")


def test_keep_current_cannot_be_drafted_under_amendment_10():
    with pytest.raises(s2.Stage2Error, match="keep-current"):
        s2.draft_official_params({"winner": 1.0, "keep_current": True}, alphas=ALPHAS)


def test_draft_ablation_is_the_other_candidate():
    official = s2.draft_official_params({"winner": 0.0, "keep_current": False})
    assert (official["alpha_primary"], official["alpha_ablation"]) == (0.0, 0.25)
    with pytest.raises(s2.Stage2Error, match="one of the two"):
        s2.draft_official_params({"winner": 0.5, "keep_current": False})


def test_stage2_alphas_are_the_amendment9_top_two():
    """Amendment 9 stage-1 recheck: top_two = [0.25, 0.0]."""
    assert s2.STAGE2_ALPHAS == (0.25, 0.0)


@pytest.mark.parametrize("over, msg", [({"expansion_require_answer": None}, "expansion_require_answer"),
                                       ({"expansion_require_answer": False}, "expansion_require_answer"),
                                       ({"leakage": None}, "leakage mode"),
                                       ({"leakage": "thread"}, "leakage mode")])
def test_rejects_run_without_amendment8_9_settings(tmp_path, over, msg):
    runs = [_run(tmp_path, a, g, **(over if (a, g) == (0.5, True) else {})) for a in ALPHAS for g in (True, False)]
    with pytest.raises(s2.Stage2Error, match=msg):
        s2.load_runs(runs, alphas=ALPHAS)


def test_rejects_runs_not_on_the_stage1_dev_questions(tmp_path):
    """Same 50 questions in all 4 runs, but not the stage-1 dev list -> rejected."""
    other = list(range(2000, 2050))
    runs = [_run(tmp_path, a, g, qids=other) for a in ALPHAS for g in (True, False)]
    with pytest.raises(s2.Stage2Error, match="stage-1 dev list"):
        s2.load_runs(runs, alphas=ALPHAS)


def test_frozen_dev_list_is_the_stage1_dev_set(monkeypatch):
    """The frozen list stage 2 is checked against is exactly the question list the Amendment 9 stage-1
    recheck ran on (positions 385-434), in the same order."""
    monkeypatch.undo()  # the real stage1_dev_ids, not the autouse stand-in
    from llm import archive_paths
    recheck = archive_paths.resolve(
        Path(s2.__file__).resolve().parent / "results" / "expansion_check" / "stage1_expansion_new_alpha0-25.jsonl")
    ids = [json.loads(line)["question_id"] for line in recheck.read_text().splitlines()]
    assert s2.stage1_dev_ids() == ids and len(ids) == 50
