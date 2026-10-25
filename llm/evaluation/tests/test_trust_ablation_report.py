"""trust_ablation_report.py (Amendment 10) on synthetic runs and judge records."""

import json
from pathlib import Path

import pytest

import trust_ablation_report as tar

QIDS = list(range(1, 11))
OFFICIAL = {"status": "selected", "c_retrieval_version": "v3", "alpha_primary": 0.25, "alpha_ablation": 0.0}


@pytest.fixture(autouse=True)
def _synthetic_sample(monkeypatch):
    monkeypatch.setattr(tar, "frozen_test_ids", lambda: QIDS)


def _judge(source: Path, qid: int, kind: str, judge_id="primary", version="judge-v2.1") -> dict:
    claims = {"hal": [{"verdict": "CONTRADICTED", "severity": "CORE"}], "ok": [{"verdict": "SUPPORTED"}],
              "abs": []}[kind]
    return {"judge_id": judge_id, "judge_version": version, "source_file": str(source), "question_id": qid,
            "answer_attempted": kind != "abs", "claims": claims, "call_failed": False, "parse_error": False}


def _cfg(alpha, grounded, **over):
    return {"condition": "C", "c_retrieval_version": "v3", "sample_split": "test", "alpha": alpha,
            "require_grounding": grounded, "expansion_require_answer": True, "expansion_overfetch": 20,
            "leakage": "thread", "max_hops": 2, **over}


def _run(tmp, alpha, grounded, role=None, qids=QIDS, **over) -> Path:
    path = tmp / f"c_a{alpha}_{'g' if grounded else 'p'}.jsonl"
    path.write_text("".join(json.dumps({"question_id": q, "config": _cfg(alpha, grounded, **over),
                                        "cosine_similarity": 0.5 + q / 100, "has_valid_citation": q % 2 == 0}) + "\n"
                            for q in qids))
    if role:
        path.with_name(path.name + ".manifest.json").write_text(json.dumps({"official_role": role}))
    return path


# question -> (C-primary kind, C-ablation kind), the same for both groundings
KINDS = {1: ("ok", "hal"), 2: ("ok", "hal"), 3: ("hal", "ok"), 4: ("hal", "hal"), 5: ("ok", "ok"),
         6: ("ok", "ok"), 7: ("abs", "hal"), 8: ("ok", "abs"), 9: ("abs", "abs"), 10: ("ok", "ok")}


def _setup(tmp):
    runs = [_run(tmp, a, g) for a in (0.25, 0.0) for g in (True, False)]
    recs = []
    for path in runs:
        idx = 0 if "a0.25" in path.name else 1
        for q in QIDS:
            recs.append(_judge(path, q, KINDS[q][idx]))
            recs.append(_judge(path, q, "hal", judge_id="secondary"))  # ignored, as in stage 2
    jf = tmp / "jv2_1.jsonl"
    jf.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return runs, jf


def test_paired_table_on_synthetic_data(tmp_path):
    runs, jf = _setup(tmp_path)
    loaded = tar.load_runs(runs, OFFICIAL)
    judged = tar.s2.load_judgments([jf], loaded, "judge-v2.1")
    rep = tar.build_report(loaded, judged)
    # ABSTAIN counted as non-factual: q1,q2,q8 ablation only; q3 primary only; q4,q7,q9 both; q5,q6,q10 neither
    assert rep["paired"]["non_factual"]["on"] == {"ablation_only": 3, "primary_only": 1, "both": 3, "neither": 3,
                                                 "n_pairs": 10, "n_excluded_abstain": 0}
    # ABSTAIN pairs (q7, q8, q9) left out
    assert rep["paired"]["exclude"]["off"] == {"ablation_only": 2, "primary_only": 1, "both": 1, "neither": 3,
                                              "n_pairs": 7, "n_excluded_abstain": 3}
    assert rep["paired"]["non_factual"]["pooled"]["n_pairs"] == 20
    assert rep["paired"]["exclude"]["pooled"]["ablation_only"] == 4
    p = rep["hallucination"]["c_primary|on"]
    assert (p["n_hallucinated"], p["n_abstain"], p["rate_excl_abstain"]) == (2, 2, 0.25)
    a = rep["hallucination"]["c_ablation|pooled"]
    assert (a["n"], a["n_hallucinated"], a["n_abstain"]) == (20, 8, 4)
    m = rep["cosine_nf2"]["c_ablation|off"]
    assert (m["n"], m["cosine_median"], m["nf2_pct_valid_citation"]) == (10, 0.555, 50.0)
    assert rep["alphas"] == {"c_primary": 0.25, "c_ablation": 0.0}
    md = tar.build_markdown(rep, "judge-v2.1")
    assert "| on | 3 | 1 | 3 | 3 | 10 | 0 |" in md


def test_rejects_configs_differing_beyond_alpha(tmp_path):
    runs = [_run(tmp_path, a, g, **({"max_hops": 1} if (a, g) == (0.0, True) else {}))
            for a in (0.25, 0.0) for g in (True, False)]
    with pytest.raises(tar.AblationError, match=r"differ in \['max_hops'\]"):
        tar.load_runs(runs, OFFICIAL)


@pytest.mark.parametrize("over, msg", [({"expansion_require_answer": False}, "expansion_require_answer"),
                                       ({"leakage": "dup+neardup0.95"}, "leakage"), ({"sample_split": "dev"}, "sample_split")])
def test_rejects_non_official_settings(tmp_path, over, msg):
    runs = [_run(tmp_path, a, g, **over) for a in (0.25, 0.0) for g in (True, False)]
    with pytest.raises(tar.AblationError, match=msg):
        tar.load_runs(runs, OFFICIAL)


def test_role_comes_from_alpha_and_manifest_must_agree(tmp_path):
    runs = [_run(tmp_path, a, g, role="c_primary") for a in (0.25, 0.0) for g in (True, False)]
    with pytest.raises(tar.AblationError, match="manifest role 'c_primary' != 'c_ablation'"):
        tar.load_runs(runs, OFFICIAL)


def test_rejects_pending_params_wrong_alpha_and_wrong_sample(tmp_path):
    runs = [_run(tmp_path, a, g) for a in (0.25, 0.0) for g in (True, False)]
    with pytest.raises(tar.AblationError, match="not 'selected'"):
        tar.load_runs(runs, {**OFFICIAL, "status": "pending_selection"})
    with pytest.raises(tar.AblationError, match="neither alpha_primary nor alpha_ablation"):
        tar.load_runs([_run(tmp_path, 0.5, True)], OFFICIAL)
    (tmp_path / "short").mkdir()
    short = [_run(tmp_path / "short", a, g, qids=QIDS[:-1]) for a in (0.25, 0.0) for g in (True, False)]
    with pytest.raises(tar.AblationError, match="frozen test sample"):
        tar.load_runs(short, OFFICIAL)
