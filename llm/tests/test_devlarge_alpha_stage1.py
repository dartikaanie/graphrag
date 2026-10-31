"""Dev-large alpha stage 1 (Amendment 19): selection rule and dry run. Offline; no judge is ever called."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import devlarge_alpha_stage1 as st  # noqa: E402


def S(primary, secondary=0.5):
    return {"primary": primary, "secondary": secondary}


def test_alpha_zero_is_never_eligible():
    sel = st.select({0.0: S(0.99, 0.9), 0.1: S(0.5), 0.25: S(0.6), 0.5: S(0.4)})
    assert 0.0 not in sel["advancing"] and sel["advancing"] == [0.25, 0.1]


def test_clear_primary_winner_outside_the_margin():
    sel = st.select({0.0: S(0.5), 0.1: S(0.50), 0.25: S(0.60), 0.5: S(0.40)})
    assert sel["advancing"] == [0.25, 0.1] and sel["rule_branches"][0].startswith("primary")


def test_primary_within_2pp_is_decided_by_secondary():
    sel = st.select({0.1: S(0.60, 0.40), 0.25: S(0.59, 0.50), 0.5: S(0.30, 0.9)})  # 1 pp apart; secondary differs by 0.10
    assert sel["advancing"][0] == 0.25 and sel["rule_branches"][0].startswith("secondary")
    assert sel["advancing"][1] == 0.1


def test_exactly_at_the_margins_counts_as_tied_and_goes_to_the_smaller_alpha():
    sel = st.select({0.1: S(0.58, 0.50), 0.25: S(0.60, 0.51), 0.5: S(0.20)})  # 2 pp and 0.01 apart
    assert sel["advancing"][0] == 0.1 and sel["rule_branches"][0].startswith("tie")


def test_beyond_the_primary_margin_the_secondary_is_ignored():
    sel = st.select({0.1: S(0.55, 0.99), 0.25: S(0.60, 0.10), 0.5: S(0.0)})  # 5 pp apart
    assert sel["advancing"][0] == 0.25


def test_three_way_tie_goes_to_the_smaller_alphas_both_picks():
    sel = st.select({0.0: S(0.7), 0.1: S(0.60, 0.5), 0.25: S(0.60, 0.5), 0.5: S(0.60, 0.5)})
    assert sel["advancing"] == [0.1, 0.25]


def _write(path, alpha, n=150, split="devlarge", version="v5", qid0=1):
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"question_id": qid0 + i, "title": "t", "tags": "x", "config_hash": "h",
                                "config": {"sample_split": split, "c_retrieval_version": version, "alpha": alpha},
                                "retrieved_context": [{"answer_id": 10 + i % 3 + int(alpha * 10), "chunk_text": f"a{i % 3}"},
                                                      {"answer_id": 99, "chunk_text": "shared"}]}) + "\n")


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "_bodies", lambda qids: {q: "<p>body</p>" for q in qids})
    import judge_versions as jv
    monkeypatch.setattr(jv, "run_purpose", lambda fs, calibration_paths=None: jv.PURPOSE_DEVELOPMENT)
    out = []
    for a in st.ALPHAS:
        p = tmp_path / f"a{a}.jsonl"
        _write(p, a)
        out.append(str(p))
    return out


def test_dry_run_prints_triples_and_cost_and_calls_nothing(files, tmp_path, capsys, monkeypatch):
    import judge_panel as jp
    monkeypatch.setattr(jp, "make_clients", lambda *a, **k: pytest.fail("a client was created in the dry run"))
    monkeypatch.setattr(st.sp, "run_stage", lambda *a, **k: pytest.fail("a judge was called in the dry run"))
    assert st.main([*files, "--out-dir", str(tmp_path / "out")]) == 0
    text = capsys.readouterr().out
    # shared "shared" text: 150 questions; per-alpha answers overlap across alphas only when the (answer, text) match
    runs = st.read_runs(files)
    n = len(st.unique_triples(runs))
    assert f"{n} unique" in text and "projected $" in text and "NO CALL WAS MADE" in text
    assert n < 4 * 150 * 2  # identical triples are shared across alphas
    assert not (tmp_path / "out" / "ctxrel_panel_devlarge_stage1.jsonl").exists()


def test_run_needs_a_cost_cap(files, tmp_path, capsys):
    assert st.main([*files, "--run", "--out-dir", str(tmp_path / "o")]) == 2
    assert "--max-cost-usd" in capsys.readouterr().out


def test_refuses_official_non_devlarge_and_non_v5_files(files, tmp_path, monkeypatch, capsys):
    bad = tmp_path / "t.jsonl"
    _write(bad, 0.5, split="test")
    assert st.main([*files[:3], str(bad)]) == 2 and "not a devlarge run" in capsys.readouterr().out
    _write(bad, 0.5, version="v4")
    assert st.main([*files[:3], str(bad)]) == 2 and "not a v5 run" in capsys.readouterr().out
    import judge_versions as jv
    monkeypatch.setattr(jv, "run_purpose", lambda fs, calibration_paths=None: jv.PURPOSE_OFFICIAL)
    assert st.main(files) == 2 and "official" in capsys.readouterr().out
    _write(bad, 0.5, n=149)
    assert st.main([*files[:3], str(bad)]) == 2


def test_report_applies_the_rule_and_enforces_the_completion_rule(files, tmp_path):
    runs = st.read_runs(files)
    out = tmp_path / "o" / "ctxrel_panel_devlarge_stage1.jsonl"
    out.parent.mkdir()
    keys = sorted({k for a in runs for ks in st.keys_by_question(runs[a]).values() for k in ks}, key=str)
    def rec(j, k, lab):
        return {"judge_id": j, "question_id": k[0], "answer_id": k[1], "text_hash": k[2], "label": lab, "parse_ok": True}
    ids = [j.judge_id for j in st.sp.JUDGES]
    with open(out, "w") as f:
        for k in keys:
            lab = "RELEVANT" if k[1] == 99 else "IRRELEVANT"  # the shared item is relevant for every alpha
            for j in ids:
                f.write(json.dumps(rec(j, k, lab)) + "\n")
    res = st.report(runs, out, out.parent)
    assert res["completion_rule_ok"] and all(v["primary"] == 1.0 for v in res["per_alpha"].values())
    assert res["selection"]["advancing"] == [0.1, 0.25] and (out.parent / "stage1_report.md").exists()
    # drop one judge's records -> more than 2% unusable for it -> no selection
    kept = [json.loads(x) for x in out.read_text().splitlines() if json.loads(x)["judge_id"] != ids[0]]
    out.write_text("".join(json.dumps(r) + "\n" for r in kept))
    assert st.report(runs, out, out.parent)["selection"] is None


def test_per_judge_detail_known_numbers(files):
    runs = st.read_runs(files)
    keys = sorted({k for a in runs for ks in st.keys_by_question(runs[a]).values() for k in ks}, key=str)
    ids = [j.judge_id for j in st.sp.JUDGES]
    recs = []
    for k in keys:
        for n, j in enumerate(ids):
            if j == ids[2] and k[1] != 99:
                continue  # judge 3: no usable record for the non-shared items
            # judge 1 and 2: shared item RELEVANT; other items PARTIAL (judge 2: IRRELEVANT on even answers)
            lab = "RELEVANT" if k[1] == 99 else ("IRRELEVANT" if n == 1 and k[1] % 2 == 0 else "PARTIAL")
            recs.append({"judge_id": j, "question_id": k[0], "answer_id": k[1], "text_hash": k[2], "label": lab,
                         "parse_ok": True})
    d = st.per_judge_detail(runs, recs, set(keys), {0.0: 0.9, 0.1: 0.8, 0.25: 0.7, 0.5: 0.6})
    j1, j2, j3 = (d["judges"][j] for j in ids)
    x = j1["per_alpha"]["0.5"]  # 150 questions x 2 items: 150 shared RELEVANT + 150 PARTIAL
    assert x["primary"] == 1.0 and x["labels"] == {"RELEVANT": 150, "PARTIAL": 150, "IRRELEVANT": 0}
    assert x["secondary"] == pytest.approx(0.75) and x["n_unusable_left_out"] == 0
    assert j3["per_alpha"]["0.5"]["n_unusable_left_out"] == 150 and j3["per_alpha"]["0.5"]["secondary"] == 1.0
    assert j1["secondary_direction"] == "not lower at highest alpha"  # constant across alphas
    assert j1["same_direction_as_panel"] is False and d["panel_secondary_direction"] == "decreasing"
    # judge 1 vs 2 differ exactly on the non-shared items whose answer id is even
    a12 = d["agreement"][f"{ids[0]}-{ids[1]}"]
    n_even = sum(1 for k in keys if k[1] != 99 and k[1] % 2 == 0)
    assert a12["n"] == len(keys) and a12["percent_agreement"] == pytest.approx(100 * (len(keys) - n_even) / len(keys), abs=0.01)
    assert d["agreement"][f"{ids[0]}-{ids[2]}"]["n"] == sum(1 for k in keys if k[1] == 99)
    assert d["unanimous_3way"]["n_with_3_usable"] == sum(1 for k in keys if k[1] == 99)
    assert d["unanimous_3way"]["percent"] == 100.0
    assert st.cohen_kappa([("A", "A"), ("A", "B"), ("B", "B"), ("B", "A")]) == 0.0
    assert st.cohen_kappa([("A", "A"), ("B", "B")]) == 1.0
