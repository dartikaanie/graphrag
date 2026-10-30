"""official_report.py: secondary_panel registry entries are skipped for hallucination matching, answer relevance is read
from the panel file, costs are separated, post-hoc exact McNemar. Offline, no registry or data files needed."""

import json

import official_report as orp


def _store(entries, tmp_path):
    s = object.__new__(orp.RegistryStore)
    s.entries = entries
    return s


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_secondary_entries_without_output_are_skipped_not_a_keyerror(tmp_path):
    src = tmp_path / "condition_a_x.jsonl"
    out = _jsonl(tmp_path / "panel_R1.jsonl", [
        {"panel_id": orp.PANEL_ID, "prompt_version": orp.PROMPT_VERSION, "source_file": str(src), "question_id": 1,
         "judge_id": "llama", "label": "FAKTUAL"}])
    entries = [{"kind": "generation", "files": {"output": str(src)}},
               {"kind": "judge", "run_id": "panel_official_R1", "cost_usd": 1.5, "files": {"output": str(out)}},
               {"kind": "judge", "run_id": "secondary_panel_ansrel@x", "cost_usd": 0.25,
                "files": {"records": str(tmp_path / "r.jsonl"), "panel": str(tmp_path / "p.jsonl")}}]
    s = _store(entries, tmp_path)
    recs = s.judge_records({"files": {"output": str(src)}})
    assert [r["question_id"] for r in recs] == [1]          # previously: KeyError 'output' on the secondary entry
    assert s.unresolved_failures({"files": {"output": str(src)}}) == 0


def test_secondary_cost_is_separate_from_hallucination_judging_cost(tmp_path):
    entries = [{"kind": "judge", "cost_usd": 1.5, "files": {"output": "a"}},
               {"kind": "judge", "cost_usd": 0.5, "files": {"output": "b"}},
               {"kind": "judge", "cost_usd": 0.25, "files": {"records": "r", "panel": "p"}},
               {"kind": "judge", "cost_usd": 0.125, "files": {"records": "r", "panel": "p"}},
               {"kind": "generation", "cost_usd": 99, "files": {"output": "g"}}]
    s = _store(entries, tmp_path)
    assert s.judge_cost() == 2.0 and s.secondary_cost() == 0.375


def test_answer_relevance_is_read_from_the_panel_file(tmp_path):
    a, b = tmp_path / "condition_a_x.jsonl", tmp_path / "condition_b_y.jsonl"
    rows = [{"key": [str(a), 1], "panel_label": "MENJAWAB", "panel_score": 1.0},
            {"key": [str(a), 2], "panel_label": "MENJAWAB_SEBAGIAN", "panel_score": 0.5},
            {"key": [str(a), 3], "panel_label": "UNRESOLVED", "panel_score": None},   # counted, not in the mean
            {"key": [str(b), 1], "panel_label": "TIDAK_MENJAWAB", "panel_score": 0.0}]
    f = _jsonl(tmp_path / "ansrel_panel.jsonl", rows)
    sec = orp.ansrel_panel_section(str(a), f)
    assert (sec["n_keys"], sec["n_judged"], sec["n_unresolved"], sec["mean_score"]) == (3, 2, 1, 0.75)
    assert orp.ansrel_panel_section(str(b), f)["mean_score"] == 0.0
    assert orp.ansrel_panel_section(str(a), tmp_path / "missing.jsonl") is None      # no panel file => no number
    assert orp.ansrel_panel_section(str(tmp_path / "condition_c_z.jsonl"), f) is None


def test_mcnemar_exact_hand_checked_values():
    assert orp.mcnemar_exact(3, 0)["p_exact"] == 0.25            # 2 * (1/8)
    assert orp.mcnemar_exact(5, 5)["p_exact"] == 1.0
    assert orp.mcnemar_exact(0, 0)["p_exact"] == 1.0
    m = orp.mcnemar_exact(10, 2)                                  # n = 12; (1 + 12 + 66) / 4096 * 2
    assert m["p_exact"] == round(2 * 79 / 4096, 6) and m["n_discordant"] == 12
    assert orp.mcnemar_exact(2, 10) == {**m, "b_only_first": 2, "c_only_other": 10}


def test_comparisons_carry_post_hoc_mcnemar_from_the_paired_cells():
    rows = lambda cats: {q: c for q, c in enumerate(cats)}                         # noqa: E731
    H, N, A = "HALLUCINATED", "NOT_HALLUCINATED", "ABSTAIN"
    runs = [{"label": "R5", "condition": "C", "results": _res(), "per_question_panel": rows([H, H, N, N, A])},
            {"label": "R1", "condition": "A", "results": _res(), "per_question_panel": rows([N, N, N, H, N])}]
    c = orp.comparisons(runs)[0]
    cells = c["paired"]["primary_excl_abstain"]["cells"]
    assert cells == {"only_c": 2, "only_other": 1, "both": 0, "neither": 1}
    mc = c["post_hoc_mcnemar_exact"]["primary_excl_abstain"]
    assert (mc["b_only_first"], mc["c_only_other"]) == (2, 1) and mc["p_exact"] == 1.0
    mf = c["post_hoc_mcnemar_exact"]["abstain_as_failed"]                           # ABSTAIN counts as failed
    assert (mf["b_only_first"], mf["c_only_other"]) == (3, 1)      # the 5th question: C ABSTAIN (failed) vs A fine


def _res():
    z = {"rate_excl_abstain": 0.1, "rate_abstain_as_failed": 0.1}
    return {"panel": z, "judges": {j: z for j in orp.JUDGES}}
