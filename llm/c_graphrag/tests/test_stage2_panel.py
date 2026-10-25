"""stage2_report.py with the panel-v2 label (Amendment 14 draft (g)) and --apply. Offline, synthetic records."""

import json
from pathlib import Path

import pytest

import stage2_report as s2

ALPHAS = (0.25, 0.0)


def _runs(tmp_path, qids=(1, 2, 3)):
    runs = {}
    for a in ALPHAS:
        for g in ("on", "off"):
            p = tmp_path / f"run_a{a}_{g}.jsonl"
            p.write_text("")
            runs[(a, g)] = {"path": p, "question_ids": set(qids), "started_at": "x"}
    return runs


def _rec(run_path, judge_id, qid, label, panel_id="panel-v2"):
    attempted = label != "ABSTAIN"
    claims = [{"verdict": "CONTRADICTED", "severity": "CORE"}] if label.startswith("HALUSINASI") else []
    return {"panel_id": panel_id, "judge_id": judge_id, "source_file": str(run_path), "question_id": qid,
            "prompt_version": "judge-v2.3", "label": label, "derived_label": label, "parse_ok": True,
            "call_failed": False, "parse_error": False, "answer_attempted": attempted, "claims": claims}


def _write(tmp_path, runs, labels):
    """labels: {judge_id: label applied to every item}, except overrides."""
    out = tmp_path / "panel_s2.jsonl"
    with open(out, "w") as f:
        for key, info in runs.items():
            for q in sorted(info["question_ids"]):
                for j, lab in labels.items():
                    f.write(json.dumps(_rec(info["path"], j, q, lab(key, q) if callable(lab) else lab)) + "\n")
    return out


def test_panel_label_is_the_2_of_3_majority_and_per_judge_kept(tmp_path):
    runs = _runs(tmp_path)
    halu_at_025 = lambda key, q: "HALUSINASI_PENUH" if key[0] == 0.25 else "FAKTUAL"  # noqa: E731
    jf = _write(tmp_path, runs, {"llama": "FAKTUAL", "deepseek": halu_at_025, "qwen3-235b": halu_at_025})
    panel, per_judge = s2.load_panel_judgments([jf], runs)
    pooled, _ = s2.stage2_metrics(panel, ALPHAS, s2._panel_category, s2._panel_severity)
    assert pooled[0.25]["hallucination_count_excl_abstain"] == 6 and pooled[0.0]["hallucination_count_excl_abstain"] == 0
    assert set(per_judge) == {"llama", "deepseek", "qwen3-235b"}
    assert s2.stage2_metrics(per_judge["llama"], ALPHAS)[0][0.25]["hallucination_count_excl_abstain"] == 0


def test_missing_judge_record_stops_stage2(tmp_path):
    runs = _runs(tmp_path)
    jf = _write(tmp_path, runs, {"llama": "FAKTUAL", "deepseek": "FAKTUAL"})  # no qwen3-235b records
    with pytest.raises(s2.Stage2Error, match="qwen3-235b has 3 missing"):
        s2.load_panel_judgments([jf], runs)


def test_other_panel_or_prompt_version_is_not_used(tmp_path):
    runs = _runs(tmp_path)
    jf = _write(tmp_path, runs, {"llama": "FAKTUAL", "deepseek": "FAKTUAL", "qwen3-235b": "FAKTUAL"})
    lines = [json.loads(x) for x in jf.read_text().splitlines()]
    lines[0]["prompt_version"] = "judge-v2.2"
    jf.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    with pytest.raises(s2.Stage2Error, match="prompt_version"):
        s2.load_panel_judgments([jf], runs)


def test_apply_writes_selected_params_once(tmp_path):
    path = tmp_path / "official_params.json"
    path.write_text(json.dumps({"status": "pending_selection", "alpha_primary": None, "alpha_ablation": None}))
    official = s2.draft_official_params({"winner": 0.25, "keep_current": False})
    s2.apply_official_params(official, "abc123", path)
    written = json.loads(path.read_text())
    assert written["status"] == "selected" and written["alpha_primary"] == 0.25 and written["alpha_ablation"] == 0.5
    assert written["selection_record"]["commit"] == "abc123"
    with pytest.raises(s2.Stage2Error, match="already applied"):
        s2.apply_official_params(official, "abc123", path)
