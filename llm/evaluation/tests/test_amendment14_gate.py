"""Amendment 14 (draft) gates for panel-v2 judging: --official tied to the approval flag, registered inputs, the
DeepInfra budget. Offline; temporary registry; no API call."""

import json

import pytest

import judge_panel as jp
import judge_versions as jv
from llm import run_registry as rr


def _test_run(tmp_path, split="test"):
    p = tmp_path / "condition_c_t.jsonl"
    p.write_text(json.dumps({"question_id": 1, "config": {"sample_split": split}}) + "\n")
    return str(p)


def test_official_needs_flag_approval_and_panel_v2(tmp_path, monkeypatch):
    run = _test_run(tmp_path)
    monkeypatch.setattr(jv, "OFFICIAL_ALLOWED", ("judge-v2.3",))
    with pytest.raises(jv.JudgeVersionError, match="--official"):
        jv.check_run_allowed("judge-v2.3", [run], log=lambda *_: None, panel_id="panel-v2")
    with pytest.raises(jv.JudgeVersionError, match="not approved"):
        jv.check_run_allowed("judge-v2.3", [run], log=lambda *_: None, official=True, panel_id="panel-v2")
    monkeypatch.setattr(jv, "AMENDMENT14_APPROVED", True)
    with pytest.raises(jv.JudgeVersionError, match="panel-v2"):
        jv.check_run_allowed("judge-v2.3", [run], log=lambda *_: None, official=True, panel_id="panel-v1")
    assert jv.check_run_allowed("judge-v2.3", [run], log=lambda *_: None, official=True,
                                panel_id="panel-v2") == jv.PURPOSE_OFFICIAL


def test_official_flag_refused_for_dev_files(tmp_path):
    with pytest.raises(jv.JudgeVersionError, match="not test-sample"):
        jv.check_run_allowed("judge-v2.3", [_test_run(tmp_path, "dev")], log=lambda *_: None, official=True,
                             panel_id="panel-v2")


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "REGISTRY_PATH", tmp_path / "run_registry.json")
    return tmp_path


def _entry(path, status):
    return {"run_id": "r", "status": status, "kind": "generation", "condition": "C", "grounding": True, "alpha": 0.25,
            "n": 50, "seed": 42, "generator": "gpt-4o-mini", "judge_config": None, "commit": None, "created_at": None,
            "files": {"output": rr.archive_paths.repo_rel(path)}, "cost_usd": None, "cost_source": None, "notes": ""}


SPEC = jp.judge_version_spec("judge-v2.3")


def test_gate_requires_registration_and_budget(registry):
    run = rr.REPO_ROOT / "llm" / "c_graphrag" / "results" / "condition_c_stage2_x.jsonl"  # never created; lookup only
    with pytest.raises(jp.PanelGuardError, match="registered as 'dev-stage2'"):
        jp.panel_v2_registry_gate([str(run)], "development", SPEC, {"total_usd": 0.5}, log=lambda *_: None)
    rr.register(_entry(run, "dev-stage2"))
    with pytest.raises(jp.PanelGuardError, match="no DeepInfra budget"):
        jp.panel_v2_registry_gate([str(run)], "development", SPEC, {"total_usd": 0.5}, log=lambda *_: None)
    rr.set_budget(1.0)
    rr.record_deepinfra_spend("screening", 0.6)
    with pytest.raises(jp.PanelGuardError, match="exceeds the remaining"):
        jp.panel_v2_registry_gate([str(run)], "development", SPEC, {"total_usd": 0.5}, log=lambda *_: None)
    gate = jp.panel_v2_registry_gate([str(run)], "development", SPEC, {"total_usd": 0.3}, log=lambda *_: None)
    assert gate["status"] == "dev-stage2" and gate["record_extra"]["frozen_params_hash"]


def test_gate_refuses_another_judge_version(registry):
    with pytest.raises(jp.PanelGuardError, match="judge-v2.3 only"):
        jp.panel_v2_registry_gate([], "development", jp.judge_version_spec("judge-v2.2"), {"total_usd": 0},
                                  log=lambda *_: None)
