"""Amendment 14/16: the panel judging path (panel-v2, judge-v2.3) never asks for human annotation or a human-agreement
freeze, for dev (stage-2) and official files alike. Offline; nothing is called.

What it shows:
  * check_run_allowed(..., panel_id="panel-v2") never touches annotation_tool.frozen_judge_version (the human freeze
    of the single-judge path) -- poisoned here, so any call fails the test;
  * for DEV files it needs neither the Amendment 14 flag nor OFFICIAL_ALLOWED;
  * for OFFICIAL files the only extra requirements are --official, panel-v2 and the approval flag (a code constant,
    not an annotation);
  * judge_panel.main asks for the annotation freeze only under --calibration-set.
"""

import json
import re
from pathlib import Path

import pytest

import judge_versions as jv


def _run_file(tmp_path, split):
    p = tmp_path / f"condition_c_{split}.jsonl"
    p.write_text(json.dumps({"question_id": 1, "config": {"sample_split": split}}) + "\n")
    return str(p)


@pytest.fixture
def no_human(monkeypatch):
    import annotation_tool

    def poisoned(*a, **k):
        raise AssertionError("the panel path asked for the human annotation freeze")

    monkeypatch.setattr(annotation_tool, "frozen_judge_version", poisoned)
    monkeypatch.setattr(annotation_tool, "assert_annotation_frozen", poisoned)


def test_dev_stage2_panel_judging_needs_neither_the_flag_nor_human_annotation(tmp_path, no_human, monkeypatch):
    monkeypatch.setattr(jv, "AMENDMENT14_APPROVED", False)
    monkeypatch.setattr(jv, "OFFICIAL_ALLOWED", ())
    # judge-v2.3 is a logged calibration-allowed version, so a dev (stage-2) panel run is allowed before approval
    assert jv.check_run_allowed("judge-v2.3", [_run_file(tmp_path, "dev")], log=lambda *_: None,
                                panel_id="panel-v2") == jv.PURPOSE_DEVELOPMENT


def test_official_panel_judging_needs_only_the_flag_and_panel_v2(tmp_path, no_human, monkeypatch):
    run = _run_file(tmp_path, "test")
    monkeypatch.setattr(jv, "AMENDMENT14_APPROVED", True)
    monkeypatch.setattr(jv, "OFFICIAL_ALLOWED", ("judge-v2.3",))
    assert jv.check_run_allowed("judge-v2.3", [run], log=lambda *_: None, official=True,
                                panel_id="panel-v2") == jv.PURPOSE_OFFICIAL


def test_official_is_still_refused_before_approval(tmp_path, no_human, monkeypatch):
    monkeypatch.setattr(jv, "AMENDMENT14_APPROVED", False)
    monkeypatch.setattr(jv, "OFFICIAL_ALLOWED", ())
    with pytest.raises(jv.JudgeVersionError, match="not approved"):
        jv.check_run_allowed("judge-v2.3", [_run_file(tmp_path, "test")], log=lambda *_: None, official=True,
                             panel_id="panel-v2")


def test_judge_panel_asks_for_the_annotation_freeze_only_for_the_calibration_set():
    src = (Path(__file__).resolve().parents[1] / "judge_panel.py").read_text()
    calls = [m.start() for m in re.finditer(r"assert_annotation_frozen\(", src)]
    assert len(calls) == 1
    before = src[:calls[0]].rsplit("\n", 4)[-4:]
    assert any("if args.calibration_set" in line for line in before)
    assert "frozen_judge_version" not in src  # the human freeze of the single-judge path is not used by the panel
