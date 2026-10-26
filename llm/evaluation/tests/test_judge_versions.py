"""Amendment 11 freeze guard (judge_versions.py + docs/JUDGE_REVISION_LOG.md)."""

import json
from pathlib import Path

import pytest

import judge_prompt_v2_2
import judge_versions as jv

HEADER = "| version | date | commit | reason | k before | k after |\n|---|---|---|---|---|---|\n"


def _log(tmp_path, versions):
    path = tmp_path / "LOG.md"
    path.write_text("# log\n\n" + HEADER + "".join(f"| {v} | 2026-10-19 | abc | r | - | - |\n" for v in versions))
    return path


def test_constants_match_the_runner_and_the_real_log():
    assert jv.BASE_VERSION == judge_prompt_v2_2.PROMPT_VERSION == jv.CALIBRATION_ALLOWED[0]
    assert len(jv.CALIBRATION_ALLOWED) == 1 + jv.MAX_REVISIONS == 4
    assert jv.logged_versions() == ["judge-v2.2", "judge-v2.3"]   # base + revision 1 (iteration 2)
    assert jv.OFFICIAL_ALLOWED in ((), ("judge-v2.3",))   # Amendment 14: no human freeze involved
    assert jv.AMENDMENT14_APPROVED == bool(jv.OFFICIAL_ALLOWED)   # flag and list change together


def test_calibration_needs_a_logged_allowed_version(tmp_path):
    log = _log(tmp_path, ["judge-v2.2"])
    jv.check_version("judge-v2.2", jv.PURPOSE_CALIBRATION, log)
    with pytest.raises(jv.JudgeVersionError, match="not logged"):
        jv.check_version("judge-v2.3", jv.PURPOSE_CALIBRATION, log)
    log = _log(tmp_path, ["judge-v2.2", "judge-v2.3", "judge-v2.4", "judge-v2.5"])
    for v in ("judge-v2.3", "judge-v2.4", "judge-v2.5"):
        jv.check_version(v, jv.PURPOSE_CALIBRATION, log)
    with pytest.raises(jv.JudgeVersionError, match="not in CALIBRATION_ALLOWED"):
        jv.check_version("judge-v2.6", jv.PURPOSE_CALIBRATION, log)
    with pytest.raises(jv.JudgeVersionError, match="not in CALIBRATION_ALLOWED"):
        jv.check_version("judge-v2.1", jv.PURPOSE_CALIBRATION, log)


def test_fourth_revision_after_base_is_refused(tmp_path):
    log = _log(tmp_path, ["judge-v2.2", "judge-v2.3", "judge-v2.4", "judge-v2.5", "judge-v2.6"])
    assert jv.revisions_after_base(jv.logged_versions(log)) == ["judge-v2.3", "judge-v2.4", "judge-v2.5", "judge-v2.6"]
    with pytest.raises(jv.JudgeVersionError, match="at most 3"):
        jv.check_version("judge-v2.5", jv.PURPOSE_CALIBRATION, log)


def test_official_needs_the_frozen_version(tmp_path):
    log = _log(tmp_path, ["judge-v2.2", "judge-v2.3"])
    with pytest.raises(jv.JudgeVersionError, match="not an official judge version"):
        jv.check_version("judge-v2.2", jv.PURPOSE_OFFICIAL, log)              # nothing frozen yet
    jv.check_version("judge-v2.3", jv.PURPOSE_OFFICIAL, log, ("judge-v2.3",), "judge-v2.3")
    with pytest.raises(jv.JudgeVersionError, match="differs from the version frozen"):
        jv.check_version("judge-v2.3", jv.PURPOSE_OFFICIAL, log, ("judge-v2.3",), "judge-v2.4")
    jv.check_version("judge-v2.3", jv.PURPOSE_DEVELOPMENT, log, ("judge-v2.3",))   # stage 2 / probes after freeze
    jv.check_version("judge-v2.2", jv.PURPOSE_DEVELOPMENT, log)                    # probes before freeze


def test_run_purpose_from_the_run_files(tmp_path):
    def run(name, split):
        p = tmp_path / name
        p.write_text(json.dumps({"question_id": 1, "config": {"sample_split": split}}) + "\n")
        return str(p)

    dev, test = run("dev.jsonl", "dev"), run("test.jsonl", "test")
    cal = run("cal.jsonl", "dev")
    calib = {str(Path(cal).resolve())}
    assert jv.run_purpose([dev], calib) == jv.PURPOSE_DEVELOPMENT
    assert jv.run_purpose([dev, test], calib) == jv.PURPOSE_OFFICIAL
    assert jv.run_purpose([cal, test], calib) == jv.PURPOSE_CALIBRATION


def test_menu_points_to_the_v2_2_runner():
    menu = (Path(jv.REPO_ROOT) / "run.py").read_text()
    assert '"script": "llm_judge_hallucination_v2_2.py"' in menu
    assert '"script": "llm_judge_hallucination_v2_1.py"' not in menu


def test_run_purpose_treats_an_absent_sample_split_as_test(tmp_path):
    """Test-sample runs do not store sample_split (only non-default values are written). Regression 2026-10-26: such
    files were classified as development, so the official judging refused them."""
    def write(name, rec):
        p = tmp_path / name
        p.write_text(json.dumps(rec) + "\n")
        return str(p)

    absent = write("absent.jsonl", {"question_id": 1, "config": {"condition": "A", "n_sample": 384}})
    test = write("test.jsonl", {"question_id": 1, "config": {"sample_split": "test"}})
    dev = write("dev.jsonl", {"question_id": 1, "config": {"sample_split": "dev"}})
    no_config = write("judge_out.jsonl", {"item_id": "x", "label": "FAKTUAL"})     # not a run file
    calib = set()
    assert jv.run_purpose([absent], calib) == jv.PURPOSE_OFFICIAL
    assert jv.run_purpose([test], calib) == jv.PURPOSE_OFFICIAL
    assert jv.run_purpose([dev], calib) == jv.PURPOSE_DEVELOPMENT
    assert jv.run_purpose([no_config], calib) == jv.PURPOSE_DEVELOPMENT
    assert jv.run_purpose([dev, absent], calib) == jv.PURPOSE_OFFICIAL
