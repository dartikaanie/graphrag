"""Amendment 12: freezing the validated annotation sheet, and the frozen-hash guard. Offline, temporary files."""

import json
from datetime import datetime, timezone

import pytest

import annotation_tool as at
import judge_panel_agreement as ag
from test_reannotation import T0, _args, _fill_all, env  # noqa: F401 -- fixture


def test_freeze_writes_record_copy_and_completion(env):
    rec = at.freeze_annotation("calibration", "annotator1", env.key_dir, now=T0)
    sheet = env.pdir / "annotator1.xlsx"
    assert rec["sha256"] == at.file_sha256(sheet) and rec["item_count"] == 60
    assert rec["note"] == "frozen before any judge run on calibration items" and rec["file_name"] == "annotator1.xlsx"
    copy = env.key_dir / "annotator1_frozen_20261101.xlsx"
    assert copy.exists() and at.file_sha256(copy) == rec["sha256"]
    assert json.loads((env.pdir / "annotator1_freeze.json").read_text()) == rec
    completion = json.loads((env.pdir / "annotator1_completed.json").read_text())
    assert completion["reannotation"] == "not_scheduled" and completion["sheet_sha256"] == rec["sha256"]
    assert at.assert_annotation_frozen("calibration", "annotator1")["sha256"] == rec["sha256"]
    # idempotent for the same sheet
    assert at.freeze_annotation("calibration", "annotator1", env.key_dir, now=T0) == rec


def test_invalid_sheet_is_not_frozen(env):
    from openpyxl import load_workbook
    path = env.pdir / "annotator1.xlsx"
    wb = load_workbook(path)
    wb["anotasi"].cell(row=2, column=at.COLUMNS.index("uncertain") + 1).value = None
    wb.save(path)
    with pytest.raises(at.AnnotationError, match="not valid"):
        at.freeze_annotation("calibration", "annotator1", env.key_dir, now=T0)
    assert not (env.pdir / "annotator1_freeze.json").exists()


def test_changed_sheet_fails_the_guard_and_cannot_be_refrozen(env):
    at.freeze_annotation("calibration", "annotator1", env.key_dir, now=T0)
    _fill_all(env.pdir / "annotator1.xlsx", flip={env.export["annotator1"][0]["item_id"]})
    with pytest.raises(at.AnnotationError, match="changed after it was frozen"):
        at.assert_annotation_frozen("calibration", "annotator1")
    with pytest.raises(at.AnnotationError, match="different version"):
        at.freeze_annotation("calibration", "annotator1", env.key_dir, now=T0)


def test_guard_without_freeze(env):
    with pytest.raises(at.AnnotationError, match="has not been frozen"):
        at.assert_annotation_frozen("calibration", "annotator1")


def test_agreement_script_refuses_unfrozen_or_changed_sheet(env, tmp_path):
    args = ["--purpose", "calibration", "--judge-output", str(tmp_path / "j.jsonl"),
            "--panel-output", str(tmp_path / "p.jsonl"), "--key-dir", str(env.key_dir)]
    assert ag.main(args) == 1                      # not frozen
    at.freeze_annotation("calibration", "annotator1", env.key_dir, now=T0)
    _fill_all(env.pdir / "annotator1.xlsx", flip={env.export["annotator1"][0]["item_id"]})
    assert ag.main(args) == 1                      # changed after freezing


def test_reannotation_refused_when_not_scheduled(env):
    at.freeze_annotation("calibration", "annotator1", env.key_dir, now=T0)
    with pytest.raises(at.AnnotationError, match="not scheduled"):
        at.cmd_export_reannotation(_args(env), now=datetime(2027, 1, 1, tzinfo=timezone.utc))
