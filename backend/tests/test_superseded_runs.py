"""Tests for logs/superseded_runs.jsonl -- run_history entries that were
genuinely valid but have since been replaced by a later re-run of the
identical config (e.g. the 2026-10-05 pilot's 3 real successes, re-run
under the config-hashed filename scheme). Hidden by default in
History/Compare, unlike invalid_runs.jsonl -- a caller can ask to see
them again. run_history.jsonl itself is never touched either way.
"""

import json

from app.services import history_service, superseded_runs_service


def test_mark_superseded_appends_without_overwriting(tmp_path, monkeypatch):
    path = tmp_path / "superseded_runs.jsonl"
    monkeypatch.setattr(superseded_runs_service, "SUPERSEDED_RUNS_PATH", path)

    superseded_runs_service.mark_superseded("C-aaa", "C", "reason one")
    superseded_runs_service.mark_superseded("D-bbb", "D", "reason two")

    assert superseded_runs_service.load_superseded_run_ids() == {"C-aaa", "D-bbb"}
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["reason"] == "reason one"


def test_load_superseded_run_ids_empty_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(superseded_runs_service, "SUPERSEDED_RUNS_PATH", tmp_path / "does_not_exist.jsonl")
    assert superseded_runs_service.load_superseded_run_ids() == set()


def _write_history(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_list_history_hides_superseded_by_default_but_shows_when_asked(tmp_path, monkeypatch):
    history_path = tmp_path / "c_graphrag" / "logs" / "run_history.jsonl"
    old = {
        "run_started_at": "2026-10-05T00:22:54.904247+00:00", "condition": "C", "status": "success",
        "provider": "openai", "model": "gpt-4o-mini", "n_sample_target": 10, "n_processed": 10,
        "seed": 42, "output_path": str(tmp_path / "old.jsonl"), "duration_sec": 1.0,
    }
    new = {
        "run_started_at": "2026-10-06T00:00:00+00:00", "condition": "C", "status": "success",
        "provider": "openai", "model": "gpt-4o-mini", "n_sample_target": 10, "n_processed": 10,
        "seed": 42, "output_path": str(tmp_path / "new.jsonl"), "duration_sec": 1.0,
    }
    _write_history(history_path, [old, new])
    monkeypatch.setitem(history_service.HISTORY_PATHS, "C", history_path)

    old_id = history_service._history_id("C", old)
    superseded_path = tmp_path / "superseded_runs.jsonl"
    monkeypatch.setattr(superseded_runs_service, "SUPERSEDED_RUNS_PATH", superseded_path)
    superseded_runs_service.mark_superseded(
        old_id, "C", "pilot 2026-10-05: re-run with config-hashed filenames and manifests",
    )

    items, total = history_service.list_history("C", None, None, 1, 25)
    assert total == 1
    assert all(item["history_id"] != old_id for item in items)

    items, total = history_service.list_history("C", None, None, 1, 25, show_superseded=True)
    assert total == 2
    assert any(item["history_id"] == old_id for item in items)
