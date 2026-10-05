"""Tests for logs/invalid_runs.jsonl -- known-bad run_history entries
(e.g. the 2026-10-05 pilot's stale-file resume collisions) that must be
hidden from History/Compare WITHOUT ever touching run_history.jsonl
itself. See invalid_runs_service.py's module docstring.
"""

import json

from app.services import history_service, invalid_runs_service


def test_mark_invalid_appends_without_overwriting(tmp_path, monkeypatch):
    path = tmp_path / "invalid_runs.jsonl"
    monkeypatch.setattr(invalid_runs_service, "INVALID_RUNS_PATH", path)

    invalid_runs_service.mark_invalid("A-aaa", "A", "reason one")
    invalid_runs_service.mark_invalid("B-bbb", "B", "reason two")

    assert invalid_runs_service.load_invalid_run_ids() == {"A-aaa", "B-bbb"}
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["reason"] == "reason one"


def test_load_invalid_run_ids_empty_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(invalid_runs_service, "INVALID_RUNS_PATH", tmp_path / "does_not_exist.jsonl")
    assert invalid_runs_service.load_invalid_run_ids() == set()


def test_list_history_hides_entries_marked_invalid(tmp_path, monkeypatch):
    history_path = tmp_path / "a_pure_llm" / "logs" / "run_history.jsonl"
    history_path.parent.mkdir(parents=True)
    good = {
        "run_started_at": "2026-10-05T01:00:00+00:00", "condition": "A", "status": "success",
        "provider": "openai", "model": "gpt-4o-mini", "n_sample_target": 10, "n_processed": 10,
        "seed": 42, "output_path": str(tmp_path / "good.jsonl"), "duration_sec": 1.0,
    }
    bad = {
        "run_started_at": "2026-10-05T00:25:22.564157+00:00", "condition": "A", "status": "no_results",
        "provider": "openai", "model": "gpt-4o-mini", "n_sample_target": 10, "n_processed": 0,
        "seed": 42, "output_path": str(tmp_path / "bad.jsonl"), "duration_sec": 9.9,
    }
    with open(history_path, "w") as f:
        f.write(json.dumps(good) + "\n")
        f.write(json.dumps(bad) + "\n")
    monkeypatch.setitem(history_service.HISTORY_PATHS, "A", history_path)

    bad_id = history_service._history_id("A", bad)
    invalid_path = tmp_path / "invalid_runs.jsonl"
    monkeypatch.setattr(invalid_runs_service, "INVALID_RUNS_PATH", invalid_path)
    invalid_runs_service.mark_invalid(bad_id, "A", "pilot 2026-10-05: stale-file resume collision")

    items, total = history_service.list_history("A", None, None, 1, 25)
    assert total == 1
    assert all(item["history_id"] != bad_id for item in items)
