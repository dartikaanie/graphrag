"""compare_conditions_stats only reads judge runs at JUDGE_TEMPERATURE (0.0);
legacy t0.1 runs (`_t0-1` files) are skipped with a warning."""

import json

from compare_conditions_stats import find_latest_manifest_entry


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_t0_1_runs_are_skipped_even_if_newer(tmp_path, capsys):
    run = tmp_path / "condition_a.jsonl"
    run.write_text("")
    manifest = tmp_path / "answer_relevance_run_history.jsonl"
    _write(manifest, [
        {"input_path": str(run), "timestamp": "2026-10-01", "judge_temperature": 0.0, "output_path": "ansrel_t0-0.jsonl"},
        {"input_path": str(run), "timestamp": "2026-10-09", "judge_temperature": 0.1, "output_path": "ansrel_t0-1.jsonl"},
    ])
    assert find_latest_manifest_entry(manifest, run)["output_path"] == "ansrel_t0-0.jsonl"
    assert "ansrel_t0-1.jsonl" in capsys.readouterr().out


def test_only_t0_1_runs_gives_none_and_warns(tmp_path, capsys):
    run = tmp_path / "condition_a.jsonl"
    run.write_text("")
    manifest = tmp_path / "answer_relevance_run_history.jsonl"
    _write(manifest, [
        {"input_path": str(run), "timestamp": "2026-09-28", "judge_temperature": 0.1, "output_path": "ansrel_t0-1.jsonl"},
        {"input_path": str(run), "timestamp": "2026-09-29", "output_path": "no_temperature_field.jsonl"},
    ])
    assert find_latest_manifest_entry(manifest, run) is None
    assert "TIDAK ADA run judge" in capsys.readouterr().out
