"""Phase 3 -- sample-consistency check (History -> Compare). No LLM calls,
no real Neo4j/DuckDB -- compute_sample_consistency() reads question_id
straight off fixture .jsonl result files via history_service's own
get_history_detail() (which itself just reads run_history.jsonl records
written to tmp_path, same pattern as test_history_service_params.py).
"""

import json

from app.services import history_service


def _write_run_history(history_path, record: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        f.write(json.dumps(record) + "\n")


def _write_result_file(path, question_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for qid in question_ids:
            f.write(json.dumps({
                "question_id": qid,
                "title": f"q{qid}",
                "retrieved_context": [{"question_id": qid + 1000, "chunk_text": "x"}],
            }) + "\n")


def _setup_run(tmp_path, monkeypatch, condition: str, run_started_at: str, question_ids: list[int]) -> str:
    history_path = tmp_path / f"logs_{condition}_{run_started_at}" / "run_history.jsonl"
    output_path = tmp_path / f"condition_{condition.lower()}_{run_started_at}.jsonl"
    record = {
        "run_started_at": run_started_at,
        "condition": condition,
        "status": "success",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "n_sample_target": len(question_ids),
        "n_processed": len(question_ids),
        "seed": 42,
        "output_path": str(output_path),
        "duration_sec": 1.0,
    }
    _write_run_history(history_path, record)
    _write_result_file(output_path, question_ids)
    monkeypatch.setitem(history_service.HISTORY_PATHS, condition, history_path)
    return history_service._history_id(condition, record)


def test_identical_samples(tmp_path, monkeypatch):
    qids = [1, 2, 3, 4, 5]
    id_a = _setup_run(tmp_path, monkeypatch, "A", "2026-09-29T00:00:00+00:00", qids)
    id_b = _setup_run(tmp_path, monkeypatch, "B", "2026-09-29T00:00:01+00:00", qids)

    result = history_service.compute_sample_consistency([id_a, id_b])

    assert result["sample_consistency"] == "identical"
    assert result["intersection_size"] == 5
    assert {r["n"] for r in result["runs"]} == {5}


def test_subset_nested_exact_prefix(tmp_path, monkeypatch):
    """Smaller run's question_ids are the first N of the larger run's, IN
    ORDER -- true nested sampling (n=10 pilot inside n=384 plan)."""
    large = [10, 20, 30, 40, 50, 60]
    small = [10, 20, 30]  # exact prefix of `large`
    id_a = _setup_run(tmp_path, monkeypatch, "C", "2026-09-29T00:00:00+00:00", large)
    id_b = _setup_run(tmp_path, monkeypatch, "D", "2026-09-29T00:00:01+00:00", small)

    result = history_service.compute_sample_consistency([id_a, id_b])

    assert result["sample_consistency"] == "subset_nested"
    assert result["intersection_size"] == 3
    assert result["is_exact_prefix"] is True


def test_subset_nested_but_not_exact_prefix(tmp_path, monkeypatch):
    """Smaller run's SET is a subset of the larger run's, but not in the
    same first-N order -- still 'subset_nested' (same underlying pool), but
    is_exact_prefix must be False, not silently True."""
    large = [10, 20, 30, 40, 50, 60]
    small = [30, 10, 20]  # same set as large[:3], different order
    id_a = _setup_run(tmp_path, monkeypatch, "A", "2026-09-29T00:00:00+00:00", large)
    id_b = _setup_run(tmp_path, monkeypatch, "B", "2026-09-29T00:00:01+00:00", small)

    result = history_service.compute_sample_consistency([id_a, id_b])

    assert result["sample_consistency"] == "subset_nested"
    assert result["is_exact_prefix"] is False


def test_different_samples(tmp_path, monkeypatch):
    id_a = _setup_run(tmp_path, monkeypatch, "A", "2026-09-29T00:00:00+00:00", [1, 2, 3])
    id_b = _setup_run(tmp_path, monkeypatch, "B", "2026-09-29T00:00:01+00:00", [1, 2, 999])

    result = history_service.compute_sample_consistency([id_a, id_b])

    assert result["sample_consistency"] == "different"
    assert result["intersection_size"] == 2
    assert result["is_exact_prefix"] is None


def test_three_way_identical_matches_real_pilot_shape(tmp_path, monkeypatch):
    """Mirrors the real five n=10 pilot runs (A/B/C-trust/C-uniform/D, all
    evaluating the exact same 10 questions in the exact same order)."""
    qids = [38118194, 76224221, 20443560, 411756, 2617170, 56612920, 3882147, 26406581, 11118023, 51061836]
    ids = [
        _setup_run(tmp_path, monkeypatch, cond, f"2026-09-28T18:{i:02d}:00+00:00", qids)
        for i, cond in enumerate(["A", "B", "C", "D"])
    ]
    result = history_service.compute_sample_consistency(ids)
    assert result["sample_consistency"] == "identical"
    assert result["intersection_size"] == 10


def test_fewer_than_two_resolvable_runs_reports_different_with_note():
    result = history_service.compute_sample_consistency(["A-doesnotexist"])
    assert result["sample_consistency"] == "different"
    assert "note" in result


def test_cache_invalidates_on_mtime_change(tmp_path, monkeypatch):
    qids = [1, 2, 3]
    id_a = _setup_run(tmp_path, monkeypatch, "A", "2026-09-29T00:00:00+00:00", qids)
    detail = history_service.get_history_detail(id_a)
    output_path = detail["output_path"]

    first = history_service._read_question_ids_cached(output_path)
    assert first == qids

    # Rewrite the file with different content but the SAME mtime forced
    # forward -- the cache must pick up the new content, not serve stale.
    import time
    time.sleep(0.01)
    _write_result_file(__import__("pathlib").Path(output_path), [1, 2, 3, 4])

    second = history_service._read_question_ids_cached(output_path)
    assert second == [1, 2, 3, 4]


def test_question_id_reader_uses_top_level_key_not_first_key_or_nested(tmp_path):
    """Regression guard for the earlier regex-based reader: this fixture
    deliberately puts "question_id" as NOT the first key, AND gives
    retrieved_context items their OWN "question_id" (a different field --
    the source thread id of that retrieved chunk) with values that would
    be picked up by a naive "first occurrence of the substring" scan.
    json.loads(line)["question_id"] must return the TOP-LEVEL value
    regardless of key order or how many nested "question_id" keys exist."""
    path = tmp_path / "condition_c_key_order_test.jsonl"
    records = [
        # question_id deliberately NOT first; retrieved_context (with its
        # own nested question_id values) appears BEFORE question_id in the
        # dict -- a first-match-wins regex would return 999001, not 555.
        {
            "title": "some question",
            "retrieved_context": [
                {"question_id": 999001, "chunk_text": "x"},
                {"question_id": 999002, "chunk_text": "y"},
            ],
            "question_id": 555,
            "llm_answer": "...",
        },
        # Second record: question_id first this time, but still has nested
        # question_id values in retrieved_context that must NOT be counted
        # as separate top-level ids.
        {
            "question_id": 777,
            "retrieved_context": [{"question_id": 999003, "chunk_text": "z"}],
        },
    ]
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    ids = history_service._read_question_ids_cached(str(path))

    assert ids == [555, 777]
    assert 999001 not in ids
    assert 999002 not in ids
    assert 999003 not in ids
