"""Tests for batch_grouping_service.py -- grouping runs launched together
(recorded batch_id) or inferred by shared params + launch-time window."""

from app.services import batch_grouping_service as svc


def _run(run_label, run_started_at, batch_id=None, n_sample_target=10, seed=42,
         provider="openai", model="gpt-4o-mini", oversample_pool=1536):
    return {
        "run_label": run_label, "run_started_at": run_started_at, "batch_id": batch_id,
        "n_sample_target": n_sample_target, "seed": seed, "provider": provider,
        "model": model, "oversample_pool": oversample_pool,
    }


def test_recorded_batch_id_groups_take_priority_and_are_not_marked_inferred():
    runs = [
        _run("A", "2026-10-05T00:00:00+00:00", batch_id="batch-1"),
        _run("B-plain", "2026-10-05T00:00:05+00:00", batch_id="batch-1"),
    ]
    batches = svc.group_into_batches(runs)
    assert len(batches) == 1
    assert batches[0]["batch_id"] == "batch-1"
    assert batches[0]["inferred"] is False
    assert len(batches[0]["runs"]) == 2


def test_inferred_grouping_clusters_by_shared_params_and_time_window():
    runs = [
        _run("A", "2026-10-05T00:00:00+00:00"),
        _run("B-plain", "2026-10-05T00:02:00+00:00"),
        _run("D-plain", "2026-10-05T00:08:00+00:00"),
    ]
    batches = svc.group_into_batches(runs)
    assert len(batches) == 1
    assert batches[0]["inferred"] is True
    assert len(batches[0]["runs"]) == 3


def test_inferred_grouping_splits_on_time_gap_beyond_window():
    runs = [
        _run("A", "2026-10-05T00:00:00+00:00"),
        _run("B-plain", "2026-10-05T01:00:00+00:00"),  # 1 hour later -- different batch
    ]
    batches = svc.group_into_batches(runs)
    assert len(batches) == 2
    assert all(b["inferred"] for b in batches)
    assert all(len(b["runs"]) == 1 for b in batches)


def test_inferred_grouping_splits_on_different_shared_params():
    runs = [
        _run("A", "2026-10-05T00:00:00+00:00", n_sample_target=10),
        _run("A", "2026-10-05T00:00:05+00:00", n_sample_target=384),  # different n_sample -- different batch
    ]
    batches = svc.group_into_batches(runs)
    assert len(batches) == 2


def test_inferred_grouping_chains_within_window_even_if_span_exceeds_it():
    """9 runs launched a few seconds apart each, spanning longer than the
    window end-to-end, must still cluster as ONE batch (chained window,
    not a fixed window from the first run)."""
    runs = [_run("A", f"2026-10-05T00:0{i}:00+00:00") for i in range(8)]  # 00:00 .. 00:07, 1 min apart
    batches = svc.group_into_batches(runs)
    assert len(batches) == 1
    assert len(batches[0]["runs"]) == 8


def test_batches_sorted_newest_launched_first():
    runs = [
        _run("A", "2026-09-01T00:00:00+00:00"),
        _run("A", "2026-10-05T00:00:00+00:00"),
    ]
    batches = svc.group_into_batches(runs)
    assert batches[0]["launched_at"] == "2026-10-05T00:00:00+00:00"
    assert batches[1]["launched_at"] == "2026-09-01T00:00:00+00:00"


def test_batch_launched_at_field_preferred_over_min_run_started_at():
    runs = [
        _run("A", "2026-10-05T00:05:00+00:00", batch_id="batch-1"),
        _run("B-plain", "2026-10-05T00:00:00+00:00", batch_id="batch-1"),
    ]
    runs[0]["batch_launched_at"] = "2026-10-05T00:00:00+00:00"
    runs[1]["batch_launched_at"] = "2026-10-05T00:00:00+00:00"
    batches = svc.group_into_batches(runs)
    assert batches[0]["launched_at"] == "2026-10-05T00:00:00+00:00"
