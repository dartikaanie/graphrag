"""Offline tests (no LLM calls, no sklearn network/model download -- just
cohen_kappa_score on in-memory label lists) for judge_agreement.py's pure
logic: pairing primary/secondary records, confusion matrix, kappa
exclusion rule, and stratified sampling for the human-annotation export.
"""

from judge_agreement import (
    build_confusion_matrix,
    compute_kappas,
    pair_primary_secondary,
    stratified_sample,
    stratified_sample_for_human_export,
)


def _record(run_id, qid, judge_id, label, call_failed=False):
    return {"run_id": run_id, "question_id": qid, "judge_id": judge_id, "label": label,
            "call_failed": call_failed, "condition": "B"}


def test_pair_primary_secondary_matches_by_run_id_and_question_id():
    records = [
        _record("r1", 1, "primary", "FAKTUAL"),
        _record("r1", 1, "secondary", "HALUSINASI_SEBAGIAN"),
        _record("r1", 2, "primary", "ABSTAIN"),  # no secondary counterpart -- must be dropped
    ]
    pairs = pair_primary_secondary(records)
    assert len(pairs) == 1
    assert pairs[0][0]["question_id"] == 1
    assert pairs[0][1]["label"] == "HALUSINASI_SEBAGIAN"


def test_pair_primary_secondary_drops_failed_calls():
    records = [
        _record("r1", 1, "primary", None, call_failed=True),
        _record("r1", 1, "secondary", "FAKTUAL"),
    ]
    assert pair_primary_secondary(records) == []


def test_confusion_matrix_counts_and_shape():
    pairs = [
        (_record("r1", 1, "primary", "FAKTUAL"), _record("r1", 1, "secondary", "FAKTUAL")),
        (_record("r1", 2, "primary", "ABSTAIN"), _record("r1", 2, "secondary", "HALUSINASI_PENUH")),
    ]
    matrix = build_confusion_matrix(pairs)
    assert matrix["FAKTUAL"]["FAKTUAL"] == 1
    assert matrix["ABSTAIN"]["HALUSINASI_PENUH"] == 1
    assert matrix["HALUSINASI_SEBAGIAN"]["ABSTAIN"] == 0  # untouched cell stays 0


def test_compute_kappas_excludes_any_pair_with_an_abstain_on_either_side():
    pairs = [
        (_record("r1", 1, "primary", "FAKTUAL"), _record("r1", 1, "secondary", "FAKTUAL")),
        (_record("r1", 2, "primary", "HALUSINASI_SEBAGIAN"), _record("r1", 2, "secondary", "HALUSINASI_PENUH")),
        (_record("r1", 3, "primary", "ABSTAIN"), _record("r1", 3, "secondary", "HALUSINASI_PENUH")),
        (_record("r1", 4, "primary", "HALUSINASI_SEBAGIAN"), _record("r1", 4, "secondary", "ABSTAIN")),
    ]
    result = compute_kappas(pairs)
    assert result["n_used"] == 2  # only the two all-ordinal pairs survive
    assert result["n_excluded"] == 2
    assert result["weighted"] is not None
    assert result["unweighted"] is not None


def test_compute_kappas_returns_none_when_nothing_survives_exclusion():
    pairs = [
        (_record("r1", 1, "primary", "ABSTAIN"), _record("r1", 1, "secondary", "ABSTAIN")),
    ]
    result = compute_kappas(pairs)
    assert result["n_used"] == 0
    assert result["weighted"] is None
    assert result["unweighted"] is None


def test_stratified_sample_splits_evenly_across_conditions():
    by_condition = {
        "A": [{"question_id": i} for i in range(10)],
        "B": [{"question_id": i} for i in range(10)],
        "C": [{"question_id": i} for i in range(10)],
        "D": [{"question_id": i} for i in range(10)],
    }
    sample = stratified_sample(by_condition, n_total=8, seed=42)
    assert len(sample) == 8  # 4 conditions, 8/4 = 2 each exactly


def test_stratified_sample_distributes_remainder_deterministically():
    by_condition = {
        "A": [{"question_id": i} for i in range(10)],
        "B": [{"question_id": i} for i in range(10)],
        "C": [{"question_id": i} for i in range(10)],
    }
    # 10 / 3 conditions = 3 each + 1 remainder -> exactly one condition gets 4
    sample1 = stratified_sample(by_condition, n_total=10, seed=42)
    sample2 = stratified_sample(by_condition, n_total=10, seed=42)
    assert len(sample1) == 10
    assert sample1 == sample2  # same seed -> reproducible sample (order included)


def test_stratified_sample_is_empty_for_empty_input():
    assert stratified_sample({}, n_total=10, seed=42) == []


def test_stratified_sample_never_pads_a_too_small_pool():
    by_condition = {
        "A": [{"question_id": 1}],  # pool smaller than its share
        "B": [{"question_id": i} for i in range(10)],
    }
    sample = stratified_sample(by_condition, n_total=10, seed=42)
    assert len(sample) == 1 + 5  # A contributes only 1 (its whole pool), not its 5-item share


# ---------------------------------------------------------------------------
# stratified_sample_for_human_export -- two-dimensional: run_label x
# primary-judge ordinal label guarantee
# ---------------------------------------------------------------------------

def _export_item(run_id, qid, run_label, primary_label):
    return {"run_id": run_id, "question_id": qid, "run_label": run_label, "primary_label": primary_label}


def _make_pool(run_labels, labels_per_run_label, start_qid=0):
    """labels_per_run_label: {run_label: [primary_label, primary_label, ...]}"""
    items = []
    qid = start_qid
    for run_label in run_labels:
        for label in labels_per_run_label[run_label]:
            items.append(_export_item(f"run-{run_label}", qid, run_label, label))
            qid += 1
    return items


def test_export_guarantees_min_per_label_when_available():
    pool = _make_pool(
        ["X", "Y"],
        {"X": ["FAKTUAL"] * 10 + ["HALUSINASI_SEBAGIAN"] * 10 + ["HALUSINASI_PENUH"] * 10,
         "Y": ["FAKTUAL"] * 10 + ["HALUSINASI_SEBAGIAN"] * 10 + ["HALUSINASI_PENUH"] * 10},
    )
    sampled, metadata = stratified_sample_for_human_export(pool, n_total=20, seed=42, min_per_label=10)

    label_counts = {}
    for it in sampled:
        label_counts[it["primary_label"]] = label_counts.get(it["primary_label"], 0) + 1
    assert label_counts.get("FAKTUAL", 0) >= 10
    assert label_counts.get("HALUSINASI_SEBAGIAN", 0) >= 10
    assert label_counts.get("HALUSINASI_PENUH", 0) >= 10
    for label in ("FAKTUAL", "HALUSINASI_SEBAGIAN", "HALUSINASI_PENUH"):
        assert metadata["label_guarantee_report"][label]["guarantee_met"] is True


def test_export_guarantee_capped_when_label_scarce():
    pool = _make_pool(
        ["X"],
        {"X": ["FAKTUAL"] * 3 + ["HALUSINASI_SEBAGIAN"] * 10 + ["HALUSINASI_PENUH"] * 10},
    )
    sampled, metadata = stratified_sample_for_human_export(pool, n_total=5, seed=42, min_per_label=10)

    faktual_count = sum(1 for it in sampled if it["primary_label"] == "FAKTUAL")
    assert faktual_count == 3  # all available, never padded
    report = metadata["label_guarantee_report"]["FAKTUAL"]
    assert report["available"] == 3
    assert report["picked"] == 3
    assert report["guarantee_met"] is False


def test_export_guarantee_spreads_across_run_label():
    """Within one label's guarantee, selection must draw from BOTH
    run_labels present for that label, not cluster entirely in one."""
    pool = _make_pool(
        ["X", "Y"],
        {"X": ["FAKTUAL"] * 10, "Y": ["FAKTUAL"] * 10},
    )
    sampled, _ = stratified_sample_for_human_export(pool, n_total=10, seed=42, min_per_label=10)
    run_labels_present = {it["run_label"] for it in sampled}
    assert run_labels_present == {"X", "Y"}


def test_export_fills_remaining_slots_after_guarantees():
    pool = _make_pool(
        ["X"],
        {"X": ["FAKTUAL"] * 5 + ["HALUSINASI_SEBAGIAN"] * 5 + ["HALUSINASI_PENUH"] * 5 + [None] * 20},
    )
    sampled, metadata = stratified_sample_for_human_export(pool, n_total=30, seed=42, min_per_label=5)
    assert metadata["n_total_sampled"] == 30
    assert len(sampled) == 30


def test_export_guarantees_not_truncated_even_if_they_exceed_n_total():
    pool = _make_pool(
        ["X"],
        {"X": ["FAKTUAL"] * 10 + ["HALUSINASI_SEBAGIAN"] * 10 + ["HALUSINASI_PENUH"] * 10},
    )
    # 3 labels x min_per_label(10) = 30 already exceeds n_total(5)
    sampled, metadata = stratified_sample_for_human_export(pool, n_total=5, seed=42, min_per_label=10)
    assert len(sampled) == 30  # guarantees honored in full, never truncated to n_total
    assert metadata["n_total_sampled"] == 30


def test_export_items_without_primary_label_never_count_toward_a_guarantee():
    pool = _make_pool(["X"], {"X": [None] * 10})  # not yet judged by primary
    sampled, metadata = stratified_sample_for_human_export(pool, n_total=5, seed=42, min_per_label=10)
    assert len(sampled) == 5  # filled from the "remaining" pool, not any label guarantee
    for label in ("FAKTUAL", "HALUSINASI_SEBAGIAN", "HALUSINASI_PENUH"):
        assert metadata["label_guarantee_report"][label]["available"] == 0


def test_export_is_deterministic_for_same_seed():
    pool = _make_pool(
        ["X", "Y", "Z"],
        {"X": ["FAKTUAL"] * 10, "Y": ["HALUSINASI_SEBAGIAN"] * 10, "Z": ["HALUSINASI_PENUH"] * 10},
    )
    sampled1, _ = stratified_sample_for_human_export(pool, n_total=15, seed=7, min_per_label=5)
    sampled2, _ = stratified_sample_for_human_export(pool, n_total=15, seed=7, min_per_label=5)
    assert sampled1 == sampled2
