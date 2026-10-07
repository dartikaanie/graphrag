"""Offline tests (no LLM calls, no sklearn network/model download -- just
cohen_kappa_score on in-memory label lists) for judge_agreement.py's pure
logic: pairing primary/secondary records, confusion matrix, kappa
exclusion rule, and stratified sampling for the human-annotation export.
"""

import pytest

from judge_agreement import (
    build_confusion_matrix,
    compute_kappas,
    compute_run_label_summary,
    pair_primary_secondary,
    pair_two_judges,
    stratified_sample,
    stratified_sample_for_human_export,
)


def _record(run_id, qid, judge_id, label, call_failed=False, run_label=None):
    return {"run_id": run_id, "question_id": qid, "judge_id": judge_id, "label": label,
            "call_failed": call_failed, "condition": "B", "run_label": run_label or "B-plain"}


def test_pair_two_judges_works_for_any_pair_not_just_primary_secondary():
    records = [
        _record("r1", 1, "fallback", "FAKTUAL"),
        _record("r1", 1, "secondary", "HALUSINASI_SEBAGIAN"),
        _record("r1", 2, "fallback", "ABSTAIN"),  # no secondary counterpart
    ]
    pairs = pair_two_judges(records, "fallback", "secondary")
    assert len(pairs) == 1
    assert pairs[0][0]["judge_id"] == "fallback"
    assert pairs[0][1]["label"] == "HALUSINASI_SEBAGIAN"


def test_pair_primary_secondary_is_pair_two_judges_specialized():
    records = [_record("r1", 1, "primary", "FAKTUAL"), _record("r1", 1, "secondary", "FAKTUAL")]
    assert pair_primary_secondary(records) == pair_two_judges(records, "primary", "secondary")


def test_compute_run_label_summary_math():
    records = [
        _record("r1", 1, "primary", "FAKTUAL", run_label="B-plain"),
        _record("r1", 2, "primary", "HALUSINASI_SEBAGIAN", run_label="B-plain"),
        _record("r1", 3, "primary", "HALUSINASI_PENUH", run_label="B-plain"),
        _record("r1", 4, "primary", "ABSTAIN", run_label="B-plain"),
        _record("r1", 5, "secondary", "FAKTUAL", run_label="B-plain"),  # different judge_id, must be filtered out
    ]
    summary = compute_run_label_summary(records, "primary")
    assert set(summary.keys()) == {"B-plain"}
    s = summary["B-plain"]
    assert s["n"] == 4
    assert s["abstention_rate"] == 0.25
    assert s["hall_rate_excl_abstain"] == round(2 / 3, 3)
    assert s["hall_rate_incl_abstain"] == round(3 / 4, 3)
    assert s["label_distribution"] == {"FAKTUAL": 1, "HALUSINASI_SEBAGIAN": 1, "HALUSINASI_PENUH": 1, "ABSTAIN": 1}


def test_compute_run_label_summary_empty_for_unknown_judge_id():
    records = [_record("r1", 1, "primary", "FAKTUAL")]
    assert compute_run_label_summary(records, "secondary") == {}


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


def test_compute_human_kappa_insufficient_data_when_no_human_labels(tmp_path):
    import csv
    import json

    from judge_agreement import compute_human_kappa

    csv_path = tmp_path / "human.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "human_label"])
        writer.writerow(["item_0001", ""])  # never labeled

    mapping_path = tmp_path / "mapping.jsonl"
    mapping_path.write_text(json.dumps({"item_id": "item_0001", "run_id": "r1", "question_id": 1}) + "\n")

    judge_output = tmp_path / "judge_out.jsonl"
    judge_output.write_text(json.dumps(_record("r1", 1, "primary", "FAKTUAL")) + "\n")

    results = compute_human_kappa(str(csv_path), str(mapping_path), [str(judge_output)])
    assert results["primary"]["insufficient_data"] is True
    assert results["primary"]["n_missing_human"] == 1


def test_compute_human_kappa_computes_weighted_and_unweighted(tmp_path):
    import csv
    import json

    from judge_agreement import compute_human_kappa

    csv_path = tmp_path / "human.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "human_label"])
        writer.writerow(["item_0001", "FAKTUAL"])
        writer.writerow(["item_0002", "HALUSINASI_PENUH"])

    mapping_path = tmp_path / "mapping.jsonl"
    with open(mapping_path, "w") as f:
        f.write(json.dumps({"item_id": "item_0001", "run_id": "r1", "question_id": 1}) + "\n")
        f.write(json.dumps({"item_id": "item_0002", "run_id": "r1", "question_id": 2}) + "\n")

    judge_output = tmp_path / "judge_out.jsonl"
    with open(judge_output, "w") as f:
        f.write(json.dumps(_record("r1", 1, "primary", "FAKTUAL")) + "\n")
        f.write(json.dumps(_record("r1", 2, "primary", "HALUSINASI_PENUH")) + "\n")

    results = compute_human_kappa(str(csv_path), str(mapping_path), [str(judge_output)], judge_ids=("primary",))
    assert results["primary"]["insufficient_data"] is False
    assert results["primary"]["n_paired"] == 2
    assert results["primary"]["weighted_kappa"] == 1.0


def test_load_judge_records_refuses_mixed_prompt_version(tmp_path):
    import json

    from judge_agreement import MixedJudgeVersionError, load_judge_records

    path = tmp_path / "mixed.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"run_id": "r1", "question_id": 1, "judge_id": "primary",
                             "prompt_version": "judge-v1", "blinding_version": "blind-v2"}) + "\n")
        f.write(json.dumps({"run_id": "r1", "question_id": 2, "judge_id": "primary",
                             "prompt_version": "judge-v2", "blinding_version": "blind-v2"}) + "\n")

    with pytest.raises(MixedJudgeVersionError):
        load_judge_records([str(path)])


def test_load_judge_records_refuses_mixed_blinding_version(tmp_path):
    import json

    from judge_agreement import MixedJudgeVersionError, load_judge_records

    path = tmp_path / "mixed_blinding.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"run_id": "r1", "question_id": 1, "judge_id": "primary",
                             "prompt_version": "judge-v1", "blinding_version": "blind-v1"}) + "\n")
        f.write(json.dumps({"run_id": "r1", "question_id": 2, "judge_id": "primary",
                             "prompt_version": "judge-v1", "blinding_version": "blind-v2"}) + "\n")

    with pytest.raises(MixedJudgeVersionError):
        load_judge_records([str(path)])


def test_load_judge_records_accepts_single_version(tmp_path):
    import json

    from judge_agreement import load_judge_records

    path = tmp_path / "clean.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"run_id": "r1", "question_id": 1, "judge_id": "primary",
                             "prompt_version": "judge-v1", "blinding_version": "blind-v2"}) + "\n")
        f.write(json.dumps({"run_id": "r1", "question_id": 2, "judge_id": "primary",
                             "prompt_version": "judge-v1", "blinding_version": "blind-v2"}) + "\n")

    assert len(load_judge_records([str(path)])) == 2


def test_compute_run_label_summary_parse_error_and_consistency_rates():
    def _rec(qid, label, parse_error=False, consistent=None):
        r = _record("r1", qid, "primary", label, run_label="A")
        r["parse_error"] = parse_error
        r["consistent"] = consistent
        return r

    records = [
        _rec(1, "FAKTUAL", consistent=True),
        _rec(2, "FAKTUAL", consistent=False),
        _rec(3, "HALUSINASI_PENUH", parse_error=True, consistent=None),  # excluded from consistency denominator
        _rec(4, "FAKTUAL", consistent=True),
    ]
    summary = compute_run_label_summary(records, "primary")["A"]
    assert summary["parse_error_rate"] == round(1 / 4, 3)
    # consistency_rate excludes the parse-error record (consistent=None):
    # 2 True out of 3 non-None values.
    assert summary["consistency_rate"] == round(2 / 3, 3)


def test_compute_run_label_summary_consistency_rate_none_when_no_data():
    records = [_record("r1", 1, "primary", "FAKTUAL", run_label="A")]
    summary = compute_run_label_summary(records, "primary")["A"]
    assert summary["parse_error_rate"] == 0.0
    assert summary["consistency_rate"] is None


def test_load_judge_records_refuses_mixed_version_across_separate_files(tmp_path):
    """Each file individually is internally consistent (all judge-v1,
    all judge-v2), but combining them in one load_judge_records() call
    must still be refused -- the per-file check alone would miss this."""
    import json

    from judge_agreement import MixedJudgeVersionError, load_judge_records

    v1_path = tmp_path / "jv1.jsonl"
    v1_path.write_text(json.dumps({"run_id": "r1", "question_id": 1, "judge_id": "primary",
                                    "prompt_version": "judge-v1", "blinding_version": "blind-v2"}) + "\n")
    v2_path = tmp_path / "jv2.jsonl"
    v2_path.write_text(json.dumps({"run_id": "r1", "question_id": 2, "judge_id": "primary",
                                    "prompt_version": "judge-v2", "blinding_version": "blind-v2"}) + "\n")

    with pytest.raises(MixedJudgeVersionError):
        load_judge_records([str(v1_path), str(v2_path)])


def test_load_judge_records_accepts_separate_calls_per_version(tmp_path):
    """The v1-vs-v2 comparison view's correct usage pattern: one call
    per version, never combined."""
    import json

    from judge_agreement import load_judge_records

    v1_path = tmp_path / "jv1.jsonl"
    v1_path.write_text(json.dumps({"run_id": "r1", "question_id": 1, "judge_id": "primary",
                                    "prompt_version": "judge-v1", "blinding_version": "blind-v2"}) + "\n")
    v2_path = tmp_path / "jv2.jsonl"
    v2_path.write_text(json.dumps({"run_id": "r1", "question_id": 1, "judge_id": "primary",
                                    "prompt_version": "judge-v2", "blinding_version": "blind-v2"}) + "\n")

    v1_records = load_judge_records([str(v1_path)])
    v2_records = load_judge_records([str(v2_path)])
    assert v1_records[0]["prompt_version"] == "judge-v1"
    assert v2_records[0]["prompt_version"] == "judge-v2"


def test_compute_run_label_summary_v2_extra_math():
    from judge_agreement import compute_run_label_summary_v2_extra

    def v2_record(qid, label, completeness, claims=None, attempted_claims_conflict=False):
        r = _record("r1", qid, "primary", label, run_label="A")
        r["completeness"] = completeness
        r["claims"] = claims or []
        r["attempted_claims_conflict"] = attempted_claims_conflict
        return r

    records = [
        v2_record(1, "FAKTUAL", "FULL"),
        v2_record(2, "FAKTUAL", "PARTIAL"),
        v2_record(3, "HALUSINASI_SEBAGIAN", "FULL",
                   claims=[{"claim": "x", "reference_conflict": True}]),
        v2_record(4, "ABSTAIN", "NONE", attempted_claims_conflict=True),
    ]
    summary = compute_run_label_summary_v2_extra(records, "primary")["A"]
    assert summary["completeness_distribution"] == {"FULL": 2, "PARTIAL": 1, "NONE": 1}
    # only record 1 is FAKTUAL + FULL
    assert summary["factual_and_complete_rate"] == round(1 / 4, 3)
    # only record 3 has a reference_conflict claim
    assert summary["pct_with_reference_conflict"] == round(100 * 1 / 4, 2)
    # only record 4 has attempted_claims_conflict=True
    assert summary["attempted_claims_conflict_rate"] == round(1 / 4, 3)


def test_compute_run_label_summary_v2_extra_empty_for_unknown_judge_id():
    from judge_agreement import compute_run_label_summary_v2_extra

    records = [_record("r1", 1, "primary", "FAKTUAL")]
    assert compute_run_label_summary_v2_extra(records, "secondary") == {}


def test_compute_kappas_never_returns_nan_json_incompatible(tmp_path):
    """A degenerate single-ordinal-pair case (or any case with zero
    label variance) makes sklearn return NaN, not raise -- NaN is not
    valid JSON, so every API response serializing this dict would 500
    without normalizing it to None (same meaning as the n_used=0 case)."""
    import math

    from judge_agreement import compute_kappas

    pairs = [
        (_record("r1", 1, "primary", "ABSTAIN"), _record("r1", 1, "secondary", "ABSTAIN")),
        (_record("r1", 2, "primary", "FAKTUAL"), _record("r1", 2, "secondary", "FAKTUAL")),
    ]
    result = compute_kappas(pairs)
    assert result["n_used"] == 1  # only the FAKTUAL/FAKTUAL pair is ordinal
    assert result["weighted"] is None or not (isinstance(result["weighted"], float) and math.isnan(result["weighted"]))
    assert result["unweighted"] is None or not (isinstance(result["unweighted"], float) and math.isnan(result["unweighted"]))
    import json
    json.dumps(result)  # must not raise
