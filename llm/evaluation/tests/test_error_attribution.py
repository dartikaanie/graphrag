"""Test offline (tanpa I/O manifest, murni fungsi murni) untuk aturan
deterministik di analyze_error_attribution.py."""

import analyze_error_attribution as attr


# ---------------------------------------------------------------------
# retrieval_status
# ---------------------------------------------------------------------

def test_retrieval_status_empty_from_no_context():
    assert attr.compute_retrieval_status({"status": "no_context", "n_items": 0}) == "EMPTY"


def test_retrieval_status_none_when_missing():
    assert attr.compute_retrieval_status(None) is None


def test_retrieval_status_none_when_failed():
    assert attr.compute_retrieval_status({"status": "failed", "n_items": 3}) is None


def test_retrieval_status_sufficient():
    rec = {"status": "ok", "n_items": 3, "sufficiency": "CUKUP", "item_labels": [{"label": "RELEVAN"}]}
    assert attr.compute_retrieval_status(rec) == "SUFFICIENT"


def test_retrieval_status_irrelevant():
    rec = {"status": "ok", "n_items": 3, "sufficiency": "TIDAK_CUKUP",
           "item_labels": [{"label": "TIDAK_RELEVAN"}, {"label": "SEBAGIAN"}]}
    assert attr.compute_retrieval_status(rec) == "IRRELEVANT"


def test_retrieval_status_partial_default():
    rec = {"status": "ok", "n_items": 3, "sufficiency": "SEBAGIAN",
           "item_labels": [{"label": "TIDAK_RELEVAN"}, {"label": "RELEVAN"}]}
    assert attr.compute_retrieval_status(rec) == "PARTIAL"


def test_retrieval_status_partial_when_relevant_item_exists_despite_tidak_cukup():
    # TIDAK_CUKUP tapi ADA item RELEVAN -> bukan IRRELEVANT (butuh TIDAK ADA
    # item relevan SEKALIGUS TIDAK_CUKUP), jatuh ke PARTIAL.
    rec = {"status": "ok", "n_items": 2, "sufficiency": "TIDAK_CUKUP", "item_labels": [{"label": "RELEVAN"}]}
    assert attr.compute_retrieval_status(rec) == "PARTIAL"


# ---------------------------------------------------------------------
# hallucinated (loose vs strict)
# ---------------------------------------------------------------------

def test_hallucinated_loose():
    assert attr.compute_hallucinated("HALUSINASI_SEBAGIAN", strict=False) is True
    assert attr.compute_hallucinated("HALUSINASI_PENUH", strict=False) is True
    assert attr.compute_hallucinated("FAKTUAL", strict=False) is False


def test_hallucinated_strict():
    assert attr.compute_hallucinated("HALUSINASI_SEBAGIAN", strict=True) is False
    assert attr.compute_hallucinated("HALUSINASI_PENUH", strict=True) is True
    assert attr.compute_hallucinated("FAKTUAL", strict=True) is False


def test_hallucinated_none_when_missing():
    assert attr.compute_hallucinated(None, strict=False) is None


# ---------------------------------------------------------------------
# attribution -- B/C/D
# ---------------------------------------------------------------------

def test_attribution_empty_hallucinated():
    assert attr.compute_attribution("C", "EMPTY", True, None) == "ANCHORING_OR_COVERAGE_FAILURE"


def test_attribution_irrelevant_hallucinated():
    assert attr.compute_attribution("C", "IRRELEVANT", True, None) == "RETRIEVAL_FAILURE"


def test_attribution_partial_hallucinated():
    assert attr.compute_attribution("C", "PARTIAL", True, None) == "PARTIAL_RETRIEVAL"


def test_attribution_sufficient_hallucinated():
    assert attr.compute_attribution("C", "SUFFICIENT", True, None) == "GENERATION_FAILURE"


def test_attribution_off_target():
    assert attr.compute_attribution("C", "SUFFICIENT", False, "TIDAK_MENJAWAB") == "OFF_TARGET_ANSWER"
    assert attr.compute_attribution("C", "PARTIAL", False, "TIDAK_MENJAWAB") == "OFF_TARGET_ANSWER"


def test_attribution_correct_from_parametric():
    assert attr.compute_attribution("C", "EMPTY", False, "MENJAWAB") == "CORRECT_FROM_PARAMETRIC"
    assert attr.compute_attribution("C", "IRRELEVANT", False, "MENJAWAB") == "CORRECT_FROM_PARAMETRIC"


def test_attribution_success():
    assert attr.compute_attribution("C", "SUFFICIENT", False, "MENJAWAB") == "SUCCESS"
    assert attr.compute_attribution("C", "PARTIAL", False, "MENJAWAB") == "SUCCESS"


def test_attribution_unknown_when_data_missing():
    assert attr.compute_attribution("C", None, None, None) == "UNKNOWN"
    assert attr.compute_attribution("C", None, True, None) == "UNKNOWN"


# ---------------------------------------------------------------------
# attribution -- Kondisi A (tidak ada retrieval_status sama sekali)
# ---------------------------------------------------------------------

def test_attribution_condition_a_hallucinated():
    assert attr.compute_attribution("A", None, True, None) == "PARAMETRIC_HALLUCINATION"


def test_attribution_condition_a_off_target():
    assert attr.compute_attribution("A", None, False, "TIDAK_MENJAWAB") == "OFF_TARGET_ANSWER"


def test_attribution_condition_a_success():
    assert attr.compute_attribution("A", None, False, "MENJAWAB") == "SUCCESS"


def test_attribution_condition_a_unknown_when_hallucination_missing():
    assert attr.compute_attribution("A", None, None, "MENJAWAB") == "UNKNOWN"
