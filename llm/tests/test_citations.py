"""Regression test for the NF2-validator fix (prompt v3 root-cause report,
docs/NF2_ROOT_CAUSE_PLACEHOLDER_CITATIONS.md): extract_citations() must
count a literal "[SO-<id>]"-style placeholder token as an ATTEMPTED but
INVALID citation (has_citation=True, has_valid_citation=False), never as
"no citation attempt at all" (which would wrongly conflate it with an
answer that doesn't mention citations in any form). No LLM calls.
"""

from llm.citations import (
    CITATION_PATTERN,
    CITATION_TOKEN_PATTERN,
    citation_token_breakdown,
    classify_citation_outcome,
    compute_citation_report,
    extract_citations,
)

RETRIEVED = [{"question_id": 123, "chunk_text": "some context"}]


def test_placeholder_only_counts_as_attempted_but_invalid():
    answer = "You can fix this by doing X [SO-<id>]."
    has_citation, cited_ids, has_valid_citation, valid_ids = extract_citations(answer, RETRIEVED)
    assert has_citation is True  # attempted
    assert has_valid_citation is False  # but invalid
    assert cited_ids == []  # no real digit id to report
    assert valid_ids == []


def test_multiple_placeholder_variants_all_count_as_attempted():
    answer = "See [SO-<id1>] and also [SO-<id2>] for details."
    has_citation, _, has_valid_citation, _ = extract_citations(answer, RETRIEVED)
    assert has_citation is True
    assert has_valid_citation is False


def test_real_valid_citation_still_works_unchanged():
    answer = "You can fix this by doing X [SO-123]."
    has_citation, cited_ids, has_valid_citation, valid_ids = extract_citations(answer, RETRIEVED)
    assert has_citation is True
    assert cited_ids == [123]
    assert has_valid_citation is True
    assert valid_ids == [123]


def test_real_but_hallucinated_id_is_attempted_and_invalid():
    """A well-formed digit citation whose id is NOT in the given context
    -- already correctly invalid before this fix, must remain so."""
    answer = "You can fix this by doing X [SO-999999]."
    has_citation, cited_ids, has_valid_citation, valid_ids = extract_citations(answer, RETRIEVED)
    assert has_citation is True
    assert cited_ids == [999999]
    assert has_valid_citation is False
    assert valid_ids == []


def test_no_citation_attempt_at_all_stays_false():
    answer = "You can fix this by doing X, no sources needed."
    has_citation, cited_ids, has_valid_citation, valid_ids = extract_citations(answer, RETRIEVED)
    assert has_citation is False
    assert cited_ids == []
    assert has_valid_citation is False
    assert valid_ids == []


def test_citation_token_pattern_does_not_false_positive_on_unrelated_brackets():
    assert not CITATION_TOKEN_PATTERN.search("See the [Solution] section below.")
    assert not CITATION_TOKEN_PATTERN.search("Marked as [SOLVED] by the OP.")


def test_citation_pattern_itself_unchanged_digit_only():
    """CITATION_PATTERN (the pre-existing, digit-only pattern used for
    cited_ids/valid_ids) must NOT have been widened -- only a new,
    separate CITATION_TOKEN_PATTERN was added for has_citation."""
    assert CITATION_PATTERN.findall("[SO-<id>]") == []
    assert CITATION_PATTERN.findall("[SO-123]") == ["123"]


# ---------------------------------------------------------------------------
# Mixed valid + invalid citations in the SAME answer -- the case NF2 alone
# (has_valid_citation) can't distinguish from a cleanly-valid answer, which
# is exactly why fabricated_citation_rate/citation_precision exist.
# ---------------------------------------------------------------------------

def test_mixed_valid_and_invalid_counts_both_in_token_breakdown():
    """One real valid citation, one placeholder, one hallucinated
    (well-formed but not-in-context) id -- all three in the same answer."""
    answer = "First claim [SO-123]. Second claim [SO-<id>]. Third claim [SO-999999]."
    breakdown = citation_token_breakdown(answer, RETRIEVED)
    assert breakdown["n_tokens"] == 3
    assert breakdown["n_valid_tokens"] == 1
    assert breakdown["n_invalid_tokens"] == 2
    assert breakdown["has_any_invalid_token"] is True


def test_mixed_answer_still_classifies_as_valid_for_the_3_way_split():
    """extract_citations()'s has_valid_citation is True (>=1 valid id
    exists) even though the SAME answer also has invalid tokens --
    classify_citation_outcome() must call this "valid", not "invalid_only",
    matching the 3-way split's definition (has_valid_citation drives the
    split; fabricated_citation_rate is the SEPARATE, orthogonal signal for
    "also contains an invalid token")."""
    answer = "First claim [SO-123]. Second claim [SO-<id>]."
    has_citation, _, has_valid_citation, _ = extract_citations(answer, RETRIEVED)
    assert has_citation is True
    assert has_valid_citation is True
    assert classify_citation_outcome(has_citation, has_valid_citation) == "valid"


def test_classify_citation_outcome_all_three_buckets():
    assert classify_citation_outcome(has_citation=True, has_valid_citation=True) == "valid"
    assert classify_citation_outcome(has_citation=True, has_valid_citation=False) == "invalid_only"
    assert classify_citation_outcome(has_citation=False, has_valid_citation=False) == "no_citation"


def test_compute_citation_report_on_mixed_run():
    """One run with all four outcomes present: cleanly valid, mixed
    valid+invalid (still counts as "valid" for the 3-way split, but DOES
    count toward fabricated_citation_rate), invalid-only, and no citation
    at all."""
    records = [
        {"llm_answer": "Clean claim [SO-123].", "retrieved_context": RETRIEVED,
         "has_citation": True, "has_valid_citation": True},
        {"llm_answer": "Mixed claim [SO-123] and [SO-<id>].", "retrieved_context": RETRIEVED,
         "has_citation": True, "has_valid_citation": True},
        {"llm_answer": "Invalid only [SO-<id>].", "retrieved_context": RETRIEVED,
         "has_citation": True, "has_valid_citation": False},
        {"llm_answer": "No citation at all.", "retrieved_context": RETRIEVED,
         "has_citation": False, "has_valid_citation": False},
    ]
    report = compute_citation_report(records)
    assert report["n_answers"] == 4
    assert report["pct_valid"] == 50.0  # 2 of 4 (clean + mixed)
    assert report["pct_invalid_only"] == 25.0  # 1 of 4
    assert report["pct_no_citation"] == 25.0  # 1 of 4
    # fabricated_citation_rate counts the MIXED answer too (it has an
    # invalid token alongside a valid one), so it's strictly >= pct_invalid_only
    assert report["fabricated_citation_rate"] == 50.0  # mixed + invalid_only, 2 of 4
    # tokens: clean=1 valid; mixed=1 valid+1 invalid; invalid_only=1 invalid; none=0
    # total=4 tokens, valid=2 -> precision 0.5
    assert report["citation_precision"] == 0.5


def test_compute_citation_report_empty_input():
    report = compute_citation_report([])
    assert report["n_answers"] == 0
    assert report["citation_precision"] is None
    assert report["pct_valid"] is None


def test_compute_citation_report_no_citation_tokens_anywhere_precision_is_none():
    records = [{"llm_answer": "nothing cited", "retrieved_context": RETRIEVED,
                "has_citation": False, "has_valid_citation": False}]
    report = compute_citation_report(records)
    assert report["citation_precision"] is None  # 0/0 must not be reported as 0.0
