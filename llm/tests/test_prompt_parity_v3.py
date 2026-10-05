"""Prompt-parity v3 (docs/PROMPT_PARITY_V3.md): given the SAME retrieved
context items, Condition B (require_citation=True), C, and D must produce
BYTE-IDENTICAL prompts -- for the grounded variant (require_grounding=True)
and, separately, for the non-grounded variant (require_grounding=False).
Retrieval (which items got retrieved) is the only thing allowed to differ
between the three conditions; the prompt TEXT around it must not.

Condition A (build_base_messages) is untouched by this parity requirement
-- it never mentions retrieval/citation at all, so there is nothing to
compare it against.

No LLM calls -- this only inspects the message strings these builders
return.
"""

from llm.prompts import build_graphrag_messages, build_lightrag_messages, build_rag_messages

RETRIEVED_SETS = [
    [],  # empty-context branch
    [{"question_id": 38118194, "chunk_text": "Q: some real retrieved question\nA: some real retrieved answer body"}],
    [
        {"question_id": 111, "chunk_text": "first context item"},
        {"question_id": 222, "chunk_text": "second context item"},
        {"question_id": 333, "chunk_text": "third context item"},
    ],
]


def _build_all(retrieved, require_grounding):
    b = build_rag_messages("Title", "Body", "<python><django>", retrieved,
                            require_citation=True, require_grounding=require_grounding)
    c = build_graphrag_messages("Title", "Body", "<python><django>", retrieved,
                                 require_grounding=require_grounding)
    d = build_lightrag_messages("Title", "Body", "<python><django>", retrieved,
                                 require_grounding=require_grounding)
    return b, c, d


def test_grounded_prompts_are_byte_identical_across_b_c_d():
    for retrieved in RETRIEVED_SETS:
        b, c, d = _build_all(retrieved, require_grounding=True)
        assert b == c, f"B != C (grounded) for retrieved={retrieved!r}"
        assert c == d, f"C != D (grounded) for retrieved={retrieved!r}"


def test_non_grounded_prompts_are_byte_identical_across_b_c_d():
    for retrieved in RETRIEVED_SETS:
        b, c, d = _build_all(retrieved, require_grounding=False)
        assert b == c, f"B != C (non-grounded) for retrieved={retrieved!r}"
        assert c == d, f"C != D (non-grounded) for retrieved={retrieved!r}"


def test_grounded_and_non_grounded_are_NOT_accidentally_identical_to_each_other():
    """Sanity check on the test itself: grounded vs non-grounded for the
    SAME condition must still differ (otherwise the two tests above would
    trivially pass by both branches collapsing to the same text)."""
    retrieved = RETRIEVED_SETS[1]
    grounded, _, _ = _build_all(retrieved, require_grounding=True)
    non_grounded, _, _ = _build_all(retrieved, require_grounding=False)
    assert grounded != non_grounded


def test_retrieval_metadata_fields_are_ignored_by_all_three_builders():
    """trust_weight/hop (C's historical metadata) and source_stage (D's)
    must have ZERO effect on the prompt text now that metadata is
    stripped -- proves the parity holds even when callers still pass
    this extra, condition-specific metadata through (which they do, since
    it's still stored in the result record)."""
    bare = [{"question_id": 1, "chunk_text": "same text"}]
    with_c_metadata = [{"question_id": 1, "chunk_text": "same text", "trust_weight": 0.9, "hop": 2}]
    with_d_metadata = [{"question_id": 1, "chunk_text": "same text", "source_stage": "high_level"}]

    for require_grounding in (True, False):
        b_bare, c_bare, d_bare = _build_all(bare, require_grounding)
        _, c_meta, _ = _build_all(with_c_metadata, require_grounding)
        _, _, d_meta = _build_all(with_d_metadata, require_grounding)
        assert c_bare == c_meta, "C's prompt changed when trust_weight/hop were added to retrieved items"
        assert d_bare == d_meta, "D's prompt changed when source_stage was added to retrieved items"


def test_no_retrieval_metadata_leaks_into_prompt_text():
    retrieved_c = [{"question_id": 1, "chunk_text": "x", "trust_weight": 0.75, "hop": 3}]
    retrieved_d = [{"question_id": 1, "chunk_text": "x", "source_stage": "low_level"}]
    for require_grounding in (True, False):
        c_text = "\n".join(m["content"] for m in build_graphrag_messages(
            "T", "B", "<p>", retrieved_c, require_grounding=require_grounding))
        d_text = "\n".join(m["content"] for m in build_lightrag_messages(
            "T", "B", "<p>", retrieved_d, require_grounding=require_grounding))
        assert "trust=" not in c_text
        assert "-hop" not in c_text
        assert "low_level" not in d_text
        assert "high_level" not in d_text
