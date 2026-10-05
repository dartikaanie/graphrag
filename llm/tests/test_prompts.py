"""Regression test for the [SO-1234]-style prompt-example fix (see
docs/README.md §8): every worked-example citation ID in llm/prompts.py's
instruction text must be a non-numeric placeholder ([SO-<id>]/
[SO-<id1>][SO-<id2>]), never a real-looking digit ID a model could copy
verbatim. The only digit-shaped [SO-<id>] tokens allowed anywhere in a
built prompt are the ones that come from the actual `retrieved` context
block (real question_id values), never from instructional/example text.

No LLM calls -- this only inspects the message strings these builders
return.
"""

import re

from llm.citations import CITATION_PATTERN
from llm.prompts import (
    PROMPT_VERSION,
    build_base_messages,
    build_graphrag_messages,
    build_lightrag_messages,
    build_rag_messages,
)

# Deliberately stricter than CITATION_PATTERN (which tolerates format
# variation) -- this scans for ANY digit-only [SO-<digits>]-shaped token,
# so it also catches a hypothetical future regression that CITATION_PATTERN
# itself wouldn't be sensitive to.
DIGIT_CITATION_RE = re.compile(r"\[SO[-: ]?(?:thread\s*)?\d+\]", re.IGNORECASE)

# The exact former example IDs -- asserting these literal numbers never
# appear anywhere in prompt text again, regardless of context.
FORMER_EXAMPLE_IDS = ("1234", "5678", "1234567", "2233445", "8899001")


def _all_text(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages)


def _retrieved_with_one_real_item(question_id: int) -> list[dict]:
    return [{
        "question_id": question_id,
        "answer_id": 42,
        "chunk_text": "Q: some real retrieved question\nA: some real retrieved answer body",
        "trust_weight": 0.5,
        "hop": 1,
        "source_stage": "graph_traversal",
    }]


BUILDERS_WITH_GROUNDING_FLAG = [
    ("build_graphrag_messages", build_graphrag_messages, "require_grounding"),
    ("build_lightrag_messages", build_lightrag_messages, "require_grounding"),
]


def test_condition_a_prompt_has_no_so_citation_text_at_all():
    """build_base_messages() (Condition A) never mentions SO-citations in
    any form -- it has no retrieval, so this fix must not touch it."""
    messages = build_base_messages("Some title", "Some body", "<python><django>")
    text = _all_text(messages)
    assert "[SO" not in text
    for former_id in FORMER_EXAMPLE_IDS:
        assert former_id not in text


def test_former_numeric_example_ids_are_gone_from_all_builders():
    for require_flag in (True, False):
        for name, builder, kwarg in BUILDERS_WITH_GROUNDING_FLAG:
            for retrieved in ([], _retrieved_with_one_real_item(999999)):
                messages = builder("Title", "Body", "<java>", retrieved, **{kwarg: require_flag})
                text = _all_text(messages)
                for former_id in FORMER_EXAMPLE_IDS:
                    assert former_id not in text, (
                        f"{name}({kwarg}={require_flag}, retrieved={'non-empty' if retrieved else 'empty'}) "
                        f"still contains former example id {former_id!r}"
                    )

    for require_citation in (True, False):
        for retrieved in ([], _retrieved_with_one_real_item(999999)):
            messages = build_rag_messages("Title", "Body", "<java>", retrieved, require_citation=require_citation)
            text = _all_text(messages)
            for former_id in FORMER_EXAMPLE_IDS:
                assert former_id not in text


def test_only_real_context_ids_appear_as_digit_citations_graphrag_and_lightrag():
    """With a single retrieved item whose real question_id is 999999, the
    ONLY digit-shaped [SO-<digits>] token anywhere in the prompt must be
    [SO-999999] -- proving the instructional/example text contributes zero
    digit citations of its own."""
    for name, builder, kwarg in BUILDERS_WITH_GROUNDING_FLAG:
        for require_flag in (True, False):
            messages = builder(
                "Title", "Body", "<java>", _retrieved_with_one_real_item(999999), **{kwarg: require_flag},
            )
            text = _all_text(messages)
            found_ids = {int(m) for m in CITATION_PATTERN.findall(text)}
            assert found_ids == {999999}, f"{name}({kwarg}={require_flag}): expected only {{999999}}, got {found_ids}"


def test_only_real_context_ids_appear_as_digit_citations_rag_b():
    messages = build_rag_messages(
        "Title", "Body", "<java>", _retrieved_with_one_real_item(999999), require_citation=True,
    )
    text = _all_text(messages)
    found_ids = {int(m) for m in CITATION_PATTERN.findall(text)}
    assert found_ids == {999999}


def test_zero_context_prompts_contain_no_digit_citation_at_all():
    """When retrieved=[] (no context found), there must be ZERO
    [SO-<digits>] tokens anywhere -- nothing for the model to copy."""
    for name, builder, kwarg in BUILDERS_WITH_GROUNDING_FLAG:
        for require_flag in (True, False):
            messages = builder("Title", "Body", "<java>", [], **{kwarg: require_flag})
            text = _all_text(messages)
            assert DIGIT_CITATION_RE.search(text) is None, f"{name}({kwarg}={require_flag}) empty-context prompt has a digit citation"

    messages = build_rag_messages("Title", "Body", "<java>", [], require_citation=True)
    assert DIGIT_CITATION_RE.search(_all_text(messages)) is None


def test_placeholder_ids_never_match_citation_pattern():
    """The non-numeric placeholders themselves must never be mistaken for a
    real citation by the exact regex used to validate model output --
    otherwise a judge/validator could be fooled by the instructions text
    itself (not applicable here since instructions aren't judged, but the
    property must hold regardless)."""
    assert CITATION_PATTERN.findall("[SO-<id>]") == []
    assert CITATION_PATTERN.findall("[SO-<id1>][SO-<id2>]") == []


def test_prompt_version_is_v3():
    assert PROMPT_VERSION == "v3"


def test_no_v3_prompt_string_contains_placeholder():
    """Prompt v3 (docs/NF2_ROOT_CAUSE_PLACEHOLDER_CITATIONS.md): the
    v2 worked-example sentence ("You can fix this... [SO-<id>]...") is
    the one piece of text models were copying verbatim when they had
    nothing confident to cite -- it's been removed from every builder,
    and NO built prompt string (any builder, any branch, with or without
    retrieved context) may contain the literal "<id>" placeholder token,
    or any numbered variant of it ("<id1>", "<id2>", ...), ever again.
    """
    placeholder_re = re.compile(r"<id\d*>", re.IGNORECASE)
    retrieved = _retrieved_with_one_real_item(999999)

    cases = [("A", build_base_messages("Title", "Body", "<java>"))]
    for name, builder, kwarg in BUILDERS_WITH_GROUNDING_FLAG:
        for require_flag in (True, False):
            for ctx in ([], retrieved):
                cases.append((
                    f"{name}({kwarg}={require_flag}, retrieved={'non-empty' if ctx else 'empty'})",
                    builder("Title", "Body", "<java>", ctx, **{kwarg: require_flag}),
                ))
    for require_citation in (True, False):
        for ctx in ([], retrieved):
            cases.append((
                f"build_rag_messages(require_citation={require_citation}, retrieved={'non-empty' if ctx else 'empty'})",
                build_rag_messages("Title", "Body", "<java>", ctx, require_citation=require_citation),
            ))

    for name, messages in cases:
        text = _all_text(messages)
        assert "<id>" not in text, f"{name} still contains the literal '<id>' placeholder"
        assert placeholder_re.search(text) is None, f"{name} still contains an '<idN>'-style placeholder"


def test_v3_grounded_branches_instruct_omitting_the_claim_not_just_the_citation():
    """Rule 1 (ground ONLY in context, never introduce unsupported claims)
    and the old rule 3 ("write that claim without any citation" when
    nothing supports it) directly contradicted each other -- rule 3 was
    rewritten so that when nothing supports a claim, the model must not
    make that claim at all (and say the sources don't cover it), not just
    drop the citation on an otherwise-still-made unsupported claim. This
    must hold in every GROUNDED branch (require_grounding/True), and in
    the matching system-message/reminder text too."""
    retrieved = _retrieved_with_one_real_item(999999)
    for name, builder, kwarg in BUILDERS_WITH_GROUNDING_FLAG:
        text = _all_text(builder("Title", "Body", "<java>", retrieved, **{kwarg: True}))
        assert "do not make that claim" in text, f"{name}({kwarg}=True) missing the rule-1/3-consistent instruction"
        assert "retrieved sources do not cover" in text or "sources don't cover" in text
        assert "write that claim without" not in text, (
            f"{name}({kwarg}=True) still contains the old rule 3 wording that contradicted rule 1"
        )


def test_v3_non_grounded_branches_still_allow_omitting_just_the_citation():
    """The non-grounded ablation variant (require_grounding=False / B's
    require_citation=True) never claimed "ground ONLY in context" in the
    first place, so writing an uncited claim there is NOT contradictory --
    this wording is deliberately UNCHANGED."""
    retrieved = _retrieved_with_one_real_item(999999)
    for name, builder, kwarg in BUILDERS_WITH_GROUNDING_FLAG:
        text = _all_text(builder("Title", "Body", "<java>", retrieved, **{kwarg: False}))
        assert "write that claim without" in text, f"{name}({kwarg}=False) missing the unchanged instruction"

    text = _all_text(build_rag_messages("Title", "Body", "<java>", retrieved, require_citation=True))
    assert "write that claim without" in text


def test_format_description_typo_is_fixed():
    """'no colon, no space, no the word thread' -> 'and not the word
    thread' -- was a grammar typo in the grounded branches' format
    description."""
    retrieved = _retrieved_with_one_real_item(999999)
    for name, builder, kwarg in BUILDERS_WITH_GROUNDING_FLAG:
        text = _all_text(builder("Title", "Body", "<java>", retrieved, **{kwarg: True}))
        assert "no the word" not in text, f"{name}({kwarg}=True) still has the typo"
        assert "and not the word 'thread'" in text
