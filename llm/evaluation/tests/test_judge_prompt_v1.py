"""Proves judge_prompt_v1.blind_candidate() strips the REAL [SO-<id>]
citation format (llm/prompts.py v2, llm/citations.py::CITATION_PATTERN)
AND the literal "[SO-<id>]"-style placeholder a model sometimes copies
verbatim from the prompt's own worked example (blind-v2), from real
B/C/D n=10 pilot answers, and leaves Condition A answers (which never
cite anything) unchanged. No LLM calls -- pure string processing against
existing result files on disk.
"""

import json
import re
from pathlib import Path

from judge_prompt_v1 import BLINDING_VERSION, CITATION_PATTERN, blind_candidate

REPO_ROOT = Path(__file__).resolve().parents[3]

RESULT_FILES = {
    "A": REPO_ROOT / "llm/a_pure_llm/results/condition_a_openai_gpt-4o-mini_n10_seed42.jsonl",
    "B": REPO_ROOT / "llm/b_rag/results/condition_b_openai_gpt-4o-mini_n10_seed42.jsonl",
    "C": REPO_ROOT / "llm/c_graphrag/results/condition_c_openai_gpt-4o-mini_n10_seed42_fw0-7-0-3.jsonl",
    "C_uniform": REPO_ROOT / "llm/c_graphrag/results/condition_c_openai_gpt-4o-mini_n10_seed42_uniform.jsonl",
    "D": REPO_ROOT / "llm/d_lightrag/results/condition_d_openai_gpt-4o-mini_n10_seed42_grounded.jsonl",
}


def _load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def test_citation_pattern_matches_real_so_format():
    """[SO-38118194] (real question_id, hyphen, no colon/space/word
    "thread") is the only digit format the generator actually produces --
    make sure the judge's own CITATION_PATTERN (adapted from llm/
    citations.py) still recognizes it after being moved into this
    module. (No capture group here, unlike llm/citations.py's own
    CITATION_PATTERN -- this one is sub()-only, for blinding, never used
    to extract/validate an id -- so assertions use search()/sub(), not
    findall()-with-captured-digits.)"""
    assert CITATION_PATTERN.search("solution here [SO-38118194].")
    assert CITATION_PATTERN.sub("", "fix this [SO-16915770][SO-29773039] please") == "fix this  please"


def test_citation_pattern_matches_literal_placeholder_family():
    """blind-v2: the literal "[SO-<id>]"/"[SO-<id1>]"/"[SO-<id2>]"
    placeholder text (copied verbatim from llm/prompts.py's own worked
    example by some models) must also be stripped -- it's a condition-
    revealing artifact in its own right (only ever seen in B/C/D output,
    never Condition A)."""
    assert CITATION_PATTERN.search("not grounded [SO-<id>] here")
    assert CITATION_PATTERN.sub("", "see [SO-<id1>] and [SO-<id2>] for details") == "see  and  for details"


def test_citation_pattern_does_not_match_unrelated_brackets():
    """Must require an actual separator/boundary right after "SO" --
    never match incidental brackets that merely start with those two
    letters, e.g. a "[Solution]"/"[SOLVED]" markdown heading."""
    assert not CITATION_PATTERN.search("See the [Solution] section below.")
    assert not CITATION_PATTERN.search("Marked as [SOLVED] by the OP.")


def test_blind_candidate_strips_real_placeholder_from_real_output():
    """Regression test for the actual leak found while smoke-testing
    export-human-csv: C-uniform's real n=10 output for QID 411756
    contains a literal [SO-<id>] placeholder alongside real digit
    citations -- blind_candidate() must strip ALL of them, leaving no
    "[SO-<id>]"/"[SO-<digits>]" token (and no bracket at all) behind."""
    records = _load_records(RESULT_FILES["C_uniform"])
    record = next(r for r in records if r["question_id"] == 411756)
    answer = record["llm_answer"]
    assert "[SO-<id>]" in answer  # the fixture's premise -- still true today

    blinded = blind_candidate(answer)
    assert "[SO-<id>]" not in blinded
    assert not CITATION_PATTERN.search(blinded)


def test_blinding_version_is_recorded():
    assert BLINDING_VERSION == "blind-v2"


def test_blind_candidate_never_strips_disclaimer_prose():
    """Only bracketed citation/placeholder TOKENS are stripped -- a
    disclaimer sentence like llm/prompts.py's empty-context instruction
    ("...this answer is NOT grounded in retrieved sources") is never
    bracketed, so it must survive blinding byte-for-byte, even right next
    to a citation token that DOES get stripped."""
    text = ("No relevant context was found, so this answer is NOT grounded in retrieved "
            "sources. Still, consider using [SO-<id>] as a reference if available.")
    blinded = blind_candidate(text)
    assert "NOT grounded in retrieved sources" in blinded
    assert "No relevant context was found" in blinded
    assert "[SO-<id>]" not in blinded


def test_blind_candidate_strips_real_citations_from_b_c_d():
    for condition in ("B", "C", "D"):
        records = _load_records(RESULT_FILES[condition])
        cited = [r for r in records if CITATION_PATTERN.search(r["llm_answer"])]
        assert cited, f"expected at least one real [SO-<id>] citation in Condition {condition}'s n=10 pilot file"

        for record in cited:
            answer = record["llm_answer"]
            blinded = blind_candidate(answer)
            assert not CITATION_PATTERN.search(blinded), (
                f"Condition {condition} QID {record['question_id']}: a [SO-<id>] citation survived blinding"
            )
            # Blinding only removes citation tokens (+ collapses the
            # resulting double-space) -- it must not also eat surrounding
            # prose. The de-citationed text, modulo whitespace, must still
            # be a substring-equivalent of the original with just the
            # citation tokens cut out.
            reconstructed = re.sub(r"[ \t]{2,}", " ", CITATION_PATTERN.sub("", answer)).strip()
            assert blinded == reconstructed


def test_blind_candidate_leaves_condition_a_answers_unchanged():
    """Condition A never retrieves context and never produces a [SO-<id>]
    citation (no retrieval, no citation instruction in its prompt) -- its
    answers must pass through blind_candidate() with no citation-looking
    substring removed. (Whitespace normalization can still apply -- the
    content itself must be untouched.)"""
    records = _load_records(RESULT_FILES["A"])
    assert records
    for record in records:
        answer = record["llm_answer"]
        assert not CITATION_PATTERN.search(answer), (
            f"Condition A QID {record['question_id']} unexpectedly contains a [SO-<id>]-shaped token "
            "-- re-check this fixture, the test's premise (A never cites) no longer holds"
        )
        blinded = blind_candidate(answer)
        normalized_original = re.sub(r"[ \t]{2,}", " ", answer).strip()
        assert blinded == normalized_original
