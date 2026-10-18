"""
judge_blinding.py
-----------------
blind-v3 (docs/DECISION_C_SCORING.md, Amendment 11): the candidate text the
judge-v2.2 and the annotation sheets see.

blind-v3 = blind-v2 (judge_prompt_v1.blind_candidate: citation markers such
as [SO-12345] and a trailing "Sources:" block removed) + normalisation of
META PHRASES -- phrases about "the retrieved sources" / "the provided
context" that only a RAG answer (B, C, D) writes and that would tell the
judge or the annotator the answer is not from condition A.

Only the meta phrase is touched, never the rest of its sentence, so a
technical claim in the same sentence stays visible to the judge:
  - a statement that the sources do not cover something becomes the
    neutral marker NEUTRAL_MARKER ("The retrieved sources do not cover X"
    -> "[catatan: informasi tidak tersedia] X");
  - an attribution to the sources is removed ("According to the retrieved
    sources, X" -> "X").
Phrases about the asker's own material ("the code snippet you provided")
are not meta phrases and are never touched.

The pattern list was built from the existing dev outputs (A, B, C, D runs on
positions 385-434 and the calibration runs); `meta_phrase_counts` reproduces
the counts reported in reports/p1_judge_annotation.md. judge_prompt_v1 and
blind-v2 are unchanged: judge-v1/v2/v2.1 outputs stay reproducible.
"""

import re
from collections import Counter

from judge_prompt_v1 import blind_candidate as blind_candidate_v2

BLINDING_VERSION = "blind-v3"
NEUTRAL_MARKER = "[catatan: informasi tidak tersedia]"

# what a RAG answer calls its context
_SRC = (r"(?:(?:the|your)\s+)?(?:retrieved|provided|given|available|supplied)\s+"
        r"(?:sources?|contexts?|documents?|excerpts?|snippets|search\s+results|information|materials?)"
        r"|the\s+(?:sources?|contexts?|excerpts?|search\s+results)(?:\s+(?:you\s+)?(?:provided|given|retrieved|supplied))?")
# "in the context of X" is ordinary prose, not a meta phrase
_NOT_OF = r"(?!\s+of\b)"
_LEAD = r"(?:(?:unfortunately|however|therefore|in\s+summary|that\s+said|note\s+that),?\s+|(?:since|while|as|although)\s+)?"
_NEG_VERB = (r"(?:,\s*however,)?\s*(?:do|does|did)\s*(?:not|n't)\s+(?:\w+ly\s+)?"
             r"(?:cover|provide|address|mention|include|contain|discuss|specify|offer|detail|describe|explain|give|"
             r"confirm|answer|support|state|have)")
_SAY_VERB = (r"(?:(?:do|does)\s+)?(?:suggest|indicate|mention|state|recommend|support|show|confirm|note|provide|cover|"
             r"describe|explain|highlight|discuss)(?:e?s)?")

# (name, compiled pattern, replacement) -- applied in this order
META_PATTERNS = [
    # "The retrieved sources do not cover", "Unfortunately, the context provided does not specifically address"
    ("sources_do_not_cover", re.compile(rf"\b{_LEAD}(?:{_SRC}){_NOT_OF}\s*{_NEG_VERB}\b", re.IGNORECASE),
     NEUTRAL_MARKER),
    # "There is no information in the retrieved sources about"
    ("no_information_in_sources",
     re.compile(rf"\bthere\s+(?:is|are)\s+no\s+(?:\w+\s+){{0,2}}(?:in|from|within)\s+(?:{_SRC}){_NOT_OF}"
                rf"(?:\s+(?:about|on|regarding))?\b", re.IGNORECASE), NEUTRAL_MARKER),
    # "... isn't directly covered in the sources" -> "... isn't directly covered"
    ("not_covered_in_sources",
     re.compile(rf"(\b(?:is|are|was|were)\s*(?:not|n't)\s+(?:\w+ly\s+)?(?:covered|mentioned|addressed|discussed|"
                rf"described|specified|provided|supported|confirmed))\s+(?:in|by)\s+(?:{_SRC}){_NOT_OF}\b",
                re.IGNORECASE), r"\1"),
    # "According to the retrieved sources," / "Based on the provided context," / "As mentioned in the sources,"
    ("attribution_prefix",
     re.compile(rf"\b(?:according\s+to|based\s+on|as\s+(?:mentioned|noted|shown|described|stated|indicated|suggested)"
                rf"\s+(?:in|by)|from|per|in)\s+(?:{_SRC}){_NOT_OF},?\s*", re.IGNORECASE), ""),
    # "... might be needed beyond the context provided" -> "... might be needed"
    ("beyond_sources", re.compile(rf"\s+beyond\s+(?:{_SRC}){_NOT_OF}", re.IGNORECASE), ""),
    # "The retrieved sources (explicitly) confirm that" / "The retrieved context provides"
    ("sources_say",
     re.compile(rf"\b(?:{_SRC}){_NOT_OF}\s+(?:\w+ly\s+)?{_SAY_VERB}(?:\s+that)?:?\s*", re.IGNORECASE), ""),
]


def normalize_meta(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(normalised text, [(pattern name, matched text), ...])."""
    hits: list[tuple[str, str]] = []
    for name, pattern, repl in META_PATTERNS:
        def _sub(m, name=name, repl=repl):
            hits.append((name, m.group(0)))
            return m.expand(repl) if "\\" in repl else repl
        text = pattern.sub(_sub, text)
    return text, hits


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r",([.;:!?])", r"\1", text)  # "works, based on the sources." -> "works,." -> "works."
    text = re.sub(r"(^|[.!?]\s+|\n)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)  # sentence start
    return text.strip()


def blind_candidate_v3(answer: str) -> str:
    """blind-v2 (citation markers + Sources block removed), then meta phrases normalised."""
    text = blind_candidate_v2(answer or "")
    text, hits = normalize_meta(text)
    return _tidy(text) if hits else text


def meta_phrase_counts(answers) -> Counter:
    """Counter of normalised matched meta phrases (lower case, citation markers removed) over `answers`."""
    counts: Counter = Counter()
    for a in answers:
        _, hits = normalize_meta(blind_candidate_v2(a or ""))
        for name, matched in hits:
            counts[(name, " ".join(matched.lower().split()))] += 1
    return counts
