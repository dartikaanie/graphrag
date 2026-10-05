"""llm/citations.py
=====================================
[SO-<id>] citation extraction/validation -- moved here from c_graphrag.py so
Condition B can reuse the EXACT same validation logic when its optional
`--require-citation` mode is on (see build_rag_messages() in prompts.py and
b_condition_b_rag.py). Having one shared implementation, not two copies, is
what makes a B-vs-C citation-compliance comparison actually apples-to-apples.
"""

import re

CITATION_PATTERN = re.compile(r"\[SO[-: ]?(?:thread\s*)?(\d+)\]", re.IGNORECASE)

# Broader than CITATION_PATTERN: matches ANY "[SO...]"-shaped bracket token,
# digit citation OR non-digit placeholder alike -- e.g. the literal
# "[SO-<id>]"/"[SO-<id1>]" text a model sometimes copies verbatim from
# prompts.py's own citation-format description (see docs/README.md §8 and
# the v3 prompt fix) instead of a real id. A placeholder like that can
# never be a digit, so CITATION_PATTERN alone never sees it at all --
# which used to make a placeholder-only answer indistinguishable from an
# answer with literally no citation attempt. Used ONLY to compute
# has_citation (see below); never contributes to cited_ids/valid_ids,
# since a non-digit token obviously can't be a real question_id.
# Lookahead requires an actual separator ([-:\s<\]]) right after "SO" so
# this never matches an unrelated bracket that happens to start with
# those two letters, e.g. "[Solution]"/"[SOLVED]".
CITATION_TOKEN_PATTERN = re.compile(r"\[SO(?=[-:\s<\]])[^\]]{0,20}\]", re.IGNORECASE)


def extract_citations(llm_answer: str, retrieved: list) -> tuple[bool, list, bool, list]:
    """Return (has_citation, cited_ids, has_valid_citation, valid_ids).

    has_citation: TRUE for ANY "[SO...]"-shaped bracket token, digit
    citation OR non-digit placeholder alike (CITATION_TOKEN_PATTERN) --
    toleran ke variasi format kecil ([SO-1234], [SO:1234], [SO 1234],
    [SO thread 1234]) DAN ke placeholder literal ([SO-<id>] dkk) supaya
    kedua kasus terdeteksi sbg "mencoba mengutip", bukan otomatis
    dianggap "tidak ada kutipan sama sekali" -- sebuah placeholder yang
    tidak valid (lihat has_valid_citation) tetap sebuah PERCOBAAN
    kutipan, berbeda dari benar-benar tidak menyebut sitasi apa pun.
    cited_ids tetap HANYA id digit nyata (CITATION_PATTERN) -- placeholder
    tidak punya id utk dilaporkan.

    has_valid_citation/valid_ids: versi ketat -- id yang dikutip di-cross-
    check terhadap question_id di retrieved_context yang BENAR diberikan ke
    model. Ditambahkan setelah ditemukan kasus model menghasilkan
    citation-like token dgn ID yang di-hallucinate (tidak cocok konteks asli
    sama sekali, mis. [SO:4329876] padahal konteks yg diberikan SO-5928724)
    -- format benar TIDAK CUKUP, ID-nya juga harus benar-benar ada di
    konteks yang diberikan. Sebuah placeholder literal ([SO-<id>]) SELALU
    dihitung TIDAK VALID di sini (bukan NOL/tidak terdeteksi) karena
    kontennya memang bukan id apa pun yang ada di konteks -- has_citation
    True + has_valid_citation False utk kasus ini artinya "mencoba
    mengutip, TAPI tidak valid", bukan "tidak mencoba mengutip".
    """
    ids = sorted(set(int(m) for m in CITATION_PATTERN.findall(llm_answer)))
    context_ids = {r.get("question_id") for r in (retrieved or []) if r.get("question_id") is not None}
    valid_ids = sorted(i for i in ids if i in context_ids)
    has_citation = bool(CITATION_TOKEN_PATTERN.search(llm_answer))
    return (has_citation, ids, len(valid_ids) > 0, valid_ids)


# ---------------------------------------------------------------------------
# Token-level breakdown + run-level reporting -- ADDITIVE, read-only
# analysis on top of extract_citations(). Does NOT change has_citation/
# has_valid_citation/NF2's existing meaning or any stored field -- these
# are new, separate numbers for a finer-grained view of WHY an answer
# fails NF2 (no attempt vs. attempted-but-invalid vs. mixed valid+invalid),
# per docs/NF2_ROOT_CAUSE_PLACEHOLDER_CITATIONS.md.
# ---------------------------------------------------------------------------

def citation_token_breakdown(llm_answer: str, retrieved: list) -> dict:
    """Per-answer TOKEN-level counts (every bracket occurrence, including
    repeats -- unlike cited_ids/valid_ids, which are de-duplicated sets of
    ids). Needed for citation precision (valid tokens / all tokens): an
    answer can cite the same valid id three times and a placeholder once,
    which id-set membership alone can't quantify.

    Return dict: n_tokens (every [SO...]-shaped bracket, digit or
    placeholder), n_valid_tokens (digit tokens whose id IS in the given
    context), n_invalid_tokens (everything else: placeholders, malformed
    digit tokens, or a well-formed digit id NOT in context -- a
    hallucinated-looking citation), has_any_invalid_token.
    """
    context_ids = {r.get("question_id") for r in (retrieved or []) if r.get("question_id") is not None}
    tokens = CITATION_TOKEN_PATTERN.findall(llm_answer)
    n_valid = 0
    for token in tokens:
        m = CITATION_PATTERN.fullmatch(token)
        if m and int(m.group(1)) in context_ids:
            n_valid += 1
    n_invalid = len(tokens) - n_valid
    return {
        "n_tokens": len(tokens),
        "n_valid_tokens": n_valid,
        "n_invalid_tokens": n_invalid,
        "has_any_invalid_token": n_invalid > 0,
    }


def classify_citation_outcome(has_citation: bool, has_valid_citation: bool) -> str:
    """3-way ANSWER-level split (not token-level) requested alongside NF2:
    "valid" (has_valid_citation=True -- NF2 pass, may still ALSO contain
    an invalid token alongside; see citation_token_breakdown/
    fabricated_citation_rate for that), "no_citation" (has_citation=False
    -- no attempt at all), "invalid_only" (has_citation=True AND
    has_valid_citation=False -- attempted, but every token is invalid).
    These three are mutually exclusive and exhaustive because
    has_valid_citation=True always implies has_citation=True (a valid
    digit citation is itself a "[SO...]"-shaped token)."""
    if has_valid_citation:
        return "valid"
    if has_citation:
        return "invalid_only"
    return "no_citation"


def compute_citation_report(records: list[dict]) -> dict:
    """Run-level aggregate over a list of already-generated answer
    records, each expected to have 'llm_answer', 'retrieved_context',
    'has_citation', 'has_valid_citation' (exactly the fields already
    written by every condition's result .jsonl -- no new fields required
    upstream). Read-only: never mutates the input, never re-derives
    has_citation/has_valid_citation (uses them as already stored, so this
    reports on whatever extract_citations() version actually produced
    that file).

    Returns:
      n_answers, pct_valid, pct_no_citation, pct_invalid_only (the 3-way
        split, percentages of n_answers, summing to 100),
      fabricated_citation_rate (% of answers with >=1 invalid TOKEN,
        including ones that ALSO have a valid citation elsewhere -- a
        strict superset of pct_invalid_only, since that one excludes
        answers with a valid citation),
      citation_precision (sum of valid tokens / sum of all tokens across
        the whole run -- a run-level micro-average, not a mean of per-
        answer ratios, so a handful of heavily-cited answers don't get
        diluted by many short, token-free ones; None if the run has zero
        citation tokens anywhere).
    """
    n = len(records)
    if n == 0:
        return {
            "n_answers": 0, "pct_valid": None, "pct_no_citation": None, "pct_invalid_only": None,
            "fabricated_citation_rate": None, "citation_precision": None,
        }

    n_valid = n_no_citation = n_invalid_only = n_fabricated = 0
    total_tokens = total_valid_tokens = 0

    for record in records:
        has_citation = bool(record.get("has_citation"))
        has_valid_citation = bool(record.get("has_valid_citation"))
        outcome = classify_citation_outcome(has_citation, has_valid_citation)
        if outcome == "valid":
            n_valid += 1
        elif outcome == "invalid_only":
            n_invalid_only += 1
        else:
            n_no_citation += 1

        breakdown = citation_token_breakdown(record.get("llm_answer", "") or "", record.get("retrieved_context"))
        total_tokens += breakdown["n_tokens"]
        total_valid_tokens += breakdown["n_valid_tokens"]
        if breakdown["has_any_invalid_token"]:
            n_fabricated += 1

    return {
        "n_answers": n,
        "pct_valid": round(n_valid / n * 100, 1),
        "pct_no_citation": round(n_no_citation / n * 100, 1),
        "pct_invalid_only": round(n_invalid_only / n * 100, 1),
        "fabricated_citation_rate": round(n_fabricated / n * 100, 1),
        "citation_precision": round(total_valid_tokens / total_tokens, 3) if total_tokens else None,
    }
