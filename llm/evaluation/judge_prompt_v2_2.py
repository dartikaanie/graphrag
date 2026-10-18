"""
judge_prompt_v2_2.py
--------------------
Judge-v2.2 (docs/DECISION_C_SCORING.md, Amendment 11): judge-v2.1 plus the
retrieval context, as ONE instrument for hallucination, faithfulness to the
context, citation support and approach match. judge_prompt_v2_1.py is
untouched; its judging standard, claim definition, verdicts, severity
rules and examples are reused verbatim (SYSTEM_PROMPT is built from it),
so the hallucination rule is the same instrument plus new fields.

Input (build_messages / prepare_judge_input):
  - question (title, tags, body), the accepted answer as the reference;
  - the candidate answer blinded with blind-v3 (judge_blinding.py);
  - RETRIEVED CONTEXT: for B/C/D the context items in retrieval order under
    neutral markers [K1]..[K5] (never the Stack Overflow id), each cut at
    CONTEXT_ITEM_MAX_CHARS; for A an explicit "no retrieved context";
  - CITATIONS: the candidate's statements that carried [SO-<id>] in the
    original answer, with the [Kn] items they cite (citation_map, built
    BEFORE blinding: SO question id -> every [Kn] of that question).

Output per claim (in addition to v2.1): supported_by_context (YES / NO /
N/A for A), cited_items, cited_item_supports (YES / NO, only for a claim
that carries a citation). Per answer: approach_match (SAME /
DIFFERENT_VALID / DIFFERENT_UNASSESSABLE).

Rules fixed in code (never the judge's own label):
  - hallucinated: >= 1 claim CONTRADICTED or FABRICATED (judge_binary,
    unchanged from v2.1);
  - faithfulness (B-D): claims with supported_by_context YES / all claims;
    None for A, for an ABSTAIN and for an answer without claims;
  - citation_support: cited claims with cited_item_supports YES / cited
    claims; None when no claim carries a citation;
  - approach_match never changes hallucinated.

All examples in SYSTEM_PROMPT are INVENTED -- none is taken from the test
sample (positions 1-384) or the dev set (385-434).
"""

import json
import re

import judge_prompt_v2_1 as v21
from judge_blinding import BLINDING_VERSION, blind_candidate_v3  # noqa: F401 -- blind-v3

PROMPT_VERSION = "judge-v2.2"
PARSE_VERSION = "pv2.2-1"

# temperature 0 as v2.1. max_tokens: v2.1's 1500 + room for the per-claim
# context/citation fields (about 15 extra tokens per claim, at most ~20 claims).
REQUEST_TEMPERATURE = 0.0
REQUEST_MAX_TOKENS = 2000

MAX_CHARS_EACH = 6000          # question body, reference, candidate: as v2.1
CONTEXT_ITEM_MAX_CHARS = 2500  # per [Kn] item; dev runs: max 2,366 chars, median 981 (no item cut)
MAX_CONTEXT_ITEMS = 5          # top_k of every official run
MAX_CITED_STATEMENTS = 12

CONTEXT_SUPPORT_VALUES = ("YES", "NO", "N/A")
CITED_SUPPORT_VALUES = ("YES", "NO")
APPROACH_VALUES = ("SAME", "DIFFERENT_VALID", "DIFFERENT_UNASSESSABLE")
NO_CONTEXT_TEXT = "No retrieved context: this answer was written without any retrieved context."
NO_CITATIONS_TEXT = "None: no statement of the candidate carried a citation."

_SO_TOKEN = re.compile(r"\[SO(?=[-:\s<\]])[^\]]{0,20}\]", re.IGNORECASE)
_SO_DIGITS = re.compile(r"\[SO[-: ]?(?:thread\s*)?(\d+)\]", re.IGNORECASE)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


# ---------------------------------------------------------------------------
# Prompt (v2.1 rules + context section + new output fields)
# ---------------------------------------------------------------------------

_CONTEXT_SECTION = """## Step 2b - Retrieved context and citations (for every claim)
The input contains a RETRIEVED CONTEXT section. For conditions with retrieval it lists context items [K1], [K2], ...; otherwise it says there is no retrieved context.
- supported_by_context: YES if a context item states the claim or directly implies it; NO if no context item does (including when the context contradicts it or does not address it). If there is no retrieved context, N/A for every claim.
- The context decides only supported_by_context. The verdict (SUPPORTED / CONTRADICTED / FABRICATED / UNVERIFIABLE) is still judged against the REFERENCE ANSWER and well-established fact, exactly as above: a claim the context supports can still be CONTRADICTED by the reference, and a claim the context does not mention can still be SUPPORTED by the reference.
- The CITATIONS section lists the statements of the candidate that cited context items in the original answer, with the cited items. For a claim that comes from such a statement, set cited_items to those items (e.g. ["K2"]) and cited_item_supports: YES if at least one cited item states or directly implies the claim, otherwise NO. A citation to a source that is not among the context items (shown as "not among the context items") never supports a claim. For every other claim, cited_items is [] and cited_item_supports is null.
- "[catatan: informasi tidak tersedia]" marks a place where the answer said that some information was not available. It is not a claim.

## Step 2c - Approach (only if answer_attempted is true)
approach_match compares the solution approach of the candidate with the reference:
- SAME: the same approach (the same fix, API, or technique), even if worded or detailed differently.
- DIFFERENT_VALID: a different approach that you are confident would also solve the problem.
- DIFFERENT_UNASSESSABLE: a different approach whose validity you cannot assess with confidence.
approach_match never makes an answer a hallucination: a different approach is not an error (see "Important").
If answer_attempted is false, approach_match is null.

"""

_OUTPUT_V22 = """## Output
Respond with JSON only, with exactly these keys in this order:
{
  "answer_attempted": true | false,
  "completeness": "FULL | PARTIAL | NONE",
  "approach_match": "SAME | DIFFERENT_VALID | DIFFERENT_UNASSESSABLE | null",
  "claims": [
    {"claim": "<short paraphrase that keeps its qualifiers>",
     "verdict": "SUPPORTED | CONTRADICTED | FABRICATED | UNVERIFIABLE",
     "severity": "CORE | MINOR | null",
     "reference_conflict": true | false,
     "supported_by_context": "YES | NO | N/A",
     "cited_items": ["K1", ...],
     "cited_item_supports": "YES | NO | null",
     "evidence": "<one short sentence>"}
  ],
  "reasoning": "<2-4 sentences applying the decision rules>",
  "label": "FAKTUAL | HALUSINASI_SEBAGIAN | HALUSINASI_PENUH | ABSTAIN"
}"""


def _build_system_prompt() -> str:
    base = v21.SYSTEM_PROMPT
    intro_old = ("Decide whether the CANDIDATE ANSWER contains hallucinations, judged against the REFERENCE ANSWER "
                 "(the accepted answer on Stack Overflow).")
    intro_new = (intro_old + " You also record, for each claim, whether the RETRIEVED CONTEXT given to the answer "
                 "supports it and whether the context items the answer cited support it, and whether the answer "
                 "takes the same approach as the reference.")
    step3 = "## Step 3 - Completeness (separate from hallucination)"
    output = base[base.index("## Output"):]
    for needle in (intro_old, step3, output):
        assert base.count(needle) == 1, needle[:40]  # v2.1 is frozen; fail loudly if it ever changes
    prompt = base.replace(intro_old, intro_new).replace(step3, _CONTEXT_SECTION + step3)
    return prompt.replace(output, _OUTPUT_V22)


SYSTEM_PROMPT = _build_system_prompt()

USER_TEMPLATE = """### QUESTION
Title: {title}
Tags: {tags}
Body:
{body}

### REFERENCE ANSWER (accepted on Stack Overflow)
{reference}

### RETRIEVED CONTEXT
{context}

### CITATIONS IN THE CANDIDATE ANSWER
{citations}

### CANDIDATE ANSWER
{candidate}"""


# ---------------------------------------------------------------------------
# Input preparation: context block, citation map (before blinding), blind-v3
# ---------------------------------------------------------------------------

def _cut(s, limit):
    s = s or ""
    return s if len(s) <= limit else s[:limit] + "\n[...truncated]"


def context_item_text(item: dict) -> str:
    return item.get("chunk_text") or item.get("text") or ""


def citation_map(context_items: list[dict] | None) -> dict[int, list[str]]:
    """{Stack Overflow question id: [Kn, ...]} for the shown context items (an id can appear more than once:
    several answers of one thread)."""
    out: dict[int, list[str]] = {}
    for i, item in enumerate((context_items or [])[:MAX_CONTEXT_ITEMS], start=1):
        qid = item.get("question_id")
        if qid is not None:
            out.setdefault(int(qid), []).append(f"K{i}")
    return out


def cited_statements(answer: str, cmap: dict[int, list[str]]) -> list[dict]:
    """Statements (sentences/lines) of the ORIGINAL answer that carry a citation token, with the [Kn] they cite.
    A cited id not among the context items is kept as an unmatched citation; a placeholder token such as
    [SO-<id>] has no id and is reported as unmatched too."""
    out = []
    for sentence in _SENTENCES.split(answer or ""):
        tokens = _SO_TOKEN.findall(sentence)
        if not tokens:
            continue
        ids = [int(x) for x in _SO_DIGITS.findall(sentence)]
        items = sorted({k for i in ids for k in cmap.get(i, [])}, key=lambda k: int(k[1:]))
        unmatched = [i for i in ids if i not in cmap] + ([None] * (len(tokens) - len(ids)))
        text = re.sub(r"[ \t]+([,.;:!?])", r"\1", blind_candidate_v3(sentence))
        if text:
            out.append({"statement": text, "cited_items": items, "n_unmatched": len(unmatched),
                        "cited_ids": ids})
    return out[:MAX_CITED_STATEMENTS]


def prepare_judge_input(candidate: str, context_items: list[dict] | None, has_context: bool) -> dict:
    cmap = citation_map(context_items) if has_context else {}
    if not has_context:
        context_block = NO_CONTEXT_TEXT
    elif not context_items:
        context_block = "No items were retrieved for this answer."
    else:
        context_block = "\n\n".join(f"[K{i}]\n{_cut(context_item_text(it), CONTEXT_ITEM_MAX_CHARS)}"
                                    for i, it in enumerate(context_items[:MAX_CONTEXT_ITEMS], start=1))
    statements = cited_statements(candidate, cmap) if has_context else []
    if statements:
        lines = []
        for n, s in enumerate(statements, start=1):
            cites = [f"[{k}]" for k in s["cited_items"]] + ["(a cited source not among the context items)"] * \
                (1 if s["n_unmatched"] else 0)
            lines.append(f'{n}. "{s["statement"]}" -> {", ".join(cites)}')
        citations_block = "\n".join(lines)
    else:
        citations_block = NO_CITATIONS_TEXT
    return {
        "has_context": has_context,
        "n_context_items": len((context_items or [])[:MAX_CONTEXT_ITEMS]) if has_context else 0,
        "context_block": context_block,
        "citations_block": citations_block,
        "citation_map": {str(k): v for k, v in cmap.items()},
        "cited_statements": statements,
        "offered_items": sorted({k for s in statements for k in s["cited_items"]}, key=lambda k: int(k[1:])),
        "candidate_blind": blind_candidate_v3(candidate),
    }


def build_messages(title, body, tags, reference, candidate, context_items=None, has_context=False,
                   prepared: dict | None = None):
    prepared = prepared or prepare_judge_input(candidate, context_items, has_context)
    user = USER_TEMPLATE.format(
        title=title, tags=tags, body=_cut(body, MAX_CHARS_EACH), reference=_cut(reference, MAX_CHARS_EACH),
        context=prepared["context_block"], citations=prepared["citations_block"],
        candidate=_cut(prepared["candidate_blind"], MAX_CHARS_EACH),
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Parsing + derived metrics
# ---------------------------------------------------------------------------

def _load_json(raw):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _upper_or_none(v):
    if v is None:
        return None
    s = str(v).strip().upper()
    return None if s in ("", "NULL", "NONE") else s


def faithfulness_of(record: dict) -> float | None:
    """Share of claims with supported_by_context YES. None for no context (A), ABSTAIN, unusable, no claims."""
    if not record.get("has_context") or record.get("parse_error") or record.get("call_failed") \
            or record.get("answer_attempted") is not True:
        return None
    claims = record.get("claims") or []
    if not claims:
        return None
    return round(sum(1 for c in claims if c.get("supported_by_context") == "YES") / len(claims), 4)


def citation_support_of(record: dict) -> float | None:
    """Share of cited claims with cited_item_supports YES. None when no claim carries a citation."""
    if record.get("parse_error") or record.get("call_failed") or record.get("answer_attempted") is not True:
        return None
    cited = [c for c in (record.get("claims") or []) if c.get("cited_items")]
    if not cited:
        return None
    return round(sum(1 for c in cited if c.get("cited_item_supports") == "YES") / len(cited), 4)


def parse_judgment_v2_2(raw: str, has_context: bool, offered_items=()) -> dict:
    """v2.1 parse (same verdict/severity rules, same failure cases), then the v2.2 fields.
      - supported_by_context: YES/NO when there is context; without context it is always N/A (a YES/NO is
        normalised to N/A and recorded). N/A or a missing value WITH context is a parse error.
      - cited_items: K labels; labels the CITATIONS section did not offer are dropped and recorded.
        cited_item_supports is required (YES/NO) when cited_items is non-empty, else null (normalised).
      - approach_match: required when answer_attempted is true, null otherwise (normalised)."""
    base = v21.parse_judgment_v2_1(raw)
    if base.get("parse_error"):
        return {**base, "parse_version": PARSE_VERSION, "prompt_version": PROMPT_VERSION}
    data = _load_json(raw)
    normalized = list(base.get("normalized_fields", []))
    fail = {"parse_error": True, "raw": raw, "parse_version": PARSE_VERSION, "prompt_version": PROMPT_VERSION}

    approach = _upper_or_none(data.get("approach_match"))
    if base["answer_attempted"]:
        if approach not in APPROACH_VALUES:
            return fail
    elif approach is not None:
        approach = None
        normalized.append("approach_match_on_abstain")

    offered = set(offered_items)
    claims = []
    for parsed, rawc in zip(base["claims"], data["claims"]):
        sbc = _upper_or_none(rawc.get("supported_by_context"))
        if not has_context:
            if sbc != "N/A":
                normalized.append("supported_by_context_without_context")
            sbc = "N/A"
        elif sbc not in ("YES", "NO"):
            return fail
        items_raw = rawc.get("cited_items") or []
        if not isinstance(items_raw, list):
            return fail
        items = []
        for k in items_raw:
            k = str(k).strip().strip("[]").upper()
            if k in offered:
                items.append(k)
            else:
                normalized.append("cited_item_not_offered")
        cis = _upper_or_none(rawc.get("cited_item_supports"))
        if items:
            if cis not in CITED_SUPPORT_VALUES:
                return fail
        elif cis is not None:
            cis = None
            normalized.append("cited_item_supports_without_citation")
        claims.append({**parsed, "supported_by_context": sbc, "cited_items": items, "cited_item_supports": cis})

    rec = {**base, "claims": claims, "approach_match": approach, "has_context": has_context,
           "normalized_fields": normalized, "prompt_version": PROMPT_VERSION, "parse_version": PARSE_VERSION}
    rec["n_cited_claims"] = sum(1 for c in claims if c["cited_items"])
    rec["faithfulness"] = faithfulness_of(rec)
    rec["citation_support"] = citation_support_of(rec)
    return rec
