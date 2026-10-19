"""
judge_prompt_v2_3.py
--------------------
Judge-v2.3: revision 1 after judge-v2.2 (docs/JUDGE_REVISION_LOG.md, iteration 2;
docs/DECISION_C_SCORING.md, Amendments 11-13). judge_prompt_v2_2.py is untouched;
this prompt is built from its SYSTEM_PROMPT by exact text replacements (each
asserted to occur once), so every rule not listed below is the same instrument.
Input template, context block, citation map and blinding (blind-v3) are v2.2's.

Changes (targets: buckets of the iteration-1 miss diagnosis, (a) not extracted,
(b) judged UNVERIFIABLE, (c) judged SUPPORTED; numbers as proposed in the
diagnosis report):
  #1 explicit conflict with the reference -> CONTRADICTED, not UNVERIFIABLE (b, c)
  #2 always together with #1: a different version/library/approach is not a
     contradiction unless the reference or well-established fact says the
     candidate's choice fails (guards against the iteration-1 false alarms)
  #3 identifiers in code/commands are separate claims; a confident near-miss of
     a real name is FABRICATED (a, b)
  #4 within the 8-claim limit, specific identifiers/commands/syntax/main fix
     come before general advice (a)
  #5 SUPPORTED / CONTRADICTED evidence must quote the reference ([R] "...");
     restating the candidate is never support (c)
Request: JSON mode (response_format {"type": "json_object"}) for all judges.
Parser (pv2.3-1): v2.2's rules, plus supported_by_context "N/A" on an item WITH
context is normalised to null and recorded as
"supported_by_context_na_with_context" (iteration 1: 7 of 8 parse failures);
faithfulness counts only claims with a non-null supported_by_context.

All examples are INVENTED (fictional tool "mytool", fictional libraries
"libfoo"/"libbar"); none is taken from the test sample or the dev set.
"""

import json

import judge_prompt_v2_2 as v22
from judge_prompt_v2_2 import (BLINDING_VERSION, CONTEXT_ITEM_MAX_CHARS, MAX_CHARS_EACH,  # noqa: F401
                               USER_TEMPLATE, _load_json, candidate_with_citation_labels, prepare_judge_input)

PROMPT_VERSION = "judge-v2.3"
PARSE_VERSION = "pv2.3-1"

REQUEST_TEMPERATURE = 0.0
REQUEST_MAX_TOKENS = 2000                                   # unchanged from v2.2
REQUEST_RESPONSE_FORMAT = {"type": "json_object"}           # JSON mode, all three judges

NA_WITH_CONTEXT = "supported_by_context_na_with_context"
_NA_STRINGS = ("N/A", "NA", "N.A.")

# ---------------------------------------------------------------------------
# Prompt: v2.2 + changes #1-#5 (exact replacements)
# ---------------------------------------------------------------------------

_REPLACEMENTS = [
    # #1 + #2 (always together): judging standard
    ("- Outdated reference: if a claim conflicts with the reference",
     "- Explicit conflict with the reference: when the reference explicitly states that the candidate's specific "
     "claim is false, impossible, or not supported (e.g. \"this is not possible\", \"only X is supported\", \"use A, "
     "not B\" for the same purpose), or the candidate states the opposite of an explicit statement in the reference, "
     "the claim is CONTRADICTED, not UNVERIFIABLE. Covering the same topic is not enough: the reference must make a "
     "statement that the claim conflicts with.\n"
     "- A different choice is not a contradiction: recommending a different version, library, package, setting, or "
     "approach than the reference is NOT CONTRADICTED merely because it differs from the reference. It is "
     "CONTRADICTED only if the reference (or well-established fact) says that the candidate's specific choice fails "
     "or does not exist; otherwise it is UNVERIFIABLE.\n"
     "- Outdated reference: if a claim conflicts with the reference"),
    # #3: identifiers in code and commands
    ("If it does not exist, that claim is FABRICATED. (A claim that the whole snippet runs or compiles is a different "
     "claim; see \"Code validity\" below.)",
     "If it does not exist, that claim is FABRICATED. (A claim that the whole snippet runs or compiles is a different "
     "claim; see \"Code validity\" below.)\n"
     "List each identifier the answer relies on as its own claim: every type, namespace, class, method, function, "
     "parameter, enum value, event or constant name, command, command-line flag, and SQL or configuration keyword. "
     "Check the exact name as written: a near-miss of a real name (a wrong namespace or class, a changed word order, "
     "a keyword or flag the tool does not accept) means the thing does not exist as written: FABRICATED (or "
     "CONTRADICTED for syntax the tool rejects). Mark it so only if you are confident of the correct name; otherwise "
     "the claim is UNVERIFIABLE."),
    # #5: evidence must quote the reference
    ("Code validity: a claim that a whole snippet is valid",
     "Evidence: a SUPPORTED or CONTRADICTED verdict that rests on the reference must quote the reference sentence it "
     "rests on, prefixed with [R] (e.g. [R] \"use fromisoformat\"). If you cannot quote such a sentence, the verdict "
     "is not decided by the reference: use UNVERIFIABLE, unless it rests on well-established fact (say which) or on "
     "the outdated-reference rule. Evidence that only restates the candidate (\"the candidate states ...\") is never "
     "support.\n\n"
     "Code validity: a claim that a whole snippet is valid"),
    # #4: extraction priority within the 8-claim limit
    ("The limit of 8 applies only to SUPPORTED and UNVERIFIABLE claims: list at most 8 of those, prioritizing the "
     "ones most important to the solution.",
     "The limit of 8 applies only to SUPPORTED and UNVERIFIABLE claims: list at most 8 of those, chosen in this "
     "order: (1) identifiers, commands, syntax, and configuration values the solution relies on; (2) the main fix and "
     "the stated cause of the problem; (3) other specific technical statements. General advice comes last, or is "
     "left out."),
    # examples for #1, #2, #3 (invented)
    ("\n\n## Output",
     "\n(k) A different question about the fictional tool mytool 2.x. Reference: \"Only the values utc and local are "
     "supported by the --tz option.\" Candidate: \"Run mytool --tz=auto.\"\n"
     "    -> the reference explicitly limits the supported values and auto is not one of them: CONTRADICTED, it is the "
     "recommended solution: CORE, evidence [R] \"Only the values utc and local are supported\". label=HALUSINASI_PENUH\n"
     "(l) A different question about fictional libraries. Reference: \"Downgrade libfoo to 1.0.1 and pin libbar to "
     "11.0.1.\" Candidate: \"Upgrade libfoo to 1.1.0.\"\n"
     "    -> a different version than the reference; the reference does not say that 1.1.0 fails: UNVERIFIABLE, "
     "severity null (not CONTRADICTED). label=FAKTUAL\n"
     "(m) Same question as (a)-(j). Candidate: \"Call datetime.from_iso_format(s).\"\n"
     "    -> a near-miss of the real method fromisoformat; from_iso_format does not exist: FABRICATED, CORE. "
     "label=HALUSINASI_PENUH\n\n## Output"),
    # #5: output schema
    ('"evidence": "<one short sentence>"}',
     '"evidence": "<SUPPORTED/CONTRADICTED from the reference: [R] \\"quote\\"; otherwise one short sentence>"}'),
]


def _build_system_prompt() -> str:
    prompt = v22.SYSTEM_PROMPT
    for old, new in _REPLACEMENTS:
        assert prompt.count(old) == 1, old[:50]  # v2.2 is frozen; fail loudly if it ever changes
        prompt = prompt.replace(old, new)
    return prompt


SYSTEM_PROMPT = _build_system_prompt()


def build_messages(title, body, tags, reference, candidate, context_items=None, has_context=False,
                   prepared: dict | None = None):
    """v2.2's input template and preparation (blind-v3, [K1]..[K5], citations), with the v2.3 system prompt."""
    messages = v22.build_messages(title, body, tags, reference, candidate, context_items, has_context, prepared)
    return [{"role": "system", "content": SYSTEM_PROMPT}, messages[1]]


# ---------------------------------------------------------------------------
# Parsing: v2.2 + "N/A" -> null for supported_by_context on items with context
# ---------------------------------------------------------------------------

def _faithfulness(rec: dict) -> float | None:
    if not rec.get("has_context") or rec.get("parse_error") or rec.get("answer_attempted") is not True:
        return None
    rated = [c for c in rec.get("claims") or [] if c.get("supported_by_context") is not None]
    if not rated:
        return None
    return round(sum(1 for c in rated if c["supported_by_context"] == "YES") / len(rated), 4)


def parse_judgment_v2_3(raw: str, has_context: bool, offered_items=()) -> dict:
    """parse_judgment_v2_2, except that supported_by_context "N/A" on an item WITH retrieved context is set to
    null (never imputed) and recorded in normalized_fields as "supported_by_context_na_with_context"; faithfulness
    then counts only claims with a non-null value. Every other rule and failure case is v2.2's."""
    data = _load_json(raw)
    na_idx = []
    if has_context and isinstance(data, dict) and isinstance(data.get("claims"), list):
        for i, c in enumerate(data["claims"]):
            if isinstance(c, dict) and str(c.get("supported_by_context") or "").strip().upper() in _NA_STRINGS:
                c["supported_by_context"] = "NO"  # placeholder so the v2.2 rules can run; reset to null below
                na_idx.append(i)
    rec = v22.parse_judgment_v2_2(json.dumps(data) if na_idx else raw, has_context, offered_items)
    rec = {**rec, "prompt_version": PROMPT_VERSION, "parse_version": PARSE_VERSION}
    if rec.get("parse_error"):
        return {**rec, "raw": raw}
    for i in na_idx:
        rec["claims"][i] = {**rec["claims"][i], "supported_by_context": None}
    rec["normalized_fields"] = list(rec.get("normalized_fields", [])) + [NA_WITH_CONTEXT] * len(na_idx)
    rec["faithfulness"] = _faithfulness(rec)
    return rec
