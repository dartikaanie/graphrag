"""
judge_prompt_v2.py
------------------
Judge-v2: a fix to judge-v1's prompt (judge_prompt_v1.py, FROZEN and
untouched) addressing systematic disagreement patterns found in the
pilot (weighted κ = 0.33 between primary and secondary) -- see
docs/JUDGE_V2_CHANGES.md for the full rationale with pilot question
IDs. In short, v1's prompt had no explicit "did the candidate even
attempt an answer" step, so an honest "the sources don't cover this"
non-answer could be graded as a hallucination via the "addresses a
different problem" clause, and statements ABOUT the retrieved sources
(which the judge never sees -- blinded) could be picked up as technical
claims.

v2 adds an explicit Step 1 (answer_attempted) and Step 3 (completeness,
deliberately never a hallucination signal), excludes meta-statements
about "the sources"/"the context" from claim extraction, and allows a
claim to override an outdated reference when the judge is highly
confident from its own well-established knowledge (reference_conflict).

OFFICIAL LABEL = DERIVED LABEL for v2 (unlike v1, where the judge's own
self-reported label was authoritative) -- derive_label_v2() recomputes
it deterministically from answer_attempted + claims, and that is what
downstream summary/agreement/disagreement code treats as `label`. The
judge's own self-reported label is kept too, as `judge_reported_label`,
purely to keep the consistency rate reportable -- see parse_judgment_v2().

judge-v1 is NEVER imported or modified by this module -- the two prompt
modules are fully independent, frozen instruments; judge-v1's own
pilot outputs stay exactly as they are for reproducibility.
"""

import json
import re

from judge_prompt_v1 import blind_candidate  # noqa: F401 -- re-exported, same blinding both versions use

PROMPT_VERSION = "judge-v2"
BLINDING_VERSION = "blind-v2"  # same blinding scheme as v1 -- reused, not redefined
# Bumped whenever parse_judgment_v2()'s PARSING logic changes (not the
# prompt) -- lets resume/mixed-version-guard code (see _judge_common.py's
# JUDGE_RESUME_FIELDS and judge_agreement.load_judge_records()) tell a
# record parsed under an older, stricter/looser parser apart from one
# parsed under the current logic, even though the prompt itself (and
# therefore PROMPT_VERSION) never changed. "pv1" is the first version to
# carry this field at all -- every record that predates it has
# parse_version == None, which already differs from "pv1", so the
# resume-key change below is enough to make old parse-error records
# re-judgeable without any extra special-casing.
PARSE_VERSION = "pv1"

# Narrow, explicit allow-lists for the ONLY benign non-conforming values
# tolerated during parsing (2026-10-07 follow-up: the primary judge,
# Llama-3.3-70B-Instruct, sometimes emits the JSON STRING "null" instead
# of a real null for `severity`, and the JSON STRING "false" instead of
# a real boolean for `reference_conflict`). Anything else unexpected
# still produces parse_error -- this is deliberately NOT a general
# type-coercion (Python's bare `bool(x)` on an arbitrary string is WRONG
# for exactly this reason: bool("false") == True).
_SEVERITY_NULLISH_STRINGS = {"null", "none"}
_BOOLISH_STRINGS = {"true": True, "false": False}

LABELS = ("FAKTUAL", "HALUSINASI_SEBAGIAN", "HALUSINASI_PENUH", "ABSTAIN")
COMPLETENESS_VALUES = ("FULL", "PARTIAL", "NONE")
CLAIM_VERDICTS = ("SUPPORTED", "CONTRADICTED", "FABRICATED", "UNVERIFIABLE")

SYSTEM_PROMPT = """You are an expert software engineer acting as an impartial evaluator of answers to Stack Overflow programming questions. Decide whether the CANDIDATE ANSWER contains hallucinations, using the REFERENCE ANSWER (the accepted answer on Stack Overflow) and your own well-established technical knowledge.

## Step 1 - Did the candidate attempt an answer?
answer_attempted = true if the candidate makes at least one technical claim that addresses the problem asked: a fix, an explanation of the cause, or a specific tool/API/approach and how it applies.
answer_attempted = false if the candidate only: says it cannot answer; says the information or sources are insufficient; summarizes material that does not address the problem; restates the question; or gives only generic advice (e.g. "check the documentation", "ask the community"). This holds even if the response is long.

## Step 2 - Extract claims (only if answer_attempted is true)
A CLAIM is a technical statement presented as fact about: an API, function, method, parameter, flag, option, config key, library or tool; its behavior, version or compatibility; the expected effect of code; or the proposed solution itself.
These are NOT claims; do not list them:
- statements about "the retrieved sources", "the context", "the provided information", or what they do or do not cover;
- statements about the answer itself or the assistant's ability to answer;
- style or design preferences that are matters of taste (e.g. one function per item vs. one shared function);
- suggestions to verify something or consult documentation.

Verdict for each claim:
- SUPPORTED: consistent with the reference, or correct according to well-established technical knowledge.
- CONTRADICTED: conflicts with well-established technical facts, or with the reference when the reference is correct.
- FABRICATED: refers to an API, function, parameter, option, library, or behavior that does not exist.
- UNVERIFIABLE: cannot be confirmed or refuted with confidence. This is NOT a hallucination.
The reference may be outdated or incomplete. If a claim conflicts with the reference but you are highly confident, from well-established knowledge, that the claim is correct, mark it SUPPORTED and set "reference_conflict": true.

Severity (only for CONTRADICTED or FABRICATED):
- CORE: the error would make the proposed solution fail, or makes the answer address a different problem.
- MINOR: a peripheral detail; the proposed solution would still work.

## Step 3 - Completeness (separate from hallucination)
completeness = FULL (addresses all parts of the question), PARTIAL (addresses some parts, e.g. it says other parts are not covered), or NONE (answer_attempted is false).
Incompleteness NEVER makes an answer a hallucination.

## Decision rules (apply in order)
0. answer_attempted is false -> ABSTAIN
1. any CORE claim is CONTRADICTED or FABRICATED -> HALUSINASI_PENUH
2. any MINOR claim is CONTRADICTED or FABRICATED -> HALUSINASI_SEBAGIAN
3. otherwise -> FAKTUAL

## Important
- A correct alternative solution that differs from the reference is FAKTUAL. Not using the reference's solution is not an error.
- Ignore length, style, formatting, and tone.
- Be conservative: if you are not confident a claim is wrong, mark it UNVERIFIABLE.
- List at most 8 claims, prioritizing those most important to the solution.

## Examples (illustrative only)
Question: How do I remove a column from a pandas DataFrame? Reference: use df.drop(columns="x").
(a) Candidate: "The retrieved sources do not cover removing columns, so I cannot provide a solution."
    -> answer_attempted=false, completeness=NONE, claims=[], label=ABSTAIN
(b) Candidate: "The sources do not cover the second part of your question. For removing the column, use df.drop(columns='x')."
    -> answer_attempted=true, completeness=PARTIAL, claims=[df.drop(columns='x') removes the column: SUPPORTED], label=FAKTUAL
(c) Candidate: "Use del df['x']."
    -> different from the reference but correct: SUPPORTED, label=FAKTUAL
(d) Candidate: "Use df.remove_column('x')."
    -> FABRICATED, CORE, label=HALUSINASI_PENUH
(e) Candidate: "Use df.drop(columns='x'); note that inplace defaults to True."
    -> main solution SUPPORTED; "inplace defaults to True" CONTRADICTED, MINOR; label=HALUSINASI_SEBAGIAN

## Output
Respond with JSON only, with exactly these keys in this order:
{
  "answer_attempted": true | false,
  "completeness": "FULL | PARTIAL | NONE",
  "claims": [
    {"claim": "<short paraphrase>",
     "verdict": "SUPPORTED | CONTRADICTED | FABRICATED | UNVERIFIABLE",
     "severity": "CORE | MINOR | null",
     "reference_conflict": true | false,
     "evidence": "<one short sentence>"}
  ],
  "reasoning": "<2-4 sentences applying the decision rules>",
  "label": "FAKTUAL | HALUSINASI_SEBAGIAN | HALUSINASI_PENUH | ABSTAIN"
}"""

# Identical to judge_prompt_v1.USER_TEMPLATE -- the question/reference/
# candidate framing itself didn't need to change, only the judge's own
# decision procedure (SYSTEM_PROMPT) did.
USER_TEMPLATE = """### QUESTION
Title: {title}
Tags: {tags}
Body:
{body}

### REFERENCE ANSWER (accepted on Stack Overflow)
{reference}

### CANDIDATE ANSWER
{candidate}"""


def build_messages(title, body, tags, reference, candidate, max_chars_each=6000):
    """Identical shape to judge_prompt_v1.build_messages() -- same
    truncation, same blind_candidate() call (imported from v1, not
    redefined) -- only SYSTEM_PROMPT differs."""
    def cut(s):
        s = s or ""
        return s if len(s) <= max_chars_each else s[:max_chars_each] + "\n[...truncated]"

    user = USER_TEMPLATE.format(
        title=title,
        tags=tags,
        body=cut(body),
        reference=cut(reference),
        candidate=cut(blind_candidate(candidate)),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Postprocessing: parse + deterministic official label
# ---------------------------------------------------------------------------

def derive_label_v2(answer_attempted: bool, claims: list[dict]) -> str:
    """Decision rules 0-3, deterministic. Rule 0 (answer_attempted is
    false -> ABSTAIN) takes priority over anything in `claims` -- a
    judge that lists claims despite answer_attempted=false is an
    inconsistency the caller flags separately (attempted_claims_
    conflict), not a reason to skip rule 0 here."""
    if not answer_attempted:
        return "ABSTAIN"
    errors = [c for c in claims
              if str(c.get("verdict", "")).upper() in ("CONTRADICTED", "FABRICATED")]
    if any(str(c.get("severity", "")).upper() == "CORE" for c in errors):
        return "HALUSINASI_PENUH"
    if errors:
        return "HALUSINASI_SEBAGIAN"
    return "FAKTUAL"


def parse_judgment_v2(raw: str) -> dict:
    """Parse judge-v2 output. Returns a dict with:
    label               : the OFFICIAL label == derived_label (see
                           module docstring) -- what every downstream
                           summary/agreement/disagreement reader treats
                           as "the" label for judge-v2.
    derived_label       : same value as label (kept as an explicit,
                           separate field per the user's follow-up, so
                           a reader never has to remember that "label"
                           means something different across versions --
                           label and derived_label are ALWAYS equal for
                           v2, by construction).
    judge_reported_label: what the judge itself said in its own
                           "label" JSON field -- kept only to compute
                           `consistent` (the consistency rate).
    consistent          : judge_reported_label == label (ABSTAIN from
                           answer_attempted=false counts as consistent
                           with a judge-reported ABSTAIN; mismatches are
                           real inconsistencies worth reporting).
    answer_attempted, completeness, claims, reasoning: as described in
                           SYSTEM_PROMPT.
    attempted_claims_conflict: true if answer_attempted is false but
                           claims were listed anyway (a judge self-
                           contradiction -- rule 0 still wins, but this
                           is flagged for human review).
    normalized_fields   : list of field names (one entry PER occurrence,
                           not deduplicated -- e.g. ["severity",
                           "severity", "reference_conflict"]) that had a
                           benign non-conforming value normalized into
                           its real type (see _SEVERITY_NULLISH_STRINGS/
                           _BOOLISH_STRINGS above). Empty list if every
                           field was already well-typed. Lets a caller
                           count exactly how often each kind of
                           normalization fired across a batch.
    parse_version       : PARSE_VERSION -- present on EVERY record this
                           function returns, including parse_error ones,
                           so the mixed-parse-version guard and resume
                           logic can see it even for a still-broken item.
    parse_error         : True if the JSON is unparseable OR any
                           required field is missing/invalid -- a
                           missing/invalid field is NEVER silently
                           defaulted into a valid-looking record. The
                           narrow normalizations above are the ONLY
                           exception to this rule, and only for the
                           exact benign variants named above.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return {"parse_error": True, "raw": raw, "parse_version": PARSE_VERSION}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"parse_error": True, "raw": raw, "parse_version": PARSE_VERSION}

    if not isinstance(data, dict):
        return {"parse_error": True, "raw": raw, "parse_version": PARSE_VERSION}

    answer_attempted_raw = data.get("answer_attempted")
    if not isinstance(answer_attempted_raw, bool):
        return {"parse_error": True, "raw": raw, "parse_version": PARSE_VERSION}
    answer_attempted = answer_attempted_raw

    completeness = str(data.get("completeness", "")).strip().upper()
    if completeness not in COMPLETENESS_VALUES:
        return {"parse_error": True, "raw": raw, "parse_version": PARSE_VERSION}

    claims_raw = data.get("claims")
    if not isinstance(claims_raw, list):
        return {"parse_error": True, "raw": raw, "parse_version": PARSE_VERSION}
    claims: list[dict] = []
    normalized_fields: list[str] = []
    for c in claims_raw:
        if not isinstance(c, dict):
            return {"parse_error": True, "raw": raw, "parse_version": PARSE_VERSION}
        verdict = str(c.get("verdict", "")).strip().upper()
        if verdict not in CLAIM_VERDICTS:
            return {"parse_error": True, "raw": raw, "parse_version": PARSE_VERSION}

        severity_raw = c.get("severity")
        if severity_raw is None:
            severity = None
        elif isinstance(severity_raw, str) and severity_raw.strip().lower() in _SEVERITY_NULLISH_STRINGS:
            severity = None
            normalized_fields.append("severity")
        else:
            severity = str(severity_raw).strip().upper()
        if severity not in (None, "CORE", "MINOR"):
            return {"parse_error": True, "raw": raw, "parse_version": PARSE_VERSION}

        rc_raw = c.get("reference_conflict", False)
        if isinstance(rc_raw, bool):
            reference_conflict = rc_raw
        elif isinstance(rc_raw, str) and rc_raw.strip().lower() in _BOOLISH_STRINGS:
            reference_conflict = _BOOLISH_STRINGS[rc_raw.strip().lower()]
            normalized_fields.append("reference_conflict")
        else:
            return {"parse_error": True, "raw": raw, "parse_version": PARSE_VERSION}

        claims.append({
            "claim": c.get("claim", ""), "verdict": verdict, "severity": severity,
            "reference_conflict": reference_conflict, "evidence": c.get("evidence", ""),
        })

    judge_reported_label = str(data.get("label", "")).strip().upper()
    if judge_reported_label not in LABELS:
        return {"parse_error": True, "raw": raw, "parse_version": PARSE_VERSION}

    official_label = derive_label_v2(answer_attempted, claims)
    attempted_claims_conflict = (not answer_attempted) and len(claims) > 0

    return {
        "parse_error": False,
        "label": official_label,
        "derived_label": official_label,
        "judge_reported_label": judge_reported_label,
        "consistent": judge_reported_label == official_label,
        "answer_attempted": answer_attempted,
        "completeness": completeness,
        "claims": claims,
        "reasoning": data.get("reasoning", ""),
        "attempted_claims_conflict": attempted_claims_conflict,
        "normalized_fields": normalized_fields,
        "prompt_version": PROMPT_VERSION,
        "parse_version": PARSE_VERSION,
    }
