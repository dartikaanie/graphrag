"""
judge_prompt_v2_1.py
--------------------
Judge-v2.1: the revision of judge-v2 defined in docs/DECISION_C_SCORING.md,
Amendment 4 (rules 1-7) and its Addendum (single judging standard for the
judge and the human annotators; severity handling). judge_prompt_v2.py is
FROZEN and untouched -- judge-v2 stays selectable and its outputs stay
reproducible; this module is an independent instrument.

What changes relative to judge-v2 (Amendment 4):
  1. Extracted claims keep the qualifiers that make them checkable.
  2. Statements about the sources/context are not claims.
  3. Severity only on CONTRADICTED/FABRICATED claims.
  4. CORE vs MINOR with criteria and examples.
  5. "Code is valid / compiles / works" is UNVERIFIABLE unless the
     reference contains the same code or says so.
  6. UNVERIFIABLE never makes an answer hallucinated.
  Review changes (2026-10-09): CONTRADICTED/FABRICATED claims are never
  dropped (the limit of 8 covers SUPPORTED/UNVERIFIABLE claims only);
  SUPPORTED requires the reference to state or directly imply the claim.
  7. max_tokens and temperature are explicit (REQUEST_MAX_TOKENS,
     REQUEST_TEMPERATURE below) and recorded on every record by the runner.
Addendum: the reference decides when it addresses a claim; a claim it does
not address is UNVERIFIABLE unless clearly fabricated or contradicting
well-established fact; reference_conflict stays for an outdated reference.

PRIMARY LABEL for v2.1 is the binary `hallucinated` (judge_binary.
derive_hallucinated). The severity label is descriptive only.

All examples in SYSTEM_PROMPT are INVENTED -- none is taken from the test
sample (positions 1-384) or the dev set (385-434).
"""

import json
import re

from judge_prompt_v1 import blind_candidate  # noqa: F401 -- same blinding as v1/v2
from judge_prompt_v2 import USER_TEMPLATE  # identical question/reference/candidate framing

PROMPT_VERSION = "judge-v2.1"
BLINDING_VERSION = "blind-v2"
PARSE_VERSION = "pv2.1-1"

# Amendment 4 rule 7. temperature 0.0: same as judge-v2, repeatable.
# max_tokens 1500: about twice the longest judge-v2 pilot response (714
# completion tokens; median 333-388, at most 8 claims) -- bounds cost and
# makes a runaway response visible as finish_reason == "length".
REQUEST_TEMPERATURE = 0.0
REQUEST_MAX_TOKENS = 1500

# Prompt limit on SUPPORTED + UNVERIFIABLE claims. CONTRADICTED/FABRICATED
# claims are never capped. The parser never drops a claim: a response
# with more than this many non-error claims is kept whole and flagged
# (`non_error_claims_over_cap`), so no verdict is silently lost.
MAX_NON_ERROR_CLAIMS = 8

_SEVERITY_NULLISH_STRINGS = {"null", "none", ""}
_BOOLISH_STRINGS = {"true": True, "false": False}

# HALUSINASI_SEVERITY_UNSPECIFIED: the answer IS hallucinated (binary label
# true) but its severity cannot be derived, because an error claim carries
# no severity and no CORE error settles it. Never imputed to SEBAGIAN
# (Amendment 4 Addendum).
LABELS = ("FAKTUAL", "HALUSINASI_SEBAGIAN", "HALUSINASI_PENUH", "ABSTAIN", "HALUSINASI_SEVERITY_UNSPECIFIED")
JUDGE_REPORTABLE_LABELS = ("FAKTUAL", "HALUSINASI_SEBAGIAN", "HALUSINASI_PENUH", "ABSTAIN")
COMPLETENESS_VALUES = ("FULL", "PARTIAL", "NONE")
CLAIM_VERDICTS = ("SUPPORTED", "CONTRADICTED", "FABRICATED", "UNVERIFIABLE")
ERROR_VERDICTS = ("CONTRADICTED", "FABRICATED")

SYSTEM_PROMPT = """You are an expert software engineer acting as an impartial evaluator of answers to Stack Overflow programming questions. Decide whether the CANDIDATE ANSWER contains hallucinations, judged against the REFERENCE ANSWER (the accepted answer on Stack Overflow).

## Judging standard
- When the reference addresses a claim, the reference decides: a claim the reference states or directly implies is SUPPORTED, a claim that conflicts with it is CONTRADICTED.
- When the reference does not address a claim, the claim is UNVERIFIABLE, unless it is clearly fabricated (it names an API, function, parameter, option, or library that does not exist) or it contradicts a well-established technical fact.
- Outdated reference: if a claim conflicts with the reference but you are highly confident, from well-established knowledge, that the claim is correct for the platform and version in the question, mark it SUPPORTED and set "reference_conflict": true.
- Never decide from a guess or a vague recollection. If you are not confident, the claim is UNVERIFIABLE.

## Step 1 - Did the candidate attempt an answer?
answer_attempted = true if the candidate makes at least one technical claim that addresses the problem asked: a fix, an explanation of the cause, or a specific tool/API/approach and how it applies.
answer_attempted = false if the candidate only: says it cannot answer; says the information or sources are insufficient; summarizes material that does not address the problem; restates the question; or gives only generic advice (e.g. "check the documentation", "ask the community"). This holds even if the response is long.

## Step 2 - Extract claims (only if answer_attempted is true)
A CLAIM is a technical statement presented as fact about: an API, function, method, parameter, flag, option, config key, library or tool; its behavior, version or compatibility; the expected effect of code; or the proposed solution itself.

Keep the qualifiers. Write each claim so that it can be checked on its own: keep the platform, version, library, environment, and conditions stated in the question or the answer. Write "X works in <framework> <version>", not "X works". Dropping a qualifier changes what is being claimed.

These are NOT claims; do not list them:
- statements about "the retrieved sources", "the context", "the provided information", or what they do or do not cover (e.g. "the sources do not cover X");
- statements about the answer itself or the assistant's ability to answer;
- style or design preferences that are matters of taste;
- suggestions to verify something or consult documentation.

Verdict for each claim:
- SUPPORTED: the reference states the claim or directly implies it (or an outdated-reference case, see the judging standard). Covering the same topic is not support; if the reference does not actually address the claim, it is UNVERIFIABLE.
- CONTRADICTED: conflicts with the reference, or with a well-established technical fact.
- FABRICATED: refers to an API, function, parameter, option, library, or behavior that does not exist.
- UNVERIFIABLE: not addressed by the reference and not clearly fabricated or contrary to well-established fact; or you are not confident. This is NOT a hallucination.

Code validity: a claim that code is valid, compiles, runs, or works is UNVERIFIABLE unless the reference contains the same code or states this explicitly.

Severity - only for CONTRADICTED or FABRICATED claims. For SUPPORTED and UNVERIFIABLE claims, severity is null.
- CORE: the error affects the main solution or the direct answer to the question. Following the answer would fail, or would solve a different problem.
  Examples: the recommended function does not exist; the central step does the opposite of what is claimed.
- MINOR: the error is in a secondary detail and does not change the solution.
  Examples: a wrong version number for when a feature was introduced; an incorrect side remark about a default value that the solution does not rely on.

## Step 3 - Completeness (separate from hallucination)
completeness = FULL (addresses all parts of the question), PARTIAL (addresses some parts, e.g. it says other parts are not covered), or NONE (answer_attempted is false).
Incompleteness NEVER makes an answer a hallucination.

## Decision rules (apply in order)
0. answer_attempted is false -> ABSTAIN
1. any CORE claim is CONTRADICTED or FABRICATED -> HALUSINASI_PENUH
2. any MINOR claim is CONTRADICTED or FABRICATED -> HALUSINASI_SEBAGIAN
3. otherwise -> FAKTUAL
UNVERIFIABLE claims never change the label: an answer whose claims are all SUPPORTED or UNVERIFIABLE is FAKTUAL.

## Important
- A solution that differs from the reference is not an error. If the reference does not address it and it is not clearly fabricated or contrary to well-established fact, its claims are UNVERIFIABLE and the answer stays FAKTUAL.
- Ignore length, style, formatting, and tone.
- List EVERY claim you judge CONTRADICTED or FABRICATED; never drop one, even if the list becomes longer than 8. The limit of 8 applies only to SUPPORTED and UNVERIFIABLE claims: list at most 8 of those, prioritizing the ones most important to the solution.

## Examples (illustrative only)
Question: In Python 3.11, how do I parse the string "2024-03-05T14:30:00Z" into a datetime? Reference: use datetime.fromisoformat(s); before Python 3.11 it could not parse a trailing "Z".
(a) Candidate: "The retrieved sources do not cover date parsing, so I cannot provide a solution."
    -> answer_attempted=false, completeness=NONE, claims=[], label=ABSTAIN
(b) Candidate: "The sources do not cover time zones in general. To parse the string, call datetime.fromisoformat(s)."
    -> answer_attempted=true, completeness=PARTIAL. The first sentence is about the sources: not a claim. claims=[In Python 3.11, datetime.fromisoformat(s) parses "2024-03-05T14:30:00Z": SUPPORTED, severity null], label=FAKTUAL
(c) Candidate: "Use dateutil.parser.isoparse(s) from the python-dateutil package."
    -> not addressed by the reference and not clearly fabricated: UNVERIFIABLE, severity null, label=FAKTUAL
(d) Candidate: "Call datetime.parse_iso(s)."
    -> datetime has no parse_iso: FABRICATED, CORE, label=HALUSINASI_PENUH
(e) Candidate: "Use datetime.fromisoformat(s). This function was added in Python 3.5."
    -> main solution SUPPORTED, severity null; "datetime.fromisoformat was added in Python 3.5" CONTRADICTED (well-established fact: it was added in 3.7), MINOR; label=HALUSINASI_SEBAGIAN
(f) Candidate: "datetime.fromisoformat(s) handles the trailing Z in Python 3.11. The snippet below runs without errors: <code>"
    -> claim 1 keeps its qualifier "in Python 3.11": SUPPORTED, severity null. "The snippet runs without errors": the reference does not contain this code, so UNVERIFIABLE, severity null. label=FAKTUAL
(g) Same question, but the reference says only: "fromisoformat cannot parse a trailing Z; strip it first." Candidate: "In Python 3.11, datetime.fromisoformat(s) parses the trailing Z directly."
    -> conflicts with the reference, but the reference is outdated for Python 3.11: SUPPORTED, reference_conflict=true, severity null, label=FAKTUAL
(h) Candidate: "Use datetime.fromisoformat(s). In Python 3.11 it also accepts ISO week dates such as 2024-W10-2."
    -> main solution SUPPORTED, severity null. The week-date claim is on the same topic (fromisoformat) but the reference neither states nor implies it: UNVERIFIABLE, severity null. label=FAKTUAL

## Output
Respond with JSON only, with exactly these keys in this order:
{
  "answer_attempted": true | false,
  "completeness": "FULL | PARTIAL | NONE",
  "claims": [
    {"claim": "<short paraphrase that keeps its qualifiers>",
     "verdict": "SUPPORTED | CONTRADICTED | FABRICATED | UNVERIFIABLE",
     "severity": "CORE | MINOR | null",
     "reference_conflict": true | false,
     "evidence": "<one short sentence>"}
  ],
  "reasoning": "<2-4 sentences applying the decision rules>",
  "label": "FAKTUAL | HALUSINASI_SEBAGIAN | HALUSINASI_PENUH | ABSTAIN"
}"""


def build_messages(title, body, tags, reference, candidate, max_chars_each=6000):
    """Same shape, truncation and blinding as judge-v1/v2 -- only
    SYSTEM_PROMPT differs."""
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
# Postprocessing: parse + deterministic labels
# ---------------------------------------------------------------------------

def derive_label_v2_1(answer_attempted: bool, claims: list[dict]) -> str:
    """Descriptive severity label (the PRIMARY v2.1 label is the binary
    `hallucinated`, see judge_binary.derive_hallucinated):
      not attempted                              -> ABSTAIN
      no CONTRADICTED/FABRICATED claim           -> FAKTUAL
      any error claim with severity CORE         -> HALUSINASI_PENUH
      every error claim has severity MINOR       -> HALUSINASI_SEBAGIAN
      otherwise (an error claim has no severity
      and no CORE error settles it)              -> HALUSINASI_SEVERITY_UNSPECIFIED
    A missing severity is never imputed (Amendment 4 Addendum)."""
    if not answer_attempted:
        return "ABSTAIN"
    errors = [c for c in claims if c.get("verdict") in ERROR_VERDICTS]
    if not errors:
        return "FAKTUAL"
    if any(c.get("severity") == "CORE" for c in errors):
        return "HALUSINASI_PENUH"
    if all(c.get("severity") == "MINOR" for c in errors):
        return "HALUSINASI_SEBAGIAN"
    return "HALUSINASI_SEVERITY_UNSPECIFIED"


def _fail(raw):
    return {"parse_error": True, "raw": raw, "parse_version": PARSE_VERSION}


def parse_judgment_v2_1(raw: str) -> dict:
    """Parse judge-v2.1 output. Same record shape as
    judge_prompt_v2.parse_judgment_v2 (label == derived_label,
    judge_reported_label, consistent, answer_attempted, completeness,
    claims, reasoning, attempted_claims_conflict, normalized_fields,
    parse_version, parse_error), with the Addendum's severity handling:
      - severity on a SUPPORTED/UNVERIFIABLE claim is set to null and
        recorded in normalized_fields as "severity_on_non_error";
      - a CONTRADICTED/FABRICATED claim with no severity is KEPT with
        severity null (never imputed) and counted in
        `n_error_claims_without_severity`.
    Any number of claims is accepted and NONE is dropped: error claims
    are never capped; more than MAX_NON_ERROR_CLAIMS SUPPORTED/
    UNVERIFIABLE claims sets `non_error_claims_over_cap` (the record is
    kept whole). `n_claims`/`n_error_claims` are recorded.
    A missing/invalid required field is still a parse_error."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
        if not m:
            return _fail(raw)
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return _fail(raw)

    if not isinstance(data, dict):
        return _fail(raw)

    answer_attempted = data.get("answer_attempted")
    if not isinstance(answer_attempted, bool):
        return _fail(raw)

    completeness = str(data.get("completeness", "")).strip().upper()
    if completeness not in COMPLETENESS_VALUES:
        return _fail(raw)

    claims_raw = data.get("claims")
    if not isinstance(claims_raw, list):
        return _fail(raw)

    claims: list[dict] = []
    normalized_fields: list[str] = []
    n_error_claims_without_severity = 0
    for c in claims_raw:
        if not isinstance(c, dict):
            return _fail(raw)
        verdict = str(c.get("verdict", "")).strip().upper()
        if verdict not in CLAIM_VERDICTS:
            return _fail(raw)

        severity_raw = c.get("severity")
        if severity_raw is None:
            severity = None
        elif isinstance(severity_raw, str) and severity_raw.strip().lower() in _SEVERITY_NULLISH_STRINGS:
            severity = None
            normalized_fields.append("severity")
        elif isinstance(severity_raw, str) and severity_raw.strip().upper() in ("CORE", "MINOR"):
            severity = severity_raw.strip().upper()
        else:
            return _fail(raw)

        if verdict not in ERROR_VERDICTS and severity is not None:
            severity = None
            normalized_fields.append("severity_on_non_error")
        if verdict in ERROR_VERDICTS and severity is None:
            n_error_claims_without_severity += 1

        rc_raw = c.get("reference_conflict", False)
        if isinstance(rc_raw, bool):
            reference_conflict = rc_raw
        elif isinstance(rc_raw, str) and rc_raw.strip().lower() in _BOOLISH_STRINGS:
            reference_conflict = _BOOLISH_STRINGS[rc_raw.strip().lower()]
            normalized_fields.append("reference_conflict")
        else:
            return _fail(raw)

        claims.append({
            "claim": c.get("claim", ""), "verdict": verdict, "severity": severity,
            "reference_conflict": reference_conflict, "evidence": c.get("evidence", ""),
        })

    judge_reported_label = str(data.get("label", "")).strip().upper()
    if judge_reported_label not in JUDGE_REPORTABLE_LABELS:
        return _fail(raw)

    official_label = derive_label_v2_1(answer_attempted, claims)
    n_error_claims = sum(1 for c in claims if c["verdict"] in ERROR_VERDICTS)
    n_non_error_claims = len(claims) - n_error_claims
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
        "attempted_claims_conflict": (not answer_attempted) and len(claims) > 0,
        "normalized_fields": normalized_fields,
        "n_error_claims_without_severity": n_error_claims_without_severity,
        "n_claims": len(claims),
        "n_error_claims": n_error_claims,
        "non_error_claims_over_cap": n_non_error_claims > MAX_NON_ERROR_CLAIMS,
        "prompt_version": PROMPT_VERSION,
        "parse_version": PARSE_VERSION,
    }
