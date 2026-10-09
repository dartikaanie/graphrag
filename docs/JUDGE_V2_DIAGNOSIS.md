# Judge-v2 diagnosis: missing HALUSINASI_SEBAGIAN and leniency patterns

**Diagnosis only -- no API calls, no prompt changes, no edits to judge code.** Computed on the reparsed judge-v2 files. Pilot batch (n=90 per judge) is part of the TEST sample -- nothing here is used to tune the judge; findings motivate a revision to be developed and validated on the dev set.

## 1. Corrected summary counts

| judge | v1-HALUSINASI items | -> FAKTUAL | -> HALUSINASI_PENUH | -> ABSTAIN |
|---|---|---|---|---|
| primary | 40 | 12 | 16 | 12 |
| secondary | 30 | 19 | 6 | 5 |

Corrects the chat summary given alongside `docs/JUDGE_V2_LENIENCY_REVIEW.md` (which quoted only the HALUSINASI->FAKTUAL drill-down counts, 12 and 19, as if they were the full v1-HALUSINASI totals). The review doc's own matrices and section headers ("primary (12 item(s))", "secondary (19 item(s))") were already correct -- they label the drill-down specifically, not the totals -- so no edit to that file was needed.

## 2. Why is HALUSINASI_SEBAGIAN never produced?

### `derive_label_v2` (quoted in full, `llm/evaluation/judge_prompt_v2.py`)

```python
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
```

### Prompt sections defining verdict, severity, and the label rules (quoted, `SYSTEM_PROMPT` in the same file)

```
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
```

### Exact condition for SEBAGIAN, and whether it is reachable

`derive_label_v2` returns `HALUSINASI_SEBAGIAN` iff: `answer_attempted` is true, AND at least one claim has `verdict` in `(CONTRADICTED, FABRICATED)`, AND **none** of those error claims has `severity == "CORE"`. The code is a correct, literal implementation of decision rules 1-3 above (rule 1 checked first via the `any(... == "CORE")` branch, rule 2 as the fallback `if errors`, rule 3 as the final fallback) -- there is no rule-ordering bug and no severity-value mismatch in the code itself (`severity` is always normalized to `None`/`"CORE"`/`"MINOR"` by `parse_judgment_v2`, exactly the three values this function checks against). **The rule is reachable in code.**

### Claims by verdict x severity (judge-v2, reparsed, n=90 items each)

| judge | (SUPPORTED, None) | (SUPPORTED, CORE) | (SUPPORTED, MINOR) | (CONTRADICTED, CORE) | (CONTRADICTED, MINOR) | (FABRICATED, \*) | (UNVERIFIABLE, None) |
|---|---|---|---|---|---|---|---|
| primary | 160 | 29 | 4 | 21 | 1 | 0 | 45 |
| secondary | 307 | 0 | 0 | 9 | 0 | 0 | 19 |

### Items with >=1 CONTRADICTED/FABRICATED claim severity MINOR and NO CORE error (the exact SEBAGIAN condition)

| judge | count | labels assigned |
|---|---|---|
| primary | **0** | -- |
| secondary | **0** | -- |

Primary has exactly **one** CONTRADICTED+MINOR claim in the entire 90-item pilot batch, and it co-occurs in an item that *also* has a CONTRADICTED+CORE claim -- so that item is (correctly, per the rules) HALUSINASI_PENUH, not SEBAGIAN. Secondary never emits a MINOR severity on an error claim at all (0 occurrences across 90 items, vs. 9 CORE). Neither judge ever emits a FABRICATED claim of any severity in this batch.

### Conclusion: judge behavior, not a code bug

The code correctly implements the documented rule, and the rule is reachable in principle (a MINOR-only error item would get SEBAGIAN). What's missing is the *data*: on this pilot batch, the judge-v2 models essentially never assign `severity: MINOR` to a CONTRADICTED/FABRICATED claim in isolation -- they assign CORE (29+21=50 CORE assignments across both judges, including 29 CORE assignments on claims that are not even errors, see section 3a) or they don't produce a MINOR-only contradiction at all. This reads as a severity-calibration behavior of the current judge-v2 prompt/models on this data, not a defect in `derive_label_v2`.

### Comparison with judge-v1

v1's label is the judge's own self-reported label (no `derive_label`-style recomputation), but it is still informative: v1 primary produced **10** HALUSINASI_SEBAGIAN items and v1 secondary produced **12**, driven by 9 (primary) + 3 (secondary) CONTRADICTED/MINOR claims and 1 FABRICATED/MINOR claim (secondary) -- all absent or nearly absent under v2 on the very same 90 generation outputs. Both v1 and v2 also assign CORE/MINOR severity to non-error (SUPPORTED/UNVERIFIABLE) claims (v1: e.g. 16 SUPPORTED+CORE in its own SEBAGIAN items alone), so that particular prompt-adherence looseness is not new to v2 -- what changed is how rarely MINOR gets attached to an actual CONTRADICTED/FABRICATED claim.

**If this were to be fixed** (not applied here, dev-set only): the likely lever is the SEVERITY rubric wording itself (what counts as "the proposed solution would still work" vs. "would fail") or the decision-rule framing nudging the model toward a binary CORE-or-nothing judgement rather than a graded CORE/MINOR one. Tests to add alongside any such change: (1) a `parse_judgment_v2`/`derive_label_v2` unit test fixing a MINOR-only CONTRADICTED claim and asserting `HALUSINASI_SEBAGIAN` (the code path is already covered this way in `test_judge_prompt_v2.py`, so this is about the PROMPT's ability to elicit that claim shape, not the code); (2) a dev-set-only count of verdict x severity combinations before/after any prompt revision, same shape as the table above, to confirm MINOR becomes reachable in practice without a corresponding drop in genuine CORE detections.

## 3. Leniency patterns (per judge, per run_label)

### a. Severity assigned to SUPPORTED claims (should always be `null` per the prompt -- "Severity (only for CONTRADICTED or FABRICATED)")

| judge | count |
|---|---|
| primary | 33 (29 CORE + 4 MINOR) |
| secondary | 0 |

Primary violates this instruction on over a third of its CORE/MINOR severity tags (33 of 33+21+1=55 total non-null severities are on non-error claims). This never changes a label on its own (`derive_label_v2` only reads severity on CONTRADICTED/FABRICATED claims), but it is a direct, countable instruction-following miss, concentrated entirely in the primary judge.

### b. Meta-statements extracted as claims (should not be claims at all -- "statements about \"the retrieved sources\", \"the context\", \"the provided information\", or what they do or do not cover")

Heuristic (exact, auditable regex): `retrieved sources? (?:do|does) not`, `sources? do(?:es)? not (?:cover|address|mention|discuss|provide)`, `not (?:specifically )?(?:covered|addressed) (?:by|in) the (?:sources|context|provided)`, `not addressed in the provided context`, `not covered in (?:detail )?(?:the|in) (?:retrieved )?(?:sources|context)`, `context does not (?:cover|address|mention)` (case-insensitive).

| judge | count | verdicts seen |
|---|---|---|
| primary | 3 | 2x CONTRADICTED/CORE, 1x SUPPORTED |
| secondary | 5 | all SUPPORTED |

Two of primary's three matches are the SAME underlying rule violation on question_id 2617170 (run_labels D-grounded and C-uniform-grounded): the claim *"Retrieved sources do not provide information on browser support for slashes or starting class names with numbers"* is a meta-statement about source coverage -- explicitly excluded from claim extraction by the prompt -- yet it was extracted, marked CONTRADICTED/CORE, and is the SOLE reason both of those records are HALUSINASI_PENUH. This is the opposite of "leniency": a meta-statement extraction violation here *manufactures* a hallucination finding rather than excusing one. Secondary's 5 matches are all SUPPORTED and don't change any label, but still shouldn't have been extracted as claims per the prompt.

### c. UNVERIFIABLE claims

| judge | total UNVERIFIABLE | of which CORE |
|---|---|---|
| primary | 45 | 0 |
| secondary | 19 | 0 |

`derive_label_v2` never treats UNVERIFIABLE as an error regardless of severity (only `CONTRADICTED`/`FABRICATED` enter the `errors` list) -- consistent with the prompt ("This is NOT a hallucination"). No UNVERIFIABLE claim in this batch was ever assigned CORE severity either (judges appear to follow "severity only for CONTRADICTED/FABRICATED" correctly for this verdict, even where they don't for SUPPORTED).

FAKTUAL items where >=50% of claims are UNVERIFIABLE (primary only -- secondary has none):

| question_id | run_label | UNVERIFIABLE / total claims |
|---|---|---|
| 20443560 | C-uniform-plain | 4 / 8 |
| 11118023 | B-grounded | 2 / 3 |
| 3882147 | D-plain | 3 / 4 |
| 20443560 | B-plain | 3 / 5 |
| 51061836 | C-uniform-grounded | 2 / 4 |

These are FAKTUAL purely because UNVERIFIABLE can never trigger an error, even when it's the majority of what was extracted -- i.e. an item can be labeled FAKTUAL while the judge itself expressed low confidence in most of what it checked. This is a real property of the decision rule, not a bug -- the prompt says UNVERIFIABLE is explicitly not a hallucination signal -- but it's a leniency pattern worth naming: "FAKTUAL" does not imply "verified."

### d. Claims asserting code validity marked SUPPORTED

Heuristic: `valid (?:rust )?(?:code|syntax)` or `(?:code|snippet) (?:is valid|works|compiles)` (case-insensitive).

| judge | count |
|---|---|
| primary | 2 |
| secondary | 1 |

All three are on question_id 76224221 (the Rust `let...else` question) and all three are discussed in section 4 below, where the accepted answer actually states the construct was stabilized in **Rust 1.65.0**, not 1.64 -- the claims marked SUPPORTED here are not merely "unverified code validity" but concretely wrong by the reference's own citation.

## 4. Evidence for manual review

### question_id 20443560, run_label D-plain ("binary shaders ARE/NOT recommended")

**Candidate answer (relevant part):** "While OpenGL supports pre-compiled binary shaders, they are **not recommended** for shipping as they are not portable across different hardware vendors or even different devices from the same vendor."

**Accepted answer (relevant part):** "...precompiled shader binaries **do not make sense at all**." (with further discussion of why binary formats aren't portable across vendors/architectures).

**Verdict: v2's reading matches the candidate and the reference.** The candidate said "not recommended," and the reference agrees shipping precompiled binaries doesn't make sense. v1 (primary) extracted this as *"Pre-compiled binary shaders are recommended for shipping"* -- CONTRADICTED/CORE -- which inverts the candidate's actual claim direction. This looks like a v1 extraction error on this item, not evidence that v2 is being lenient.

| judge | version | claim | verdict/severity |
|---|---|---|---|
| primary | v1 | "Pre-compiled binary shaders are recommended for shipping" | CONTRADICTED/CORE |
| primary | v1 | "Shader sources can be protected by encryption" | CONTRADICTED/MINOR |
| primary | v2 | "Pre-compiled binary shaders are not recommended for shipping due to portability issues" | SUPPORTED/None |
| secondary | v1 | "Pre-compiled binary shaders are not recommended for shipping due to portability issues." | SUPPORTED/CORE |
| secondary | v2 | "Pre-compiled binary shaders are not recommended for shipping due to portability issues" | SUPPORTED/None |

(Secondary, both versions, already read this correctly -- only primary/v1 inverted it.)

### question_id 38118194, run_label A, secondary judge (DocX/UWP)

**Candidate answer (relevant part):** "**DocX**: This library is a .NET library that allows for easy generation of Word documents. **While it's not directly available for UWP**, you could try using DocX through .NET Standard libraries if they work for your UWP project."

**Accepted answer (relevant part):** "For MS Word document... it uses Open-XML-SDK and currently it doesn't support UWP platform." (No Word-generation library is endorsed as working in UWP; DocX is not mentioned at all in the reference.)

**Verdict: the UWP qualifier was dropped by v2, not strictly fabricated by v1.** The candidate's own claim about DocX is already hedged ("not directly available... you could try"). v1 extracted *"DocX can be used to generate Word documents in UWP"* -- a flattened, unhedged version of the candidate's claim -- and marked it CONTRADICTED/CORE against the reference's silence on DocX specifically (a reasonable but strict reading: the reference affirms no good UWP Word-generation option exists, and DocX-in-UWP is exactly such an option). v2 extracted *"DocX is a .NET library that allows for easy generation of Word documents"* -- true in general, but it drops the UWP context entirely, converting a checkable (and likely wrong) UWP-specific claim into a trivially-true generic one. This is a concrete instance of the leniency mechanism: context-stripping during claim paraphrase turns a specific, falsifiable claim into a generic truism.

| judge | version | claim | verdict/severity |
|---|---|---|---|
| secondary | v1 | "DocX can be used to generate Word documents in UWP." | CONTRADICTED/CORE |
| secondary | v1 | "Open XML SDK is compatible with UWP if the appropriate assemblies are referenced." | CONTRADICTED/CORE |
| secondary | v1 | "You can use DocX to insert images into a Word document." | CONTRADICTED/CORE |
| secondary | v2 | "DocX is a .NET library that allows for easy generation of Word documents." | SUPPORTED/None |
| secondary | v2 | "Open XML SDK is compatible with UWP if you reference the appropriate assemblies." | SUPPORTED/None |

(Primary shows the same pattern: v1 CONTRADICTED/CORE on "Open XML SDK is compatible with UWP" vs. v2 SUPPORTED/CORE on the identical claim text, with `reference_conflict: true` -- i.e. primary's v2 judge explicitly invoked the reference-conflict override here, a DIFFERENT leniency mechanism from secondary's context-stripping on the same underlying question.)

### question_id 76224221, run_label B-plain -- "let...else was stabilized in Rust 1.64"

**Accepted answer states:** "the suggested code has been valid code [since 1.65.0, released November 2022](https://blog.rust-lang.org/2022/11/03/Rust-1.65.0.html#let-else-statements)."

**Candidate answer states:** "...ensure that you are using a version of Rust that supports the `let...else` syntax, which was stabilized in **Rust 1.64**."

**This is a genuine, concretely-wrong claim against the reference -- 1.64 vs. 1.65.0, with a dated citation in the reference.** It requires no outside knowledge to catch, only reading the reference's own link. Every judge/version combination on this item except primary/v1 marked it SUPPORTED:

| judge | version | claim | verdict/severity |
|---|---|---|---|
| primary | v1 | "The let...else syntax was stabilized in Rust 1.64" | **CONTRADICTED/MINOR** (correctly caught) |
| primary | v2 | "The let...else syntax was stabilized in Rust 1.64." | SUPPORTED/None (missed) |
| secondary | v1 | "The let...else syntax was stabilized in Rust 1.64." | SUPPORTED/None (missed) |
| secondary | v2 | "The `let...else` syntax was stabilized in Rust 1.64." | SUPPORTED/None (missed) |

Even the one judge/version that caught it (primary/v1) only assigned MINOR severity to a claim that is directly, citably false against the reference -- illustrating the severity-calibration question from section 2 from another angle: this model treats a dated-version factual error as "peripheral" rather than core, even when it does notice it.

