# Judge-v2: what changed from judge-v1, and why

Judge-v1 and its outputs are **frozen** for reproducibility (`llm/evaluation/judge_prompt_v1.py`, `llm_judge_hallucination_v1.py`, and every `jv1_dashboard_*.jsonl` file are untouched by this work). Judge-v2 is a new, separate instrument (`judge_prompt_v2.py`, `llm_judge_hallucination_v2.py`, `jv2_dashboard_*.jsonl`) that fixes systematic problems the pilot judging (9 runs × 10 questions) surfaced in judge-v1's prompt — this is a fix to the measurement instrument, not a re-tuning of the study's results.

## Why: weighted κ = 0.33 in the pilot, and what was actually wrong

Primary (Llama-3.3-70B) and secondary (gpt-4o-mini) disagreed on 33% of pilot items (weighted κ = 0.33, far below the conventional 0.6 "moderate" threshold). Reading the disagreements showed this wasn't random annotator noise — it was the SAME few prompt design problems recurring across many questions:

### 1. Non-answers judged as hallucinations

Judge-v1's rule 0 for ABSTAIN only fired when the candidate "refuses, or only asks for clarification." A candidate that honestly says "the retrieved sources don't cover this" was NOT a refusal in that narrow sense, so judge-v1 fell through to rule 1 ("the answer addresses a different problem than the one asked" → HALUSINASI_PENUH). An honest admission of insufficient context was being scored as the WORST possible hallucination label.

- **Q20443560, C-trust-grounded**: both primary and secondary labeled PENUH for a candidate that said the context didn't cover the question.
- **Q411756, D-grounded**: primary said PENUH, secondary said ABSTAIN — the exact disagreement pattern this causes: one judge interprets the non-answer as "addresses a different problem" (PENUH), the other as a refusal (ABSTAIN), for the identical candidate text.
- **Q76224221, B-grounded**: same pattern — primary PENUH, secondary ABSTAIN.

This directly inflates the measured hallucination rate for **grounded** runs specifically, since grounding is what causes a model to sometimes say "I don't have enough information" instead of guessing — exactly the honest behavior the study wants to measure positively, not penalize as the worst hallucination class.

**Fix**: judge-v2 adds an explicit Step 1 (`answer_attempted`) with a broader, explicit definition of what counts as a non-answer ("says it cannot answer; says the information or sources are insufficient; summarizes material that does not address the problem; restates the question; or gives only generic advice"), and rule 0 (`answer_attempted is false → ABSTAIN`) is checked FIRST, before any claim-based rule. Completeness (Step 3) is a separate, explicit field (FULL/PARTIAL/NONE) and the prompt states directly: "Incompleteness NEVER makes an answer a hallucination."

### 2. Statements about "the retrieved sources" treated as technical claims

The candidate is blinded (citation markers stripped) but still sometimes narrates its own retrieval process ("the sources mention X but not Y"). Judge-v1 had no rule excluding these meta-statements from claim extraction, so a judge could extract "the sources do not cover removing columns" as a CLAIM and then evaluate it as a technical statement — even though the judge never saw the sources and has no way to verify a claim ABOUT them.

**Fix**: judge-v2's Step 2 explicitly lists "statements about 'the retrieved sources', 'the context', 'the provided information', or what they do or do not cover" as NOT claims, with an instruction to never list them.

### 3. Correct alternative solutions penalized for differing from the reference

- **Q11118023, A**: primary labeled PENUH because the proposed solution was judged "unnecessary" relative to the reference, despite being technically correct.

Judge-v1 did state "An alternative solution... is FAKTUAL," but didn't give the judge a worked example of this exact trap, and nothing separated "differs from the reference" from "is wrong."

**Fix**: judge-v2 keeps the same rule but adds two illustrative examples ((b) and (c) in SYSTEM_PROMPT) showing a correct alternative being scored FAKTUAL, and a reference_conflict mechanism (see §6) for when the reference itself is wrong.

### 4. Style or design preferences treated as contradictions

- **Q56612920**: "one function per emotion" (a design/style choice) was marked CONTRADICTED.

Judge-v1's claim definition didn't exclude style/taste statements from being extracted as technical claims at all.

**Fix**: judge-v2's Step 2 explicitly excludes "style or design preferences that are matters of taste (e.g. one function per item vs. one shared function)" from claim extraction.

### 5. Judge label inconsistent with its own claims

Primary's self-reported label matched its own derived-from-claims label only 70–90% of the time in the pilot — the judge's own reasoning didn't reliably determine its final answer.

**Fix**: for judge-v2, **the official label is the derived label**, not the judge's self-report (see "Official label = derived label" below) — a judge's internal inconsistency can no longer silently become the measured result. The self-reported label (`judge_reported_label`) is still stored, so the consistency rate (and `attempted_claims_conflict`, a related new metric) stays reportable as a diagnostic.

### 6. Outdated or incomplete reference answers

- **Q38118194**: the reference says no UWP PDF library exists, but XFINIUM.PDF does support UWP — a candidate correctly naming it would be WRONG according to the reference, but right according to current technical fact.

Judge-v1 had no mechanism for a judge to override an outdated reference from its own well-established knowledge.

**Fix**: judge-v2's Step 2 adds: "The reference may be outdated or incomplete. If a claim conflicts with the reference but you are highly confident, from well-established knowledge, that the claim is correct, mark it SUPPORTED and set `reference_conflict: true`." Every such claim is also surfaced in a dedicated "Reference conflicts" list for human review (never silently trusted) — see the dashboard's v1-vs-v2 comparison page.

## Official label = derived label

Judge-v1 stored `label` (the judge's own self-reported JSON field) as authoritative and `derived_label` (recomputed deterministically from the claims list) as a diagnostic, with `consistent = (label == derived_label)`.

For judge-v2, the decision is reversed: the **derived label is now authoritative**. To avoid ever confusing which field means what across the two versions, judge-v2's output records store it the other way round:

- `label` — the OFFICIAL, deterministic label (== `derived_label`, always, by construction).
- `derived_label` — kept as an explicit, separate field equal to `label`, per an explicit requirement that a reader never has to remember "label means something different in v1 vs v2."
- `judge_reported_label` — what the judge itself said in its own JSON `"label"` field. Used only to compute `consistent` (the consistency rate) and `attempted_claims_conflict`.
- `consistent` — `judge_reported_label == label`.
- `attempted_claims_conflict` — `true` if `answer_attempted` was `false` but the judge listed claims anyway (rule 0 still wins and the official label is still ABSTAIN; this flag surfaces the judge's internal inconsistency for review).

Every downstream summary/agreement/disagreement reader (`judge_agreement.py`) treats `label` as authoritative — unchanged code, because judge-v2's `label` field is already the right value by construction.

## Validation discipline

Judge-v2 will be validated on a **separate development set** (new questions, not drawn from the 9-run pilot) and against **human annotation**, exactly as judge-v1 was. **None of judge-v2's design decisions were tuned against the pilot questions' labels** — the pilot's disagreements were used only to identify *categories* of prompt-design problems (the six above), and the fix targets those categories generically (e.g. "exclude meta-statements about sources" applies to every future question, not specifically to Q38118194). The pilot questions remain a held-out test set for judge-v2 exactly as they were for judge-v1; re-judging them with v2 (which the user will trigger from the dashboard, not this task) is a validation run, not a tuning run.

## What never changed

- `judge_prompt_v1.py`, `llm_judge_hallucination_v1.py`, and every existing `jv1_dashboard_*.jsonl` output file: untouched.
- The legacy (non-`_v1`) faithfulness judge: untouched.
- Blinding (`blind_candidate()`, `BLINDING_VERSION = "blind-v2"`): reused as-is, not redefined.
- The question/reference/candidate framing (`USER_TEMPLATE`): identical to v1 — only the judge's own decision procedure (`SYSTEM_PROMPT`) changed.
- Shared runner plumbing (retry, resume, failures/resolution logging, served-model check, plan-table cost estimation) was refactored into shared helpers (`_judge_common.py`, `engine_service.py`'s generic plan/dashboard-driver functions) that judge-v1 and judge-v2 now both call — proven behavior-preserving for v1 by v1's own existing test suite passing unchanged throughout the refactor.
