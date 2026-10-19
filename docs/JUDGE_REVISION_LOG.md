# Judge revision log

The hallucination judge's prompt versions (docs/DECISION_C_SCORING.md, Amendment 11, change 3).

- judge-v2.2 is the base version before calibration. It is not a revision.
- At most three revisions may follow (judge-v2.3, judge-v2.4, judge-v2.5). The count starts at the first run of judge-v2.2 against the human labels.
- Every version is committed and logged here before it is run on calibration data. The calibration runner (`llm/evaluation/judge_versions.py`) refuses a version that is not in this table, and refuses a fourth version after judge-v2.2.
- κ is the unweighted 3-category Cohen's κ, primary judge vs researcher, on the 60 calibration items, with its bootstrap 95% CI (`annotation_tool.py agreement`).
- From Amendment 12 on, κ is the PANEL's κ vs the frozen researcher labels (`judge_panel_agreement.py`). From Amendment 13 on (judge-v2.3 and later), a version is accepted only if the panel 3-category κ ≥ 0.6 AND the panel binary κ (ABSTAIN excluded) ≥ 0.4; both are logged.

| version | date | commit | reason | κ before (95% CI) | κ after (95% CI) |
|---|---|---|---|---|---|
| judge-v2.2 | 2026-10-19 | 79b907c | Base version (Amendment 11): retrieval context, supported_by_context, cited_item_supports, approach_match, blind-v3. Not a revision. | – | panel-v1 (Amendment 12), run 35fab00: 3-category κ 0.551 (95% CI 0.329–0.731); panel binary κ 0.195 (ABSTAIN excluded). Below the 0.6 target. |
| judge-v2.3 | 2026-10-19 | db12f5e | Iteration 2, revision 1 of 3 after judge-v2.2. Prompt changes #1+#2 (always together), #3, #4, #5; JSON mode (response_format json_object) for all three judges; parser pv2.3-1 (supported_by_context "N/A" with context → null). Details and rationale below. | judge-v2.2: 3-category 0.551 (0.329–0.731); binary (ABSTAIN excl.) 0.195 | panel-v1, run cc140db: 3-category κ 0.529; panel binary κ 0.180 (ABSTAIN excluded); recall 2/15. Fails both Amendment 13 conditions. |

## Iteration 2: judge-v2.3 (2026-10-19)

**Design caveat.** These changes were designed using the calibration set: the iteration-1 miss diagnosis read the judges' outputs on the 60 calibration items against the frozen human labels. The calibration κ of judge-v2.3 is therefore expected to be optimistic. The 63-item confirmation set (Amendment 11, change 5), annotated blind before its judge labels are viewed, provides the out-of-sample estimate.

**Unchanged:** panel composition and aggregation (Amendment 12), blind-v3 (Amendment 12 (h)), input template and context block, temperature 0, max_tokens 2000, the frozen human labels (`annotator1_freeze.json`), the acceptance rule (Amendment 13). judge-v2.2 stays available and unchanged (`judge_prompt_v2_2.py`; `judge_panel.py --judge-version judge-v2.2`).

**Iteration-1 diagnosis** (12 human-hallucinated items the panel passed; panel-level buckets): (a) wrong claim not extracted 2; (b) extracted but judged UNVERIFIABLE 6; (c) extracted but judged SUPPORTED 1; (d) flagged by one judge but outvoted 2; flagged but lost to a parse failure 1. DeepSeek also raised 4 false alarms, 3 of them treating a different dependency version as a contradiction.

| change | what | targets | rationale |
|---|---|---|---|
| #1 | Explicit conflict with the reference ("not possible", "only X is supported", "use A, not B") → CONTRADICTED, not UNVERIFIABLE; covering the same topic is not enough | (b), (c) | Several misses were claims the reference explicitly rules out but the judges marked UNVERIFIABLE or SUPPORTED. |
| #2 | Always with #1: a different version/library/package/setting/approach is not a contradiction unless the reference or well-established fact says the candidate's choice fails | guard against new false alarms | Iteration-1 false alarms were exactly this over-application; #1 is never used without #2. |
| #3 | Every identifier (type, namespace, method, enum/event name, command, flag, SQL/config keyword) is its own claim; a confident near-miss of a real name → FABRICATED (CONTRADICTED for rejected syntax), otherwise UNVERIFIABLE | (a), (b) | Misses on wrong API/type/event names and syntax were not extracted or were left UNVERIFIABLE. |
| #4 | Within the 8-claim limit for SUPPORTED/UNVERIFIABLE claims: identifiers/commands/syntax/config values first, then the main fix and the stated cause, general advice last | (a) | The limit was reached and the wrong claim left out in several judge outputs. |
| #5 | SUPPORTED/CONTRADICTED from the reference must quote it ([R] "..."); restating the candidate is never support | (c) | Several SUPPORTED verdicts had circular or misread evidence. |
| JSON mode | `response_format: {"type": "json_object"}` for all three judges | parse failures | One iteration-1 failure was invalid JSON (unescaped quotes); retries at temperature 0 repeated it. |
| pv2.3-1 | `supported_by_context` "N/A" on an item with context → null, recorded as `supported_by_context_na_with_context`; faithfulness counts non-null values only; verdicts untouched | parse failures | 7 of 8 iteration-1 parse failures, with complete hallucination verdicts, were lost to this field. |

**Amendment within iteration 2, before any judge-v2.3 run (6036213, 2026-10-19).** Example (l) first reused version numbers that occur in calibration items: 1.0.1 and 11.0.1 (the reference of CAL-016/017/058) and 1.1.0 (candidates of CAL-016/017/032/040/058). They were replaced by 2.4.0, 7.3.2 and 2.6.0, which occur in no calibration item. Check performed on the full texts of all 60 calibration items (question with body, reference, blinded and original candidate, context): no identifier or version of examples (k)-(m) occurs in any item (the word "auto" of (k) occurs only in unrelated senses, e.g. CSS `height: auto`, Android `res-auto`), and no 5-word phrase of the examples occurs in any item. The judge-v2.3 table row's commit db12f5e is the original version; 6036213 is the version that runs.
