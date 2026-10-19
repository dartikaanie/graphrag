# Judge revision log

The hallucination judge's prompt versions (docs/DECISION_C_SCORING.md, Amendment 11, change 3).

- judge-v2.2 is the base version before calibration. It is not a revision.
- At most three revisions may follow (judge-v2.3, judge-v2.4, judge-v2.5). The count starts at the first run of judge-v2.2 against the human labels.
- Every version is committed and logged here before it is run on calibration data. The calibration runner (`llm/evaluation/judge_versions.py`) refuses a version that is not in this table, and refuses a fourth version after judge-v2.2.
- κ is the unweighted 3-category Cohen's κ, primary judge vs researcher, on the 60 calibration items, with its bootstrap 95% CI (`annotation_tool.py agreement`).

| version | date | commit | reason | κ before (95% CI) | κ after (95% CI) |
|---|---|---|---|---|---|
| judge-v2.2 | 2026-10-19 | 79b907c | Base version (Amendment 11): retrieval context, supported_by_context, cited_item_supports, approach_match, blind-v3. Not a revision. | – | panel-v1 (Amendment 12), run 35fab00: 3-category κ 0.551 (95% CI 0.329–0.731); panel binary κ 0.195 (ABSTAIN excluded). Below the 0.6 target. |
