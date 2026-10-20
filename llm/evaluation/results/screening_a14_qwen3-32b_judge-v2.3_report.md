# Screening: Qwen/Qwen3-32B (judge-v2.3, 60 calibration items)

**Eligible: NO** (rule: reports/amendment14_draft.md)

| Criterion | Result | Pass |
|---|---|---|
| (1) HALLUCINATED >= 5% of non-ABSTAIN | 6/52 = 0.115 | True |
| (2) candidate text as [R] quote | 1 violations in 20 quotes (16 in neither text) | False |
| (3) usable, no reasoning | 60/60 usable; reasoning 0; <think> 0; first-attempt failures 0 | True |
| validity: served-model mismatch | 0 | True |

Cost $0.0399; latency {'p50': 35.94, 'p95': 52.632, 'n': 60}

## Agreement with the frozen v1 labels (information only)

| Rater | 3-cat kappa | binary kappa (excl.) | recall (excl.) | false alarm (excl.) | recall per condition |
|---|---|---|---|---|---|
| qwen3-32b | 0.484 | 0.145 | 3/15 | 3/37 | A [dev] 1/8, B-grounded [dev] 0/1, B-plain [dev] 2/6 |
| llama (stored) | 0.529 | 0.180 | 2/15 | 0/37 | A [dev] 1/8, B-grounded [dev] 0/1, B-plain [dev] 1/6 |
| deepseek (stored) | 0.566 | 0.329 | 5/15 | 2/36 | A [dev] 1/8, B-grounded [dev] 1/1, B-plain [dev] 3/6 |
| qwen (stored) | 0.442 | 0.000 | 0/15 | 0/37 | A [dev] 0/8, B-grounded [dev] 0/1, B-plain [dev] 0/6 |
| panel | 0.512 | 0.182 | 3/15 | 2/37 | A [dev] 1/8, B-grounded [dev] 0/1, B-plain [dev] 2/6 |

`panel` = panel-v2 (stored Llama + stored DeepSeek + the screened judge, 2-of-3).

## panel-v2 from stored outputs (no extra calls)

- labels: {'FAKTUAL': 47, 'ABSTAIN': 8, 'HALUSINASI_PENUH': 5}
- UNRESOLVED: 0; incomplete panels: 0
- Fleiss' kappa, 3 categories (complete items): 0.789 (n = 60)
- Fleiss' kappa, binary, ABSTAIN excluded: 0.557 (n = 51)
