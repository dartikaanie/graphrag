# Criterion (2) fairness check: stored judge-v2.3 calibration outputs

Source: `results/panel_calibration_judge-v2.3.jsonl`. Method: judge_screening.quote_check (criterion (2) of the eligibility rule); stored outputs only, no API call.

| Judge | Model | Usable records | [R] quotes | Violations | Affected items | Items | Quotes in neither text |
|---|---|---|---|---|---|---|---|
| deepseek | deepseek-ai/DeepSeek-V3.2 | 60 | 112 | 1 | 1 | CAL-033 | 32 |
| llama | meta-llama/Llama-3.3-70B-Instruct-Turbo | 60 | 0 | 0 | 0 | -- | 0 |
| qwen | Qwen/Qwen2.5-72B-Instruct | 60 | 75 | 8 | 4 | CAL-005, CAL-007, CAL-041, CAL-057 | 13 |

## Where the violating quotes come from (added by hand, 2026-10-20; information only)

Checked against the question title + body (questions parquet), stored data only. "Q" = the quoted text is also in
the question (the candidate answer repeated it); "candidate only" = found only in the candidate answer.

| Judge | Item | Quotes |
|---|---|---|
| deepseek (stored) | CAL-033 | 1 Q |
| qwen (Qwen2.5-72B, stored) | CAL-005 | 3 candidate only |
| qwen (Qwen2.5-72B, stored) | CAL-007 | 2 candidate only |
| qwen (Qwen2.5-72B, stored) | CAL-041 | 2 Q |
| qwen (Qwen2.5-72B, stored) | CAL-057 | 1 candidate only |
| qwen3-235b (screening) | CAL-033 | 2 Q |
| qwen3-32b (screening) | CAL-041 | 1 Q |

Llama-3.3 produced no `[R]` quote at all in the 60 records (0 quotes), so criterion (2) cannot fail for it.
