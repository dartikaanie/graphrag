# Stage 1 recheck with answerable-only semantic expansion (Amendment 9; dev set, ctxrel-v2 primary)

- Time: 20261018T042953Z  |  git commit: `38c4f6fa88bca2597f225c65bf34401d9bcb01d2`  |  duration 199.4 s
- Reproduction (switch off) vs stage-1 result of record (Amendment 6): identical (50 x 5); recomputed metrics identical to the recorded ones
- New items judged: 119  |  judge model: meta-llama/Llama-3.3-70B-Instruct  |  estimated cost $0.0141

| alpha | % q >=1 RELEVANT (old -> new) | mean score (old -> new) | % accepted (old -> new) | questions changed |
|---|---|---|---|---|
| 0.0 | 48.0 -> 50.0 | 0.3263 -> 0.3525 | 37.29 -> 38.52 | 31 / 50 |
| 0.25 | 48.0 -> 48.0 | 0.3305 -> 0.3504 | 41.95 -> 47.13 | 34 / 50 |
| 0.5 | 48.0 -> 48.0 | 0.3157 -> 0.3299 | 53.39 -> 61.89 | 38 / 50 |
| 0.75 | 46.0 -> 44.0 | 0.2818 -> 0.2848 | 66.95 -> 76.23 | 35 / 50 |
| 1.0 | 34.0 -> 28.0 | 0.2225 -> 0.2029 | 72.88 -> 85.66 | 33 / 50 |

- Selection old: top_two [0.5, 0.25] (keep_current False)
- Selection new: top_two [0.25, 0.0] (keep_current False)
- New reasoning: Ranked by primary criterion (% questions with >=1 RELEVANT item; tied within 2.0 pp), then mean relevance score among primary ties (tied within 0.02), then larger alpha among ties on both. Full ranking: 0.25, 0.0, 0.5, 0.75, 1.0. alpha=0.25 (48.0%, mean score 0.3504) and alpha=0.0 (50.0%, mean score 0.3525) advance to stage 2.

## Retrieval per alpha (old -> new)

- alpha=0.0: empty expansion 64.1% -> 0.0%; expansion items in top-5 43 -> 90; final_k total 236 -> 244 (<5: 5 -> 2); latency p95 0.684 -> 0.241 s
- alpha=0.25: empty expansion 64.1% -> 0.0%; expansion items in top-5 37 -> 88; final_k total 236 -> 244 (<5: 5 -> 2); latency p95 0.1 -> 0.102 s
- alpha=0.5: empty expansion 64.1% -> 0.0%; expansion items in top-5 36 -> 82; final_k total 236 -> 244 (<5: 5 -> 2); latency p95 0.091 -> 0.106 s
- alpha=0.75: empty expansion 64.1% -> 0.0%; expansion items in top-5 37 -> 83; final_k total 236 -> 244 (<5: 5 -> 2); latency p95 0.113 -> 0.119 s
- alpha=1.0: empty expansion 64.1% -> 0.0%; expansion items in top-5 42 -> 92; final_k total 236 -> 244 (<5: 5 -> 2); latency p95 0.1 -> 0.123 s
