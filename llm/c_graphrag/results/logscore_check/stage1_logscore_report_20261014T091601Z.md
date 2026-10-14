# Stage 1 recheck with trustScore v3-logscore (dev set, ctxrel-v2 primary)

- Time: 20261014T091601Z  |  git commit: `e5394238049cf2352d9ae4b9fa6da6bfd7682906`
- Reproduction v2 vs stage-1 result of record (Amendment 5): identical (50 x 5)
- New items judged: 3  |  judge model: meta-llama/Llama-3.3-70B-Instruct  |  P99 Answer score: 46.0  |  duration 98.6 s

| alpha | % q >=1 RELEVANT (v2 -> v3) | mean score (v2 -> v3) | % accepted (v2 -> v3) |
|---|---|---|---|
| 0.0 | 48.0 -> 48.0 | 0.3263 -> 0.3263 | 37.29 -> 37.29 |
| 0.25 | 48.0 -> 48.0 | 0.3242 -> 0.3305 | 41.95 -> 41.95 |
| 0.5 | 48.0 -> 48.0 | 0.3178 -> 0.3157 | 52.12 -> 53.39 |
| 0.75 | 46.0 -> 46.0 | 0.286 -> 0.2818 | 67.8 -> 66.95 |
| 1.0 | 36.0 -> 34.0 | 0.2225 -> 0.2225 | 74.58 -> 72.88 |

- Selection helper v2: top_two [0.5, 0.25] (keep_current False)
- Selection helper v3: top_two [0.5, 0.25] (keep_current False)
- v3 reasoning: Top two by primary criterion (% questions with >=1 RELEVANT item), tie-broken by mean relevance score then larger alpha: alpha=0.5 (48.0%), alpha=0.25 (48.0%) advance to stage 2.

## v2 -> v3 per alpha (descriptive)

- alpha=0.0: delta % q = 0.0, delta score = 0.0
- alpha=0.25: delta % q = 0.0, delta score = 0.0063
- alpha=0.5: delta % q = 0.0, delta score = -0.0021
- alpha=0.75: delta % q = 0.0, delta score = -0.0042
- alpha=1.0: delta % q = -2.0, delta score = 0.0
