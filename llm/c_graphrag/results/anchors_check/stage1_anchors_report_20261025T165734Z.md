# Stage 1 recheck with answerable-only anchors (Amendment 17; C v4; dev set, ctxrel-v2 primary)

- Time: 20261025T165734Z | leakage `thread` | judge meta-llama/Llama-3.3-70B-Instruct | new items judged 254 | actual cost $0.02994 (estimate $0.0299)
- Previous stage 1 (C v3, plain anchors): stage1_expansion_report_20261025T160942Z.json

| alpha | % q >=1 RELEVANT (v3 anchors -> v4 anchors) | mean score (v3 -> v4) | % accepted (v3 -> v4) |
|---|---|---|---|
| 0.0 | 50.0 -> 56.0 | 0.3525 -> 0.376 | 38.52 -> 36.8 |
| 0.25 | 48.0 -> 56.0 | 0.3504 -> 0.374 | 47.13 -> 51.2 |
| 0.5 | 48.0 -> 58.0 | 0.3299 -> 0.372 | 61.89 -> 59.6 |
| 0.75 | 44.0 -> 56.0 | 0.2848 -> 0.332 | 76.23 -> 71.2 |
| 1.0 | 28.0 -> 40.0 | 0.2029 -> 0.256 | 85.66 -> 82.8 |

- Selection with v3 anchors: top_two [0.25, 0.0]
- Selection with v4 anchors: top_two [0.5, 0.25] (keep_current False)
- Reasoning: Ranked by primary criterion (% questions with >=1 RELEVANT item; tied within 2.0 pp), then mean relevance score among primary ties (tied within 0.02), then larger alpha among ties on both. Full ranking: 0.5, 0.25, 0.0, 0.75, 1.0. alpha=0.5 (58.0%, mean score 0.372) and alpha=0.25 (56.0%, mean score 0.374) advance to stage 2.

## Retrieval per alpha

- alpha=0.0: final_k total 250 (<5: 0, 0: 0); questions with <3 seeds 0; expansion items in top-5 63; latency p95 2.779 s
- alpha=0.25: final_k total 250 (<5: 0, 0: 0); questions with <3 seeds 0; expansion items in top-5 56; latency p95 1.006 s
- alpha=0.5: final_k total 250 (<5: 0, 0: 0); questions with <3 seeds 0; expansion items in top-5 52; latency p95 2.357 s
- alpha=0.75: final_k total 250 (<5: 0, 0: 0); questions with <3 seeds 0; expansion items in top-5 47; latency p95 1.277 s
- alpha=1.0: final_k total 250 (<5: 0, 0: 0); questions with <3 seeds 0; expansion items in top-5 57; latency p95 1.532 s
