# Stage 1 Report -- C retrieval v3 alpha sweep

Generated: 2026-10-07T18:46:48.685748+00:00

Context-relevance judge version: ctxrel-v2

Code commit (run start, clean tree): 43b3b3b21a92a1a04a5db93531cf2cdf959fac59

Controls gate (not re-run): `llm/c_graphrag/results/controls/ctxrel_v2_controls_positive-easy_negative-hard_negative_20261007T165007Z_summary.json` (sha256 ae35d135b293fbf3..., produced at commit 1138eb8096f11ca88db2ca8e6bde13badfe386e3, clean tree; ctxrel-v2, passed per primary: positive 94% (>= 90%, n=50), easy_negative 100% (>= 90%, n=50), hard_negative 100% (>= 80%, n=47))

| alpha | % q w/ >=1 RELEVANT | % RELEVANT | % PARTIAL | % IRRELEVANT | mean score | mean sim (sel) | mean trust (sel) | % accepted (sel) | Jaccard vs a=0 | Jaccard vs a=1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 48.0 | 22.03 | 21.19 | 56.78 | 0.3263 | 0.5839 | 0.2949 | 37.29 | 1.0 | 0.3791 |
| 0.25 | 48.0 | 21.19 | 23.31 | 55.51 | 0.3284 | 0.5815 | 0.3292 | 41.95 | 0.879 | 0.416 |
| 0.5 | 48.0 | 19.49 | 23.31 | 57.2 | 0.3114 | 0.5691 | 0.3765 | 52.54 | 0.701 | 0.5148 |
| 0.75 | 46.0 | 17.8 | 21.61 | 60.59 | 0.286 | 0.5321 | 0.427 | 68.22 | 0.5402 | 0.6717 |
| 1.0 | 34.0 | 13.98 | 17.8 | 68.22 | 0.2288 | 0.4745 | 0.4435 | 74.58 | 0.3791 | 1.0 |

## Relevance by hop and edge type (per alpha)

**alpha=0.0**

| hop | n | % RELEVANT | % PARTIAL |
|---|---|---|---|
| 1 | 35 | 20.0 | 37.14 |
| 2 | 158 | 23.42 | 18.99 |
| semantic | 43 | 18.6 | 16.28 |

| edge type | n | % RELEVANT | % PARTIAL |
|---|---|---|---|
| EMBED_SIM->HAS_ACCEPTED_ANSWER | 52 | 23.08 | 32.69 |
| EMBED_SIM->HAS_ANSWER | 72 | 23.61 | 16.67 |
| HAS_ACCEPTED_ANSWER | 27 | 37.04 | 14.81 |
| HAS_ANSWER | 51 | 9.8 | 31.37 |
| IS_RELATED_TO->HAS_ANSWER | 8 | 62.5 | 0.0 |
| TAG_COOCCUR->HAS_ACCEPTED_ANSWER | 9 | 0.0 | 0.0 |
| TAG_COOCCUR->HAS_ANSWER | 17 | 17.65 | 5.88 |

**alpha=0.25**

| hop | n | % RELEVANT | % PARTIAL |
|---|---|---|---|
| 1 | 41 | 21.95 | 34.15 |
| 2 | 158 | 21.52 | 21.52 |
| semantic | 37 | 18.92 | 18.92 |

| edge type | n | % RELEVANT | % PARTIAL |
|---|---|---|---|
| EMBED_SIM->HAS_ACCEPTED_ANSWER | 55 | 23.64 | 32.73 |
| EMBED_SIM->HAS_ANSWER | 66 | 19.7 | 18.18 |
| HAS_ACCEPTED_ANSWER | 31 | 32.26 | 19.35 |
| HAS_ANSWER | 47 | 12.77 | 31.91 |
| IS_RELATED_TO->HAS_ACCEPTED_ANSWER | 2 | 0.0 | 100.0 |
| IS_RELATED_TO->HAS_ANSWER | 8 | 50.0 | 12.5 |
| TAG_COOCCUR->HAS_ACCEPTED_ANSWER | 11 | 0.0 | 0.0 |
| TAG_COOCCUR->HAS_ANSWER | 16 | 25.0 | 6.25 |

**alpha=0.5**

| hop | n | % RELEVANT | % PARTIAL |
|---|---|---|---|
| 1 | 43 | 25.58 | 32.56 |
| 2 | 159 | 17.61 | 20.75 |
| semantic | 34 | 20.59 | 23.53 |

| edge type | n | % RELEVANT | % PARTIAL |
|---|---|---|---|
| EMBED_SIM->HAS_ACCEPTED_ANSWER | 70 | 20.0 | 28.57 |
| EMBED_SIM->HAS_ANSWER | 49 | 18.37 | 18.37 |
| HAS_ACCEPTED_ANSWER | 35 | 28.57 | 22.86 |
| HAS_ANSWER | 42 | 19.05 | 33.33 |
| IS_RELATED_TO->HAS_ACCEPTED_ANSWER | 2 | 0.0 | 100.0 |
| IS_RELATED_TO->HAS_ANSWER | 5 | 40.0 | 20.0 |
| TAG_COOCCUR->HAS_ACCEPTED_ANSWER | 17 | 0.0 | 0.0 |
| TAG_COOCCUR->HAS_ANSWER | 16 | 18.75 | 6.25 |

**alpha=0.75**

| hop | n | % RELEVANT | % PARTIAL |
|---|---|---|---|
| 1 | 40 | 27.5 | 30.0 |
| 2 | 158 | 15.19 | 20.25 |
| semantic | 38 | 18.42 | 18.42 |

| edge type | n | % RELEVANT | % PARTIAL |
|---|---|---|---|
| EMBED_SIM->HAS_ACCEPTED_ANSWER | 75 | 18.67 | 30.67 |
| EMBED_SIM->HAS_ANSWER | 24 | 16.67 | 20.83 |
| HAS_ACCEPTED_ANSWER | 42 | 23.81 | 19.05 |
| HAS_ANSWER | 36 | 22.22 | 30.56 |
| IS_RELATED_TO->HAS_ACCEPTED_ANSWER | 3 | 33.33 | 66.67 |
| IS_RELATED_TO->HAS_ANSWER | 4 | 50.0 | 25.0 |
| TAG_COOCCUR->HAS_ACCEPTED_ANSWER | 41 | 0.0 | 0.0 |
| TAG_COOCCUR->HAS_ANSWER | 11 | 27.27 | 9.09 |

**alpha=1.0**

| hop | n | % RELEVANT | % PARTIAL |
|---|---|---|---|
| 1 | 29 | 31.03 | 27.59 |
| 2 | 164 | 11.59 | 17.07 |
| semantic | 43 | 11.63 | 13.95 |

| edge type | n | % RELEVANT | % PARTIAL |
|---|---|---|---|
| EMBED_SIM->HAS_ACCEPTED_ANSWER | 71 | 14.08 | 32.39 |
| EMBED_SIM->HAS_ANSWER | 15 | 26.67 | 6.67 |
| HAS_ACCEPTED_ANSWER | 46 | 19.57 | 17.39 |
| HAS_ANSWER | 26 | 19.23 | 23.08 |
| IS_RELATED_TO->HAS_ACCEPTED_ANSWER | 4 | 25.0 | 50.0 |
| IS_RELATED_TO->HAS_ANSWER | 3 | 33.33 | 33.33 |
| TAG_COOCCUR->HAS_ACCEPTED_ANSWER | 55 | 0.0 | 0.0 |
| TAG_COOCCUR->HAS_ANSWER | 16 | 18.75 | 6.25 |

## Selection helper output

keep_current: False
top_two (stage 2 candidates): [0.5, 0.25]
reasoning: Top two by primary criterion (% questions with >=1 RELEVANT item), tie-broken by mean relevance score then larger alpha: alpha=0.5 (48.0%), alpha=0.25 (48.0%) advance to stage 2.
