# Exposure of hallucinated vs non-hallucinated answers to low-trust context

Official C run (alpha 0.5, n=384, 5 items each). Join check passed: 70 hallucinated (panel-v2 PENUH or SEBAGIAN), 242 not hallucinated (FAKTUAL), 72 ABSTAIN (reported separately, not pooled). Effect = mean(hallucinated) - mean(not hallucinated) per question; bootstrap 95% CI (questions resampled within group, 2000 draws, seed 42); permutation p (2000 shuffles). Full table, all four thresholds and the expansion-excluded variants: `exposure_table.csv`. Script: `exposure.py`.

**Share of a question's 5 context items with trust_weight < T** (hallucinated / not halluc. / abstain; difference in percentage points with CI, perm p)

| T | all items | difference (CI), p | excluding expansion items | difference (CI), p |
|---|---|---|---|---|
| 0.3 | 3.7% / 6.3% / 5.6% | -2.6 (-4.9, -0.1), p=0.10 | 2.6% / 3.3% / 2.2% | -0.7 (-2.8, +1.5), p=0.63 |
| 0.4 | 4.3% / 7.5% / 6.7% | -3.2 (-5.9, -0.5), p=0.054 | 3.1% / 4.5% / 3.3% | -1.3 (-3.4, +1.1), p=0.35 |
| 0.5 | 39.1% / 41.1% / 41.1% | -1.9 (-6.6, +2.7), p=0.47 | 19.7% / 21.2% / 20.8% | -1.4 (-7.2, +3.7), p=0.61 |
| 0.6 | 61.1% / 59.4% / 62.8% | +1.7 (-4.6, +7.8), p=0.60 | 41.7% / 39.5% / 42.5% | +2.2 (-4.9, +8.9), p=0.52 |

**Questions with at least one low-trust item:** at T=0.5, 94.3% vs 93.8% (+0.5 pp, CI -5.6 to +6.2); at T=0.4, 20.0% vs 30.6% (-10.6 pp, CI -21.0 to -0.1, p=0.10); at T=0.6, 98.6% vs 97.9%. Excluding expansion items at T=0.5: 60.0% vs 68.2% (-8.2 pp, CI -21.7 to +4.5).

**Means:** context trust_weight 0.622 vs 0.616 (+0.006, CI -0.023 to +0.036); similarity 0.602 vs 0.600 (+0.002, CI -0.018 to +0.023). Abstain: 0.608 and 0.551.

## What the numbers show
At no threshold are low-trust items over-represented in the contexts of hallucinated answers. Where the difference is visible (T=0.3 and 0.4), hallucinated answers have *fewer* low-trust items, and that gap disappears once expansion items are excluded, so it comes from expansion items (their proxy weight is capped at 0.4, which is why T=0.3/0.4 mostly count expansion items). At T=0.5 and 0.6 the groups are indistinguishable; mean trust and mean similarity are the same. With the like-for-like comparison, the earlier "40.7% vs 8.6%" contrast is replaced by 39.1% vs 41.1% at the same threshold (T=0.5).

## What it does NOT show
It cannot show that trust has no effect on correctness: the groups are unbalanced (70 vs 242), the comparison is observational (question difficulty, topic and the retrieval pool differ between questions), 26 metrics were computed without multiplicity correction (the T=0.4 result with p=0.054 should not be read as a finding), and exposure to low-trust items says nothing about whether the model used them.
