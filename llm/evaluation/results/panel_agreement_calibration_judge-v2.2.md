# Panel agreement report -- calibration set

Generated: 2026-10-19T14:10:20.329547+00:00. Unweighted Cohen's kappa, 95% CI = percentile bootstrap (2,000 resamples, fixed seed). Confusion matrices: rows = human, columns = judge. The panel composition is fixed; per-judge rows are informational.

## Acceptance: 3-category kappa (ABSTAIN / hallucinated / not hallucinated)

Rule (Amendment 11, Amendment 12 (e)): panel 3-category kappa (ABSTAIN / hallucinated / not hallucinated) >= 0.6, as pre-registered in Amendment 11 (Amendment 12 (e)); binary metrics and per-judge results are supporting statistics and do not decide.

**Panel 3-category kappa = 0.5508; target met: False.**

| rater | n | missing | raw agreement | 3-category kappa | 95% CI |
|---|---|---|---|---|---|
| llama | 60 | 0 | 46/60 (76.7%) | 0.4859 | [0.2674, 0.6747] |
| deepseek | 54 | 6 | 41/54 (75.9%) | 0.5439 | [0.2992, 0.7398] |
| qwen | 58 | 2 | 45/58 (77.6%) | 0.5046 | [0.284, 0.6999] |
| panel | 59 | 1 | 47/59 (79.7%) | 0.5508 | [0.3287, 0.731] |

llama:

| human \ judge | ABSTAIN | NOT_HALLUCINATED | HALLUCINATED |
|---|---|---|---|
| ABSTAIN | 8 | 0 | 0 |
| NOT_HALLUCINATED | 0 | 37 | 0 |
| HALLUCINATED | 0 | 14 | 1 |

deepseek:

| human \ judge | ABSTAIN | NOT_HALLUCINATED | HALLUCINATED |
|---|---|---|---|
| ABSTAIN | 8 | 0 | 0 |
| NOT_HALLUCINATED | 1 | 28 | 4 |
| HALLUCINATED | 0 | 8 | 5 |

qwen:

| human \ judge | ABSTAIN | NOT_HALLUCINATED | HALLUCINATED |
|---|---|---|---|
| ABSTAIN | 8 | 0 | 0 |
| NOT_HALLUCINATED | 0 | 36 | 0 |
| HALLUCINATED | 0 | 13 | 1 |

panel:

| human \ judge | ABSTAIN | NOT_HALLUCINATED | HALLUCINATED |
|---|---|---|---|
| ABSTAIN | 8 | 0 | 0 |
| NOT_HALLUCINATED | 0 | 37 | 0 |
| HALLUCINATED | 0 | 12 | 2 |

# Supporting statistics (do not decide acceptance)

## Judge vs human -- ABSTAIN excluded

| rater | n | missing | raw agreement | kappa | 95% CI | recall | unsafe pass | false alarm |
|---|---|---|---|---|---|---|---|---|
| llama | 52 | 0 | 38/52 (73.1%) | 0.0923 | [0.0, 0.2877] | 1/15 (6.7%) | 14/15 (93.3%) | 0/37 (0.0%) |
| deepseek | 45 | 6 | 33/45 (73.3%) | 0.2857 | [-0.0318, 0.5683] | 5/13 (38.5%) | 8/13 (61.5%) | 4/32 (12.5%) |
| qwen | 50 | 2 | 37/50 (74.0%) | 0.0997 | [0.0, 0.3191] | 1/14 (7.1%) | 13/14 (92.9%) | 0/36 (0.0%) |
| panel | 51 | 1 | 39/51 (76.5%) | 0.1947 | [0.0, 0.4333] | 2/14 (14.3%) | 12/14 (85.7%) | 0/37 (0.0%) |

llama:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 37 | 0 |
| HALLUCINATED | 14 | 1 |

deepseek:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 28 | 4 |
| HALLUCINATED | 8 | 5 |

qwen:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 36 | 0 |
| HALLUCINATED | 13 | 1 |

panel:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 37 | 0 |
| HALLUCINATED | 12 | 2 |

## Judge vs human -- ABSTAIN counted as not hallucinated

| rater | n | missing | raw agreement | kappa | 95% CI | recall | unsafe pass | false alarm |
|---|---|---|---|---|---|---|---|---|
| llama | 60 | 0 | 46/60 (76.7%) | 0.0968 | [0.0, 0.2941] | 1/15 (6.7%) | 14/15 (93.3%) | 0/45 (0.0%) |
| deepseek | 54 | 6 | 42/54 (77.8%) | 0.3208 | [0.0116, 0.5948] | 5/13 (38.5%) | 8/13 (61.5%) | 4/41 (9.8%) |
| qwen | 58 | 2 | 45/58 (77.6%) | 0.1045 | [0.0, 0.3256] | 1/14 (7.1%) | 13/14 (92.9%) | 0/44 (0.0%) |
| panel | 59 | 1 | 47/59 (79.7%) | 0.2027 | [0.0, 0.4587] | 2/14 (14.3%) | 12/14 (85.7%) | 0/45 (0.0%) |

llama:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 45 | 0 |
| HALLUCINATED | 14 | 1 |

deepseek:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 37 | 4 |
| HALLUCINATED | 8 | 5 |

qwen:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 44 | 0 |
| HALLUCINATED | 13 | 1 |

panel:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 45 | 0 |
| HALLUCINATED | 12 | 2 |

## ABSTAIN agreement (ABSTAIN yes/no)

| rater | n | raw agreement | kappa | 95% CI | ABSTAIN human | ABSTAIN rater |
|---|---|---|---|---|---|---|
| llama | 60 | 60/60 (100.0%) | 1.0 | [1.0, 1.0] | 8 | 8 |
| deepseek | 54 | 53/54 (98.2%) | 0.9302 | [0.7395, 1.0] | 8 | 9 |
| qwen | 58 | 58/58 (100.0%) | 1.0 | [1.0, 1.0] | 8 | 8 |
| panel | 59 | 59/59 (100.0%) | 1.0 | [1.0, 1.0] | 8 | 8 |

## Pairwise judge agreement (binary)

| pair | mode | n | raw agreement | kappa | 95% CI |
|---|---|---|---|---|---|
| llama_vs_deepseek | abstain_excluded | 45 | 37/45 (82.2%) | 0.1667 | [0.0, 0.4643] |
| llama_vs_deepseek | abstain_counted | 54 | 46/54 (85.2%) | 0.1724 | [0.0, 0.5055] |
| llama_vs_qwen | abstain_excluded | 50 | 48/50 (96.0%) | -0.0204 | [-0.0504, 0.0] |
| llama_vs_qwen | abstain_counted | 58 | 56/58 (96.5%) | -0.0175 | [-0.0432, 0.0] |
| deepseek_vs_qwen | abstain_excluded | 43 | 36/43 (83.7%) | 0.1887 | [0.0, 0.5409] |
| deepseek_vs_qwen | abstain_counted | 52 | 45/52 (86.5%) | 0.1947 | [0.0, 0.5649] |

## n per condition

```json
{
  "A [dev]": {
    "n_items": 20,
    "n_human_labelled": 20,
    "human_labels": {
      "NOT_HALLUCINATED": 12,
      "HALLUCINATED": 8
    },
    "n_llama_usable": 20,
    "n_deepseek_usable": 20,
    "n_qwen_usable": 19,
    "n_panel_usable": 20
  },
  "B-grounded [dev]": {
    "n_items": 20,
    "n_human_labelled": 20,
    "human_labels": {
      "ABSTAIN": 8,
      "NOT_HALLUCINATED": 11,
      "HALLUCINATED": 1
    },
    "n_llama_usable": 20,
    "n_deepseek_usable": 18,
    "n_qwen_usable": 20,
    "n_panel_usable": 20
  },
  "B-plain [dev]": {
    "n_items": 20,
    "n_human_labelled": 20,
    "human_labels": {
      "HALLUCINATED": 6,
      "NOT_HALLUCINATED": 14
    },
    "n_llama_usable": 20,
    "n_deepseek_usable": 16,
    "n_qwen_usable": 19,
    "n_panel_usable": 19
  }
}
```
