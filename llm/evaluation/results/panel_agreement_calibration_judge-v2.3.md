# Panel agreement report -- calibration set

Generated: 2026-10-19T15:26:59.601169+00:00. Unweighted Cohen's kappa, 95% CI = percentile bootstrap (2,000 resamples, fixed seed). Confusion matrices: rows = human, columns = judge. The panel composition is fixed; per-judge rows are informational.

## Acceptance: 3-category kappa (ABSTAIN / hallucinated / not hallucinated)

Rule (Amendment 11, Amendment 12 (e)): panel 3-category kappa (ABSTAIN / hallucinated / not hallucinated) >= 0.6 (Amendment 11, 12 (e)) AND panel binary kappa, ABSTAIN excluded, >= 0.4 (Amendment 13); the other binary metrics and per-judge results are supporting statistics.

**Panel 3-category kappa = 0.529 (>= 0.6: False); panel binary kappa, ABSTAIN excluded = 0.1796 (>= 0.4: False). Accepted: False.**

| rater | n | missing | raw agreement | 3-category kappa | 95% CI |
|---|---|---|---|---|---|
| llama | 60 | 0 | 47/60 (78.3%) | 0.529 | [0.3137, 0.7156] |
| deepseek | 60 | 0 | 47/60 (78.3%) | 0.5655 | [0.3424, 0.7576] |
| qwen | 60 | 0 | 45/60 (75.0%) | 0.4417 | [0.2241, 0.6324] |
| panel | 60 | 0 | 47/60 (78.3%) | 0.529 | [0.3137, 0.7156] |
| SENSITIVITY: llama | 60 | 0 | 46/60 (76.7%) | 0.507 | [0.298, 0.6923] |
| SENSITIVITY: deepseek | 60 | 0 | 46/60 (76.7%) | 0.5415 | [0.3191, 0.7319] |
| SENSITIVITY: qwen | 60 | 0 | 44/60 (73.3%) | 0.4231 | [0.2143, 0.6141] |
| SENSITIVITY: panel | 60 | 0 | 46/60 (76.7%) | 0.507 | [0.298, 0.6923] |

SENSITIVITY rows: SENSITIVITY: human labels with the post-freeze edits (docs/ANNOTATION_POSTFREEZE_RECONSIDERATIONS.md); saved after the iteration-1 run, post-dating the freeze; not used for acceptance. Edited items: CAL-010, CAL-028, CAL-041.

llama:

| human \ judge | ABSTAIN | NOT_HALLUCINATED | HALLUCINATED |
|---|---|---|---|
| ABSTAIN | 8 | 0 | 0 |
| NOT_HALLUCINATED | 0 | 37 | 0 |
| HALLUCINATED | 0 | 13 | 2 |

deepseek:

| human \ judge | ABSTAIN | NOT_HALLUCINATED | HALLUCINATED |
|---|---|---|---|
| ABSTAIN | 8 | 0 | 0 |
| NOT_HALLUCINATED | 1 | 34 | 2 |
| HALLUCINATED | 0 | 10 | 5 |

qwen:

| human \ judge | ABSTAIN | NOT_HALLUCINATED | HALLUCINATED |
|---|---|---|---|
| ABSTAIN | 8 | 0 | 0 |
| NOT_HALLUCINATED | 0 | 37 | 0 |
| HALLUCINATED | 0 | 15 | 0 |

panel:

| human \ judge | ABSTAIN | NOT_HALLUCINATED | HALLUCINATED |
|---|---|---|---|
| ABSTAIN | 8 | 0 | 0 |
| NOT_HALLUCINATED | 0 | 37 | 0 |
| HALLUCINATED | 0 | 13 | 2 |

# Supporting statistics (do not decide acceptance)

## Judge vs human -- ABSTAIN excluded

| rater | n | missing | raw agreement | kappa | 95% CI | recall | unsafe pass | false alarm |
|---|---|---|---|---|---|---|---|---|
| llama | 52 | 0 | 39/52 (75.0%) | 0.1796 | [0.0, 0.4091] | 2/15 (13.3%) | 13/15 (86.7%) | 0/37 (0.0%) |
| deepseek | 51 | 0 | 39/51 (76.5%) | 0.3289 | [0.0445, 0.5995] | 5/15 (33.3%) | 10/15 (66.7%) | 2/36 (5.6%) |
| qwen | 52 | 0 | 37/52 (71.2%) | 0.0 | [0.0, 0.0] | 0/15 (0.0%) | 15/15 (100.0%) | 0/37 (0.0%) |
| panel | 52 | 0 | 39/52 (75.0%) | 0.1796 | [0.0, 0.4091] | 2/15 (13.3%) | 13/15 (86.7%) | 0/37 (0.0%) |
| SENSITIVITY: llama | 52 | 0 | 38/52 (73.1%) | 0.1651 | [0.0, 0.3863] | 2/16 (12.5%) | 14/16 (87.5%) | 0/36 (0.0%) |
| SENSITIVITY: deepseek | 51 | 0 | 38/51 (74.5%) | 0.3014 | [0.0325, 0.5723] | 5/16 (31.2%) | 11/16 (68.8%) | 2/35 (5.7%) |
| SENSITIVITY: qwen | 52 | 0 | 36/52 (69.2%) | 0.0 | [0.0, 0.0] | 0/16 (0.0%) | 16/16 (100.0%) | 0/36 (0.0%) |
| SENSITIVITY: panel | 52 | 0 | 38/52 (73.1%) | 0.1651 | [0.0, 0.3863] | 2/16 (12.5%) | 14/16 (87.5%) | 0/36 (0.0%) |

llama:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 37 | 0 |
| HALLUCINATED | 13 | 2 |

deepseek:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 34 | 2 |
| HALLUCINATED | 10 | 5 |

qwen:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 37 | 0 |
| HALLUCINATED | 15 | 0 |

panel:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 37 | 0 |
| HALLUCINATED | 13 | 2 |

## Judge vs human -- ABSTAIN counted as not hallucinated

| rater | n | missing | raw agreement | kappa | 95% CI | recall | unsafe pass | false alarm |
|---|---|---|---|---|---|---|---|---|
| llama | 60 | 0 | 47/60 (78.3%) | 0.1875 | [0.0, 0.4141] | 2/15 (13.3%) | 13/15 (86.7%) | 0/45 (0.0%) |
| deepseek | 60 | 0 | 48/60 (80.0%) | 0.3514 | [0.0551, 0.625] | 5/15 (33.3%) | 10/15 (66.7%) | 2/45 (4.4%) |
| qwen | 60 | 0 | 45/60 (75.0%) | 0.0 | [0.0, 0.0] | 0/15 (0.0%) | 15/15 (100.0%) | 0/45 (0.0%) |
| panel | 60 | 0 | 47/60 (78.3%) | 0.1875 | [0.0, 0.4141] | 2/15 (13.3%) | 13/15 (86.7%) | 0/45 (0.0%) |
| SENSITIVITY: llama | 60 | 0 | 46/60 (76.7%) | 0.1732 | [0.0, 0.3868] | 2/16 (12.5%) | 14/16 (87.5%) | 0/44 (0.0%) |
| SENSITIVITY: deepseek | 60 | 0 | 47/60 (78.3%) | 0.3253 | [0.0476, 0.5833] | 5/16 (31.2%) | 11/16 (68.8%) | 2/44 (4.5%) |
| SENSITIVITY: qwen | 60 | 0 | 44/60 (73.3%) | 0.0 | [0.0, 0.0] | 0/16 (0.0%) | 16/16 (100.0%) | 0/44 (0.0%) |
| SENSITIVITY: panel | 60 | 0 | 46/60 (76.7%) | 0.1732 | [0.0, 0.3868] | 2/16 (12.5%) | 14/16 (87.5%) | 0/44 (0.0%) |

llama:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 45 | 0 |
| HALLUCINATED | 13 | 2 |

deepseek:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 43 | 2 |
| HALLUCINATED | 10 | 5 |

qwen:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 45 | 0 |
| HALLUCINATED | 15 | 0 |

panel:

| human \ judge | NOT_H | H |
|---|---|---|
| NOT_HALLUCINATED | 45 | 0 |
| HALLUCINATED | 13 | 2 |

## ABSTAIN agreement (ABSTAIN yes/no)

| rater | n | raw agreement | kappa | 95% CI | ABSTAIN human | ABSTAIN rater |
|---|---|---|---|---|---|---|
| llama | 60 | 60/60 (100.0%) | 1.0 | [1.0, 1.0] | 8 | 8 |
| deepseek | 60 | 59/60 (98.3%) | 0.9315 | [0.7429, 1.0] | 8 | 9 |
| qwen | 60 | 60/60 (100.0%) | 1.0 | [1.0, 1.0] | 8 | 8 |
| panel | 60 | 60/60 (100.0%) | 1.0 | [1.0, 1.0] | 8 | 8 |

## Pairwise judge agreement (binary)

| pair | mode | n | raw agreement | kappa | 95% CI |
|---|---|---|---|---|---|
| llama_vs_deepseek | abstain_excluded | 51 | 46/51 (90.2%) | 0.4084 | [0.0, 0.7901] |
| llama_vs_deepseek | abstain_counted | 60 | 55/60 (91.7%) | 0.4141 | [0.0, 0.7826] |
| llama_vs_qwen | abstain_excluded | 52 | 50/52 (96.2%) | 0.0 | [0.0, 0.0] |
| llama_vs_qwen | abstain_counted | 60 | 58/60 (96.7%) | 0.0 | [0.0, 0.0] |
| deepseek_vs_qwen | abstain_excluded | 51 | 44/51 (86.3%) | 0.0 | [0.0, 0.0] |
| deepseek_vs_qwen | abstain_counted | 60 | 53/60 (88.3%) | 0.0 | [0.0, 0.0] |

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
    "n_qwen_usable": 20,
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
    "n_deepseek_usable": 20,
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
    "n_deepseek_usable": 20,
    "n_qwen_usable": 20,
    "n_panel_usable": 20
  }
}
```
