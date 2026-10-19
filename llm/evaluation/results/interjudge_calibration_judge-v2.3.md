# Inter-judge agreement -- calibration

## judge-v2.3

### Inter-judge kappa -- binary, ABSTAIN excluded (PRIMARY)

| pair | n | raw agreement | Cohen's kappa |
|---|---|---|---|
| llama vs deepseek | 51 | 46/51 (90.2%) | 0.408 |
| llama vs qwen | 52 | 50/52 (96.2%) | 0.000 |
| deepseek vs qwen | 51 | 44/51 (86.3%) | 0.000 |
| **Fleiss' kappa (panel, complete items)** | 51 | unanimous 44/51 (86.3%) | **0.174** |

Matrix (Cohen's kappa; raw agreement):

| | llama | deepseek | qwen |
|---|---|---|---|
| llama | -- | 0.408 (90.2%, n=51) | 0.000 (96.2%, n=52) |
| deepseek | 0.408 (90.2%, n=51) | -- | 0.000 (86.3%, n=51) |
| qwen | 0.000 (96.2%, n=52) | 0.000 (86.3%, n=51) | -- |

### Inter-judge kappa -- binary, ABSTAIN counted as not hallucinated

| pair | n | raw agreement | Cohen's kappa |
|---|---|---|---|
| llama vs deepseek | 60 | 55/60 (91.7%) | 0.414 |
| llama vs qwen | 60 | 58/60 (96.7%) | 0.000 |
| deepseek vs qwen | 60 | 53/60 (88.3%) | 0.000 |
| **Fleiss' kappa (panel, complete items)** | 60 | unanimous 53/60 (88.3%) | **0.181** |

### Inter-judge kappa -- 3-category (ABSTAIN / NOT_HALLUCINATED / HALLUCINATED)

| pair | n | raw agreement | Cohen's kappa |
|---|---|---|---|
| llama vs deepseek | 60 | 54/60 (90.0%) | 0.726 |
| llama vs qwen | 60 | 58/60 (96.7%) | 0.872 |
| deepseek vs qwen | 60 | 52/60 (86.7%) | 0.613 |
| **Fleiss' kappa (panel, complete items)** | 60 | unanimous 52/60 (86.7%) | **0.723** |

### Prevalence of "hallucinated" (and ABSTAIN) per rater

| rater | H, ABSTAIN excluded | H, ABSTAIN counted | ABSTAIN |
|---|---|---|---|
| llama | 2/52 (3.9%) | 2/60 (3.3%) | 8/60 (13.3%) |
| deepseek | 7/51 (13.7%) | 7/60 (11.7%) | 9/60 (15.0%) |
| qwen | 0/52 (0.0%) | 0/60 (0.0%) | 8/60 (13.3%) |
| human | 15/52 (28.8%) | 15/60 (25.0%) | 8/60 (13.3%) |

qwen labelled no item HALLUCINATED: every binary Cohen's kappa involving it is 0 by construction (expected agreement equals observed agreement on the other rater's NOT_HALLUCINATED items), even where raw agreement is high. The binary kappa values reflect the low prevalence of HALLUCINATED, not only disagreement.

### Severity distribution (no kappa)

| rater | FAKTUAL | SEBAGIAN | PENUH | UNSPECIFIED | ABSTAIN | missing |
|---|---|---|---|---|---|---|
| llama | 50 | 0 | 2 | 0 | 8 | 0 |
| deepseek | 44 | 0 | 7 | 0 | 9 | 0 |
| qwen | 52 | 0 | 0 | 0 | 8 | 0 |
| human | 37 | 9 | 6 | 0 | 8 | 0 |

Judge SEBAGIAN labels in this run: 0. Severity of a judge's HALLUCINATED label = PENUH if any error claim is CORE, SEBAGIAN if every error claim is MINOR (judge_binary.severity_of).

### Missing / unusable judge labels (not imputed)

- none: every item has a usable label from all three judges

### Item categories (3-category labels vs frozen annotator1.xlsx)

| category | meaning | n | items |
|---|---|---|---|
| A | panel unanimous, equal to the human (easy) | 42 (70.0%) | (see JSON) |
| B | panel unanimous, different from the human (re-check the human label) | 10 (16.7%) | CAL-002, CAL-005, CAL-010, CAL-013, CAL-014, CAL-023, CAL-031, CAL-037, CAL-038, CAL-059 |
| C | panel split (ambiguous / hard) | 8 (13.3%) | CAL-016, CAL-017, CAL-018, CAL-021, CAL-024, CAL-034, CAL-050, CAL-057 |
| D | incomplete: a judge has no usable label | 0 (0.0%) |  |

SENSITIVITY (human labels with the post-freeze edits CAL-010, CAL-028, CAL-041; not the frozen labels): A=41, B=11, C=8, D=0.

Category C items (3-category labels):

| item | human | llama | deepseek | qwen |
|---|---|---|---|---|
| CAL-016 | NOT_HALLUCINATED | NOT_HALLUCINATED | HALLUCINATED | NOT_HALLUCINATED |
| CAL-017 | NOT_HALLUCINATED | NOT_HALLUCINATED | HALLUCINATED | NOT_HALLUCINATED |
| CAL-018 | HALLUCINATED (PENUH) | HALLUCINATED | HALLUCINATED | NOT_HALLUCINATED |
| CAL-021 | HALLUCINATED (PENUH) | NOT_HALLUCINATED | HALLUCINATED | NOT_HALLUCINATED |
| CAL-024 | HALLUCINATED (PENUH) | HALLUCINATED | HALLUCINATED | NOT_HALLUCINATED |
| CAL-034 | HALLUCINATED (PENUH) | NOT_HALLUCINATED | HALLUCINATED | NOT_HALLUCINATED |
| CAL-050 | NOT_HALLUCINATED | NOT_HALLUCINATED | ABSTAIN | NOT_HALLUCINATED |
| CAL-057 | HALLUCINATED (SEBAGIAN) | NOT_HALLUCINATED | HALLUCINATED | NOT_HALLUCINATED |
