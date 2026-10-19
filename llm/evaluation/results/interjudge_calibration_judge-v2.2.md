# Inter-judge agreement -- calibration

## judge-v2.2

### Inter-judge kappa -- binary, ABSTAIN excluded (PRIMARY)

| pair | n | raw agreement | Cohen's kappa |
|---|---|---|---|
| llama vs deepseek | 45 | 37/45 (82.2%) | 0.167 |
| llama vs qwen | 50 | 48/50 (96.0%) | -0.020 |
| deepseek vs qwen | 43 | 36/43 (83.7%) | 0.189 |
| **Fleiss' kappa (panel, complete items)** | 43 | unanimous 35/43 (81.4%) | **0.133** |

Matrix (Cohen's kappa; raw agreement):

| | llama | deepseek | qwen |
|---|---|---|---|
| llama | -- | 0.167 (82.2%, n=45) | -0.020 (96.0%, n=50) |
| deepseek | 0.167 (82.2%, n=45) | -- | 0.189 (83.7%, n=43) |
| qwen | -0.020 (96.0%, n=50) | 0.189 (83.7%, n=43) | -- |

### Inter-judge kappa -- binary, ABSTAIN counted as not hallucinated

| pair | n | raw agreement | Cohen's kappa |
|---|---|---|---|
| llama vs deepseek | 54 | 46/54 (85.2%) | 0.172 |
| llama vs qwen | 58 | 56/58 (96.5%) | -0.018 |
| deepseek vs qwen | 52 | 45/52 (86.5%) | 0.195 |
| **Fleiss' kappa (panel, complete items)** | 52 | unanimous 44/52 (84.6%) | **0.145** |

### Inter-judge kappa -- 3-category (ABSTAIN / NOT_HALLUCINATED / HALLUCINATED)

| pair | n | raw agreement | Cohen's kappa |
|---|---|---|---|
| llama vs deepseek | 54 | 45/54 (83.3%) | 0.600 |
| llama vs qwen | 58 | 56/58 (96.5%) | 0.871 |
| deepseek vs qwen | 52 | 44/52 (84.6%) | 0.628 |
| **Fleiss' kappa (panel, complete items)** | 52 | unanimous 43/52 (82.7%) | **0.687** |

### Prevalence of "hallucinated" (and ABSTAIN) per rater

| rater | H, ABSTAIN excluded | H, ABSTAIN counted | ABSTAIN |
|---|---|---|---|
| llama | 1/52 (1.9%) | 1/60 (1.7%) | 8/60 (13.3%) |
| deepseek | 9/45 (20.0%) | 9/54 (16.7%) | 9/54 (16.7%) |
| qwen | 1/50 (2.0%) | 1/58 (1.7%) | 8/58 (13.8%) |
| human | 15/52 (28.8%) | 15/60 (25.0%) | 8/60 (13.3%) |

### Severity distribution (no kappa)

| rater | FAKTUAL | SEBAGIAN | PENUH | UNSPECIFIED | ABSTAIN | missing |
|---|---|---|---|---|---|---|
| llama | 51 | 0 | 1 | 0 | 8 | 0 |
| deepseek | 36 | 0 | 9 | 0 | 9 | 6 |
| qwen | 49 | 0 | 1 | 0 | 8 | 2 |
| human | 37 | 9 | 6 | 0 | 8 | 0 |

Judge SEBAGIAN labels in this run: 0. Severity of a judge's HALLUCINATED label = PENUH if any error claim is CORE, SEBAGIAN if every error claim is MINOR (judge_binary.severity_of).

### Missing / unusable judge labels (not imputed)

- deepseek / CAL-014: no record; failures file: parse_error
- deepseek / CAL-021: no record; failures file: parse_error
- deepseek / CAL-029: no record; failures file: parse_error
- deepseek / CAL-030: no record; failures file: parse_error
- deepseek / CAL-039: no record; failures file: parse_error
- deepseek / CAL-051: no record; failures file: parse_error
- qwen / CAL-008: no record; failures file: parse_error
- qwen / CAL-024: no record; failures file: parse_error

### Item categories (3-category labels vs frozen annotator1.xlsx)

| category | meaning | n | items |
|---|---|---|---|
| A | panel unanimous, equal to the human (easy) | 35 (58.3%) | (see JSON) |
| B | panel unanimous, different from the human (re-check the human label) | 8 (13.3%) | CAL-002, CAL-005, CAL-010, CAL-013, CAL-031, CAL-037, CAL-038, CAL-059 |
| C | panel split (ambiguous / hard) | 9 (15.0%) | CAL-016, CAL-017, CAL-018, CAL-023, CAL-034, CAL-049, CAL-050, CAL-057, CAL-058 |
| D | incomplete: a judge has no usable label | 8 (13.3%) | CAL-008, CAL-014, CAL-021, CAL-024, CAL-029, CAL-030, CAL-039, CAL-051 |

SENSITIVITY (human labels with the post-freeze edits CAL-010, CAL-028, CAL-041; not the frozen labels): A=34, B=9, C=9, D=8.

Category C items (3-category labels):

| item | human | llama | deepseek | qwen |
|---|---|---|---|---|
| CAL-016 | NOT_HALLUCINATED | NOT_HALLUCINATED | HALLUCINATED | NOT_HALLUCINATED |
| CAL-017 | NOT_HALLUCINATED | NOT_HALLUCINATED | HALLUCINATED | NOT_HALLUCINATED |
| CAL-018 | HALLUCINATED (PENUH) | HALLUCINATED | HALLUCINATED | NOT_HALLUCINATED |
| CAL-023 | HALLUCINATED (SEBAGIAN) | NOT_HALLUCINATED | HALLUCINATED | NOT_HALLUCINATED |
| CAL-034 | HALLUCINATED (PENUH) | NOT_HALLUCINATED | HALLUCINATED | NOT_HALLUCINATED |
| CAL-049 | NOT_HALLUCINATED | NOT_HALLUCINATED | HALLUCINATED | NOT_HALLUCINATED |
| CAL-050 | NOT_HALLUCINATED | NOT_HALLUCINATED | ABSTAIN | NOT_HALLUCINATED |
| CAL-057 | HALLUCINATED (SEBAGIAN) | NOT_HALLUCINATED | HALLUCINATED | HALLUCINATED |
| CAL-058 | NOT_HALLUCINATED | NOT_HALLUCINATED | HALLUCINATED | NOT_HALLUCINATED |
