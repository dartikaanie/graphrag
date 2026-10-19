# Inter-judge agreement and hard cases -- calibration set

Panel panel-v1 (llama = Llama-3.3-70B-Instruct-Turbo, deepseek = DeepSeek-V3.2, qwen = Qwen2.5-72B-Instruct). Main result: judge-v2.3; comparison: judge-v2.2. Human reference: frozen annotator1.xlsx (guideline v1), 60 items. Descriptive statistics only (kappa, percentages); no inferential tests. Saved judge outputs only; no judge API call.

Conventions:

- Labels: judge_binary.category_of (derived from the judge's claims), the same 3-category label as judge_panel_agreement.py.
- PRIMARY = binary "hallucinated" with ABSTAIN excluded (an item/pair is left out when a rater in it says ABSTAIN); companion = ABSTAIN counted as not hallucinated; 3-category as a supplement.
- Cohen's kappa is unweighted, pairwise, over the items both judges labelled; Fleiss' kappa only on complete items (all three judges usable in that mode), n given.
- Kappa can be low while raw agreement is high when one label is rare (see prevalence); "undefined" = both raters used one single category.
- Severity: distribution only, no inter-judge kappa.
- Categories A-D use the 3-category labels; failed / unparsed judge records are category D and are never imputed.

## Overview

| version | Fleiss' kappa binary excl. (n) | Fleiss' kappa binary counted (n) | Fleiss' kappa 3-cat (n) | A | B | C | D | judge SEBAGIAN |
|---|---|---|---|---|---|---|---|---|
| judge-v2.3 | 0.174 (51) | 0.181 (60) | 0.723 (60) | 42 | 10 | 8 | 0 | 0 |
| judge-v2.2 | 0.133 (43) | 0.145 (52) | 0.687 (52) | 35 | 8 | 9 | 8 | 0 |

Judge SEBAGIAN labels across these runs: 0. No judge produced a SEBAGIAN label, so judge severity is PENUH-or-nothing while the human labels hold both SEBAGIAN and PENUH; severity is therefore shown as a distribution only. The judge prompt was not changed for this analysis.

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


## Patterns in category C

Hand-written notes from reading the judge reasoning and claims of the category-C items (main run judge-v2.3; items
that are C only in judge-v2.2 are marked). They describe where the judges diverge; they do not say which judge or
which human label is right. Reviewer decisions belong in the review sheet.

Overall shape (judge-v2.3): in 7 of the 8 C items deepseek is the judge that says HALLUCINATED, alone (5 items) or
with llama (2 items); qwen labels no item HALLUCINATED in this run. In judge-v2.2 the same shape holds, with one
item (CAL-057) where qwen joins deepseek.

1. **Different version numbers / different fix than the reference.** CAL-016, CAL-017 and CAL-058 (v2.2) are
   three candidates for the same Android/Firebase dependency question. The candidate proposes other library
   versions than the accepted answer (e.g. constraint-layout 1.1.0 / 1.1.3 vs 1.0.1, Firebase 11.4.2 vs
   11.0.1). deepseek marks these version recommendations CONTRADICTED by the reference; llama and qwen describe
   them as a different but valid approach. Human label: NOT_HALLUCINATED in all three.

2. **Alternative solution whose mechanism differs from the reference.** CAL-018 (match_parent instead of a
   NestedScrollView with layout_behavior), CAL-021 (cast to StepExecutionListener instead of ChunkListener),
   CAL-024 (expect a Supertest promise to reject with the backend error instead of checking the HTTP response).
   The disagreement is again "contradicts the reference" vs "different approach", but here the human labelled
   HALLUCINATED (PENUH) and wrote a mechanism-level reason. llama splits: HALLUCINATED on CAL-018 and CAL-024,
   NOT_HALLUCINATED on CAL-021; qwen calls all three "different but valid". Patterns 1 and 2 have the same
   judge-side disagreement and opposite human labels.

3. **Time-dependent facts versus the reference date.** CAL-034 (Java 11 on AWS Lambda; the 2019 reference says
   only Java 8 is supported) and CAL-049 (v2.2; the reference describes Mesa as limited for OpenGL 4). deepseek
   follows the reference statement literally; qwen states on CAL-034 that the candidate is "correct, despite the
   outdated reference"; llama treats the claims as unverifiable. Human: HALLUCINATED (PENUH) on CAL-034 (the
   time_rule_check item), NOT_HALLUCINATED on CAL-049, so the two items differ in which side the human label
   is on.

4. **One wrong detail inside an otherwise aligned answer.** CAL-057 (`brew install python@3.3`, a versioned
   formula next to the accepted `brew install python3`) and CAL-023 (v2.2; "there is no direct method to load an
   OGR source into GeoPandas" vs the accepted answer's `gpd.read_file` with the OGR_GRASS driver). The human
   labelled both HALLUCINATED (SEBAGIAN). The judges that flag them label the claim CORE, so their label is PENUH;
   the judges that do not flag them summarise the answer as aligned with the reference. This is where the missing
   SEBAGIAN level on the judge side becomes visible.

5. **Where "attempted" ends and ABSTAIN begins.** CAL-050: the candidate says the context does not cover the
   problem and calls the user's own Redis-SET idea "reasonable". deepseek reads this as no technical claim
   (ABSTAIN); llama and qwen read it as an attempted answer that matches the reference's approach. Human:
   NOT_HALLUCINATED.

## Sources

- human labels: `annotation/calibration/annotator1.xlsx`
- judge-v2.3: `results/panel_calibration_judge-v2.3.jsonl`
- judge-v2.2: `results/panel_calibration_judge-v2.2.jsonl (+ panel_calibration_judge-v2.2_failures.jsonl)`
