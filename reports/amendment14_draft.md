# Amendment 14 (DRAFT, not approved, not committed): frozen judge panel without a human-agreement gate

**Status: draft for the author's review.** Written 2026-10-20, after the Qwen screening (section c) and before any
alpha stage-2 run and any official run on the test sample (n = 384). No stage-2 or test-sample output exists. The
eligibility rule of section (c) was written before any screening call; its byte-exact pre-registered text is
`reports/amendment14_eligibility_rule_preregistered.md` (SHA-256 `62569fc9bb11…`, the value stored in both screening
runs' history entries, `llm/evaluation/logs/judge_panel_run_history.jsonl`, field `screening.rule_sha256`).

## (a) Human validation of the judge is deferred

Human validation of the LLM judge is deferred. The judge panel is used without a human-agreement acceptance gate;
human annotation resumes only if the supervisor asks (section k). The kappa acceptance gates of Amendment 11
(panel 3-category kappa >= 0.6) and Amendment 13 (panel binary kappa, ABSTAIN excluded, >= 0.4) are **suspended**.
They were **not met**, and this is reported, not hidden: on the 60 calibration items against the frozen v1 human
labels, judge-v2.3 / panel-v1 reached 3-category kappa 0.529 and binary kappa 0.180 (section h). Consequently every
rate this thesis reports is a **panel-detected hallucination rate**, not a validated hallucination rate.

## (b) Official frozen judge configuration

| Item | Value |
|---|---|
| Prompt | judge-v2.3 (`llm/evaluation/judge_prompt_v2_3.py`) |
| Parser | pv2.3-1 |
| Blinding | blind-v3 (citation markers removed from the candidate) |
| Request | temperature 0, max_tokens 2000, JSON mode (`response_format: {"type": "json_object"}`), at most 2 retries |
| Panel | **panel-v2** = `meta-llama/Llama-3.3-70B-Instruct-Turbo` + `deepseek-ai/DeepSeek-V3.2` (`reasoning_effort: "none"`) + `Qwen/Qwen3-235B-A22B-Instruct-2507` (DeepInfra) |
| Aggregation | 2-of-3 majority (`judge_panel_aggregate.aggregate_labels`), unchanged |

No further prompt, parser, blinding, model or aggregation change is allowed for stage 2 or the official runs. In the
approval commit, `judge_versions.AMENDMENT14_APPROVED` is set to `True` and `judge_versions.OFFICIAL_ALLOWED` to
`("judge-v2.3",)` (approved by the author, 2026-10-20); until then `judge_panel.py --official` refuses every run, and
it accepts only panel-v2.

## (c) Why Qwen2.5-72B was replaced, and how the replacement was chosen

**Reasons.** In calibration Qwen2.5-72B-Instruct was a degenerate rater: it labelled **0 of 60** items hallucinated
with judge-v2.3 and **1 of 58** with judge-v2.2 (the human labelled 15). It presented candidate text as reference
quotes (criterion (2) below: **8 violations on 4 items**, CAL-005, CAL-007, CAL-041, CAL-057). It was the most
expensive panel member ($0.36 / $0.40 per 1M tokens; $0.138 of the $0.289 judge-v2.3 calibration run).

**Pre-registered eligibility rule** (written before any screening call; verbatim in
`reports/amendment14_eligibility_rule_preregistered.md`). On the 60 calibration items with judge-v2.3, a Qwen
candidate must (1) label at least 5% of its non-ABSTAIN items HALLUCINATED; (2) never present candidate text as a
reference quote: a `[R]`-quoted evidence string (normalised) that is not in the reference answer but is in the
candidate answer is a violation, zero violations allowed; (3) give a usable record for all 60 items with no
reasoning/thinking content. Screening, not tuning: prompt, parser, blinding, request parameters and aggregation
frozen. Candidates in order: Qwen3-235B-A22B-Instruct-2507, then Qwen3-32B with thinking disabled
(`reasoning_effort: "none"` and `chat_template_kwargs: {"enable_thinking": false}`).

**Screening results** (commit `a9a884c`; `llm/evaluation/results/screening_a14_*`; cost cap $0.10 each):

| | Qwen3-235B-A22B-Instruct-2507 | Qwen3-32B |
|---|---|---|
| (1) HALLUCINATED / non-ABSTAIN | 14/50 = 28.0% (pass) | 6/52 = 11.5% (pass) |
| (2) violations / affected items | 2 / 1 (CAL-033) | 1 / 1 (CAL-041) |
| (3) usable, reasoning content, `<think>` | 60/60, 0, 0 (pass) | 60/60, 0, 0 (pass) |
| served-model mismatch | 0 | 0 |
| actual cost | $0.0538 | $0.0399 |

**Criterion (2) fairness check** (stored judge-v2.3 calibration outputs, same function, no API call;
`llm/evaluation/results/screening_a14_criterion2_fairness_check.md`):

| Judge | `[R]` quotes | Violations | Affected items |
|---|---|---|---|
| Llama-3.3-70B-Instruct-Turbo | 0 | 0 | 0 (it never quotes the reference) |
| DeepSeek-V3.2 | 112 | 1 | 1 (CAL-033) |
| Qwen2.5-72B-Instruct | 75 | 8 | 4 |

For information: every violation of the two Qwen3 candidates and of DeepSeek quotes text that also appears in the
**question** (the candidate answer repeated it), while 6 of Qwen2.5's 8 violations quote text found only in the
candidate answer.

**Post-results revision of criterion (2) (decided by the author on 2026-10-20, after the screening results were
known).** Criterion (2) was originally zero-tolerance. Both Qwen3 candidates failed it on a single item each
(Qwen3-235B: CAL-033; Qwen3-32B: CAL-041). Because the criterion's purpose is to exclude systematic evidence
fabrication (Qwen2.5-72B: 8 violations), it is revised to "at most 1 affected item out of 60". This revision was made
after the screening results were known. The fairness check above shows that the retained panel member DeepSeek-V3.2
would also have failed the zero-tolerance version (1 affected item, CAL-033) and passes the revised one; Llama-3.3
cannot fail it because it produces no `[R]` quote.

**Selected judge: `Qwen/Qwen3-235B-A22B-Instruct-2507`.** Both candidates meet the revised rule. The basis of the
choice is the pre-registered criterion (1): Qwen3-235B is the more sensitive rater (28.0% vs 11.5% of non-ABSTAIN
items labelled hallucinated). Agreement with the frozen v1 labels was computed and seen (information only; not the
basis): Qwen3-235B 3-category kappa 0.424, binary kappa 0.175, recall 6/15, false alarms 8/35; Qwen3-32B 0.484,
0.145, 3/15, 3/37. Qwen3-235B-A22B-Instruct-2507 is a non-thinking model by design (no reasoning content in any of
the 60 responses), served in FP8, and is called with JSON mode like the other judges.

**panel-v2 on the calibration set** (stored Llama and DeepSeek records + the Qwen3-235B screening records, no extra
call): 3-category kappa 0.566, binary kappa (ABSTAIN excluded) 0.329, recall 5/15, false alarms 2/36; per condition
recall A [dev] 1/8, B-plain [dev] 3/6, B-grounded [dev] 1/1; labels FAKTUAL 44, HALUSINASI_PENUH 7, ABSTAIN 9;
0 UNRESOLVED; Fleiss' kappa between the three judges 0.646 (3 categories, n = 60) and 0.384 (binary, ABSTAIN
excluded, n = 50). These numbers are reported, not used as a gate.

## (d) Role of each estimate

All estimates are computed from the **same stored judge outputs**; no estimate needs an extra call.

- **Panel rate (primary):** the 2-of-3 panel label. It is the conservative estimate and likely a **lower bound**: in
  calibration the panel missed most human-labelled hallucinations (panel-v1 recall 2/15; panel-v2 5/15) while
  raising few false alarms.
- **DeepSeek-only rate (sensitive):** the most sensitive judge in calibration; it includes some false alarms on
  valid alternative approaches (inter-judge pattern 1).
- **Llama-only and Qwen-only rates:** reported as supporting information.

## (e) Interpretation rule (recorded before any official run)

A pre-registered comparison (R5 vs R1, R5 vs R3, R5 vs R9, R5 vs R7; and R4 vs R2, R4 vs R8, R4 vs R6 when the
stage-2 runs of the official matrix are available) is called **"consistent"** only if the panel difference and the
DeepSeek-only difference (hallucination rate excluding ABSTAIN) have the **same sign**; otherwise it is reported as
**"inconclusive"**. A difference of exactly 0 under either estimate is "inconclusive" (approved by the author,
2026-10-20). The rule labels the direction
only; the significance tests of the Selection procedure / Amendment 4 are unchanged and are run on the panel label.

## (f) Time standard of the metric

The judge prompt lets well-established knowledge override an outdated reference answer. This defines the measured
quantity: a claim is judged against knowledge valid at evaluation time, not only against the reference. Limitation:
in calibration the judges applied this inconsistently (inter-judge pattern 3: CAL-034 and CAL-049; DeepSeek follows
the reference literally, Qwen2.5 overrides it, Llama treats the claims as unverifiable).

## (g) Alpha stage 2 uses the frozen panel

Alpha stage 2 is judged with the frozen panel-v2 configuration of section (b), which is not validated against human
labels. The **panel label** is the label of the pre-registered stage-2 selection rule (binary hallucination rate
excluding ABSTAIN, pooled over grounding on/off, tie margin 2 answers, tie-breakers as in the Selection procedure);
per-judge results are reported alongside. `stage2_report.py` is adapted to read the panel file instead of the
"primary" judge (no change to the rule).

## (h) Reported limitations (numbers from the archived calibration reports)

Source: `results/_archive/2026-10-19_pre-official/llm/evaluation/results/` (`panel_agreement_calibration_judge-v2.3.md`,
`interjudge_summary_calibration.md`, `interjudge_patterns_C_calibration.md`).

- Calibration agreement against the frozen v1 labels (60 items, judge-v2.3, panel-v1): 3-category kappa **0.529**
  (95% CI 0.314-0.716), binary kappa (ABSTAIN excluded) **0.180** (CI 0.0-0.409); panel recall **2/15**, unsafe pass
  13/15, false alarms 0/37. Per-condition panel recall: A [dev] 1/8, B-plain [dev] 1/6, B-grounded [dev] 0/1.
  Single-annotator labels; human labels post-dating the freeze change these figures slightly (sensitivity rows of
  the report).
- Inter-judge disagreement patterns (`interjudge_patterns_C_calibration.md`, archived): (1) valid alternative vs
  contradiction (different versions / different fix than the reference); (2) alternative solution with a different
  mechanism; (3) time-dependent facts vs the reference date; (4) a single wrong detail inside an otherwise aligned
  answer (no SEBAGIAN level on the judge side); (5) the ABSTAIN boundary.
- The judge-side severity scale does not reproduce the human SEBAGIAN level (pattern 4).
- The Qwen seat was filled under a criterion revised after the screening results (section c).
- Rates are called **"panel-detected hallucination rates"** throughout.

## (i) Reporting rule for every official run (and stage 2)

Per condition / run: (1) hallucination rate excluding ABSTAIN, (2) ABSTAIN rate, (3) hallucination rate over all
answers (ABSTAIN counted as not hallucinated) -- each **for the panel and for each judge**; plus Fleiss' kappa and
Krippendorff's alpha (nominal) between the three judges, with n. Items with an UNRESOLVED or incomplete panel are
counted and reported, never imputed. Also per run: cosine similarity (mean, median), citation validity (NF2), and
latency from the per-answer timing of record schema r3 (`answer_started_at_utc`, `answer_finished_at_utc`,
`end_to_end_latency_sec` around retrieval + generation, and `generation_latency_sec`), added before any stage-2 or
official run. **NF3 = the p95 of per-answer end-to-end latency of a Condition C run, target <= 15 s.** The run-duration
proxy (run duration / answers) is used only for archived runs without r3 timing and is labelled as a fallback; a run
without r3 timing cannot be registered as stage-2 or official.

## (j) Staged execution with the same frozen configuration

1. Alpha stage 2 first (4 dev runs, 200 answers; writes alpha\* and alpha_abl to `official_params.json`).
2. Official stage 1: R1, R3, R5, R7, R9 (the pre-registered primary comparisons and the trust effect).
3. Official stage 2: R2, R4, R6, R8 -- **deferred, not dropped**: it runs after a later top-up of the DeepInfra budget,
   with the same frozen configuration. These four runs are supporting analyses (Bab III). If they are never run,
   this is reported as a limitation.
Each step starts only after the author's explicit "go". DeepInfra budget for judging: $9.00 (the account balance is
$9.20; $0.20 stays untouched as a margin; the screening spend of $0.094 is already paid and counted). Projected
judging cost with a 10% buffer: alpha stage 2 $0.76, official stage 1 $6.80, official stage 2 $6.30
(`results/cost_projection.md`).

## (k) Reactivation plan (only if the supervisor asks)

Annotator 1 annotates under guideline v2 (`docs/ANNOTATION_GUIDELINE_v2.md`). The v2 sheet
`llm/evaluation/annotation/calibration/annotator1_v2.xlsx` was exported on 2026-10-19, is not filled in and is not
used. A confirmation set of 30 items is drawn with seed 42 from the official C-run outputs and annotated blind,
without AI assistance; agreement is computed on the **stored** judge outputs, without re-running any judge.

## (l) Archive note

All pre-official data were archived on 2026-10-19: `results/_archive/2026-10-19_pre-official/ARCHIVE_INDEX.md`
(commit `c14e430`).

## (m) Scope

This amendment supersedes **only the acceptance parts** of Amendments 11-13 (the kappa gates and the confirmation-set
acceptance). Everything else in those amendments, and every earlier amendment, stands.
