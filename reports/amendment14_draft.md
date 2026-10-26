# Amendment 14 (confirmed 2026-10-26): frozen judge panel for the four official runs

**Status: confirmed by the author on 2026-10-26.** Rewritten 2026-10-25 for the four-run design of Amendment 16 (A, B-grounded,
C-primary-grounded, D-grounded). The first version of 2026-10-20 was written for nine runs and carried human-agreement
gates; both are gone. Written before any alpha stage-2 run and any official run on the test sample (n = 384). No
stage-2 or test-sample output exists. The Qwen eligibility rule of section (c) was written before any screening call;
its byte-exact pre-registered text is `reports/amendment14_eligibility_rule_preregistered.md` (SHA-256
`62569fc9bb11…`, the value stored in both screening runs' history entries,
`llm/evaluation/logs/judge_panel_run_history.jsonl`, field `screening.rule_sha256`).

**Update note (2026-10-26):** Stage 1 was re-run under Amendment 17 with leakage thread; the stage-1 of record is stage1_anchors_report_20261025T165734Z. Stage 2 was run on 2026-10-25/26 (see docs/STAGE2_RESULT_AND_RUNTIME_NOTE.md); α* = 0.5 at the tie-break step. Statements above that say otherwise are historical, as written when the draft was made. The drafts of Amendments 14, 15, 16 and 17 were written before stage 2 but are committed after it.

## (a) No human gate

The judge panel is used without any human annotation, human-agreement threshold or human-validated acceptance step.
The kappa gates of Amendments 11 and 13 and the confirmation-set acceptance are **removed** for the official
evaluation (`judge_panel.py` never asks for them on the panel path;
`llm/evaluation/tests/test_panel_path_needs_no_human_annotation.py`). Consequently every rate this thesis reports is a
**panel-detected hallucination rate**, not a validated hallucination rate, and the thesis says so.

## (b) Official frozen judge configuration

| Item | Value |
|---|---|
| Prompt | judge-v2.3 (`llm/evaluation/judge_prompt_v2_3.py`) |
| Parser | pv2.3-1 |
| Blinding | blind-v3 (citation markers and "the retrieved sources …" meta phrases removed from the candidate) |
| Request | temperature 0, max_tokens 2000, JSON mode (`response_format: {"type": "json_object"}`), at most 2 retries |
| Panel | **panel-v2** = `meta-llama/Llama-3.3-70B-Instruct-Turbo` + `deepseek-ai/DeepSeek-V3.2` (`reasoning_effort: "none"`) + `Qwen/Qwen3-235B-A22B-Instruct-2507` (DeepInfra) |
| Aggregation | 2-of-3 majority (`judge_panel_aggregate.aggregate_labels`), unchanged: at least two ABSTAIN gives ABSTAIN, otherwise the median of the ordinal scores |

No further prompt, parser, blinding, model or aggregation change is allowed for stage 2 or the official runs. In the
approval commit, `judge_versions.AMENDMENT14_APPROVED` is set to `True` and `judge_versions.OFFICIAL_ALLOWED` to
`("judge-v2.3",)` (the exact change is prepared in `docs/AMENDMENT14_APPROVAL_CHANGE.md`); until then
`judge_panel.py --official` refuses every run, and it accepts only panel-v2.

## (c) Why Qwen2.5-72B was replaced, and how the replacement was chosen

**Reasons.** In the 60-item calibration run Qwen2.5-72B-Instruct was a degenerate rater: it labelled 0 of 60 items
hallucinated with judge-v2.3 (1 of 58 with judge-v2.2). It presented candidate text as reference quotes (criterion (2)
below: 8 violations on 4 items) and was the most expensive panel member.

**Pre-registered eligibility rule** (written before any screening call). On the 60 calibration items with judge-v2.3, a
Qwen candidate must (1) label at least 5% of its non-ABSTAIN items HALLUCINATED; (2) never present candidate text as a
reference quote; (3) give a usable record for all 60 items with no reasoning/thinking content. Screening, not tuning:
prompt, parser, blinding, request parameters and aggregation frozen. Candidates in order: Qwen3-235B-A22B-Instruct-2507,
then Qwen3-32B with thinking disabled.

**Screening results** (commit `a9a884c`; cap $0.10 each):

| | Qwen3-235B-A22B-Instruct-2507 | Qwen3-32B |
|---|---|---|
| (1) HALLUCINATED / non-ABSTAIN | 14/50 = 28.0% (pass) | 6/52 = 11.5% (pass) |
| (2) violations / affected items | 2 / 1 (CAL-033) | 1 / 1 (CAL-041) |
| (3) usable, reasoning content, `<think>` | 60/60, 0, 0 (pass) | 60/60, 0, 0 (pass) |
| actual cost | $0.0538 | $0.0399 |

**Post-results revision of criterion (2) (author, 2026-10-20, after the screening results were known).** Criterion (2)
was zero-tolerance; both Qwen3 candidates failed it on a single item. Because its purpose is to exclude systematic
evidence fabrication (Qwen2.5-72B: 8 violations), it was revised to "at most 1 affected item out of 60". The retained
panel member DeepSeek-V3.2 would also have failed the zero-tolerance version (1 affected item, CAL-033). This revision
is reported as a limitation.

**Selected judge: `Qwen/Qwen3-235B-A22B-Instruct-2507`**, on the pre-registered criterion (1): it is the more sensitive
rater (28.0% vs 11.5%). It is a non-thinking model by design (no reasoning content in any of the 60 responses), served
in FP8, and is called with JSON mode like the other judges.

## (d) Role of each estimate

All estimates are computed from the **same stored judge outputs**; no estimate needs an extra call.

- **Panel rate (primary):** the 2-of-3 panel label.
- **DeepSeek-only rate (sensitive):** DeepSeek-V3.2 was the most sensitive judge in the calibration run; it includes
  some false alarms on valid alternative approaches.
- **Llama-only and Qwen-only rates:** reported as supporting information.

## (e) Pre-registered comparisons and interpretation rule

Three comparisons on the hallucination rate of the panel label with ABSTAIN excluded (version (a) of section (i)):

1. **C vs A** (C-primary-grounded vs plain LLM),
2. **C vs B** (C vs B-grounded, conventional RAG),
3. **C vs D** (C vs D-grounded, dual-level retrieval).

A comparison is called **"consistent"** only if the panel difference and the DeepSeek-only difference (rate excluding
ABSTAIN) have the **same sign**; otherwise it is **"inconclusive"**. A difference of exactly 0 under either estimate is
"inconclusive". The rule labels the direction only. Nothing else is pre-registered as a primary comparison: there is
no plain-vs-grounded comparison and no trust-ablation comparison in the official evaluation. The report additionally
gives, per comparison and per question, how many times C is better, equal or worse on the binary label (section (i)); the 2 x 2 table
(only C hallucinated, only the other, both, neither) is the **primary paired report**.

## (f) Time standard of the metric

The judge prompt lets well-established knowledge override an outdated reference answer. A claim is judged against
knowledge valid at evaluation time, not only against the reference. Limitation: the judges applied this inconsistently
in the calibration run (DeepSeek follows the reference literally, Qwen2.5 overrode it, Llama treated such claims as
unverifiable).

## (g) Alpha stage 2 uses the frozen panel

Alpha stage 2 (dev set, n = 50) is judged with the frozen panel-v2 of section (b). The candidates are the top two alphas of the stage-1 re-run of Amendment 17 (0.5 and 0.25); alpha 0 is run as an exploratory reference and is not eligible for selection. The label of the pre-registered selection rule is the panel label: binary hallucination rate excluding ABSTAIN, pooled over grounding on and off (100 answers per alpha), tie margin 2 answers, then ABSTAIN counted as failed, then the larger alpha. Stage 2 is a dev-split run and does not need the approval flag of section (b). No official run uses the second candidate. Result: docs/STAGE2_RESULT_AND_RUNTIME_NOTE.md.

## (h) Reported limitations

- Rates are "panel-detected hallucination rates"; the panel is not validated against human labels.
- The Qwen seat was filled under a criterion revised after the screening results (section (c)).
- The judge-side severity scale has no counterpart to a human "partial" level; severity is reported descriptively.
- Inter-judge disagreement patterns found in the archived calibration run (valid alternative vs contradiction,
  different mechanism, time-dependent facts, single wrong detail, the ABSTAIN boundary) are listed in the archived
  reports `results/_archive/2026-10-19_pre-official/llm/evaluation/results/` and are cited as qualitative limitations,
  not as acceptance numbers.
- Context relevance and answer relevance (section (i)) come from one judge, not the panel. That judge is
  `meta-llama/Llama-3.3-70B-Instruct`, while the panel uses the Turbo variant (`Llama-3.3-70B-Instruct-Turbo`).
- In stage 2 Llama flagged very few answers as hallucinated (1, 1 and 4 of about 92 answers for alpha 0.5, 0.25 and 0), so
  the panel label (2 of 3) is in practice decided by DeepSeek and Qwen.

## (i) Reporting rule for every official run

Per run and condition: **(a)** hallucination rate **excluding ABSTAIN** = hallucinated / (hallucinated + not
hallucinated), the primary figure; **(b)** hallucination rate **with ABSTAIN counted as failed** = (hallucinated +
ABSTAIN) / all usable answers; the ABSTAIN rate; the severity split (full / partial hallucination, shares of the
hallucinated answers); the share of unverifiable claims; each **for the panel and for each judge**. Items with an
UNRESOLVED or incomplete panel are counted and reported, never imputed.

Also per run: cosine similarity (mean, median); citation validity (NF2); latency from the per-answer timing of record
schema r3 (`end_to_end_latency_sec`, `generation_latency_sec`, `retrieval_latency_sec`), where **NF3 = the p95 of
per-answer end-to-end latency of the Condition C run, target <= 15 s**, and retrieval p50/p95 for every run.

**Faithfulness and citation support (B, C, D only; no new call).** Aggregated from the per-claim labels already stored on
the panel records. Faithfulness = claims with `supported_by_context = YES` / claims with a YES or NO value.
Citation support = cited claims with `cited_item_supports = YES` / cited claims with a YES or NO value. Claims with a
null value are left out of the denominator, never imputed. Reported pooled over the three judges' claim lists and per
judge (micro), plus the per-answer mean (macro). Condition A has no context and is excluded.

**Secondary metrics (one judge, not the panel).** Context relevance (B, C, D; prompt ctxrel-v2) and answer relevance
(A, B, C, D; prompt ansrel-v1) by `meta-llama/Llama-3.3-70B-Instruct` at temperature 0, blinded to condition and scores
(`llm/evaluation/secondary_metrics.py`; dry run first, official files only after the approval of section (b)).

**Agreement.** Fleiss' kappa and Krippendorff's alpha (nominal) between the three judges with n, and the share of
answers on which all three judges gave the same label.

**Paired counts.** For C vs A, C vs B and C vs D, per question: C better, equal, worse on the binary label of the panel
(version (a): pairs where either answer is ABSTAIN or unusable are left out and counted; version (b): ABSTAIN counts as
failed). The 2 x 2 table (only C hallucinated, only the other, both, neither) is the primary paired report; the better / equal / worse counts are secondary.

## (j) Execution

1. Alpha stage 2 first: two candidates plus one exploratory reference, grounding on and off, 300 answers; run 2026-10-25/26 (alpha\* = 0.5 at the tie-break step; applied to `official_params.json` on 2026-10-26).
2. The four official runs (A, B-grounded, C-primary-grounded, D-grounded), then judging with the frozen panel, then the
   secondary metrics, then `llm/evaluation/official_report.py`.

Each step starts only after the author's explicit "go". DeepInfra budget for judging: $9.00 (set in
`results/run_registry.json`). **Stage-2 judging spend, three numbers that do not agree** (no account or billing endpoint
was called for this reconciliation): (1) the registry ledger holds $0.6087 for the stage-2 judging (entries $0.5017 plus
an explicit adjustment entry of about $0.107 for the first aborted run; ledger total including the two Qwen screenings
$0.7024); (2) the record-level estimate, the sum of `estimated_cost_usd` over the 900 kept judge records, is $1.081; (3) the
provider's usage figures, read earlier in a filtered form, are about $0.95 to $1.0. The ledger is therefore still lower
than the other two by about $0.35 to $0.47 and under-records the resumed runs; the registry is not corrected beyond the
adjustment entry. For the budget, use the higher figure: stage 2 consumed about $1.08 plus the screenings ($0.094), so
about $7.83 of the $9.00 remains for the official judging. The cost projection for the four-run design is
`results/cost_projection.md` (regenerated by `llm/cost_projection.py`; the file under `results/` is the author's to
regenerate).

## (k) Archive note

All pre-official data were archived on 2026-10-19: `results/_archive/2026-10-19_pre-official/ARCHIVE_INDEX.md`
(commit `c14e430`).

## (m) Scope

This amendment supersedes the acceptance parts of Amendments 11-13 (the kappa gates and the confirmation-set
acceptance). Everything else in those amendments, and every earlier amendment, stands, except where Amendment 16 says
otherwise (the run matrix, the second stage-2 candidate, Condition D's scoring).

## (n) Runtime change in `judge_panel.py` (2026-10-26)

During stage 2 the judge run stalled several times on one hanging connection. Two changes were made: the client timeout
is 180 s per request (the SDK default is 600 s), and each result is written as soon as its call finishes
(`as_completed`) instead of in submission order. Judge version, prompts, parser, blinding, request parameters, panel
and aggregation are not changed, and no hash changed. Details and the checks: `docs/STAGE2_RESULT_AND_RUNTIME_NOTE.md`,
section 2.

## Record
* Confirmation by the author: 2026-10-26
* Commit: the approval commit (subject line below)
* Subject line: approve amendments 14-17 and unlock official judging
