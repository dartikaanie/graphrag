# Alpha stage 2: result, and a note on the judge-panel runtime change

Recorded 2026-10-26. This note adds to the record; no amendment is edited. Status of every number below: dev split
(n = 50), **not** an official result. Nothing in `official_params.json` or any frozen file was changed.

## 1. Result of alpha stage 2 (dev set, C retrieval v4, thread leakage)

Generation: gpt-4o-mini, 50 questions, grounding on and off, 5.0 context items per answer in all six runs. Judging:
panel-v2 (Llama-3.3-70B-Turbo, DeepSeek-V3.2, Qwen3-235B-A22B-Instruct-2507), judge-v2.3, pv2.3-1, blind-v3, 2-of-3.
Candidates are the top two of the Amendment 17 stage-1 re-run (`stage1_anchors_report_20261025T165734Z.json`).

| alpha | role | hallucinated / non-ABSTAIN (rate) | ABSTAIN | hallucinated + ABSTAIN / 100 |
|---|---|---|---|---|
| 0.5 | candidate | **23 / 91** (0.2527) | 9 | 32 |
| 0.25 | candidate | **21 / 90** (0.2333) | 10 | 31 |
| 0 | exploratory, not eligible | **21 / 90** (0.2333) | 10 | 31 |

Per judge (hallucinated / non-ABSTAIN answers of that judge, pooled over grounding), order alpha 0.5 | 0.25 | 0:

| judge | 0.5 | 0.25 | 0 |
|---|---|---|---|
| Llama-3.3-70B-Turbo | 1 / 93 | 1 / 91 | 4 / 92 |
| DeepSeek-V3.2 | 26 / 88 | 31 / 89 | 31 / 90 |
| Qwen3-235B-A22B-Instruct-2507 | 30 / 89 | 32 / 88 | 35 / 88 |

**Llama flagged very few answers** (1, 1 and 4 of about 92), so the panel (2 of 3) is in practice decided by DeepSeek
and Qwen. DeepSeek and Qwen alone rate alpha 0.5 lowest; the panel rate puts 0.5 slightly above 0.25.

**Selection.** The pre-registered rule (hallucinated count excl. ABSTAIN, margin 2 answers; then with ABSTAIN counted
as failed, margin 2; then larger alpha) selects **alpha = 0.5 at the tie-break step**: the primary difference is 2
answers (23 vs 21) and the secondary difference is 1 answer (32 vs 31), both within the margin. Per Amendment 10, a
tie-break win is not evidence of an effect of trust. Alpha 0 was not worse than 0.25 on the dev set. If alpha 0.5 is
accepted as alpha\*, alpha 0.25 is the unused alpha_ablation. The change that would record this is prepared, not applied:
`docs/PREPARED_ALPHA_SELECTION_CHANGE.md`.

**Report files.** `llm/c_graphrag/results/stage2/stage2_report_20261025T221359Z.{md,json}` is the first report; its
selection sentence ("has the lowest pooled hallucination rate") is wrong for this result and is kept unchanged.
`stage2_report_20261025T232940Z.{md,json}` was regenerated from the same run and judge files after the wording fix
below; the numbers are identical.

**Wording fix (2026-10-26).** `ctxrel_sweep.select_stage2_winner` now names the tier that decided the selection
(`deciding_tier`: primary, secondary or tie-break) and never claims the winner has the lowest rate. Selection logic and
margins are unchanged. For this result it says: "alpha=0.5 selected at the tie-break step; alpha=0.5 vs 0.25 differ by
2 hallucinated answers (23 vs 21), within the margin."

**Cost.** Generation $0.164; judging about $1.08 by record-level estimate (DeepInfra usage figures give about $0.95 to
$1.0); total about $1.25, under the $1.50 cap. The registry ledger held $0.5954 of judging cost; the first aborted
judging run (about $0.107) was added on 2026-10-26 as an explicit adjustment entry (`"adjustment": true`) in
`results/run_registry.json`; no earlier entry was changed. The ledger total is now $0.7024 (an estimate for the added
part).

## 2. Judge-panel runtime change: what it does and does not change

`llm/evaluation/judge_panel.py` stalled several times during stage 2 (one hanging connection held a run for about
30 minutes). Two changes were made (committed in `1784f5a`):

1. `make_clients` passes `timeout=CLIENT_TIMEOUT_SEC` (180 s per request; the SDK default is 600 s).
2. `run_tasks` writes each result as soon as its call finishes (`as_completed`) instead of in submission order
   (`pool.map`).

Neither touches the judge version (judge-v2.3), the prompts, the parser (pv2.3-1), blinding, the request parameters
(temperature 0, max_tokens 2000, JSON mode, 2 retries), the panel definition or the aggregation
(`judge_panel_aggregate.aggregate_labels`). Checked by `git diff 29f59a1 HEAD -- llm/evaluation/judge_panel.py` (only
these two changes) and by the hashes: `messages_sha256` is computed from the messages only, so it is unchanged; the
record `config_hash` comes from the generation file; the registry `judge_config` does not contain the timeout; and
`python3 llm/frozen_parameters.py --check` reports the frozen-parameter table (hash `d3b805af7798`) up to date, so **no
hash changed**. The only observable effect is the order of lines in the output file and that a stalled call fails after
180 s and is retried within the existing retry limit instead of after 600 s. A call that exceeds the timeout twice
would be recorded as a failed call, exactly as a failure of any other kind.
