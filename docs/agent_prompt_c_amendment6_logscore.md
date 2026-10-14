# Agent Prompt — Condition C, Amendment 6: Log-normalised upvote score in trustScore

> Suggested repo path: `docs/agent_prompt_c_amendment6_logscore.md`
> Repo: `dartikaanie/graphrag` (branch `master`) · Local: `/Users/dartikaanie/SMART SYSTEM/THESIS_2/project/graphrag/`
> Before you start, open `docs/agent_prompt_c_retrieval_v3_devset.md` and `docs/DECISION_C_SCORING.md` to match the existing conventions (naming, logging, decision-record style).

---

## 0. HARD RULES (read first, apply throughout)

1. **Do NOT run** `14_apply_logscore_trust.py --apply` or `--restore`, **do NOT call any judge/LLM API** (OpenAI, DeepInfra, Anthropic, Ollama), and **do NOT `git commit` / `git push`** without my explicit confirmation in chat. You may run: `--dry-run`, read-only Neo4j queries, DuckDB reads, `grep`, and `pytest`.
2. **Do NOT touch the official test sample** (n=384, seed=42). Do not load, score, rank, or inspect its items. Only the dev set is in scope.
3. **Do NOT change** the `HAS_ANSWER` edge weights (per-thread min-max, fallback 0.5), the `combined_trust` formula, the final ranking formula, the α grid, or the stage-1 decision rule.
4. **Do NOT overwrite or restart** the uncommitted work already in the repo (see Task 0). Audit it, then build on it. If you must replace something, say what and why first.
5. Environment: MacBook M2, 8 GB RAM, pyenv Python 3.10.9. Keep DuckDB pragmas `memory_limit='2GB'`, `threads=2`, `preserve_insertion_order=false`. Any `SentenceTransformer` must use `device="cpu"`. Neo4j: `bolt://localhost:7687`, database `graphrag`. Never print or log the contents of `.env` / API keys.
6. Terminal commands you give me must be **zsh-compatible**, one command per line, **no `#` comments on the same line** as a command.
7. Talk to me in **simple Bahasa Indonesia, step by step**. Technical terms stay in English. Code, comments, docstrings, and docs stay in English.
8. If anything in the repo contradicts this prompt (e.g., a formula differs from what I describe as "current"), **stop and report** instead of guessing.

---

## 1. Context

Thesis: GraphRAG over Stack Overflow (SORD) to reduce LLM hallucination. The KG is in Neo4j: 2,685,814 `Question`, 725,795 `Answer` (270,988 accepted).

**Condition C** ranks candidate answers with:

```
combined_trust = 0.7 · path_weight + 0.3 · trustScore_Answer
final          = (1 − α) · sim_norm + α · combined_trust
```

α is selected in two stages on the **dev set**. Stage 1 (context-relevance judge) is done. Stage 2 is **not** started. The official n=384 run is **not** started.

**Current trustScore** (`01_data_cleaning/7_build_knowledge_graph.py`, `compute_trust_scores`):

```
trust_Q = 0.7 · minmax(score_Q) + 0.3 · minmax(log1p(views))
trust_A = 0.5 · minmax(score_A) + 0.3 · is_accepted + 0.2 · w_auth
w_auth  = clip(log1p(rep) / log1p(P99_rep), 0, 1)
```

`w_auth` came from Amendment 5, applied 2026-10-13 via `13_apply_reputation_trust.py`; current `trustScoreVersion = "v2-reputation"`.

**Problem.** Answer scores range from −26 to 6,095 (median 1, P99 ≈ 46, 11,925 negative). With global min-max, 90% of answers have `minmax(score_A) < 0.0053`, so the upvote signal is essentially zero inside trustScore. Current `trustScore_A`: median 0.189, p90 0.495, min 0.0017, max 0.984. This is inconsistent with the thesis rationale: reputation is already log-normalised *because* its distribution is skewed; vote score has the same problem but was not treated the same way.

---

## 2. Decision: Amendment 6

> ⚠️ **CONFIRMATION GATE.** All four points below must be confirmed by me in chat before any `--apply`. Implement them as written, but in your final report list them explicitly as "awaiting confirmation", with the actual P99 values from the dry-run.

1. **Formula.** `norm_log(score) = clip(log1p(max(score, 0)) / log1p(P99_score), 0, 1)`. Negative scores → 0.
2. **P99 population.** P99 is computed over the **full population of the same node type**: all Answers for `trust_A` (expected ≈ 46), all Questions for `trust_Q`, each separately. Compute on the raw `score` column (negatives included), state the exact quantile method used (e.g., DuckDB `quantile_cont` / numpy `linear`), and use the **same** method in build and apply. Write both actual P99 values to the log. If a P99 ≤ 0, raise an error.
3. **Weights unchanged.** A: 0.5 / 0.3 / 0.2. Q: 0.7 / 0.3. `w_auth` unchanged. The `views` term in `trust_Q` (`minmax(log1p(views))`) is **unchanged** — only the score term changes. Flag this in your report so I can confirm it.
4. **trust_Q also changes** for consistency, even though it is not used in Condition C ranking. You **must verify by grep** that `trust_Q` / Question `trustScore` is not used in Conditions A–D, evaluation scripts, the backend, or reports (see Task 0.3). If it is used anywhere, stop and report the file:line.

**Resulting formulas:**

```
trust_Q = 0.7 · norm_log(score_Q; P99_Q) + 0.3 · minmax(log1p(views))
trust_A = 0.5 · norm_log(score_A; P99_A) + 0.3 · is_accepted + 0.2 · w_auth
trustScoreVersion = "v3-logscore"
```

**Justification (must appear in the decision record, in these terms):** a measurement-validity correction based on the score distribution in the data, **not** on hallucination results. It is made **before** stage 2 and **before** the test sample has been touched.

---

## 3. Tasks (in order)

### Task 0 — Audit existing uncommitted work (do this FIRST)

0.1 Run `git status` and `git diff`. Expected uncommitted changes from another session:
- modified: `7_build_knowledge_graph.py`, `docs/DECISION_C_SCORING.md`, `stage2_report.py`, `test_stage2_report.py`
- new: `14_apply_logscore_trust.py`, `trust_formulas.py`, `logscore_stage1_recheck.py`, `tests/test_logscore_trust.py`

Locate each file's actual path (use `git status` output; don't assume directories). If the set differs from this list, report the difference.

0.2 Read every changed/new file fully and audit it against Section 2 and Tasks 1–7. Produce a table:

| File | Correct | Wrong | Missing | Action |
|---|---|---|---|---|

Check specifically: negative → 0 before `log1p`; clip to [0,1]; P99 per node type over the full population; weights unchanged; `views` term unchanged; `w_auth` unchanged; build and apply import the **same** function (no duplicated formula); version string `"v3-logscore"`; no edits to `HAS_ANSWER` weights.

0.3 Grep for every consumer of trust values and report results (file:line):

```
grep -rn "trustScore" --include="*.py" --include="*.ts" --include="*.tsx" --include="*.cypher" .
grep -rn "trust_Q\|trust_q\|q_trust\|question_trust" --include="*.py" .
grep -rn "trustScoreVersion\|v2-reputation" --include="*.py" --include="*.md" .
grep -rn "TRUST_V2_APPLIED_AT" .
```

Classify each hit: (a) Answer trust used in ranking, (b) Question trust used anywhere, (c) write/build path, (d) docs/tests. Also look for **cached copies** of trustScore (parquet/JSONL candidate caches, dev-set feature files) that would go stale after apply, and list them.

0.4 Read `13_apply_reputation_trust.py`, `reputation_ablation_check.py`, `reputation_stage1_recheck.py`, and `stage2_report.py` (the pre-change version via `git show HEAD:<path>`) to learn the patterns you must follow.

**After Task 0, send me the audit in Indonesian and continue** unless you found a contradiction under Hard Rule 8.

### Task 1 — Amendment 6 in `docs/DECISION_C_SCORING.md`

Follow the format of Amendment 5. Include: problem statement with the numbers from Section 1; old vs new formulas; the four decision points (marked "pending confirmation" until I confirm); justification as in Section 2; scope (Answer and Question trustScore; `HAS_ANSWER` weights untouched); expected effect (dry-run distribution old vs new); and a **Record** subsection with empty placeholders to be filled **after** apply: apply timestamp (ISO 8601 with timezone), P99_A, P99_Q, quantile method, rows updated, backup path, verify result (max diff), git commit hash.

### Task 2 — Single source of truth: `trust_formulas.py`

- Pure, side-effect-free, vectorised (numpy/pandas) functions, e.g. `norm_log(score, p99)`, `compute_p99(scores, method=...)`, `trust_answer(...)`, `trust_question(...)`, plus constants for weights and `TRUST_SCORE_VERSION = "v3-logscore"`.
- Keep `w_auth` logic here too if it isn't already, so all trust math lives in one module.
- `7_build_knowledge_graph.py` (`compute_trust_scores`) and `14_apply_logscore_trust.py` must both import from it. No duplicated formula anywhere.
- A fresh KG build and the apply script must produce **bit-identical** float64 values from the same parquet source.

### Task 3 — `14_apply_logscore_trust.py`

Mirror `13_apply_reputation_trust.py`:
- Modes: `--dry-run` (default), `--apply`, `--restore`, `--verify-applied`. `--apply` and `--restore` must require an explicit flag; no interactive defaults that write.
- **Dry-run:** compute new values in memory; report old vs new distribution (min / median / p90 / max, plus count at 0 and at 1) for Answer and Question separately; report P99_A and P99_Q and the quantile method; report how many nodes would change. No writes.
- **Apply:** check free disk space first; back up current `trustScore` + `trustScoreVersion` (node id → values) to a timestamped parquet before writing; batch write to Neo4j (`UNWIND` batches, size consistent with script 13); set `trustScoreVersion = "v3-logscore"`; after writing, read back and verify **max |diff| = 0** against computed values; print the apply timestamp to paste into the decision record and `stage2_report.py`.
- **Restore:** restore from a given backup parquet (explicit path argument), then verify max diff = 0 against the backup.
- **Verify-applied:** read-only; recompute and compare with Neo4j; report max diff and version counts.
- Timestamped log file in the same log directory/style as script 13. Do not add new node properties beyond what script 13 writes.

### Task 4 — Read-only top-5 impact check (pattern: `reputation_ablation_check.py`)

- Dev set only. Compare Condition C top-5 contexts under **v2** trust (from the backup parquet or recomputed v2 formula) vs **v3** trust, at α = 0.25 and α = 0.5. Also run α = 0 as a sanity check: **0 changes expected** (pure similarity); assert it.
- Report per α: number of dev questions whose top-5 set changed, mean number of items swapped, mean Jaccard of top-5 sets, and a few example question IDs.
- No writes to Neo4j, no API calls. It must also work before apply (compute v3 in memory via `trust_formulas.py`), but my planned order runs it after verify.

### Task 5 — `logscore_stage1_recheck.py` (pattern: `reputation_stage1_recheck.py`)

- Re-run stage 1 for α ∈ {0, 0.25, 0.5, 0.75, 1} on the dev set with v3 trust.
- **Reuse cached judge labels.** Cache key must identify the item exactly (e.g., dev question id + answer/context id + judge model + judge prompt version/hash). Only items not in the cache are new and need judging. Report how many new items there are per α **before** any judge call.
- Judge calls are behind an explicit flag (e.g. `--run-judge`) and I will run it myself. Default mode = plan/estimate only (count of new items, estimated calls).
- Apply the **existing** stage-1 decision rule unchanged (import or reuse it; do not re-implement differently). Output in the same format as the reputation recheck so results are comparable.

### Task 6 — `stage2_report.py`

- Replace `TRUST_V2_APPLIED_AT` with a v3 constant (e.g. `TRUST_V3_APPLIED_AT`). Only stage-2 runs that **started after** the v3 apply are valid.
- Until I fill it in after apply, the constant must be a placeholder that makes the report **fail closed** (clear error, no report), not silently include everything.
- Compare timestamps **timezone-aware** (my machine is Asia/Jakarta, UTC+7; check what format run logs use and normalise both sides).
- Also check runs' recorded `trustScoreVersion` if run metadata has it; treat mismatch as invalid and report it.

### Task 7 — Unit tests (pytest, fakes only — no Neo4j, no network)

Cover at least:
- `norm_log`: negative → 0; 0 → 0; score = P99 → 1; score > P99 → clipped to 1; monotonic; NaN handling (state the chosen behaviour).
- `compute_p99`: matches the documented quantile method on a known array; raises if P99 ≤ 0.
- Weights: `trust_A` and `trust_Q` match hand-computed values on small fixtures; result within [0,1].
- **Build vs apply equivalence:** the build path and the apply path produce identical arrays (`np.array_equal`) on the same fixture.
- Apply script logic with a fake Neo4j session: dry-run performs zero writes; apply writes backup before any write; restore writes back exactly the backup values; verify detects a non-zero diff.
- `stage2_report`: placeholder → fails closed; run before apply excluded; run after apply included; timezone edge case (same instant in UTC vs +07:00); version mismatch excluded.
- Top-5 check: α = 0 gives zero changes on a fixture.

Run the whole suite. **All existing tests must still pass.** If an old test breaks because of an intended change, explain it and ask before editing it.

---

## 4. Final report (in Bahasa Indonesia, simple and step by step)

1. **Audit result** of the pre-existing uncommitted work (the table from Task 0.2, short).
2. **Grep result** for trust_Q usage and stale caches (Task 0.3).
3. **Files changed/created**, one line each on what changed.
4. **Test results** (counts passed/failed, command used).
5. **Dry-run output**: P99_A, P99_Q, quantile method, old vs new distributions.
6. **Points awaiting my confirmation** — the four Amendment 6 points, plus the `views` term staying unchanged, with the actual numbers.
7. **Commands for me to run, in order** (zsh, no inline comments): dry-run → (I confirm) → apply → verify-applied → top-5 check → stage-1 recheck plan → stage-1 recheck with judge → fill the Record in `DECISION_C_SCORING.md` and the timestamp in `stage2_report.py`.
8. **Open decisions** I need to make, and anything you found that contradicts this prompt.

Do not commit. Leave all changes uncommitted for my review.
