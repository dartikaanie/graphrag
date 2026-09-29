# GraphRAG Thesis Project — README

Master's thesis (Informatics, ITB): **"Development of a GraphRAG Framework Based on
Stack Overflow Community Knowledge to Reduce LLM Hallucination"**, supervised by
Dr. Ir. Arry Akhmad Arman, MT.

This document is a running log of what has actually been built in this repo, kept
up to date as work progresses. For the dashboard's detailed design spec, see
[`PLAN_UI_UX.md`](./PLAN_UI_UX.md) in this same folder.

---

## 1. What this project is

The thesis compares four conditions for answering Stack Overflow-style
programming questions with an LLM, using the SORD dataset (~2.68M questions):

- **Condition A — Pure LLM baseline** (`llm/a_pure_llm/`): replicates Da Silva,
  Samhi & Khomh (2025) — a single LLM call per question, no retrieval.
- **Condition B — Conventional RAG** (`llm/b_rag/`): Condition A + dense
  retrieval (FAISS) over a flat chunked corpus of SO answers.
- **Condition C — GraphRAG** (`llm/c_graphrag/`): Condition A + hybrid
  retrieval over a trust-weighted Knowledge Graph in Neo4j — vector anchoring →
  graph traversal → semantic expansion — with enforced `[SO-<id>]` citation
  grounding. This is the thesis's novel contribution, benchmarked against A and B.
- **Condition D — Dual-Level Retrieval (LightRAG-adapted)** (`llm/d_lightrag/`):
  an adaptation of LightRAG's (Guo et al.) dual-level (low-level + high-level)
  retrieval mechanism, running on the exact same Neo4j knowledge graph and
  FAISS cache as Condition C (no new graph, no LLM-based entity extraction) —
  isolates the contribution of trust-weighted fusion specifically, by swapping
  in a trust-free ranking mechanism over identical data.

All four conditions share the exact same 4-turn prompt skeleton
(`llm/prompts.py`) so that presence/absence/type of retrieval — and, for C/D,
the grounding constraint specifically — is the only variable. That's the
controlled comparison the thesis is built on. A full write-up of each
condition's design, and a side-by-side comparison table, lives on the
dashboard's **Methodology** page (`/methodology`) — see §7.

Beyond the four conditions' own cosine-similarity/NF2/NF3 metrics, a
separate post-hoc evaluation layer (LLM-as-Judge for Faithfulness/Answer
Relevance/Hallucination Rate, plus retrieval Precision@k/MRR@k) can be run
on top of any already-completed result file without re-running the
condition itself — see §6.

A web dashboard (`backend/` + `frontend/`) has been built on top of this
research engine so runs, results, history, and the knowledge graph can be
demoed/driven from a browser instead of the terminal. See §7 below and
`PLAN_UI_UX.md` for the full spec.

---

## 2. Repository layout

```
00_datasource/          Raw SORD CSV/XML dumps + merged Parquet (gitignored, large)
01_data_cleaning/       Data pipeline: merge → dedup → EDA → KG build → densify
llm/                    The research engine (main project)
  client_factory.py       LLM provider abstraction (openai/anthropic/local/ollama)
  prompts.py               Shared 4-turn message builders for Condition A/B/C/D
  citations.py             Shared [SO-<id>] citation extraction/validation (B/C/D)
  a_pure_llm/              Condition A script + its results/logs
  b_rag/                   Condition B script + its results/logs/index_cache
  c_graphrag/              Condition C script + its results/logs
  d_lightrag/              Condition D script + its results/logs
  evaluation/              Post-hoc LLM-as-Judge script + its own results/logs (see §6)
backend/                FastAPI dashboard backend
frontend/               React (Vite) dashboard frontend
config/                 One-off diagnostic scripts (accepted-answer match, KG feasibility, LLM connection test)
report/                 Generated EDA/integrity reports (markdown + charts)
docs/                   This file + PLAN_UI_UX.md
run.py                  Interactive CLI launcher (menu wrapper around the scripts above)
compare_condition_c_runs.py   Compares run_history.jsonl across A/B/C/D (+ judge_run_history.jsonl) side by side
analyze_retrieval_quality.py  Precision@k/MRR@k (tag-overlap proxy) purely from existing B/C/D result files, no new LLM calls
```

`llm/a_pure_llm`, `llm/b_rag`, `llm/c_graphrag` were originally named
`02_baseline_replication`, `03_rag`, `04_graphrag` and lived at repo root;
they were moved and renamed into `llm/` to group the actual research engine
into one place, separate from the dashboard's `backend/`/`frontend/`.

---

## 3. Data pipeline (`01_data_cleaning/`)

Run in order (each is also wired into `run.py`'s menu):

1. `1_preview_files.py` — quick preview of raw SORD CSVs.
2. `2_check_data_integrity.py` — validates row counts vs. the SORD paper, headers, duplicate IDs.
3. `3_merge_sord_sources.py` — merges Contain + LikeMinusContain sources into `*_raw_union.parquet`.
4. `3b_dedup_merged_sources.py` — dedups the merged Parquet by `Id` (run once after every merge).
5. `4_test_infra.py` — validates Neo4j + FAISS connectivity before KG construction.
6. `5_eda_trust_signals.py` — 19-chart EDA on data quality, content, trust signals, time, KG feasibility.
7. `6_postlinks_to_sord.py` — enriches SORD with Question↔Question `Linked`/`Duplicate` edges from StackExchange `PostLinks.xml`.
8. `7_build_knowledge_graph.py` — builds the base Neo4j KG (see schema in §4).
9. `8_convert_users.py`, `9_load_author_trust.py` — loads `User` nodes and `AUTHOR_TRUST` edges.
10. `10_densify_tag_cooccurrence.py`, `11_densify_embedding_similarity.py` — adds `TAG_COOCCUR` and `EMBED_SIM` edges to densify the graph beyond the sparse base structure.
11. `12_validate_densification.py` — sanity-checks the densification results.

Diagnostic one-offs live in `config/`: `check_accepted_answer_match.py`
(accepted-answer coverage in the Answers Parquet) and
`analyze_kg_feasibility.py` (tag/comment/score distribution of the matched
subset).

---

## 4. Neo4j Knowledge Graph schema

**Nodes**

| Label | Key properties |
|---|---|
| `Question` | `id`, `title`, `body`, `score`, `viewCount`, `creationDate`, `domainTag`, `trustScore` |
| `Answer` | `id`, `body`, `score`, `isAccepted`, `authorReputation`, `trustScore` |
| `Tag` | `name`, `questionCount` |
| `User` | `id`, `reputation` |

**Edges** (trust-weight hierarchy matters — it's a visual/methodological
argument in the thesis defense)

| Edge | Direction | Weight | Source |
|---|---|---|---|
| `HAS_ACCEPTED_ANSWER` | Question → Answer | 1.0 | base KG |
| `HAS_ANSWER` | Question → Answer | variable | base KG |
| `TAGGED_WITH` | Question → Tag | 1.0 (structural) | base KG |
| `IS_RELATED_TO` | Question → Question | variable, `link_type` Linked/Duplicate | PostLinks enrichment |
| `AUTHOR_TRUST` | Answer → User | variable | base KG |
| `TAG_COOCCUR` | Question → Question | 0.6 (categorical), `jaccard` (raw) | densification |
| `EMBED_SIM` | Question → Question | 0.4 (categorical), `cosine_sim` (raw) | densification |

Current live counts (from the dashboard's `/api/stats/summary`): **2,685,814**
Questions, **725,795** Answers, **56,435** Tags, **32,100,676** edges total.

Neo4j indexes added while building the dashboard (Phase 1, see §7): range
indexes on `Question.score`, `Answer.score`, `Tag.questionCount`, plus
full-text indexes `question_title_fts` and `answer_body_fts` — needed
because unindexed `ORDER BY`/`CONTAINS` queries over a 2.7M-node graph would
otherwise hang (see the "lessons learned" note in §8).

---

## 5. The four experiment conditions

Each condition script (`a_baseline_replication.py`, `b_condition_b_rag.py`,
`c_graphrag.py`, `d_lightrag.py`) is resumable (JSONL append + skip-already-
done), reads config from the repo-root `.env`, auto-names its output file
from provider+model+n+seed(+ablation variant), and logs a per-run summary to
its own `logs/run_history.jsonl`.

- **Condition A**: sample 384 questions (95% CI / 5% margin, matching the
  original paper) → 1 LLM call per question → cosine similarity (MiniLM) vs.
  the accepted answer.
- **Condition B**: same sampling, + FAISS top-k retrieval over a flat chunked
  corpus of SO answers. Citation labeling/validation is optional
  (`--require-citation`, same `[SO-<id>]` format as C/D) for a direct NF2
  comparison, but never includes a grounding constraint.
- **Condition C**: same sampling, + hybrid retrieval — vector-anchor into the
  KG, 1–2 hop trust-weighted graph traversal, semantic expansion — fused into
  a `[SO-<id>]`-labeled context under a dual constraint (grounding + mandatory
  citation). Measures NF2 (citation compliance) and NF3 (retrieval latency
  ≤15s) alongside cosine similarity. Ships with three independent ablation
  switches: `--fusion-mode {trust_weighted,uniform}` (isolates trust-
  weighting itself), `--no-require-grounding` (isolates the grounding
  constraint from the citation constraint), and
  `--no-enable-semantic-expansion` (isolates the semantic-expansion stage).
  **`c_retrieval_version`** in `run_history.jsonl`: `"v2"` (current) once
  deterministic tie-breaks were added (see below); absent entirely on
  records predating this field (equivalent to unversioned "v1").
- **Condition D**: same sampling and KG/FAISS cache as Condition C, but a
  dual-level retrieval mechanism (adapted from LightRAG, Guo et al.) instead
  of trust-weighted fusion — low-level (1-hop from vector-search anchors) +
  high-level (2-hop via tag/relatedness/embedding-similarity edges), ranked
  purely by relevance score, never trust. Shares Condition C's
  `--require-grounding` ablation switch so the grounding-constraint
  experiment can be run identically on both graph-based conditions; semantic
  expansion is not implemented here (only two retrieval levels, by design).
  **`d_retrieval_version`** in `run_history.jsonl`: `"v1"` (buggy, see
  below), `"v2"` (EMBED_SIM added, LIMIT-before-join bug still present),
  `"v3"` (current — both retrieval fixes below); absent entirely on records
  predating the field.

  **Bug fix 1 — missing EMBED_SIM (found during n=10 pilot diagnosis, before
  the real n=384 run):** `retrieve_high_level()`'s 2-hop Cypher only matched
  `IS_RELATED_TO|TAG_COOCCUR` — its own docstring said the pattern was
  copied from Condition C's `traverse_graph()` "bagian
  IS_RELATED_TO|TAG_COOCCUR|EMBED_SIM", but `EMBED_SIM` was missing from
  the actual query. For any anchor whose only outgoing Question↔Question
  edges are `EMBED_SIM` (common — `TAGGED_WITH`/`EMBED_SIM` are frequently
  an anchor's *only* edges), high-level retrieval silently returned zero
  candidates even when Condition C, querying the same anchors with
  `EMBED_SIM` included, found several relevant ones.

  **Bug fix 2 — LIMIT before the answer join:** even with fix 1,
  `retrieve_high_level()`'s `ORDER BY relevance_score DESC LIMIT
  $n_high_level` still ran *before* the join that checks whether a
  candidate actually has an answer — and `EMBED_SIM` edges all carry the
  same flat weight (0.4), so with scores tied, `LIMIT` could pick
  candidates that happen to have zero answers while candidates ranked just
  outside the limit did have one. Fixed by restricting to
  answer-having questions (`EXISTS { ... }`) *before* `ORDER BY`/`LIMIT`.
  Traced read-only against live Neo4j (no LLM calls) on the two n=10 pilot
  questions that hit both bugs: before either fix, 56612920 → 0 candidates
  and 3882147 → 0 candidates; after fix 1 alone, 56612920 → 2 and 3882147
  → 0 (still broken by bug 2); after both fixes, 56612920 → 5 and 3882147
  → 3.

  **Determinism fix — deterministic tie-breaking:** every Cypher query with
  `ORDER BY`/`LIMIT` in both C (`traverse_graph`'s 2-hop) and D
  (`retrieve_high_level`), plus the two 1-hop queries that had no `ORDER
  BY` at all (C's `traverse_graph` 1-hop, D's `retrieve_low_level`), now
  sort on an explicit secondary key (`a.id ASC`, or `q2.id ASC, a.id ASC`
  for D's high-level). The Python-side final-ranking sorts
  (`fuse_and_rank()` in C, `fuse_dual_level()` in D) now use
  `(-score, answer_id)` as the sort key instead of score alone — without
  this, Python's *stable* sort falls back to whatever order Neo4j happened
  to return matching rows in, which is not guaranteed stable across
  identical query executions. **Proven, not assumed**: each condition's
  full retrieval pipeline was run twice, read-only, on the 10 n=10 pilot
  `question_id`s, with no LLM calls — `retrieved_context` came back
  byte-identical across both runs, for all 10 questions, both conditions.
  **This changed C's retrieved context for 7 of the 10 pilot questions**
  (some just reordered, some with a genuinely different item surviving the
  by-`answer_id` dedup step on a tied score) even though C had no
  correctness bug — ties were simply resolved by an undefined Neo4j return
  order before, and by `answer_id` now. Ranking/trust logic itself was not
  changed in either condition.

See the dashboard's **Methodology** page (`/methodology`) for the full
design write-up per condition and a side-by-side comparison table, or read
the top-of-file docstring in each script above for exact implementation
detail (leakage-exclusion queries, prompt text, CLI flags).

`run.py` is a menu-driven launcher that wraps all of the above (plus the data
pipeline scripts) so they can be run without remembering exact CLI flags.
`compare_condition_c_runs.py` reads all four `run_history.jsonl` files
(`--condition ABCD`, or any subset e.g. `CD`) and prints a side-by-side
comparison table across conditions/models/runs.

**Verified working** (Sept 2026, n=1 smoke tests, gpt-4o-mini): all four
conditions run end-to-end — Condition A similarity 0.634, Condition B 0.634
(retrieval + FAISS cache load OK), Condition C 0.650/0.707 across
trust_weighted/uniform fusion-mode ablation runs (Neo4j + KG workspace +
citation validation OK, NF2/NF3 both PASS), Condition D verified via the
dashboard's `/api/runs` endpoint (dual-level retrieval + grounding-toggle
wiring confirmed end-to-end, cancelled before an LLM call to avoid pilot
cost — see the run.py/dashboard smoke-test notes in git history for detail).

---

## 6. Post-hoc evaluation: LLM-as-Judge & retrieval quality (`llm/evaluation/`)

Two evaluation layers run entirely on top of ALREADY-COMPLETED result
files — neither re-runs a condition's generator, and neither is required
before a condition's own core metrics (cosine similarity, NF2, NF3) are
usable.

**LLM-as-Judge** (`llm/evaluation/llm_judge_hallucination.py`) scores an
existing result file's answers for Faithfulness, Answer Relevance, and a
3-class Hallucination Rate (`FAKTUAL` / `HALUSINASI_SEBAGIAN` /
`HALUSINASI_PENUH`):

- Two judging modes, chosen automatically from `--condition`: `no_context`
  for Condition A (no retrieval exists, so faithfulness is `null` and the
  judge falls back to its own general knowledge — methodologically
  equivalent to a "Fabricated Claim Rate"), and `context_grounded` for
  B/C/D (faithfulness is checked against the exact `retrieved_context`
  the generator actually used, re-read from that file, never re-retrieved).
- The judge prompt explicitly withholds which condition/provider/model
  produced the answer, so a judge can never rate an answer higher just
  because it "knows" it came from the graph-based condition.
- Output is a **separate, new file** in `llm/evaluation/results/`, named
  deterministically from (source file, judge provider, judge model,
  temperature, majority rounds) — never merged back into a condition's own
  result file. Re-running with an identical config detects
  `already_complete` and makes zero LLM calls; a different config always
  writes a different file, so ablation runs never collide.
- `--majority-rounds N` (N > 1) re-judges each question N times at the
  same temperature and takes the majority-vote label (median for the
  numeric scores) — recommended only for a Cohen's Kappa validation
  subsample, not a full batch, since cost scales linearly.
- `--kappa-validation` re-judges a random subsample with a second
  ("secondary") judge and computes Cohen's Kappa inter-rater agreement
  between the two (Landis & Koch bands, collapsed to 4: <0.4 weak, 0.4–0.6
  moderate, 0.6–0.8 substantial, >0.8 almost perfect) — prints an explicit
  warning if the primary and secondary judge are configured identically,
  since that measures self-consistency, not self-enhancement bias.
- Every invocation (including a no-op `already_complete` check) appends one
  line to `llm/evaluation/logs/judge_run_history.jsonl` — this manifest is
  what the dashboard's backend joins against to surface judge results on
  the existing History pages (see §7's Phase 6 note) without a second,
  separate "judge results" page.

**Retrieval quality** (`analyze_retrieval_quality.py`, repo root) computes
Precision@k and MRR@k for Condition B/C/D's `retrieved_context` — purely
from files already on disk, zero new LLM calls. Relevance is a **tag-
overlap (Jaccard) proxy**, not human-labeled ground truth, and the
script says so both in its docstring and in its printed output — this is
an explicit, documented limitation for the thesis's Bab V, not a hidden
assumption. Recall@k is deliberately **not** computed: it requires knowing
the full candidate pool before the top-k cutoff, which existing pilot runs
never recorded. An opt-in `--log-full-candidates` flag was added to
`c_graphrag.py`/`b_condition_b_rag.py` for *future* runs to capture that
list (`all_candidate_question_ids`), enabling real Recall@k in a later
version of this script — it is not implemented yet.

Both tools are wired into `run.py`'s menu and into the dashboard: Settings
gains a "LLM-as-Judge Configuration" section (judge/secondary-judge
provider+model+temperature, Kappa sample size, majority rounds — reusing
the same API keys as the generator providers, not a separate secret), and
`/evaluation/judge` triggers a judge run from the browser (condition → file
picker narrowed by provider/model/n_sample/seed, config pre-filled from
Settings and overridable per run, a `force` toggle, and — if the exact
same source file + config was already judged — the page shows the cached
result and offers "re-judge" instead of a plain Run button). Results
appear automatically on the existing History Detail page for the judged
run (a new "Hasil LLM-as-Judge" table) and on its per-question detail page
(a "Penilaian Judge" section) — there is no separate results page to
check.

**Context-Relevance Judge** (`llm/evaluation/llm_judge_context_relevance.py`,
Condition B/C/D only) and **Answer-Relevance Judge**
(`llm/evaluation/llm_judge_answer_relevance.py`, all four conditions) fill
the two remaining dimensions of Kamalipour, Asadi & Amiri Chimeh's (2026,
*Computer Science Review* 61, 100925, §6.1) four-dimension RAG evaluation
framework — retrieval relevance and answer relevance — that neither the
tag-overlap proxy nor the original hallucination judge actually measures
from text content. Both share small helpers in `llm/evaluation/
_judge_common.py` (output-path/resume/manifest conventions, HTML
stripping for question bodies pulled from `QUESTIONS_PARQUET`, Cohen's
Kappa) without touching `llm_judge_hallucination.py` at all.

- **Why an LLM judge for context relevance, not query↔context cosine
  similarity**: for Condition B that would be circular — FAISS already
  ranks its top-k by that exact cosine similarity, so "evaluating" it with
  the same metric it was optimized for carries no independent
  information. Conditions C/D aren't purely cosine-ranked either (graph
  traversal + trust weighting), so a metric usable across all three has
  to judge the retrieved *text content* against the question, independent
  of whatever ranking mechanism produced it.
- Context-Relevance Judge makes **two separate LLM calls per question**:
  a reference-free per-item call (`RELEVAN` / `SEBAGIAN` / `TIDAK_RELEVAN`
  for each retrieved chunk, judged only against the question — never
  shown the reference answer, so per-item labels can't be contaminated by
  already knowing "the right answer") and a reference-aware sufficiency
  call (`CUKUP` / `SEBAGIAN` / `TIDAK_CUKUP` — does the retrieved context,
  taken as a whole, contain what's needed for the reference answer's key
  solution). The prompt for both calls contains **only `chunk_text`** —
  never `trust_weight`/`combined_score`/`score`/`relevance_score`/
  `source_stage`/`hop`/`rel_type`/`is_accepted`, and never which
  condition/provider/model produced the retrieval — so a judge can never
  rate an item relevant just because it knows Condition C's trust
  weighting ranked it highly. Zero-context questions make no LLM call at
  all (`status: "no_context"`). Derived per-question metrics
  (`context_precision_strict`/`_lenient`, `first_relevant_rank`,
  `reciprocal_rank`) are computed from the item labels, not re-judged.
- **Why a new reference-free Answer-Relevance metric, when
  `llm_judge_hallucination.py` already returns `answer_relevance_score`**:
  that existing score is produced in the *same prompt call* that also
  shows the reference answer and the hallucination rubric — it is
  reference-conditioned, not an independent relevance measurement. A judge
  that has already read "the correct answer" can blend "does this address
  what was asked" with "is this consistent with the reference," which are
  conceptually different (an answer can be highly relevant yet wrong, or
  correct yet not actually address the question). Answer-Relevance Judge
  is given **only the question and the answer** — no reference answer, no
  retrieved context, no hallucination rubric in the same call — making it
  the answer-relevance metric this thesis treats as primary;
  `answer_relevance_score` from the hallucination judge is **not
  removed or changed** and remains a secondary/complementary signal.
- Both judges print an explicit **self-judging warning** to stderr (not a
  hard stop) if the judge model matches the `llm_model` of records being
  judged, since that risks self-preference bias (a judge rating answers
  from its own model family more favorably).
- Output files (`ctxrel__*.jsonl`, `ansrel__*.jsonl`) follow the exact
  same deterministic-naming / resume / `already_complete` conventions as
  the hallucination judge, but write to **separate manifests**
  (`context_relevance_run_history.jsonl`, `answer_relevance_run_history.
  jsonl`) rather than `judge_run_history.jsonl`, so the existing dashboard
  join (§7) is completely unaffected by these additions.

**Error Attribution** (`llm/evaluation/analyze_error_attribution.py`) and
**Trust vs Relevance** (`llm/evaluation/analyze_trust_vs_relevance.py`)
are pure joins/statistics over already-judged output — zero new LLM
calls. Error Attribution auto-resolves the latest hallucination/ctxrel/
ansrel judge output per source file from their manifests (matched on
resolved `input_path`) and applies deterministic rules (documented in the
script's own docstring, meant to be quoted directly in Bab III) to
attribute each hallucinated answer to retrieval failure, entity-
anchoring/coverage failure, or generation failure — Condition A gets a
simpler hallucination × answer-relevance outcome only, since it has no
retrieval stage to attribute anything to. Trust vs Relevance joins
`item_labels` back to `retrieved_context` on `(question_id, answer_id)`
and reports Spearman/Mann-Whitney at the **item level**, plus a
**question-level** Wilcoxon test on the per-question (mean trust of
relevant items − mean trust of non-relevant items) — the question-level
view exists specifically because items *within* one question are not
independent observations, so item-level significance alone would be
anti-conservative (pseudo-replication). An optional `--uniform-path`
adds the `--fusion-mode uniform` ablation comparison (paired Wilcoxon +
Cliff's delta on `context_precision_strict`).

**Compare Conditions Stats** (`llm/evaluation/compare_conditions_stats.py`)
is the paired statistical design this thesis needed and didn't have
before: for every pair of supplied conditions, it pairs on the
`question_id` intersection (reporting the paired *n* explicitly, since
not every question has every metric) and runs `scipy.stats.wilcoxon`
(two-sided, `zero_method="wilcox"` — ties/zero-differences are common on
the small-scale ordinal metrics here, so they're dropped from the test
rather than rank-included) plus Cliff's delta (Romano et al. bands) per
metric per pair, with Holm–Bonferroni correction applied **per metric**
across all its comparisons (both raw and adjusted p-values reported).
`hallucination_ordinal` (FAKTUAL=0/HALUSINASI_SEBAGIAN=1/
HALUSINASI_PENUH=2) is the one metric where **lower is better** — the
opposite direction from every other metric in the same table, called out
explicitly in the script's output.

Cliff's delta and the Holm correction live in a tiny shared
`llm/evaluation/_stats_common.py` (no LLM calls), used by both
Trust vs Relevance and Compare Conditions Stats.

All five new tools are wired into `run.py`'s menu (same
`--input-path`/manual-args pattern as the existing two). **Dashboard
integration is future work** — these five tools write to their own
result files and manifests, so the backend/frontend are completely
unaffected by their addition; surfacing their output on the dashboard
(similar to §7's judge-results tables) has not been built yet.

---

## 7. Dashboard (`backend/` + `frontend/`)

Full design spec: [`PLAN_UI_UX.md`](./PLAN_UI_UX.md). Built phase-by-phase,
pausing for review after each phase.

**Phase 1 — Backend skeleton (done).** FastAPI + Neo4j + DuckDB. Endpoints for
Questions/Answers/Tags (list, detail, search, pagination) and
`/api/stats/summary`. Built and fixed against the *live* 2.7M-node graph, not
just written and assumed correct — see the lessons-learned note in §8.

**Phase 2 — Frontend skeleton (done).** Vite + React + TypeScript + Tailwind
v4, design tokens from `PLAN_UI_UX.md` §2.1 (flat, blue-accent, no "AI
slop"). Sidebar layout, routing, Home (stats cards), Question/Answer/Tag list
+ detail pages, generic `AttributePanel`, `DataTable`, `Pagination`,
`SearchInput`.

**Phase 3 — Graph visualization (done).** `/api/graph/partial` (top-N
questions by score + their answers/tags) and
`/api/graph/node/{question|answer|tag}/{id}?hops=1|2`, every query
explicitly capped (no unbounded traversal — see §8). `GraphView`
(react-force-graph-2d) + `EdgeLegend` wired into Home and all three detail
pages: nodes show a short `Q#<id>`/`A#<id>` label on-canvas with full detail
(title, score, trust, accepted status) on hover; the edge-type legend sits
below the graph so it never requires horizontal scrolling.

**Phase 4 — Engine wrapper & Run endpoints (done).** `backend/app/services/
engine_service.py` imports and calls each condition script's own functions
(`get_candidate_questions`, `process_sample`, `append_run_history`, ...)
directly rather than re-implementing retrieval/generation logic — one
`run_condition_x()` per condition, all registered in a `RUNNERS` dict.
`POST /api/runs` (batch or single-question, any condition A–D)
+ `GET /api/runs/{id}/stream` (SSE progress, polls run state so a page
refresh mid-run recovers cleanly) + cancel support. All four conditions
share one `oversample_pool` constant (`DEFAULT_OVERSAMPLE_POOL = 1536`,
exposed via `GET /api/config/defaults` so the frontend never hard-codes it)
so a smaller pilot `n_sample` is always an exact prefix of the official
n=384 sample rather than an independently-drawn one — see §8. A
"Release cached FAISS index" action (Settings page → System,
`POST /api/config/release-faiss-cache`) frees the shared C/D FAISS/
embedding memmap (~4GB) on demand without a backend restart.

**Phase 5 — Frontend Run pages (done).** One `RunConditionPage` per
condition (batch/single mode, all condition-specific parameters — top_k,
n_anchor/n_semantic_expansion, fusion-mode ablation weights,
n_low_level/n_high_level, the grounding-constraint toggle — each with an
inline tooltip explaining what it does, backed by one shared glossary so
wording is identical everywhere a parameter appears), a live
`ProgressRunPanel`, and per-question `RunResultDetailPage` showing the full
prompt transcript, retrieved context (ranked, with an explicit "Rank #N" and
the actual ranking score used), and — for C/D — the real retrieval-path
subgraph actually touched by that run (not a generic node-centered
subgraph). **"Run All" (`/experiment/all`)** fires all four conditions with
identical sampling parameters for a direct apples-to-apples comparison run
— **not actually in parallel**: all four `POST /api/runs` requests are sent
at once, but the backend (`engine_service.py`) queues all of A, B, C, and D
behind one shared lock (`_HEAVY_RUN_LOCK`) so only one condition ever
executes at a time — Condition A also runs a DuckDB sampling query and must
not overlap C/D's ~4GB FAISS footprint on an 8GB machine, so it is not
exempted from this lock. A queued run's status shows `"queued"` (with its
position in the queue) until its turn comes up; submission still uses
`Promise.allSettled` so one condition failing to even start (e.g. a bad
parameter) doesn't block the others from starting. Judge runs
(`POST /api/judge/runs`) use a *separate* bound — up to 2 concurrent judges
via a semaphore, independent of `_HEAVY_RUN_LOCK` by default — plus an
opt-out Settings toggle ("Judges wait while a heavy run is active", default
on) that makes a judge run ALSO wait for the A/B/C/D lock to be free before
starting, on top of the 2-judge cap. See §8 for the full resource-budget
reasoning.

**Phase 6 — History / run comparison (done).** `/api/history` reads each
condition's `run_history.jsonl` directly (CLI runs and dashboard runs are
indistinguishable to this UI) into one unified, paginated, filterable table;
select 2–4 runs (any mix of conditions) to compare parameters and metrics
side by side, with a per-row delete action (removes the transaction-log
entry only — the underlying results `.jsonl` on disk is never touched by a
UI delete). Compare also computes and displays a **sample-consistency
banner** (`GET /api/history/compare/consistency?ids=...`, backed by
`history_service.compute_sample_consistency()`) — green "identical" when
every selected run's `question_id` set matches exactly, blue "nested" when
they're prefixes of each other at different `n` (the expected relationship
for pilot vs. full runs drawn from the same `oversample_pool`/seed), or red
"different" when they aren't comparable samples at all — included in the
page's Markdown export too, so a comparison can't be read at face value
without knowing whether the underlying samples actually match. A dedicated
**Methodology page** (`/methodology`) documents each
condition's design/retrieval mechanism/leakage-prevention approach and a
full A/B/C/D comparison table, so the dashboard is self-documenting rather
than requiring this file to be read alongside it. LLM-as-Judge results
(§6) are joined onto this same History Detail page server-side by
`output_path` — there is no separate "judge results" page to check.

**Phase 7 — Settings (done).** Provider/model/API-key/Neo4j config lives in
a Fernet-encrypted file *outside* the git repo
(`~/.graphrag-dashboard/config.json`, chmod 600) rather than the repo-root
`.env` — see §8 for why this pattern exists. A per-run UI override (e.g.
Condition B's citation toggle) takes precedence over both Settings and
`.env` without persisting the override anywhere.

**Phase 8 — Demo polish (done).** Accepted-answer highlighting + auto-
centering in graph views, edge hover tooltips (type, weight, endpoints) in
every graph visualization, Duration/NF2/latency metrics shown generically
(presence-based, not condition-gated) across History and Run views.

---

## 8. Repo hygiene / lessons learned along the way

A few things surfaced during an audit and while building the dashboard that
are worth keeping visible rather than losing in chat history:

- **`.env` with live `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` was committed to git
  history** at some point and has not been rotated/scrubbed. Not remediated
  by this project's tooling — a decision left to the repo owner. The
  dashboard's Settings page (Phase 7) is designed to not repeat this pattern.
- **`.gitignore` had a malformed line** (two patterns fused:
  `03_rag/logs/*00_datasource/PostLinks.xml`) and `.DS_Store` was tracked in
  6 locations — fixed during the `llm/` folder reorganization.
- **`run.py` referenced a typo'd filename** (`a_baseline_replecation.py`)
  for Condition A, silently breaking that menu item — fixed, and the
  Condition C menu item (previously marked "not built yet" even though the
  script existed) was wired up too.
- **Backend query performance on a 2.7M-node graph is not "just add an
  index and it's fine" — it required real profiling.** Concretely:
  - A stats query chaining `MATCH...WITH...MATCH...WITH` made Neo4j abandon
    its fast label-count-store path; splitting into 4 independent `count()`
    queries fixed it.
  - `ORDER BY score` combined with an aggregate *before* pagination forced a
    full sort of the entire population before returning 5 rows; fixed by
    adding range indexes and restructuring to page first, hydrate second.
  - Unindexed `CONTAINS` search combined with index-ordered pagination could
    scan nearly the whole 2.7M-question population before finding a match;
    fixed with real full-text indexes (`question_title_fts`,
    `answer_body_fts`).
  - Every graph-visualization query (Phase 3) is deliberately capped with
    `LIMIT`/`CALL` subqueries per fan-out — an uncapped 2-hop traversal on
    this graph is not just slow, it can hang for minutes, and a
    force-directed layout is unreadable past a few hundred nodes anyway.
- **Folder rename**: `02_baseline_replication` → `llm/a_pure_llm`,
  `03_rag` → `llm/b_rag`, `04_graphrag` → `llm/c_graphrag`. All relative
  imports (`sys.path.insert` depth), KG workspace paths, `run.py` menu
  entries, and `.gitignore` patterns were updated and re-verified by actually
  executing each condition script's import machinery and by running all
  three end-to-end with n=1, not just by reading the diffs.
- **"Run All Conditions" crashed an 8GB RAM MacBook M2 at n_sample=384.**
  It originally fired A/B/C/D as four fully-parallel background threads
  (`threading.Thread` per condition, no coordination) — fine at small pilot
  sizes, but at n=384 the combined footprint (2x Neo4j driver, 2x FAISS
  index + embedding memmap loaded independently for C and D even though
  they read the identical on-disk files, up to 4 concurrent DuckDB
  connections each free to use unlimited memory/threads, 4x embedding
  model) was enough to exhaust memory and hang the machine. Fixed three
  ways in `engine_service.py`, all still allowing A/B/C/D to be *triggered*
  together from the UI: (1) a module-level `threading.Lock` serializes
  actual execution of A/B/C/D to one at a time — Condition A was initially
  left unlocked on the theory that it's materially cheaper and doesn't
  touch Neo4j/FAISS, but it still runs its own DuckDB sampling query and
  was later folded into the same lock so it can't overlap C/D's ~4GB FAISS
  footprint on an 8GB machine either — with
  the run showing status `"queued"` while it waits; (2) Condition D's
  `load_faiss_cache()` reads the exact same files as Condition C's (by
  design — D reuses C's KG), so it's now loaded once via a shared
  `@lru_cache`-wrapped helper instead of twice; (3) every
  `duckdb.connect()` in that file now goes through one helper that applies
  `memory_limit='2GB'`/`threads=2`/`preserve_insertion_order=false`, the
  same pragmas the standalone analysis scripts already used, instead of
  defaulting to "use whatever the machine has." Lesson: a background-thread
  fan-out that's fine to *trigger* concurrently from an API is not
  automatically fine to *execute* concurrently — those are separate
  decisions, and on a resource-constrained target machine the second one
  needs its own explicit bound, not just a for-loop across `RUNNERS`.
- **Condition D's `retrieve_high_level()` had a docstring/implementation
  mismatch that silently zeroed out retrieval for some questions.** The
  docstring said the 2-hop Cypher pattern was copied from Condition C's
  `traverse_graph()` including `EMBED_SIM`; the actual `MATCH` clause only
  had `IS_RELATED_TO|TAG_COOCCUR`. Found by tracing two n=10 pilot
  questions (56612920, 3882147) where D returned zero context but B and C
  both retrieved 5 relevant items for the exact same questions — reading
  D's own comments literally (rather than assuming the code matched them)
  is what surfaced it. Lesson: when a docstring says "copied from X",
  diff the actual query against X, don't just trust the sentence — and
  when a condition scores worse than a baseline on a metric it shouldn't
  structurally lose on, check whether it's a real methodological gap
  before writing it up as one in Bab V.
- **The `[SO-1234]` "hallucinated" citation wasn't a fabrication — it was
  the model copying the prompt's own worked example.** `llm/prompts.py`'s
  citation instructions for B/C/D use `[SO-1234]`/`[SO-5678]`/
  `[SO-1234567]` etc. as illustrative example IDs; when retrieved context
  was too weak/irrelevant to actually cite, especially with smaller local
  models, the model sometimes echoed the example number verbatim instead
  of refusing to cite or citing nothing. A scan of all existing result
  files found 8/772 answers doing this (7 from small local Ollama models,
  1 from gpt-4o-mini). Fixed by making every example ID in the prompt
  non-numeric (`[SO-<id>]`) so it can never look like a real, copyable
  citation — see `PROMPT_VERSION` below. Lesson: any illustrative example
  embedded in a prompt is something the model can and eventually will
  copy literally, especially under instruction pressure ("cite EVERY
  claim") when it has nothing good to actually cite.
- **"Ranking unchanged" is not the same as "reproducible."** Fixing
  Condition D's `LIMIT`-before-answer-join bug (§5) surfaced a second,
  broader issue: several Cypher `ORDER BY ... LIMIT` queries in both C and
  D had no tie-break, and the Python sorts consuming their results used
  score alone as the sort key. None of that is wrong *logic* — Python's
  `sorted()` is stable and the scores are computed correctly — but Neo4j
  doesn't guarantee a stable row order for an unordered `MATCH`, so on a
  tie (common: `EMBED_SIM` edges all carry the same flat 0.4 weight),
  which candidate "won" depended on Neo4j's internal, run-to-run-unstable
  return order rather than on any criterion in the code. This had been
  running unnoticed since C was first built — measured directly:
  re-running C's retrieval (post-fix) against the existing n=10 pilot file
  changed the retrieved context for 6 of the 10 questions (4 re-ordered
  only: 20443560, 411756, 2617170, 56612920; 2 with genuinely different
  retrieved items: 26406581, 51061836; the other 4 — 38118194, 76224221,
  3882147, 11118023 — were unchanged), despite zero change to what
  "better" means. Lesson: whenever a ranking step's key can tie, add an
  explicit secondary key (e.g. `answer_id ASC`) even if the primary
  criterion is "obviously" correct — a stable sort over an unordered
  upstream source is not itself a source of determinism, and the gap only
  shows up as run-to-run variance that's easy to mistake for noise rather
  than a reproducibility bug. Verified with an actual double-run test (not
  just code review) — run each condition's retrieval twice, read-only, no
  LLM calls, and diff the IDs. Bumped `C_RETRIEVAL_VERSION` to `"v2"` and
  `D_RETRIEVAL_VERSION` to `"v3"` so History can tell which fix era a
  stored result belongs to instead of silently comparing pre-fix and
  post-fix runs as if they used the same method.
- **A regex reading a JSONL field by pattern-matching text is fragile even
  when it "always" matches the first occurrence today.**
  `history_service._read_question_ids_cached()` originally used
  `re.compile(r'"question_id"\s*:\s*(-?\d+)').search(line)`, relying on
  `question_id` happening to be the first key serialized in every record —
  true in practice, but never a guaranteed contract, and every record also
  has *nested* `question_id` keys inside `retrieved_context` items that a
  naive `.search()` could just as easily have matched instead. Replaced
  with `json.loads(line)["question_id"]`, which reads the actual top-level
  key regardless of key order or nesting, and added a regression test
  (`test_question_id_reader_uses_top_level_key_not_first_key_or_nested`)
  with `question_id` deliberately placed after other keys and multiple
  nested `question_id` values inside `retrieved_context` that must NOT be
  picked up. Lesson: a regex over already-structured data (JSON, in this
  case) is a bet on a formatting convention holding forever; parsing the
  actual structure costs one line and removes the bet entirely.

---

## 9. Local setup quick reference

Start **Neo4j Desktop** (the local DB the dashboard/Condition C/D connect
to) before running Condition C or D, or "Run All Conditions" — the backend
does not start it for you, and a C/D run will fail immediately without it.
On an 8GB machine, remember that "Run All Conditions" *submits* A/B/C/D
together but *executes* them one at a time (§7/§8) — a run showing
`"queued"` is expected and not stuck; check its queue position rather than
assuming it hung.

```bash
# Backend
cd backend
uvicorn app.main:app --reload --port 8000   # http://localhost:8000/docs

# Frontend
cd frontend
npm run dev                                  # http://localhost:5173

# Any condition, quick smoke test (cheap: n=1)
cd llm/a_pure_llm   # or llm/b_rag, llm/c_graphrag, llm/d_lightrag
python3 <script>.py --n-sample 1 --seed 42
# add --provider ollama --model phi3:mini for a free local test run

# Compare run_history.jsonl across all four conditions
python3 compare_condition_c_runs.py --condition ABCD --group

# LLM-as-Judge on an already-completed result file (no re-run of the condition)
cd llm/evaluation
python3 llm_judge_hallucination.py --input-path ../c_graphrag/results/<file>.jsonl --condition C
python3 llm_judge_hallucination.py --input-path ../c_graphrag/results/<file>.jsonl --condition C \
    --kappa-validation --secondary-judge-provider openai --secondary-judge-model gpt-4o --kappa-sample-size 5

# Retrieval quality (Precision@k/MRR@k) -- zero new LLM calls
cd ../..
python3 analyze_retrieval_quality.py --input-path llm/c_graphrag/results/<file>.jsonl
```

Requires: Neo4j Desktop running (`bolt://localhost:7687`, database
`graphrag`), the SORD Parquet files available at the paths configured in the
root `.env` (currently on an external "T7 Shield" drive), and — for Ollama
smoke tests — `ollama serve` running locally with a model already pulled.
