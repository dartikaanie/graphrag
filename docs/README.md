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
- **Condition D**: same sampling and KG/FAISS cache as Condition C, but a
  dual-level retrieval mechanism (adapted from LightRAG, Guo et al.) instead
  of trust-weighted fusion — low-level (1-hop from vector-search anchors) +
  high-level (2-hop via tag/relatedness edges), ranked purely by relevance
  score, never trust. Shares Condition C's `--require-grounding` ablation
  switch so the grounding-constraint experiment can be run identically on
  both graph-based conditions; semantic expansion is not implemented here
  (only two retrieval levels, by design).

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
refresh mid-run recovers cleanly) + cancel support.

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
subgraph). **"Run All" (`/experiment/all`)** fires all four conditions in
parallel with identical sampling parameters for a direct apples-to-apples
comparison run.

**Phase 6 — History / run comparison (done).** `/api/history` reads each
condition's `run_history.jsonl` directly (CLI runs and dashboard runs are
indistinguishable to this UI) into one unified, paginated, filterable table;
select 2–4 runs (any mix of conditions) to compare parameters and metrics
side by side, with a per-row delete action (removes the transaction-log
entry only — the underlying results `.jsonl` on disk is never touched by a
UI delete). A dedicated **Methodology page** (`/methodology`) documents each
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

---

## 9. Local setup quick reference

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
