# GraphRAG Thesis Project — README

Master's thesis (Informatics, ITB): **"Development of a GraphRAG Framework Based on
Stack Overflow Community Knowledge to Reduce LLM Hallucination"**, supervised by
Dr. Ir. Arry Akhmad Arman, MT.

This document is a running log of what has actually been built in this repo, kept
up to date as work progresses. For the dashboard's detailed design spec, see
[`PLAN_UI_UX.md`](./PLAN_UI_UX.md) in this same folder.

---

## 1. What this project is

The thesis compares three conditions for answering Stack Overflow-style
programming questions with an LLM, using the SORD dataset (~2.68M questions):

- **Condition A — Pure LLM baseline** (`llm/a_pure_llm/`): replicates Da Silva,
  Samhi & Khomh (2025) — a single LLM call per question, no retrieval.
- **Condition B — Conventional RAG** (`llm/b_rag/`): Condition A + dense
  retrieval (FAISS) over a flat chunked corpus of SO answers.
- **Condition C — GraphRAG** (`llm/c_graphrag/`): Condition A + hybrid
  retrieval over a trust-weighted Knowledge Graph in Neo4j — vector anchoring →
  graph traversal → semantic expansion — with enforced `[SO-<id>]` citation
  grounding. This is the thesis's novel contribution, benchmarked against A and B.

All three conditions share the exact same 4-turn prompt skeleton
(`llm/prompts.py`) so that presence/absence/type of retrieval is the only
variable — that's the controlled comparison the thesis is built on.

A web dashboard (`backend/` + `frontend/`) is being built on top of this
research engine so runs, results, and the knowledge graph can be
demoed/driven from a browser instead of the terminal. See §4 below and
`PLAN_UI_UX.md` for the full spec.

---

## 2. Repository layout

```
00_datasource/          Raw SORD CSV/XML dumps + merged Parquet (gitignored, large)
01_data_cleaning/       Data pipeline: merge → dedup → EDA → KG build → densify
llm/                    The research engine (main project)
  client_factory.py       LLM provider abstraction (openai/anthropic/local/ollama)
  prompts.py               Shared 4-turn message builders for Condition A/B/C
  a_pure_llm/              Condition A script + its results/logs
  b_rag/                   Condition B script + its results/logs/index_cache
  c_graphrag/              Condition C script + its results/logs
backend/                FastAPI dashboard backend
frontend/               React (Vite) dashboard frontend
config/                 One-off diagnostic scripts (accepted-answer match, KG feasibility, LLM connection test)
report/                 Generated EDA/integrity reports (markdown + charts)
docs/                   This file + PLAN_UI_UX.md
run.py                  Interactive CLI launcher (menu wrapper around the scripts above)
compare_condition_c_runs.py   Compares run_history.jsonl across A/B/C side by side
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
8. `7_build_knowledge_graph.py` — builds the base Neo4j KG (see schema in §5).
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

Neo4j indexes added while building the dashboard (Phase 1, see §6): range
indexes on `Question.score`, `Answer.score`, `Tag.questionCount`, plus
full-text indexes `question_title_fts` and `answer_body_fts` — needed
because unindexed `ORDER BY`/`CONTAINS` queries over a 2.7M-node graph would
otherwise hang (see the "lessons learned" note in §7).

---

## 5. The three experiment conditions

Each condition script (`a_baseline_replication.py`, `b_condition_b_rag.py`,
`c_graphrag.py`) is resumable (JSONL append + skip-already-done), reads
config from the repo-root `.env`, auto-names its output file from
provider+model+n+seed, and logs a per-run summary to its own
`logs/run_history.jsonl`.

- **Condition A**: sample 384 questions (95% CI / 5% margin, matching the
  original paper) → 1 LLM call per question → cosine similarity (MiniLM) vs.
  the accepted answer.
- **Condition B**: same sampling, + FAISS top-k retrieval over a flat chunked
  corpus of SO answers, inserted into the prompt as generic "Reference N" context.
- **Condition C**: same sampling, + hybrid retrieval — vector-anchor into the
  KG, 1–2 hop trust-weighted graph traversal, semantic expansion — fused into
  a `[SO-<id>]`-labeled context with a dual constraint (grounding + mandatory
  citation). Measures NF2 (citation compliance) and NF3 (retrieval latency
  ≤15s) alongside cosine similarity.

`run.py` is a menu-driven launcher that wraps all of the above (plus the data
pipeline scripts) so they can be run without remembering exact CLI flags.
`compare_condition_c_runs.py` reads all three `run_history.jsonl` files and
prints a side-by-side comparison table across conditions/models/runs.

**Verified working** (Sept 2026, n=1 smoke test, gpt-4o-mini): all three
conditions run end-to-end after the `llm/` folder reorganization — Condition
A similarity 0.634, Condition B 0.634 (retrieval + FAISS cache load OK),
Condition C 0.650 (Neo4j + KG workspace + citation validation OK, NF2/NF3 both PASS).

---

## 6. Dashboard (`backend/` + `frontend/`)

Full design spec: [`PLAN_UI_UX.md`](./PLAN_UI_UX.md). Built phase-by-phase,
pausing for review after each phase.

**Phase 1 — Backend skeleton (done).** FastAPI + Neo4j + DuckDB. Endpoints for
Questions/Answers/Tags (list, detail, search, pagination) and
`/api/stats/summary`. Built and fixed against the *live* 2.7M-node graph, not
just written and assumed correct — see the lessons-learned note in §7.

**Phase 2 — Frontend skeleton (done).** Vite + React + TypeScript + Tailwind
v4, design tokens from `PLAN_UI_UX.md` §2.1 (flat, blue-accent, no "AI
slop"). Sidebar layout, routing, Home (stats cards), Question/Answer/Tag list
+ detail pages, generic `AttributePanel`, `DataTable`, `Pagination`,
`SearchInput`.

**Phase 3 — Graph visualization (done).** `/api/graph/partial` (top-N
questions by score + their answers/tags) and
`/api/graph/node/{question|answer|tag}/{id}?hops=1|2`, every query
explicitly capped (no unbounded traversal — see §7). `GraphView`
(react-force-graph-2d) + `EdgeLegend` wired into Home and all three detail
pages: nodes show a short `Q#<id>`/`A#<id>` label on-canvas with full detail
(title, score, trust, accepted status) on hover; the edge-type legend sits
below the graph so it never requires horizontal scrolling.

**Phase 4 — Engine wrapper & Run endpoints (not started).** Refactor
`a_/b_/c_*.py` into callable functions; `/api/runs` + SSE progress.

**Phase 5 — Frontend Run pages (not started).**

**Phase 6 — History / run comparison (not started).**

**Phase 7 — Settings (not started).** Notably: provider/model/API-key/Neo4j
config must move to an encrypted file *outside* the git repo
(`~/.graphrag-dashboard/config.json`) rather than a repo-root `.env` — see §7.

**Phase 8 — Demo polish (not started).**

---

## 7. Repo hygiene / lessons learned along the way

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

## 8. Local setup quick reference

```bash
# Backend
cd backend
uvicorn app.main:app --reload --port 8000   # http://localhost:8000/docs

# Frontend
cd frontend
npm run dev                                  # http://localhost:5173

# Any condition, quick smoke test (cheap: n=1)
cd llm/a_pure_llm   # or llm/b_rag, llm/c_graphrag
python3 <script>.py --n-sample 1 --seed 42
# add --provider ollama --model phi3:mini for a free local test run
```

Requires: Neo4j Desktop running (`bolt://localhost:7687`, database
`graphrag`), the SORD Parquet files available at the paths configured in the
root `.env` (currently on an external "T7 Shield" drive), and — for Ollama
smoke tests — `ollama serve` running locally with a model already pulled.
