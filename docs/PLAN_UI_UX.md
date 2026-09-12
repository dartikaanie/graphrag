# GraphRAG Thesis Dashboard — Full Plan (Architecture, Backend, Frontend, UI/UX)

> Use this document as the primary instruction (system/task prompt) for a coding agent (Claude Code, Cursor, etc.) that will build this application. Copy the whole document as the initial prompt, then continue iterating phase by phase per the "Implementation Phases" section near the end.

---

## 0. Project Context

I (Dara) am working on a Master's thesis in Informatics at ITB titled **"Development of a GraphRAG Framework Based on Stack Overflow Community Knowledge to Reduce LLM Hallucination,"** supervised by Dr. Ir. Arry Akhmad Arman, MT.

I already have a **research backend engine** (Python) consisting of:
- **Condition A** — `llm/a_pure_llm/a_baseline_replication.py`: pure LLM baseline
- **Condition B** — `llm/b_rag/b_condition_b_rag.py`: conventional dense-retrieval RAG (FAISS)
- **Condition C** — `llm/c_graphrag/c_graphrag.py`: GraphRAG with a trust-weighted Knowledge Graph in Neo4j
- Data source: the SORD dataset (~2.685M Question nodes), stored as **Parquet** (queried via DuckDB for deterministic sampling, `seed=42`) and in **Neo4j** (database `graphrag`, `bolt://localhost:7687`) for graph traversal
- A shared, provider-agnostic `llm/` package: `client_factory.py` (openai / anthropic / local / ollama) and `prompts.py` (per-condition message builders)
- Each run produces evaluation results (cosine similarity, ROUGE, BERTScore, hallucination rate, etc.) that are **currently written to `.jsonl` files**

**Goal of this application**: build a web dashboard that lets me (and my supervisor, during presentations/defense) do the following:
1. Browse master data (questions, answers, tags, etc.) together with their graph relationships, visually
2. Run all three experimental conditions (A/B/C) directly from the UI — both in batch mode (n-sample) and single-question mode — and watch progress and results in real time
3. View the history of all previous runs (from `.jsonl` files) as a clean, drill-down-able "transaction" table
4. Manage configuration (LLM provider, model, API key, other parameters) from a single Settings page

Top priority: **ease of demoing to my supervisor** — the UI must clearly show "what is happening" (retrieval graph, evidence, `[SO-<id>]` citations, A vs B vs C comparison) without me needing to open a terminal or notebook.

---

## 1. Ground Truth Used for This Plan

This plan is grounded in the actual `dartikaanie/graphrag` repository (branch `master`), which I pulled and inspected directly — not generic assumptions. The following facts come from real code.

### 1.1 Neo4j Node Schema (from `01_data_cleaning/7_build_knowledge_graph.py`)

| Label | Properties |
|---|---|
| `Question` | `id`, `title`, `body`, `score`, `viewCount`, `creationDate`, `domainTag`, `trustScore` |
| `Answer` | `id`, `body`, `score`, `isAccepted`, `authorReputation`, `trustScore` |
| `Tag` | `name`, `questionCount` |
| `User` | `id`, `reputation` |

### 1.2 Neo4j Edge Schema (5 from the base KG + 2 from densification)

| Edge | Direction | Properties | Source |
|---|---|---|---|
| `HAS_ACCEPTED_ANSWER` | Question → Answer | `weight=1.0` | Base KG |
| `HAS_ANSWER` | Question → Answer | `weight` (variable) | Base KG |
| `TAGGED_WITH` | Question → Tag | `weight=1.0` | Base KG |
| `IS_RELATED_TO` | Question → Question | `weight`, `link_type` (`Linked`/`Duplicate`), `creation_date` | PostLinks (script 6) |
| `AUTHOR_TRUST` | Answer → User | `weight` | Base KG |
| `TAG_COOCCUR` | Question → Question | `weight=0.6` (categorical), `jaccard` (raw score) | Densification (script 10) |
| `EMBED_SIM` | Question → Question | `weight=0.4` (categorical), `cosine_sim` (raw score) | Densification (script 11) |

Trust-weight ordering note: `HAS_ACCEPTED_ANSWER` (1.0) > `TAG_COOCCUR` (0.6) > `EMBED_SIM` (0.4) — this is an **important visual argument** for the defense: the graph legend must reflect this real trust hierarchy, not arbitrary colors.

### 1.3 Run Output Schema (from `c_graphrag.py`, `b_condition_b_rag.py`, `a_baseline_replication.py`) — fields per `.jsonl` line

| Field | A (baseline) | B (RAG) | C (GraphRAG) |
|---|:---:|:---:|:---:|
| `question_id`, `title`, `tags`, `n_tokens`, `view_count`, `question_score` | ✓ | ✓ | ✓ |
| `accepted_answer_id`, `ground_truth_answer` | ✓ | ✓ | ✓ |
| `llm_answer`, `llm_model`, `cosine_similarity` | ✓ | ✓ | ✓ |
| `retrieved_context` (list, key `chunk_text`, optional `trust_weight`/`hop`/`question_id`) | – | ✓ | ✓ |
| `n_anchors`, `n_graph_candidates`, `n_expansion_candidates` | – | – | ✓ |
| `retrieval_latency_sec` | – | – | ✓ (logs a WARNING if > 15 s — this is NF3) |
| `has_citation`, `cited_source_ids`, `has_valid_citation`, `valid_cited_source_ids` | – | – | ✓ |

Run history is stored in `run_history.jsonl` per condition folder (`llm/a_pure_llm/logs/`, `llm/b_rag/logs/`, `llm/c_graphrag/logs/`), written via `append_run_history()`.

### 1.4 How Runs Currently Work

Each condition is a CLI script (`a_baseline_replication.py`, `b_condition_b_rag.py`, `c_graphrag.py`) with arguments `--n-sample`, `--seed`, `--oversample-pool`, `--provider`, `--model`, `--top-k` (B/C), `--n-anchor`/`--n-semantic-expansion` (C only), plus `run.py` as an interactive menu launcher. This dashboard effectively **replaces `run.py` and the terminal** with a UI, calling the same underlying functions programmatically (not via CLI `subprocess` — see the wrapper notes in §4.7).

Core functions confirmed to exist with consistent names across all three scripts: `get_candidate_questions`, `filter_by_token_limit`, `sample_questions`, `get_accepted_answers`, `compute_similarity`, `build_output_path`, `append_run_history`, `main`. Condition C adds: `connect_neo4j`, `load_faiss_cache`, `anchor_via_vector_search`, `traverse_graph`, `semantic_expansion`, `fuse_and_rank`, `extract_citations`.

### 1.5 Repo Audit Findings (Claude Code / VSCode audit) and Their Impact on This Plan

| # | Finding | Impact on this plan |
|---|---|---|
| 1 | `.env` with live `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` is **committed in git history**, not yet remediated (keys not rotated, history not scrubbed) | Settings storage design (§4.6, §5.9) must not repeat this pattern — config lives outside the git repo, keys are never echoed back raw. Rotating the leaked keys and cleaning git history remains your decision, outside this dashboard's scope. |
| 2 | `results_old/` folders exist alongside active `results/` per condition, plus stray `.tmp` DuckDB spill files | `history_service.py` must read `.jsonl` only from an explicitly configured active path per condition — never a recursive glob that would also pick up `results_old/` or `.tmp` files (§4.5). |
| 3 | `run.py:145` references a nonexistent file `a_baseline_replecation.py` (typo) — the real file is `a_baseline_replication.py`; the CLI launcher's "Run Condition A" menu item is currently broken | **No functional impact** on the dashboard — we call Python functions directly, not `run.py`/`subprocess`. The correct filename is used consistently throughout this plan. Once the dashboard exists, `run.py` is effectively retired for day-to-day use; the typo only matters if you still use the old CLI for quick debugging. |
| 4 | `compare_condition_c_runs.py` exists in two ambiguous locations (repo root vs `llm/c_graphrag/`) | **No impact** — the "Compare" feature in History (§5.9) is designed as native backend logic (reading multiple `run_history.jsonl` files and building a side-by-side table), not a call into this script. |
| 5 | `.gitignore` line 14 is broken (two patterns fused into one: `llm/b_rag/logs/*00_datasource/PostLinks.xml`); `.DS_Store` is tracked in 6 locations | No direct impact on the dashboard's UI/UX or API contract — pure repo-hygiene items, noted here so they aren't lost before thesis submission. |

### 1.6 Still Open (optional to answer before implementation)

1. Parquet columns outside the Neo4j schema (e.g. raw `CreationDate`, `OwnerUserId`) that you might also want surfaced on detail pages — not yet confirmed. Not blocking: `AttributePanel` renders generically, so extra fields can be added later without restructuring pages.
2. A real sample line from `results/*.jsonl` for Condition C (answer text can be redacted) — useful to confirm there are no format edge cases (e.g. `retrieved_context: []`), but the field-level schema above is already sufficient to design the layout.
3. Whether the `graphrag-dashboard-backend-fase1.zip` scaffold mentioned in earlier sessions should be reused — it was not found in the `master` branch, so this plan assumes a **clean start**.

If nothing here needs correcting, I'll proceed with the assumptions above.

---

## 2. Design Principles — Minimalist, Blue Theme, Not "AI Slop"

**"AI slop" patterns to explicitly AVOID:**
- Purple-to-pink or blue-to-purple gradients as decorative backgrounds
- Neon glow borders, heavy drop-shadows, glassmorphism (`backdrop-blur` everywhere)
- Emoji as a substitute for icons in the production UI (emoji are fine in this document, not in the app)
- Rounded-full on every element (buttons, cards, badges) — reads as a generic SaaS template
- Decorative stock illustrations or 3D blobs that carry no information
- Too many distinct accent colors on one screen ("rainbow UI")
- Cards with heavy shadows floating over a light-gray background (the over-used "Notion-clone" pattern)

**Principles USED instead:**
- **Flat, functional, data-dense.** This is a research dashboard, not a marketing landing page — prioritize table/graph readability over decoration.
- **One accent color (blue), used sparingly.** Blue is reserved for primary interactive elements (links, primary buttons, active sidebar state, node highlighting). Everything else is neutral (grayscale/slate).
- **Thin borders, not heavy shadows**, to separate sections. Shadows, when used at all, are very subtle and only on floating elements (dropdowns, modals).
- **A single typeface** (one font family, only 3 weights: regular/medium/semibold); hierarchy comes from size and color, not from mixing fonts.
- **Deliberate whitespace**, 8px grid, consistent left alignment.
- **Graph edge colors follow the real trust hierarchy** (§1.2), not arbitrary per-type colors.

### 2.1 Design Tokens

```css
/* Base / Neutral (Slate) */
--color-bg:            #F8FAFC;   /* slate-50 — app background */
--color-surface:       #FFFFFF;   /* card, table, panel */
--color-border:        #E2E8F0;   /* slate-200 — all thin borders */
--color-border-strong: #CBD5E1;   /* slate-300 — borders that need more contrast (e.g. table header) */
--color-text-primary:  #0F172A;   /* slate-900 */
--color-text-secondary:#64748B;   /* slate-500 */
--color-text-muted:    #94A3B8;   /* slate-400 — placeholders, small timestamps */

/* Accent — Blue, used sparingly */
--color-primary:       #2563EB;   /* blue-600 — primary buttons, links, active state */
--color-primary-hover: #1D4ED8;   /* blue-700 */
--color-primary-soft:  #EFF6FF;   /* blue-50 — light highlight background, badges */
--color-primary-border:#BFDBFE;   /* blue-200 — border on blue-highlighted elements */

/* Semantic (used VERY sparingly: run status & metrics only) */
--color-success:       #16A34A;   /* green-600 — "completed" status, high similarity */
--color-warning:       #D97706;   /* amber-600 — NF3 latency warning, partial hallucination */
--color-danger:        #DC2626;   /* red-600  — failed run, full hallucination */

/* Graph edges — follow the real trust-weight hierarchy (§1.2), NOT arbitrary colors */
--edge-accepted-answer:#2563EB;   /* HAS_ACCEPTED_ANSWER (w=1.0) — strongest, primary blue */
--edge-has-answer:     #93C5FD;   /* HAS_ANSWER (w variable) — light blue */
--edge-related-to:     #0EA5E9;   /* IS_RELATED_TO (w=1.0/cosine) — cyan-blue */
--edge-tag-cooccur:    #64748B;   /* TAG_COOCCUR (w=0.6) — neutral slate */
--edge-embed-sim:      #94A3B8;   /* EMBED_SIM (w=0.4) — faded slate (lowest trust) */
--edge-tagged-with:    #CBD5E1;   /* TAGGED_WITH — very faint gray, structural not trust-based */
--edge-author-trust:   #C4B5FD;   /* AUTHOR_TRUST — faded purple (the one deliberate deviation, since it points to a User node, a different context) */

/* Typography */
--font-family: 'Inter', system-ui, -apple-system, sans-serif;
--font-mono:   'JetBrains Mono', 'SF Mono', monospace;   /* for [SO-<id>] citations, question_id, code */

/* Radius & Spacing — consistently small, NOT rounded-full */
--radius-sm: 4px;   /* badge, input */
--radius-md: 6px;   /* card, button */
--radius-lg: 8px;   /* modal, large panel */
--spacing-unit: 8px; /* base grid */
```

### 2.2 Component Rules (Summary)

| Component | Rule |
|---|---|
| **Sidebar** | Background `--color-surface`, 1px right border `--color-border`. Active item: background `--color-primary-soft`, text `--color-primary`, no emoji icons — use a monochrome outline icon set (Lucide/Heroicons), color follows state. |
| **Primary button** | Background `--color-primary`, white text, `--radius-md`, NO shadow, hover → `--color-primary-hover`. Only 1 primary button per section/form. |
| **Secondary button** | Transparent background, `--color-border-strong` border, `--color-text-primary` text. |
| **Table** | Header background `--color-bg`, bottom border `--color-border-strong`, row hover `--color-bg`, no colorful zebra striping — just thin row dividers. |
| **Badge/Chip** | `--radius-sm` (not a full pill), small padding, color by semantics (blue for citations, green/amber/red for hallucination class). |
| **Card** | 1px `--color-border`, NO shadow (or a very subtle `0 1px 2px rgba(0,0,0,0.04)` at most), `--radius-lg`. |
| **Progress bar** | Track `--color-border`, fill `--color-primary`, thin height (6–8px), no gradient/glow animation. |
| **Graph node** | Flat shape per type (Question = blue circle, Answer = slate-green square, Tag = gray diamond/triangle), size proportional to `score`/`trustScore`, label appears on hover only (not always-on, to avoid clutter). |
| **Citation badge `[SO-<id>]`** | Mono font, `--color-primary-soft` background, `--color-primary` text; click → scroll/navigate to the source in the retrieval evidence panel. |

---

## 3. System Architecture

```
┌─────────────────────┐        REST + SSE/WebSocket        ┌──────────────────────┐
│   Frontend (React)   │ ◄─────────────────────────────────►│  Backend (FastAPI)    │
│   Vite + TS          │                                     │                       │
└─────────────────────┘                                     └──────┬───────┬───────┘
                                                                     │       │
                                                          ┌──────────┘       └──────────┐
                                                          ▼                             ▼
                                                 ┌─────────────────┐          ┌──────────────────┐
                                                 │ Neo4j (graphrag) │          │ DuckDB + Parquet  │
                                                 │ bolt://localhost │          │ (T7 Shield drive) │
                                                 └─────────────────┘          └──────────────────┘
                                                          ▲
                                                          │ called by
                                                 ┌─────────────────────────────┐
                                                 │ Existing engine (a_/b_/c_.py)│
                                                 │  wrapped as a service layer  │
                                                 │  invoked by FastAPI          │
                                                 └─────────────────────────────┘
```

**Required stack:**
- Frontend: **React (Vite + TypeScript)**, React Router, TanStack Query (data fetching + cache), Zustand or Context for light state
- Backend: **FastAPI**, Pydantic v2, `neo4j` Python driver, `duckdb`, Uvicorn
- Real-time progress: **Server-Sent Events (SSE)** (simpler than WebSocket for one-way progress push; use WebSocket only if two-way communication is needed)
- Graph visualization: **react-force-graph-2d** (lightweight, fits the partial graph on Home & detail pages) — avoid heavy libraries like Cytoscape unless advanced features are actually needed
- Styling: TailwindCSS + shadcn/ui (table, dialog, progress bar components are ready-made, saves time)

---

## 4. Information Architecture (Site Map)

```
/                              Home (summary dashboard)
/questions                     Question list
/questions/:id                 Question detail
/answers                       Answer list
/answers/:id                   Answer detail
/tags                          Tag list
/tags/:name                    Tag detail
/experiment/a                  Run Condition A (Pure LLM)
/experiment/b                  Run Condition B (LLM + RAG)
/experiment/c                  Run Condition C (LLM + GraphRAG)
/experiment/:condition/runs/:run_id          Run results (live or completed)
/experiment/:condition/runs/:run_id/q/:qid   Single-question result detail within a run
/history                       All run history ("transactions")
/history/compare               Compare 2–3 runs side by side
/history/:run_id               Historical run detail (reuses the run-result view)
/settings                      Provider/model/API key/path configuration
```

**Sidebar menu structure:**

```
Home
Master Data
   ├─ Questions
   ├─ Answers
   └─ Tags
Experiments
   ├─ Condition A — Pure LLM
   ├─ Condition B — LLM + RAG
   └─ Condition C — LLM + GraphRAG
History
Settings
```

(No emoji in the actual sidebar UI — use monochrome outline icons per §2.2.)

---

## 5. Backend (FastAPI) — Specification

### 5.1 Folder Structure

```
backend/
├── app/
│   ├── main.py                     # FastAPI app, CORS, router registration
│   ├── config.py                   # Load env/config, Pydantic Settings
│   ├── db/
│   │   ├── neo4j_client.py         # Singleton driver, session helper
│   │   └── duckdb_client.py        # Read parquet, deterministic reservoir sampling (seed=42)
│   ├── models/
│   │   └── schemas.py              # Pydantic response/request models
│   ├── services/
│   │   ├── master_data_service.py  # Query questions/answers/tags from DuckDB/Neo4j
│   │   ├── graph_service.py        # Fetch n-hop subgraph for visualization
│   │   ├── engine_service.py       # Wrapper calling a_/b_/c_.py (see §5.7)
│   │   ├── history_service.py      # Read/parse .jsonl files → transactions
│   │   └── settings_service.py     # CRUD config (provider, model, API key)
│   ├── routers/
│   │   ├── master_data.py
│   │   ├── graph.py
│   │   ├── runs.py                 # trigger run + SSE progress
│   │   ├── history.py
│   │   └── settings.py
│   └── core/
│       └── security.py             # Encrypted API key storage (see §5.6)
├── requirements.txt
└── .env.example
```

### 5.2 Endpoints — Master Data

```
GET  /api/questions?page=&page_size=&search=&tag=          → paginated list
GET  /api/questions/{question_id}                          → full attribute detail
GET  /api/questions/{question_id}/answers                  → answers belonging to this question
GET  /api/questions/{question_id}/accepted-answer           → accepted answer (if any)

GET  /api/answers?page=&page_size=&search=
GET  /api/answers/{answer_id}                               → full attribute detail

GET  /api/tags?page=&page_size=&search=
GET  /api/tags/{tag_name}                                    → detail + questions tagged with it

GET  /api/stats/summary                                      → total questions, answers, tags, edges (for Home)
```

Every detail endpoint **must return all node attributes** as-is from SORD/Neo4j (don't drop fields), plus lightweight graph metadata (outgoing edge count, 2-hop reachable count) so the frontend can show a "N relations found" badge.

### 5.3 Endpoints — Graph Visualization

```
GET /api/graph/partial?limit=100                                    → random/representative subgraph for Home
GET /api/graph/node/{node_type}/{node_id}?hops=2                    → subgraph centered on one node (question/answer/tag)
```

Response format (consumed directly by react-force-graph):
```json
{
  "nodes": [{ "id": "Q-123", "type": "Question", "label": "...", "properties": {...} }],
  "links": [{ "source": "Q-123", "target": "A-456", "type": "HAS_ANSWER", "weight": 0.82 }]
}
```
Include `edge type` (`HAS_ANSWER`, `TAG_COOCCUR`, `EMBED_SIM`, `IS_RELATED_TO` Linked/Duplicate) with distinct colors on the frontend — important for showing your supervisor how the trust-weighted KG actually works.

**Note:** with ~2.6M `Question` nodes, `/api/graph/partial` must **always** sample (e.g. top-N by degree/`trustScore`), never return anything close to the full graph, or the force-directed layout becomes unusable noise. Default assumption: `limit=100` representative nodes.

### 5.4 Endpoints — Run Engine (Condition A/B/C)

Two run modes, both through the same endpoint with different payloads:

```
POST /api/runs                     → start a new run, return run_id immediately (async job)

Body (batch mode):
{
  "condition": "A" | "B" | "C",
  "mode": "batch",
  "n_sample": 30,
  "seed": 42,
  "oversample_pool": 90,
  "model": "phi3:mini",
  "provider": "ollama"
}

Body (single mode):
{
  "condition": "A" | "B" | "C",
  "mode": "single",
  "question_id": "12345678",
  "model": "phi3:mini",
  "provider": "ollama"
}

GET  /api/runs/{run_id}/stream     → SSE, progress events: {step, total, current_question_id, status}
GET  /api/runs/{run_id}            → status + summary results, while running or after completion
GET  /api/runs/{run_id}/results    → list of processed questions (click → per-question detail)
GET  /api/runs/{run_id}/results/{question_id}  → full detail of one result (LLM answer, accepted answer, retrieved context/graph path, metrics, [SO-<id>] citations)
POST /api/runs/{run_id}/cancel     → stop a running job (optional but recommended)
```

**Important — run as a background task**, not a blocking request:
- Use FastAPI `BackgroundTasks` for simple jobs, or `asyncio.create_task` + an in-memory job registry (`run_id → status`) for more granular progress
- If a run can take a long time (n=384) and needs to survive a server restart, consider a lightweight queue (e.g. `arq`, or even just a thread + file lock) — for a thesis-scale project, an in-memory registry plus writing progress to a file is sufficient

**Required run summary metrics** (from the existing evaluation schema):
- Number of questions processed, average Cosine Similarity, ROUGE-1/2/L, BERTScore
- Hallucination Rate (LLM-as-Judge) + κ info if available
- For Condition C: NF2 compliance rate (% of answers with ≥1 `[SO-<id>]` citation)
- Total duration, model & provider used, parameters (seed, pool, n)

### 5.5 Endpoints — History / Transactions

```
GET /api/history?condition=&date_from=&date_to=&page=&page_size=
GET /api/history/{run_id}          → same shape as /api/runs/{run_id} but for historical data (read from .jsonl)
```

The backend reads `.jsonl` files (path configured in Settings), parses each line into one "run transaction," and **groups** result lines by `run_id`/timestamp into a single history entry with a summary (never raw JSONL dumped to the UI). Display history as a table: `Timestamp | Condition | Mode | N | Model | Avg Similarity | Hallucination Rate | NF2% | Status`.

**Note from the repo audit (§1.5):** the repo currently has a `results_old/` folder next to the active `results/` per condition, plus leftover `.tmp` DuckDB spill files. `history_service.py` **must**:
- Read `.jsonl` only from an explicit active path per condition (configured in Settings, e.g. `llm/c_graphrag/results/`) — never a recursive `glob("**/*.jsonl")` that would also pick up `results_old/`.
- Skip (not crash on) lines that fail `json.loads()`, so any stray non-JSONL file in the same folder doesn't take down the endpoint.
- Show the currently active path explicitly in the Settings UI, so during a demo it's clear which data is being displayed.

### 5.6 Endpoints — Settings

```
GET  /api/settings
PUT  /api/settings
Body: { "provider": "ollama"|"openai"|"anthropic"|"local", "model": "...", "api_key": "...", "ollama_host": "...", "num_ctx": 8192, "jsonl_path": "...", "parquet_dir": "...", "neo4j_uri": "...", "neo4j_user": "...", "neo4j_password": "..." }
```
- **Never** return `api_key`/`neo4j_password` raw in a `GET` response — mask it (`sk-...abcd`) or return an `is_set: true` flag.
- **Important, from the repo audit (§1.5):** this repo previously had a `.env` with a live API key committed to git, not yet fully remediated. To avoid the dashboard reopening the same leak path, config **must** be stored in a file **outside the Git repo folder** (e.g. `~/.graphrag-dashboard/config.json`, lightly encrypted with `cryptography.fernet`) — **never** written to an `.env` inside any repo root that could be accidentally committed.
- Provide connection validation: `POST /api/settings/test-connection` to check Neo4j and the LLM provider are reachable before saving.

### 5.7 Engine Wrapper (`engine_service.py`)

Since the `a_/b_/c_*.py` scripts are currently CLI scripts, do a **minimal refactor** (not a rewrite) so they can be called as plain Python functions:
- Extract each condition's core logic into a function `run_condition_a(question_ids: list[str], model: str, provider: str, on_progress: Callable) -> list[ResultRecord]`
- The same function is used for both batch mode (long list) and single mode (a list with one id)
- `on_progress` is called after each completed question → used by the backend to push SSE events
- Don't change the Condition A/B function signatures in `llm/prompts.py` while adding Condition C features (this is already your convention)
- Keep writing results to `.jsonl` **as usual** (don't remove this — it stays the source of truth); the backend re-reads this file for history, and also exposes freshly completed results directly from memory for fast responses

Core functions confirmed to exist with consistent names in all three scripts (see §1.4): `get_candidate_questions`, `filter_by_token_limit`, `sample_questions`, `get_accepted_answers`, `compute_similarity`, `build_output_path`, `append_run_history`, `main` — Condition C additionally has `connect_neo4j`, `load_faiss_cache`, `anchor_via_vector_search`, `traverse_graph`, `semantic_expansion`, `fuse_and_rank`, `extract_citations`.

**Note from the repo audit:** the correct filename for Condition A is **`llm/a_pure_llm/a_baseline_replication.py`** (not `a_baseline_replecation.py` — there's a typo at `run.py:145` referencing that wrong filename, which is why the old CLI launcher's "Run Condition A" menu item currently fails). Since the dashboard calls Python functions directly (not `run.py`/`subprocess`), that typo has **no effect** on `engine_service.py` as long as you import from the correct path — but it's still worth fixing separately in `run.py` if you plan to keep using the old CLI for quick debugging outside the dashboard.

---

## 6. Frontend (React) — Specification

### 6.1 Folder Structure

```
frontend/
├── src/
│   ├── main.tsx / App.tsx (router)
│   ├── layouts/
│   │   └── DashboardLayout.tsx     # Sidebar + topbar + content outlet
│   ├── components/
│   │   ├── GraphView.tsx           # react-force-graph wrapper, reusable
│   │   ├── EdgeLegend.tsx          # edge-type color legend, shared across all GraphView usages
│   │   ├── ProgressRunPanel.tsx    # progress bar + live SSE log
│   │   ├── DataTable.tsx           # generic table with pagination/search
│   │   ├── AttributePanel.tsx      # generic node-attribute renderer
│   │   ├── MetricSummaryCards.tsx  # run summary metric cards (reused: live run + history detail)
│   │   ├── AnswerComparisonPanel.tsx # LLM answer vs ground truth, side by side
│   │   ├── RetrievalEvidenceList.tsx # retrieved_context list (B/C only)
│   │   └── CitationBadge.tsx       # renders [SO-<id>] as a link to the source
│   ├── pages/
│   │   ├── HomePage.tsx
│   │   ├── QuestionListPage.tsx
│   │   ├── QuestionDetailPage.tsx
│   │   ├── AnswerListPage.tsx
│   │   ├── AnswerDetailPage.tsx
│   │   ├── TagListPage.tsx / TagDetailPage.tsx
│   │   ├── RunConditionPage.tsx    # reused 3x (A/B/C) via a route param
│   │   ├── RunResultPage.tsx
│   │   ├── HistoryPage.tsx
│   │   ├── HistoryComparePage.tsx
│   │   ├── HistoryDetailPage.tsx
│   │   └── SettingsPage.tsx
│   ├── api/                        # TanStack Query hooks per resource
│   └── types/                      # TS types mirroring the Pydantic schemas
```

### 6.2 Reusable Components — Where They're Used

| Component | Used in |
|---|---|
| `AttributePanel` | Question detail, Answer detail |
| `GraphView` | Home, Question detail, Answer detail, single-result detail (mini path graph) |
| `EdgeLegend` | Every page that renders a `GraphView` |
| `DataTable` | Question/Answer/Tag lists, per-question run results, History |
| `ProgressRunPanel` | Run Condition A/B/C pages |
| `MetricSummaryCards` | Run summary (new run & history detail — same component, reused) |
| `AnswerComparisonPanel` | Single-result detail (LLM answer vs ground truth) |
| `RetrievalEvidenceList` | Single-result detail (B/C only) |
| `CitationBadge` | Inside LLM answer text & `RetrievalEvidenceList` |

---

## 7. Page-by-Page Layout (Wireframe Level)

### 7.1 Home (`/`)

```
┌───────────┬─────────────────────────────────────────────────────────┐
│           │  Knowledge Graph Summary                                  │
│  Sidebar  │  ┌──────────┬──────────┬──────────┬──────────┐          │
│  240px    │  │ Question │  Answer  │   Tag    │  Edges   │          │
│  fixed    │  │ 2,685,814│  725,795 │  56,435  │  N total │          │
│           │  └──────────┴──────────┴──────────┴──────────┘          │
│           │                                                          │
│           │  Partial Knowledge Graph            [Edge type legend]   │
│           │  ┌────────────────────────────────────────┐  ● Accepted │
│           │  │                                          │  ● HasAnsw │
│           │  │        (force-directed graph)            │  ● Related │
│           │  │                                          │  ● TagCooc│
│           │  └────────────────────────────────────────┘  ● EmbedSim │
│           │                                                          │
│           │  Latest Runs                                             │
│           │  ┌─Condition A──┐ ┌─Condition B──┐ ┌─Condition C──┐     │
│           │  │ sim: 0.xx    │ │ sim: 0.xx    │ │ sim: 0.xx    │     │
│           │  │ n=xx  →detail│ │ n=xx  →detail│ │ NF2: xx% →det│     │
│           │  └──────────────┘ └──────────────┘ └──────────────┘     │
└───────────┴─────────────────────────────────────────────────────────┘
```
- 4 stat cards from `/api/stats/summary`.
- `GraphView` shows the partial graph (`/api/graph/partial`); clicking a node navigates to `/questions/:id`, `/answers/:id`, or `/tags/:name` per its type.
- Last-run summary: 3 small cards (latest results for Condition A/B/C, if any) linking to History.

### 7.2 Master Data Lists (`/questions`, `/answers`, `/tags`)

- Full-width search bar above the table; tag filter (Questions only) as a dropdown/chip.
- Minimal columns:
  - **Questions**: `id`, `title` (truncated), `domainTag`, `score`, `viewCount`, `trustScore` (small badge), #answers
  - **Answers**: `id`, `body` (truncated), `score`, `isAccepted` (green outline badge, not solid fill), `authorReputation`
  - **Tags**: `name`, `questionCount`
- Server-side pagination below the table (not infinite scroll — more predictable for demos).

### 7.3 Question Detail (`/questions/:id`)

```
┌──────────────────────────────┬───────────────────────────────┐
│  Question Attributes           │  2-hop Graph (centered here)    │
│  ─────────────────────────    │  ┌───────────────────────────┐ │
│  #id · domainTag · trustScore │  │                             │ │
│  Title                        │  │      (force graph)          │ │
│  ─────────────────────────    │  │                             │ │
│  Body (markdown/code render)  │  └───────────────────────────┘ │
│  ─────────────────────────    │  Edge legend (same as Home)     │
│  score · viewCount ·          │                                 │
│  creationDate                 ├───────────────────────────────┤
│                                │  Answers (N)                    │
│                                │  ┌─────────────────────────┐   │
│                                │  │ ✓ Accepted Answer         │   │
│                                │  │ score: xx  →detail        │   │
│                                │  ├─────────────────────────┤   │
│                                │  │ Answer #2  score: xx →det │   │
│                                │  └─────────────────────────┘   │
└──────────────────────────────┴───────────────────────────────┘
```
- `AttributePanel` on the left renders generically from the object, but field order is prioritized to match the schema table in §1.1 (not raw JSON key order).
- The accepted answer is always shown first in the answer list with a `✓ Accepted` badge (thin green border, not solid fill — stays minimal).
- Graph panel: `/api/graph/node/Question/{id}?hops=2`, edges colored by type.

### 7.4 Answer Detail (`/answers/:id`)

Same layout as Question detail (`AttributePanel` left + `GraphView` top right, centered on the answer), plus a breadcrumb "← Back to Question #id" at the top.

### 7.5 Run Condition (`/experiment/a|b|c`)

One component, 3 routes (`/experiment/a`, `/experiment/b`, `/experiment/c`), differentiated by a `condition` prop.

**Input form** — tab/toggle between 2 modes:
- **Batch mode**: `n_sample`, `seed` (default 42), `oversample_pool`, `provider`/`model` dropdowns (prefilled from Settings, overridable)
- **Single mode**: `question_id` input (with autocomplete/search against `/api/questions`)

```
┌────────────────────────────────────────────────────────────────┐
│  Condition C — LLM + GraphRAG                                    │
│  ┌─ Mode ─────────────────────────────────────────────┐         │
│  │  [Batch]  [Single Question]      ← tabs, not radio buttons   │
│  └────────────────────────────────────────────────────┘         │
│                                                                    │
│  Batch:                                                           │
│    n_sample [  30]  seed [ 42]  oversample_pool [ 90]            │
│    provider [ollama ▾]  model [phi3:mini ▾]                      │
│    top_k [5]  n_anchor [3]  n_semantic_expansion [3]              │
│                                                    [▶ Run]         │
│                                                                    │
│  ⚠ seed/oversample_pool differ from the last Condition A/B run    │
│     (if detected — inline warning, not a blocking modal)          │
└────────────────────────────────────────────────────────────────┘

--- after "Run" ---

┌────────────────────────────────────────────────────────────────┐
│  Progress: ████████████░░░░░░░░  18 / 30      elapsed 02:14      │
│  Live log (auto-scroll, max height, monospace):                   │
│    ✓ Id=1234 similarity=0.712 citation=✓ latency=2.1s              │
│    ✓ Id=5678 similarity=0.655 citation=✓ latency=8.7s ⚠           │
│  [■ Stop]                                                          │
└────────────────────────────────────────────────────────────────┘

--- after completion ---

┌─ Result Summary ──────────────────────────────────────────────────┐
│  Avg Cosine Sim   ROUGE-1/2/L   BERTScore   Hallucination Rate    │
│     0.7195           …             …          FACTUAL: 78%        │
│  NF2 (citation compliance): 100%     Avg retrieval latency: 2.81s │
└────────────────────────────────────────────────────────────────┘

┌─ Per-Question Results (table, click a row → detail) ──────────────┐
│  id   │ similarity │ citation │ n_anchors │ latency │ status      │
└────────────────────────────────────────────────────────────────┘
```

"Run" → `POST /api/runs` → get `run_id` → immediately open `ProgressRunPanel` subscribed to SSE `/api/runs/{run_id}/stream`.

After completion, show:
- **Result Summary** (metric cards: Avg Cosine Sim, ROUGE-1/2/L, BERTScore, Hallucination Rate, and NF2% for Condition C specifically)
- Table of processed questions → click a row → `RunResultPage` (full detail: LLM answer with highlighted/linked `[SO-<id>]` citations, ground-truth answer next to it for visual side-by-side comparison, per-question metrics, and — Condition C only — a mini `GraphView` showing the Entity Anchoring → Graph Traversal → Semantic Expansion retrieval path used)

### 7.6 Single-Result Detail (`/experiment/:condition/runs/:run_id/q/:qid`)

This is the single most important panel for demoing to your supervisor — a direct side-by-side comparison:

```
┌───────────────────────────────┬───────────────────────────────┐
│  LLM Answer (Condition C)       │  Accepted Answer (Ground Truth) │
│  ...answer text with           │  ...ground_truth_answer text    │
│  [SO-1234] highlighted blue,    │                                  │
│  click → scroll to evidence     │                                  │
│  below                          │                                  │
├───────────────────────────────┴───────────────────────────────┤
│  Metrics: cosine_similarity · has_valid_citation · retrieval_latency│
├─────────────────────────────────────────────────────────────────┤
│  Retrieval Evidence (B/C only) — retrieved_context                 │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ [SO-1234]  trust=1.00  0-hop   "...chunk_text..."  →open Q  │   │
│  │ [SO-5678]  trust=0.60  1-hop   "...chunk_text..."  →open Q  │   │
│  └───────────────────────────────────────────────────────────┘   │
│  Mini graph: Entity Anchoring → Graph Traversal → Semantic Expansion│
│  (Condition C only — the 3 stages visualized as a graph path)      │
└─────────────────────────────────────────────────────────────────┘
```

### 7.7 History (`/history`)

Transaction table (read from each condition's `run_history.jsonl`):

| Timestamp | Condition | Mode | n | Provider/Model | Avg Sim | Hallucination | NF2% | Status |
|---|---|---|---|---|---|---|---|---|

A checkbox on the left of each row selects 2–3 runs → a "Compare" button appears in the table's toolbar (not a floating action button).

### 7.8 History Compare (`/history/compare`)

Side-by-side metric table (columns = selected runs, rows = metrics). Highlight the best value per row with bold text + a subtle green tint (not a trophy icon/emoji).

### 7.9 Settings (`/settings`)

Single-column form, grouped into sections with a thin divider (not separate floating cards):
- **LLM Provider** — provider dropdown, model field, API key (masked, with a "show" toggle), Ollama host field (Ollama only)
- **Data Source** — parquet directory path, per-condition `results/`/`logs/` paths, Neo4j URI/user/password
- "Test Connection" button per section — result shown as small text under the field (`✓ Connected` green / `✗ Failed: <error message>` red), not an auto-dismissing toast, so it can be read calmly during a demo.
- **Security note (from the repo audit, §1.5):** since this repo previously had an `.env` with a live API key committed to git, the API key field **must** (a) never be sent back raw from `GET /api/settings` — only masked (`sk-...abcd`) or as `is_set: true`, (b) be stored by the backend in a file **outside** the git repo folder (e.g. an encrypted `~/.graphrag-dashboard/config.json`), never in an `.env` at any repo root, so the same leak path can't reopen. Add a small line under this section: *"Keys are never displayed again or written to logs."*

### 7.10 Methodology (`/methodology`)

Read-only reference page — documents each condition's (A/B/C/D) design and
retrieval mechanism, not something interactive. Tabbed by condition
(A/B/C/D), each tab showing overview, retrieval design, prompting,
leakage-prevention approach, and metrics measured, plus a single
always-visible comparison table summarizing all four conditions
side by side. Content is defined in-component (`MethodologyPage.tsx`)
rather than a separate data file, since it changes only when a
condition's actual design changes.

### 7.11 Metrik Evaluasi (`/docs/metrics`)

Also read-only/reference, not interactive — the counterpart to §7.10 but
for evaluation metrics rather than condition design: what each metric
measures, how it's computed, which condition(s) produce it, how to
interpret it, and its limitations (mandatory on every entry, never left
blank). Content lives in `frontend/src/content/evaluationMetrics.ts` as
typed data (not hardcoded JSX), so metric definitions can be
reviewed/edited independently of the page's rendering logic — and so the
same content can be linked to by id from other pages (see below) without
duplicating the explanation.

Layout: tabs by category (Semantic Quality / Citation & Grounding /
Hallucination & Faithfulness / Retrieval Quality / Efficiency, plus an
"All" tab), a checkbox filter by condition (A/B/C/D) alongside it, and
each metric rendered as a collapsed-by-default card — name + per-condition
badges + a one-line summary always visible, full detail (formula,
interpretation guide, limitations, source file paths, "see also"
cross-links to related metrics) behind expand. Anchored by metric id
(`#cosine-similarity`, `#hallucination-rate`, etc.) so other pages can
deep-link straight to one metric's definition.

**Cross-linking convention**: any page that displays a metric's value
(`MetricSummaryCards`, `RunResultDetailPage`, `HistoryResultDetailPage`,
the "Hasil LLM-as-Judge"/"Penilaian Judge" sections) shows a small inline
info icon (`MetricInfoLink` component) next to that metric's label,
linking to `/docs/metrics#<metric-id>` — never a second copy of the
explanation. This keeps metric definitions in exactly one place so they
can't drift out of sync between pages.

---

## 8. Things Not to Miss

1. **Sampling consistency**: batch mode in the UI must send the same `seed` & `oversample_pool` across conditions if the user wants a fair A/B/C comparison — show an inline warning if the user changes the default 42/90 without realizing it.
2. **NF2 citation** must be **visually distinct** in results — this is the core evidence for the thesis's novelty claim. Render `[SO-<id>]` as a colored badge with a link back to the source question/answer.
3. **Graph edge types** must have a consistent color legend across every page (`HAS_ANSWER`, `HAS_ACCEPTED_ANSWER`, `TAG_COOCCUR`, `EMBED_SIM`, `IS_RELATED_TO` Linked/Duplicate).
4. **Don't block the UI** during long runs (n=384 can take a while on an M2 8GB) — ensure SSE auto-reconnects if the connection drops, and run status must remain checkable even after a page refresh (`GET /api/runs/{run_id}` must be idempotent, reading from a job registry/file, not only in-memory state that's lost on refresh).
5. **Local-only deployment**: no need for complex auth (single user, running on localhost), but still never expose the API key in the Settings `GET` response.
6. **Generic attribute schema**: since SORD has many fields, `AttributePanel` should render automatically from the JSON object (a key-value list) instead of hardcoding each field — this keeps it resilient to future Neo4j/parquet schema changes.
7. **`results_old/` and stray `.tmp` files** (per the audit, §1.5): History must read only from the explicitly configured active results path, never scan directories blindly.
8. **Settings storage location** (per the audit, §1.5): config must live outside the git repo folder, encrypted, never in a repo-root `.env`.

---

## 9. Implementation Phases (work sequentially, ask for my review at the end of each phase)

1. **Phase 1 — Backend skeleton**: FastAPI app + Neo4j & DuckDB connections + master-data endpoints (questions/answers/tags list & detail) + stats summary endpoint. Test via `/docs` (Swagger) first, no frontend yet.
2. **Phase 2 — Frontend skeleton**: sidebar layout, routing, Home page (stats only, no graph yet), List & Detail pages for Questions/Answers/Tags (no graph view yet).
3. **Phase 3 — Graph visualization**: `/api/graph/*` endpoints + `GraphView` component integrated into Home and the detail pages.
4. **Phase 4 — Engine wrapper & Run endpoints**: refactor `a_/b_/c_*.py` into callable functions, `/api/runs` endpoint + SSE progress, test with Condition A first (simplest), then B, then C.
5. **Phase 5 — Frontend Run pages**: 2-mode input form, progress bar, Result Summary, per-question result detail.
6. **Phase 6 — History**: read your existing `.jsonl` files, display as a transaction table + detail + compare feature.
7. **Phase 7 — Settings**: provider/model/API key/path config, test connection.
8. **Phase 8 — Polish for the demo**: consistent graph legend colors, loading states, empty states, responsive layout for presenting on a projector during the defense.

---

## 10. Additional Notes for the Coding Agent

- Follow existing conventions in the `dartikaanie/graphrag` repo (numbered STEP logging, `_kg_workspace/`, `write_batched()` UNWIND writes to Neo4j) when integrating the engine wrapper — don't rewrite retrieval/generation logic that has already been validated.
- Don't modify the Condition A/B functions in `llm/prompts.py` while working on the wrapper — only add new functions or backward-compatible optional parameters.
- Display all numeric metric fields (cosine similarity, ROUGE, etc.) with 3–4 decimal places and include the unit/scale (0–1) in the UI so nothing is ambiguous when explaining to your supervisor.