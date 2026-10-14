# Agent Prompt — Dashboard: "Process & Parameters" page (end-to-end transparency)

> Suggested repo path: `docs/agent_prompt_ui_process_page.md`
> Repo: `dartikaanie/graphrag` (branch `master`) · Local: `/Users/dartikaanie/SMART SYSTEM/THESIS_2/project/graphrag/`
> Before you start, read `GraphRAG_Dashboard_Plan.md`, `docs/DECISION_C_SCORING.md`, and `docs/agent_prompt_c_amendment6_logscore.md` to match existing conventions.

---

## 0. HARD RULES

1. **Read-only feature.** The new page and its API must never write to Neo4j, never run `--apply` / `--restore`, never call a judge/LLM API, and never start a run. No buttons that trigger pipeline actions.
2. **Do NOT `git commit` / `git push`** without my explicit confirmation in chat.
3. **Never invent a rationale.** Every "why this value" text must come from an existing source: `docs/DECISION_C_SCORING.md`, other files in `docs/`, or code docstrings/comments. Store the source reference (file + section/line) next to it. If no documented rationale exists, show the value with the label **"Rationale not documented"** and list it in your report for me to fill in. Do not paraphrase in a way that changes the meaning.
4. **No hand-copied values.** Every parameter value shown must be read from the code/config/artifact that actually uses it (import the constant, read the config, parse the log/report). A test must fail if a displayed value drifts from the runtime value.
5. **Do not leak blinded or held-out material.** Judge calibration (Amendment 4) is **in progress** and I am annotator 1. The page must NOT display:
   - the annotation key, condition/run labels of calibration items, or any judge label on calibration items;
   - any item content (questions, answers, judge outputs) from the official test sample (positions 1–384);
   - stage-2 outputs, once they exist, before the confirmation set has been annotated.
   Aggregate status only for these (e.g., "calibration: in progress", "official run: not started"). Make these restrictions explicit in code (a single guard module) and cover them with tests.
6. **Secrets.** Never read, return, or render `.env` or any API key. Any log/config text sent to the frontend passes through a redaction function (keys, tokens, passwords, `Authorization` headers, connection strings with credentials). Test it.
7. **Logs from an allowlist only.** Read log files only from explicitly configured directories (no recursive glob over the repo; there are stray `results_old/` folders). Validate paths to block traversal (`..`, symlinks out of the allowed roots). Cap file size / tail length.
8. Environment: MacBook M2, 8 GB RAM, pyenv Python 3.10.9. Neo4j `bolt://localhost:7687`, database `graphrag`. The page must still load when Neo4j is off (show "Neo4j unavailable" for live fields). Heavy Neo4j queries (e.g., counts over 3.4M nodes) must not run on every page load: use the latest verify log/report as the default source, with an explicit read-only "Refresh live values" action that is cached.
9. Terminal commands for me: zsh-compatible, one per line, no `#` comments on the same line.
10. Talk to me in **simple Bahasa Indonesia, step by step**. Code, comments, and docs in English. UI text: follow the language already used in the dashboard; if it is mixed, ask me at the plan stop.
11. UI style: minimalist, white background, professional typography. No gradients, glow effects, decorative blobs, or emoji. Reuse existing shadcn/ui components and Tailwind tokens; do not add new dependencies (e.g., a math renderer) without asking.

---

## 1. Goal

One dedicated page (e.g. route `/process`, nav label "Process & Parameters") that explains the **whole experiment end to end**: what each step does, which script/module implements it, which parameters it uses, **why each value was chosen**, which decision-record amendment set it, its current status, and the evidence (logs, reports, commit hashes). A reader (me, my advisor, an examiner during the demo) should be able to follow the pipeline from raw data to final metrics without opening the repo.

---

## 2. Content to cover

Verify every item against the repo; if something here is wrong or missing in the code, report it instead of following this list blindly.

**A. Pipeline steps (in order).** For each step: purpose (1–2 sentences), script/module path, inputs → outputs, parameters (with values + rationale), status (done / in progress / not started / blocked by), date, evidence links.
1. Data: SORD source, dedup, question/answer counts.
2. Cleaning & sampling: seed, oversample pool, token filter, test sample (positions 1–384) vs dev set (positions 385–434), leakage exclusion of evaluation answers from the retrieval corpus.
3. Knowledge graph build: node/edge types and edge weights (`HAS_ACCEPTED_ANSWER`, `HAS_ANSWER`, `TAGGED_WITH`, `IS_RELATED_TO`, `AUTHOR_TRUST`, `TAG_COOCCUR`, `EMBED_SIM`) with their thresholds; densification results.
4. Trust scoring: current formulas for `trust_A` and `trust_Q` from `trust_formulas.py`, P99 values and quantile method, `trustScoreVersion` history (v1 → v2-reputation → v3-logscore) with apply timestamps, backups, and verify results.
5. Retrieval conditions A, B, C, D (and grounding on/off): what differs between them.
6. Condition C ranking: `combined_trust`, `final` score, α, top_k, candidate generation stages (entity anchoring → graph traversal → semantic expansion).
7. α selection: stage 1 (ctxrel judge, criteria, tie margins, ranking rule, result table v2 → v3, selected α) and stage 2 (rule, required checks in `stage2_report.py`, status "waiting for judge freeze (Amendment 4)").
8. Hallucination judge: models, judge version, binary primary metric + severity secondary, calibration protocol and κ target, status (aggregate only, per Rule 5).
9. Official run: n=384, generation model, freeze list, status "not started".
10. Metrics & statistics: cosine similarity, hallucination rate, NF2, latency (NF3), Wilcoxon + Cliff's delta, κ.

**B. Parameter registry (table view).** One row per parameter: name, value, unit/range, step, defined in (file:line), used by, rationale (+ source reference), set by (amendment / decision), status (frozen / pending / candidate), last changed (commit hash + date). Searchable and filterable by step and status.

**C. Decision timeline.** Amendments in `DECISION_C_SCORING.md` in date order: what changed, why, when applied, commit hashes. Parse from the document; do not retype.

**D. Run & config provenance.** For runs in the configured results directories: run id, condition, split (dev/test), α, grounding, `trustScoreVersion`, `config_hash`, start time, commit. Dev/test visually distinct. Respect Rule 5 (metadata only for held-out material, no content).

**E. Logs viewer.** List allowlisted log files (apply logs, recheck/top-5 reports, KG build/densification logs), newest first; view a redacted tail; link each log to the step it belongs to.

---

## 3. Architecture (proposal; confirm or adjust at the plan stop)

- **Backend (FastAPI):** a read-only router, e.g. `/api/process/...`, with endpoints for steps, parameters, decisions, runs, and logs. Values come from importing the real constants/config and parsing existing artifacts (verify logs, recheck `.json` reports, manifests, the decision record).
- **Rationale registry:** a single structured file (e.g. `docs/process_registry.yaml` or a Python module) that maps each parameter/step to its **source pointer** for the value and its **rationale source** (document + section). It must not contain the values themselves where a runtime source exists.
- **Frontend (React + Vite + TS, TanStack Query):** one page with a step-by-step overview (vertical flow or numbered sections), a parameter table, a decision timeline, a run list, and a log viewer. Each step links to its parameters, decisions, and logs. Show an "as of" timestamp for every data source.

---

## 4. Tasks (in order)

**Task 0 — Inventory and plan, then STOP.**
- Inventory the existing dashboard: routes, nav, pages, API services, shared components, existing endpoints that already expose config/runs/logs. Reuse where possible.
- Inventory all parameters in Section 2 with their real locations (file:line) and the rationale source you found (or "not documented").
- Report in Indonesian: the inventory, the proposed page layout (sections and what each shows), endpoint list, registry format, the list of parameters without documented rationale, any conflict between this prompt and the repo, and questions for me. **Wait for my approval.**

**Task 1 — Backend** (after approval): guard module (Rule 5), redaction (Rule 6), log allowlist (Rule 7), registry loader, endpoints, Neo4j live values with cache and graceful fallback (Rule 8).

**Task 2 — Frontend:** the page and nav entry, following existing patterns and Rule 11. Readable at laptop width; tables scroll horizontally only inside their container.

**Task 3 — Tests.**
- Backend (pytest, fakes only, no Neo4j/network): every displayed value equals the runtime value (drift test); redaction; path traversal blocked; only allowlisted logs listed; calibration key, calibration judge labels, test-sample content, and stage-2 outputs never returned; page works with Neo4j unavailable; registry entries without rationale are flagged, not filled.
- Frontend: type check, lint, build, and existing frontend tests clean.
- All existing test suites (`llm/tests`, `llm/c_graphrag/tests`, `llm/evaluation/tests`, `backend/tests`) still pass. If an old test must change, explain and ask first.

**Task 4 — Final report (Bahasa Indonesia, simple):**
1. Files changed/created, one line each.
2. Test results with the commands used.
3. Commands for me to start backend and frontend and open the page (zsh).
4. The list of parameters marked "Rationale not documented", so I can write them.
5. Anything shown on the page that differs from what I described in this prompt.
6. Proposed commit file list (explicit paths) and commit message. Do not commit until I reply "commit".
