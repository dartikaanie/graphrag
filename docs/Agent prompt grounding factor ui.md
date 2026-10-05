# Task: Grounding as an experimental factor + dashboard support for the v3 pilot

## Context (decisions already made)

- Generation prompt is now **v3**. Grounded and non-grounded prompts are byte-identical across B/C/D (parity tests in `llm/tests/test_prompt_parity_v3.py`). Retrieval metadata (`trust=`, hop, source stage) never appears in prompt text. Condition A is untouched.
- The study is now a **factorial design**: retrieval method × grounding (on/off), plus A.

  | | Grounding OFF | Grounding ON |
  |---|---|---|
  | B (FAISS) | `B-plain` | `B-grounded` |
  | C uniform | `C-uniform-plain` | `C-uniform-grounded` |
  | C trust_weighted | `C-trust-plain` | `C-trust-grounded` |
  | D (LightRAG) | `D-plain` | `D-grounded` |
  | A | `A` | — |

- Judge-v1 (`llm_judge_hallucination_v1.py`) is the **reference-based hallucination** metric (primary: Llama-3.3-70B via DeepInfra; secondary: gpt-4o-mini). The legacy `llm_judge_hallucination.py` is the **context-grounded faithfulness** metric and must keep working unchanged.
- `DEEPINFRA_API_KEY` has been added to `.env` (verified untracked and ignored). Judge-v1 is fully implemented (Steps 0–5 of the previous prompt), and the sanity check passed 12/12 for both judges.
- I will run the pilot (n=10, 9 runs) **through the dashboard UI**, so the UI must support everything below.

## Step 0: Plan first. Do NOT write code yet.

Inspect the backend (FastAPI) and frontend (React + Vite + TS) and report:
1. How runs are currently launched from the UI (router, service, form fields), and how `fusion_mode` and other condition-specific params reach the generation scripts.
2. Which UI pages and components show runs (history list, history detail, run comparison) and judge results (the existing "Run Judge" flow via `routers/judge.py` → `engine_service.run_judge_batch_dashboard()` → legacy script; `judge_lookup_service.py`).
3. Your proposed file-by-file plan for Steps 1–6 below.
4. Also show me the verbatim **grounded prompt with empty retrieval** for B/C/D, and confirm it is byte-identical across them.

Wait for my approval before implementing.

## Step 1: Grounding as a factor in the pipeline (backend + llm)

1. Make `require_grounding` a run parameter for **B, C (both fusion modes), and D**, settable from the CLI and from the dashboard.
2. Record in `run_history.jsonl` for every new run: `grounding` (`on`/`off`), `prompt_version`, `fusion_mode` (C), retrieval versions, seed, n_sample, oversample pool, top_k, and the other existing params.
3. Add a single source of truth for **`run_label`**, derived from metadata (never from filenames). Format: `A`, `B-plain`, `B-grounded`, `C-uniform-plain`, `C-uniform-grounded`, `C-trust-plain`, `C-trust-grounded`, `D-plain`, `D-grounded`. Use it in:
   - `history_service.py` (dashboard run IDs and labels)
   - `llm/evaluation/_run_metadata.py` (judge records). Keep the existing cross-check test that both produce identical results, and extend it to `run_label`.
   - `compare_condition_c_runs.py` and the Run Comparison report (add a **Grounding** row/column and the new citation metrics: NoCit%, InvOnly%, Fabric%, Precis).
4. Older runs without a `grounding` field: label them by their historical behavior (B → `plain`, C/D → `grounded`) and mark them `(inferred)`, the same way prompt version is inferred today. Never rewrite old records.
5. Judge-v1 summary, agreement script, and stratified human-annotation export must all group by the full `run_label`.

## Step 2: Dashboard: launching runs

1. In the run launch form, add a **Grounding** toggle (on/off) for B, C, and D. Hide it for A. Keep the existing C fusion-mode selector.
2. Show the **prompt version** (read-only, `v3`) and the full parameter set before launch.
3. Add a **"Pilot batch"** action that queues all 9 runs (A + 4 retrieval methods × 2 grounding) with identical shared parameters (seed=42, n=10, oversample pool 1536, top_k=5, ...). It must:
   - show a **confirmation screen** listing each run's full config (dry-run), including the C and D retrieval versions, before anything executes;
   - run sequentially (8GB RAM machine), with per-run status (queued / running / done / failed);
   - skip a run whose identical config already completed (reuse existing idempotency via `run_history.jsonl`).
4. Respect the existing idempotency and resumability. Never overwrite existing result files.

## Step 3: Dashboard: judge-v1

1. Next to the existing **"Run Judge"** (legacy faithfulness), add **"Run Hallucination Judge (judge-v1)"**. The legacy flow must keep working unchanged.
   - Options: judge = primary / secondary / both; scope = selected runs.
   - Before launching, show a **cost/size confirmation**: number of API calls per judge and an estimated token count. Default scope is the selected runs only.
   - Run via `llm_judge_hallucination_v1.py` (import it, don't duplicate logic), with judge outputs written to an **explicitly configured** directory (no recursive glob).
   - Show progress, failures (from `<out>_failures.jsonl`), and allow resume.
2. **Label naming in the UI:** the legacy metric is "Faithfulness (context-grounded)"; judge-v1 is "Hallucination (reference-based)". Never show both under the same unlabeled column.
3. **Results views:**
   - **Run Comparison:** per `run_label`: label distribution (FAKTUAL / SEBAGIAN / PENUH / ABSTAIN), abstention rate, hallucination rate computed two ways (excluding ABSTAIN; counting ABSTAIN as non-factual), for the primary judge, with secondary shown alongside.
   - **Agreement panel:** 4×4 confusion matrix (primary vs secondary), linear-weighted and unweighted Cohen's κ, label-vs-derived_label consistency rate, parse-error rate per judge.
   - **History Detail (per question):** both judges' labels side by side, claims table (claim, verdict, severity, evidence), reasoning, the **blinded** candidate answer and the reference answer. Highlight rows where the two judges disagree.
   - **Disagreement list:** a filterable list of all items where primary and secondary labels differ, so I can review them quickly. (First known example: QID 38118194, C-uniform: secondary FAKTUAL, primary HALUSINASI_PENUH over whether Open XML SDK supports UWP.)
4. **Human annotation:** a button to export the blinded stratified CSV (existing `export-human-csv`), and an import that computes human vs. judge κ once `human_label` is filled. Optional, only if simple: an in-app labeling view that **hides run_label/condition** while labeling.

## Step 4: Judge throughput (primary latency is ~30 s per item)

The judge sanity check is already done for both judges (12/12 each). But a primary call takes ~30 s (≈1.2k prompt + ≈0.6k completion tokens). The full study is 9 runs × 384 items ≈ 3,456 items per judge, which is ~29 hours if sequential. So:
1. Make judge concurrency configurable (`--workers`, and a field in the UI judge dialog). Default 8 for primary, 4 for secondary. Check DeepInfra's documented rate/concurrency limits and cap the default below them; report what you found.
2. Concurrency must be safe with the append-only output: one writer thread/queue, no interleaved partial lines, and resumability still works after an interrupted run.
3. Show live throughput (items/min) and an ETA in the UI progress view, and an estimated wall-clock time in the pre-launch confirmation.
4. Note: generation runs stay sequential (8GB RAM). Only the judge (pure API calls) runs concurrently.

## Step 5: Data retention and audit trail (everything must be kept)

All logs, raw responses, and results are thesis evidence. Nothing may be lost, overwritten, or summarized away.

1. **Judge records (both judges):** for every call, store the **full raw response** exactly as returned by the API, not only parsed fields: the complete message content, response `id`, `model`, `created`, `system_fingerprint` (if present), `finish_reason`, and `usage`. Also store the request parameters (model, temperature, response_format, max_tokens), the exact `messages` sent (or a SHA-256 of them plus enough fields to rebuild them deterministically), `prompt_version`, `blinding_version`, attempt count, and per-attempt error messages for retried calls. If a field is currently truncated or dropped, fix that.
2. **Failures:** keep `<out>_failures.jsonl` with the full error text, HTTP status, attempt count, and timestamps. Never delete failure records, even after a successful retry. Mark them `resolved: true` in a separate resolution log instead.
3. **Generation records (all conditions):** verify the same standard holds for gpt-4o-mini generation outputs: full raw answer text, response metadata, usage, the exact prompt (or its hash + rebuild fields), retrieved context with IDs and retrieval metadata (trust, hop, stage), and all run parameters. Report any gaps you find.
4. **Run manifest:** for every generation run and every judge run, write a manifest JSON next to the output with: full config, `run_label`, prompt/judge/blinding versions, git commit SHA (and whether the working tree was dirty), start/end timestamps (UTC), Python and key package versions (`openai`, `sentence-transformers`, `faiss`, `neo4j`, `duckdb`), item counts (attempted/succeeded/failed), total tokens, and a SHA-256 checksum of each output file.
5. **Append-only, never overwrite:** outputs, manifests, and logs are append-only or written to new files. A re-run creates a new file or appends; it never truncates existing data. Add a test for this.
6. **Archive:** add a script (and a `run.py` menu item) that copies all results, logs, manifests, and judge outputs to a configurable archive directory (default from an env var, e.g. `GRAPHRAG_ARCHIVE_DIR`, which I'll point at my external T7 drive). It should preserve the folder structure, write a `SHA256SUMS` file, verify the checksums after copying, and never delete anything at the source.
7. **Secrets:** never write API keys, auth headers, or full request headers into any log or record. Add a test that scans new log/record output for key-like strings.
8. **Git:** large results and logs stay out of git (check `.gitignore`). Small manifests and summary tables may be committed if I choose. Tell me which files fall into which group.

## Step 6: Pre-pilot checks (run these, report results)

1. Full test suite.
2. A dry-run listing of the 9-run pilot config as the UI would launch it.
3. A judge-v1 concurrency smoke test: `--limit 5 --workers 4` on one existing run file, for the primary judge only. Report wall-clock time and confirm the output has 5 well-formed records, no duplicates, the full raw response in each record, and a manifest with checksums.
4. Run the archive script once against a temporary directory and show that checksum verification passes.
5. Existing judge-v1 smoke-test outputs were created before `run_label` included grounding (e.g. `C-uniform`). Don't migrate them. Treat them as obsolete test data and make sure the UI doesn't mix them into results for new v3 runs.
6. Do NOT execute the pilot generation or a full judge run. I will launch them from the UI.

## Constraints

- `backend/` may depend on `llm/`, never the reverse.
- Don't modify or delete existing result/log files. Don't break the legacy judge flow, existing pages, or the CLI menu in `run.py` (add new menu items if useful).
- UI style: minimalist, white background, professional typography. No gradients, glow, emoji, or decorative elements. Reuse existing shadcn/ui components and the existing layout.
- Hardware: MacBook M2, 8GB RAM. Run things sequentially and stream JSONL.
- Add tests for: `run_label` derivation (all 9 labels plus inferred legacy labels), pilot-batch config generation (shared params identical across runs), and the judge-v1 API endpoints.

## When done, report

- Files created or changed (one line each)
- Screenshots or a short description of each new or changed UI view
- Sanity-check table (both judges), test results, pilot dry-run listing
- Exact steps for me to launch the pilot and the judge from the UI
- Anything ambiguous you decided, and what you chose