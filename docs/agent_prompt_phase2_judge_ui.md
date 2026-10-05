# Task: Phase 2 — Judge-v1 in the dashboard (UI + throughput)

## Current state (context)

- 9 pilot runs (prompt v3, n=10, config-hashed filenames, manifests) exist and are valid: A, B-plain, B-grounded, C-uniform-plain, C-uniform-grounded, C-trust-plain, C-trust-grounded, D-plain, D-grounded.
- Judge-v1 backend is done: `llm/evaluation/llm_judge_hallucination_v1.py` (primary: Llama-3.3-70B via DeepInfra, BF16; secondary: gpt-4o-mini), full raw-response capture, failures + resolution logs, manifests, blinding `blind-v2`, `judge_agreement.py` (κ, confusion matrix, stratified human export).
- Run identity: `run_label` (9 factorial labels) + `config_hash`. Invalid runs (`logs/invalid_runs.jsonl`) and superseded runs (`logs/superseded_runs.jsonl`) are hidden by default. Mixed-config files raise `MixedConfigHashError`.
- Primary judge latency is ~30 s per item. Full study ≈ 9 runs × 384 items ≈ 3,456 items per judge.
- The legacy judge (`llm_judge_hallucination.py`, "Faithfulness (context-grounded)") and its UI flow must keep working unchanged.

## Step 0: Plan first. Do NOT write code yet.

Report:
1. Your file-by-file plan for Steps 1–5 below.
2. DeepInfra's documented rate/concurrency limits for `meta-llama/Llama-3.3-70B-Instruct` (cite where you found them), and the default worker counts you propose for primary and secondary.
3. How judge-v1 outputs will be linked to generation runs (by `run_id` + `config_hash`), and where judge outputs are stored (explicit configured directory, no recursive glob).
4. How existing judge-v1 smoke-test entries (pre-grounding labels like `C-uniform`, and any `/tmp` outputs) will be excluded from the UI.

Wait for my approval before implementing.

## Step 1: Backend

1. New router (e.g. `routers/judge_v1.py`) and service (`engine_service.start_judge_v1_run()`, `judge_v1_lookup_service.py`), separate from the legacy judge files.
2. Endpoints:
   - **Plan / dry-run:** given selected run IDs and judge choice (primary / secondary / both), return: items per run, items already judged (resume), API calls to make, estimated tokens (from average tokens of existing judge records), estimated cost (use per-token prices from config, with the date they were set), and estimated wall-clock time at the chosen worker count.
   - **Launch:** start a judge-v1 job for the selected runs. Refuse invalid or superseded runs, and runs whose output file fails `assert_single_config_hash`.
   - **Status:** per run and per judge: queued / running / done / failed, items done / total, failures, throughput (items/min), ETA.
   - **Cancel:** stop gracefully. Completed records stay; a re-launch resumes.
   - **Results:** per-run summary, agreement, per-question detail, disagreement list (see Step 3).
   - **Human annotation:** export the blinded stratified CSV (existing `export-human-csv`), import a filled CSV, and return human-vs-judge κ.
3. Add an `on_progress` callback to `llm_judge_hallucination_v1.run_batch()` (keep CLI behavior unchanged) for live progress, throughput, and ETA.
4. Concurrency: configurable workers per judge (defaults from Step 0), single-writer output, resumable after cancel or crash. Judge jobs may run while a generation run is running (they're API-only), but only one judge-v1 job at a time.
5. **Judge model registry (selectable, but with fixed roles):** replace the hard-coded primary/secondary pair with a registry in config. Each entry has: `judge_id`, display name, provider, `base_url`, `model`, API key env var name, precision note, per-token prices with the date set, and a **role**:
   - `primary` (official): `meta-llama/Llama-3.3-70B-Instruct` via DeepInfra (BF16)
   - `secondary` (official): `gpt-4o-mini` via OpenAI
   - `fallback` (pre-registered, used only if primary fails human-κ validation): `meta-llama/Llama-3.1-70B-Instruct` (BF16 provider; propose one in Step 0)
   - `exploratory`: any model I add later
   Exactly one `primary` and one `secondary` at any time. Changing which model holds an official role must be an explicit config change that is recorded in a `logs/judge_role_changes.jsonl` (timestamp, old, new, reason). The UI must not let me change roles casually.
   All judges use the identical `judge-v1` prompt and settings (temperature 0, JSON mode). Every judge record stores `judge_id`, `judge_model`, `judge_role` (at the time of judging), and `judge_base_url`.
6. Every judge record must carry the generation run's `run_id`, `run_label`, and `config_hash`. Each judge job writes a manifest (status completed / interrupted / failed, counts, tokens, checksums, log path) and a per-job log file.

## Step 2: Launch UI

1. Add a **"Hallucination Judge (judge-v1)"** page or tab next to the legacy "Run Judge". Rename the legacy one's heading to "Faithfulness Judge (context-grounded, legacy)".
2. Run selection: a list of valid runs grouped by batch/date with their `run_label`, plus a quick "select all 9 runs of this factorial batch" action. Hide invalid and superseded runs.
3. Options: **judge models** as a multi-select from the registry, showing each model's role badge (Primary / Secondary / Fallback / Exploratory). Default selection = primary + secondary. Workers per judge model.
4. **Pre-launch confirmation** (from the plan endpoint): table per run × judge with items to judge, already done, est. tokens, est. cost, est. time; totals at the bottom. Nothing runs until I confirm.
5. Progress view: **one card per row** (same style as the factorial results page): header (run_label, judge, status), progress bar with current/total, throughput + ETA, failures count with a link to details, Cancel button.

## Step 3: Results UI

1. **Run Comparison** (`HistoryComparePage`): add a clearly labeled section **"Hallucination (reference-based, judge-v1)"**, per run, for primary and secondary side by side:
   - label distribution: FAKTUAL / HALUSINASI_SEBAGIAN / HALUSINASI_PENUH / ABSTAIN (counts and %)
   - abstention rate
   - hallucination rate (SEBAGIAN + PENUH), computed two ways: excluding ABSTAIN, and counting ABSTAIN as non-factual
   - parse-error rate and label-vs-derived_label consistency rate
   Keep the legacy faithfulness metrics in their own separately labeled section.
   The **official** numbers come only from the model holding the `primary` role. Results from fallback or exploratory judges appear in a separate, clearly labeled "Other judges" block, never mixed into the official columns.
2. **Agreement panel** (overall and per run_label): pick any two judge models from those that have judged the same items (default: primary vs secondary). Show the 4×4 confusion matrix, linear-weighted and unweighted Cohen's κ, number of pairs, consistency and parse-error rates. Show "insufficient data" when there are too few pairs, rather than a misleading κ.
3. **Per-question detail** (HistoryResultDetailPage or a new view): both judges' labels side by side, claims table (claim, verdict, severity, evidence), reasoning, the **blinded** candidate answer, the reference answer, and links to the raw judge responses. Highlight disagreements.
4. **Disagreement list:** filterable by run_label and by label pair (e.g. primary PENUH vs secondary FAKTUAL), linking to the per-question detail.
5. **Human annotation:** export button (with `--min-per-label` and n options), import button, and a results card with human-vs-primary and human-vs-secondary weighted κ. Optional, only if simple: an in-app labeling view that **hides run_label, condition, and both judges' labels** while I label.

## Step 4: Tests

- Plan endpoint: counts, resume-awareness, refusal of invalid / superseded / mixed-hash runs.
- Launch → status → cancel → resume, with a mocked judge client.
- Summary math: label distribution, both hallucination-rate definitions, abstention rate.
- Agreement: κ on a small known fixture, and the "insufficient data" path.
- Judge records carry `run_id`, `run_label`, `config_hash`, `judge_id`, `judge_model`, `judge_role`.
- Registry validation: exactly one primary and one secondary; a role change is logged; official summaries use only the primary-role model.
- Full suite passes; frontend `tsc --noEmit`, `npm run build`, `oxlint` clean.

## Step 5: Smoke test (report results, then stop)

- Judge **2 items** from one pilot run with **both** judges through the UI's backend path (not the full 9 runs).
- Show the resulting summary, agreement output, and one per-question detail payload.
- Do NOT run the full judge on the 9 pilot runs. I'll launch that from the UI.

## Constraints

- Don't modify the legacy judge script, its manifest, or its lookup service.
- `backend/` may depend on `llm/`, never the reverse.
- Don't modify or delete existing result, log, or judge files.
- UI style: minimalist, white background, no gradients, glow, or emoji. Reuse existing shadcn/ui components.
- Never log or display API keys.

## When done, report

- Files created or changed (one line each)
- A short description of each new UI view
- Test results and smoke-test output
- The exact steps for me to judge the 9 pilot runs from the UI
- Anything ambiguous you decided, and what you chose
