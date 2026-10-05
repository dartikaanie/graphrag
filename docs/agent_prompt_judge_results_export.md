# Task: Judge results page + exports (summary, agreement, per-question detail, context relevance)

## Why

The pilot judge results (9 runs, n=10) show an unexpected pattern: grounding raises the hallucination rate for C, and C-trust-grounded abstains on 4/10 questions even though it always receives 5 context items. Before the n=384 run I need to check whether this is a real effect or a measurement problem. That needs three things exported as files I can share for review:
1. both judges' results and their agreement;
2. per-question detail (question, reference, retrieved context, answer, both judges' reasoning);
3. a direct measure of how relevant the retrieved context is to each question.

## Step 0: Plan first. Do NOT write code yet.

Report your file-by-file plan, and:
- whether a context-relevance judge already exists in the repo (an earlier plan mentioned `llm_judge_context_relevance.py`). If it exists, describe it and say whether it can be reused as-is.
- whether the retrieved context for a question is **byte-identical** between the plain and grounded run of the same retrieval method (e.g. C-trust-plain vs C-trust-grounded), checked on the 9 pilot runs. Retrieval doesn't depend on grounding, so it should be. Report any differences.

Wait for my approval.

## Step 1: Judge results page

1. A **"Judge results"** section on the Hallucination Judge (judge-v1) page, listing completed judge jobs: date/time (Asia/Jakarta), batch, judge models, run labels covered, items judged / failed, warnings (missing file, served-model mismatch). Newest first, grouped by batch.
2. Clicking a batch opens a **results view** with its own URL (e.g. `/judge-v1/results/:batchId`), containing:
   - **Summary table per run_label**, rows in factorial order (A, B-plain, B-grounded, C-uniform-plain, C-uniform-grounded, C-trust-plain, C-trust-grounded, D-plain, D-grounded), with **primary and secondary side by side**: FAKTUAL / SEBAGIAN / PENUH / ABSTAIN counts and %, abstention rate, hallucination rate both ways, parse-error and consistency rates.
   - **Agreement panel** (existing component, with the judge pair picker): overall and per run_label.
   - **Disagreement list** (existing component, filterable), each row linking to the per-question detail.
   - **Context relevance** summary (Step 3), once available.
3. Reuse the existing summary / agreement / disagreement endpoints and components. Don't duplicate logic.

## Step 2: Exports

All exports are markdown (`.md`) with a header block: generated time, batch, run IDs + run_labels + config_hash (short), generation prompt version, judge models with role, served model, precision, judge prompt version, blinding version.

1. **Run Comparison export** (History → Compare): add the judge-v1 section when results exist for the compared runs:
   - per-run summary table (primary and secondary side by side);
   - agreement block: judge pair, weighted and unweighted κ, n pairs, 4×4 confusion matrix (overall, plus per run_label);
   - disagreement list: question id, run_label, both labels, one-line reason from each judge;
   - context relevance summary (Step 3), if available.
   Keep the legacy faithfulness section separate and labeled, if present.
2. **Judge results export** ("Download report (.md)" on the results view): same content as above for the whole batch.
3. **Per-question detail export:** from the results view, export selected questions, or all questions of a selected run_label, or all questions where the judges disagree. For each question:
   - question id, title, body (truncate to ~1,500 chars with a note), tags;
   - reference (accepted) answer (truncate to ~1,500 chars);
   - retrieved context items: rank, SO question id, answer id, is_accepted, trust weight, hop, source stage, combined/relevance score, the context text (truncate each to ~600 chars), and the context-relevance label + reason (Step 3);
   - the candidate answer: raw text, and the blinded version the judge saw;
   - citation outcome (valid / no citation / invalid-only);
   - for each judge: label, derived label, consistency flag, claims table (claim, verdict, severity, evidence), reasoning, served model.
   Also offer a quick action: "Export C-trust-grounded (all 10)".
4. Exports must never include API keys or request headers.

## Step 3: Context relevance metric

1. If a context-relevance judge exists and fits, reuse it. Otherwise add one (`llm/evaluation/llm_judge_context_relevance.py`, prompt version `ctxrel-v1`):
   - Input: the question (title, body, tags) and **one** retrieved context item. Never the candidate answer, the reference answer, the run label, or the trust score.
   - Output (JSON): `label` = RELEVANT (directly addresses this problem) / PARTIAL (related topic, but a different problem or only part of it) / IRRELEVANT, plus a one-sentence `reason`.
   - Uses the judge registry: primary-role model by default, temperature 0, JSON mode, same raw-response capture, manifests, failures log, resume, and served-model check as judge-v1. Output filename includes the prompt version.
2. **Deduplicate:** judge each unique (question_id, context answer_id, context text hash) once and reuse the result across runs. Plain and grounded runs of the same retrieval method should share contexts, so this roughly halves the calls.
3. **Per-run metrics:** % RELEVANT / PARTIAL / IRRELEVANT over all context items; per question, whether at least one item is RELEVANT; mean relevance score (RELEVANT=1, PARTIAL=0.5, IRRELEVANT=0); and relevance by rank position (rank 1 vs ranks 2–5).
4. **Link to outcomes** (descriptive only, no statistical tests at n=10): per run_label, a table of hallucination label × "has at least one RELEVANT context item", and for grounded runs, ABSTAIN × "has at least one RELEVANT context item".
5. Add it to the judge-v1 page as a separate launch option ("Context relevance"), with the same plan/confirm/progress flow and cost estimate.
6. Condition A has no context, so it is excluded from this metric.

## Step 4: Tests

- Export contents: header block, judge-v1 sections, per-question detail fields, truncation, no secrets.
- Context relevance: dedup across runs, metrics math, exclusion of A, resume.
- Results view endpoints (if new).
- Full suite passes; frontend `tsc --noEmit`, `npm run build`, `oxlint` clean.

## Step 5: Smoke test (then stop)

- Context relevance on **2 questions** from C-trust-grounded with the primary judge (≈10 context items).
- Produce one per-question detail export for those 2 questions, and the Run Comparison export for the 9 pilot runs (with the judge-v1 section). Report where the files are.
- Do NOT run context relevance on all 9 runs. I'll launch that from the UI.

## Constraints

- Don't modify the legacy judge or existing result/log/judge files.
- `backend/` may depend on `llm/`, never the reverse.
- UI style: minimalist, white background, existing plain-Tailwind conventions.

## When done, report

- Files created or changed (one line each)
- The answers to the two Step 0 checks
- Test results, smoke-test output, and paths of the exported files
- Steps for me to run context relevance on the 9 pilot runs and export everything
