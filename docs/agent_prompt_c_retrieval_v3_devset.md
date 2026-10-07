# Task: Decision record + development set + C retrieval v3 (α sweep tooling)

Start this only after judge-v2 is complete.

## Why

Pilot evidence (prompt v3, n=10): context relevance (ctxrel-v1) was 0% RELEVANT for every retrieval method, and C's ranking score (`0.7 × path trust + 0.3 × answer trust`) contains **no query-relevance term**. After FAISS anchoring, candidates are ranked by trust only, so trusted but off-topic answers reach the top (many contexts share identical scores like 0.5407 / 0.4007, and rank-1 relevance is no better than ranks 2–5).

Decision (made by me, documented below): add a relevance term with a linear combination, following the trust-aware rescoring form in Li & Chen (2026, TrustPropRAG, arXiv 2609.00543):

```
final_score = (1 − α) × sim_norm(q, candidate) + α × trust
```

α is chosen on a **separate development set** by a procedure fixed **before** any dev results are seen. α = 1 (equivalent to the current ranking) is one of the candidates, so the procedure can keep the current method.

## Step 0: Plan first. Do NOT write code yet.

Report:
1. How sampling works today (pool, token filter, shuffle, `head(n)`), and exactly how you'll take positions 385–434 for the dev set without changing test-set sampling.
2. Where in `c_graphrag.py` the new scoring goes, how candidate texts are embedded (model, device, batching, caching), and the expected extra latency per question.
3. How `alpha`, `sample_split`, and `c_retrieval_version` enter the run params, `config_hash`, `run_label`, the factorial batch page, and the dry-run table.
4. Your file-by-file plan for Steps 1–6 (including Step 4b).
5. Whether AUTHOR_TRUST edges exist in the current Neo4j graph (count them), and your proposed formula for `use_author_trust`.

Wait for my approval.

## Step 1: Decision record (write first, before anything runs)

Create `docs/DECISION_C_SCORING.md` with the content below (fill in only the bracketed parts), and show it to me. I'll commit it with its date **before** any dev-set run.

```markdown
# Decision record: C ranking score (C retrieval v3)

Date written: [YYYY-MM-DD]  (must precede any dev-set run)
Author: Dartika Anie Marian

## Problem (pilot evidence, prompt v3, n=10, seed 42)
- Context relevance (ctxrel-v1, primary judge): 0% RELEVANT for B, C-uniform, C-trust, and D; C-trust 22% PARTIAL / 78% IRRELEVANT.
- C's ranking score uses trust only (0.7 path trust + 0.3 answer trust); rank-1 relevance is not higher than ranks 2-5.
- The pilot questions belong to the official test sample (nested sampling), so they are NOT used for tuning.

## Options considered
0. Keep the current score.
1. Linear: final = (1 - α) * sim_norm + α * trust  (precedent: Li & Chen 2026, TrustPropRAG).
2. Multiplicative: final = sim * (1 - λ + λ * trust)  (no direct RAG precedent).
3. Filter by similarity threshold, then rank by trust (threshold-sensitive).

Chosen: option 1. Reasons: direct precedent, one interpretable parameter, and α = 0 gives a clean relevance-only ablation (C-uniform), so C-uniform vs C-trust differ only by the trust term.

## Definitions
- sim(q, c): cosine similarity between the query embedding and the embedding of the candidate text as shown to the LLM ("Q: <title>\nA: <answer>", same truncation), using all-MiniLM-L6-v2.
- sim_norm: min-max normalization of sim over the candidate pool of each question (after dedup, before top-k). If all sims are equal, sim_norm = 1.
- trust: the existing combined trust (0.7 path trust + 0.3 answer trust; semantic-expansion candidates capped as before).
- Ranking: final_score descending, ties broken by answer_id ascending; keep top_k = 5.

## Development set
- Same pool and filter as the test sample (seed 42, oversample pool 1536, token filter), same deterministic order.
- Test sample = positions 1-384. Dev set = positions 385-434 (n = 50). Disjoint by construction.
- All tuning (α, generation prompt wording, judge-v2 wording) happens on the dev set only.

## Selection procedure (fixed before seeing dev results)
- Candidates: α ∈ {0, 0.25, 0.5, 0.75, 1}.
- Stage 1 (retrieval only, no generation): for each α, retrieve contexts for the 50 dev questions and judge context relevance (ctxrel-v1, primary).
  Primary criterion: % of questions with ≥1 RELEVANT context item.
  Tie: differences of ≤1 question (2 percentage points) are ties.
- Stage 2: for the two best α from stage 1, generate answers (grounding off and on) and judge them with judge-v2 (primary).
  Secondary criterion: hallucination rate (ABSTAIN excluded), then hallucination rate (ABSTAIN counted).
- Tie rule: if still tied, choose the larger α (the contribution under test is trust).
- Keep-current rule: if the chosen α is 1, or every α is tied with α = 1 on the primary criterion, keep the current score (C retrieval v2).
- Context-relevance judge validity: before stage 1, run positive/negative controls (own accepted answer must be RELEVANT; an unrelated item must be IRRELEVANT). If controls fail, fix ctxrel first and record that here.

## Exploratory switches (not used for selection)
max_hops, edge_types, use_author_trust, and accepted_only may be varied on the dev set to understand the system (e.g. relevance by hop and by edge type). Their results are reported descriptively. They do NOT change the selection outcome; the official runs use their defaults:
max_hops = 2, edge_types = [all currently used], use_author_trust = false, accepted_only = false.
Any change to this rule must be added to this document, with its date, BEFORE the corresponding dev runs.

## Freeze
After selection: freeze α, the generation prompt, judge-v2, and all other parameters (including the switch defaults above). Run the official test sample (n = 384) once.

## Disclosure
The 10 pilot questions (test positions 1-10) were seen during pipeline debugging and motivated judge-v2 and this decision. Judge-v2 agreement is reported on the dev set and against human annotation, not on the pilot.

## Outcome (filled in after the procedure)
[Stage 1 table, stage 2 table, chosen α, date]
```

## Step 2: Development set

1. Add a sampling option: `sample_split` = `test` (default: positions 1..n, unchanged) or `dev` (positions 385..385+n−1, default n=50). Implement it as an explicit offset on the same deterministic order, not a new seed.
2. Guarantees, with tests: dev never overlaps positions 1–384 regardless of n; the dev and test question-ID sets are disjoint; test-set sampling is byte-identical to today's (same IDs, same order).
3. `sample_split` (and the offset) are part of `config_hash`, the run metadata, the run label display (e.g. a "dev" badge), the batch metadata, and the dry-run table.
4. Factorial batch page: add a **"Dev (n=50)"** preset. Dev and test batches must be visually distinct everywhere (History, Compare, judge pages, exports).

## Step 3: C retrieval v3

1. Set `C_RETRIEVAL_VERSION = "v3"` for the new scoring; keep v2 available and selectable (`c_retrieval_version` param) so old runs stay reproducible. v2's code path must not change.
2. New param `alpha` (float, 0–1) for v3. Run labels for v3: `C-uniform` when α = 0, `C-trust` otherwise, with α shown next to the label (e.g. `C-trust (α=0.5)`). `alpha` is part of `config_hash`.
3. Scoring exactly as in the decision record: build the candidate pool as today (anchors → traversal → semantic expansion, dedup by answer_id), embed each candidate's LLM-facing text with all-MiniLM-L6-v2 on CPU (batched), compute cosine vs the query embedding, min-max normalize per question, combine with the existing trust, rank, top-5.
4. Cache candidate-text embeddings on disk, keyed by (answer_id, text hash), so sweeps don't re-encode.
5. Store per context item: `sim`, `sim_norm`, `trust`, `final_score`, `alpha`, `rank`, plus the existing fields. Store the candidate pool size per question.
6. D and B are unchanged.

## Step 4: α sweep tooling (stage 1 and stage 2)

1. **Retrieval-only mode** for C: run retrieval for a sample without calling the LLM, writing a contexts-only output file + manifest (config hash includes alpha and split). Available from the CLI and the dashboard.
2. **Sweep launcher:** given the dev split and an α list, run retrieval-only C for each α, then context relevance (ctxrel-v1, primary) on all resulting contexts (dedup across α, since many contexts repeat). Plan/confirm/progress flow with cost and time estimates.
3. **Controls first:** a button/command to run the ctxrel positive/negative controls on the dev questions (own accepted answer → expected RELEVANT; an unrelated-tag item → expected IRRELEVANT), reported as expected vs actual. These are smoke-test data, not mixed with real results.
4. **Stage 1 report** (dashboard view + markdown export), per α: % questions with ≥1 RELEVANT item (primary criterion), % RELEVANT / PARTIAL / IRRELEVANT items, mean relevance score, mean sim of selected items, mean trust of selected items, % accepted answers among selected items, mean Jaccard overlap of selected answer_ids vs α = 1 (current ranking) and vs α = 0.
5. **Selection helper:** a function that applies the decision record's stage-1 rule (ties ≤1 question) and outputs the two α values for stage 2, with the reasoning shown. It must implement the rules exactly as written; no extra heuristics.
6. **Stage 2:** the factorial batch page can launch, on the dev split, C runs for the two selected α values with grounding off and on (plus B, D, A for context if I choose). After judge-v2, a stage-2 report applies the secondary criterion, tie rule, and keep-current rule, and outputs the final α with reasoning. I'll copy the outcome into the decision record.

## Step 4b: Configurable C retrieval switches (exploratory)

Add these as run parameters for C (v3 only; v2 behavior unchanged). **Defaults must reproduce current behavior exactly.** All are part of `config_hash`, the run metadata, and the dry-run table, and are shown in run labels when non-default (e.g. `C-trust (α=0.5, hop1)`).

1. `max_hops` (1 or 2, default 2): limit graph traversal depth.
2. `edge_types` (subset of the traversed edge types, default all currently used): which relationship types traversal may follow (e.g. HAS_ACCEPTED_ANSWER, HAS_ANSWER, IS_RELATED_TO, TAG_COOCCUR, EMBED_SIM).
3. `use_author_trust` (bool, default false): when true, include author reputation via the AUTHOR_TRUST edge in the trust term. Propose the exact formula in Step 0; the trust term must stay in [0, 1]. Report whether AUTHOR_TRUST edges actually exist in the current graph (an earlier audit suggested the users file was missing at KG build time).
4. `accepted_only` (bool, default false): keep only accepted answers in the candidate pool.

Per context item, also store `hop`, `rel_type` (already present), and `via_edge_types`, so the stage-1 report can break down relevance by hop and by edge type.

**Stage 1 report additions:** % RELEVANT and % PARTIAL by hop (1 vs 2) and by edge type, for each α.

**Dashboard:** these switches live in an "Advanced (exploratory)" section of the C parameters, collapsed by default, with the note: "Non-default values are exploratory unless pre-registered in DECISION_C_SCORING.md." The Official (n=384) preset locks every C parameter to the values recorded in the decision record, and the dry-run shows a clear warning if any value differs.

**Sweep launcher:** allows adding exploratory switch values to a dev sweep, but the selection helper only applies the pre-registered rules (α); exploratory results are reported descriptively, separately from the selection outcome.

## Step 5: Tests

- Dev/test disjointness and unchanged test sampling.
- `config_hash` includes `sample_split`/offset and `alpha`; changing either changes the hash; `batch_id` still doesn't.
- **α = 1 under v3 gives the same ranking as v2 `trust_weighted`** on a fixture (trust = the existing combined score), and α = 0 ranks purely by similarity.
- Min-max normalization edge cases (all equal, single candidate).
- Embedding cache hit/miss.
- Selection helper on fixtures for every rule path (clear winner, tie → larger α, keep-current).
- Exploratory switches: defaults reproduce current v3 behavior; each switch changes `config_hash`; `max_hops=1` never returns hop-2 items; `edge_types` filtering; `accepted_only` returns only accepted answers; `use_author_trust` keeps trust in [0, 1].
- The Official preset locks C parameters to the decision record's values and warns on any difference.
- Full suite passes 3 times in a row; frontend `tsc --noEmit`, `npm run build`, `oxlint` clean.

## Step 6: Smoke test (then stop)

- Retrieval-only C v3 on **3 dev questions** for α ∈ {0, 1}: show the top-5 per α with sim, sim_norm, trust, final_score, and confirm α = 1 matches v2 trust_weighted's top-5 for the same questions.
- Don't run the full sweep, ctxrel, or any generation. I'll launch those from the UI after committing the decision record.

## Constraints

- Don't modify existing result, log, or judge files. Don't change v2 retrieval behavior, B, or D.
- `backend/` may depend on `llm/`, never the reverse.
- Embedding on CPU only (M2 8GB); stream outputs, keep memory low.
- UI style: minimalist, white background, existing plain-Tailwind conventions.

## When done, report

- Files created or changed (one line each)
- The filled-in `docs/DECISION_C_SCORING.md` (for me to review and commit)
- Test results and the Step 6 smoke-test output
- Measured extra retrieval latency per question for v3
- Exact steps for me to: commit the decision record → run the controls → run stage 1 → run stage 2
