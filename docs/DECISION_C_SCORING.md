# Decision record: C ranking score (C retrieval v3)

Date written: 2026-10-06  (must precede any dev-set run)
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
- sim(q, c): cosine similarity between the query embedding and the embedding of the candidate text **as shown to the LLM** (the final, post-truncation `chunk_text`: "Q: <title>\nA: <answer>", truncated to the token-chunk limit exactly as stored in the output record), using all-MiniLM-L6-v2. MiniLM's tokenizer reads at most 256 word pieces, so for longer `chunk_text`, `sim` reflects only the leading portion. This limitation already applies wherever this model is used in the pipeline (e.g. `compute_similarity()`); it is not introduced by v3.
- sim_norm: min-max normalization of sim over the candidate pool of each question (after dedup, before top-k). If all sims are equal, sim_norm = 1.
- trust: the existing combined trust (0.7 path trust + 0.3 answer trust; semantic-expansion candidates capped as before).
- Ranking: final_score descending, ties broken by answer_id ascending; keep top_k = 5.

## Embedding consistency (verified before writing any v3 code)
- The FAISS-cache question embeddings (`01_data_cleaning/11_densify_embedding_similarity.py`, `--stage encode`) are produced with `SentenceTransformer("all-MiniLM-L6-v2")` and explicitly L2-normalized before being written to the memmap. The FAISS index is `IndexIVFFlat` with `METRIC_INNER_PRODUCT`, a correct cosine proxy only because the stored vectors are pre-normalized.
- `c_graphrag.py::embed_query()` (fallback when a question isn't in the cache) also L2-normalizes its output.
- v3's candidate-text embeddings use the same model and the same explicit L2 normalization, so `sim(q, c)` is a true cosine similarity on both sides. The existing `query_emb` is reused as-is.

## Development set
- Same pool and filter as the test sample (seed 42, oversample pool 1536, token filter), same deterministic order (one permutation with `df.sample(frac=1.0, random_state=42)`).
- Test sample = positions 1-384 (1-based), i.e. 0-based rows `iloc[0:384]`.
- Dev set = positions 385-434 (1-based), i.e. 0-based rows `iloc[384:434]` (n = 50). Disjoint from the test sample by construction.
- All tuning (α, generation prompt wording, judge-v2 wording, context-relevance judge wording) happens on the dev set only.

## Selection procedure (fixed before seeing dev results)
- Candidates: α ∈ {0, 0.25, 0.5, 0.75, 1}.
- Stage 1 (retrieval only, no generation): for each α, retrieve contexts for the 50 dev questions and judge context relevance (primary judge; judge version as recorded in Amendments 1-2).
  - Primary criterion: % of questions with ≥1 RELEVANT context item. Differences of ≤1 question (2 percentage points) are ties.
  - Stage-1 tie-breaker: mean relevance score over all selected context items (RELEVANT = 1, PARTIAL = 0.5, IRRELEVANT = 0). Differences ≤ 0.02 are ties.
  - Ranking: by primary criterion, then tie-breaker, then larger α. The two highest-ranked α values go to stage 2.
- Keep-current rule (stage 1): if α = 1 ranks first, or every α is tied with α = 1 on both the primary criterion and the tie-breaker, keep the current score (C retrieval v2) and skip stage 2.
- Stage 2: for the two α values from stage 1, generate answers for the 50 dev questions with grounding off and on (100 answers per α) and judge them with judge-v2 (primary).
  - Secondary criterion: hallucination rate with ABSTAIN excluded, pooled over both grounding settings. Differences of ≤2 answers (2 percentage points) are ties.
  - Then: hallucination rate with ABSTAIN counted as non-factual, pooled, same tie margin.
  - If still tied: choose the larger α (the contribution under test is trust).
- Keep-current rule (stage 2): if the chosen α is 1, keep the current score (C retrieval v2).
- Judge-v2 must be frozen before stage 2 (prompt version, parser version, max_tokens, model registry entries). Record those here before running stage 2.
- Context-relevance judge validity: before stage 1, run positive/negative controls. If controls fail, fix the context-relevance judge first and record that here (see Amendments 1-2).

## Exploratory switches (not used for selection)
max_hops, edge_types, use_author_trust, and accepted_only may be varied on the dev set to understand the system (e.g. relevance by hop and by edge type). Their results are reported descriptively. They do NOT change the selection outcome; the official runs use their defaults:
max_hops = 2, edge_types = [all currently used], use_author_trust = false, accepted_only = false.
Any change to this rule must be added to this document, with its date, BEFORE the corresponding dev runs.

### `use_author_trust` formula (2026-10-06)
When `use_author_trust = true`: `trust' = (1 − β) × trust + β × author_weight`, with **β = 0.3** (fixed) and `author_weight = 0` when the answer's author has no `AUTHOR_TRUST` edge (e.g. deleted/anonymous user). `author_weight` is the `AUTHOR_TRUST` edge's `weight` property, already in `[0, 1]` (`normalize(log1p(reputation))`, clipped at P99; see `01_data_cleaning/9_load_author_trust.py`). Since both terms are in `[0, 1]` and this is a convex combination, `trust'` stays in `[0, 1]`. Verified against the real Neo4j graph: 715,604 `AUTHOR_TRUST` edges and 293,078 `User` nodes exist. Exploratory only, default `false`, never affects the selection outcome.

## Freeze
After selection: freeze α, the generation prompt, judge-v2, the context-relevance judge version, and all other parameters (including the switch defaults above). Run the official test sample (n = 384) once.

## Disclosure
The 10 pilot questions (test positions 1-10) were seen during pipeline debugging and motivated judge-v2 and this decision. Judge-v2 agreement is reported on the dev set and against human annotation, not on the pilot.

---

## Amendment 1 (2026-10-07): context-relevance controls and dev offset

Written after the first control run and BEFORE any stage-1 run.

### What happened
- The first control run used 1 positive and 1 negative item. The positive control (the dev question's own accepted answer) was judged PARTIAL instead of RELEVANT by ctxrel-v1 (primary). The negative control was correctly IRRELEVANT.
- The run log printed `dev_offset=385`. The dev set is defined above as 0-based rows `iloc[384:434]`. Verification result (2026-10-07): the log printed a 1-based number; the code converts it with `start = dev_offset - 1`, so the rows used were 0-based `iloc[384:434]` (contiguous), as defined. First 3 question IDs: 20952138, 55753657, 60887798 (0-based positions 384, 385, 386; identical to the v3 smoke test). No overlap with the test sample (last test row, `iloc[383]`, is question 52558999). No functional fix was needed; the log line now prints both the 1-based number and the 0-based range, and a test pins the dev rows to 0-based 384-433.
- No stage-1 retrieval relevance was judged. No selection-relevant result has been seen.

### Redefined controls
One item is too few to evaluate the judge, so the controls are redefined:
- **Positive controls:** for each of the 50 dev questions, its own accepted answer as the context item, formatted exactly like a retrieved context item ("Q: <title>\nA: <answer>", same truncation). Expected: RELEVANT.
- **Negative controls:** for each dev question, one accepted answer from a different question that shares no tag with it (selected with seed 42), same formatting. Expected: IRRELEVANT.
- **Judges:** primary and secondary. The primary judge decides pass/fail; the secondary is reported for reference.
- **Pass thresholds:** ≥ 90% of positive controls judged RELEVANT AND ≥ 90% of negative controls judged IRRELEVANT.

### If ctxrel-v1 fails the redefined controls
- A revised context-relevance judge (ctxrel-v2) may be written.
- It may be tuned ONLY using the control items. It must never be tuned using stage-1 retrieved contexts or per-α results.
- The controls are re-run with the revised judge; it must pass the same thresholds.
- Stage 1 uses the judge version that passes. Its prompt version is recorded below before stage 1 runs.
- The pilot's ctxrel-v1 results remain as reported, labeled with their version.

### Record
- Control fix (2026-10-07): the first control run omitted the question body; real stage-1 judging includes it. The controls were rebuilt to match stage-1 inputs exactly (question title, tags, and body; context item truncated to 400 tokens with cl100k_base, as in `fuse_and_rank`; raw HTML kept, as in retrieval). All control results below use the rebuilt controls.
- Truncation check: 15 of 50 positive items were truncated; the primary judge rated 14 of those 15 RELEVANT (vs 30 of 35 untruncated), so truncation does not explain the failures.
- Control results, ctxrel-v1 (primary / secondary): positives RELEVANT 88% / 74% (PARTIAL 12% / 26%, IRRELEVANT 0% / 0%); negatives IRRELEVANT 100% / 100%. Pass: **no** (primary 44/50 positives RELEVANT; threshold 45/50).
- Failure pattern: every miss was a positive judged PARTIAL. The judges' reasons show RELEVANT being read as "completely solves the problem in the asker's exact way": accepted answers giving a workaround, an alternative approach, or the cause were judged PARTIAL.
- Outputs: `llm/c_graphrag/results/controls/ctxrel_v1_controls_20261007T154329Z.jsonl` and `_summary.json` (control data only).
- Revision: see Amendment 2.
- Date: 2026-10-07

---

## Amendment 2 (2026-10-07): ctxrel-v2 and hard-negative controls

Written after the ctxrel-v1 control results and BEFORE any ctxrel-v2 run or stage-1 run.

### ctxrel-v2
ctxrel-v2 changes only the label definitions; inputs, output format, and parser are the same as v1.
- RELEVANT: the item contains information that directly helps solve or explain THIS question's problem: a fix, a workaround, the cause, a recommended approach or tool, or a key part of the solution. It need not be complete, match the exact versions or wording, or follow the approach the asker tried; an alternative approach that achieves the asker's underlying goal counts. The item may be cut off; only what is shown is judged.
- PARTIAL: same technology or topic, but a different problem, or only tangentially useful.
- IRRELEVANT: unrelated to this problem.
- Sharing a technology, library, or keyword with the question is not enough on its own for RELEVANT.

Full prompt committed at: `4c1c73528302797f5b8fab78373c6a12753dc414` (`llm/evaluation/ctxrel_prompt_v2.py`; the v1 prompt text is unchanged and pinned by a hash test).

### Third control type: hard negatives
The existing negatives (no shared tag) cannot detect a judge that is too lenient on same-technology, different-problem items, which is the RELEVANT/PARTIAL boundary stage 1 depends on. A third control set is added:
- **Hard negatives:** for each of the 50 dev questions, one accepted answer from a different question in the same candidate pool that shares at least one tag with it (selected with seed 42), excluding questions linked to it by `IS_RELATED_TO` (Linked/Duplicate). Same formatting and truncation as stage-1 items. Expected: NOT RELEVANT (PARTIAL or IRRELEVANT).
- Dev questions with no eligible hard negative are reported and excluded from the hard-negative denominator.

### Pass thresholds (primary judge decides; secondary reported)
- Positives: ≥ 90% RELEVANT
- Easy negatives (no shared tag): ≥ 90% IRRELEVANT
- Hard negatives (shared tag): ≥ 80% NOT RELEVANT

The hard-negative threshold is lower because a same-tag question can occasionally address a genuinely similar problem.

ctxrel-v1 is also run on the hard negatives, for comparison only (it already failed on positives and cannot be used for stage 1).

### Iteration limit
- If ctxrel-v2 fails, at most one further revision (ctxrel-v3) is allowed. It is tuned only on control items and recorded here, with its commit hash, before it runs.
- If no version passes after that, stage 1 uses the version with the highest primary positive-control rate among those meeting both negative thresholds, and this is reported as a limitation.

### Record
The hard-negative definition above was replaced by Amendment 3 before any judge run. Results are recorded under Amendment 3.

---

## Amendment 3 (2026-10-07): harder hard negatives, donors restricted to non-test questions

Written after a dry build of the Amendment 2 hard negatives on real data (no judge calls) and BEFORE any ctxrel-v2 run or stage-1 run.

### Why
- The Amendment 2 hard negatives were mostly easy: 46 of 47 shared exactly one tag, usually a broad one (`android` 8, `javascript` 5, `python` 3, `html`, `php`, `java`). Example: "Best way to convert RTMP to MP4" received a CSS background-colour answer because both are tagged `html`. Such items give little evidence about the RELEVANT/PARTIAL boundary these controls exist to test.
- 14 of the 47 donors were test-sample questions (positions 1-384), so their accepted answers would have been used to tune the context-relevance judge.
- The `IS_RELATED_TO` exclusion removed nothing in practice (six dev questions have linked neighbours, none of them in the pool).

### New hard-negative definition (replaces the Amendment 2 definition)
- **Donor pool:** questions in the same candidate pool (seed 42, oversample pool 1536, token filter) that are NOT in the test sample (0-based positions 0-383), excluding the dev question itself and any question linked to it by `IS_RELATED_TO` (either direction).
- **Eligibility:** the donor shares at least one tag with the dev question.
- **Selection:** among eligible donors, the one whose question embedding (all-MiniLM-L6-v2, L2-normalized, the existing FAISS-cache vectors) has the **highest cosine similarity** to the dev question. Ties broken by question ID ascending. No randomness.
- **Item:** the donor's accepted answer, formatted and truncated exactly like a stage-1 context item.
- **Expected:** NOT RELEVANT (PARTIAL or IRRELEVANT).
- Dev questions with no eligible donor are reported and excluded from the hard-negative denominator.
- The donor's cosine similarity, number of shared tags, and shared tags are recorded per item.

### Unchanged
- ctxrel-v2 prompt (committed at `4c1c735`), positive controls, easy negatives, pass thresholds (positives ≥ 90% RELEVANT, easy negatives ≥ 90% IRRELEVANT, hard negatives ≥ 80% NOT RELEVANT; primary judge decides), and the iteration limit from Amendment 2.
- Because the most-similar same-tag donor can occasionally address a genuinely similar problem, the 80% threshold is kept rather than raised.

### Reproducibility
All code used for the control runs (runner, judge module with ctxrel-v2, hard-negative builder, dev-offset test) is committed before any run; control outputs and manifests record that commit.

### Record (fill in before stage 1)
- Code commit used for the control runs: [commit hash]
- Hard negatives: [n eligible] of 50 dev questions had an eligible donor; donor cosine similarity median [x] (range [x]-[x]); donors sharing ≥2 tags: [n].
- ctxrel-v1 hard negatives (primary / secondary): NOT RELEVANT [x]% / [x]%
- ctxrel-v2 (primary / secondary): positives RELEVANT [x]% / [x]%; easy negatives IRRELEVANT [x]% / [x]%; hard negatives NOT RELEVANT [x]% / [x]%. Pass: [yes/no]
- Context-relevance judge version used for stage 1: [ ]
- Date: [YYYY-MM-DD]

---

## Outcome (filled in after the procedure)
[Stage 1 table, stage 2 table, chosen α, date]

## Machine-readable official parameters

The block below is the single source of truth the dashboard's "Official (n=384)" preset reads to lock Condition C's parameters (`llm/c_graphrag/official_params.json` is kept byte-identical to this block by `llm/c_graphrag/tests/test_official_params_sync.py`). While `status` is `pending_selection`, the dashboard refuses to launch any Official run, regardless of `c_retrieval_version`. Update this block (and `official_params.json`) only in the same commit that fills in the Outcome section, never before.

```json official_params
{
  "status": "pending_selection",
  "c_retrieval_version": null,
  "alpha": null,
  "sample_split": "test",
  "max_hops": 2,
  "edge_types": ["EMBED_SIM", "HAS_ACCEPTED_ANSWER", "HAS_ANSWER", "IS_RELATED_TO", "TAG_COOCCUR"],
  "use_author_trust": false,
  "accepted_only": false
}
```