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
- sim(q, c): cosine similarity between the query embedding and the embedding of the candidate text **as shown to the LLM** (the final, post-truncation `chunk_text` — "Q: <title>\nA: <answer>", truncated to the token-chunk limit exactly as stored in the output record), using all-MiniLM-L6-v2. Note: MiniLM's own tokenizer truncates to 256 word pieces internally regardless of how long the input text is, so for a `chunk_text` longer than ~256 word pieces, `sim` reflects only the leading portion MiniLM actually reads — the same leading-portion-only limitation already implicit in using this model anywhere else in the pipeline (`compute_similarity()`), not a new limitation introduced by v3.
- sim_norm: min-max normalization of sim over the candidate pool of each question (after dedup, before top-k). If all sims are equal, sim_norm = 1.
- trust: the existing combined trust (0.7 path trust + 0.3 answer trust; semantic-expansion candidates capped as before).
- Ranking: final_score descending, ties broken by answer_id ascending; keep top_k = 5.

## Embedding consistency (verified before writing any v3 code)
- The FAISS-cache question embeddings (`01_data_cleaning/11_densify_embedding_similarity.py`, `--stage encode`) are produced with `SentenceTransformer("all-MiniLM-L6-v2")` and are **explicitly L2-normalized** before being written to the memmap (`embeddings = embeddings / norms`) — the FAISS index itself is `IndexIVFFlat` with `METRIC_INNER_PRODUCT`, which is only a correct cosine-similarity proxy because the stored vectors are pre-normalized.
- `c_graphrag.py::embed_query()` (used for the fallback path when a question isn't already in the cache) also explicitly L2-normalizes its output the same way.
- Both the cache-hit path and the fallback path therefore produce vectors in the **same model, same normalization convention** as each other. v3's candidate-text embeddings will be computed with the same model and explicitly L2-normalized the same way, so `sim(q, c)` is a dot product of two L2-normalized vectors on both sides — a true cosine similarity, consistent with the existing query-embedding convention. No on-the-fly re-embedding of the query is needed; the existing `query_emb` (cache hit or `embed_query()` fallback) is reused as-is for v3's scoring.

## Development set
- Same pool and filter as the test sample (seed 42, oversample pool 1536, token filter), same deterministic order.
- Test sample = positions 1-384. Dev set = positions 385-434 (n = 50). Disjoint by construction.
- All tuning (α, generation prompt wording, judge-v2 wording) happens on the dev set only.

## Selection procedure (fixed before seeing dev results)
- Candidates: α ∈ {0, 0.25, 0.5, 0.75, 1}.
- Stage 1 (retrieval only, no generation): for each α, retrieve contexts for the 50 dev questions and judge context relevance (ctxrel-v1, primary).
  - Primary criterion: % of questions with ≥1 RELEVANT context item. Differences of ≤1 question (2 percentage points) are ties.
  - Stage-1 tie-breaker: mean relevance score over all selected context items (RELEVANT = 1, PARTIAL = 0.5, IRRELEVANT = 0). Differences ≤ 0.02 are ties.
  - Ranking: by primary criterion, then tie-breaker, then larger α. The two highest-ranked α values go to stage 2.
- Keep-current rule (stage 1): if α = 1 ranks first, or every α is tied with α = 1 on both the primary criterion and the tie-breaker, keep the current score (C retrieval v2) and skip stage 2.
- Stage 2: for the two α values from stage 1, generate answers for the 50 dev questions with grounding off and on (100 answers per α) and judge them with judge-v2 (primary).
  - Secondary criterion: hallucination rate with ABSTAIN excluded, pooled over both grounding settings. Differences of ≤2 answers (2 percentage points) are ties.
  - Then: hallucination rate with ABSTAIN counted as non-factual, pooled, same tie margin.
  - If still tied: choose the larger α (the contribution under test is trust).
- Keep-current rule (stage 2): if the chosen α is 1, keep the current score (C retrieval v2).
- Judge-v2 must be frozen before stage 2 (prompt version, parser version, max_tokens). Record those versions here before running stage 2.
- Context-relevance judge validity: before stage 1, run positive/negative controls (own accepted answer must be RELEVANT; an unrelated item must be IRRELEVANT). If controls fail, fix ctxrel first and record that here.

## Exploratory switches (not used for selection)
max_hops, edge_types, use_author_trust, and accepted_only may be varied on the dev set to understand the system (e.g. relevance by hop and by edge type). Their results are reported descriptively. They do NOT change the selection outcome; the official runs use their defaults:
max_hops = 2, edge_types = [all currently used], use_author_trust = false, accepted_only = false.
Any change to this rule must be added to this document, with its date, BEFORE the corresponding dev runs.

### `use_author_trust` formula (2026-10-06)
When `use_author_trust = true`: `trust' = (1 − β) × trust + β × author_weight`, with **β = 0.3** (fixed) and `author_weight = 0` when the answer's author has no `AUTHOR_TRUST` edge (e.g. deleted/anonymous user). `author_weight` is the `AUTHOR_TRUST` edge's own `weight` property, already in `[0, 1]` (`normalize(log1p(reputation))`, clipped at P99 — see `01_data_cleaning/9_load_author_trust.py`). Since `trust` and `author_weight` are both already in `[0, 1]` and this is a convex combination (`β ∈ [0, 1]`), `trust'` stays in `[0, 1]` by construction. Verified against the real Neo4j graph before proposing this: **715,604 `AUTHOR_TRUST` edges, 293,078 `User` nodes exist** — the edges are present (an earlier audit's "users file missing at KG build time" concern does not hold for the current graph). This switch is exploratory only, default `false`, and never affects the selection outcome.

## Freeze
After selection: freeze α, the generation prompt, judge-v2, and all other parameters (including the switch defaults above). Run the official test sample (n = 384) once.

## Disclosure
The 10 pilot questions (test positions 1-10) were seen during pipeline debugging and motivated judge-v2 and this decision. Judge-v2 agreement is reported on the dev set and against human annotation, not on the pilot.

## Outcome (filled in after the procedure)
[Stage 1 table, stage 2 table, chosen α, date]

## Machine-readable official parameters

The block below is the single source of truth the dashboard's "Official
(n=384)" preset reads to lock Condition C's parameters (see
`llm/c_graphrag/official_params.json`, kept byte-identical to this block
by `llm/c_graphrag/tests/test_official_params_sync.py`). Until the
Outcome section above is filled in, `status` is `"pending_selection"` --
the dashboard must refuse to launch an Official run whenever `status`
is `"pending_selection"`, **regardless of `c_retrieval_version`**, and
point the user at this decision record. Update this block (and
`official_params.json` alongside it) on the same commit that fills in
the Outcome section -- set `status` to `"locked"` at that point, never
before.

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
