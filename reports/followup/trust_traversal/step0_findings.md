# Step 0 findings (read-only; no code changed, no LLM/judge calls)

Official run: `llm/c_graphrag/results/condition_c_..._v4-alpha0-5_v3_95d2f39d2c.jsonl` (n=384, alpha 0.5, fw 0.7/0.3).
Code: `llm/c_graphrag/c_graphrag.py`.

## 1. Caps on the candidate pool
- Hop-1 fetch (:551-563): **no LIMIT, no per-anchor cap**. Only `NOT a.id IN $exclude_a`, ORDER BY a.id. All answers of all 3 anchors reach fusion.
- Hop-2 (:566-583): `ORDER BY edge_weight DESC, a.id ASC LIMIT 50`, a single global cap over all anchors. edge_weight = r2.weight * r3.weight (relation edge x HAS_ANSWER edge weight), NOT Answer.trustScore. Bound in only 10/384 questions (hop-2 count = 50).
- Expansion (:715-721, :647): seeds = first 3 distinct via_questions in traversal order (hop-1 items first, then hop-2); n_expansion neighbours per seed; ALL answers of those questions fetched (no LIMIT).
- Dedup (fuse_and_rank): by answer_id, keep highest combined_score. No other cap before the final rerank. Top-k=5 is the only cut.
- Conclusion: "no cap" on hop-1, so ordering alone cannot change the result. Per your rule, STOP here.

## 2. Candidates reaching the final rerank (official run, from stored `candidate_pool_size`)
mean 24.8, median 22, min 11, max 69 distinct answer_ids (5 are kept).
Composition per question: hop-1 mean 3.89, hop-2 mean 13.75, expansion mean 11.41 (before dedup).
Origin of the 1920 final items: hop1 1005 (52%), hop2 533 (28%), expansion 382 (20%).

## 3. Trust distributions of hop-1 candidates (1492 candidates, recomputed from Neo4j, read-only)
Two different quantities are both called "trust" in the code and in your brief:

| quantity | where used | ==0.0 | <0.5 | ==0.5 | in (0.5,1) | ==1.0 |
|---|---|---|---|---|---|---|
| `Answer.trustScore` (continuous, 1203 distinct values) | 0.3 weight inside combined_score | 0.9% | **69.0%** | 1.0% | 29.5% | 0.5% |
| edge `r.weight` (what the provenance file calls `trust_weight`; 50 distinct values) | 0.7 weight inside combined_score | 12.4% | 16.1% | **41.6%** | 0.4% | **42.0%** |

Histogram of trustScore (bin start: n): 0:93 (80 exactly-ish 0..0.1 incl 13 at 0), 0.1:321, 0.2:303, 0.3:171, 0.4:142, 0.5(exact):15, 0.5-0.6:176, 0.6:164, 0.7:57, 0.8:22, 0.9:21, 1.0:7.
Edge weight is almost bimodal: 0.5 (620), 1.0 (626), 0.0 (185), other (61).

## Conflicts with the brief, need your decision
1. **MIN_TRUST=0.5 on `Answer.trustScore` would prune 69% of hop-1 answers.** The 0.5 threshold is sharp only for the edge weight (a plateau of 41.6% exactly at 0.5), not for trustScore. T1 as specified (trustScore < 0.5) is a very aggressive filter; on edge weight it is a mild one (drops the 16% below 0.5, i.e. the 0.0s).
2. **Which trust do T1/T2/T3 use?** trustScore, edge weight, or combined_score (0.7*edge + 0.3*trustScore)? I'd suggest combined_score-free options only if you want "trust at traversal"; please pick.
3. **Hop-1 is thin:** 80% of anchors have exactly 1 answer (mean 1.30, max 11); only 5.6% of anchors have >2. So T2 (top-K=2 per anchor) can only touch ~5.6% of anchors, and T1 with the keep-one rule can only remove answers from anchors that have >=2. Hop-1 variants will therefore change little by construction.
4. The brief says trust "only acts in hop-2 pruning". Accurate refinement: hop-2 pruning uses the edge-weight product (not trustScore) and binds in only 10/384 questions; trust otherwise acts via combined_score in fusion/rerank (alpha=0.5).
5. Hop-1 items are 52% of the final context, so the hop-1 stage is where most context originates; but with so few candidates there, pruning there mostly removes items that would otherwise be selected or replaced by hop-2/expansion items.

Not yet done (waiting for your answer): any code changes, replay, T0 reproducibility check, metrics.
