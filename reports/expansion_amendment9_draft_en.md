## Amendment 9 (2026-10-18): semantic expansion selects only answerable Questions

**Status: confirmed by the author. Written before any run of the change** (no retrieval, no judging and no stage-1 recheck had been run with it). Only the Outcome section was filled in afterwards, from the recheck. Written after stage 1 and before stage 2 and the official evaluation on the test sample (n = 384). The decision was made before stage 2 and before any official n = 384 run. Technical details are in `reports/expansion_answered.md`.

### Finding
- In Condition C, stage 3 (Semantic Expansion) picks, per seed question, the `n_semantic_expansion` (= 3) Questions most similar to the seed, then fetches their answers. Many of those Questions have no Answer in the KG at all: the KG holds 2,685,814 Questions but only 725,795 Answers (602,900 Questions with at least one Answer).
- In the leakage audit (Amendment 8, dev set n = 50, α = 0.5), **64% of the expansion Questions (277 of 432) contributed no answer**, and none of these was caused by the leakage exclusion: every one of them has no Answer in the KG. The expansion stage therefore fills most of its slots with candidates that cannot contribute context.
- Among the first 18 FAISS candidates of each seed, 32.7% of the Questions have an Answer; only 11 of 144 dev seeds get 3 answerable Questions from the current top-3.

### Change
1. **Only Semantic Expansion changes.** Entity Anchoring, Graph Traversal, the trust scores, α and `fuse_and_rank()` are unchanged. Condition D is unchanged (its high-level step already requires an answer with `EXISTS`).
2. **Over-fetch:** per seed, `search_n = 20 · top_n + |exclude_ids|` FAISS candidates (60 + excluded ids for top_n = 3). The factor 20 was chosen from the distribution of Answer existence in the KG on dev-50 — share of seeds that get top_n answerable Questions: 6× → 83%, 12× → 97%, 20× → 100% (largest rank needed: 51) — not from hallucination or relevance results.
3. **Verification:** the candidates of all seeds are checked in one parameterised Cypher round-trip; a Question is kept only if it has at least one Answer not in the leakage exclusion (`EXISTS { (q)-[:HAS_ANSWER|HAS_ACCEPTED_ANSWER]->(a:Answer) WHERE NOT a.id IN $exclude_a }`, honouring the edge-type filter).
4. **Selection:** per seed, in order, the first `top_n` verified candidates by similarity (tie-break: question id ascending) that no earlier seed has taken. If fewer than `top_n` are verified, the available ones are used, and the record logs `n_expansion_requested` and `n_expansion_answerable`.
5. **Setting** `EXPANSION_REQUIRE_ANSWER` (default True) is part of C's config and config_hash (`"expansion_require_answer": true, "expansion_overfetch": 20`); the keys are absent when it is False, so the config_hash of every earlier run is unchanged. False reproduces the previous expansion exactly and is for audit only. The CLI, the dashboard and `stage1_sweep.py` follow the setting; `reputation_ablation_check.py` and `investigate_zero_context.py` deliberately keep the previous expansion to reproduce Amendment 5.

### Stage-1 recheck (protocol fixed before any result)
Because the change alters C's retrieval, stage 1 is checked again with **exactly the same protocol**: the same dev set (positions 385–434, n = 50), the same α grid {0, 0.25, 0.5, 0.75, 1}, the same primary context-relevance judge (ctxrel-v2, temperature 0), the same metrics and the same selection rule (`select_stage1_candidates`). Judge labels are reused only when question_id, answer_id, text_hash, judge model and prompt version are all identical; every other item is judged anew. **The result is accepted whatever it is, including if the selected α changes.**

### Outcome
Recheck run on 2026-10-18 (report `llm/c_graphrag/results/expansion_check/stage1_expansion_report_20261018T042953Z.md`).
- **Reproduction:** with the switch off, the top-5 is identical to the Amendment 6 stage-1 result of record (50 × 5), the recomputed metrics equal the recorded ones, and top_two stays [0.5, 0.25].
- **Expansion:** empty expansion slots fall from 64.1% to 0% (432 expansion Questions on 50 dev questions, every α). Context items from expansion in the top-5 roughly double (α = 0.5: 36 → 82; α = 0.25: 37 → 88). Questions with fewer than 5 context items fall from 5 to 2. Anchors and traversal candidates are unchanged for every question.
- **Questions whose top-5 changed:** 38 / 50 at α = 0.5 and 34 / 50 at α = 0.25.
- **Latency:** C retrieval-only p95 changes by about +0.01–0.02 s, for example α = 0.5 from 0.091 to 0.106 s.
- **Judging:** 405 cached labels were reused under the identity rule, and 119 new items were judged (ctxrel-v2, primary, `meta-llama/Llama-3.3-70B-Instruct`, temperature 0; 0 failures). The new labels are in `results/expansion_check/ctxrel_v2_expansion_primary.jsonl`. Cost was $0.0137.

| α | % q ≥1 RELEVANT (old → new) | mean score (old → new) | % accepted (old → new) |
|---|---|---|---|
| 0 | 48.0 → 50.0 | 0.3263 → 0.3525 | 37.29 → 38.52 |
| 0.25 | 48.0 → 48.0 | 0.3305 → 0.3504 | 41.95 → 47.13 |
| 0.5 | 48.0 → 48.0 | 0.3157 → 0.3299 | 53.39 → 61.89 |
| 0.75 | 46.0 → 44.0 | 0.2818 → 0.2848 | 66.95 → 76.23 |
| 1 | 34.0 → 28.0 | 0.2225 → 0.2029 | 72.88 → 85.66 |

- **Ranking** under the unchanged rule: 0.25, 0.0, 0.5, 0.75, 1.0.
  - α = 0, 0.25 and 0.5 tie on the primary criterion (within 2 pp).
  - α = 0.5 falls more than 0.02 below α = 0 on mean score.
  - α = 0.25 and α = 0 tie on both criteria, so the larger α ranks first.
- **top_two = [0.25, 0.0]** (previously [0.5, 0.25]). keep_current = False.
- As pre-registered, this result is accepted. **Stage-2 candidates of record: α = 0.25 and α = 0.** At α = 0, `final_score` is similarity only.

### Justification
This is a structural correction (an expansion slot must be able to contribute context), decided on the structure of the KG, not on hallucination results. It was decided before stage 2 and before the official evaluation on the test sample.
