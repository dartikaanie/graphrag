## Amendment 8 (2026-10-18): per-query leakage exclusion for B, C and D

**Status: confirmed by the author.** Written after stage 1 and before stage 2 and the official evaluation on the test sample (n = 384). The decision was made before stage 2 and before any official n = 384 run. No retrieval or generation was run on the test sample; for the test sample only aggregate counts of related questions were computed. Technical details are in `reports/leakage_audit.md`.

### Finding
- Every retrieval corpus (B: all KG answers; C and D: the KG) also contains the evaluation questions and their answers, because the KG is built from the whole Stack Overflow population.
- Without any exclusion, retrieval almost always returns the evaluation question's own thread. On the dev set (positions 385–434, n = 50, retrieval only, no LLM), the share of questions whose context contains an id from the question's own thread is **B 84% (42/50), C 100% (50/50), D 100% (50/50)**. C and D reach 100% because both anchor on the question embedding, and the question itself is the top anchor when it is not excluded.
- In the sampling pool (1,471 questions), 13 questions (0.9%) have an `IS_RELATED_TO` Duplicate, 3 (0.2%) have a near-duplicate question (cosine ≥ 0.95), and 209 (14.2%) have a Linked question. Dev set (n = 50): 0, 0 and 6. Test sample (aggregate counts only): 2 Duplicate, 1 near-duplicate.

### Decision
1. **Exclusion set per evaluation question q** (one shared function for B, C and D, `build_exclusion()` in `llm/common/leakage.py`): q and every answer in q's thread (including the accepted answer); every Question linked to q by `IS_RELATED_TO {link_type: 'Duplicate'}`, either direction, with all its answers; every Question whose Title+Body embedding has cosine ≥ 0.95 with q's (C's question embeddings and index), with all its answers. Exclusion is by question identity and similarity only, never by the content of q's accepted answer.
2. **Linked is not excluded** (`LEAKAGE_EXCLUDE_LINKED=False`). A Linked question is community knowledge that legitimately relates to q, not q's ground truth. The audit supports this: on the dev set no retrieved context item came from a Linked neighbour, in any condition; for the 98 non-test pool questions with answered Linked neighbours, no neighbour answer reached cosine ≥ 0.95 with q's accepted answer (maximum 0.9468). Excluding Linked is kept only as a sensitivity mode.
3. **Near-duplicate threshold 0.95.** At 0.95 the threshold catches 3 pool questions (7 related questions). The 0.9468 Linked case shows that a lower threshold starts to catch legitimate on-topic answers.
4. **Settings** `LEAKAGE_FILTER` (default True), `LEAKAGE_EXCLUDE_LINKED` (default False) and `LEAKAGE_NEAR_DUP_COSINE` (default 0.95) are part of each condition's config and config_hash as `"leakage": "dup+neardup0.95"`. The key is absent for the earlier thread-only exclusion, so the config_hash of every earlier run is unchanged. `LEAKAGE_FILTER=False` is for the audit only.
5. **Implementation.** The related-question part is precomputed offline for the whole sampling pool into `results/leakage/exclusion_sets.parquet` (byte-identical on rebuild); B reads it and never accesses Neo4j at runtime. Every condition asserts per query that retrieval results ∩ exclusion set = ∅, and logs the mode, the number of excluded questions and answers, the number of filtered hits and the final number of context items.

### Effect
- With the exclusion, ID-level self-retrieval is **0% for B, C and D** in every mode, and no context item has cosine ≥ 0.95 with the accepted answer.
- **No effect on stage 1 (α selection).** In the default mode, 0 of the 50 stage-1 dev exclusion sets change compared with the thread-only exclusion used in stage 1, and C's retrieval is identical for 50/50 questions at α = 0.5 and at α = 0.25 (ids, order, scores and text). Stage 1 is therefore not re-run.

### Metrics
The official leakage metrics are the ID-level self-retrieval rate and the cosine similarity (≥ 0.95) between each context item and q's accepted answer. The cosine is a measurement only, never a filter. A third metric, an identical substring of ≥ 50 characters between context and accepted answer, was tried and abandoned: on raw HTML every match found with the filter on (26 of 26 checked) was boilerplate (Android XML attributes, documentation URLs, Gradle lines, standard imports), not answer content.

### Justification
This is a correction of evaluation validity (ground truth must not reach the prompt through retrieval), decided on the structure of the corpus and the audit, not on hallucination results. It was decided before stage 2 and before the official evaluation on the test sample.
