## Amendment 10 (2026-10-18): trust ablation on the official evaluation

**Status: confirmed by the author.** Written after Amendment 9 and **before stage 2 and before the official evaluation on the test sample (n = 384)**. No stage-2 result exists; no generation or judging has been run for stage 2 or on the test sample.

### Motivation
- The Amendment 9 stage-1 recheck gave top_two = [0.25, 0.0], and the two are nearly tied (% questions with ≥1 RELEVANT item 48.0 vs 50.0; mean relevance score 0.3504 vs 0.3525). α = 0.25 ranks first through the "larger α" tie-break.
- Stage 2 only chooses α on the dev set (n = 50). It is a selection step, not a measurement of the effect of trustScore, and its result may again be decided by a tie-break.
- The effect of trustScore is the core contribution of the thesis, so it is measured on the test data.

### Decision
1. **Both stage-2 candidates (α = 0.25 and α = 0) are run in the official evaluation on the test sample (n = 384), whatever the stage-2 result.**
   - **C-primary** = the stage-2 winner under the existing selection rule.
   - **C-ablation** = the other stage-2 candidate.
2. **Identical procedure; only α differs.** Sample, generation prompt, generation model, grounding on and off, judge, and the whole retrieval pipeline are the same: edge types, max hops, leakage exclusion (Amendment 8), answerable-only expansion (Amendment 9), `use_author_trust` and `accepted_only`.
3. **Separate runs.** Each run has its own config_hash (α is part of the config). Run labels are `c_primary` and `c_ablation`; they are set from the stage-2 result, not by hand.
4. **Judge.** C-primary and C-ablation, like every condition in the official evaluation, are judged with the same frozen judge as stage 2, using the binary hallucination metric of Amendment 4.
5. **Main analysis unchanged.** The comparison of C-primary with A, B and D stays the main analysis. C-ablation is reported in addition to it and does not replace any condition.

### What α = 0 removes (naming)
In C v3, `final_score = (1 − α) · sim_norm + α · trust`. At α = 0 the final ranking is similarity only:
- trustScore (votes, accepted flag, author reputation) does not enter the ranking;
- `use_author_trust` and `accepted_only` are off in the official parameters;
- anchoring, 1-hop traversal and semantic expansion do not use any trust signal.

**C-ablation (α = 0) removes trustScore from the final ranking; the accepted/vote signal remains only indirectly, through the order in which 2-hop candidates are cut (LIMIT 50; active for 2 of 50 dev questions).**

That cut orders 2-hop candidates by `r2.weight · r3.weight`, where r3 is `HAS_ACCEPTED_ANSWER` (1.0) or `HAS_ANSWER` (the vote score min-max normalised per question). The 2-hop path is not changed. The thesis therefore calls C-ablation "without trustScore in the final ranking", not "without any trust signal".

### Analysis of C-primary vs C-ablation (descriptive, no inferential test)
On the n = 384 test sample, per grounding setting (on, off) and pooled:
- **Binary hallucination rate** for each run, with ABSTAIN excluded and with ABSTAIN counted as non-factual (Amendment 4).
- **Paired counts per question** (same question, same grounding setting): hallucinated in C-ablation but not in C-primary; hallucinated in C-primary but not in C-ablation; hallucinated in both; hallucinated in neither.
  - The table is given with ABSTAIN counted as non-factual.
  - It is also given on the pairs where neither answer is ABSTAIN, together with the number of pairs left out.
- **Cosine similarity** to the accepted answer and **NF2** (share of answers with a valid citation), reported for both runs in the same way as for the other conditions.

### Limits of the claims (fixed now)
- If stage 2 is tied and C-primary is chosen through the "larger α" tie-break, the thesis does not cite stage 2 as evidence of an effect of trust.
- Any claim about the effect of trustScore rests only on the comparison of C-primary with C-ablation on n = 384.
- Bab III states the "larger α" tie-break rule explicitly, for stage 1 and for stage 2.

### Unchanged
The stage-2 selection rule, the stage-2 candidates (α = 0.25 and α = 0), the judge, the prompts, the samples (dev and test), the 2-hop path, and conditions A, B and D.

### Justification
This decision adds a measurement; it does not change any selection. It was made before stage 2 and before any official n = 384 run, so it cannot depend on their results.
