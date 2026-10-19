Hand-written notes from reading the judge reasoning and claims of the category-C items (main run judge-v2.3; items
that are C only in judge-v2.2 are marked). They describe where the judges diverge; they do not say which judge or
which human label is right. Reviewer decisions belong in the review sheet.

Overall shape (judge-v2.3): in 7 of the 8 C items deepseek is the judge that says HALLUCINATED, alone (5 items) or
with llama (2 items); qwen labels no item HALLUCINATED in this run. In judge-v2.2 the same shape holds, with one
item (CAL-057) where qwen joins deepseek.

1. **Different version numbers / different fix than the reference.** CAL-016, CAL-017 and CAL-058 (v2.2) are
   three candidates for the same Android/Firebase dependency question. The candidate proposes other library
   versions than the accepted answer (e.g. constraint-layout 1.1.0 / 1.1.3 vs 1.0.1, Firebase 11.4.2 vs
   11.0.1). deepseek marks these version recommendations CONTRADICTED by the reference; llama and qwen describe
   them as a different but valid approach. Human label: NOT_HALLUCINATED in all three.

2. **Alternative solution whose mechanism differs from the reference.** CAL-018 (match_parent instead of a
   NestedScrollView with layout_behavior), CAL-021 (cast to StepExecutionListener instead of ChunkListener),
   CAL-024 (expect a Supertest promise to reject with the backend error instead of checking the HTTP response).
   The disagreement is again "contradicts the reference" vs "different approach", but here the human labelled
   HALLUCINATED (PENUH) and wrote a mechanism-level reason. llama splits: HALLUCINATED on CAL-018 and CAL-024,
   NOT_HALLUCINATED on CAL-021; qwen calls all three "different but valid". Patterns 1 and 2 have the same
   judge-side disagreement and opposite human labels.

3. **Time-dependent facts versus the reference date.** CAL-034 (Java 11 on AWS Lambda; the 2019 reference says
   only Java 8 is supported) and CAL-049 (v2.2; the reference describes Mesa as limited for OpenGL 4). deepseek
   follows the reference statement literally; qwen states on CAL-034 that the candidate is "correct, despite the
   outdated reference"; llama treats the claims as unverifiable. Human: HALLUCINATED (PENUH) on CAL-034 (the
   time_rule_check item), NOT_HALLUCINATED on CAL-049, so the two items differ in which side the human label
   is on.

4. **One wrong detail inside an otherwise aligned answer.** CAL-057 (`brew install python@3.3`, a versioned
   formula next to the accepted `brew install python3`) and CAL-023 (v2.2; "there is no direct method to load an
   OGR source into GeoPandas" vs the accepted answer's `gpd.read_file` with the OGR_GRASS driver). The human
   labelled both HALLUCINATED (SEBAGIAN). The judges that flag them label the claim CORE, so their label is PENUH;
   the judges that do not flag them summarise the answer as aligned with the reference. This is where the missing
   SEBAGIAN level on the judge side becomes visible.

5. **Where "attempted" ends and ABSTAIN begins.** CAL-050: the candidate says the context does not cover the
   problem and calls the user's own Redis-SET idea "reasonable". deepseek reads this as no technical claim
   (ABSTAIN); llama and qwen read it as an attempted answer that matches the reference's approach. Human:
   NOT_HALLUCINATED.
