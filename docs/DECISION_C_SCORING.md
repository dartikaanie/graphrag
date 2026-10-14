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

**Note (2026-10-13, Amendment 5):** since the graph update, every Answer's `trustScore` already contains reputation (`0.2 × author_weight`). Turning `use_author_trust` on therefore adds reputation a second time. The switch stays exploratory and off by default; official and stage-2 runs never use it.

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

### Record

- Code commit used for the control runs: control code at `1c1b99e3a377bc5f7a8ad1d780b54061a22eb70b`; the runs execute at a later HEAD that differs only by untracking `.DS_Store` (`68755f8`) and this decision-record update. The run outputs record the exact HEAD and a clean tree.
- Control runs executed at HEAD `1138eb8096f1` with `git_tree_clean = true` (as recorded in the control summary JSONs). The paid stage-1 sweep references the ctxrel-v2 control summary as its gate (`--controls-summary`) instead of re-running the controls; the gate checks version, all three control sets, pass status, thresholds, the 50 dev questions, and a clean-tree commit in HEAD's history (gate code at `ce04d23`).
- Hard negatives (dry build, 2026-10-07): donor pool 1,087 questions (1,471 candidates minus 384 test-sample questions; all have cached vectors). 47 of 50 dev questions had an eligible donor; no eligible donor for 61453942 (teiid), 4316334 (blackberry), 50814048 (hyperledger-fabric). Test-sample donors: 0; 4 donors are other dev questions. Donor cosine similarity median 0.457 (range 0.009-0.801, IQR 0.381-0.571); donors sharing ≥2 tags: 5; truncated items: 14 of 47. Low-similarity donors (when few eligible donors exist) behave like easy negatives; this is reported as a limitation.
- ctxrel-v1 hard negatives (primary / secondary): NOT RELEVANT 100% / 100% (R/P/I 0 / 14.9 / 85.1 and 0 / 6.4 / 93.6; n = 47). Comparison only; ctxrel-v1 had already failed on positives.
- ctxrel-v2 (primary / secondary):
  - positives RELEVANT 94% / 96% (R/P/I 94 / 6 / 0 and 96 / 4 / 0; n = 50);
  - easy negatives IRRELEVANT 100% / 100% (n = 50);
  - hard negatives NOT RELEVANT 100% / 97.9% (R/P/I 0 / 17.0 / 83.0 and 2.1 / 19.1 / 78.7; n = 47).
  - Pass: **yes** (primary decides: 94% ≥ 90%, 100% ≥ 90%, 100% ≥ 80%).
- ctxrel-v2 positives not judged RELEVANT by primary (all PARTIAL): q45387122 (the accepted answer is a dependency-version tweak), q12278877 (a link-out to Flex 3 vs Flex 4 differences), q53432433 (background on environment configuration). Three positives missed by v1 are RELEVANT under v2: q26179020, q4316334, q50814048.
- Hard negatives judged RELEVANT: none by primary (v1 or v2). Secondary (v2) judged one RELEVANT: q57331553 → donor q46370607 ("Didn't find class android.support.v7.widget.CardView"), cosine 0.801, the most similar pair in the set. Primary PARTIAL hard negatives all had donor similarity 0.298-0.687.
- Outputs: `llm/c_graphrag/results/controls/ctxrel_v1_controls_hard_negative_20261007T164814Z.jsonl` (+ `_summary.json`) and `llm/c_graphrag/results/controls/ctxrel_v2_controls_positive-easy_negative-hard_negative_20261007T165007Z.jsonl` (+ `_summary.json`).
- Context-relevance judge version used for stage 1: **ctxrel-v2**
- Date: 2026-10-07

---

## Amendment 4 (2026-10-09): judge-v2 revision, binary primary metric, human calibration

Written after the pilot judge diagnostics (`docs/JUDGE_V2_LENIENCY_REVIEW.md`, `docs/JUDGE_V2_DIAGNOSIS.md`) and BEFORE any dev-set generation for judge calibration, any judge-v2.x run on dev data, and any stage-2 run. Stage 1 (ctxrel-v2) is not affected.

### Why
Pilot diagnostics (test positions 1-10, 9 runs, n = 90 items per judge; used for diagnosis only, not for tuning):
- judge-v2 never produced HALUSINASI_SEBAGIAN (0/90 for both judges). `derive_label_v2` matches the documented rule; the judges almost never output CONTRADICTED/FABRICATED + MINOR (primary 1 claim, secondary 0). judge-v1 produced SEBAGIAN on the same generation outputs.
- Primary-secondary agreement after the parser fix (`parse_version = pv1`): κ = 0.243 (judge-v1: weighted 0.327, unweighted 0.246).
- Judge errors in both directions: qualifiers dropped during claim extraction (e.g. "in UWP"), meta-statements ("the sources do not cover X") extracted as claims (one producing a false HALUSINASI_PENUH), severity assigned to SUPPORTED claims (33, primary), code-validity claims marked SUPPORTED, and a wrong version number marked SUPPORTED by almost every judge/version combination.

### Primary hallucination metric (replaces the 3-class primary metric)
- **Primary:** binary hallucination rate = share of answers with ≥1 claim judged CONTRADICTED or FABRICATED (CORE or MINOR). Reported with ABSTAIN excluded and with ABSTAIN counted, as before.
- **Secondary (descriptive):** severity distribution (HALUSINASI_PENUH / HALUSINASI_SEBAGIAN), ABSTAIN rate, and unverifiable rate (share of claims judged UNVERIFIABLE).
- The stage-2 secondary criterion in "Selection procedure" uses the binary rate.

### judge-v2.1 prompt changes (judge-v2 stays available for reproducibility)
1. Extracted claims keep the qualifiers that make them checkable (platform, version, library, environment, conditions stated in the question or answer).
2. Statements about the sources/context themselves ("the sources do not cover X") are not claims and are not extracted.
3. Severity is assigned only to CONTRADICTED or FABRICATED claims; SUPPORTED and UNVERIFIABLE claims have no severity.
4. CORE vs MINOR with criteria and examples: CORE = the error affects the main solution or the direct answer to the question; MINOR = an error in a secondary detail (e.g. a wrong version number, a side remark) that does not change the solution.
5. Claims that code is valid, compiles, or works are UNVERIFIABLE unless the reference contains the same code or states it explicitly.
6. UNVERIFIABLE claims do not make an answer hallucinated; they are reported via the unverifiable rate.
7. `max_tokens` and `temperature` are set explicitly and recorded on every record.

### Calibration set (independent of α)
- Questions: dev set (positions 385-434, n = 50).
- Conditions: A, B-plain, B-grounded (none depend on C's α); same generation model (`gpt-4o-mini`) and generation prompt as the pilot. 150 answers.
- Annotation sample: 60 items, stratified 20 per condition, drawn with seed 42. Annotator 1 (Dartika Anie Marian) labels all 60; annotator 2 labels a fixed 30-item subset (10 per condition, seed 42).

### Confirmation set (not used for tuning)
- 30 items drawn with seed 42 from the stage-2 outputs (C runs), stratified across the stage-2 runs. Annotated by annotator 1 after judge-v2.x is frozen and BEFORE judge labels for these items are viewed. The judge-vs-human κ on this set is the reported out-of-sample estimate.

### Annotation protocol
- A written annotation guideline (`docs/ANNOTATION_GUIDELINE.md`), using the same definitions as the judge-v2.1 prompt, is committed before annotation starts.
- Blind: items shuffled; condition, run label, and judge output hidden. Annotators see the question, the accepted answer (reference), and the candidate answer only.
- Labels per item: ABSTAIN yes/no; if not ABSTAIN, hallucinated yes/no; if hallucinated, severity (PENUH / SEBAGIAN).
- Calibration annotation is completed before any judge-v2.x run on the calibration set.
- Annotator disagreements on the 30 overlapping items are resolved by discussion; the adjudicated label is the reference. Human-human κ is reported before adjudication.

### Acceptance and iteration
- Agreement measured on 3 categories (ABSTAIN / not hallucinated / hallucinated), unweighted Cohen's κ, primary judge vs human reference on the 60 calibration items.
- Target: κ ≥ 0.6. At most 3 prompt versions (v2.1, v2.2, v2.3); each version is committed before it is run.
- If no version reaches 0.6: freeze the version with the highest κ, and report κ and the limitation in Bab V.
- Secondary-judge κ and human-human κ are reported alongside and do not decide.

### Freeze
After acceptance: freeze the judge version, model, temperature, and max_tokens; "judge-v2" in the Freeze section above means this frozen version. Then run stage 2. No judge changes after the confirmation set is annotated.

### Addendum (2026-10-09, before any calibration generation or annotation)
- Judging standard (judge and human annotators): when the reference addresses a claim,
  the reference decides. A claim the reference does not address is UNVERIFIABLE, unless
  it is clearly fabricated or contradicts well-established fact. `reference_conflict`
  remains available for an outdated reference. Human annotators may consult official
  documentation only to check whether a named API, function, or option exists.
- Severity handling in v2.1: severity on SUPPORTED/UNVERIFIABLE claims is set to null and
  counted as a normalization; an error claim without severity keeps its record, counts
  for the binary label, and is reported as "severity unspecified" (not imputed).
- The annotation key is kept outside the repository until adjudication is complete; its
  SHA-256 hash is committed when the annotation files are created.
- Reported alongside the acceptance κ: binary κ (hallucinated vs not) on items that both
  the human reference and the judge label as non-ABSTAIN, with n.

### Record (fill in)
- Judge version frozen: [ ]  commit: [ ]
- κ human-human (30 items, pre-adjudication): [ ]
- κ primary judge vs human, calibration (60), per version: [ ]
- κ secondary judge vs human, calibration (60): [ ]
- κ primary judge vs human, confirmation (30): [ ]
- Date frozen: [YYYY-MM-DD]

---

## Amendment 5 (2026-10-10): contributor reputation in trustScore

Written after the exploratory reputation check and BEFORE any judging of reputation-fixed contexts, any stage-2 run, and any change to the graph.

### Finding
- `7_build_knowledge_graph.py` defines `trustScore_A = 0.5*norm(score) + 0.3*accepted + 0.2*norm(reputation)`, but the KG build (log `20260829T075916Z`) did not find `FilteredUsers.csv`, so reputation was empty for every answer. `minmax_normalize` of an all-zero series returns 0.5, so the reputation term is the constant 0.1 for every answer (log: `trustScore Answer -- min=0.100`). `9_load_author_trust.py` later wrote 293,078 `User` nodes and 715,604 `AUTHOR_TRUST` edges but did not recompute `trustScore`. Reputation has therefore never affected ranking, including in stage 1 (run at `43b3b3b`).
- Exploratory check (`reputation_ablation_check.py`, dev set n = 50, no API calls, not used for selection): with the fix below, the top-5 set changes for 0 / 4 / 6 / 10 / 40% of questions at α = 0 / 0.25 / 0.5 / 0.75 / 1. AUTHOR_TRUST covers 99.2% of candidates.

### Fix under evaluation (`repfix`)
`trust_A' = trust_A − 0.2 × 0.5 + 0.2 × author_weight`, where `author_weight` is the `AUTHOR_TRUST` edge weight (`log1p(rep) / log1p(rep_P99)`, clipped to [0, 1]; 0 when the answer has no `AUTHOR_TRUST` edge). Log-scaled weights are used instead of min-max on raw reputation because a few users with very high reputation would compress almost everyone else to ≈ 0. Everything else in C retrieval v3 is unchanged.

### Procedure (`llm/c_graphrag/reputation_stage1_recheck.py`)
1. Same dev set, retrieval, and `fuse_and_rank` v3 as stage 1. The `base` top-5 must reproduce the committed stage-1 retrieval files exactly, and the recomputed `base` metrics must equal the stage-1 report; otherwise the procedure stops.
2. Context items of `repfix` that stage 1 did not judge are judged with ctxrel-v2, primary judge, same judge input. Items already judged reuse their stage-1 labels.
3. Stage-1 metrics and the selection helper are computed for `repfix` with the same code.

### Decision rule (fixed before judging)
Tie margins as in stage 1: ≤ 2 percentage points (1 question) on the primary criterion, ≤ 0.02 on the mean relevance score.
- **R1.** At α = 0.25 and α = 0.5 (the current stage-2 candidates), `repfix` is not worse than `base` beyond the tie margin on the primary criterion AND on the mean relevance score.
- If R1 holds: **include reputation.** `trustScore` of Answer nodes is updated in Neo4j with the formula above (and `authorReputation` filled) in a separate, logged step; the `repfix` stage-1 result becomes the stage-1 result of record, and its `top_two` are the stage-2 candidates.
- If R1 fails: **do not include reputation.** Stage 1 as run at `43b3b3b` stands; reputation is reported as an exploratory finding, and Bab I–III describe trust as upvote score plus accepted status.

### Record
- Recheck run: 2026-10-10T02:26:34Z at commit `6b0b40c7da280be362c7193e8be3daf9aebdc4a0`. `base` reproduced the stage-1 retrieval files exactly (50 questions × 5 α), and the recomputed `base` metrics equalled the stage-1 report.
- New judged items: 18 (ctxrel-v2, primary); all other items reused stage-1 labels.

| α | % q ≥1 RELEVANT (base → repfix) | mean score (base → repfix) | % accepted (base → repfix) |
|---|---|---|---|
| 0.0 | 48.0 → 48.0 | 0.3263 → 0.3263 | 37.29 → 37.29 |
| 0.25 | 48.0 → 48.0 | 0.3284 → 0.3242 | 41.95 → 41.95 |
| 0.5 | 48.0 → 48.0 | 0.3114 → 0.3178 | 52.54 → 52.12 |
| 0.75 | 46.0 → 46.0 | 0.2860 → 0.2860 | 68.22 → 67.80 |
| 1.0 | 34.0 → 36.0 | 0.2288 → 0.2225 | 74.58 → 74.58 |

- R1: α = 0.25: Δ% q = 0.0, Δ score = −0.0042 (pass); α = 0.5: Δ% q = 0.0, Δ score = +0.0064 (pass).
- **Decision: include reputation.** The `repfix` stage-1 result is the stage-1 result of record.
- Selection helper (`repfix`): top_two = [0.5, 0.25], keep_current = False, identical to the original stage 1. **Stage-2 candidates of record: α = 0.5 and α = 0.25.**
- Interpretation: on the dev set, reputation is neutral for context relevance at the stage-2 candidate α values; it is included so that the trust score uses all three community signals stated in the design (G2), not because it improves relevance.

### Applying the decision
1. `01_data_cleaning/13_apply_reputation_trust.py --apply` updates `trustScore` and `authorReputation` of every Answer node with the formula above (backup of the old values written first, verified after writing, every node tagged `trustScoreVersion = "v2-reputation"`). `tests/test_apply_reputation_trust.py` pins the update formula to the evaluated `repfix` formula.
2. `reputation_stage1_recheck.py --verify-applied` must then confirm that retrieval from the updated graph reproduces the `repfix` top-5 exactly (50 × 5).
3. `7_build_knowledge_graph.py` now computes the reputation term from the AUTHOR_TRUST weight, so a rebuild uses the same definition.
4. The graph state is not part of `config_hash`; runs are attributed to the updated graph by date and by the `trustScoreVersion` tag. Every stage-2 and official run happens after step 2.

- DB update applied: 2026-10-13, 20:04:20–20:08:03 WIB (13:04:20–13:08:03 UTC), log `01_data_cleaning/logs/20261013T130420Z_apply_reputation_trust.log`. 725,795 Answer nodes read (715,604 with an `AUTHOR_TRUST` edge; 10,191 without, w = 0); precondition check OK (`authorReputation != 0`: 0 nodes; `trustScore` min 0.1001). Backup: `01_data_cleaning/_kg_workspace/answer_trust_backup_pre_amendment5.parquet`. `trustScore` before: min 0.1001, max 0.8916, mean 0.2144; after: min 0.0017, max 0.9840, mean 0.2611. Read-back verification: max difference 0.00e+00; 725,795 / 725,795 tagged `v2-reputation`. (Dry-run just before: log `20261013T130302Z_apply_reputation_trust.log`, same numbers.)
- Verify-applied: no saved output of the `reputation_stage1_recheck.py --verify-applied` run was found in the repository. The same property was re-checked on 2026-10-14 (WIB) by `logscore_stage1_recheck.py` (no judge calls): ranking from the stored `v2-reputation` values reproduces the `repfix` top-5 exactly (50 questions × 5 α).

---

## Amendment 6 (2026-10-14): log-normalised upvote score in trustScore

Written after the Amendment 5 graph update and BEFORE any stage-2 run, any run on the test sample (n = 384), and any change to the graph for this amendment. Stage 1 is done; stage 2 has not started.

**Status: all four decision points and the stage-1 rule below were confirmed by the author on 2026-10-14. Not yet applied to the graph.**

### Finding
- Answer `score` ranges from −26 to 6,095 (n = 725,795; median 1; P99 = 46; 11,925 negative; 249,542 equal to 0). `trustScore_A` normalises it with a global min-max, so the range is set by a few extreme answers: `minmax(score_A)` has median 0.0044 and P90 0.0052, i.e. for 90% of answers the upvote term contributes less than 0.5 × 0.0053 ≈ 0.003 to `trustScore_A`. The upvote signal is effectively absent from the trust score.
- Current `trustScore_A` (`v2-reputation`): min 0.0017, median 0.189, P90 0.495, max 0.984.
- Question `score` has the same shape (−42 to 11,651; n = 2,685,814; median 1; P99 = 30; `minmax(score_Q)` P90 0.0039).
- This is inconsistent with the rationale already recorded in Amendment 5: reputation is log-scaled *because* a few very high values would compress everyone else to ≈ 0. Vote score has the same skew but was not treated the same way.

### Formulas

Old (`7_build_knowledge_graph.py`, with `w_auth` from Amendment 5):

```
trust_Q = 0.7 · minmax(score_Q) + 0.3 · minmax(log1p(views))
trust_A = 0.5 · minmax(score_A) + 0.3 · is_accepted + 0.2 · w_auth
w_auth  = clip(log1p(rep) / log1p(P99_rep), 0, 1)
```

New:

```
norm_log(score; P99) = clip(log1p(max(score, 0)) / log1p(P99), 0, 1)
trust_Q = 0.7 · norm_log(score_Q; P99_Q) + 0.3 · minmax(log1p(views))
trust_A = 0.5 · norm_log(score_A; P99_A) + 0.3 · is_accepted + 0.2 · w_auth
trustScoreVersion = "v3-logscore"
```

### Decision points (all confirmed 2026-10-14)
1. **Formula** *(confirmed 2026-10-14)*. `norm_log` as above. Negative scores give 0; scores at or above P99 give 1.
2. **P99 population** *(confirmed 2026-10-14)*. P99 is computed on the raw `score` (negatives included) over the full population of the same node type: all Answer nodes for `trust_A`, all Question nodes for `trust_Q`, separately. Quantile method: numpy `linear` at 0.99 (the numpy/pandas default; the same definition as DuckDB `quantile_cont`), in the build and in the update. A P99 ≤ 0 is an error. Dry-run values: **P99_A = 46.0, P99_Q = 30.0**.
3. **Weights unchanged** *(confirmed 2026-10-14)*. Answer 0.5 / 0.3 / 0.2; Question 0.7 / 0.3; `w_auth` unchanged. The `views` term of `trust_Q`, `minmax(log1p(views))`, is **unchanged**: only the score term changes.
4. **`trust_Q` also changes** *(confirmed 2026-10-14: Answer and Question nodes are both updated)*, for consistency, although Question `trustScore` is not used in any ranking. Code search (2026-10-14): Conditions A, B, C, D and the evaluation scripts never read Question `trustScore`; C reads only `a.trustScore` (`c_graphrag.py:511`, `:536`, `:623`). The dashboard **displays** Question `trustScore` as a node property (`backend/app/services/graph_service.py`, `frontend/src/components/GraphView.tsx`, `frontend/src/pages/QuestionDetailPage.tsx`); it does not sort, filter, or compute with it. This display-only use is accepted: after the update the dashboard shows the v3 values.

### Justification
This is a measurement-validity correction based on the distribution of the score in the data, not on hallucination results. It is made before stage 2 and before the test sample has been touched.

Order of events: the formula, the P99 population, the unchanged weights, the inclusion of `trust_Q`, and this justification were fixed in writing (`docs/agent_prompt_c_amendment6_logscore.md`, 2026-10-14 01:05 WIB) from the score distribution alone, before the dry-run of the new values and before any dev-set impact number was produced. The dev-set impact seen afterwards was small: in the stage-1 recheck plan run (no judge calls), the v3 top-5 contain 0 / 1 / 0 / 2 / 2 context items without a cached ctxrel-v2 label at α = 0 / 0.25 / 0.5 / 0.75 / 1, out of 236 selected items per α (3 unique items to judge).

### Scope
- Changes: `trustScore` and `trustScoreVersion` of Answer and Question nodes.
- Unchanged: `HAS_ANSWER` edge weights (per-thread min-max, fallback 0.5) and every other edge weight, `combined_trust = 0.7 × path trust + 0.3 × answer trust`, the final ranking formula, the α candidates, the stage-1 decision rule, `w_auth`, and `authorReputation`.
- All trust formulas live in one module, `01_data_cleaning/trust_formulas.py`, imported by the KG build and by the update script, so a rebuild and the update give identical values for the same inputs.

### Expected effect (dry-run, 2026-10-14, no writes)

| Node | Version | min | median | P90 | max | nodes at 0 | nodes at 1 |
|---|---|---|---|---|---|---|---|
| Answer | v2-reputation | 0.0017 | 0.1888 | 0.4952 | 0.9840 | 0 | 0 |
| Answer | v3-logscore | 0.0000 | 0.3161 | 0.6658 | 1.0000 | 3,528 | 1,434 |
| Question | (untagged) | 0.0025 | 0.1022 | 0.1467 | 0.9853 | 0 | 0 |
| Question | v3-logscore | 0.0000 | 0.2028 | 0.4761 | 1.0000 | 1 | 1 |

Nodes whose `trustScore` changes: 725,794 of 725,795 Answer; 2,685,812 of 2,685,814 Question. The score term itself (`norm_log`) has median 0.180 and P90 0.505 for Answer (was 0.0044 and 0.0052).

### Procedure
1. `01_data_cleaning/14_apply_logscore_trust.py` (dry-run by default). `--apply` checks the graph state and free disk space, backs up `trustScore` + `trustScoreVersion` of every node to a timestamped parquet, writes the new values, and requires max |diff| = 0 on read-back. `--restore --backup <file>` undoes it; `--verify-applied` re-checks without writing.
2. `llm/c_graphrag/logscore_top5_check.py` (read-only, no API calls): top-5 under v2 vs v3 on the dev set at α = 0.25 and 0.5; α = 0 must give 0 changes.
3. `llm/c_graphrag/logscore_stage1_recheck.py`: stage 1 again with v3 for the five α values. The v2 ranking must reproduce the stage-1 result of record (Amendment 5) exactly. Cached ctxrel-v2 primary labels are reused; only new items are judged, and only with `--run-judge`. The stage-1 selection rule is applied unchanged.
4. `llm/c_graphrag/stage2_report.py` accepts only stage-2 runs that started after the update (`TRUST_V3_APPLIED_AT`, compared timezone-aware) and whose manifest records `v3-logscore`; it refuses to produce a report while `TRUST_V3_APPLIED_AT` is empty.

### Stage-1 rule (fixed 2026-10-14, BEFORE any judging of v3 items)
- The v3 stage-1 recheck is **binding**: after it, the v3 result is the stage-1 result of record, replacing the Amendment 5 (`repfix`) result.
- It is computed with the unchanged stage-1 procedure: same dev set, same ctxrel-v2 primary judge, same metrics code, same selection helper and tie margins ("Selection procedure"), including the keep-current rule.
- If the v3 `top_two` equal {0.5, 0.25}, the stage-2 candidates stay α = 0.5 and α = 0.25.
- If the v3 `top_two` differ, the stage-2 candidates become the v3 `top_two`, and `STAGE2_ALPHAS` in `stage2_report.py` is updated to them before any stage-2 run. Until the recheck is done, `STAGE2_ALPHAS` stays (0.5, 0.25).
- No other outcome of the recheck changes Amendment 6: it is not a test of whether to apply the correction.

### Run attribution and archive
- Condition C runs record the graph's `trustScoreVersion` in their manifest (`graph.trustScoreVersion`, read from the Answer nodes at run start; CLI and dashboard runs). `stage2_report.py` rejects a run whose manifest does not record `v3-logscore`, in addition to the start-time check.
- A KG rebuild writes `trustScoreVersion = "v3-logscore"` on Question and Answer nodes; `9_load_author_trust.py` takes `w_auth` from `trust_formulas.py`.
- The v2 stage-1 files are kept unchanged as an archive: the five stage-1 retrieval files `llm/c_graphrag/results/condition_c_*_dev_v3_*.jsonl`, `results/reputation_check/stage1_repfix_alpha*.jsonl`, their reports, and the judge label files. The v3 files are written to a separate directory, `results/logscore_check/`.

### Record (fill in after apply)
- Decision points 1–4 and the stage-1 rule confirmed: 2026-10-14
- Code commit used for apply (pre-registration commit #1): [ ]
- Apply timestamp (ISO 8601 with timezone): [ ]
- P99_A: [ ]  |  P99_Q: [ ]  |  quantile method: [ ]
- Rows updated: Answer [ ]  |  Question [ ]
- Backup path: [ ]
- Verify result (max |diff|): [ ]
- Top-5 check (v2 vs v3, dev set): [report path, % questions changed at α = 0.25 / 0.5]
- Stage-1 recheck: [report path, new judged items, v3 `top_two`, `STAGE2_ALPHAS` changed yes/no]
- Git commit: [ ]

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