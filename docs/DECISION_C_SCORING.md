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
- Outputs: `results/_archive/2026-10-19_pre-official/llm/c_graphrag/results/controls/ctxrel_v1_controls_20261007T154329Z.jsonl` and `_summary.json` (control data only).
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
- Outputs: `results/_archive/2026-10-19_pre-official/llm/c_graphrag/results/controls/ctxrel_v1_controls_hard_negative_20261007T164814Z.jsonl` (+ `_summary.json`) and `results/_archive/2026-10-19_pre-official/llm/c_graphrag/results/controls/ctxrel_v2_controls_positive-easy_negative-hard_negative_20261007T165007Z.jsonl` (+ `_summary.json`).
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

> Update 2026-10-14: applied to the graph at 2026-10-14T15:56:53+07:00 (code 6a42942); see Record.

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
- The v2 stage-1 files are kept unchanged as an archive: the five stage-1 retrieval files `results/_archive/2026-10-19_pre-official/llm/c_graphrag/results/condition_c_*_dev_v3_*.jsonl`, `results/_archive/2026-10-19_pre-official/llm/c_graphrag/results/reputation_check/stage1_repfix_alpha*.jsonl`, their reports, and the judge label files. The v3 files are written to a separate directory, `results/logscore_check/`.

### Record (fill in after apply)
- Decision points 1–4 and the stage-1 rule confirmed: 2026-10-14
- Code commit used for apply (pre-registration commit #1): `6a42942`. No file used by the apply (`14_apply_logscore_trust.py`, `trust_formulas.py`), the top-5 check or the stage-1 recheck changed after `6a42942`; see the note on `e539423` below.
- Apply timestamp (ISO 8601 with timezone): **2026-10-14T15:56:53+07:00** (= 2026-10-14T08:56:53+00:00), the APPLIED_AT line of log `01_data_cleaning/logs/20261014T085255Z_apply_logscore_trust.log`. Dry-runs before it: `20261013T181349Z` and `20261014T085014Z` (same numbers as the apply).
- P99_A: 46.0  |  P99_Q: 30.0  |  quantile method: numpy `linear` at 0.99
- Rows updated: Answer 725,795 (725,794 with a changed value)  |  Question 2,685,814 (2,685,812 with a changed value); all tagged `trustScoreVersion = "v3-logscore"`. Graph state before: Answer `v2-reputation` 725,795; Question untagged 2,685,814.
- Backup path: `01_data_cleaning/_kg_workspace/trust_backup_pre_amendment6_20261014T085414Z.parquet` (3,411,609 rows; not in git)
- Verify result (max |diff|): 0 for Answer and Question on read-back after writing; `--verify-applied` re-check (log `20261014T085801Z_apply_logscore_trust.log`): max |diff| = 0, 0 nodes differing, Answer 725,795 / 725,795 and Question 2,685,814 / 2,685,814 `v3-logscore`.
- Top-5 check (v2 vs v3, dev set): `results/_archive/2026-10-19_pre-official/llm/c_graphrag/results/logscore_check/logscore_top5_20261014T090618Z.md`; top-5 set changed for 4 / 50 questions (8%) at α = 0.25 and 11 / 50 (22%) at α = 0.5; α = 0 sanity check 0 / 50 (passed).
- Stage-1 recheck: `results/_archive/2026-10-19_pre-official/llm/c_graphrag/results/logscore_check/stage1_logscore_report_20261014T091601Z.md` (run 2026-10-14T09:16:01Z with `--run-judge`, at commit `e539423`); v2 reproduced the Amendment 5 stage-1 result of record exactly (50 × 5); 3 new items judged (ctxrel-v2, primary, `meta-llama/Llama-3.3-70B-Instruct`; 1 at α = 0.25, 2 at α = 0.75, 2 at α = 1, 3 unique), labels in `results/_archive/2026-10-19_pre-official/llm/c_graphrag/results/logscore_check/ctxrel_v2_logscore_primary.jsonl`; v3 `top_two` = [0.5, 0.25], keep_current = False; **`STAGE2_ALPHAS` changed: no**. The binding v3 recheck confirms the stage-2 candidates of record: **α = 0.5 and α = 0.25**. The v3 result is now the stage-1 result of record.
- Note on `e539423` ("update trustscore v3", 2026-10-14 16:14:08 +07:00): a post-apply commit, made after the apply (backup `085414Z`) and the top-5 check, and before the `--run-judge` recheck (16:16). It adds the apply logs, the top-5 report (`.md`/`.json`/`.csv`) and the five stage-1 v3 retrieval files `stage1_logscore_alpha*.jsonl`. The retrieval files were rewritten by the `--run-judge` run with byte-identical content (same git blob), so the committed versions are final. Its code changes only make the FAISS/embedding cache directory configurable through `KG_WORKSPACE_DIR` (`01_data_cleaning/11_densify_embedding_similarity.py`, `01_data_cleaning/12_validate_densification.py`, `backend/app/config.py`, `backend/app/services/engine_service.py`); none of these files is imported by the apply, the top-5 check or the recheck. These changes were already in the working tree, uncommitted, when the apply and the top-5 check ran; that is why the top-5 report records `6a42942`. From then on the cache was read from external storage (`KG_WORKSPACE_DIR`). Its five files are byte-identical (SHA-256) to the local cache used before. No secrets in the diff.
- Git commit: the commit that adds this Record (Amendment 6 commit #2).

### Results (dev set, n = 50)

Top-5 change, v2-reputation → v3-logscore (Condition C, top_k = 5; a change is not an improvement):

| α | Top-5 set changed | Mean Jaccard | Rank-1 changed | Order changed | Accepted in / out |
|---|---|---|---|---|---|
| 0.0 | 0 / 50 (0%) | 1.0000 | 0 | 0 | 0 / 0 |
| 0.25 | 4 / 50 (8%) | 0.9733 | 4 | 15 | 2 / 2 |
| 0.5 | 11 / 50 (22%) | 0.9267 | 2 | 23 | 6 / 3 |

Stage 1, v2 (Amendment 5 result of record) → v3 (ctxrel-v2 primary):

| α | % q ≥1 RELEVANT (v2 → v3) | mean score (v2 → v3) | % accepted (v2 → v3) |
|---|---|---|---|
| 0.0 | 48.0 → 48.0 | 0.3263 → 0.3263 | 37.29 → 37.29 |
| 0.25 | 48.0 → 48.0 | 0.3242 → 0.3305 | 41.95 → 41.95 |
| 0.5 | 48.0 → 48.0 | 0.3178 → 0.3157 | 52.12 → 53.39 |
| 0.75 | 46.0 → 46.0 | 0.2860 → 0.2818 | 67.80 → 66.95 |
| 1.0 | 36.0 → 34.0 | 0.2225 → 0.2225 | 74.58 → 72.88 |

Selection with the unchanged rule ("Selection procedure"; `select_stage1_candidates`, `llm/c_graphrag/ctxrel_sweep.py`), v3:
1. Primary criterion, tie margin 2 pp: best 48.0%. α = 0.0, 0.25, 0.5 (48.0) and α = 0.75 (46.0) are within 2 pp and form one tied group; α = 1.0 (34.0) is ranked after it.
2. Tie-breaker within that group, margin 0.02: best mean score 0.3305 (α = 0.25). α = 0.0 (0.3263, −0.0042) and α = 0.5 (0.3157, −0.0148) are tied with it; α = 0.75 (0.2818, −0.0487) is not and is ranked after them.
3. Larger α among ties on both: 0.5 > 0.25 > 0.0.

Ranking: 0.5, 0.25, 0.0, 0.75, 1.0 → `top_two` = [0.5, 0.25]. α = 1 does not rank first, and not every α is tied with α = 1, so keep-current does not apply. v2 gives the same ranking.

---

## Amendment 7 (2026-10-17): corpus and query-text parity of Condition B with Condition C

**Status: confirmed by the author.** Written after stage 1 and before stage 2 and the official evaluation on the test sample (n = 384). This change does not touch the test sample; the only new run is Condition B on the dev split (positions 385–419, n = 35). Earlier pilots that used some test items have been archived and are not used as the basis for any decision, including this one. Technical details are in `reports/b_corpus_parity.md`.

### Finding
- Condition B built its FAISS index from a **random sample of 8,000 answers** (`INDEX_POOL`) from the parquet. Condition C reaches **all 725,795 Answer nodes** in the Neo4j KG (`graphrag`). B therefore saw only about 1.1% of the answers C can reach.
- B's query text (`f"{Title}\n{Body}"`, full body) differed from C's (`f"{Title} {str(Body)[:1000]}"`, the same text `11_densify_embedding_similarity.py` encodes).
- As a result, differences in results between B and C could be caused by corpus size or by different query input, not by the graph structure and trust-weighting under test. This is a threat to the validity of the comparison, not a finding from hallucination results.

### Decision
1. **B's corpus = all Answer nodes of the same KG as C** (725,795 answers; every answer has exactly one parent Question via `HAS_ANSWER`/`HAS_ACCEPTED_ANSWER`; 0 orphan answers, 0 with two parents).
2. **Answer text representation identical to C**: one vector per answer, text `Q: <parent question title>\nA: <answer body>` cut to 400 cl100k_base tokens with the same truncation function as C (moved unchanged to `llm/common/text.py`). No multi-chunking.
3. **Query text identical to C**: `build_query_text(title, body) = f"{title} {str(body)[:1000]}"`, one shared function for B and C. B's query vector equals C's within a tolerance of 1e-6 (maximum measured difference 1.8e-7 over 35 dev questions).
4. **Neo4j is only an offline export source.** `build_kg_answer_index.py --stage export` reads Neo4j once (read-only) and writes `kg_answers.parquet`. At runtime, Condition B **does not access Neo4j at all**: retrieval stays pure FAISS (`IndexFlatIP`, exact search, L2-normalised all-MiniLM-L6-v2 embeddings), text is taken from the parquet. B uses no graph, relations or trust.
5. **B's leakage exclusion becomes per query**, the same as C: over-fetch, drop the question being answered together with every answer in its thread (and its accepted answer), then cut to `top_k`. Its definition is one shared function for B, C and D (`llm/common/leakage.py`), with a per-query assert: retrieval results ∩ exclusion set = ∅. The exclusion currently covers the question's own thread. Extending it to Duplicate and near-duplicate questions (and a Linked option) is decided separately in Amendment 8.
6. **Condition C's behaviour does not change.** C is only switched to call the shared functions. C's retrieval on 12 dev questions × 4 configurations is identical to the snapshot taken before the refactor (ids, scores and hashes of the context text). Trust weights, α values and the Neo4j data are not changed.
7. The old corpus is kept behind `--corpus legacy_sample`, only to reproduce the old pilot. KG-corpus runs have their own `config_hash` (`"corpus": "kg_answers_v1_tok400"`), so they are never mixed with old runs.

### Unchanged
Sampling and split, prompt (`PROMPT_VERSION` v3), generator model, `top_k = 5`, `TOKEN_CHUNK_LIMIT = 400`, embedding model, evaluation metrics, and all behaviour of Conditions A, C and D.

The LLM-as-Judge calibration batch (`calibration_generate.py`) keeps using the `legacy_sample` corpus, because the batch and its annotations were already made with that corpus. Calibration tests the reliability of the judge, not the quality of B's retrieval, so the difference in corpus does not affect its validity.

### Impact on pilot results
The old B pilot results (n = 10, **test items**, 8,000-answer corpus) and the old B dev run (n = 50, 8,000-answer corpus) were produced with a different corpus and query text. Neither **can be compared directly** with new B runs or with C; both remain recorded as `legacy_sample` corpus results. The comparison of the new vs old dev run on the same 35 dev questions is reported in `reports/b_corpus_parity.md` §7 for information only, not as a basis for selection. All official B runs (n = 384) use the KG corpus.

### Justification
This is a correction of experimental-design validity (corpus and input parity between the compared conditions), decided on the structure of the pipeline, not on hallucination results. The change was made before the test sample was touched.

---

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

---

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
Recheck run on 2026-10-18 (report `results/_archive/2026-10-19_pre-official/llm/c_graphrag/results/expansion_check/stage1_expansion_report_20261018T042953Z.md`).
- **Reproduction:** with the switch off, the top-5 is identical to the Amendment 6 stage-1 result of record (50 × 5), the recomputed metrics equal the recorded ones, and top_two stays [0.5, 0.25].
- **Expansion:** empty expansion slots fall from 64.1% to 0% (432 expansion Questions on 50 dev questions, every α). Context items from expansion in the top-5 roughly double (α = 0.5: 36 → 82; α = 0.25: 37 → 88). Questions with fewer than 5 context items fall from 5 to 2. Anchors and traversal candidates are unchanged for every question.
- **Questions whose top-5 changed:** 38 / 50 at α = 0.5 and 34 / 50 at α = 0.25.
- **Latency:** C retrieval-only p95 changes by about +0.01–0.02 s, for example α = 0.5 from 0.091 to 0.106 s.
- **Judging:** 405 cached labels were reused under the identity rule, and 119 new items were judged (ctxrel-v2, primary, `meta-llama/Llama-3.3-70B-Instruct`, temperature 0; 0 failures). The new labels are in `results/_archive/2026-10-19_pre-official/llm/c_graphrag/results/expansion_check/ctxrel_v2_expansion_primary.jsonl`. Cost was $0.0137.

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

---

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

### Record (fill in after stage 2)
- α\* (C-primary, stage-2 winner): [ ]
- α ablation (C-ablation, the other stage-2 candidate): [ ]
- Stage-2 report: [path]
- Date selected: [YYYY-MM-DD]
- Commit that fills in this Record, the Outcome section and the official_params block: [ ]

---

## Amendment 11 (2026-10-19): judge-v2.2 with retrieval context, single-annotator validation, 63-item confirmation set

**Status: confirmed by the author.** Written after Amendment 10 and before any calibration annotation, any judge-v2.x run on human-labelled data, stage 2, and the official evaluation on the test sample (n = 384). No judge-v2.1 or judge-v2.2 output exists. The calibration sheets exported on 2026-10-13 are still empty (0 of 60 and 0 of 30 rows filled), so their format can be replaced. Technical details are in `reports/p1_judge_annotation.md`.

### Motivation
- judge-v2.1 sees only the question, the accepted answer and the candidate answer. Faithfulness to the retrieved context (B–D), citation support and the match of the solution approach cannot be measured with it (`reports/evaluation_implementation_audit.md`, P1-1).
- `blind_candidate` leaves phrases such as "the retrieved sources do not cover …" in the candidate. They tell the judge and the annotator that an answer comes from a RAG condition (P1-5).
- Only one human annotator (the researcher) is available. Amendment 4's human-human κ and adjudication cannot be the main reliability evidence.

### Change 1: judge-v2.2, one instrument (replaces judge-v2.1 as the instrument to calibrate)
1. **Input:** the question (title, tags, body), the accepted answer as the reference, the blinded candidate answer, and a **retrieval context block**.
   - For B, C and D, each context item is shown under a neutral marker `[K1]` … `[K5]`, in retrieval order, never with its Stack Overflow id.
   - Each item is cut at 2,500 characters. On the existing dev runs, no item is longer than 2,366 characters (median 981).
   - For A, the block states explicitly that there is no context.
2. **Output per claim:** the verdict of judge-v2.1 (SUPPORTED / CONTRADICTED / FABRICATED / UNVERIFIABLE) and its severity, plus:
   - `supported_by_context`: yes / no, or n/a for A;
   - `cited_item_supports`: yes / no, only for a claim that carries a citation in the original answer.
3. **Output per answer:** `answer_attempted` (ABSTAIN), and `approach_match`: same / different_valid / different_unassessable.
4. **Fixed rules in code, never the judge's own label:**
   - HALU = ≥ 1 claim CONTRADICTED or FABRICATED (unchanged from Amendment 4).
   - Faithfulness (B–D) = claims with `supported_by_context` = yes ÷ all claims. It is null for A and for an answer without claims.
   - Citation support = cited claims with `cited_item_supports` = yes ÷ all cited claims. It is null when the answer has no cited claim.
   - `approach_match` never changes HALU. The rule "a different approach is not a hallucination" stays in the prompt.
5. **Citations:** before blinding, each `[SO-<id>]` in the answer is mapped to the `[Kn]` of the same context item. The judge sees `[Kn]` only in the claim list it is asked to check, never the Stack Overflow id.
6. `PROMPT_VERSION = "judge-v2.2"`, temperature 0, recorded on every record and in the manifest. judge-v2.1 stays in the repository unchanged.

### Change 2: blind-v3
`BLINDING_VERSION = "blind-v3"`.
- Citation markers are removed as in blind-v2.
- In addition, meta phrases that reveal a RAG answer ("the retrieved sources do not cover …", "based on the provided context …", "according to the sources …") are replaced by the neutral marker `[catatan: informasi tidak tersedia]` or removed, following a documented pattern list taken from the existing dev outputs.
- The judge and the annotation sheets use the same blinding.

### Change 3: judge-v2.2 is the base version; revision limit
- judge-v2.2 is the **base version before calibration**. It is not counted as a revision.
- The limit of 3 prompt revisions (Amendment 4) starts at the first run of judge-v2.2 against the human labels. At most three revisions (v2.3, v2.4, v2.5) may follow; a fourth version after v2.2 is refused by the calibration runner.
- Every version is committed before it is run and is logged in `docs/JUDGE_REVISION_LOG.md` (date, commit, reason, κ before and after).
- The runners accept only the versions on an explicit list in the code, one list for calibration and one for official runs.

### Change 4: annotation by one person, and its mitigations
- **Annotator:** the researcher labels the calibration set (60 items: A, B-plain, B-grounded, 20 each, the same items, seed and hidden key as the 2026-10-13 export).
- **Blind annotation:** condition, run, model and judge output stay hidden. The annotator sees the question, the accepted answer, the blind-v3 candidate and the `[K1]` … `[K5]` context (empty for A).
- **Evidence per label:** a row with hallucinated = yes must list at least one wrong claim with its type (contradicted / fabricated) and evidence (a documentation URL or a quote from the thread or the context). Rows without it are rejected.
- **Further columns:** `faithfulness_level` (B–D: all_supported / partly_supported / not_supported), `approach_match`, `domain_familiarity` (familiar / not_familiar), `uncertain` and `notes`, and an `annotator_id` per row.
- **Intra-annotator reliability:** 20% of the calibration items are re-annotated with new ids and a new order, at least 14 days after the first sheet is completed. Any override of the 14 days is recorded in the manifest. κ between the two passes is reported.
- **Reported agreement**, each as unweighted κ (3 categories: ABSTAIN / not hallucinated / hallucinated, and binary on items both raters label non-ABSTAIN) with a bootstrap 95% CI (2,000 resamples, fixed seed) and a confusion matrix:
  - primary judge vs researcher (this decides acceptance, target κ ≥ 0.6, as in Amendment 4);
  - secondary judge vs researcher;
  - primary vs secondary judge;
  - researcher first pass vs re-annotation;
  - a second annotator vs the researcher, if one becomes available (the code supports it without change; adjudication as in Amendment 4 then applies).
- Agreement on faithfulness (judge ratio mapped to all / partly / not vs `faithfulness_level`) and separate results for familiar and not-familiar items are reported descriptively.

### Change 5: confirmation set
- The confirmation set becomes **63 items = 7 per run × the 9 official test runs (R1–R9)**, drawn with seed 42 from the frozen test sample.
- It is exported only after the judge version is frozen (the Record below is filled in) and all nine official runs exist.
- It is annotated blind by the researcher with the same sheet format, before the judge labels of these items are viewed.
- It replaces Amendment 4's 30 items from the stage-2 runs.

### Change 6: judge bias probes (dev only, descriptive)
1. **Authority:** 40 B/C/D dev answers judged twice, once blind-v3 and once with the `[Kn]` citation labels left in; the flip rate of HALU is reported.
2. **Length:** point-biserial correlation of answer length (tokens) with HALU, per condition.
3. **Order:** 40 B/C/D dev answers judged with the `[K1]` … `[K5]` order shuffled (fixed seed); the flip rate is reported.

### Unchanged
The binary HALU rule and the metrics of Amendment 4; the acceptance target κ ≥ 0.6 for the primary judge; the judges and their models (primary Llama-3.3-70B, secondary gpt-4o-mini, temperature 0); retrieval, generators and generation prompts; stage 1, the stage-2 selection rule and Amendment 10.

### Justification
These are measurement-validity corrections, decided on the design of the instrument and on the availability of annotators, before any human label or judge-v2.x label exists. They cannot depend on agreement or hallucination results.

### Record (fill in)
- Judge version frozen: [ ]  commit: [ ]
- Revisions after v2.2 (see docs/JUDGE_REVISION_LOG.md): [ ]
- κ primary judge vs researcher, calibration (60), per version, with 95% CI: [ ]
- κ secondary judge vs researcher, calibration (60), with 95% CI: [ ]
- κ primary vs secondary judge, calibration (60), with 95% CI: [ ]
- κ intra-annotator (20% re-annotation), with 95% CI, and days between passes: [ ]
- κ second annotator vs researcher, if any: [ ]
- κ primary judge vs researcher, confirmation (63), with 95% CI: [ ]
- Date frozen: [YYYY-MM-DD]

---

## Amendment 12 (2026-10-19): LLM-as-Judge panel

Amendment 12 – LLM-as-Judge panel.
(a) The generator for the initial report remains gpt-4o-mini for all conditions.
(b) The LLM-as-Judge is a fixed panel of three judges from different model families,
served via DeepInfra: meta-llama/Llama-3.3-70B-Instruct-Turbo (served by DeepInfra in FP8
quantization), deepseek-ai/DeepSeek-V3.2 (non-thinking mode), and Qwen/Qwen2.5-72B-Instruct. gpt-4o-mini is no longer a judge, to
avoid self-preference bias toward the generator's own outputs (Verga et al., 2024).
DeepSeek and Qwen families were selected in part because recent evidence shows them to
agree substantially with expert labels on evidence-based hallucination judgment
(Zhu et al., 2026) and to be the strongest open-source judge family (Gu et al., 2026).
(c) All judges use the same frozen judge prompt version, temperature 0, explicit
max_tokens, and no reasoning/thinking mode.
(d) Aggregation: ABSTAIN if >=2 judges say ABSTAIN; otherwise the median of ordinal scores
(FAKTUAL/ABSTAIN = 0, SEBAGIAN = 1, PENUH = 2). The binary hallucination outcome equals a
2-of-3 majority vote (Gu et al., 2026: majority voting over mean). Missing-judge handling as
implemented; UNRESOLVED items are reported, never dropped.
(e) Validation: against the human calibration labels, each judge and the panel are
reported with raw agreement, Cohen's kappa, hallucination recall, unsafe pass rate and false
alarm rate (Zhu et al., 2026), with ABSTAIN excluded and counted. Acceptance is decided by the
panel's 3-category Cohen's kappa (ABSTAIN / hallucinated / not hallucinated) against the human
labels, target >= 0.6, as pre-registered in Amendment 11. Binary kappa (ABSTAIN excluded and
ABSTAIN counted), raw agreement, hallucination recall, unsafe pass rate and false alarm rate are
reported as supporting statistics and do not decide acceptance. Raw agreement is always reported alongside kappa because kappa is
sensitive to label prevalence (Çetinkaya, 2027). The panel composition is fixed before any
judge run on calibration items and is not changed based on these results. If the panel
kappa < 0.6, the existing protocol applies (up to 3 prompt iterations on the dev
calibration set, applied identically to all three judges).
(f) Human annotation is done by a single annotator; agreement therefore measures the
validity of judge labels, not inter-annotator reliability (Çetinkaya, 2027; Zhu et al., 2026).
(g) This supersedes the earlier choice of Llama as primary and gpt-4o-mini as secondary judge.
(h) blind-v3 does not mask every source-limitation phrase (e.g. 'The resources provided do not
cover…'), so some RAG answers remain recognizable to both the annotator and the judges. blind-v3
is kept unchanged so that judges and the annotator see identical input.

---

## Amendment 13 (2026-10-19): stricter acceptance for judge iterations after judge-v2.2

**Status: confirmed by the author.** Written after the first panel run against the human labels (judge-v2.2, panel-v1, run 35fab00) and BEFORE any further judge run (judge-v2.3 or later) on calibration data.

### Finding (iteration 1)
- Panel 3-category κ = 0.551 (95% CI 0.329–0.731), below the 0.6 target. The 3-category κ was dominated by ABSTAIN agreement (8/8 human ABSTAIN items matched, ABSTAIN κ = 1.0), while hallucination detection was poor: panel binary κ (ABSTAIN excluded) = 0.195, hallucination recall 2/14, unsafe pass rate 12/14.
- A judge version could therefore come close to the 3-category target while still passing most hallucinated answers as not hallucinated.

### Change: acceptance criterion for every later iteration (judge-v2.3, v2.4, v2.5)
A judge version is accepted only if BOTH hold, for the panel against the frozen human labels on the 60 calibration items:
1. the panel's unweighted 3-category Cohen's κ (ABSTAIN / hallucinated / not hallucinated) ≥ 0.6 (unchanged, Amendment 11 / Amendment 12 (e)); AND
2. the panel's binary Cohen's κ (hallucinated vs not), ABSTAIN excluded, ≥ 0.4.

Raw agreement, binary κ with ABSTAIN counted, hallucination recall, unsafe pass rate and false alarm rate remain supporting statistics, reported as before, and do not decide acceptance. If no version meets both conditions within the revision limit (Amendment 11, change 3), the existing fallback applies: freeze the version with the highest panel 3-category κ among those run, and report both κ values and the limitation.

### This makes the criterion stricter, not looser
Condition 1 is unchanged. Condition 2 is added. Every version that passes the new criterion also passes the old one, and some versions that pass the old one fail the new one. No result already obtained changes status: judge-v2.2 fails under both criteria.

### Recorded before any further judge run
This amendment is committed before any judge-v2.3 prompt is written into the repository, before any further judge call on calibration items, and before any judge run on the confirmation set or the official test sample. The panel composition (Amendment 12 (b)), the frozen human labels (annotator1_freeze.json, sha256 bff6e290…8290b) and blind-v3 are unchanged.

### Justification
The 0.4 threshold is the conventional lower bound of "moderate" agreement (Landis & Koch, 1977); it is set to require that the panel detects hallucinations better than chance by a non-trivial margin, which the 3-category κ alone does not guarantee when ABSTAIN items are easy to agree on. The threshold was chosen on the basis of iteration 1's structure (ABSTAIN-dominated κ), not tuned to a value that any future version is known to reach.

---

## Amendment 15 (2026-10-25): leakage exclusion reduced to the evaluation question and its answers

**Status: DRAFT, awaiting author confirmation.** Written BEFORE stage 2 and BEFORE any official run (R1-R9) on the test sample; no stage-2 or official run exists, `official_params.json` is still `pending_selection`. Numbering: Amendment 14 is still an untracked draft in `reports/` and is not part of this file; this amendment takes the next free number. If Amendment 14 is committed first, renumber consistently.

### Change
| | Amendment 8 (superseded for the official evaluation) | Amendment 15 |
|---|---|---|
| Mode string | `dup+neardup0.95` | `thread` |
| Excluded for evaluation question q | q, every answer of q's thread (incl. accepted), `IS_RELATED_TO` Duplicate questions (both directions) with all their answers, near-duplicate questions (cosine >= 0.95) with all their answers | q and every answer of q's thread (incl. accepted answer). Nothing else |
| Duplicate / Linked / near-duplicate questions | Duplicate and near-duplicate excluded, Linked kept | all three stay retrievable |

Amendment 8's text is not edited. This amendment only states that, for the official evaluation of B, C and D, its exclusion rule (Decision items 1 and 3, and the default in item 4) is replaced by the `thread` rule. Its finding, its audit and its metrics stay as recorded. The modes `dup`, `dup+neardup0.95` and `dup+linked+neardup0.95` remain available as sensitivity modes (`--leakage-mode`, `LEAKAGE_EXCLUDE_DUP`, `LEAKAGE_EXCLUDE_LINKED`, `LEAKAGE_NEAR_DUP_COSINE`) and in the audit.

### Rationale
TODO(author): isi alasan desain. (Not written by the assistant. It should answer Amendment 8's argument that a Duplicate is the evaluation question's ground truth in disguise, preferably as a design argument about scenario realism rather than one based on scores.)

### Known impact
- **Test sample (n = 384):** Amendment 8 counted, in aggregate only, 2 questions with a Duplicate and 1 with a near-duplicate. Under `thread` their duplicate's / near-duplicate's answers can enter the context. The frozen test list contains 10825926, 61698964 (Duplicate) and 50363191 (near-duplicate) per `results/leakage/exclusion_sets.parquet`; no retrieval has been run on them.
- **Stage 1 is not affected.** It ran with the thread-only exclusion; in the default mode 0 of 50 dev exclusion sets differed, and C's retrieval was identical for 50/50 questions at alpha 0.5 and 0.25 (Amendment 8). Stage 1 is not re-run. Stage 2 has not been run and will run under `thread`.
- **Linked questions:** unchanged (never excluded in the official mode).
- **Metric:** the ID-level self-retrieval rate must still be 0 (`assert_no_leakage` runs per query for q and its answers). The cosine >= 0.95 check between each context item and q's accepted answer is no longer guaranteed to be 0: it is MEASURED and reported, not enforced. It is never used as a filter.

### Consequences for hashes and the registry
- `config_leakage_key("thread")` now writes `"leakage": "thread"`. Only `leakage_mode=None` (the unspecified mode of every run before Amendment 8) omits the key, so the `config_hash` of all earlier runs is unchanged (asserted against the real pilot histories), while a new `thread` run hashes differently and therefore gets a different output filename. Resume/append, which keys on that filename and hash, can never mix a new run with an old one.
- The builders in B, C and D now default `leakage_mode` to `None` (legacy); the CLI, the dashboard (`LeakageSettings`) and `official_run_matrix.LEAKAGE_MODE` pass `"thread"`.
- `LEAKAGE_MODE = "thread"` is the single constant read by the run matrix (`--leakage-mode thread` for every B, C, D cell), `frozen_parameters.py`, `run_registry.expected_generation_config`, `stage2_report.py` and `trust_ablation_report.py`. Those two reports compare `config["leakage"]` with it and so now accept only `thread`; a stage-2 or ablation run made under `dup+neardup0.95` is rejected. All earlier dev runs are archived and are not stage-2 inputs.
- `results/frozen_parameters.json` and `docs/FROZEN_PARAMETERS.md` are generated files. They were NOT overwritten; regenerate them after this amendment is confirmed.

### Code and documents changed
`llm/common/leakage.py`, `llm/official_run_matrix.py`, `llm/frozen_parameters.py`, `llm/build_archive_index.py`, `llm/evaluation/official_report.py`, `llm/c_graphrag/stage2_report.py` (messages), `llm/b_rag/b_condition_b_rag.py`, `llm/c_graphrag/c_graphrag.py`, `llm/d_lightrag/d_lightrag.py` (only the `leakage_mode` default and comments; no retrieval logic), `backend/app/services/engine_service.py` (docstring), the tests `llm/tests/test_leakage_exclusion.py`, `test_official_run_matrix.py`, `test_run_registry.py`, `llm/c_graphrag/tests/test_config_hash_backward_compatibility.py`, `test_stage2_report.py`, `test_logscore_trust.py`, `llm/evaluation/tests/test_trust_ablation_report.py`, and the documents `docs/OFFICIAL_RUN_MATRIX.md` and `reports/leakage_audit.md` (note only).

### Smoke test (retrieval only, no LLM call, cost $0; dev set has no Duplicate or near-duplicate, so two unused-pool questions were used; no test-sample question was touched)
Q 35278729 (Duplicate 59533843), Q 20528668 (near-duplicate 66893076, answer 66893206), Q 19951718 (dev, Linked 44881990). Mode `thread`: exclusion set = {q} and q's thread answers only (1 question; 2, 1, 1 answers); mode `dup+neardup0.95` excluded 2 questions for the first two. Own-thread ids in the context: 0 for B, C, D in every case. For 20528668 the near-duplicate's accepted answer entered the context under `thread`: B via vector search only, C via graph traversal (1-hop `HAS_ACCEPTED_ANSWER` from anchor 66893076), D via low-level retrieval (anchor 66893076); under `dup+neardup0.95` it did not. Max cosine of a context item with q's accepted answer for that question: 0.9391 (B, C, D; context text includes the question title), versus 0.58 / 0.53 / 0.28 under `dup+neardup0.95`; items with cosine >= 0.95: 0 in all 18 cells. The Duplicate of 35278729 and the Linked question of 19951718 did not surface in any condition.

---

## Amendment 16 (2026-10-25): the four-run design; Condition D scored by one cosine; secondary metrics

**Status: DRAFT, awaiting author confirmation.** Written BEFORE stage 2 and BEFORE any official run; no stage-2 or official
run exists and `official_params.json` is still `pending_selection`. This is a note that follows the thesis design of
Bab III; it does not edit the text of any earlier amendment (history stays as written).

### What changes for the official evaluation
| | Before (Amendments 10, 11, 14 draft) | Amendment 16 |
|---|---|---|
| Official runs | nine cells R1-R9: A, B/C/D each plain and grounded, C at two alphas | **four**: A, B-grounded, C-primary-grounded, D-grounded (cells R1, R3, R5, R9) |
| Grounding + citation instructions | on and off | **on** for B, C, D (A has none) |
| Alpha | C-primary (alpha\*) and C-ablation | **one alpha, alpha\***. Stage 2 still produces the second candidate and `official_params.json` still stores it, but **no official run uses it** |
| Pre-registered comparisons | R5 vs R1, R3, R9, R7; R4 vs R2, R8, R6 | **C vs A, C vs B, C vs D** (R5 vs R1, R3, R9) |
| Legacy cells | part of the matrix | still defined (R2, R4, R6, R7, R8) so old hashes and data stay reproducible; refused on the test sample unless `--allow-legacy-cells` (API `allow_legacy_cells`) is passed; never started by the Official preset, the default listings or "Run all" |
| `C_RETRIEVAL_VERSION` default | v2 | **v3** (CLI and API) |

Amendment 10 ("trust ablation on the official evaluation") is therefore **not applied** to the official evaluation: its
C-ablation runs are legacy cells. Its stage-2 candidate rule (two candidates, 0.25 and 0) is unchanged, and so is its
statement that stage 2 alone is not evidence for an effect of trust. The thesis claims no effect of trustScore on the
basis of a trust ablation; the C vs D comparison differs in several components at once (Bab III).

### Condition D, retrieval version v4 (v3 stays reproducible)
v3 scored the low level by the cosine of the anchor and the high level by the raw weight of the edge (constants 0.6 for
TAG_COOCCUR, 0.4 for EMBED_SIM; 1.0 or a cosine for IS_RELATED_TO): two scales that cannot be compared, and an edge
type fixed the order whatever the similarity. v4 scores **both levels by the same quantity: the cosine between the
query embedding and the embedding of the question that owns the answer** (the anchor for the low level, the neighbour
question for the high level), from the cached embeddings of the FAISS index (title + first 1,000 characters,
normalised). Edge weights are neither read nor used. The high level collects the distinct neighbour questions of the
anchors over IS_RELATED_TO, TAG_COOCCUR and EMBED_SIM that have at least one non-excluded answer, **scores at most 100
of them per query** (`HIGH_LEVEL_CANDIDATE_CAP`; the natural bound is 20 TAG_COOCCUR + 10 EMBED_SIM + the question's
links per anchor; when exceeded the smallest question ids are kept, deterministic and blind to content, and the record
stores `high_level_cap_hit`), and keeps the three best (`N_HIGH_LEVEL`). Unchanged: same graph, index and leakage rule
as C; no trust; no accepted-answer bonus; no semantic expansion; dedup keeps the highest score; ties by answer id;
top-5; `N_LOW_LEVEL = N_HIGH_LEVEL = 3`. `d_retrieval_version` is part of `config_hash`; the official matrix pins v4
(`--d-retrieval-version v4`). No retrieval has been run with v4 yet: before the official run, run a retrieval-only dev
check (`--retrieval-only`, no LLM call).

### Reporting (`llm/evaluation/official_report.py`)
* Hallucination rate in two versions: (a) ABSTAIN excluded from the denominator (primary), (b) ABSTAIN counted as failed =
  (hallucinated + ABSTAIN) / all usable answers. The earlier "all answers" figure counted ABSTAIN as not hallucinated;
  that was an error against the thesis definition and is replaced by (b). Also the severity split (full / partial) and the
  share of unverifiable claims per run.
* Faithfulness and citation support (B, C, D) aggregated from the per-claim labels of the panel records, no new call.
  Denominators: faithfulness = claims with `supported_by_context = YES` / claims rated YES or NO; citation support = cited
  claims with `cited_item_supports = YES` / cited claims rated YES or NO; null values are left out, never imputed.
* Paired counts per question for C vs A, B, D (C better / equal / worse on the binary label of the panel; version (a)
  leaves out pairs with an ABSTAIN or unusable answer and counts them, version (b) counts ABSTAIN as failed) and the share
  of answers on which all three judges gave the same label. Fleiss, Krippendorff, bias probes and the
  consistent / inconclusive flag are unchanged.
* Secondary metrics, one judge (`meta-llama/Llama-3.3-70B-Instruct`, temperature 0), blinded: context relevance (B, C, D;
  ctxrel-v2) and answer relevance (A-D; question + blinded answer only). `llm/evaluation/secondary_metrics.py`
  (dry run prints the number of calls; official files are refused until Amendment 14 is approved).
* Latency unchanged: NF3 = p95 of end-to-end latency of Condition C <= 15 s; retrieval p50 / p95 for every run.

### Other
* The 63-item confirmation set of Amendment 11 (7 per run x 9 runs) belonged to the human-agreement gate that Amendment 14
  removes; the code (`annotation_tool.py`) is kept but dormant and is not used by the official evaluation.
* Densification script defaults now equal the values of the final graph (tag df <= 200, Jaccard >= 0.6, top-20; embedding
  cosine >= 0.7, top-10). The graph was not rebuilt.
* The Question-level trustScore stays in the data but is hidden in the dashboard unless a setting is turned on; retrieval
  never reads it.
* Stage 1 of record (Amendment 9 recheck) ran with leakage mode `dup+neardup0.95` (the default at the time); on the 50 dev
  questions that exclusion equals the `thread` exclusion (every record: `n_excluded_q = 1`; 0 Duplicate and 0 near-duplicate
  in `results/leakage/exclusion_sets.parquet`). See `IMPLEMENTATION_STATUS.md`.

### Record
* Confirmation by the author: [ ]
* Commit: [ ]

---

## Amendment 17 (2026-10-25): answerable anchors for C and D; D fills up to k

**Status: DRAFT, awaiting author confirmation.** Written BEFORE stage 2 and BEFORE any official run. Earlier amendments are
not edited.

### Reason
Dev-set diagnosis (50 questions, `thread`, read-only): only **28% of the entity anchors had an answer** (42 of 150; 22 of
50 questions had no answered anchor at all), so the 1-hop step contributed 1.12 answers per question; and a neighbour
question has one answer in 75% of cases. Condition D therefore returned only **3.94 context items per question** on
average (20 of 50 questions had 5). C reached 5 items in 48 of 50 questions only through its answerable-only expansion.

### Change (nothing else changes: trust, expansion in C, fusion, alpha, prompts, leakage rule)
1. **Entity anchoring, C and D** (`llm/common/anchors.py::select_answerable_anchors`, one shared function): a question can
   be a seed only if it has at least one non-excluded answer. Questions without an answer are skipped and the scan
   continues down the same FAISS list until `n_anchor` (3) answerable seeds are found, scanning at most **400** candidates.
   If fewer are found, the ones found are used and `anchors_found` / `anchors_requested` are recorded. The evaluation
   question (and its answers) stays excluded as before.
2. **Condition D high level**: from the neighbours (which already need an answer), take the top 3 by cosine to the query,
   then keep adding neighbours in score order until at least **k = 5 distinct answers** exist (answers already found by the
   low level count), or no neighbours are left; then keep the top 5 answers by score. The 100-candidate cap stays.
3. **Version labels**: **C retrieval `v4`** (= v3 ranking + answerable anchors) and **D retrieval `v5`** (= v4 scoring +
   answerable anchors + fill up to k). v3 (C) and v3 / v4 (D) stay reproducible with `--c-retrieval-version v3` and
   `--d-retrieval-version v3|v4`. The versions are part of `config_hash`, pinned in the run matrix and in the frozen
   parameters; `official_params.json` now carries `c_retrieval_version: "v4"`.

### Consequence for stage 1
The stage-1 result of record (Amendment 9 recheck) was made with plain anchors (C v3). Because C's retrieval changes, stage 1
is repeated with v4 anchors, the same judge (ctxrel-v2, Llama-3.3-70B-Instruct, temperature 0) and the same label cache;
only new items are judged. The rule is unchanged and the result is accepted whatever it is. Report:
`llm/c_graphrag/results/anchors_check/`.

**Result (2026-10-25, dev set, `thread`, 254 new items judged, $0.03):** % of questions with at least one RELEVANT item
and mean score per alpha: 0: 56.0 / 0.376; 0.25: 56.0 / 0.374; 0.5: 58.0 / 0.372; 0.75: 56.0 / 0.332; 1: 40.0 / 0.256.
Under the unchanged rule the two best alphas are **0.5 and 0.25** (previously 0.25 and 0). They are the stage-2
candidates of record (`llm/c_graphrag/stage2_candidates.py` reads them from `stage1_anchors_report_20261025T165734Z.json`).
Alpha 0 is run in stage 2 as a pre-declared exploratory reference, not eligible for selection.

### Record
* Confirmation by the author: [ ]
* Commit: [ ]

---

## Outcome (filled in after the procedure)
[Stage 1 table, stage 2 table, chosen α, date]

## Machine-readable official parameters

The block below is the single source of truth the dashboard's "Official (n=384)" preset reads to lock Condition C's parameters (`llm/c_graphrag/official_params.json` is kept byte-identical to this block by `llm/c_graphrag/tests/test_official_params_sync.py`). While `status` is not `selected`, the dashboard refuses to launch any Official run and `c_graphrag.py` refuses every `--sample-split test` run (`llm/c_graphrag/official_params.py`). Fill in its values (and `official_params.json`) only in the same commit that fills in the Outcome section and the Amendment 10 Record, never before.

Schema (Amendment 10, 2026-10-18; values unchanged, still pending): `alpha` is replaced by `alpha_primary` (C-primary, α\*) and `alpha_ablation` (C-ablation, the other stage-2 candidate); `c_retrieval_version` is `"v3"`, since both stage-2 candidates are v3 values and the stage-2 keep-current rule (α = 1) cannot apply to them; `selection_record` points to the record that fills in the values; `status` becomes `selected` when they are filled in. The official run matrix built from this block is `docs/OFFICIAL_RUN_MATRIX.md`.

```json official_params
{
  "status": "pending_selection",
  "c_retrieval_version": "v4",
  "alpha_primary": null,
  "alpha_ablation": null,
  "selection_record": {"path": "docs/DECISION_C_SCORING.md#amendment-10-2026-10-18-trust-ablation-on-the-official-evaluation", "commit": null},
  "sample_split": "test",
  "max_hops": 2,
  "edge_types": ["EMBED_SIM", "HAS_ACCEPTED_ANSWER", "HAS_ANSWER", "IS_RELATED_TO", "TAG_COOCCUR"],
  "use_author_trust": false,
  "accepted_only": false
}
```