"""
Process & Parameters registry: the pipeline steps and, per parameter, a
POINTER to where its value lives (`Src`, resolved at request time by
sources.resolve) and pointers to where its rationale / specification is
written (`Quote`, extracted verbatim by sources.quote). No parameter value
is typed here.

Fields of `Param`:
  rationale  WHY the value was chosen, quoted from docs/ or code comments.
             None -> the page shows "Rationale not documented" (Rule 3);
             never fill this with new text, only with a pointer to an
             existing sentence.
  spec       WHERE the rule/value is laid down (usually the decision
             record). A spec is not a rationale.
  set_by     decision-timeline id ("decision", "amendment-N") or None.
  status     frozen | pending | candidate | exploratory | observed, or
             "judge" (= frozen once the Amendment 4 judge is frozen,
             otherwise pending; resolved in service.py).
  kind       "parameter" (a choice) or "result" (an observed number; no
             rationale expected).
  note       factual pointer for the reader (e.g. a known inconsistency in
             the repo), never a rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.process.sources import Src

DEC = "docs/DECISION_C_SCORING.md"
README = "docs/README.md"
A = "llm/a_pure_llm/a_baseline_replication.py"
B = "llm/b_rag/b_condition_b_rag.py"
C = "llm/c_graphrag/c_graphrag.py"
D = "llm/d_lightrag/d_lightrag.py"
PROMPTS = "llm/prompts.py"
SPLIT = "llm/sample_split.py"
ENGINE = "backend/app/services/engine_service.py"
POSTLINKS = "01_data_cleaning/6_postlinks_to_sord.py"
KG7 = "01_data_cleaning/7_build_knowledge_graph.py"
AUTHOR9 = "01_data_cleaning/9_load_author_trust.py"
TAG10 = "01_data_cleaning/10_densify_tag_cooccurrence.py"
EMB11 = "01_data_cleaning/11_densify_embedding_similarity.py"
VAL12 = "01_data_cleaning/12_validate_densification.py"
REP13 = "01_data_cleaning/13_apply_reputation_trust.py"
LOG14 = "01_data_cleaning/14_apply_logscore_trust.py"
TF = "01_data_cleaning/trust_formulas.py"
SWEEP = "llm/c_graphrag/stage1_sweep.py"
CTXSWEEP = "llm/c_graphrag/ctxrel_sweep.py"
STAGE2 = "llm/c_graphrag/stage2_report.py"
LOGSCORE_RECHECK = "llm/c_graphrag/logscore_stage1_recheck.py"
OFFICIAL = "llm/c_graphrag/official_params.json"
JUDGE_CLIENTS = "llm/evaluation/judge_clients.py"
JUDGE_V21 = "llm/evaluation/judge_prompt_v2_1.py"
CTXREL_V2 = "llm/evaluation/ctxrel_prompt_v2.py"
ANNOT = "llm/evaluation/annotation_tool.py"
CALGEN = "llm/evaluation/calibration_generate.py"
STATS = "llm/evaluation/compare_conditions_stats.py"
STATS_COMMON = "llm/evaluation/_stats_common.py"

KG_LOGS = "01_data_cleaning/logs"

STATUSES = ("frozen", "pending", "candidate", "exploratory", "observed", "judge")


@dataclass(frozen=True)
class Quote:
    file: str
    needle: str  # must occur on exactly one line of `file`


@dataclass(frozen=True)
class Step:
    key: str
    number: int
    title: str
    purpose: str
    modules: tuple[str, ...]
    inputs: str
    outputs: str
    quotes: tuple[Quote, ...] = ()  # verbatim background passages shown with the step


@dataclass(frozen=True)
class Param:
    key: str
    name: str
    step: str
    source: Src
    unit: str = ""
    used_by: tuple[str, ...] = ()
    rationale: Quote | None = None
    spec: Quote | None = None
    set_by: str | None = None
    status: str = "frozen"
    kind: str = "parameter"
    note: str | None = None


STEPS: tuple[Step, ...] = (
    Step("data", 1, "Data: SORD source and deduplication",
         "Merge the SORD question/answer CSV variants into one deduplicated parquet per entity and add "
         "Stack Overflow PostLinks (Linked/Duplicate) as question links.",
         ("01_data_cleaning/3_merge_sord_sources.py", "01_data_cleaning/3b_dedup_merged_sources.py", POSTLINKS),
         "SORD CSVs (00_datasource/raw), PostLinks.xml",
         "00_datasource/merged/*_raw_union.parquet, question_links.parquet"),
    Step("sampling", 2, "Cleaning and sampling",
         "Draw the evaluation questions from one fixed permutation of the token-filtered candidate pool: the "
         "test sample is positions 1-384, the development set positions 385-434. Evaluation answers are "
         "excluded from every retrieval corpus.",
         (SPLIT, C, B, ENGINE),
         "Merged question/answer parquet",
         "Ordered evaluation questions (test / dev split)",
         (Quote(DEC, "Test sample = positions 1-384 (1-based)"),
          Quote(DEC, "Dev set = positions 385-434 (1-based)"),
          Quote(B, "PENCEGAHAN DATA LEAKAGE (ketat, by design)"))),
    Step("kg_build", 3, "Knowledge graph build and densification",
         "Load Question, Answer, Tag and User nodes into Neo4j with weighted edges, then add TAG_COOCCUR and "
         "EMBED_SIM edges so that most questions have graph neighbours.",
         (KG7, AUTHOR9, TAG10, EMB11, VAL12),
         "Merged parquet, users.parquet, question_links.parquet",
         "Neo4j database `graphrag`, FAISS index + embedding cache (KG_WORKSPACE_DIR)"),
    Step("trust", 4, "Trust scoring",
         "Compute the intrinsic trustScore of every Answer and Question node from community signals. All "
         "formulas live in one module used by the KG build and by the update scripts.",
         (TF, REP13, LOG14),
         "Answer/Question score, accepted flag, views, AUTHOR_TRUST weight",
         "trustScore + trustScoreVersion on Answer and Question nodes, backup parquet, apply/verify logs",
         (Quote(TF, "norm_log(score; P99) = clip(log1p(max(score, 0)) / log1p(P99), 0, 1)"),
          Quote(DEC, "Log-scaled weights are used instead of min-max on raw reputation"),
          Quote(DEC, "it is included so that the trust score uses all three community signals"))),
    Step("conditions", 5, "Retrieval conditions A, B, C, D",
         "Four generation conditions share one sampling and one 4-turn prompt skeleton and differ only in "
         "retrieval: none (A), flat dense retrieval (B), trust-weighted graph retrieval (C), dual-level graph "
         "retrieval without trust (D). Grounding can be switched on or off for C and D.",
         (A, B, C, D, PROMPTS),
         "Evaluation questions, Neo4j graph, FAISS cache",
         "Run JSONL + manifest per condition",
         (Quote(README, "- **Condition A**: sample 384 questions"),
          Quote(README, "- **Condition B**: same sampling, + FAISS top-k retrieval"),
          Quote(README, "- **Condition C**: same sampling, + hybrid retrieval"),
          Quote(README, "- **Condition D**: same sampling and KG/FAISS cache as Condition C"))),
    Step("c_ranking", 6, "Condition C ranking",
         "Candidates come from entity anchoring (FAISS), graph traversal (1-2 hops) and semantic expansion, "
         "are deduplicated by answer, scored, and the top_k are kept. C retrieval v3 ranks by "
         "final = (1 - alpha) * sim_norm + alpha * combined_trust.",
         (C,),
         "Query embedding, Neo4j graph, FAISS cache",
         "Top-k context items per question",
         (Quote(DEC, "1. Linear: final = (1 - α) * sim_norm + α * trust"),
          Quote(DEC, "- sim_norm: min-max normalization of sim over the candidate pool"),
          Quote(DEC, "- trust: the existing combined trust (0.7 path trust + 0.3 answer trust"))),
    Step("alpha_selection", 7, "Alpha selection",
         "Stage 1 judges the relevance of retrieved contexts on the dev set for each alpha candidate and keeps "
         "the two best; stage 2 compares their hallucination rates after the judge is frozen.",
         (SWEEP, CTXSWEEP, LOGSCORE_RECHECK, STAGE2),
         "Dev-set retrieval per alpha, ctxrel-v2 primary judge labels",
         "Stage-1 report of record, stage-2 candidates, (later) stage-2 report",
         (Quote(DEC, "Primary criterion: % of questions with ≥1 RELEVANT context item."),
          Quote(DEC, "Ranking: by primary criterion, then tie-breaker, then larger α."),
          Quote(DEC, "- Keep-current rule (stage 1):"),
          Quote(DEC, "The v3 stage-1 recheck is **binding**"))),
    Step("judge", 8, "Hallucination judge",
         "An LLM judge labels each generated answer against the accepted answer. The binary hallucination rate "
         "is the primary metric; the judge is calibrated against human annotation before it is frozen.",
         (JUDGE_V21, JUDGE_CLIENTS, ANNOT, CALGEN),
         "Generated answers, accepted answers",
         "Judge labels (binary + severity), calibration kappa",
         (Quote(DEC, "**Primary:** binary hallucination rate"),
          Quote(DEC, "**Secondary (descriptive):** severity distribution"),
          Quote(DEC, "Agreement measured on 3 categories"))),
    Step("official", 9, "Official run",
         "After selection and freeze, every condition runs once on the official test sample (n = 384).",
         (ENGINE, OFFICIAL, "backend/app/routers/config.py"),
         "Frozen parameters (official_params.json)",
         "Official run files for A, B, C, D",
         (Quote(DEC, "After selection: freeze α, the generation prompt"),)),
    Step("metrics", 10, "Metrics and statistics",
         "Answers are scored with cosine similarity to the accepted answer, the judge's hallucination rate, "
         "citation compliance (NF2) and retrieval latency (NF3); conditions are compared pairwise with "
         "Wilcoxon signed-rank tests, Holm correction and Cliff's delta.",
         (STATS, STATS_COMMON, C, VAL12),
         "Run files, judge labels",
         "Comparison tables (analysis/)"),
)


P = Param  # brevity below

PARAMETERS: tuple[Param, ...] = (
    # ---------------------------------------------------------------- 1 data
    P("data.valid_link_types", "PostLinks types kept", "data", Src("const", POSTLINKS, "VALID_LINK_TYPES"),
      "LinkTypeId -> label", (POSTLINKS, KG7)),
    P("data.questions", "Question nodes", "data",
      Src("log_field", KG_LOGS, r"Question node\s*:\s*([\d,]+)", glob="*_build_knowledge_graph.log"),
      "nodes", kind="result", status="observed"),
    P("data.answers", "Answer nodes", "data",
      Src("log_field", KG_LOGS, r"Answer node\s*:\s*([\d,]+)", glob="*_build_knowledge_graph.log"),
      "nodes", kind="result", status="observed"),
    P("data.accepted", "Accepted answers", "data",
      Src("log_field", KG_LOGS, r"Answer accepted \(HAS_ACCEPTED_ANSWER\):\s*([\d,]+)",
          glob="*_build_knowledge_graph.log"),
      "answers", kind="result", status="observed"),
    P("data.tags", "Tag nodes", "data",
      Src("log_field", KG_LOGS, r"Tag node \(unik\)\s*:\s*([\d,]+)", glob="*_build_knowledge_graph.log"),
      "nodes", kind="result", status="observed"),

    # ---------------------------------------------------------------- 2 sampling
    P("sampling.seed", "Sampling seed", "sampling", Src("cli_default", C, "--seed"), "",
      (SPLIT, A, B, C, D), spec=Quote(DEC, "Same pool and filter as the test sample"), set_by="decision"),
    P("sampling.n_sample", "Official sample size n", "sampling", Src("cli_default", C, "--n-sample"), "questions",
      (A, B, C, D), rationale=Quote(README, "- **Condition A**: sample 384 questions"),
      spec=Quote(DEC, "Run the official test sample (n = 384) once.")),
    P("sampling.oversample_pool", "Oversample pool", "sampling", Src("const", ENGINE, "DEFAULT_OVERSAMPLE_POOL"),
      "candidate questions", (ENGINE, "backend/app/routers/config.py"),
      rationale=Quote(ENGINE, "Nested sampling -- the property that an n=10 pilot sample is"),
      spec=Quote(DEC, "Same pool and filter as the test sample"), set_by="decision"),
    P("sampling.oversample_pool_calibration", "Oversample pool (copy in calibration generation)", "sampling",
      Src("const", CALGEN, "OVERSAMPLE_POOL"), "candidate questions", (CALGEN,),
      note="Separate constant with the same role as DEFAULT_OVERSAMPLE_POOL; a test checks both stay equal."),
    P("sampling.token_limit", "Question token filter", "sampling", Src("const", C, "TOKEN_LIMIT"),
      "tokens (cl100k_base)", (A, B, C, D),
      rationale=Quote(A, "TOKEN_LIMIT = 2048  # sesuai batasan paper asli")),
    P("sampling.dev_offset", "Dev split start (1-based)", "sampling", Src("const", SPLIT, "DEFAULT_DEV_OFFSET"),
      "position", (SPLIT, C), rationale=Quote(DEC, "All tuning (α, generation prompt wording"),
      spec=Quote(DEC, "Dev set = positions 385-434 (1-based)"), set_by="decision"),
    P("sampling.n_dev", "Dev set size", "sampling", Src("const", STAGE2, "N_DEV"), "questions", (STAGE2, SWEEP),
      rationale=Quote(DEC, "All tuning (α, generation prompt wording"),
      spec=Quote(DEC, "Dev set = positions 385-434 (1-based)"), set_by="decision"),

    # ---------------------------------------------------------------- 3 KG build
    P("kg.w_has_accepted_answer", "HAS_ACCEPTED_ANSWER weight", "kg_build",
      Src("source_literal", KG7, r"MERGE \(q\)-\[r:HAS_ACCEPTED_ANSWER\]->\(a\)\s*\n\s*SET r\.weight = ([0-9.]+)"),
      "edge weight", (KG7, C)),
    P("kg.w_has_answer_fallback", "HAS_ANSWER weight when all scores in a thread are equal", "kg_build",
      Src("source_literal", KG7, r"fillna\(([0-9.]+)\)", function="compute_answer_weights"),
      "edge weight (otherwise per-thread min-max of score)", (KG7, C),
      spec=Quote(DEC, "Unchanged: `HAS_ANSWER` edge weights (per-thread min-max, fallback 0.5)"),
      set_by="amendment-6"),
    P("kg.w_tagged_with", "TAGGED_WITH weight", "kg_build",
      Src("source_literal", KG7, r"MERGE \(q\)-\[r:TAGGED_WITH\]->\(t\)\s*\n\s*SET r\.weight = ([0-9.]+)"),
      "edge weight", (KG7,)),
    P("kg.w_related_duplicate", "IS_RELATED_TO weight (Duplicate)", "kg_build",
      Src("source_literal", KG7, r'== "Duplicate", "weight"\] = ([0-9.]+)'),
      "edge weight (Linked = embedding cosine)", (KG7, C)),
    P("kg.w_related_linked_fallback", "IS_RELATED_TO weight (Linked, --skip-embeddings only)", "kg_build",
      Src("source_literal", KG7, r'links\.loc\[linked_mask, "weight"\] = ([0-9.]+)'),
      "edge weight", (KG7,),
      note="Fallback only; the build log 20260829T075916Z computed cosine similarity for the Linked edges."),
    P("kg.tag_max_df", "TAG_COOCCUR: max tag document frequency", "kg_build", Src("const", TAG10, "DEFAULT_MAX_DF"),
      "questions per tag", (TAG10,)),
    P("kg.tag_jaccard", "TAG_COOCCUR: Jaccard threshold", "kg_build",
      Src("const", TAG10, "DEFAULT_JACCARD_THRESHOLD"), "Jaccard", (TAG10,)),
    P("kg.tag_top_k", "TAG_COOCCUR: neighbours per question", "kg_build", Src("const", TAG10, "DEFAULT_TOP_K"),
      "edges", (TAG10,)),
    P("kg.tag_weight", "TAG_COOCCUR weight", "kg_build", Src("const", TAG10, "DEFAULT_TRUST_WEIGHT"),
      "edge weight", (TAG10, C, D), rationale=Quote(TAG10, "DEFAULT_TRUST_WEIGHT = 0.6  #")),
    P("kg.emb_top_k", "EMBED_SIM: neighbours per question", "kg_build", Src("const", EMB11, "DEFAULT_TOP_K"),
      "edges", (EMB11,)),
    P("kg.emb_cosine", "EMBED_SIM: cosine threshold", "kg_build", Src("const", EMB11, "DEFAULT_COSINE_THRESHOLD"),
      "cosine", (EMB11,)),
    P("kg.emb_weight", "EMBED_SIM weight", "kg_build", Src("const", EMB11, "DEFAULT_TRUST_WEIGHT"),
      "edge weight", (EMB11, C, D), rationale=Quote(EMB11, "DEFAULT_TRUST_WEIGHT = 0.4  #")),
    P("kg.emb_nprobe", "EMBED_SIM: FAISS nprobe", "kg_build", Src("cli_default", EMB11, "--nprobe"),
      "IVF lists probed", (EMB11,)),
    P("kg.target_isolated", "Densification target: isolated questions", "kg_build",
      Src("const", VAL12, "TARGET_ISOLATED_PCT"), "% (at most)", (VAL12,)),
    P("kg.target_mean_degree", "Densification target: mean degree", "kg_build",
      Src("const", VAL12, "TARGET_MEAN_DEGREE"), "edges (at least)", (VAL12,)),
    P("kg.target_median_degree", "Densification target: median degree", "kg_build",
      Src("const", VAL12, "TARGET_MEDIAN_DEGREE"), "edges (at least)", (VAL12,)),
    # results
    P("kg.n_has_accepted_answer", "HAS_ACCEPTED_ANSWER edges", "kg_build",
      Src("log_field", KG_LOGS, r"Edge HAS_ACCEPTED_ANSWER\s*:\s*([\d,]+)", glob="*_build_knowledge_graph.log"),
      "edges", kind="result", status="observed"),
    P("kg.n_has_answer", "HAS_ANSWER edges", "kg_build",
      Src("log_field", KG_LOGS, r"Edge HAS_ANSWER\s*:\s*([\d,]+)", glob="*_build_knowledge_graph.log"),
      "edges", kind="result", status="observed"),
    P("kg.n_tagged_with", "TAGGED_WITH edges", "kg_build",
      Src("log_field", KG_LOGS, r"Edge TAGGED_WITH\s*:\s*([\d,]+)", glob="*_build_knowledge_graph.log"),
      "edges", kind="result", status="observed"),
    P("kg.n_is_related_to", "IS_RELATED_TO edges", "kg_build",
      Src("log_field", KG_LOGS, r"Edge IS_RELATED_TO\s*:\s*([\d,]+)", glob="*_build_knowledge_graph.log"),
      "edges", kind="result", status="observed"),
    P("kg.n_author_trust", "AUTHOR_TRUST edges", "kg_build",
      Src("log_field", KG_LOGS, r"Neo4j AUTHOR_TRUST:\s*([\d,]+)", glob="*_load_author_trust.log"),
      "edges", kind="result", status="observed"),
    P("kg.n_tag_cooccur", "TAG_COOCCUR edges", "kg_build",
      Src("log_field", KG_LOGS, r"Neo4j TAG_COOCCUR:\s*([\d,]+)", glob="*_densify_tag_cooccurrence.log"),
      "edges", kind="result", status="observed"),
    P("kg.n_embed_sim", "EMBED_SIM edges", "kg_build",
      Src("log_field", KG_LOGS, r"Neo4j EMBED_SIM:\s*([\d,]+)", glob="*_densify_embedding_similarity.log"),
      "edges", kind="result", status="observed"),
    P("kg.isolated_pct", "Isolated questions after densification", "kg_build",
      Src("log_field", "logs", r"Question isolated \(deg=0\):\s*[\d,]+ \(([\d.]+)%\)",
          glob="*_validate_densification.log"),
      "%", kind="result", status="observed"),
    P("kg.mean_degree", "Mean question degree after densification", "kg_build",
      Src("log_field", "logs", r"Mean degree\s*:\s*([\d.]+)", glob="*_validate_densification.log"),
      "edges", kind="result", status="observed"),
    P("kg.median_degree", "Median question degree after densification", "kg_build",
      Src("log_field", "logs", r"Median degree\s*:\s*([\d.]+)", glob="*_validate_densification.log"),
      "edges", kind="result", status="observed"),

    # ---------------------------------------------------------------- 4 trust
    P("trust.w_a_score", "trust_A weight: upvote score", "trust", Src("const", TF, "W_A_SCORE"), "weight",
      (TF, KG7, LOG14), spec=Quote(DEC, "3. **Weights unchanged** *(confirmed 2026-10-14)*"), set_by="amendment-6"),
    P("trust.w_a_accepted", "trust_A weight: accepted", "trust", Src("const", TF, "W_A_ACCEPTED"), "weight",
      (TF, KG7, LOG14), spec=Quote(DEC, "3. **Weights unchanged** *(confirmed 2026-10-14)*"), set_by="amendment-6"),
    P("trust.w_a_author", "trust_A weight: author reputation (w_auth)", "trust", Src("const", TF, "W_A_AUTHOR"),
      "weight", (TF, KG7, LOG14), spec=Quote(DEC, "3. **Weights unchanged** *(confirmed 2026-10-14)*"),
      set_by="amendment-5"),
    P("trust.w_q_score", "trust_Q weight: upvote score", "trust", Src("const", TF, "W_Q_SCORE"), "weight",
      (TF, KG7, LOG14), spec=Quote(DEC, "3. **Weights unchanged** *(confirmed 2026-10-14)*"), set_by="amendment-6"),
    P("trust.w_q_views", "trust_Q weight: views", "trust", Src("const", TF, "W_Q_VIEWS"), "weight",
      (TF, KG7, LOG14), spec=Quote(DEC, "3. **Weights unchanged** *(confirmed 2026-10-14)*"), set_by="amendment-6"),
    P("trust.version", "trustScore version (log-normalised score)", "trust",
      Src("const", TF, "TRUST_SCORE_VERSION"), "trustScoreVersion", (TF, KG7, LOG14, AUTHOR9),
      rationale=Quote(DEC, "This is inconsistent with the rationale already recorded in Amendment 5"),
      spec=Quote(DEC, "1. **Formula** *(confirmed 2026-10-14)*"), set_by="amendment-6"),
    P("trust.p99_quantile", "P99 quantile", "trust", Src("const", TF, "P99_QUANTILE"), "quantile", (TF, LOG14),
      spec=Quote(DEC, "2. **P99 population** *(confirmed 2026-10-14)*"), set_by="amendment-6"),
    P("trust.quantile_method", "Quantile method", "trust", Src("const", TF, "QUANTILE_METHOD"), "numpy method",
      (TF, LOG14), rationale=Quote(DEC, "2. **P99 population** *(confirmed 2026-10-14)*"), set_by="amendment-6"),
    P("trust.p99_answer", "P99 of Answer score", "trust",
      Src("log_field", KG_LOGS, r"P99 score -- Answer: ([\d.]+)", glob="*_apply_logscore_trust.log"),
      "upvotes", (LOG14,), spec=Quote(DEC, "- P99_A:"), set_by="amendment-6", kind="result", status="observed"),
    P("trust.p99_question", "P99 of Question score", "trust",
      Src("log_field", KG_LOGS, r"P99 score -- Answer: [\d.]+ \| Question: ([\d.]+)", glob="*_apply_logscore_trust.log"),
      "upvotes", (LOG14,), spec=Quote(DEC, "- P99_A:"), set_by="amendment-6", kind="result", status="observed"),
    P("trust.applied_at", "v3-logscore applied at", "trust", Src("const", STAGE2, "TRUST_V3_APPLIED_AT"),
      "ISO 8601", (STAGE2,), spec=Quote(DEC, "- Apply timestamp (ISO 8601 with timezone)"),
      set_by="amendment-6", kind="result", status="observed"),

    # ---------------------------------------------------------------- 5 conditions
    P("cond.prompt_version", "Generation prompt version", "conditions", Src("const", PROMPTS, "PROMPT_VERSION"),
      "", (A, B, C, D), rationale=Quote(PROMPTS, '"v3" (PROMPT_VERSION saat ini)')),
    P("cond.generation_model", "Generation model", "conditions", Src("cli_default", C, "--model"), "",
      (A, B, C, D), spec=Quote(DEC, "same generation model (`gpt-4o-mini`)"), set_by="amendment-4"),
    P("cond.c_retrieval_version", "C retrieval version (code default)", "conditions",
      Src("const", C, "C_RETRIEVAL_VERSION"), "", (C,), rationale=Quote(C, '"v2" (C_RETRIEVAL_VERSION saat ini)'),
      note="v3 (alpha ranking) is selected per run with --c-retrieval-version v3; stage 1 ran with v3."),
    P("cond.d_retrieval_version", "D retrieval version", "conditions", Src("const", D, "D_RETRIEVAL_VERSION"), "",
      (D,)),
    P("cond.b_top_k", "B: top_k", "conditions", Src("cli_default", B, "--top-k"), "chunks", (B,)),
    P("cond.d_n_low_level", "D: low-level anchors", "conditions", Src("cli_default", D, "--n-low-level"),
      "questions", (D,)),
    P("cond.d_n_high_level", "D: high-level neighbours", "conditions", Src("cli_default", D, "--n-high-level"),
      "questions", (D,)),

    # ---------------------------------------------------------------- 6 C ranking
    P("crank.alpha_candidates", "alpha candidates", "c_ranking", Src("const", SWEEP, "DEFAULT_ALPHA_VALUES"), "",
      (SWEEP, LOGSCORE_RECHECK), rationale=Quote(DEC, "Chosen: option 1. Reasons:"),
      spec=Quote(DEC, "- Candidates: α ∈ {0, 0.25, 0.5, 0.75, 1}."), set_by="decision", status="candidate"),
    P("crank.alpha_official", "alpha (official)", "c_ranking", Src("json_field", OFFICIAL, path=("alpha",)), "",
      (OFFICIAL, "backend/app/routers/config.py"), spec=Quote(DEC, "After selection: freeze α, the generation prompt"),
      set_by="decision", status="pending"),
    P("crank.fusion_w_path", "combined_trust weight: path trust", "c_ranking",
      Src("const", C, "DEFAULT_FUSION_W_PATH_TRUST"), "weight", (C,),
      spec=Quote(DEC, "- trust: the existing combined trust (0.7 path trust + 0.3 answer trust"), set_by="decision"),
    P("crank.fusion_w_intrinsic", "combined_trust weight: answer trustScore", "c_ranking",
      Src("const", C, "DEFAULT_FUSION_W_ANSWER_INTRINSIC_TRUST"), "weight", (C,),
      spec=Quote(DEC, "- trust: the existing combined trust (0.7 path trust + 0.3 answer trust"), set_by="decision"),
    P("crank.semantic_cap", "Semantic-expansion trust cap", "c_ranking",
      Src("const", C, "DEFAULT_SEMANTIC_EXPANSION_TRUST_CAP"), "trust", (C,),
      rationale=Quote(C, "Proxy trust utk kandidat yang HANYA ditemukan lewat semantic expansion")),
    P("crank.top_k", "top_k", "c_ranking", Src("cli_default", C, "--top-k"), "context items", (C,),
      spec=Quote(DEC, "keep top_k = 5"), set_by="decision"),
    P("crank.n_anchor", "Entity anchors (FAISS)", "c_ranking", Src("cli_default", C, "--n-anchor"), "questions",
      (C,)),
    P("crank.n_semantic_expansion", "Semantic-expansion anchors", "c_ranking",
      Src("cli_default", C, "--n-semantic-expansion"), "questions", (C,)),
    P("crank.token_chunk_limit", "Context item token limit", "c_ranking", Src("cli_default", C, "--token-chunk-limit"),
      "tokens (cl100k_base)", (C, B)),
    P("crank.embed_model", "Embedding model (retrieval and cosine metric)", "c_ranking",
      Src("cli_default", C, "--embed-model"), "", (C, EMB11, A, B, D)),
    P("crank.max_hops", "max_hops", "c_ranking", Src("const", C, "DEFAULT_MAX_HOPS"), "hops", (C,),
      rationale=Quote(DEC, "may be varied on the dev set to understand the system"), set_by="decision"),
    P("crank.edge_types", "Traversal edge types", "c_ranking", Src("const", C, "DEFAULT_EDGE_TYPES"), "", (C,),
      rationale=Quote(DEC, "may be varied on the dev set to understand the system"), set_by="decision"),
    P("crank.use_author_trust", "use_author_trust", "c_ranking", Src("const", C, "DEFAULT_USE_AUTHOR_TRUST"), "",
      (C,), rationale=Quote(DEC, "Turning `use_author_trust` on therefore adds reputation a second time"),
      set_by="amendment-5"),
    P("crank.author_trust_beta", "beta for use_author_trust", "c_ranking", Src("const", C, "AUTHOR_TRUST_BETA"), "",
      (C,), spec=Quote(DEC, "with **β = 0.3** (fixed)"), set_by="decision", status="exploratory"),
    P("crank.accepted_only", "accepted_only", "c_ranking", Src("const", C, "DEFAULT_ACCEPTED_ONLY"), "", (C,),
      rationale=Quote(DEC, "may be varied on the dev set to understand the system"), set_by="decision"),
    P("crank.c_retrieval_version_official", "C retrieval version (official)", "c_ranking",
      Src("json_field", OFFICIAL, path=("c_retrieval_version",)), "", (OFFICIAL,),
      spec=Quote(DEC, "After selection: freeze α, the generation prompt"), set_by="decision", status="pending"),

    # ---------------------------------------------------------------- 7 alpha selection
    P("sel.ctxrel_version", "Context-relevance judge (stage 1)", "alpha_selection",
      Src("const", CTXREL_V2, "PROMPT_VERSION"), "", (SWEEP, LOGSCORE_RECHECK),
      rationale=Quote(DEC, "Pass: **yes** (primary decides"),
      spec=Quote(DEC, "Context-relevance judge version used for stage 1: **ctxrel-v2**"), set_by="amendment-3"),
    P("sel.control_positive", "Control threshold: positives RELEVANT", "alpha_selection",
      Src("const", SWEEP, "CONTROL_THRESHOLDS", path=("positive", 1)), "share (at least)", (SWEEP,),
      spec=Quote(DEC, "- Positives: ≥ 90% RELEVANT"), set_by="amendment-2"),
    P("sel.control_easy", "Control threshold: easy negatives IRRELEVANT", "alpha_selection",
      Src("const", SWEEP, "CONTROL_THRESHOLDS", path=("easy_negative", 1)), "share (at least)", (SWEEP,),
      spec=Quote(DEC, "- Easy negatives (no shared tag): ≥ 90% IRRELEVANT"), set_by="amendment-2"),
    P("sel.control_hard", "Control threshold: hard negatives NOT RELEVANT", "alpha_selection",
      Src("const", SWEEP, "CONTROL_THRESHOLDS", path=("hard_negative", 1)), "share (at least)", (SWEEP,),
      rationale=Quote(DEC, "The hard-negative threshold is lower because"),
      spec=Quote(DEC, "- Hard negatives (shared tag): ≥ 80% NOT RELEVANT"), set_by="amendment-2"),
    P("sel.tie_pp", "Stage 1 tie margin: primary criterion", "alpha_selection",
      Src("const", CTXSWEEP, "TIE_THRESHOLD_PERCENTAGE_POINTS"), "percentage points", (CTXSWEEP,),
      spec=Quote(DEC, "Differences of ≤1 question (2 percentage points) are ties."), set_by="decision"),
    P("sel.tie_score", "Stage 1 tie margin: mean relevance score", "alpha_selection",
      Src("const", CTXSWEEP, "RELEVANCE_SCORE_TIE_THRESHOLD"), "score", (CTXSWEEP,),
      spec=Quote(DEC, "Differences ≤ 0.02 are ties."), set_by="decision"),
    P("sel.stage2_tie", "Stage 2 tie margin: hallucinated answers", "alpha_selection",
      Src("const", CTXSWEEP, "STAGE2_ANSWER_COUNT_TIE_MARGIN"), "answers", (CTXSWEEP, STAGE2),
      rationale=Quote(DEC, "If still tied: choose the larger α (the contribution under test is trust)."),
      spec=Quote(DEC, "Differences of ≤2 answers (2 percentage points) are ties."), set_by="decision"),
    P("sel.stage2_alphas", "Stage 2 candidates", "alpha_selection", Src("const", STAGE2, "STAGE2_ALPHAS"), "alpha",
      (STAGE2,), rationale=Quote(DEC, "Ranking: 0.5, 0.25, 0.0, 0.75, 1.0"),
      spec=Quote(DEC, "The binding v3 recheck confirms the stage-2 candidates of record"),
      set_by="amendment-6", status="candidate"),

    # ---------------------------------------------------------------- 8 judge
    P("judge.primary_model", "Primary judge model", "judge",
      Src("call_kwarg", JUDGE_CLIENTS, "JUDGE_REGISTRY", path=("primary", "model")), "", (JUDGE_CLIENTS,),
      status="judge"),
    P("judge.secondary_model", "Secondary judge model", "judge",
      Src("call_kwarg", JUDGE_CLIENTS, "JUDGE_REGISTRY", path=("secondary", "model")), "", (JUDGE_CLIENTS,),
      status="judge"),
    P("judge.version", "Hallucination judge prompt version", "judge", Src("const", JUDGE_V21, "PROMPT_VERSION"), "",
      (JUDGE_V21,), rationale=Quote(DEC, "Pilot diagnostics (test positions 1-10, 9 runs"),
      spec=Quote(DEC, "Target: κ ≥ 0.6. At most 3 prompt versions"), set_by="amendment-4", status="judge"),
    P("judge.parse_version", "Judge parser version", "judge", Src("const", JUDGE_V21, "PARSE_VERSION"), "",
      (JUDGE_V21,), set_by="amendment-4", status="judge"),
    P("judge.temperature", "Judge temperature", "judge", Src("const", JUDGE_V21, "REQUEST_TEMPERATURE"), "",
      (JUDGE_V21,), spec=Quote(DEC, "`max_tokens` and `temperature` are set explicitly"), set_by="amendment-4",
      status="judge"),
    P("judge.max_tokens", "Judge max_tokens", "judge", Src("const", JUDGE_V21, "REQUEST_MAX_TOKENS"), "tokens",
      (JUDGE_V21,), spec=Quote(DEC, "`max_tokens` and `temperature` are set explicitly"), set_by="amendment-4",
      status="judge"),
    P("judge.kappa_target", "Calibration kappa target", "judge", Src("const", ANNOT, "ACCEPTANCE_KAPPA"),
      "Cohen's kappa (at least)", (ANNOT,), spec=Quote(DEC, "Target: κ ≥ 0.6. At most 3 prompt versions"),
      set_by="amendment-4"),
    P("judge.n_annotation", "Calibration annotation items", "judge", Src("const", CALGEN, "N_ANNOTATION_ITEMS"),
      "items", (CALGEN, ANNOT), spec=Quote(DEC, "Annotation sample: 60 items"), set_by="amendment-4"),

    # ---------------------------------------------------------------- 9 official
    P("off.status", "Official parameters status", "official", Src("json_field", OFFICIAL, path=("status",)), "",
      (OFFICIAL, "backend/app/routers/config.py"),
      spec=Quote(DEC, "While `status` is `pending_selection`, the dashboard refuses"), status="pending"),
    P("off.n", "Official n", "official", Src("const", ENGINE, "MAX_PLANNED_N_SAMPLE"), "questions", (ENGINE,),
      rationale=Quote(README, "- **Condition A**: sample 384 questions"),
      spec=Quote(DEC, "Run the official test sample (n = 384) once."), set_by="decision"),
    P("off.sample_split", "Official sample split", "official", Src("json_field", OFFICIAL, path=("sample_split",)),
      "", (OFFICIAL,), set_by="decision"),
    P("off.max_hops", "Official max_hops", "official", Src("json_field", OFFICIAL, path=("max_hops",)), "hops",
      (OFFICIAL,), rationale=Quote(DEC, "may be varied on the dev set to understand the system"), set_by="decision"),
    P("off.edge_types", "Official edge types", "official", Src("json_field", OFFICIAL, path=("edge_types",)), "",
      (OFFICIAL,), rationale=Quote(DEC, "may be varied on the dev set to understand the system"), set_by="decision"),
    P("off.use_author_trust", "Official use_author_trust", "official",
      Src("json_field", OFFICIAL, path=("use_author_trust",)), "", (OFFICIAL,), rationale=Quote(DEC, "may be varied on the dev set to understand the system"), set_by="decision"),
    P("off.accepted_only", "Official accepted_only", "official",
      Src("json_field", OFFICIAL, path=("accepted_only",)), "", (OFFICIAL,), rationale=Quote(DEC, "may be varied on the dev set to understand the system"), set_by="decision"),

    # ---------------------------------------------------------------- 10 metrics
    P("met.hallucination_metric_stats", "Hallucination metric in the comparison script", "metrics",
      Src("const", STATS, "METRICS_HALLUCINATION"), "", (STATS,),
      spec=Quote(DEC, "**Primary:** binary hallucination rate"), set_by="amendment-4",
      note="compare_conditions_stats.py still uses the 3-class ordinal label; Amendment 4 makes the binary "
           "rate the primary metric."),
    P("met.wilcoxon_zero_method", "Wilcoxon zero_method", "metrics",
      Src("source_literal", STATS, r'zero_method="(\w+)"', function="paired_comparisons"), "", (STATS,),
      rationale=Quote(STATS, '`zero_method="wilcox"` (default scipy')),
    P("met.wilcoxon_alternative", "Wilcoxon alternative", "metrics",
      Src("source_literal", STATS, r'alternative="([\w-]+)"', function="paired_comparisons"), "", (STATS,)),
    P("met.cliff_negligible", "Cliff's delta: negligible below", "metrics",
      Src("source_literal", STATS_COMMON, r"\bif ad < ([0-9.]+):"), "|delta|", (STATS_COMMON,),
      rationale=Quote(STATS_COMMON, "Pita interpretasi (Romano dkk. 2006")),
    P("met.cliff_small", "Cliff's delta: small below", "metrics",
      Src("source_literal", STATS_COMMON, r'elif ad < ([0-9.]+):\s*\n\s*band = "small"'), "|delta|", (STATS_COMMON,),
      rationale=Quote(STATS_COMMON, "Pita interpretasi (Romano dkk. 2006")),
    P("met.cliff_medium", "Cliff's delta: medium below", "metrics",
      Src("source_literal", STATS_COMMON, r'elif ad < ([0-9.]+):\s*\n\s*band = "medium"'), "|delta|",
      (STATS_COMMON,), rationale=Quote(STATS_COMMON, "Pita interpretasi (Romano dkk. 2006")),
    P("met.latency_target", "NF3 latency target", "metrics", Src("const", VAL12, "TARGET_LATENCY_SEC"),
      "seconds (at most)", (VAL12,)),
)

PARAMS_BY_KEY = {p.key: p for p in PARAMETERS}
STEPS_BY_KEY = {s.key: s for s in STEPS}
