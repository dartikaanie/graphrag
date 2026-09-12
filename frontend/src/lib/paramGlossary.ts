/**
 * Single source of truth for what each run-configuration keyword means, so
 * the explanation is worded identically everywhere the field appears
 * (RunConditionPage for A/B/C individually, and RunAllConditionsPage).
 */
export const PARAM_GLOSSARY = {
  n_sample: 'Number of questions to evaluate in this batch run.',
  seed: 'Random seed for sampling the question pool. Use the same seed across A/B/C so every condition is evaluated on the exact same questions.',
  oversample_pool:
    'Size of the candidate pool drawn before filtering/sampling down to n_sample (e.g. questions that fit the token limit). Leave blank to auto-size it.',
  provider: 'Which LLM API to send requests to.',
  model: 'The specific LLM model name to use, e.g. gpt-4o-mini.',
  question_id: 'Stack Overflow question ID to run a single evaluation against, instead of a batch.',
  top_k: 'Number of retrieved context chunks (answers) included in the prompt sent to the LLM.',
  n_anchor: 'Number of similar questions used as entry points ("anchors") into the knowledge graph before traversal.',
  n_semantic_expansion: 'Number of extra candidates pulled in by embedding similarity, in addition to graph traversal, to fill out the retrieved context.',
  require_citation:
    'Require the LLM to cite its sources inline as [SO-<id>] for each claim, so citation compliance (NF2) can be measured and compared against Condition C.',
  fusion_mode:
    'Ablation switch for how retrieved candidates are ranked. "trust_weighted" (default) ranks by combined trust score. "uniform" ignores trust entirely and keeps candidates in the order they were discovered, isolating trust-weighting’s actual contribution.',
  fusion_w_path_trust:
    'Weight given to graph-path trust (edge trust along the traversal path) when computing a candidate’s combined score. Only used in "trust_weighted" mode.',
  fusion_w_intrinsic:
    'Weight given to an answer’s own intrinsic trust score (independent of how it was reached in the graph) when computing its combined score. Only used in "trust_weighted" mode.',
  semantic_expansion_trust_cap:
    'Maximum trust score assignable to candidates found only via semantic (embedding) expansion, since they lack a graph path to derive trust from. Only used in "trust_weighted" mode.',
  n_low_level:
    'Number of similar questions used as entry points ("anchors") for low-level retrieval — the answers directly attached to each anchor question, ranked purely by anchor similarity (no trust score).',
  n_high_level:
    'Number of related questions (via tag co-occurrence / relatedness edges) used for high-level retrieval — broader, theme-level context beyond the direct anchors, ranked by the raw edge relevance score (no trust score).',
  require_grounding:
    'Whether the LLM is instructed to ground its answer ONLY in the retrieved context, in addition to citing sources. Turning this off keeps the citation instruction but drops the "don’t introduce facts outside the context" constraint — isolating the effect of the grounding constraint itself.',
  enable_semantic_expansion:
    'Whether the semantic expansion stage runs at all. When off, retrieval is limited to entity anchoring + graph traversal only (no second-round vector search from traversal results) — isolating the contribution of the semantic expansion stage itself.',
  judge_condition:
    'Which condition\'s (A/B/C/D) result file to evaluate. Determines the judging mode: Condition A has no retrieved context (faithfulness is not applicable), while B/C/D are judged against the exact context they retrieved.',
  judge_input_file:
    'The result .jsonl file (already-completed run) to evaluate. Narrow it down by provider, model, n_sample, and seed below — the picker resolves to a single file once those identify it uniquely.',
  judge_input_provider: 'Filter available result files by which LLM provider generated them.',
  judge_input_model: 'Filter available result files by which model generated them.',
  judge_input_n_sample: 'Filter available result files by how many questions that run evaluated.',
  judge_input_seed: 'Filter available result files by the sampling seed that run used.',
  judge_provider: 'Which LLM API to send judging requests to. Independent of the provider/model that generated the answers being judged.',
  judge_model: 'The specific LLM model used as the judge, e.g. gpt-4o-mini. Reuses the API key configured for this provider in Settings.',
  judge_temperature:
    'Sampling temperature for the judge\'s own calls (separate from the generator\'s config). Low values (e.g. 0.1) make judging more consistent/deterministic run to run.',
  judge_majority_rounds:
    'Number of independent judging rounds per question, taking the majority-vote label and median scores across rounds (variation comes from the LLM\'s own randomness at the same temperature, not from raising temperature). Recommended >1 ONLY for a Cohen\'s Kappa validation subsample, not the full batch — API cost scales linearly with it.',
  kappa_validation:
    'Re-judges a random subsample with a second ("secondary") judge and computes Cohen\'s Kappa agreement between the two — validates that the primary judge\'s labels are reliable rather than idiosyncratic to one model.',
  secondary_judge_provider: 'Which LLM API the secondary (validation) judge uses. Should ideally differ from the primary judge to actually test for self-enhancement bias.',
  secondary_judge_model: 'The specific model used as the secondary judge for Cohen\'s Kappa validation.',
  judge_force:
    'Re-evaluate every question from scratch for this exact source file + judge config, discarding any existing results for it. Without this, a run resumes from where a previous one left off (or does nothing at all if it already fully completed).',
  kappa_sample_size:
    'How many already-judged questions to re-judge with the secondary judge for the Kappa comparison. Kept small by default since this doubles (or more, with majority rounds) the API cost for just that subsample.',
  api_key: 'Your OpenAI API key, used whenever provider is set to openai (generator runs and, separately, an openai judge).',
  anthropic_api_key: 'Your Anthropic API key (from console.anthropic.com, not a Claude.ai subscription), used whenever provider is set to anthropic.',
  ollama_host: 'The local Ollama server address (default http://localhost:11434) — dev/testing provider only, not used for final report results.',
  neo4j_uri: 'Bolt connection URI for the Neo4j knowledge graph, e.g. bolt://localhost:7687.',
  neo4j_user: 'Neo4j database username.',
  neo4j_password: 'Neo4j database password. Stored encrypted outside the repo, never displayed again once set.',
  neo4j_database: 'Name of the Neo4j database to connect to (Condition C/D and the master-data browsing pages all query this database).',
  questions_parquet: 'Filesystem path to the merged Questions Parquet file (SORD dataset) that every condition samples its evaluation questions from.',
  answers_parquet: 'Filesystem path to the merged Answers Parquet file (SORD dataset), used to look up accepted answers and retrieval corpora.',
} as const
