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
} as const
