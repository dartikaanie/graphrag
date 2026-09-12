export type MetricCategory =
  | 'Semantic Quality'
  | 'Citation & Grounding'
  | 'Hallucination & Faithfulness'
  | 'Retrieval Quality'
  | 'Efficiency'

export interface EvaluationMetric {
  id: string
  name: string
  category: MetricCategory
  appliesTo: ('A' | 'B' | 'C' | 'D')[]
  whatItMeasures: string
  howComputed: string
  formula?: string
  interpretationGuide: string
  limitations: string
  implementedIn: string[]
  relatedTo?: string[]
  /** Not a per-condition output metric -- a process-validation check on
   * the judge itself (inter-rater reliability), shown with a distinct
   * visual treatment on the reference page. */
  isValidationMetric?: boolean
}

/**
 * Single source of truth for every evaluation metric actually implemented
 * in this repo, rendered on /docs/metrics (MetricsReferencePage). Kept as
 * data, not JSX, so the content can be reviewed/edited independently of
 * the page's rendering logic.
 *
 * Every entry here is verified against the CURRENT implementation (not
 * design intent) as of this writing -- see the `implementedIn` paths for
 * exactly where each metric's logic lives.
 */
export const EVALUATION_METRICS: EvaluationMetric[] = [
  {
    id: 'cosine-similarity',
    name: 'Cosine Similarity',
    category: 'Semantic Quality',
    appliesTo: ['A', 'B', 'C', 'D'],
    whatItMeasures:
      'How semantically close the LLM\'s answer is to the accepted (ground-truth) Stack Overflow answer for that question.',
    howComputed:
      'Both the LLM answer and the accepted answer (AcceptedAnswerBody) are embedded with the same sentence-transformers model (all-MiniLM-L6-v2), and cosine similarity is computed between the two embedding vectors. Identical for all four conditions — this is the one metric every condition produces, since it is the baseline metric being improved on. Aggregated per run as mean, median, and % of questions scoring above 0.5.',
    formula: 'cosine_similarity = cos_sim(embed(llm_answer), embed(accepted_answer_body))',
    interpretationGuide:
      '0.0–1.0, higher is better. In practice, scores cluster well below 1.0 even for good answers (paraphrasing, different code style, extra explanation). Compare relative differences between conditions/runs on the same sample rather than treating any single absolute value as a hard pass/fail threshold.',
    limitations:
      'This is a semantic-similarity proxy, NOT a measure of factual correctness. A correct answer that is phrased very differently from the accepted answer (different wording, different but equally valid code) can score low; a wrong answer that reuses a lot of the same vocabulary as the accepted answer can score misleadingly high. It also cannot detect whether an answer is complete, safe, or up to date — only how close its wording/meaning is to one specific accepted answer.',
    implementedIn: [
      'llm/a_pure_llm/a_baseline_replication.py',
      'llm/b_rag/b_condition_b_rag.py',
      'llm/c_graphrag/c_graphrag.py',
      'llm/d_lightrag/d_lightrag.py',
    ],
    relatedTo: ['answer-relevance-score'],
  },
  {
    id: 'nf2-citation-validity',
    name: 'NF2 — Citation Validity',
    category: 'Citation & Grounding',
    appliesTo: ['B', 'C', 'D'],
    whatItMeasures:
      'Whether the LLM actually cites its sources using the [SO-<id>] format, and — more strictly — whether the ids it cites are real sources that were actually given to it in the retrieved context (rather than invented/hallucinated ids).',
    howComputed:
      'A regex (CITATION_PATTERN) scans the LLM answer for citation-like tokens, tolerating minor format variation ([SO-1234], [SO:1234], [SO 1234], [SO thread 1234]) — this loose match produces has_citation/cited_source_ids. Separately, each matched id is cross-checked against the question_id set actually present in that answer\'s retrieved_context; only ids that are really there count toward has_valid_citation/valid_cited_source_ids. Shared by Condition B (only when --require-citation is on), C, and D — the exact same function is reused by all three so the comparison is apples-to-apples.',
    interpretationGuide:
      '% of answers with has_valid_citation = true, aggregated per run. Target for the thesis is 100% (every answer cites at least one real source) for conditions where citation is required. A gap between the loose has_citation % and the strict has_valid_citation % specifically indicates hallucinated citation ids (right format, wrong/nonexistent source).',
    limitations:
      'Condition A produces no retrieved context at all, so this metric does not apply to it in the same sense — a "citation" from Condition A would necessarily reference nothing real, which is a different failure mode (fabrication from a blank slate) than B/C/D\'s failure mode (fabricating on top of real-but-unused context). That distinction is tracked separately as a "Fabricated Citation Rate" framing rather than as this metric. Also, the regex only detects the specific [SO-<id>] format this project defines — a model could cite sources in free text without matching the pattern, which would be undercounted here as "no citation" even if it named a real source in words.',
    implementedIn: ['llm/citations.py', 'llm/prompts.py'],
    relatedTo: ['faithfulness-score'],
  },
  {
    id: 'nf3-retrieval-latency',
    name: 'NF3 — Retrieval Latency',
    category: 'Efficiency',
    appliesTo: ['C', 'D'],
    whatItMeasures:
      'How long the retrieval step (before the LLM is even called) takes per question — a non-functional requirement specifically for the graph-based conditions, since their retrieval involves multiple sequential Neo4j/FAISS round-trips instead of one flat vector search.',
    howComputed:
      'Timed from immediately before entity anchoring/low-level retrieval starts to immediately after the fused context is ready (right before the prompt is built) — via time.time() deltas around the retrieval stages only. Explicitly excludes LLM generation time; it measures the graph/vector retrieval pipeline in isolation. Aggregated per run as mean and p95.',
    formula: 'retrieval_latency_sec = t_after_fusion - t_before_anchoring  (LLM call time NOT included)',
    interpretationGuide:
      'Target (NF3): average ≤15 seconds per question, checked against both the run-level average and p95. A per-question warning is logged if any single question exceeds 15s, even if the run average still passes.',
    limitations:
      'Measured on whatever hardware/network conditions the run happened to execute under (local Neo4j instance, local FAISS index) — not a controlled benchmark, so absolute numbers are not directly comparable across machines. It also only covers retrieval, not the LLM\'s response time, so it says nothing about total end-to-end answer latency a real user would experience.',
    implementedIn: ['llm/c_graphrag/c_graphrag.py', 'llm/d_lightrag/d_lightrag.py'],
  },
  {
    id: 'hallucination-rate',
    name: 'Hallucination Rate (3-class)',
    category: 'Hallucination & Faithfulness',
    appliesTo: ['A', 'B', 'C', 'D'],
    whatItMeasures:
      'A post-hoc judgment of how much of the LLM\'s answer is factually sound versus fabricated or contradictory, expressed as one of three categorical labels rather than a single blended score.',
    howComputed:
      'Computed entirely separately from the generator run, by llm/evaluation/llm_judge_hallucination.py reading an already-completed result file and sending each question to a second LLM acting as an impartial judge (never told which condition/provider/model produced the answer being judged, to avoid identity-based bias). Two judging modes are selected automatically from --condition: "no_context" for Condition A (no retrieved_context exists, so the judge falls back to its own general knowledge plus the reference answer), and "context_grounded" for B/C/D (the judge checks the answer against the exact retrieved_context that generator run actually used). The label is FAKTUAL, HALUSINASI_SEBAGIAN, or HALUSINASI_PENUH. When --majority-rounds > 1, the judge is called that many times at the same temperature and the majority-vote label is kept (not an average).',
    interpretationGuide:
      '% distribution across the three labels per run (pct_faktual / pct_halusinasi_sebagian / pct_halusinasi_penuh). Higher %FAKTUAL and lower %HALUSINASI_PENUH is better. Because mode differs by condition (A vs B/C/D), only compare hallucination rate WITHIN the same mode meaningfully, or note the mode difference explicitly when comparing across it.',
    limitations:
      'This is an LLM\'s subjective judgment, not ground truth — it inherits whatever biases the judge model has, including self-enhancement bias (a judge may rate answers from a similar model/family more favorably) and sensitivity to phrasing/position. This is exactly why a secondary judge + Cohen\'s Kappa validation exists: a single judge\'s labels should not be trusted at face value without checking inter-rater agreement (see the Cohen\'s Kappa entry).',
    implementedIn: ['llm/evaluation/llm_judge_hallucination.py'],
    relatedTo: ['faithfulness-score', 'answer-relevance-score', 'cohens-kappa'],
  },
  {
    id: 'faithfulness-score',
    name: 'Faithfulness Score',
    category: 'Hallucination & Faithfulness',
    appliesTo: ['B', 'C', 'D'],
    whatItMeasures:
      'The fraction of factual claims in the LLM\'s answer that are actually supported by the context it was given — a groundedness check, independent of whether those claims happen to also match the ground-truth answer.',
    howComputed:
      'The same LLM-as-judge call that produces the hallucination label also returns faithfulness_score (0.0–1.0) in "context_grounded" mode: the judge is instructed to check every claim in the answer against the given retrieved_context and report the fraction that is supported. When --majority-rounds > 1, the median across rounds is kept (not the mean).',
    formula: 'faithfulness_score ≈ (# claims supported by retrieved_context) / (total # claims in the answer)',
    interpretationGuide:
      '0.0–1.0, higher is better; 1.0 means every claim in the answer is traceable to the given context. Mean faithfulness_score is reported per run.',
    limitations:
      'Not applicable to Condition A by definition — there is no retrieved_context for A to be faithful to, so this field is always null for Condition A rather than a low score (a null result is a different thing from a bad result and must not be treated as 0). It is also possible for an answer to be highly faithful to its context yet still wrong, if the retrieved context itself was misleading or irrelevant — faithfulness measures grounding, not correctness.',
    implementedIn: ['llm/evaluation/llm_judge_hallucination.py'],
    relatedTo: ['nf2-citation-validity', 'hallucination-rate'],
  },
  {
    id: 'answer-relevance-score',
    name: 'Answer Relevance Score',
    category: 'Hallucination & Faithfulness',
    appliesTo: ['A', 'B', 'C', 'D'],
    whatItMeasures:
      'How directly and completely the answer addresses what the question actually asked — separate from whether the content is factually correct.',
    howComputed:
      'Returned by the same LLM-as-judge call (both judging modes) as answer_relevance_score (0.0–1.0), evaluated regardless of the factual-correctness assessment. Mean answer_relevance_score is reported per run.',
    interpretationGuide:
      '0.0–1.0, higher is better. A low relevance score with a high faithfulness/FAKTUAL label would flag an answer that is accurate but off-topic or incomplete relative to what was actually asked.',
    limitations:
      'Like the hallucination label, this is a judge LLM\'s subjective call, not a deterministic measurement — it is exposed to the same judge-model biases described under Hallucination Rate.',
    implementedIn: ['llm/evaluation/llm_judge_hallucination.py'],
    relatedTo: ['cosine-similarity', 'hallucination-rate'],
  },
  {
    id: 'cohens-kappa',
    name: "Cohen's Kappa (inter-judge agreement)",
    category: 'Hallucination & Faithfulness',
    appliesTo: ['A', 'B', 'C', 'D'],
    isValidationMetric: true,
    whatItMeasures:
      'Whether the primary judge\'s hallucination-label verdicts are reliable, by checking how much a second, independent judge agrees with them on the same questions — a validation check ON the judging process itself, not a metric of any one condition\'s answer quality.',
    howComputed:
      'A random subsample (fixed seed) of already-judged questions is re-judged by a "secondary" judge (a different provider/model, ideally), and Cohen\'s Kappa is computed between the two judges\' hallucination_label values via sklearn\'s cohen_kappa_score over the 3-class label set. If the primary and secondary judge are configured with the identical provider+model, a warning is printed explicitly, since that setup cannot detect self-enhancement bias — it only measures the judge\'s own run-to-run variance.',
    formula: 'κ computed via sklearn.metrics.cohen_kappa_score(primary_labels, secondary_labels, labels=[FAKTUAL, HALUSINASI_SEBAGIAN, HALUSINASI_PENUH])',
    interpretationGuide:
      'Interpreted with 4 bands (a simplified collapse of the original 6-band Landis & Koch 1977 scale, used as-is per this project\'s convention): < 0.4 weak, 0.4–0.6 moderate, 0.6–0.8 substantial, > 0.8 almost perfect agreement.',
    limitations:
      'If the primary and secondary judge are the same model, a high kappa only shows the judge is internally consistent with itself — it says nothing about whether its judgments are actually correct or free of self-enhancement bias. Kappa also only validates the categorical hallucination_label, not the numeric faithfulness/relevance scores, and is computed on a small random subsample (kappa_sample_size, default 50) rather than the full dataset, for cost reasons — it is a spot-check, not exhaustive validation.',
    implementedIn: ['llm/evaluation/llm_judge_hallucination.py'],
    relatedTo: ['hallucination-rate'],
  },
  {
    id: 'precision-at-k',
    name: 'Precision@k',
    category: 'Retrieval Quality',
    appliesTo: ['B', 'C', 'D'],
    whatItMeasures:
      'Of the items actually retrieved for a question (up to top-k), what fraction are judged relevant to that question.',
    howComputed:
      'Computed entirely post-hoc from an already-completed result file\'s retrieved_context field, via analyze_retrieval_quality.py — no new retrieval or LLM calls. Relevance is a PROXY: an item is considered relevant if the Jaccard tag overlap between the evaluation question\'s tags and the retrieved item\'s source question\'s tags exceeds --tag-overlap-threshold (default 0.0, i.e. at least one shared tag). Precision is computed over the number of items ACTUALLY retrieved for that question (which can be less than top-k if retrieval came back short), not a fixed denominator.',
    formula: 'precision@k = (# retrieved items judged relevant) / (# items actually retrieved, ≤ k)',
    interpretationGuide:
      '0.0–1.0, higher is better, reported as mean/median per result file. A precision noticeably below what you\'d expect from the tag-overlap baseline may indicate the retrieval mechanism is surfacing off-topic context.',
    limitations:
      'Relevance here is a TAG-OVERLAP PROXY, not a human relevance judgment — two questions can share tags without being substantively related, or be substantively related without sharing any tag, so this number should be read as a rough signal, not ground truth. It also only evaluates within the top-k that was actually returned; it says nothing about relevant items that existed but were never retrieved at all (that is what Recall@k would measure, and Recall@k is explicitly NOT implemented yet — see analyze_retrieval_quality.py\'s docstring for why).',
    implementedIn: ['analyze_retrieval_quality.py'],
    relatedTo: ['mrr-at-k'],
  },
  {
    id: 'mrr-at-k',
    name: 'MRR@k (Mean Reciprocal Rank)',
    category: 'Retrieval Quality',
    appliesTo: ['B', 'C', 'D'],
    whatItMeasures:
      'How near the top of the ranked retrieved list the FIRST relevant item appears — rewards retrieval that surfaces a good item early, not just eventually.',
    howComputed:
      'Also computed post-hoc by analyze_retrieval_quality.py from retrieved_context, using the SAME tag-overlap relevance proxy as Precision@k. Walks the retrieved_context list in its original ranked order (never re-sorted) and takes the reciprocal of the rank (1-indexed) of the first item judged relevant; 0.0 if none of the retrieved items are relevant.',
    formula: 'mrr@k = 1 / rank_of_first_relevant_item   (0.0 if no relevant item was retrieved at all)',
    interpretationGuide:
      '0.0–1.0, higher is better. A value of 1.0 means the very first retrieved item was already relevant; a low value with high Precision@k would suggest relevant items are being retrieved but ranked poorly.',
    limitations:
      'Same tag-overlap-proxy caveat as Precision@k applies — relevance is not human-verified. MRR also only cares about the FIRST relevant hit and ignores everything about the rest of the ranking, so two very differently-ranked result sets can have identical MRR if their first relevant item lands at the same rank.',
    implementedIn: ['analyze_retrieval_quality.py'],
    relatedTo: ['precision-at-k'],
  },
]
