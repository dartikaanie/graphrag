import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Badge } from '@/components/Badge'

interface ConditionSpec {
  key: 'A' | 'B' | 'C' | 'D'
  title: string
  tagline: string
  color: string
  overview: string
  retrieval: { heading: string; body: string }[]
  prompting: string
  leakage: string
  metrics: string[]
}

const CONDITIONS: ConditionSpec[] = [
  {
    key: 'A',
    title: 'Condition A — Pure LLM Baseline',
    tagline: 'No retrieval at all — the LLM answers from parametric knowledge only.',
    color: 'border-l-text-muted',
    overview:
      'Replicates the baseline methodology from Da Silva, Samhi & Khomh (2025), "LLMs and Stack Overflow discussions: Reliability, impact, and challenges," applied to the SORD dataset. This is the control condition: a single LLM call per question with no external context of any kind, establishing the floor that Conditions B, C, and D are measured against.',
    retrieval: [],
    prompting:
      'A 4-turn chat structure replicated verbatim from the original paper\'s get_base_message(): (1) system — general software-engineering persona, (2) user — a persona statement built from the question\'s tags, (3) assistant — the model priming/confirming that persona, (4) user — the question title + body. No retrieved context is inserted anywhere. This exact 4-turn skeleton is reused unchanged by B, C, and D, so the only variable across conditions is the presence, source, and constraints of retrieved context.',
    leakage:
      'Not applicable — there is no retrieval corpus to leak from. The accepted answer is only used afterward, as the ground truth the LLM\'s answer is scored against.',
    metrics: ['Cosine similarity (all-MiniLM-L6-v2) vs. the accepted answer'],
  },
  {
    key: 'B',
    title: 'Condition B — Conventional Dense RAG',
    tagline: 'Flat FAISS retrieval over a chunked corpus of Stack Overflow answers, no graph structure.',
    color: 'border-l-primary',
    overview:
      'Condition A plus a standard dense-retrieval RAG pipeline: a flat, un-structured corpus of (question title + answer body) pairs indexed with FAISS, retrieved top-k per evaluation question and inserted into the prompt. This is the industry-standard RAG baseline that Condition C\'s graph-based retrieval is benchmarked against — same LLM, same sampling, same base prompt skeleton, different retrieval mechanism only.',
    retrieval: [
      {
        heading: 'Corpus construction',
        body: 'Every (Question.Title + Answer.Body) pair in the candidate pool is treated as one document, independent of any other document — there is no graph, no edges, no relationships between documents. Documents are chunked to a fixed token limit (tiktoken cl100k_base) with no overlap.',
      },
      {
        heading: 'Indexing & retrieval',
        body: 'Chunks are embedded with all-MiniLM-L6-v2 and indexed with FAISS IndexFlatIP over normalized vectors (cosine similarity). For each evaluation question, the top-k nearest chunks are retrieved by a single vector search — one retrieval pass, no traversal, no multi-hop reasoning. The index is cached to disk so it isn\'t rebuilt on every run.',
      },
    ],
    prompting:
      'Same 4-turn skeleton as Condition A, with retrieved chunks inserted at the start of the final user turn. Citation is optional and off by default in the methodology sense — it can be toggled on (require_citation) to label each chunk "[SO-<id>]" and instruct the model to cite it, using the exact same citation format and validation logic as Condition C, so NF2 (citation compliance) can be compared B vs. C on equal footing. Critically, Condition B never includes a "ground your answer only in the given context" instruction even when citation is on — that grounding constraint is what distinguishes Condition C\'s (and optionally D\'s) prompting.',
    leakage:
      'Prevented at index build time: the accepted answer for every evaluation question, and every other (non-accepted) answer belonging to any evaluation question, is explicitly excluded from the retrieval corpus before the FAISS index is even built. Retrieval can never rediscover the answer it is being asked to predict.',
    metrics: [
      'Cosine similarity vs. the accepted answer',
      'Retrieval latency',
      'NF2 — citation compliance (only when require_citation is on)',
    ],
  },
  {
    key: 'C',
    title: 'Condition C — GraphRAG (trust-weighted Knowledge Graph)',
    tagline: 'The thesis\'s core contribution: hybrid retrieval over a trust-weighted Neo4j knowledge graph, with dual-constraint grounded generation.',
    color: 'border-l-success',
    overview:
      'Condition A\'s sampling and base prompt, plus a three-stage hybrid retrieval pipeline over a Knowledge Graph built from the same SORD community data — Question/Answer/Tag/User nodes connected by trust-weighted edges (HAS_ACCEPTED_ANSWER, HAS_ANSWER, TAGGED_WITH, IS_RELATED_TO, TAG_COOCCUR, EMBED_SIM, AUTHOR_TRUST). Retrieval is not a single vector search — it is anchor → traverse → expand, fused into a ranked context under a dual grounding + citation constraint. This is the design being evaluated against B (does graph structure + trust weighting beat flat dense retrieval?).',
    retrieval: [
      {
        heading: 'a. Entity anchoring',
        body: 'A FAISS vector search (reusing the same index built during graph densification) over the evaluation question\'s embedding finds the top-N_ANCHOR most similar Question nodes already in the graph — these become entry points into the KG. The evaluation question\'s own id is excluded from the results (leakage prevention #1).',
      },
      {
        heading: 'b. Graph traversal',
        body: 'From each anchor, 1-hop (HAS_ACCEPTED_ANSWER / HAS_ANSWER → the anchor\'s own answers) and 2-hop (IS_RELATED_TO / TAG_COOCCUR / EMBED_SIM → a related Question → its answers) edges are followed, collecting Answer nodes together with the trust weight of the path that reached them.',
      },
      {
        heading: 'c. Semantic expansion',
        body: 'Up to 3 Question nodes discovered during traversal are used as new anchors for a second vector-search pass, catching semantically relevant content that has no explicit graph edge to the original anchors. Optional — can be disabled entirely (ablation) to isolate its contribution.',
      },
      {
        heading: 'Fusion & ranking',
        body: 'All candidates from (b) and (c) are deduplicated by answer id and ranked by a combined trust score: 0.7 × path trust (edge weight) + 0.3 × the answer\'s own intrinsic trust score, by default. An ablation switch (fusion_mode) can instead rank purely by discovery order ("uniform"), isolating exactly how much of Condition C\'s effect comes from trust-weighting itself versus the graph structure.',
      },
    ],
    prompting:
      'Same 4-turn skeleton as A/B, with each context item labeled "[SO-<question_id>]" (not a generic "Reference N") plus transparency metadata (trust weight, hop count). A dual constraint is added by default: (a) RAG grounding — the answer must be based only on the given context, no invented claims, and (b) citation — every factual claim must cite its "[SO-<id>]" source. The grounding half of this constraint is itself an ablation toggle (require_grounding): turning it off keeps the citation instruction but drops the "don\'t invent facts" rule, isolating the grounding constraint\'s own contribution from the citation constraint\'s.',
    leakage:
      'Handled dynamically at every query, not once at build time, because the KG is one shared graph containing the evaluation questions themselves as ordinary nodes. Every anchor search, every traversal query, and every expansion search explicitly excludes the evaluation question\'s own id and all of its answer ids (accepted and otherwise) from the results — at every stage, not just the final one.',
    metrics: [
      'Cosine similarity vs. the accepted answer',
      'Retrieval latency (target ≤15s)',
      'NF2 — citation compliance, validated against the actual retrieved context (not just format)',
    ],
  },
  {
    key: 'D',
    title: 'Condition D — Dual-Level Retrieval (LightRAG-adapted)',
    tagline: 'Same knowledge graph and data as Condition C, but a different retrieval mechanism entirely: dual-level (low-level + high-level), no trust weighting.',
    color: 'border-l-warning',
    overview:
      'An adaptation of the dual-level retrieval mechanism from LightRAG (Guo et al.) — deliberately NOT a reimplementation of LightRAG\'s own entity-extraction pipeline. Condition D reads the exact same Neo4j knowledge graph and FAISS cache that Condition C uses, as-is: no new graph is built, no LLM-based entity extraction is performed. This is a controlled methodological choice so that Condition C vs. D isolates the retrieval MECHANISM (trust-weighted fusion vs. dual-level, trust-free ranking) as the only variable, not the underlying data or graph.',
    retrieval: [
      {
        heading: 'Low-level retrieval',
        body: 'Identical entity anchoring to Condition C (FAISS vector search → top-N_LOW_LEVEL Question anchors), then a single 1-hop lookup of each anchor\'s directly attached answers. The relevance score used for ranking is the anchor\'s own cosine similarity from the vector search — never an edge weight or trust score.',
      },
      {
        heading: 'High-level retrieval',
        body: 'From the same anchors, a 2-hop traversal via TAG_COOCCUR / IS_RELATED_TO edges reaches thematically related questions (capped to the top-N_HIGH_LEVEL by raw edge weight), and their answers are collected. The relevance score is the raw edge weight itself (a similarity/Jaccard-style score between two questions) — it is never combined with an answer\'s trust score the way Condition C\'s traversal is.',
      },
      {
        heading: 'Fusion',
        body: 'Low-level and high-level candidates are deduplicated by answer id (keeping the highest relevance score seen), labeled with their source_stage ("low_level"/"high_level") as transparency metadata only, sorted by relevance score, and capped to top-k. No trust score of any kind participates anywhere in this pipeline.',
      },
    ],
    prompting:
      'The same 4-turn skeleton and "[SO-<id>]" labeling as Condition C, but context metadata shows source_stage (low_level/high_level) instead of trust weight/hop, since Condition D has no trust score to show. Grounding is the same optional toggle as Condition C (require_grounding, default on) — dual grounding+citation constraint when on, citation-only when off — letting the grounding-constraint ablation be run identically across both graph-based conditions.',
    leakage:
      'Identical exclusion discipline to Condition C: the evaluation question\'s own id and all of its answer ids are excluded from every anchor search and every traversal query, at every stage.',
    metrics: [
      'Cosine similarity vs. the accepted answer',
      'Retrieval latency',
      'NF2 — citation compliance, validated against the actual retrieved context',
    ],
  },
]

interface ComparisonRow {
  label: string
  a: string
  b: string
  c: string
  d: string
}

const COMPARISON_ROWS: ComparisonRow[] = [
  { label: 'Retrieval', a: 'None', b: 'Flat dense vector search (FAISS)', c: 'Hybrid: anchor → graph traversal → semantic expansion', d: 'Dual-level: low-level (1-hop) + high-level (2-hop)' },
  { label: 'Data structure', a: '—', b: 'Flat chunked corpus, no relationships', c: 'Trust-weighted Neo4j knowledge graph', d: 'Same knowledge graph as C, read-only, no new extraction' },
  { label: 'Ranking signal', a: '—', b: 'Cosine similarity only', c: 'Combined trust score (0.7 × path trust + 0.3 × intrinsic trust), or uniform (ablation)', d: 'Pure relevance score — anchor similarity (low) / edge weight (high), never trust' },
  { label: 'Trust weighting', a: 'No', b: 'No', c: 'Yes (default) — ablatable via fusion_mode', d: 'No, by design (isolates the mechanism from trust)' },
  { label: 'Grounding constraint', a: 'N/A', b: 'Never present', c: 'On by default — ablatable (require_grounding)', d: 'On by default — ablatable (require_grounding)' },
  { label: 'Citation constraint', a: 'N/A', b: 'Optional (require_citation)', c: 'Always on when context exists', d: 'Always on when context exists' },
  { label: 'Semantic expansion stage', a: 'N/A', b: 'N/A', c: 'Yes — optional, ablatable (enable_semantic_expansion)', d: 'Not implemented (only low-level + high-level)' },
  { label: 'Leakage prevention', a: 'N/A', b: 'Static exclusion at index build time', c: 'Dynamic exclusion at every query (shared graph)', d: 'Dynamic exclusion at every query (shared graph)' },
  { label: 'Key metrics', a: 'Cosine similarity', b: 'Cosine similarity, retrieval latency, NF2 (optional)', c: 'Cosine similarity, retrieval latency, NF2', d: 'Cosine similarity, retrieval latency, NF2' },
  { label: 'Script', a: 'llm/a_pure_llm/a_baseline_replication.py', b: 'llm/b_rag/b_condition_b_rag.py', c: 'llm/c_graphrag/c_graphrag.py', d: 'llm/d_lightrag/d_lightrag.py' },
]

export function MethodologyPage() {
  const [activeKey, setActiveKey] = useState<ConditionSpec['key']>('A')
  const active = CONDITIONS.find((c) => c.key === activeKey) ?? CONDITIONS[0]

  return (
    <div className="max-w-4xl">
      <h1 className="text-lg font-semibold text-text-primary mb-1">Methodology — Conditions A, B, C &amp; D</h1>
      <p className="text-sm text-text-secondary mb-6">
        All four conditions share the same evaluation protocol — question sampling, the base 4-turn prompt skeleton
        (<code className="text-xs bg-bg border border-border rounded px-1 py-0.5">llm/prompts.py</code>), the same embedding
        model for scoring, and the same LLM provider factory — so that retrieval mechanism and prompting constraint are the
        only variables under comparison. See{' '}
        <Link to="/experiment/a" className="text-primary hover:underline">
          Run
        </Link>{' '}
        pages to execute a condition, or{' '}
        <Link to="/history" className="text-primary hover:underline">
          History
        </Link>{' '}
        to inspect past results.
      </p>

      <div className="flex gap-1 mb-4 border-b border-border">
        {CONDITIONS.map((c) => (
          <button
            key={c.key}
            onClick={() => setActiveKey(c.key)}
            className={`px-4 py-2 text-sm border-b-2 -mb-px ${
              activeKey === c.key ? 'border-primary text-primary font-medium' : 'border-transparent text-text-secondary'
            }`}
          >
            {c.key}
          </button>
        ))}
      </div>

      <div className={`border-l-4 ${active.color} border-y border-r border-border rounded-r-lg bg-surface p-5 mb-10`}>
        <div className="flex items-center gap-2 mb-1">
          <Badge tone="primary">{active.key}</Badge>
          <h2 className="text-base font-semibold text-text-primary">{active.title}</h2>
        </div>
        <p className="text-sm text-text-secondary italic mb-3">{active.tagline}</p>
        <p className="text-sm text-text-primary mb-3">{active.overview}</p>

        {active.retrieval.length > 0 && (
          <div className="mb-3">
            <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1.5">Retrieval design</h3>
            <div className="flex flex-col gap-2">
              {active.retrieval.map((r) => (
                <div key={r.heading} className="text-sm">
                  <span className="font-medium text-text-primary">{r.heading}: </span>
                  <span className="text-text-secondary">{r.body}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="mb-3">
          <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1.5">Prompting</h3>
          <p className="text-sm text-text-secondary">{active.prompting}</p>
        </div>

        <div className="mb-3">
          <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1.5">Leakage prevention</h3>
          <p className="text-sm text-text-secondary">{active.leakage}</p>
        </div>

        <div>
          <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1.5">Metrics measured</h3>
          <div className="flex flex-wrap gap-1.5">
            {active.metrics.map((m) => (
              <Badge key={m} tone="neutral">
                {m}
              </Badge>
            ))}
          </div>
        </div>
      </div>

      <h2 className="text-sm font-semibold text-text-primary mb-2">Comparison summary</h2>
      <div className="overflow-x-auto border border-border rounded-lg mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-bg border-b border-border-strong">
              <th className="text-left font-medium text-text-secondary px-4 py-2.5 align-top">Aspect</th>
              <th className="text-left font-medium text-text-secondary px-4 py-2.5 align-top">A — Pure LLM</th>
              <th className="text-left font-medium text-text-secondary px-4 py-2.5 align-top">B — Dense RAG</th>
              <th className="text-left font-medium text-text-secondary px-4 py-2.5 align-top">C — GraphRAG</th>
              <th className="text-left font-medium text-text-secondary px-4 py-2.5 align-top">D — Dual-Level</th>
            </tr>
          </thead>
          <tbody>
            {COMPARISON_ROWS.map((row) => (
              <tr key={row.label} className="border-b border-border last:border-b-0 align-top">
                <td className="px-4 py-2.5 font-medium text-text-primary whitespace-nowrap">{row.label}</td>
                <td className="px-4 py-2.5 text-text-secondary">{row.a}</td>
                <td className="px-4 py-2.5 text-text-secondary">{row.b}</td>
                <td className="px-4 py-2.5 text-text-secondary">{row.c}</td>
                <td className="px-4 py-2.5 text-text-secondary">{row.d}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="text-xs text-text-muted border border-border rounded-lg bg-bg px-4 py-3">
        For the full implementation detail behind each design decision (leakage-exclusion queries, exact prompt text,
        CLI/ablation flags), see the docstring at the top of each condition's script listed in the table above, or{' '}
        <span className="font-mono">docs/README.md</span> in the repo.
      </div>
    </div>
  )
}
