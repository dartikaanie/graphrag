# Error analysis: where trust_weight acts in the GraphRAG pipeline, and what the hallucinations look like

Read-only analysis. No pipeline code was modified and no LLM/judge call was made. Everything is recomputed from stored files. Scripts: `scratchpad/ea.py`, `rep.py` (copy them into the repo if you want them versioned). Step 2 output: `reports/context_provenance_C.jsonl`.

## 0. Read this first (what could and could not be done)

1. **Only one configuration exists at full scale.** The only judged C run with n=384 is the official one (C v4, alpha=0.5, 2 hops, expansion on, trust in ranking). Judged **1-hop-only, 2-hop-only, no-expansion** runs do **not exist**; the `max_hops`/`enable_semantic_expansion` switches exist (`c_graphrag.py:518`, `:1167`) but were never run through the judge panel. "+/- trust" is available only as dev runs (n=50) via alpha: alpha=0.0 ranks by similarity only (`c_graphrag.py:1039`), alpha=0.25/0.5 include trust. I report those 6 dev runs instead and **do not invent** the missing configurations. Per-origin trust distributions are given as a substitute for hop-stratified numbers.
2. **Gold evidence cannot separate (b) from (c).** The leakage policy (`leakage: thread`) removes the question's own accepted answer and all its answers from retrieval, so the gold answer is absent from the context in **all** hallucinated answers by construction (0/70 official). "Gold answer not in context" is therefore always true and says nothing. The context-sufficiency judge was never run (`results/ch5/REPORT.md`, blocker 1). So **(b) and (c) are reported as one combined bucket "b|c"**, not split.
3. **No trust threshold is defined anywhere in the repo.** I use `trust_weight < 0.5` (weaker than a plain non-accepted direct answer, `HAS_ANSWER` weight 0.5, and below every expansion proxy, which is capped at 0.4). It is a parameter; sensitivity is in section 4.
4. **"Hallucinated"** = panel-v2 label HALUSINASI_PENUH or SEBAGIAN (3 judges, `aggregate_labels`). ABSTAIN is not hallucinated. Counts reproduce `results/ch5/hallucination_panel.csv` (C: 70). Claim evidence comes from the judges that voted hallucinated.
5. **There is no separate "seed" origin among context items.** Seeds are the anchor Questions; their answers enter as hop1. Origins in the log are therefore hop1 (answers of seed/anchor questions), hop2, expansion.

## 1. Step 1: where trust_weight is computed and used (`llm/c_graphrag/c_graphrag.py` unless noted)

| Stage | Uses trust? | Where | Detail |
|---|---|---|---|
| Graph build | computes it | `01_data_cleaning/7_build_knowledge_graph.py`, `13_apply_reputation_trust.py`, `14_apply_logscore_trust.py`, `trust_formulas.py` | Edge `weight` and node `trustScore` (version `v3-logscore`) stored in Neo4j. |
| **Seed (anchor) selection** | **NO** | `:488-508`, `llm/common/anchors.py:41` (`select_answerable_anchors`, called `:1196`) | Pure FAISS cosine of the question vs Question embeddings, plus an "answerable" filter. Trust is not read. |
| **Hop-1 fetch** | partly | `:551-563` | Reads `r.weight` as `edge_weight` and `a.trustScore`, but `ORDER BY a.id ASC` (`:560`): **no trust filter or ordering**; all answers of the anchors are fetched. |
| **Hop-2 edge scoring** | **YES** | `:574-591` | `edge_weight = r2.weight * r3.weight`, `ORDER BY edge_weight DESC ... LIMIT 50` (`:581-583`). This is the only place where trust prunes candidates during traversal. |
| **Seeds for expansion** | **NO** | `:715-721` | The first 3 distinct `via_question` in traversal order become expansion seeds (order = hop1 by answer_id, then hop2 by edge weight). No trust test. |
| **Expansion neighbour choice** | **NO** | `:625-683` (answerable path, `:647` sorts by cosine desc), `:731-740` | Neighbours chosen by cosine only. |
| **Expansion items' weight** | **proxy, not a measured weight** | `:752-756`, `:943-952` | Fetches `r.weight` of the answer edge, then `proxy_weight = min(edge_weight,1) * 0.4` (`:945`) and `combined_score = 0.7*proxy + 0.3*trustScore`. The stored `trust_weight` of every expansion item is therefore this synthetic value (max 0.4), **not empty**. |
| Fusion score | YES | `:933-952` | `combined_score = 0.7*min(edge_weight,1) + 0.3*answer trustScore` (weights `:179-180`). |
| Dedup | YES | `:1010-1014` | Keeps the highest `combined_score` per answer_id. |
| **Final rerank / top-5** | **YES, mixed with similarity** | `:1035-1042` (v4, alpha 0.5) | `final = (1-alpha)*sim_norm + alpha*combined_score`; sorted desc, cut to `top_k=5`. At alpha=0 trust has no say. `alpha_ranking` at `:211`. |
| Context ordering | follows the rerank | `:1046-1064` | Rank 1..5 = final_score order. |
| **LLM prompt** | **NO (deliberately)** | `llm/prompts.py:149-162` (`_format_context_block`), `:236-262` | The prompt shows only `[SO-<question_id>] <chunk_text>`. `trust_weight`, hop and stage are stored in the record but never shown. The model cannot weigh by trust. |

Stages that bypass trust: seed selection, hop-1 fetch, expansion seed choice, expansion neighbour choice, and the prompt. Trust acts in only three places: hop-2 pruning/LIMIT, the fusion score, and the final rerank.

## 2. Step 2: context provenance log

File: `reports/context_provenance_C.jsonl` (3,420 rows, 7 configurations: official n=384 plus 6 dev runs n=50, all with 5 items per question). One row per context item sent to the LLM, fields: `config, question_id, rank, answer_id, context_label_qid, origin (hop1|hop2|expansion), rel_type, trust_weight (empty is never the case; see below), trust_weight_is_expansion_proxy, trust_combined, sim, final_score, is_accepted`.

- `trust_weight` is **never empty** in the stored data. For expansion items it is the proxy described above (flagged by `trust_weight_is_expansion_proxy=true`). A real hop-1 weight of 0.0 exists in 8 official items.
- `sim` is the cosine of the question to the final chunk text (v4 records). Rank is the position in the prompt.
- Conditions A, B, D are not logged here (only C uses trust).

Official context composition: hop1 1,005 items (52%), hop2 533 (28%), expansion 382 (20%).


## 3. Step 3-4: Classification of hallucinated answers

### 3.1 Rules (deterministic, no LLM)

For each hallucinated answer, take the claims judged CONTRADICTED or FABRICATED by the judges that voted hallucinated, and map each claim's `cited_items` (K labels) to context items through the judge's `citation_map`.

- **a**: claim has `supported_by_context=YES`, the judge says the cited item supports it (`cited_item_supports=YES`), and that item has `trust_weight < 0.5`.
- **a'** (extra bucket): same but the supporting item has `trust_weight >= 0.5`. The context did contain the bad evidence, but it was not low-trust, so it is not (a), and forcing it into (b) or (c) would be wrong.
- **a-unattributed**: context supports the incorrect claim but the judge named no supporting item, so trust cannot be read.
- **b|c**: no context item supports the incorrect claim. (b) "evidence missing" vs (c) "evidence present but ignored" needs a gold-evidence or sufficiency label that does not exist (section 0, point 2).


### OFFICIAL a=0.5 grounded (n=384)

Hallucinated (panel-v2, includes SEBAGIAN): **70** of 384 questions. Gold accepted answer present in context: **0/70** (by design, see section 0, point 2).

| Label | Meaning | Count | % of hallucinated |
|---|---|---|---|
| a | (a) context supports the incorrect claim AND the supporting item has trust_weight < 0.5 | 6 | 8.6% |
| a-high | (a') context supports the incorrect claim but the supporting item has trust_weight >= 0.5 (outside a/b/c) | 8 | 11.4% |
| a-unattributed | context supports the incorrect claim but judge named no supporting item, so trust unknown | 6 | 8.6% |
| b|c | (b) or (c): no context item supports the claim; b vs c NOT separable (no sufficiency or gold-evidence label) | 50 | 71.4% |

trust_weight distribution of items sent to the LLM (min / median / max / std):

| Subset | n items | min / median / max / std |
|---|---|---|
| all context items | 1920 | 0.000 / 0.500 / 1.000 / 0.278 |
| origin=hop1 | 1005 | 0.000 / 1.000 / 1.000 / 0.254 |
| origin=hop2 | 533 | 0.133 / 0.400 / 1.000 / 0.131 |
| origin=expansion | 382 | 0.000 / 0.400 / 0.400 / 0.071 |

Origin of the offending low-trust item in (a): expansion=3, hop2=3

**(a) examples (6 total; first 5 by question_id order)**

| question_id | question | incorrect claim (verdict) | related context item (origin, rank, trust_weight) |
|---|---|---|---|
| 56998713 | Tomcat classpath loader javax.mail.Session conflict with spring-boot-s | Use Spring Boot's built-in features: configure SMTP settings in application.properties without needing javax.mail.jar in Tomcat (CONTRADICTED) [qwen3-235b] | SO-33213060 expansion, rank 5, tw=0.4, rel=HAS_ACCEPTED_ANSWER |
| 33988487 | Should I be using IEquatable to ease testing of factories? | Implementing IEquatable aligns with best practices for establishing clear and consistent equality logic, especially useful in collections or scenarios requiring equality  (CONTRADICTED) [deepseek] | SO-104158 hop2, rank 2, tw=0.4727, rel=IS_RELATED_TO->HAS_ANSWER |
| 14586569 | Where to store flat files in CakePHP? | This approach will ensure that the files are not directly accessible through a web URL. (CONTRADICTED) [deepseek] | SO-2339649 expansion, rank 3, tw=0.4, rel=HAS_ACCEPTED_ANSWER |
| 15465867 | Video Specs across iOS Platforms | The video technologies in iOS support MPEG-4 video up to 2.5 Mbps at 640 by 480 pixels and 30 frames per second, with AAC-LC audio up to 160 Kbps (48 kHz, stereo). (CONTRADICTED) [deepseek] | SO-1535836 hop2, rank 4, tw=0.4, rel=EMBED_SIM->HAS_ACCEPTED_ANSWER |
| 39312836 | Hamburgers and sidebarEffects Collision | The hamburger icon structure should have child elements (like `.bar1`, `.bar2`, `.bar3`) to apply transformations. (CONTRADICTED) [deepseek] | SO-66758345 hop2, rank 3, tw=0.4, rel=EMBED_SIM->HAS_ACCEPTED_ANSWER |

**(a') examples (8 total; first 5 by question_id order)**

| question_id | question | incorrect claim (verdict) | related context item (origin, rank, trust_weight) |
|---|---|---|---|
| 812945 | ASP.NET PageMethods and JQuery AJAX Post – How secure really is it? | Implement rate limiting and protection against brute force attacks, such as logging failed login attempts and temporarily blocking IP addresses that exceed a certain numb (CONTRADICTED) [deepseek] | SO-14140037 hop1, rank 2, tw=1.0, rel=HAS_ANSWER |
| 43809688 | authentication error when publishing from gradle to nexus | The issue is caused by using HTTP in the Nexus URL, leading to authentication problems (CONTRADICTED) [qwen3-235b] | SO-68370067 hop1, rank 1, tw=1.0, rel=HAS_ACCEPTED_ANSWER |
| 69053706 | Uncaught ReferenceError: firebase is not defined in Flutter | Firebase scripts should be included without type="module" in index.html (CONTRADICTED) [qwen3-235b] | SO-69796555 hop1, rank 2, tw=1.0, rel=HAS_ACCEPTED_ANSWER |
| 52387886 | Android NDK: Static or shared C++ runtime for 3rd-party Java library | It is generally recommended to use a shared library variant for your C++ runtime to avoid these conflicts, especially when multiple shared libraries are involved. (CONTRADICTED) [deepseek] | SO-38529945 hop1, rank 2, tw=1.0, rel=HAS_ACCEPTED_ANSWER |
| 26494671 | Export Google BigQuery data to Python Pandas dataframe | It is advisable to switch to the official Google BigQuery Client Library due to gbq being discontinued (CONTRADICTED) [qwen3-235b] | SO-59373681 hop1, rank 1, tw=0.5, rel=HAS_ANSWER |

**(b)/(c) examples (50 total; first 5 by question_id order)**

| question_id | question | incorrect claim (verdict) | related context item (origin, rank, trust_weight) |
|---|---|---|---|
| 76224221 | how to rephrase this code to get past clippy::manual_let_else | The suggested fix in the error message is valid Rust syntax that should compile correctly. (CONTRADICTED) [deepseek] | none named by judge (claim not supported by context) |
| 3882147 | How to refer to dataTable parent within the dataTable? | In the case of ui:repeat, you may want to format the render ID like so: render=":wrapperForm:blocksRepeat:#{blocksRepeat.index}:rowsWrapper". (CONTRADICTED) [deepseek] | none named by judge (claim not supported by context) |
| 51061836 | Making a security definition request to FIX adapter | In ToAdmin, you can check if msg.Header.GetString(Tags.MsgType) == "D" to ensure it is for your Security Definition Request. (CONTRADICTED) [deepseek] | none named by judge (claim not supported by context) |
| 58418666 | Razor Pages/ ASP.NET CORE Routing Issue on Deployed Site (404 Error) | If the Index page is not in the same controller as the OnGetAsync method, consider using RedirectToAction("Action", "Controller") to clarify to which action or page you’r (CONTRADICTED) [deepseek] | none named by judge (claim not supported by context) |
| 21164127 | Auto Creating a Table in SQL | Auto-creating a table for each user is a suitable approach (CONTRADICTED) [llama] | none named by judge (claim not supported by context) |

### DEV a=0.0 grounded (n=50)

Hallucinated (panel-v2, includes SEBAGIAN): **8** of 50 questions. Gold accepted answer present in context: **0/8** (by design, see section 0, point 2).

| Label | Meaning | Count | % of hallucinated |
|---|---|---|---|
| a | (a) context supports the incorrect claim AND the supporting item has trust_weight < 0.5 | 4 | 50.0% |
| b|c | (b) or (c): no context item supports the claim; b vs c NOT separable (no sufficiency or gold-evidence label) | 4 | 50.0% |

trust_weight distribution of items sent to the LLM (min / median / max / std):

| Subset | n items | min / median / max / std |
|---|---|---|
| all context items | 250 | 0.000 / 0.400 / 1.000 / 0.301 |
| origin=hop1 | 105 | 0.000 / 0.500 / 1.000 / 0.348 |
| origin=hop2 | 82 | 0.000 / 0.350 / 0.768 / 0.161 |
| origin=expansion | 63 | 0.000 / 0.200 / 0.400 / 0.119 |

Origin of the offending low-trust item in (a): hop2=2, expansion=2

### DEV a=0.0 ungrounded (n=50)

Hallucinated (panel-v2, includes SEBAGIAN): **13** of 50 questions. Gold accepted answer present in context: **0/13** (by design, see section 0, point 2).

| Label | Meaning | Count | % of hallucinated |
|---|---|---|---|
| a | (a) context supports the incorrect claim AND the supporting item has trust_weight < 0.5 | 3 | 23.1% |
| a-high | (a') context supports the incorrect claim but the supporting item has trust_weight >= 0.5 (outside a/b/c) | 1 | 7.7% |
| a-unattributed | context supports the incorrect claim but judge named no supporting item, so trust unknown | 2 | 15.4% |
| b|c | (b) or (c): no context item supports the claim; b vs c NOT separable (no sufficiency or gold-evidence label) | 7 | 53.8% |

trust_weight distribution of items sent to the LLM (min / median / max / std):

| Subset | n items | min / median / max / std |
|---|---|---|
| all context items | 250 | 0.000 / 0.400 / 1.000 / 0.301 |
| origin=hop1 | 105 | 0.000 / 0.500 / 1.000 / 0.348 |
| origin=hop2 | 82 | 0.000 / 0.350 / 0.768 / 0.161 |
| origin=expansion | 63 | 0.000 / 0.200 / 0.400 / 0.119 |

Origin of the offending low-trust item in (a): hop2=3

### DEV a=0.25 grounded (n=50)

Hallucinated (panel-v2, includes SEBAGIAN): **8** of 50 questions. Gold accepted answer present in context: **0/8** (by design, see section 0, point 2).

| Label | Meaning | Count | % of hallucinated |
|---|---|---|---|
| a | (a) context supports the incorrect claim AND the supporting item has trust_weight < 0.5 | 3 | 37.5% |
| b|c | (b) or (c): no context item supports the claim; b vs c NOT separable (no sufficiency or gold-evidence label) | 5 | 62.5% |

trust_weight distribution of items sent to the LLM (min / median / max / std):

| Subset | n items | min / median / max / std |
|---|---|---|
| all context items | 250 | 0.000 / 0.400 / 1.000 / 0.301 |
| origin=hop1 | 118 | 0.000 / 1.000 / 1.000 / 0.310 |
| origin=hop2 | 76 | 0.000 / 0.400 / 0.817 / 0.162 |
| origin=expansion | 56 | 0.200 / 0.400 / 0.400 / 0.093 |

Origin of the offending low-trust item in (a): hop2=3

### DEV a=0.25 ungrounded (n=50)

Hallucinated (panel-v2, includes SEBAGIAN): **13** of 50 questions. Gold accepted answer present in context: **0/13** (by design, see section 0, point 2).

| Label | Meaning | Count | % of hallucinated |
|---|---|---|---|
| a | (a) context supports the incorrect claim AND the supporting item has trust_weight < 0.5 | 3 | 23.1% |
| a-high | (a') context supports the incorrect claim but the supporting item has trust_weight >= 0.5 (outside a/b/c) | 1 | 7.7% |
| a-unattributed | context supports the incorrect claim but judge named no supporting item, so trust unknown | 1 | 7.7% |
| b|c | (b) or (c): no context item supports the claim; b vs c NOT separable (no sufficiency or gold-evidence label) | 8 | 61.5% |

trust_weight distribution of items sent to the LLM (min / median / max / std):

| Subset | n items | min / median / max / std |
|---|---|---|
| all context items | 250 | 0.000 / 0.400 / 1.000 / 0.301 |
| origin=hop1 | 118 | 0.000 / 1.000 / 1.000 / 0.310 |
| origin=hop2 | 76 | 0.000 / 0.400 / 0.817 / 0.162 |
| origin=expansion | 56 | 0.200 / 0.400 / 0.400 / 0.093 |

Origin of the offending low-trust item in (a): hop2=3

### DEV a=0.5 grounded (n=50)

Hallucinated (panel-v2, includes SEBAGIAN): **11** of 50 questions. Gold accepted answer present in context: **0/11** (by design, see section 0, point 2).

| Label | Meaning | Count | % of hallucinated |
|---|---|---|---|
| a | (a) context supports the incorrect claim AND the supporting item has trust_weight < 0.5 | 2 | 18.2% |
| a-high | (a') context supports the incorrect claim but the supporting item has trust_weight >= 0.5 (outside a/b/c) | 2 | 18.2% |
| b|c | (b) or (c): no context item supports the claim; b vs c NOT separable (no sufficiency or gold-evidence label) | 7 | 63.6% |

trust_weight distribution of items sent to the LLM (min / median / max / std):

| Subset | n items | min / median / max / std |
|---|---|---|
| all context items | 250 | 0.000 / 0.500 / 1.000 / 0.282 |
| origin=hop1 | 128 | 0.000 / 1.000 / 1.000 / 0.259 |
| origin=hop2 | 70 | 0.200 / 0.400 / 0.817 / 0.131 |
| origin=expansion | 52 | 0.200 / 0.400 / 0.400 / 0.083 |

Origin of the offending low-trust item in (a): hop2=1, expansion=1

### DEV a=0.5 ungrounded (n=50)

Hallucinated (panel-v2, includes SEBAGIAN): **12** of 50 questions. Gold accepted answer present in context: **0/12** (by design, see section 0, point 2).

| Label | Meaning | Count | % of hallucinated |
|---|---|---|---|
| a | (a) context supports the incorrect claim AND the supporting item has trust_weight < 0.5 | 1 | 8.3% |
| a-unattributed | context supports the incorrect claim but judge named no supporting item, so trust unknown | 1 | 8.3% |
| b|c | (b) or (c): no context item supports the claim; b vs c NOT separable (no sufficiency or gold-evidence label) | 10 | 83.3% |

trust_weight distribution of items sent to the LLM (min / median / max / std):

| Subset | n items | min / median / max / std |
|---|---|---|
| all context items | 250 | 0.000 / 0.500 / 1.000 / 0.282 |
| origin=hop1 | 128 | 0.000 / 1.000 / 1.000 / 0.259 |
| origin=hop2 | 70 | 0.200 / 0.400 / 0.817 / 0.131 |
| origin=expansion | 52 | 0.200 / 0.400 / 0.400 / 0.083 |

Origin of the offending low-trust item in (a): hop2=1

## 4. Sensitivity of (a) to the threshold (official run)

Offending supporting items (a + a') by `trust_weight`: expansion proxy 0.4 (3), hop2 0.4 (2), hop2 0.4727 (1), hop1 0.5 (2), hop1 1.0 (6). Count of (a) if the threshold were 0.3 / 0.4 / 0.5 / 0.6 / none: **0 / 0 / 6 / 8 / 14**. The (a) count is a direct function of the arbitrary cut-off.

Base rate: 782 of 1,920 official context items (40.7%) have `trust_weight < 0.5`, yet only 6 of 70 hallucinated answers (8.6%) are attributed to such an item. The data give no sign that low-trust items are over-represented in hallucinations.

## 5. Conclusions

1. **Dominant category: (b)/(c), 50 of 70 official hallucinations (71%).** The judges found no context item supporting the incorrect claim, so the error originates in generation or in missing evidence, not in a low-trust item. Only 6 (8.6%) are strictly (a); 14 (20%) have context that supports the wrong claim, and 8 of those 14 come from items with trust_weight >= 0.5 (6 are hop-1 items with weight 1.0). A split of (b) vs (c) is **not possible** with the stored labels, so this conclusion is "not caused by low-trust evidence", not "caused by the LLM". In the dev runs the pattern is the same (b|c is the largest bucket in all 6).
2. **Stages that bypass trust_weight:** seed selection (`:488-508`), hop-1 fetch (`:560`), expansion seed and neighbour choice (`:715-721`, `:647`), and the prompt (`prompts.py:149-162`). Expansion items carry a synthetic proxy weight (max 0.4) rather than a measured one; 3 of the 6 (a) cases in the official run come from such expansion items, and the other 3 from hop-2 items (0.40-0.47), so low-trust evidence does reach the LLM, but it is a small share of the failures.
3. **Next ablation:** run the missing configurations through the existing panel at n=384 (or a fixed n=100 subset): `--max-hops 1` vs `--max-hops 2` and `--no-enable-semantic-expansion` (switches `--max-hops`, `--enable-semantic-expansion` exist), each with alpha=0.5 vs alpha=0.0, to get the 1-hop / 2-hop / +expansion x +/- trust grid the task asked for. First run the context-sufficiency judge (blocked, not run) so (b) and (c) can be split; without it the dominant bucket stays unresolved. Judge runs need your explicit "go", so I have not started any; the plan and cost projection would come first.
