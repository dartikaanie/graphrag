# Result Step 1: Data Preparation Counts (SORD → Neo4j KG)

All numbers were computed read-only. No repo file was modified except this report.

**Where the data lives**
- Parquets and PostLinks.xml: `/Volumes/T7 Shield/S2/THESIS GRAPHRAG/project/graphrag copy/00_datasource/merged/`
- Pre-dedup questions: `questions_raw_union.parquet.predup_backup`
- Scripts used (scratchpad, not in repo): `q1.py`–`q5.py`, `tok.py`, `n1.py`–`n3.py` in `/private/tmp/claude-501/-Users-dartikaanie-SMART-SYSTEM-THESIS-2-project-graphrag/6d86ab9e-2a3e-4828-86ef-f701527c7ecd/scratchpad/`
- The raw CSVs are not on disk; per-file row counts come from `report/2_integrity_report.md` and were cross-checked against `match_source` counts in the parquets.

## Corrections to the premise

1. **The SORD download has 8 Contain/LikeMinusContain files, not 12.** `report/2_integrity_report.md` lists 12 files, but 4 are `additional-metadata` (Badges, Tags, Users, Votes). There are 4 categories × 2 sources: QuestionsTitle, QuestionsBody, Answers, Comments.
2. **725,795 is not "unique Answers after dedup."** `answers_raw_union.parquet` has 3,655,888 rows, all with unique Ids, so dedup removed none. The KG keeps only the 725,795 answers whose `ParentId` is a SORD Question. The other 2,930,093 answers have a parent Question outside SORD.
3. **The 47.4% figure was computed before dedup** (see section 3).

## 1. Raw rows and dedup

| File (source) | Rows | Source |
|---|---|---|
| QuestionsTitle_Contain | 73,901 | integrity report; equals `match_source` count in predup backup |
| QuestionsTitle_LikeMinusContain | 90,685 | same |
| QuestionsBody_Contain | 1,135,148 | same |
| QuestionsBody_LikeMinusContain | 1,511,800 | same |
| Answers_Contain | 2,228,118 | integrity report; equals `match_source` count |
| Answers_LikeMinusContain | 1,427,770 | same |
| Comments_Contain | 1,900,242 | integrity report; equals `match_source` count |
| Comments_LikeMinusContain | 2,114,468 | same |

| Metric | Value |
|---|---|
| Raw Question rows | 2,811,534 |
| Questions after dedup by Id | **2,685,814** (confirmed) |
| Question rows removed | 125,720 |
| Removed by source | `title_contain` −53,297 (73,901→20,604); `title_like` −72,423 (90,685→18,262); both body sources unchanged |
| Raw Answer rows | 3,655,888 |
| Distinct Answer Ids | 3,655,888 (0 removed) |
| Answers with a parent in the deduped Questions | **725,795** (confirmed) |
| Raw Comment rows | 4,014,710 (4,014,710 distinct, 0 removed) |
| Raw rows, Questions + Answers + Comments | 10,482,132 |

Commands: `select match_source,count(*)` and `select count(*),count(distinct Id)` on the predup backup, the deduped questions parquet and the answers parquet. Dedup script: `01_data_cleaning/3b_dedup_merged_sources.py`. KG build log: `01_data_cleaning/logs/20260829T075916Z_build_knowledge_graph.log` (2,685,814 and 725,795).

## 2. Answer coverage

| Metric | Value |
|---|---|
| Questions with ≥1 Answer in the SORD Answers table | **602,900** |
| Questions with no Answer | **2,082,914** (602,900 + 2,082,914 = 2,685,814) |
| Answers per answered Question | min 1, median 1, mean 1.204, max 54 (p75 = 1, p99 = 4) |
| Distinct `ParentId` over all 3,655,888 answers | 3,251,697 |

Answered Questions are counted as 602,900 because only Questions in the 2,685,814 set can be nodes. The 3,251,697 figure includes parents outside SORD. Do not use the `AnswerCount` column: it says 2,253,995 Questions have ≥1 answer, which counts answers that were never downloaded.

## 3. The 47.4% figure

- **Source:** chart A1 in `01_data_cleaning/5_eda_trust_signals.py:chart_a1`, written to `report/5_eda_report.md` ("47.4% dari 2,811,534 pertanyaan").
- **Numerator:** 1,333,247 rows with `AcceptedAnswerId` not null and ≠ 0.
- **Denominator:** 2,811,534, all Question rows **before dedup**, answered or not. Ratio 47.42%.
- **After dedup:** 1,275,276 / 2,685,814 = 47.48% (rounds to 47.5%, so the report's 47.4% is slightly stale).
- `AcceptedAnswerId` = 0 is the "none" marker (1,410,538 Questions); there are no nulls. The number only says an accepted id is recorded, not that the answer exists.

Reconciliation with 270,988 / 10.1%:

| Step | Count | % of 2,685,814 |
|---|---|---|
| Accepted id recorded (≠ 0) | 1,275,276 | 47.48% |
| Accepted id found in the SORD Answers table (rule a) | 270,988 | 10.09% |

270,988 is 21.2% of the 1,275,276 Questions with an accepted id; the other 1,004,288 point to an answer not in the download. Pre-dedup (chart A2): 295,975 of 1,333,247 = 22.2%.

## 4. Eligibility population

| Metric | Value | Query |
|---|---|---|
| Rule (a), before the Answers join | 1,275,276 | `AcceptedAnswerId is not null and !=0` |
| Rule (a), after the join (**population for the 1,536 pool**) | **270,988** | add `AcceptedAnswerId in (select Id from answers)`; same logic as `llm/a_pure_llm/a_baseline_replication.py` (~line 220) |
| Of those, title+body+tags ≤ 2,048 tokens (rule b) | **258,556 (95.41%)** | `tok.py`: tiktoken `cl100k_base` on `f"{Title} {Body} {Tags}"`, as in `filter_by_token_limit` |
| Excluded by rule (b) | 12,432 (4.59%) | |

Rule (b) is not applied to the whole population in the pipeline: the 1,536 pool is a reservoir sample (seed 42) of the 270,988, and the token filter is applied afterwards. The 258,556 figure is a full-population computation done for this report. The EDA chart A4's "~5.2%" was a sample-based estimate.

## 5. HAS_ACCEPTED_ANSWER integrity

| Check | Result |
|---|---|
| Distinct Questions with an accepted edge | 270,988 = edge count (Neo4j and parquet) |
| Distinct accepted Answers | 270,988 |
| Questions with more than one accepted edge | 0 |
| Accepted Answer whose parent ≠ the Question | 0 |
| 270,988 + 454,807 | = 725,795 exactly |
| Answers in Neo4j with no incoming Question edge | 0 |
| Answers with more than one incoming edge | 0 |
| Answers with a missing parent Question | 0 in the 725,795 (0 null `ParentId`; the 2,930,093 orphans are outside the KG) |

Neo4j queries: `n2.py`, `n3.py` over `HAS_ACCEPTED_ANSWER|HAS_ANSWER`.

## 6. Other counts

| Metric | Value |
|---|---|
| Answers with author known (AUTHOR_TRUST) | 715,604 (to 293,078 distinct Users) |
| Answers with no AUTHOR_TRUST edge | **10,191** (confirmed in Neo4j) |
| ↳ `OwnerUserId` = 0 (deleted/anonymous user) | 8,674; 8,638 have an `OwnerDisplayName` |
| ↳ positive id, user missing from `users.parquet` | 1,517 (1,333 distinct ids), no display name |
| Distinct authors needed | 294,412 (294,411 positive ids + the 0 id) |
| Users with reputation in the dump | 293,078 |
| Users missing | 1,333 (positive ids only; the 0 id isn't a user) |
| Tags per Question | mean 3.174, median 3, min 1, max 6; **0 Questions with zero tags**; 8,525,257 pairs; 56,435 distinct Tags |
| PostLinks raw rows | 6,552,590 (the "filter to types 1 and 3" step removed nothing) |
| PostLinks dropped (an endpoint not in SORD) | **6,391,943** (leaves 160,647) |
| PostLinks matched on one side only | `PostId` 806,123; `RelatedPostId` 730,936 |

- Community wiki cannot be checked: `CommunityOwnedDate` is a 1970-01-01 placeholder for null; only 131 of the 10,191 unknown-author answers have a real value.
- `users.parquet` has 22.5M users but not these 1,333.

## Discrepancy: IS_RELATED_TO is 151,094, not 160,647

| | Linked | Duplicate | Total |
|---|---|---|---|
| `question_links.parquet` (6_postlinks and KG build logs) | 149,808 | 10,839 | 160,647 |
| Neo4j now | 145,021 | 6,073 | **151,094** |

- Only 151,094 distinct directed (post, related) pairs exist, so 9,553 pairs appear in the raw links as both Linked and Duplicate. `MERGE (q1)-[r:IS_RELATED_TO]->(q2)` in `01_data_cleaning/7_build_knowledge_graph.py:830` collapses each into one edge, and the stored `link_type` is whichever was written last. Hence Duplicate −4,766 and Linked −4,787.
- Safe thesis wording: "160,647 links in SORD scope (149,808 Linked, 10,839 Duplicate), loaded as 151,094 distinct edges." The 6,073 Duplicate edges understate the true Duplicate links.
- Neo4j also holds `TAG_COOCCUR` and `EMBED_SIM` edges from the densification step, so do not total all relationship types.
