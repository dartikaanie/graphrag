"""Answerable entity anchoring, shared by Condition C (retrieval v4) and Condition D (retrieval v5)
(docs/DECISION_C_SCORING.md, Amendment 17).

A question can be a seed (anchor) only if it has at least one NON-EXCLUDED answer. Questions without an answer are
skipped and the scan continues down the SAME FAISS similarity list until n answerable seeds are found, up to
ANCHOR_SCAN_DEPTH candidates. If fewer than n are found, what was found is used and the shortfall is recorded. The
evaluation question (and anything else in the exclusion set) stays excluded exactly as before.

One function, used by both conditions, so C and D always get the same seeds for the same question. The only
Neo4j access is one batched existence check; the driver is passed in (this module imports neither neo4j nor faiss).
"""

from __future__ import annotations

ANCHOR_SCAN_DEPTH = 400  # candidates scanned down the FAISS list (tested on the dev set: 3 answerable seeds always found)
FETCH_TYPES = ("HAS_ACCEPTED_ANSWER", "HAS_ANSWER")


def answerable_question_ids(driver, database, question_ids, exclude_answer_ids, fetch_types=FETCH_TYPES) -> set:
    """Subset of `question_ids` with at least one answer that is not in `exclude_answer_ids` (one Cypher round trip)."""
    ids = sorted({int(q) for q in question_ids})
    if not ids:
        return set()
    exclude_a = list(exclude_answer_ids) or [-1]
    with driver.session(database=database) as session:
        rows = session.run(
            f"""
            UNWIND $qids AS qid
            MATCH (q:Question {{id: qid}})
            WHERE EXISTS {{
                MATCH (q)-[:{"|".join(fetch_types)}]->(a:Answer)
                WHERE NOT a.id IN $exclude_a
            }}
            RETURN qid
            """,
            qids=ids, exclude_a=exclude_a,
        ).data()
    return {int(r["qid"]) for r in rows}


def select_answerable_anchors(query_emb, index, ids_arr, n: int, exclude_question_ids, exclude_answer_ids,
                              driver, database, depth: int = ANCHOR_SCAN_DEPTH, stats: dict | None = None) -> list:
    """The first `n` answerable questions of the FAISS list for `query_emb`, as [(question_id, cosine)] in rank order.

    Candidates are the hits of the list that are not excluded (the first `depth` of them are scanned). `stats`
    (optional dict) receives n_hits_filtered (excluded hits dropped), and the record fields anchors_requested,
    anchors_found, anchor_scan_depth, anchor_candidates_scanned (position, among the scanned candidates, of the last
    seed taken) and anchors_skipped_unanswered (scanned candidates above that position without an answer)."""
    exclude_q = set(int(q) for q in exclude_question_ids)
    scores, idxs = index.search(query_emb, depth + len(exclude_q) + 5)  # buffer: excluded hits do not count against depth
    candidates, filtered = [], 0
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        qid = int(ids_arr[idx])
        if qid in exclude_q:
            filtered += 1
            continue
        candidates.append((qid, float(score)))
        if len(candidates) >= depth:
            break
    answerable = answerable_question_ids(driver, database, [q for q, _ in candidates], exclude_answer_ids)
    chosen, last_pos = [], 0
    for pos, (qid, score) in enumerate(candidates, start=1):
        if qid in answerable:
            chosen.append((qid, score))
            last_pos = pos
            if len(chosen) >= n:
                break
    if stats is not None:
        stats["n_hits_filtered"] = stats.get("n_hits_filtered", 0) + filtered
        stats.update(anchors_requested=n, anchors_found=len(chosen), anchor_scan_depth=depth,
                     anchor_candidates_scanned=last_pos if chosen else len(candidates),
                     anchors_skipped_unanswered=last_pos - len(chosen) if chosen else len(candidates))
    return chosen


def anchor_record_fields(stats: dict) -> dict:
    """The answerable-anchor facts a record stores (empty when the stats hold none, i.e. the earlier versions)."""
    keys = ("anchors_requested", "anchors_found", "anchor_scan_depth", "anchor_candidates_scanned",
            "anchors_skipped_unanswered")
    return {k: stats[k] for k in keys} if "anchors_found" in stats else {}
