"""Master-data queries (Questions/Answers/Tags) against Neo4j.

Neo4j is the source of truth here (not the raw Parquet) because it already
carries the derived fields the dashboard needs to show (domainTag,
trustScore) per PLAN_UI_UX.md §1.1 — built by
01_data_cleaning/7_build_knowledge_graph.py.
"""

import math

from app.db.neo4j_client import run_query


def list_questions(page: int, page_size: int, search: str | None, tag: str | None) -> tuple[list[dict], int]:
    skip = (page - 1) * page_size

    if search:
        # Plain `WHERE toLower(q.title) CONTAINS ...` was tried first and had
        # to be abandoned: with no index to back it, Cypher would stream
        # Questions in score order (to satisfy ORDER BY) and apply the CONTAINS
        # filter lazily row-by-row, which could scan nearly the whole ~2.7M
        # population before finding `limit` matches -- observed hanging past
        # 20s+ in testing. A full-text index (`question_title_fts`) makes
        # search itself index-backed instead.
        return _search_questions(page, page_size, search, tag)

    params: dict = {"skip": skip, "limit": page_size}
    match_clause = "MATCH (q:Question)"
    if tag:
        match_clause = "MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag {name: $tag})"
        params["tag"] = tag

    count_query = f"{match_clause} RETURN count(q) AS total"
    total = run_query(count_query, params)[0]["total"]

    # Page first (ORDER BY q.score DESC is index-backed via `question_score`,
    # so this avoids sorting the full ~2.7M-node population), THEN compute
    # answer_count only for the page_size rows actually returned. Doing the
    # OPTIONAL MATCH aggregate before SKIP/LIMIT (i.e. across the whole
    # population before paging) made this endpoint hang during testing.
    list_query = f"""
        {match_clause}
        RETURN q.id AS id, q.title AS title, q.domainTag AS domain_tag,
               q.score AS score, q.viewCount AS view_count
        ORDER BY q.score DESC
        SKIP $skip LIMIT $limit
    """
    rows = run_query(list_query, params)
    if not rows:
        return rows, total

    ids = [r["id"] for r in rows]
    counts_query = """
        UNWIND $ids AS qid
        MATCH (q:Question {id: qid})
        OPTIONAL MATCH (q)-[:HAS_ANSWER|HAS_ACCEPTED_ANSWER]->(a:Answer)
        RETURN qid, count(a) AS answer_count
    """
    count_rows = run_query(counts_query, {"ids": ids})
    counts_by_id = {r["qid"]: r["answer_count"] for r in count_rows}
    for r in rows:
        r["answer_count"] = counts_by_id.get(r["id"], 0)
    return rows, total


def _search_questions(page: int, page_size: int, search: str, tag: str | None) -> tuple[list[dict], int]:
    skip = (page - 1) * page_size
    # db.index.fulltext.queryNodes returns a `score` column (Lucene relevance)
    # which we deliberately ignore in favor of ORDER BY q.score DESC below,
    # to keep result ordering consistent with the unfiltered list.
    fts_clause = "CALL db.index.fulltext.queryNodes('question_title_fts', $search) YIELD node AS q"
    if tag:
        fts_clause += " MATCH (q)-[:TAGGED_WITH]->(:Tag {name: $tag})"
    params = {"search": search, "tag": tag, "skip": skip, "limit": page_size}

    total = run_query(f"{fts_clause} RETURN count(q) AS total", params)[0]["total"]

    list_query = f"""
        {fts_clause}
        RETURN q.id AS id, q.title AS title, q.domainTag AS domain_tag,
               q.score AS score, q.viewCount AS view_count
        ORDER BY q.score DESC
        SKIP $skip LIMIT $limit
    """
    rows = run_query(list_query, params)
    for r in rows:
        r["answer_count"] = None  # not computed for search results (kept snappy)
    return rows, total


def get_question_detail(question_id: int) -> dict | None:
    query = """
        MATCH (q:Question {id: $id})
        OPTIONAL MATCH (q)-[:HAS_ANSWER|HAS_ACCEPTED_ANSWER]->(a:Answer)
        OPTIONAL MATCH (q)-[r]-()
        RETURN q AS question, count(DISTINCT a) AS answer_count, count(r) AS relation_count
    """
    rows = run_query(query, {"id": question_id})
    if not rows:
        return None
    row = rows[0]
    attributes = dict(row["question"])
    return {
        "id": question_id,
        "attributes": attributes,
        "graph_meta": {
            "answer_count": row["answer_count"],
            "relation_count": row["relation_count"],
        },
    }


def get_question_answers(question_id: int) -> list[dict]:
    query = """
        MATCH (q:Question {id: $id})-[:HAS_ANSWER|HAS_ACCEPTED_ANSWER]->(a:Answer)
        OPTIONAL MATCH (q)-[:HAS_ACCEPTED_ANSWER]->(a)
        RETURN a AS answer, (a IS NOT NULL AND exists((q)-[:HAS_ACCEPTED_ANSWER]->(a))) AS is_accepted
        ORDER BY is_accepted DESC, a.score DESC
    """
    rows = run_query(query, {"id": question_id})
    return [dict(r["answer"]) for r in rows]


def get_accepted_answer(question_id: int) -> dict | None:
    query = """
        MATCH (q:Question {id: $id})-[:HAS_ACCEPTED_ANSWER]->(a:Answer)
        RETURN a AS answer
        LIMIT 1
    """
    rows = run_query(query, {"id": question_id})
    return dict(rows[0]["answer"]) if rows else None


def list_answers(page: int, page_size: int, search: str | None) -> tuple[list[dict], int]:
    skip = (page - 1) * page_size
    params: dict = {"skip": skip, "limit": page_size}

    if search:
        # Same reasoning as list_questions: unindexed CONTAINS over ~726k
        # Answer bodies risks the same lazy-scan hang, so use the full-text
        # index instead.
        params["search"] = search
        match_clause = "CALL db.index.fulltext.queryNodes('answer_body_fts', $search) YIELD node AS a"
    else:
        match_clause = "MATCH (a:Answer)"

    total = run_query(f"{match_clause} RETURN count(a) AS total", params)[0]["total"]

    query = f"""
        {match_clause}
        OPTIONAL MATCH (q:Question)-[:HAS_ANSWER|HAS_ACCEPTED_ANSWER]->(a)
        RETURN a.id AS id, a.body AS body, a.score AS score,
               a.isAccepted AS is_accepted, q.id AS question_id
        ORDER BY a.score DESC
        SKIP $skip LIMIT $limit
    """
    rows = run_query(query, params)
    return rows, total


def get_answer_detail(answer_id: int) -> dict | None:
    query = """
        MATCH (a:Answer {id: $id})
        OPTIONAL MATCH (q:Question)-[:HAS_ANSWER|HAS_ACCEPTED_ANSWER]->(a)
        OPTIONAL MATCH (a)-[r]-()
        RETURN a AS answer, q.id AS question_id, count(r) AS relation_count
    """
    rows = run_query(query, {"id": answer_id})
    if not rows:
        return None
    row = rows[0]
    attributes = dict(row["answer"])
    return {
        "id": answer_id,
        "attributes": attributes,
        "graph_meta": {
            "question_id": row["question_id"],
            "relation_count": row["relation_count"],
        },
    }


def list_tags(page: int, page_size: int, search: str | None) -> tuple[list[dict], int]:
    skip = (page - 1) * page_size
    where_sql = ""
    params: dict = {"skip": skip, "limit": page_size}
    if search:
        where_sql = "WHERE toLower(t.name) CONTAINS toLower($search)"
        params["search"] = search

    total = run_query(f"MATCH (t:Tag) {where_sql} RETURN count(t) AS total", params)[0]["total"]

    query = f"""
        MATCH (t:Tag)
        {where_sql}
        RETURN t.name AS name, t.questionCount AS question_count
        ORDER BY t.questionCount DESC
        SKIP $skip LIMIT $limit
    """
    rows = run_query(query, params)
    return rows, total


def get_tag_detail(tag_name: str, limit: int = 50) -> dict | None:
    tag_rows = run_query(
        "MATCH (t:Tag {name: $name}) RETURN t.name AS name, t.questionCount AS question_count",
        {"name": tag_name},
    )
    if not tag_rows:
        return None

    # Two-step, same reasoning as list_questions: pull only `limit` question
    # ids for this tag (index-backed ORDER BY on score, cheap even for a tag
    # with hundreds of thousands of questions like "android"), rather than
    # collect()-ing every matching Question before limiting.
    questions_query = """
        MATCH (t:Tag {name: $name})<-[:TAGGED_WITH]-(q:Question)
        RETURN q.id AS id, q.title AS title, q.domainTag AS domainTag,
               q.score AS score, q.viewCount AS viewCount
        ORDER BY q.score DESC
        LIMIT $limit
    """
    questions = run_query(questions_query, {"name": tag_name, "limit": limit})

    row = tag_rows[0]
    return {
        "name": row["name"],
        "question_count": row["question_count"],
        "questions": questions,
    }


def get_stats_summary() -> dict:
    # Deliberately 4 separate single-label/pattern queries rather than one
    # query chained with WITH: chaining independent MATCH...count()...WITH
    # steps made the Neo4j planner stop using the (very fast) label/relationship
    # count store and instead hang on this ~2.7M node / 32M edge graph.
    total_questions = run_query("MATCH (q:Question) RETURN count(q) AS c")[0]["c"]
    total_answers = run_query("MATCH (a:Answer) RETURN count(a) AS c")[0]["c"]
    total_tags = run_query("MATCH (t:Tag) RETURN count(t) AS c")[0]["c"]
    total_edges = run_query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    return {
        "total_questions": total_questions,
        "total_answers": total_answers,
        "total_tags": total_tags,
        "total_edges": total_edges,
    }


def total_pages(total: int, page_size: int) -> int:
    return max(1, math.ceil(total / page_size))
