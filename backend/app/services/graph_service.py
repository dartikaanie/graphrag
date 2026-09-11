"""Subgraph queries for react-force-graph visualization (Home + detail pages).

Every query here is deliberately capped (LIMIT on every fan-out, per-node caps
via CALL subqueries) rather than a broad MATCH -- Phase 1 already showed that
an uncapped traversal or aggregate over this graph (~2.7M Question nodes,
~32M edges) can hang for minutes. A force-directed layout is also unusable
past a few hundred nodes anyway, so capping is correct for both performance
and readability (PLAN_UI_UX.md §5.3).
"""

from app.db.neo4j_client import run_query

NODE_LABELS = {"question": "Question", "answer": "Answer", "tag": "Tag"}


def _qnode(row_id, title, score, trust_score, domain_tag) -> dict:
    return {
        "id": f"Question-{row_id}",
        "type": "Question",
        "label": (title or "")[:80],
        "properties": {"id": row_id, "score": score, "trustScore": trust_score, "domainTag": domain_tag},
    }


def _anode(row_id, score, trust_score, is_accepted) -> dict:
    return {
        "id": f"Answer-{row_id}",
        "type": "Answer",
        "label": f"Answer #{row_id}",
        "properties": {"id": row_id, "score": score, "trustScore": trust_score, "isAccepted": is_accepted},
    }


def _tnode(name, question_count) -> dict:
    return {
        "id": f"Tag-{name}",
        "type": "Tag",
        "label": name,
        "properties": {"name": name, "questionCount": question_count},
    }


def _unode(row_id, reputation) -> dict:
    return {
        "id": f"User-{row_id}",
        "type": "User",
        "label": f"User #{row_id}",
        "properties": {"id": row_id, "reputation": reputation},
    }


def _link(source, target, edge_type, weight=None) -> dict:
    return {"source": source, "target": target, "type": edge_type, "weight": weight}


def get_partial_graph(limit: int = 100) -> dict:
    """Representative subgraph for Home: top `limit` Questions by score, each
    with up to 2 answers and up to 2 tags -- enough to show real structure
    without producing an unreadable force-directed hairball.
    """
    top_questions = run_query(
        """
        MATCH (q:Question)
        RETURN q.id AS id, q.title AS title, q.score AS score,
               q.trustScore AS trustScore, q.domainTag AS domainTag
        ORDER BY q.score DESC
        LIMIT $limit
        """,
        {"limit": limit},
    )
    ids = [q["id"] for q in top_questions]
    if not ids:
        return {"nodes": [], "links": []}

    nodes: dict[str, dict] = {}
    links: list[dict] = []

    for q in top_questions:
        node = _qnode(q["id"], q["title"], q["score"], q["trustScore"], q["domainTag"])
        nodes[node["id"]] = node

    answer_rows = run_query(
        """
        UNWIND $ids AS qid
        MATCH (q:Question {id: qid})
        CALL (q) {
            MATCH (q)-[r:HAS_ANSWER]->(a:Answer)
            RETURN a, r.weight AS weight
            ORDER BY a.score DESC
            LIMIT 2
        }
        OPTIONAL MATCH (q)-[:HAS_ACCEPTED_ANSWER]->(a)
        RETURN qid AS question_id, a.id AS answer_id, a.score AS score, a.trustScore AS trustScore,
               weight, (a IS NOT NULL AND exists((q)-[:HAS_ACCEPTED_ANSWER]->(a))) AS accepted
        """,
        {"ids": ids},
    )
    for r in answer_rows:
        anode = _anode(r["answer_id"], r["score"], r["trustScore"], r["accepted"])
        nodes[anode["id"]] = anode
        edge_type = "HAS_ACCEPTED_ANSWER" if r["accepted"] else "HAS_ANSWER"
        weight = 1.0 if r["accepted"] else r["weight"]
        links.append(_link(f"Question-{r['question_id']}", anode["id"], edge_type, weight))

    tag_rows = run_query(
        """
        UNWIND $ids AS qid
        MATCH (q:Question {id: qid})
        CALL (q) {
            MATCH (q)-[:TAGGED_WITH]->(t:Tag)
            RETURN t
            LIMIT 2
        }
        RETURN qid AS question_id, t.name AS name, t.questionCount AS questionCount
        """,
        {"ids": ids},
    )
    for r in tag_rows:
        tnode = _tnode(r["name"], r["questionCount"])
        nodes[tnode["id"]] = tnode
        links.append(_link(f"Question-{r['question_id']}", tnode["id"], "TAGGED_WITH"))

    return {"nodes": list(nodes.values()), "links": links}


def _question_neighbors(question_id: int) -> tuple[dict, list, list]:
    """1-hop neighbors of a Question: answers, tags, and related/cooccur/embed-sim
    questions. Returns (nodes_by_id, links, related_question_ids) -- the third
    value feeds the optional hop-2 expansion.
    """
    nodes: dict[str, dict] = {}
    links: list[dict] = []
    related_qids: list[int] = []

    answers = run_query(
        """
        MATCH (q:Question {id: $id})-[r:HAS_ANSWER]->(a:Answer)
        OPTIONAL MATCH (q)-[:HAS_ACCEPTED_ANSWER]->(a)
        RETURN a.id AS id, a.score AS score, a.trustScore AS trustScore, r.weight AS weight,
               (a IS NOT NULL AND exists((q)-[:HAS_ACCEPTED_ANSWER]->(a))) AS accepted
        ORDER BY accepted DESC, a.score DESC
        LIMIT 10
        """,
        {"id": question_id},
    )
    for a in answers:
        anode = _anode(a["id"], a["score"], a["trustScore"], a["accepted"])
        nodes[anode["id"]] = anode
        edge_type = "HAS_ACCEPTED_ANSWER" if a["accepted"] else "HAS_ANSWER"
        weight = 1.0 if a["accepted"] else a["weight"]
        links.append(_link(f"Question-{question_id}", anode["id"], edge_type, weight))

    tags = run_query(
        """
        MATCH (q:Question {id: $id})-[:TAGGED_WITH]->(t:Tag)
        RETURN t.name AS name, t.questionCount AS questionCount
        LIMIT 10
        """,
        {"id": question_id},
    )
    for t in tags:
        tnode = _tnode(t["name"], t["questionCount"])
        nodes[tnode["id"]] = tnode
        links.append(_link(f"Question-{question_id}", tnode["id"], "TAGGED_WITH"))

    for rel_type in ("IS_RELATED_TO", "TAG_COOCCUR", "EMBED_SIM"):
        weight_field = "cosine_sim" if rel_type == "EMBED_SIM" else ("jaccard" if rel_type == "TAG_COOCCUR" else "weight")
        rows = run_query(
            f"""
            MATCH (q:Question {{id: $id}})-[r:{rel_type}]-(other:Question)
            RETURN other.id AS id, other.title AS title, other.score AS score,
                   other.trustScore AS trustScore, other.domainTag AS domainTag,
                   coalesce(r.{weight_field}, r.weight) AS weight
            ORDER BY weight DESC
            LIMIT 8
            """,
            {"id": question_id},
        )
        for r in rows:
            qnode = _qnode(r["id"], r["title"], r["score"], r["trustScore"], r["domainTag"])
            nodes[qnode["id"]] = qnode
            links.append(_link(f"Question-{question_id}", qnode["id"], rel_type, r["weight"]))
            related_qids.append(r["id"])

    return nodes, links, related_qids


def get_question_subgraph(question_id: int, hops: int = 2) -> dict:
    center = run_query(
        """
        MATCH (q:Question {id: $id})
        RETURN q.id AS id, q.title AS title, q.score AS score, q.trustScore AS trustScore, q.domainTag AS domainTag
        """,
        {"id": question_id},
    )
    if not center:
        return {"nodes": [], "links": []}

    nodes: dict[str, dict] = {}
    links: list[dict] = []
    c = center[0]
    center_node = _qnode(c["id"], c["title"], c["score"], c["trustScore"], c["domainTag"])
    nodes[center_node["id"]] = center_node

    hop1_nodes, hop1_links, related_qids = _question_neighbors(question_id)
    nodes.update(hop1_nodes)
    links.extend(hop1_links)

    if hops >= 2:
        # Shallow second layer: for a handful of the related Questions found at
        # hop 1, pull just their accepted answer + top tag -- enough to show
        # depth without repeating the full fan-out (which would blow up fast
        # since TAG_COOCCUR/EMBED_SIM neighbors can themselves have thousands
        # of edges).
        for qid in related_qids[:8]:
            extra = run_query(
                """
                MATCH (q:Question {id: $id})
                OPTIONAL MATCH (q)-[:HAS_ACCEPTED_ANSWER]->(a:Answer)
                OPTIONAL MATCH (q)-[:TAGGED_WITH]->(t:Tag)
                WITH q, a, collect(DISTINCT t)[0..1] AS tags
                RETURN a.id AS answer_id, a.score AS answer_score, a.trustScore AS answer_trust,
                       [t IN tags | {name: t.name, questionCount: t.questionCount}] AS tags
                """,
                {"id": qid},
            )
            if not extra:
                continue
            row = extra[0]
            if row["answer_id"] is not None:
                anode = _anode(row["answer_id"], row["answer_score"], row["answer_trust"], True)
                nodes[anode["id"]] = anode
                links.append(_link(f"Question-{qid}", anode["id"], "HAS_ACCEPTED_ANSWER", 1.0))
            for t in row["tags"]:
                tnode = _tnode(t["name"], t["questionCount"])
                nodes[tnode["id"]] = tnode
                links.append(_link(f"Question-{qid}", tnode["id"], "TAGGED_WITH"))

    return {"nodes": list(nodes.values()), "links": links}


def get_answer_subgraph(answer_id: int, hops: int = 2) -> dict:
    center = run_query(
        "MATCH (a:Answer {id: $id}) RETURN a.id AS id, a.score AS score, a.trustScore AS trustScore, a.isAccepted AS isAccepted",
        {"id": answer_id},
    )
    if not center:
        return {"nodes": [], "links": []}

    nodes: dict[str, dict] = {}
    links: list[dict] = []
    c = center[0]
    center_node = _anode(c["id"], c["score"], c["trustScore"], c["isAccepted"])
    nodes[center_node["id"]] = center_node

    owner = run_query(
        """
        MATCH (q:Question)-[:HAS_ANSWER]->(a:Answer {id: $id})
        OPTIONAL MATCH (q)-[:HAS_ACCEPTED_ANSWER]->(a)
        RETURN q.id AS id, q.title AS title, q.score AS score, q.trustScore AS trustScore, q.domainTag AS domainTag,
               (a IS NOT NULL AND exists((q)-[:HAS_ACCEPTED_ANSWER]->(a))) AS accepted
        """,
        {"id": answer_id},
    )
    question_id = None
    if owner:
        o = owner[0]
        question_id = o["id"]
        qnode = _qnode(o["id"], o["title"], o["score"], o["trustScore"], o["domainTag"])
        nodes[qnode["id"]] = qnode
        edge_type = "HAS_ACCEPTED_ANSWER" if o["accepted"] else "HAS_ANSWER"
        links.append(_link(qnode["id"], center_node["id"], edge_type, 1.0 if o["accepted"] else None))

    authors = run_query(
        """
        MATCH (a:Answer {id: $id})-[r:AUTHOR_TRUST]->(u:User)
        RETURN u.id AS id, u.reputation AS reputation, r.weight AS weight
        LIMIT 5
        """,
        {"id": answer_id},
    )
    for u in authors:
        unode = _unode(u["id"], u["reputation"])
        nodes[unode["id"]] = unode
        links.append(_link(center_node["id"], unode["id"], "AUTHOR_TRUST", u["weight"]))

    if hops >= 2 and question_id is not None:
        hop1_nodes, hop1_links, _related = _question_neighbors(question_id)
        for nid, n in hop1_nodes.items():
            if nid not in nodes:
                nodes[nid] = n
        links.extend(hop1_links)

    return {"nodes": list(nodes.values()), "links": links}


def get_tag_subgraph(tag_name: str, hops: int = 2) -> dict:
    center = run_query(
        "MATCH (t:Tag {name: $name}) RETURN t.name AS name, t.questionCount AS questionCount",
        {"name": tag_name},
    )
    if not center:
        return {"nodes": [], "links": []}

    nodes: dict[str, dict] = {}
    links: list[dict] = []
    c = center[0]
    center_node = _tnode(c["name"], c["questionCount"])
    nodes[center_node["id"]] = center_node

    questions = run_query(
        """
        MATCH (t:Tag {name: $name})<-[:TAGGED_WITH]-(q:Question)
        RETURN q.id AS id, q.title AS title, q.score AS score, q.trustScore AS trustScore, q.domainTag AS domainTag
        ORDER BY q.score DESC
        LIMIT 15
        """,
        {"name": tag_name},
    )
    top_qids = []
    for q in questions:
        qnode = _qnode(q["id"], q["title"], q["score"], q["trustScore"], q["domainTag"])
        nodes[qnode["id"]] = qnode
        links.append(_link(qnode["id"], center_node["id"], "TAGGED_WITH"))
        top_qids.append(q["id"])

    if hops >= 2:
        for qid in top_qids[:5]:
            accepted = run_query(
                """
                MATCH (q:Question {id: $id})-[:HAS_ACCEPTED_ANSWER]->(a:Answer)
                RETURN a.id AS id, a.score AS score, a.trustScore AS trustScore
                """,
                {"id": qid},
            )
            for a in accepted:
                anode = _anode(a["id"], a["score"], a["trustScore"], True)
                nodes[anode["id"]] = anode
                links.append(_link(f"Question-{qid}", anode["id"], "HAS_ACCEPTED_ANSWER", 1.0))

    return {"nodes": list(nodes.values()), "links": links}


def get_node_subgraph(node_type: str, node_id: str, hops: int = 2) -> dict:
    node_type = node_type.lower()
    if node_type == "question":
        return get_question_subgraph(int(node_id), hops)
    if node_type == "answer":
        return get_answer_subgraph(int(node_id), hops)
    if node_type == "tag":
        return get_tag_subgraph(node_id, hops)
    raise ValueError(f"Unknown node_type '{node_type}', expected question/answer/tag")
