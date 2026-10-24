"""Per-item retrieval provenance: one schema for the items in `retrieved_context`
of Conditions B, C and D (provenance_version "v3").

v3 vs v2: `path` steps carry node ids and edge types ONLY (no per-edge weights);
every item has `owner_question_id`; D no longer records trust_weight; C/D records
carry a bounded `not_selected` / `below_cutoff` trace; provenance_version is part
of config_hash. v2 records stay readable (extra step keys are tolerated).

Scope: retrieval-side provenance for audit and reproducibility -- which KG path
(or which vector hit) produced each context item. It is NOT shown to the LLM
(llm/prompts.py::_format_context_block emits only `[SO-<id>] <text>`; see
llm/tests/test_prompt_no_provenance.py) and it is NOT a user-facing explanation.
Provenance is additive: every field the item carried before is unchanged.
(Not to be confused with llm/common/provenance.py, which records RUN provenance
in the manifest.)

Item fields added by this module (on top of the existing question_id,
answer_id, chunk_text, hop, rel_type, source_stage, is_accepted, trust_weight /
relevance_score / combined_score / sim / sim_norm / final_score / alpha):

    owner_question_id the Question that owns the answer (== question_id == via_question_id)
    via_question_id   alias of question_id, kept from v2
    rank              1-based position in the final list
    anchor_id         Question id the item originated from == path[0].node_id;
                      None for vector_only (B has no anchor)
    path              ordered steps {node_type, node_id, edge_type_to_next}; the
                      last step is the Answer
    alt_paths         other paths that reach the SAME answer and lost the dedup
                      (may be []); each {path, source_stage, hop, rel_type,
                      <score key>: value}
    seed_similarity   semantic_expansion only: cosine between seed and the
                      expanded Question (FAISS)
    trust_weight      edge weight for the item (C only; None for B, which has no edge,
                      and for D, which does not use trust weighting)

Record-level (C and D, `build_trace`): `not_selected` = {answer_ids, question_ids,
n_answer_ids, n_question_ids} -- ids on candidate paths that did not reach the final
list (lists capped at 100, the n_* give the uncapped count) -- and `below_cutoff` =
[[answer_id, rank, score], ...] for the (max 20) answers ranked just under top-k.
Both are read from the candidates the fusion already holds; ids cut INSIDE a Cypher
query (LIMIT) never reach Python and are not listed.

Path shapes (node_type Q = Question, A = Answer):

    graph_traversal hop 1   Q(anchor) -[HAS_ACCEPTED_ANSWER|HAS_ANSWER]-> A
    graph_traversal hop 2   Q(anchor) -[IS_RELATED_TO|TAG_COOCCUR|EMBED_SIM]-> Q(q2) -[HAS_*]-> A
    semantic_expansion      Q(seed) -[SEMANTIC_SIM]-> Q(expanded) -[HAS_*]-> A
                            (SEMANTIC_SIM is a FAISS nearest-neighbour link, not a KG
                            edge; its cosine is the item's seed_similarity)
    low_level (D)           same as hop 1
    high_level (D)          same as hop 2
    vector_only (B)         A (a single step; no edges)

`rel_type` is kept as each condition always wrote it (C hop 2: "r2->r3"; D
high_level: the r2 type only; semantic: the HAS_* type). For analysis across
conditions use path[].edge_type_to_next, which is uniform.
"""

from __future__ import annotations

from typing import Callable, Optional, TypedDict

PROVENANCE_VERSION = "v3"
SUPPORTED_VERSIONS = ("v2", "v3")   # versions the readers (lookup, analysis) understand

STAGE_GRAPH = "graph_traversal"
STAGE_SEMANTIC = "semantic_expansion"
STAGE_VECTOR = "vector_only"
STAGE_LOW = "low_level"
STAGE_HIGH = "high_level"
SOURCE_STAGES = (STAGE_GRAPH, STAGE_SEMANTIC, STAGE_VECTOR, STAGE_LOW, STAGE_HIGH)

EDGE_SEMANTIC_SIM = "SEMANTIC_SIM"
FETCH_EDGES = ("HAS_ACCEPTED_ANSWER", "HAS_ANSWER")
RELATION_EDGES = ("IS_RELATED_TO", "TAG_COOCCUR", "EMBED_SIM")

NODE_QUESTION = "Question"
NODE_ANSWER = "Answer"

# Item keys this module adds. A snapshot of the PRE-provenance item shape (the
# C retrieval fixtures) is compared after dropping these.
ADDED_ITEM_KEYS = ("via_question_id", "owner_question_id", "anchor_id", "path", "alt_paths", "seed_similarity")
TRACE_CAP_IDS = 100
TRACE_CAP_BELOW_CUTOFF = 20


class PathStep(TypedDict):
    node_type: str
    node_id: Optional[int]
    edge_type_to_next: Optional[str]


class AltPath(TypedDict, total=False):
    path: list
    source_stage: str
    hop: object
    rel_type: Optional[str]


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------

def _int_or_none(v):
    return None if v is None else int(v)


def _float_or_none(v):
    return None if v is None else float(v)


def path_step(node_type: str, node_id, edge_type_to_next: str | None = None) -> PathStep:
    return {"node_type": node_type, "node_id": _int_or_none(node_id), "edge_type_to_next": edge_type_to_next}


def split_rel_type(rel_type: str | None) -> list[str]:
    """"TAG_COOCCUR->HAS_ANSWER" -> ["TAG_COOCCUR", "HAS_ANSWER"]."""
    return [t for t in (rel_type or "").split("->") if t]


def one_hop_path(anchor_id, answer_id, fetch_type) -> list[PathStep]:
    return [path_step(NODE_QUESTION, anchor_id, fetch_type), path_step(NODE_ANSWER, answer_id)]


def two_hop_path(anchor_id, q2_id, answer_id, relation_type, fetch_type) -> list[PathStep]:
    return [path_step(NODE_QUESTION, anchor_id, relation_type),
            path_step(NODE_QUESTION, q2_id, fetch_type),
            path_step(NODE_ANSWER, answer_id)]


def semantic_path(seed_id, expanded_q_id, answer_id, fetch_type) -> list[PathStep]:
    return [path_step(NODE_QUESTION, seed_id, EDGE_SEMANTIC_SIM),
            path_step(NODE_QUESTION, expanded_q_id, fetch_type),
            path_step(NODE_ANSWER, answer_id)]


def vector_only_path(answer_id) -> list[PathStep]:
    return [path_step(NODE_ANSWER, answer_id)]


def _stage_order(source_stage: str, hop) -> int:
    return {STAGE_GRAPH: 0, STAGE_LOW: 0, STAGE_HIGH: 1, STAGE_SEMANTIC: 2, STAGE_VECTOR: 3}.get(source_stage, 9) * 10 + (
        hop if isinstance(hop, int) else 5)


def alt_paths_for(winner: dict, candidates: list[dict], score_key: str) -> list[dict]:
    """Paths of every other candidate that reaches the winner's answer_id.

    Candidates are the dicts the condition built BEFORE its dedup; each carries
    `_path` (the path), `source_stage`, `hop`, `rel_type` and `score_key`. A
    candidate whose (path, rel_type) equals the winner's is not an alternative.
    Order is deterministic: score descending, then stage/hop, then the path's
    node ids. The winner itself is chosen by the condition's own, unchanged
    dedup rule (C: highest combined_score, first seen wins a tie -- v2/v3
    trust_weighted; first seen -- uniform; D: highest relevance_score, first
    seen wins a tie)."""
    seen = {_path_key(winner["_path"], winner.get("rel_type"))}
    alts = []
    for c in candidates:
        if c is winner or c["answer_id"] != winner["answer_id"]:
            continue
        key = _path_key(c["_path"], c.get("rel_type"))
        if key in seen:
            continue
        seen.add(key)
        alts.append({"path": c["_path"], "source_stage": c["source_stage"], "hop": c.get("hop"),
                     "rel_type": c.get("rel_type"), score_key: _score_for_json(c.get(score_key))})
    alts.sort(key=lambda a: (-(a.get(score_key) or 0.0), _stage_order(a["source_stage"], a["hop"]),
                             [(s["node_id"] if s["node_id"] is not None else -1) for s in a["path"]],
                             a["rel_type"] or ""))
    return alts


def _score_for_json(v):
    return None if v is None else round(float(v), 4)


def _path_key(path: list, rel_type) -> tuple:
    return (tuple((s["node_type"], s["node_id"], s["edge_type_to_next"]) for s in path), rel_type)


def provenance_fields(*, source_stage: str, hop, path: list, rank: int, via_question_id=None,
                      alt_paths: list | None = None, seed_similarity: float | None = None) -> dict:
    """The provenance keys for one item (everything the module adds except
    trust_weight/rel_type/scores, which each condition already writes)."""
    anchor = None if source_stage == STAGE_VECTOR else path[0]["node_id"]
    return {
        "owner_question_id": _int_or_none(via_question_id),
        "via_question_id": _int_or_none(via_question_id),
        "rank": int(rank),
        "anchor_id": anchor,
        "path": path,
        "alt_paths": list(alt_paths or []),
        "seed_similarity": _float_or_none(seed_similarity),
    }


def build_trace(all_candidates: list[dict], final_items: list[dict], below_cutoff: list) -> dict:
    """Record-level `not_selected` / `below_cutoff` from what the fusion already holds:
    `all_candidates` (each with `_path`, in discovery order), the final items, and
    `below_cutoff` = [[answer_id, rank, score], ...] for the answers ranked just under
    top-k. Nothing is queried or copied beyond the id lists."""
    kept_answers = {i["answer_id"] for i in final_items}
    kept_questions = {s["node_id"] for i in final_items for s in i["path"] if s["node_type"] == NODE_QUESTION}
    answers, questions = [], []
    seen_a, seen_q = set(), set()
    for c in all_candidates:
        if c["answer_id"] not in kept_answers and c["answer_id"] not in seen_a:
            seen_a.add(c["answer_id"])
            answers.append(int(c["answer_id"]))
        for step in c["_path"]:
            nid = step["node_id"]
            if step["node_type"] == NODE_QUESTION and nid is not None and nid not in kept_questions and nid not in seen_q:
                seen_q.add(nid)
                questions.append(nid)
    return {"not_selected": {"answer_ids": answers[:TRACE_CAP_IDS], "question_ids": questions[:TRACE_CAP_IDS],
                             "n_answer_ids": len(answers), "n_question_ids": len(questions)},
            "below_cutoff": [[int(a), int(r), round(float(sc), 4)] for a, r, sc in below_cutoff[:TRACE_CAP_BELOW_CUTOFF]]}


def legacy_view(item: dict, drop_rank: bool = False) -> dict:
    """The item as it was before provenance v2 (drops ADDED_ITEM_KEYS, and `rank`
    when the pre-provenance shape had none -- C v2 and D)."""
    drop = set(ADDED_ITEM_KEYS) | ({"rank"} if drop_rank else set())
    return {k: v for k, v in item.items() if k not in drop}


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

_REQUIRED_STEP_KEYS = ("node_type", "node_id", "edge_type_to_next")
_REQUIRED_ITEM_KEYS = ("question_id", "answer_id", "chunk_text", "source_stage", "hop", "rel_type",
                       "via_question_id", "owner_question_id", "rank", "anchor_id", "path", "alt_paths", "trust_weight", "is_accepted")
_HOP_BY_STAGE = {STAGE_GRAPH: (1, 2), STAGE_SEMANTIC: ("semantic",), STAGE_VECTOR: (None,),
                 STAGE_LOW: (1,), STAGE_HIGH: (2,)}


def _check_path_shape(path, stage: str, hop, answer_id, prefix: str) -> list[str]:
    errs: list[str] = []
    if not isinstance(path, list) or not path:
        return [f"{prefix}path is not a non-empty list"]
    for i, s in enumerate(path):
        if not isinstance(s, dict) or any(k not in s for k in _REQUIRED_STEP_KEYS):
            return [f"{prefix}path[{i}] lacks keys {_REQUIRED_STEP_KEYS}"]
    types = [s["node_type"] for s in path]
    edges = [s["edge_type_to_next"] for s in path[:-1]]
    if path[-1]["node_type"] != NODE_ANSWER or path[-1]["edge_type_to_next"] is not None:
        errs.append(f"{prefix}path must end at an Answer step with edge_type_to_next=None")
    if answer_id is not None and path[-1]["node_id"] != answer_id:
        errs.append(f"{prefix}path ends at {path[-1]['node_id']}, item answer_id is {answer_id}")
    if any(s["node_id"] is None for s in path):
        errs.append(f"{prefix}path has a step with node_id None")
    if stage == STAGE_VECTOR:
        if len(path) != 1:
            errs.append(f"{prefix}vector_only path must have exactly 1 step, has {len(path)}")
        return errs
    one_hop = stage == STAGE_LOW or (stage == STAGE_GRAPH and hop == 1)
    two_hop = stage == STAGE_HIGH or (stage == STAGE_GRAPH and hop == 2)
    if one_hop:
        if types != [NODE_QUESTION, NODE_ANSWER] or edges[0] not in FETCH_EDGES:
            errs.append(f"{prefix}1-hop path must be Question -[HAS_*]-> Answer, got {types} {edges}")
    elif two_hop:
        if types != [NODE_QUESTION, NODE_QUESTION, NODE_ANSWER] or len(edges) != 2 \
                or edges[0] not in RELATION_EDGES or edges[1] not in FETCH_EDGES:
            errs.append(f"{prefix}2-hop path must be Question -[relation]-> Question -[HAS_*]-> Answer, got {types} {edges}")
    elif stage == STAGE_SEMANTIC:
        if types != [NODE_QUESTION, NODE_QUESTION, NODE_ANSWER] or len(edges) != 2 \
                or edges[0] != EDGE_SEMANTIC_SIM or edges[1] not in FETCH_EDGES:
            errs.append(f"{prefix}semantic path must be Question -[SEMANTIC_SIM]-> Question -[HAS_*]-> Answer, got {types} {edges}")
    else:
        errs.append(f"{prefix}unknown source_stage/hop {stage!r}/{hop!r}")
    return errs


def _rel_type_ok(rel_type, stage: str, path: list) -> bool:
    edges = [s["edge_type_to_next"] for s in path[:-1]]
    if not edges:
        return rel_type is None
    options = {"->".join(edges)}
    if stage == STAGE_HIGH:
        options.add(edges[0])      # D's high_level rel_type is the r2 type only
    if stage == STAGE_SEMANTIC:
        options.add(edges[-1])     # semantic rel_type is the HAS_* type
    return rel_type in options


def validate_item(item: dict) -> list[str]:
    """Schema errors for one retrieved_context item ([] = valid)."""
    missing = [k for k in _REQUIRED_ITEM_KEYS if k not in item]
    if missing:
        return [f"missing keys {missing}"]
    errs: list[str] = []
    stage, hop, path = item["source_stage"], item["hop"], item["path"]
    if stage not in SOURCE_STAGES:
        return [f"unknown source_stage {stage!r}"]
    if hop not in _HOP_BY_STAGE[stage]:
        errs.append(f"hop {hop!r} not allowed for source_stage {stage!r}")
    if item["via_question_id"] != item["question_id"] or item["owner_question_id"] != item["question_id"]:
        errs.append("via_question_id / owner_question_id != question_id")
    if not isinstance(item["rank"], int) or item["rank"] < 1:
        errs.append("rank must be an int >= 1")
    if item["trust_weight"] is not None and not isinstance(item["trust_weight"], (int, float)):
        errs.append("trust_weight must be a number or None")
    errs += _check_path_shape(path, stage, hop, item["answer_id"], "")
    if errs:
        return errs
    if stage in (STAGE_VECTOR, STAGE_LOW, STAGE_HIGH) and item["trust_weight"] is not None:
        errs.append(f"{stage} trust_weight must be None (B has no edge; D does not use trust weighting)")
    if stage == STAGE_VECTOR:
        if item["anchor_id"] is not None:
            errs.append("vector_only anchor_id must be None")
    else:
        if item["anchor_id"] != path[0]["node_id"]:
            errs.append(f"anchor_id {item['anchor_id']} != path[0].node_id {path[0]['node_id']}")
        if path[-2]["node_id"] != item["via_question_id"]:
            errs.append("path[-2] (the Question owning the Answer) != via_question_id")
        if not _rel_type_ok(item["rel_type"], stage, path):
            errs.append(f"rel_type {item['rel_type']!r} does not match the path's edge types")
    if stage == STAGE_SEMANTIC:
        if item.get("seed_similarity") is None:
            errs.append("semantic item: seed_similarity missing")
    if not isinstance(item["alt_paths"], list):
        errs.append("alt_paths must be a list")
    else:
        for j, alt in enumerate(item["alt_paths"]):
            errs += _check_path_shape(alt.get("path"), alt.get("source_stage"), alt.get("hop"), item["answer_id"],
                                      f"alt_paths[{j}].")
    return errs


def validate_record(record: dict) -> list[str]:
    """Errors for a whole record: provenance_version plus every item."""
    errs = []
    if record.get("provenance_version") != PROVENANCE_VERSION:
        errs.append(f"provenance_version is {record.get('provenance_version')!r}, expected {PROVENANCE_VERSION!r}")
    if record.get("condition") in ("C", "D"):
        ns, bc = record.get("not_selected"), record.get("below_cutoff")
        if not isinstance(ns, dict) or not all(k in ns for k in ("answer_ids", "question_ids", "n_answer_ids", "n_question_ids")):
            errs.append("C/D record lacks not_selected {answer_ids, question_ids, n_answer_ids, n_question_ids}")
        elif len(ns["answer_ids"]) > TRACE_CAP_IDS or len(ns["question_ids"]) > TRACE_CAP_IDS:
            errs.append("not_selected lists exceed the cap")
        if not isinstance(bc, list) or len(bc) > TRACE_CAP_BELOW_CUTOFF or any(len(e) != 3 for e in bc):
            errs.append("below_cutoff must be a list of at most 20 [answer_id, rank, score]")
    for i, item in enumerate(record.get("retrieved_context") or []):
        errs += [f"item[{i}]: {e}" for e in validate_item(item)]
    return errs


# --- validation against the KG ----------------------------------------

# edge_lookup(src_type, src_id, edge_type, dst_type, dst_id) -> non-empty list when the directed edge
# exists in the KG, [] when it does not
EdgeLookup = Callable[[str, int, str, str, int], list]

_NODE_LABELS = (NODE_QUESTION, NODE_ANSWER)


def validate_path_in_kg(path: list, edge_lookup: EdgeLookup) -> list[str]:
    """Every KG edge of `path` exists. SEMANTIC_SIM steps are FAISS links, not KG
    edges: they are not looked up."""
    errs = []
    for a, b in zip(path, path[1:]):
        edge = a["edge_type_to_next"]
        if edge == EDGE_SEMANTIC_SIM:
            continue
        if not edge_lookup(a["node_type"], a["node_id"], edge, b["node_type"], b["node_id"]):
            errs.append(f"no {edge} edge {a['node_type']}({a['node_id']}) -> {b['node_type']}({b['node_id']}) in the KG")
    return errs


def validate_item_in_kg(item: dict, edge_lookup: EdgeLookup) -> list[str]:
    errs = validate_path_in_kg(item["path"], edge_lookup)
    for j, alt in enumerate(item.get("alt_paths") or []):
        errs += [f"alt_paths[{j}]: {e}" for e in validate_path_in_kg(alt["path"], edge_lookup)]
    return errs


def neo4j_edge_lookup(driver, database: str) -> EdgeLookup:
    """EdgeLookup over a live Neo4j (read-only). Labels and edge types are
    checked against whitelists before they are put in the query text."""
    allowed_edges = set(FETCH_EDGES) | set(RELATION_EDGES)

    def lookup(src_type, src_id, edge_type, dst_type, dst_id):
        if src_type not in _NODE_LABELS or dst_type not in _NODE_LABELS or edge_type not in allowed_edges:
            raise ValueError(f"not a KG edge: {src_type} -[{edge_type}]-> {dst_type}")
        with driver.session(database=database) as session:
            result = session.run(
                f"MATCH (s:{src_type} {{id: $s}})-[r:{edge_type}]->(d:{dst_type} {{id: $d}}) RETURN 1 AS w",
                s=int(src_id), d=int(dst_id))
            return [record["w"] for record in result]

    return lookup
