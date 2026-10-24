"""On-request provenance lookup: how was each cited source retrieved?

Default behaviour of the pipeline is unchanged -- the user gets the answer and its
citations only. This module is the opt-in read-back: given a run and a question it reads
the STORED run data (no Neo4j, no LLM, nothing is re-run) and shows, for every source the
answer cites with [SO-<id>], the path that retrieved it, plus the sources that were
retrieved but not cited.

Where the data lives: one record per question in the run's output file
(`llm/<condition>/results/*.jsonl`; archived runs under `results/_archive/...`). The record
carries `run_id`, `question_id`, `condition`, `llm_answer` (the answer text) and
`retrieved_context` (the items, each with its `path` since provenance_version v2), so answer
and provenance are joined by construction -- the [SO-<id>] in `llm_answer` is matched to the
items' `question_id` (llm/citations.py::extract_citations, the same rule NF2 uses).

    from llm.provenance_lookup import lookup_provenance, render_lookup
    result = lookup_provenance(run_id="...", question_id=123456)      # or results_file=Path(...)
    print(render_lookup(result))

    python -m llm.provenance_lookup --run-id <uuid> --question-id 123456 [--file run.jsonl] [--json]

What the path means: how the source was retrieved from the knowledge graph -- NOT why the
model used it (the model never sees the path). Condition A has no retrieval; B has no KG
path (vector hit); C shows the full path (anchor -> [question] -> answer, or the semantic-
expansion link); D shows its question/answer path. Records written before provenance, or
without `path`, give a clear message instead of a path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm.citations import extract_citations  # noqa: E402
from llm.common.item_provenance import EDGE_SEMANTIC_SIM, SUPPORTED_VERSIONS  # noqa: E402

CONDITION_DIRS = ("a_pure_llm", "b_rag", "c_graphrag", "d_lightrag")
NOTE = "Note: the path shows how the source was retrieved from the knowledge graph, not why the model used it."
SEMANTIC_NOTE = "SEMANTIC_SIM is a similarity link from the vector index, not a knowledge-graph edge."


def default_search_dirs() -> list[Path]:
    dirs = [REPO_ROOT / "llm" / c / "results" for c in CONDITION_DIRS]
    archive = REPO_ROOT / "results" / "_archive"
    if archive.is_dir():
        dirs += sorted(archive.glob("*/llm/*/results"))
    return [d for d in dirs if d.is_dir()]


def _first_record(path: Path) -> dict | None:
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    return json.loads(line)
    except (OSError, json.JSONDecodeError):
        return None
    return None


def find_run_file(run_id: str, search_dirs=None) -> Path | None:
    """The output .jsonl whose records carry this run_id (the UUID written into every record)."""
    for d in (search_dirs if search_dirs is not None else default_search_dirs()):
        for path in sorted(Path(d).glob("*.jsonl")):
            first = _first_record(path)
            if first and first.get("run_id") == run_id:
                return path
    return None


def path_text(path: list) -> str:
    """"Question 111 -(TAG_COOCCUR)-> Question 222 -(HAS_ACCEPTED_ANSWER)-> Answer 333"."""
    parts = []
    for step in path:
        parts.append(f"{step['node_type']} {step['node_id']}")
        if step.get("edge_type_to_next"):
            parts.append(f"-({step['edge_type_to_next']})->")
    return " ".join(parts)


def _item_view(item: dict, cond: str) -> dict:
    path = item.get("path")
    return {"source_id": item.get("question_id"), "answer_id": item.get("answer_id"), "rank": item.get("rank"),
            "source_stage": item.get("source_stage"), "hop": item.get("hop"),
            "path": path, "path_text": path_text(path) if path else None,
            "has_semantic_link": bool(path) and any(s.get("edge_type_to_next") == EDGE_SEMANTIC_SIM for s in path)}


def _result(status: str, message: str, **extra) -> dict:
    return {"status": status, "message": message, "note": NOTE, **extra}


def _read_records(path: Path, run_id: str | None, question_id: int) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("question_id") == question_id and (run_id is None or rec.get("run_id") == run_id):
                out.append(rec)
    return out


def _condition(rec: dict, path: Path | None) -> str | None:
    cond = rec.get("condition") or ((rec.get("config") or {}).get("condition"))
    if cond:
        return cond
    if path is not None:
        for c in "ABCD":
            if path.name.startswith(f"condition_{c.lower()}_"):
                return c
    return None


def lookup_provenance(run_id: str | None = None, question_id: int | None = None, *, results_file=None,
                      search_dirs=None) -> dict:
    """Provenance of the sources cited in the stored answer for (run_id, question_id).

    Returns a dict with `status` (ok | retrieval_only | no_retrieval | no_provenance | run_not_found |
    question_not_found | error), a human `message`, the `note`, and -- when a record is found --
    `condition`, `run_id`, `question_id`, `answer`, `cited` (one entry per cited [SO-id]: its retrieved
    items with `path` / `path_text`), `cited_not_retrieved` and `not_cited`. Never raises on old,
    partial or missing data."""
    if question_id is None or (run_id is None and results_file is None):
        return _result("error", "Give a question_id and either a run_id or the run's results file.")
    question_id = int(question_id)
    try:
        path = Path(results_file) if results_file else find_run_file(run_id, search_dirs)
        if path is None:
            return _result("run_not_found", f"No run file with run_id {run_id} was found "
                           "(runs written before run_id existed cannot be found by id; pass the results file).")
        if not path.exists():
            return _result("run_not_found", f"Results file not found: {path}")
        records = _read_records(path, run_id, question_id)
    except OSError as e:
        return _result("error", f"Could not read the run data: {e}")
    if not records:
        return _result("question_not_found", f"Question {question_id} has no record in {path.name}"
                       + (f" for run {run_id}." if run_id else "."), results_file=str(path))
    rec = records[-1]                                 # a resumed/re-run question: the last record wins
    cond = _condition(rec, path)
    base = {"results_file": str(path), "condition": cond, "run_id": rec.get("run_id"), "question_id": question_id,
            "answer": rec.get("llm_answer"), "n_records_for_question": len(records),
            "provenance_version": rec.get("provenance_version")}
    if cond == "A":
        return _result("no_retrieval", "Condition A has no retrieval: the answer was generated without any "
                       "context, so there are no sources and no paths.", **base)
    items = rec.get("retrieved_context")
    if not isinstance(items, list):
        return _result("no_provenance", "This record has no retrieved_context, so nothing can be shown.", **base)
    answer = rec.get("llm_answer")
    views = [_item_view(i, cond) for i in items]
    have_paths = any(v["path"] for v in views)
    version = rec.get("provenance_version")
    if answer is None:
        return _result("retrieval_only", "This is a retrieval-only record (no answer text), so no citations can be "
                       "matched; the retrieved sources are listed.", **base, cited=[], cited_not_retrieved=[],
                       not_cited=views)
    _, cited_ids, _, valid_ids = extract_citations(answer, items)
    cited = [{"source_id": sid, "items": [v for v in views if v["source_id"] == sid]} for sid in valid_ids]
    not_cited = [v for v in views if v["source_id"] not in set(valid_ids)]
    common = dict(base, cited=cited, cited_not_retrieved=[i for i in cited_ids if i not in valid_ids], not_cited=not_cited)
    if not have_paths or (version is not None and version not in SUPPORTED_VERSIONS):
        why = ("this record was written before per-item provenance existed (no provenance_version/path)"
               if version is None else f"its provenance_version {version!r} is not understood")
        return _result("no_provenance", f"No path can be shown: {why}. The retrieved and cited sources are listed "
                       "without paths.", **common)
    if cond == "B":
        return _result("ok", "Condition B retrieves by flat vector search: there is no knowledge-graph path, only "
                       "the vector-search rank of each source.", **common)
    return _result("ok", "", **common)


def render_lookup(result: dict) -> str:
    """Readable text for lookup_provenance()'s result."""
    lines = []
    head = [str(result.get("condition") or "?")]
    if result.get("run_id"):
        head.append(f"run {result['run_id']}")
    if result.get("question_id") is not None:
        head.append(f"question {result['question_id']}")
    lines.append("Provenance of cited sources -- " + ", ".join(head))
    if result.get("message"):
        lines.append(result["message"])
    cond = result.get("condition")
    semantic = False

    def describe(v):
        nonlocal semantic
        semantic = semantic or v["has_semantic_link"]
        where = f"rank {v['rank']}" if v.get("rank") else "rank ?"
        if v["path_text"]:
            label = f"{v['source_stage']}, {where}" if cond != "B" else f"vector search, {where}"
            return f"  - {v['path_text']}   [{label}]" if cond != "B" else f"  - Answer {v['answer_id']}   [{label}; no knowledge-graph path]"
        return f"  - Answer {v['answer_id']} of Question {v['source_id']}   [{where}; no path recorded]"

    for entry in result.get("cited") or []:
        lines.append(f"Cited [SO-{entry['source_id']}]:")
        lines += [describe(v) for v in entry["items"]]
    if result.get("status") in ("ok", "no_provenance") and not result.get("cited"):
        lines.append("The answer cites no source that was in the retrieved context.")
    if result.get("cited_not_retrieved"):
        lines.append("Cited but not in the retrieved context: " + ", ".join(f"SO-{i}" for i in result["cited_not_retrieved"]))
    not_cited = result.get("not_cited") or []
    if not_cited:
        lines.append("Retrieved but not cited: " + "; ".join(
            f"SO-{v['source_id']} (answer {v['answer_id']}, rank {v['rank'] if v.get('rank') else '?'})" for v in not_cited))
    if semantic:
        lines.append(SEMANTIC_NOTE)
    lines.append(result.get("note", NOTE))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", help="run_id (UUID) written into every record of the run")
    parser.add_argument("--question-id", type=int, required=True)
    parser.add_argument("--file", help="the run's results .jsonl (skips the search by run_id)")
    parser.add_argument("--search-dir", action="append", help="directory of run .jsonl files to search (repeatable)")
    parser.add_argument("--json", action="store_true", help="print the structured result instead of text")
    args = parser.parse_args(argv)
    result = lookup_provenance(args.run_id, args.question_id, results_file=args.file, search_dirs=args.search_dir)
    print(json.dumps(result, indent=1, default=str) if args.json else render_lookup(result))
    return 0 if result["status"] in ("ok", "no_retrieval", "no_provenance", "retrieval_only") else 1


if __name__ == "__main__":
    sys.exit(main())
