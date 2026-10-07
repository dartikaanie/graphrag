"""
stage1_sweep.py
=====================================
End-to-end stage-1 CLI runner for the C retrieval v3 α sweep (Step 4 of
docs/agent_prompt_c_retrieval_v3_devset.md, decision procedure in
docs/DECISION_C_SCORING.md). For the dev split and α in
DEFAULT_ALPHA_VALUES:

  (a) runs the context-relevance controls (positive, easy negative, hard
      negative; Amendment 2) with --ctxrel-version and STOPS if they fail;
  (b) runs retrieval-only C v3 for each α (no LLM calls -- free, config-
      hashed + manifested like every other run);
  (c) runs --ctxrel-version (primary judge) on every resulting context, deduped
      across α (via llm_judge_context_relevance_v1.load_items(), which
      already dedups on (question_id, answer_id, text_hash));
  (d) writes the stage-1 markdown report (per-α relevance metrics, hop/
      edge-type breakdown, Jaccard vs α=0 and α=1);
  (e) prints the selection helper's output with reasoning.

Retrieval (b) has NO API cost, so it runs first to get REAL dedup counts
-- the plan shown before confirmation is then exact, not an estimate of
an estimate. The only real-money step is context-relevance judging (controls +
main sweep), which is gated behind an explicit "show plan -> confirm ->
run" flow, same shape as the dashboard's judge-v1 plan/launch.

CARA PAKAI (run from the repo root; QUESTIONS_PARQUET/ANSWERS_PARQUET can
come from .env instead of flags, same as c_graphrag.py's own CLI)
----------------------------------------------------------------------------
    # (a) controls only -- cheap sanity check before committing to the sweep
    # (--control-sets hard_negative to judge one set only)
    python3 llm/c_graphrag/stage1_sweep.py --controls-only --ctxrel-version ctxrel-v2

    # (b) the full stage-1 sweep: builds the dev sample, runs retrieval-only
    # for every alpha (free), shows the plan, asks for y/N confirmation,
    # then (if confirmed) runs the controls and the context-relevance judging and
    # writes the report. Add --yes to skip the confirmation prompt, or
    # --plan-only to stop right after the plan (no API call at all).
    python3 llm/c_graphrag/stage1_sweep.py --ctxrel-version ctxrel-v2
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

DEFAULT_ALPHA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
RESULTS_DIR = Path(__file__).resolve().parent / "results"  # absolute -- stable regardless of invocation cwd


# ---------------------------------------------------------------------
# (b) Retrieval-only runs per alpha -- no API cost.
# ---------------------------------------------------------------------

class IncompleteRetrievalOutput(RuntimeError):
    """A retrieval-only output file does not hold exactly one valid record
    per dev question -- it must never be used as an α result."""


def scan_retrieval_output(output_path, sample_qids, config_hash: str) -> dict:
    """Reads a retrieval-only JSONL. A record is VALID iff it parses,
    its question_id is a dev question, its config_hash matches, and it
    has a `retrieved_context` list. Returns {"valid": {qid: line}, (first
    valid record per qid, file order), "duplicates": [qid, ...],
    "invalid_lines": [(line_no, reason), ...]}."""
    sample_qids = {int(q) for q in sample_qids}
    valid: dict[int, str] = {}
    duplicates, invalid = [], []
    path = Path(output_path)
    if not path.exists():
        return {"valid": valid, "duplicates": duplicates, "invalid_lines": invalid}
    with open(path, encoding="utf-8") as f:
        for n, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                invalid.append((n, "unparseable (truncated?) line"))
                continue
            qid = rec.get("question_id") if isinstance(rec, dict) else None
            if not isinstance(qid, int) or qid not in sample_qids:
                invalid.append((n, f"question_id {qid!r} is not a dev question"))
            elif rec.get("config_hash") != config_hash:
                invalid.append((n, f"config_hash {rec.get('config_hash')!r} != {config_hash!r}"))
            elif not isinstance(rec.get("retrieved_context"), list):
                invalid.append((n, "no retrieved_context list"))
            elif qid in valid:
                duplicates.append(qid)
            else:
                valid[qid] = line
    return {"valid": valid, "duplicates": duplicates, "invalid_lines": invalid}


def verify_retrieval_output_complete(output_path, sample_qids, config_hash: str) -> int:
    """Raises IncompleteRetrievalOutput unless the file holds exactly one
    valid record for EVERY dev question and nothing else. Returns n."""
    scan = scan_retrieval_output(output_path, sample_qids, config_hash)
    missing = sorted({int(q) for q in sample_qids} - set(scan["valid"]))
    if missing or scan["duplicates"] or scan["invalid_lines"]:
        raise IncompleteRetrievalOutput(
            f"{output_path}: {len(scan['valid'])}/{len(set(sample_qids))} dev questions complete; "
            f"missing={missing[:10]}{'...' if len(missing) > 10 else ''} duplicates={scan['duplicates'][:10]} "
            f"invalid_lines={scan['invalid_lines'][:5]}"
        )
    return len(scan["valid"])


def _resume_sidecar(output_path) -> Path:
    return Path(str(output_path) + ".resume.json")  # git-ignored (*.resume.json)


def prepare_retrieval_resume(output_path, sample_qids, config_hash: str, git_commit: str | None) -> set[int]:
    """Decides what an existing retrieval-only file may contribute:
      - no file, no sidecar, or a sidecar from a different config_hash /
        git commit (or with no commit) -> start FRESH (file removed): a
        record written by different code must never be mixed in;
      - otherwise -> keep only VALID records (scan_retrieval_output), one
        per question, rewrite the file with just those (dropping
        truncated, foreign, or duplicate lines), and return their qids
        as `already_done`.
    Writes/refreshes the sidecar {config_hash, git_commit} either way."""
    path, sidecar = Path(output_path), _resume_sidecar(output_path)
    meta = None
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text())
        except (json.JSONDecodeError, OSError):
            meta = None
    reusable = (path.exists() and isinstance(meta, dict) and git_commit is not None
                and meta.get("config_hash") == config_hash and meta.get("git_commit") == git_commit)
    already_done: set[int] = set()
    if reusable:
        scan = scan_retrieval_output(path, sample_qids, config_hash)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for line in scan["valid"].values():
                f.write(line + "\n")
        os.replace(tmp, path)
        already_done = set(scan["valid"])
    elif path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({"config_hash": config_hash, "git_commit": git_commit}))
    return already_done


def run_retrieval_only_for_alpha(
    sample_df, alpha: float, driver, database, faiss_index, faiss_ids, faiss_embeddings,
    id_to_row, all_answer_ids_map, embed_model, embedding_cache, questions_parquet: str,
    output_dir: str, seed: int, n_sample: int, provider: str = "openai", model: str = "gpt-4o-mini",
    top_k: int = 5, n_anchor: int = 3, n_semantic_expansion: int = 3, token_chunk_limit: int = 400,
    oversample_pool: int = 1536, ctxrel_version: str | None = None, git_commit: str | None = None,
) -> dict:
    """Runs C retrieval-only v3 for one alpha on the dev split, writes a
    config-hashed output file + manifest + run_history entry (so
    _run_metadata.resolve_run_metadata() -- which ctxrel-v1's load_items()
    requires -- can resolve it), and returns {"output_path", "config_hash",
    "n_questions", "n_resumed", "history_path"}.

    RESUME (prepare_retrieval_resume): a partial file left by a crashed
    run of the SAME config and SAME git commit is resumed -- only its
    valid per-question records are kept, and only the missing questions
    are retrieved. COMPLETENESS: before the manifest/run_history entry is
    written, verify_retrieval_output_complete() requires exactly one valid
    record per dev question; otherwise IncompleteRetrievalOutput is raised
    and NO run_history entry exists, so load_items() refuses the file
    (RunMetadataNotFoundError) -- a partial file can never become an α
    result."""
    import c_graphrag as cg
    from llm.manifest import compute_config_hash, write_manifest

    fusion_mode = "uniform" if alpha == 0 else "trust_weighted"
    output_path = cg.build_output_path(
        output_dir, provider, model, n_sample, seed, fusion_mode,
        oversample_pool=oversample_pool, top_k=top_k, n_anchor=n_anchor,
        n_semantic_expansion=n_semantic_expansion, c_retrieval_version="v3", alpha=alpha,
        sample_split="dev",
    )
    config = cg.build_config(
        provider, model, n_sample, seed, oversample_pool, top_k, n_anchor, n_semantic_expansion,
        fusion_mode, cg.DEFAULT_FUSION_W_PATH_TRUST, cg.DEFAULT_FUSION_W_ANSWER_INTRINSIC_TRUST,
        cg.DEFAULT_SEMANTIC_EXPANSION_TRUST_CAP, True, True,
        c_retrieval_version="v3", alpha=alpha, sample_split="dev",
    )
    config_hash = compute_config_hash(config)
    sample_qids = [int(q) for q in sample_df["Id"]]
    already_done = prepare_retrieval_resume(output_path, sample_qids, config_hash, git_commit)

    run_started_at = datetime.now(timezone.utc)
    results, stats = cg.process_sample(
        sample_df, None, None, embed_model, driver, database,
        faiss_index, faiss_ids, faiss_embeddings, id_to_row, all_answer_ids_map,
        top_k, n_anchor, n_semantic_expansion, token_chunk_limit, model, output_path,
        already_done=already_done,
        c_retrieval_version="v3", alpha=alpha, embedding_cache=embedding_cache, retrieval_only=True,
        config=config, config_hash=config_hash,
    )
    finished_at = datetime.now(timezone.utc)
    n_complete = verify_retrieval_output_complete(output_path, sample_qids, config_hash)

    history_path = cg.append_run_history({
        "prompt_version": "retrieval-only", "c_retrieval_version": "v3", "alpha": alpha,
        "sample_split": "dev", "run_started_at": run_started_at.isoformat(), "condition": "C",
        "status": "success", "provider": provider, "model": model, "n_sample_target": n_sample,
        "n_processed": n_complete, "n_resumed": len(already_done), "seed": seed,
        "oversample_pool": oversample_pool,
        "fusion_mode": fusion_mode, "output_path": str(output_path), "config_hash": config_hash,
        "duration_sec": round((finished_at - run_started_at).total_seconds(), 1),
        "retrieval_only": True, "ctxrel_version": ctxrel_version, "git_commit": git_commit,
        "embedding_cache_disabled": getattr(embedding_cache, "disabled", None),
    })
    write_manifest(
        output_path, run_label=None, status="completed",
        config=config, config_hash=config_hash,
        started_at_utc=run_started_at.isoformat(), finished_at_utc=finished_at.isoformat(),
        item_counts={"attempted": len(sample_df), "succeeded": n_complete, "failed": 0,
                     "resumed": len(already_done)},
        prompt_version="retrieval-only", judge_version=ctxrel_version,
    )
    return {"output_path": str(output_path), "config_hash": config_hash, "n_questions": n_complete,
            "n_resumed": len(already_done), "history_path": str(history_path)}


# ---------------------------------------------------------------------
# (a) Controls -- docs/DECISION_C_SCORING.md Amendments 1 and 2. Control
# data, written to results/controls/, never mixed with real sweep results.
# ---------------------------------------------------------------------

CONTROL_SETS = ("positive", "easy_negative", "hard_negative")
# (labels that count as a match, minimum match rate) -- Amendment 2.
CONTROL_THRESHOLDS = {
    "positive": (("RELEVANT",), 0.90),
    "easy_negative": (("IRRELEVANT",), 0.90),
    "hard_negative": (("PARTIAL", "IRRELEVANT"), 0.80),  # NOT RELEVANT
}
CONTROL_EXPECTED = {"positive": "RELEVANT", "easy_negative": "IRRELEVANT", "hard_negative": "NOT_RELEVANT"}
CONTROL_JUDGE_IDS = ("primary", "secondary")  # primary decides; secondary reported for reference
CONTROLS_DIR = RESULTS_DIR / "controls"


def _tagset(raw) -> set:
    import re

    return set(re.findall(r"<([^>]+)>", str(raw or "")))


def build_control_items(sample_df, token_chunk_limit: int = 400, seed: int = 42,
                        hard_pool_df=None, related_ids: dict | None = None,
                        question_vectors: dict | None = None,
                        control_sets=CONTROL_SETS) -> tuple[list[dict], dict]:
    """Per dev question in `sample_df` (needs Id/Title/Tags/
    AcceptedAnswerId/AcceptedAnswerBody):
      - positive: its OWN accepted answer -> expected RELEVANT.
      - easy_negative: the accepted answer of a DIFFERENT dev question
        sharing NO tag with it, drawn with random.Random(seed).
      - hard_negative (Amendment 3): donors are `hard_pool_df` (the
        candidate pool MINUS the test sample -- see
        _build_hard_negative_pool), excluding the dev question itself and
        its IS_RELATED_TO neighbours (`related_ids`: {question_id:
        set(neighbor ids)}, either direction). Eligible = shares >= 1
        tag. Selected = the eligible donor with the HIGHEST cosine
        similarity between question embeddings (`question_vectors`:
        {question_id: L2-normalized vector}, the FAISS-cache vectors);
        ties by question id ascending; no randomness. Eligible donors
        without a cached vector are skipped (counted in info). Records
        donor_cosine_sim / n_shared_tags / shared_tags. Expected NOT
        RELEVANT.
    Easy-negative donor lists are sorted by Id before drawing. chunk_text
    is built EXACTLY like a retrieved item in c_graphrag.fuse_and_rank():
    "Q: <donor title>\nA: <answer body, raw HTML>", truncated to
    `token_chunk_limit` cl100k_base tokens.

    Returns (items, info): info["no_hard_negative"] / ["no_easy_negative"]
    list the dev question ids with no eligible donor (excluded from that
    set's denominator, per Amendments 2/3); info["hard_negative_missing_
    vector"] lists dev question ids with no cached vector and
    info["n_eligible_donors_without_vector"] counts skipped donors.
    """
    import random

    import tiktoken

    import c_graphrag as cg

    enc = tiktoken.get_encoding("cl100k_base")
    rows = sample_df.to_dict("records")
    related_ids = related_ids or {}

    def _item(question_row, source_row, control_type):
        full = f"Q: {source_row['Title']}\nA: {source_row['AcceptedAnswerBody']}"
        n_tokens_full = len(enc.encode(full))
        return {
            "question_id": int(question_row["Id"]), "answer_id": int(source_row["AcceptedAnswerId"]),
            "source_question_id": int(source_row["Id"]),
            "chunk_text": cg._truncate_doc_text(enc, full, token_chunk_limit),
            "title": question_row["Title"], "tags": question_row["Tags"],
            "expected_label": CONTROL_EXPECTED[control_type], "control_type": control_type,
            "n_tokens_full": n_tokens_full, "truncated": n_tokens_full > token_chunk_limit,
            "donor_cosine_sim": None, "n_shared_tags": None, "shared_tags": None,
        }

    items: list[dict] = []
    info = {"no_easy_negative": [], "no_hard_negative": [], "hard_negative_missing_vector": [],
            "n_eligible_donors_without_vector": 0}
    if "positive" in control_sets:
        items += [_item(r, r, "positive") for r in rows]

    if "easy_negative" in control_sets:
        rng = random.Random(seed)
        donors_all = sorted(rows, key=lambda d: int(d["Id"]))
        for r in rows:
            donors = [d for d in donors_all if d["Id"] != r["Id"] and not (_tagset(d["Tags"]) & _tagset(r["Tags"]))]
            if donors:
                items.append(_item(r, rng.choice(donors), "easy_negative"))
            else:
                info["no_easy_negative"].append(int(r["Id"]))

    if "hard_negative" in control_sets:
        import numpy as np

        if hard_pool_df is None or question_vectors is None:
            raise ValueError("hard_negative controls need hard_pool_df (non-test candidate pool with accepted "
                             "answers) and question_vectors (FAISS-cache question embeddings)")
        pool = [d for d in hard_pool_df.to_dict("records") if isinstance(d.get("AcceptedAnswerBody"), str)]
        for r in rows:
            qid = int(r["Id"])
            linked = related_ids.get(qid, set())
            tags_r = _tagset(r["Tags"])
            eligible = [d for d in pool
                        if int(d["Id"]) != qid and int(d["Id"]) not in linked and (_tagset(d["Tags"]) & tags_r)]
            q_vec = question_vectors.get(qid)
            if q_vec is None:
                info["hard_negative_missing_vector"].append(qid)
                info["no_hard_negative"].append(qid)
                continue
            scored = []
            for d in eligible:
                d_vec = question_vectors.get(int(d["Id"]))
                if d_vec is None:
                    info["n_eligible_donors_without_vector"] += 1
                    continue
                scored.append((float(np.dot(np.asarray(q_vec, dtype=np.float64),
                                            np.asarray(d_vec, dtype=np.float64))), int(d["Id"]), d))
            if not scored:
                info["no_hard_negative"].append(qid)
                continue
            sim, _, donor = min(scored, key=lambda t: (-t[0], t[1]))
            shared = sorted(_tagset(donor["Tags"]) & tags_r)
            item = _item(r, donor, "hard_negative")
            item.update({"donor_cosine_sim": round(sim, 6), "n_shared_tags": len(shared), "shared_tags": shared,
                         "n_eligible_donors": len(scored)})
            items.append(item)

    return items, info


def fetch_is_related_to_neighbors(driver, database: str, question_ids: list[int]) -> dict[int, set]:
    """{question_id: ids of Questions linked by IS_RELATED_TO (Linked/
    Duplicate), EITHER direction} -- the hard-negative exclusion set."""
    out = {int(q): set() for q in question_ids}
    with driver.session(database=database) as session:
        result = session.run(
            """
            UNWIND $ids AS qid
            MATCH (q:Question {id: qid})-[:IS_RELATED_TO]-(o:Question)
            RETURN qid, collect(DISTINCT o.id) AS neighbors
            """,
            ids=[int(q) for q in question_ids],
        )
        for rec in result:
            out[int(rec["qid"])] = {int(n) for n in rec["neighbors"]}
    return out


CONTROL_OUTPUT_PREFIX = "llm/c_graphrag/results/controls/"


REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_relative(path) -> str | None:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def parse_porcelain_z(raw: str) -> list[tuple[str, str]]:
    """`git status --porcelain -z` -> [(XY, path)] (rename/copy entries
    carry an extra NUL-separated source path, which is skipped). -z
    avoids git's quoting of paths with spaces/special characters."""
    tokens = raw.split("\0")
    entries, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if not tok:
            continue
        xy, path = tok[:2], tok[3:]
        entries.append((xy, path))
        if "R" in xy or "C" in xy:
            i += 1
    return entries


def git_state(exempt_paths=()) -> dict:
    """Commit + working-tree state of the repo, recorded on every control
    output/manifest. `git_tree_clean` ignores ONLY: entries under
    results/controls/ (control outputs -- an earlier control run's files
    must not make the next run "dirty"), the exact files in `exempt_paths`
    (this same run's own outputs, e.g. its retrieval-only files and run
    history), and git-ignored files (never listed by git status). Every
    other change -- modified, staged, deleted, or untracked -- is dirty.
    All entries are recorded verbatim."""
    import subprocess

    def _git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout

    exempt = {r for r in (_repo_relative(p) for p in exempt_paths) if r}
    try:
        commit = _git("rev-parse", "HEAD").strip()
        entries = parse_porcelain_z(_git("status", "--porcelain", "-z", "--untracked-files=all"))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return {"git_commit": None, "git_tree_clean": False, "git_status_porcelain": [f"<git error: {e}>"],
                "git_exempted": []}
    exempted = [p for _, p in entries if p.startswith(CONTROL_OUTPUT_PREFIX) or p in exempt]
    relevant = [p for _, p in entries if p not in exempted]
    return {"git_commit": commit, "git_tree_clean": not relevant,
            "git_status_porcelain": [f"{xy} {p}" for xy, p in entries], "git_exempted": exempted}


def _load_question_bodies_text(questions_parquet: str, question_ids: list[int]) -> dict[int, str]:
    """Question body as the judge sees it in the REAL stage-1 judging
    (llm_judge_context_relevance_v1.run_batch): html_to_text(Body)."""
    from _judge_common import html_to_text, load_question_bodies

    bodies_html = load_question_bodies(questions_parquet, question_ids)
    return {qid: html_to_text(html) if html else "" for qid, html in bodies_html.items()}


def summarize_controls(results: list[dict], judge_ids, deciding_judge_id: str = "primary") -> dict:
    """Per judge and control set present in `results`: label % (call
    failures / parse errors counted in the denominator as non-matching)
    and pass/fail against CONTROL_THRESHOLDS. A judge's overall `pass` is
    None unless ALL THREE sets were judged (a partial run, e.g. hard
    negatives only, cannot pass or fail the judge). `passed` is the
    deciding judge's overall result."""
    summary = {}
    for jid in judge_ids:
        per_set = {}
        for cset in CONTROL_SETS:
            rs = [r for r in results if r["judge_id"] == jid and r["control_type"] == cset]
            if not rs:
                continue
            n = len(rs)
            counts = {lab: sum(1 for r in rs if r["label"] == lab) for lab in ("RELEVANT", "PARTIAL", "IRRELEVANT")}
            counts["ERROR"] = n - sum(counts.values())
            ok_labels, threshold = CONTROL_THRESHOLDS[cset]
            n_ok = sum(counts[lab] for lab in ok_labels)
            per_set[cset] = {
                "n": n, "counts": counts,
                "pct": {lab: round(100 * c / n, 1) for lab, c in counts.items()},
                "pct_not_relevant": round(100 * (counts["PARTIAL"] + counts["IRRELEVANT"]) / n, 1),
                "match_rate": round(n_ok / n, 4), "threshold": threshold, "pass": n_ok / n >= threshold,
            }
        complete = all(c in per_set for c in CONTROL_SETS)
        summary[jid] = {**per_set, "complete": complete,
                        "pass": all(per_set[c]["pass"] for c in CONTROL_SETS) if complete else None}
    return {"per_judge": summary, "deciding_judge_id": deciding_judge_id,
            "passed": summary.get(deciding_judge_id, {}).get("pass")}


def run_controls(sample_df, ctxrel_version: str, judge_ids=CONTROL_JUDGE_IDS, questions_parquet: str = "",
                 workers: int = 4, deciding_judge_id: str = "primary", token_chunk_limit: int = 400,
                 seed: int = 42, control_sets=CONTROL_SETS, hard_pool_df=None,
                 related_ids: dict | None = None, question_vectors: dict | None = None,
                 require_clean_tree: bool = True, exempt_paths=()) -> dict:
    """Judges every control item in `control_sets` with every judge in
    `judge_ids`, using context-relevance prompt `ctxrel_version` and the
    SAME judge input as the real stage-1 judging (title + tags +
    html_to_text(question body) + one chunk_text). Writes one JSONL line
    per judgment (label + reason) to results/controls/<version>_controls_
    <sets>_<UTC timestamp>.jsonl, a summary JSON, and a manifest."""
    from concurrent.futures import ThreadPoolExecutor

    import llm_judge_context_relevance_v1 as ctxrel
    from judge_clients import get_judge_client
    from llm.manifest import write_manifest

    build = ctxrel.get_prompt_builder(ctxrel_version)
    if isinstance(judge_ids, str):
        judge_ids = (judge_ids,)
    if deciding_judge_id not in judge_ids:
        deciding_judge_id = judge_ids[0]

    git = git_state(exempt_paths)
    if require_clean_tree and not git["git_tree_clean"]:
        dirty = [e for e in git["git_status_porcelain"] if e[3:] not in git["git_exempted"]]
        return {"passed": False, "results": [], "info": {}, "git": git,
                "error": "working tree is not clean (outside results/controls/ and this run's own outputs) -- "
                         f"commit the code first; dirty entries: {dirty}"}

    items, info = build_control_items(sample_df, token_chunk_limit=token_chunk_limit, seed=seed,
                                      hard_pool_df=hard_pool_df, related_ids=related_ids,
                                      question_vectors=question_vectors, control_sets=control_sets)
    missing = [c for c in control_sets if not any(i["control_type"] == c for i in items)]
    if missing:
        return {"passed": False, "results": [], "info": info,
                "error": f"could not build any control items for set(s) {missing} from this dev sample"}

    started = datetime.now(timezone.utc)
    clients = {jid: get_judge_client(jid) for jid in judge_ids}
    bodies = _load_question_bodies_text(questions_parquet, sorted({i["question_id"] for i in items}))

    def _judge(task):
        item, jid = task
        client, config = clients[jid]
        messages = build(item["title"], bodies.get(item["question_id"], ""),
                         ctxrel.format_tags(item["tags"]), item["chunk_text"])
        payload, attempts = ctxrel.call_judge_with_retry(client, config, messages)
        parsed = ctxrel.parse_judgment(payload["raw"]) if payload else {"label": None, "reason": None, "parse_error": True}
        label = parsed["label"] if not parsed["parse_error"] else None
        ok_labels, _ = CONTROL_THRESHOLDS[item["control_type"]]
        return {
            **{k: item.get(k) for k in ("control_type", "question_id", "answer_id", "source_question_id",
                                         "n_tokens_full", "truncated", "chunk_text", "donor_cosine_sim",
                                         "n_shared_tags", "shared_tags", "n_eligible_donors")},
            "expected": item["expected_label"], "judge_id": jid, "judge_model": getattr(config, "model", None),
            "prompt_version": ctxrel_version, "body_available": item["question_id"] in bodies,
            "label": label, "reason": parsed.get("reason"), "match": label in ok_labels,
            "call_failed": payload is None, "raw_response": payload["raw"] if payload else None,
            "response_model": payload.get("response_model") if payload else None,
            "prompt_tokens": payload.get("prompt_tokens") if payload else None,
            "completion_tokens": payload.get("completion_tokens") if payload else None,
            "error": None if payload else (attempts[-1]["error"] if attempts else None),
        }

    tasks = [(item, jid) for item in items for jid in judge_ids]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(_judge, tasks))
    finished = datetime.now(timezone.utc)

    summary = summarize_controls(results, judge_ids, deciding_judge_id)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    sets_tag = "-".join(c for c in CONTROL_SETS if c in control_sets)
    CONTROLS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONTROLS_DIR / f"{ctxrel_version.replace('-', '_')}_controls_{sets_tag}_{stamp}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps({"timestamp_utc": stamp, "git_commit": git["git_commit"],
                                "git_tree_clean": git["git_tree_clean"], **r}, default=str) + "\n")
    config = {
        "ctxrel_version": ctxrel_version, "control_sets": list(control_sets), "judge_ids": list(judge_ids),
        "deciding_judge_id": deciding_judge_id, "token_chunk_limit": token_chunk_limit, "seed": seed,
        "thresholds": {k: v[1] for k, v in CONTROL_THRESHOLDS.items()},
        "dev_question_ids": [int(q) for q in sample_df["Id"]],
        "hard_negative_definition": "amendment-3" if "hard_negative" in control_sets else None,
        **git,
    }
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps({
        "timestamp_utc": stamp, **config, "n_items": len(items),
        "n_items_per_set": {c: sum(1 for i in items if i["control_type"] == c) for c in control_sets},
        **info, **summary,
    }, indent=2))
    write_manifest(
        out_path, run_label=None, config=config,
        started_at_utc=started.isoformat(), finished_at_utc=finished.isoformat(),
        item_counts={"attempted": len(results), "succeeded": sum(1 for r in results if not r["call_failed"]),
                     "failed": sum(1 for r in results if r["call_failed"])},
        prompt_version=ctxrel_version, judge_version=ctxrel_version, extra_output_files=[summary_path],
    )
    return {"passed": summary["passed"], "summary": summary, "results": results, "info": info, "git": git,
            "output_path": str(out_path), "summary_path": str(summary_path)}


# ---------------------------------------------------------------------
# Token/latency estimate for the plan -- same fallback-chain shape as
# backend/app/services/engine_service.py's _ctxrel_token_latency_stats,
# duplicated here (not imported) since backend/ depends on llm/, never
# the reverse.
# ---------------------------------------------------------------------

def _token_latency_stats(judge_model: str, results_dir: Path | None = None) -> tuple:
    results_dir = results_dir if results_dir is not None else RESULTS_DIR  # read at call time (monkeypatch-friendly)
    input_vals, output_vals, latency_vals = [], [], []
    if results_dir.exists():
        for path in sorted(results_dir.glob("ctxrel_v1_*.jsonl")):
            if "controls" in path.name or "smoke_test" in path.name:
                continue
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        if r.get("judge_model") != judge_model or r.get("call_failed"):
                            continue
                        if isinstance(r.get("prompt_tokens"), (int, float)):
                            input_vals.append(r["prompt_tokens"])
                        if isinstance(r.get("completion_tokens"), (int, float)):
                            output_vals.append(r["completion_tokens"])
                        if isinstance(r.get("latency_s"), (int, float)):
                            latency_vals.append(r["latency_s"])
            except OSError:
                continue
    if input_vals and output_vals:
        return (sum(input_vals) / len(input_vals), sum(output_vals) / len(output_vals),
                sum(latency_vals) / len(latency_vals) if latency_vals else None, "from history")
    return None, None, None, None


def build_plan(unique_items: int, n_controls: int, judge_id: str, workers: int = 4) -> dict:
    from judge_clients import JUDGE_REGISTRY

    config = JUDGE_REGISTRY.get(judge_id)
    total_items = unique_items + n_controls
    if config is None:
        return {"items": total_items, "unique_items": unique_items, "n_controls": n_controls,
                "judge_id": judge_id, "cost_unknown": True, "est_cost_usd": None, "est_time_sec": None}

    avg_in, avg_out, avg_lat, source = _token_latency_stats(config.model)
    if avg_in is None:
        avg_in, avg_out = config.default_input_tokens_per_item, config.default_output_tokens_per_item
        source = "default estimate"
    if avg_lat is None:
        avg_lat = config.default_latency_sec_per_item

    est_tokens_in = round(avg_in * total_items)
    est_tokens_out = round(avg_out * total_items)
    est_cost = round(est_tokens_in / 1_000_000 * config.price_per_m_input
                      + est_tokens_out / 1_000_000 * config.price_per_m_output, 4)
    w = max(1, workers)
    est_time_sec = round(total_items / w * avg_lat)

    return {
        "items": total_items, "unique_items": unique_items, "n_controls": n_controls,
        "judge_id": judge_id, "judge_model": config.model, "cost_unknown": False,
        "est_tokens_in": est_tokens_in, "est_tokens_out": est_tokens_out,
        "est_cost_usd": est_cost, "est_time_sec": est_time_sec, "token_source": source,
    }


def format_plan(plan: dict) -> str:
    lines = [
        "=== STAGE 1 SWEEP -- PLAN (no API calls made yet) ===",
        f"Controls: {plan['n_controls']} item(s)",
        f"Unique (question, context item) pairs to judge (deduped across all alpha values): {plan['unique_items']}",
        f"Total items: {plan['items']}  (judge_id={plan['judge_id']})",
    ]
    if plan["cost_unknown"]:
        lines.append("Estimated cost: unknown (judge_id not in registry)")
    else:
        lines.append(f"Estimated tokens (in/out): {plan['est_tokens_in']} / {plan['est_tokens_out']} "
                      f"(source: {plan['token_source']})")
        lines.append(f"Estimated cost: ${plan['est_cost_usd']:.4f}")
        lines.append(f"Estimated time: ~{plan['est_time_sec']}s")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# (d) Stage-1 markdown report
# ---------------------------------------------------------------------

def compute_alpha_metrics(retrieval_output_path: str, ctxrel_by_key: dict, selected_answer_ids_by_alpha: dict,
                           this_alpha: float) -> dict:
    """Per-alpha stage-1 metrics for ONE retrieval-only output file,
    joined against the ctxrel-v1 judged records (keyed by (question_id,
    answer_id, text_hash), content-based, never by run_id -- same join
    pattern as llm_judge_context_relevance_v1.compute_run_context_
    relevance()). `selected_answer_ids_by_alpha` is {alpha: {qid:
    set(answer_ids)}}, used for the Jaccard overlap vs alpha=0/alpha=1."""
    import llm_judge_context_relevance_v1 as ctxrel_mod

    LABEL_SCORE = {"RELEVANT": 1.0, "PARTIAL": 0.5, "IRRELEVANT": 0.0}
    n_judged = 0
    label_counts: dict[str, int] = {}
    scores = []
    sims_selected = []
    trusts_selected = []
    n_accepted_selected = 0
    n_total_selected = 0
    questions_with_relevant: set = set()
    all_questions: set = set()
    by_hop: dict = {}
    by_edge_type: dict = {}

    with open(retrieval_output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = record["question_id"]
            all_questions.add(qid)
            for item in record.get("retrieved_context") or []:
                n_total_selected += 1
                if item.get("is_accepted"):
                    n_accepted_selected += 1
                sims_selected.append(item.get("sim"))
                trusts_selected.append(item.get("trust"))

                key = (qid, item.get("answer_id"), ctxrel_mod._text_hash(item.get("chunk_text", "") or ""))
                judged = ctxrel_by_key.get(key)
                if judged is None or judged.get("label") not in LABEL_SCORE:
                    continue
                n_judged += 1
                label = judged["label"]
                label_counts[label] = label_counts.get(label, 0) + 1
                scores.append(LABEL_SCORE[label])
                if label == "RELEVANT":
                    questions_with_relevant.add(qid)

                hop = item.get("hop")
                by_hop.setdefault(hop, {"n": 0, "n_relevant": 0, "n_partial": 0})
                by_hop[hop]["n"] += 1
                if label == "RELEVANT":
                    by_hop[hop]["n_relevant"] += 1
                elif label == "PARTIAL":
                    by_hop[hop]["n_partial"] += 1

                rel_type = item.get("rel_type")
                by_edge_type.setdefault(rel_type, {"n": 0, "n_relevant": 0, "n_partial": 0})
                by_edge_type[rel_type]["n"] += 1
                if label == "RELEVANT":
                    by_edge_type[rel_type]["n_relevant"] += 1
                elif label == "PARTIAL":
                    by_edge_type[rel_type]["n_partial"] += 1

    from ctxrel_sweep import jaccard_overlap

    this_selected = selected_answer_ids_by_alpha.get(this_alpha, {})

    def _overlap_vs(other_alpha):
        other_selected = selected_answer_ids_by_alpha.get(other_alpha, {})
        if not this_selected:
            return None
        overlaps = []
        for qid, aids in this_selected.items():
            other_aids = other_selected.get(qid, set())
            overlaps.append(jaccard_overlap(aids, other_aids))
        return round(sum(overlaps) / len(overlaps), 4) if overlaps else None

    return {
        "n_questions": len(all_questions),
        "n_questions_with_relevant": len(questions_with_relevant),
        "pct_questions_with_relevant": round(100 * len(questions_with_relevant) / len(all_questions), 2) if all_questions else None,
        "pct_relevant": round(100 * label_counts.get("RELEVANT", 0) / n_judged, 2) if n_judged else None,
        "pct_partial": round(100 * label_counts.get("PARTIAL", 0) / n_judged, 2) if n_judged else None,
        "pct_irrelevant": round(100 * label_counts.get("IRRELEVANT", 0) / n_judged, 2) if n_judged else None,
        "mean_relevance_score": round(sum(scores) / len(scores), 4) if scores else None,
        "mean_sim_selected": round(sum(s for s in sims_selected if s is not None) / len(sims_selected), 4) if sims_selected else None,
        "mean_trust_selected": round(sum(t for t in trusts_selected if t is not None) / len(trusts_selected), 4) if trusts_selected else None,
        "pct_accepted_selected": round(100 * n_accepted_selected / n_total_selected, 2) if n_total_selected else None,
        "jaccard_vs_alpha1": _overlap_vs(1.0),
        "jaccard_vs_alpha0": _overlap_vs(0.0),
        "by_hop": by_hop, "by_edge_type": by_edge_type,
    }


def build_stage1_markdown(per_alpha_metrics: dict[float, dict], selection: dict,
                          ctxrel_version: str | None = None) -> str:
    lines = [
        "# Stage 1 Report -- C retrieval v3 alpha sweep", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        f"Context-relevance judge version: {ctxrel_version}", "",
        "| alpha | % q w/ >=1 RELEVANT | % RELEVANT | % PARTIAL | % IRRELEVANT | mean score | mean sim (sel) | mean trust (sel) | % accepted (sel) | Jaccard vs a=0 | Jaccard vs a=1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for alpha in sorted(per_alpha_metrics):
        m = per_alpha_metrics[alpha]
        lines.append(
            f"| {alpha} | {m['pct_questions_with_relevant']} | {m['pct_relevant']} | {m['pct_partial']} | "
            f"{m['pct_irrelevant']} | {m['mean_relevance_score']} | {m['mean_sim_selected']} | "
            f"{m['mean_trust_selected']} | {m['pct_accepted_selected']} | {m['jaccard_vs_alpha0']} | {m['jaccard_vs_alpha1']} |"
        )
    lines.append("")
    lines.append("## Relevance by hop and edge type (per alpha)")
    lines.append("")
    for alpha in sorted(per_alpha_metrics):
        m = per_alpha_metrics[alpha]
        lines.append(f"**alpha={alpha}**")
        lines.append("")
        lines.append("| hop | n | % RELEVANT | % PARTIAL |")
        lines.append("|---|---|---|---|")
        for hop, h in sorted(m["by_hop"].items(), key=lambda kv: str(kv[0])):
            pct_rel = round(100 * h["n_relevant"] / h["n"], 2) if h["n"] else None
            pct_part = round(100 * h["n_partial"] / h["n"], 2) if h["n"] else None
            lines.append(f"| {hop} | {h['n']} | {pct_rel} | {pct_part} |")
        lines.append("")
        lines.append("| edge type | n | % RELEVANT | % PARTIAL |")
        lines.append("|---|---|---|---|")
        for et, e in sorted(m["by_edge_type"].items(), key=lambda kv: str(kv[0])):
            pct_rel = round(100 * e["n_relevant"] / e["n"], 2) if e["n"] else None
            pct_part = round(100 * e["n_partial"] / e["n"], 2) if e["n"] else None
            lines.append(f"| {et} | {e['n']} | {pct_rel} | {pct_part} |")
        lines.append("")

    lines.append("## Selection helper output")
    lines.append("")
    lines.append(f"keep_current: {selection['keep_current']}")
    if not selection["keep_current"]:
        lines.append(f"top_two (stage 2 candidates): {selection['top_two']}")
    lines.append(f"reasoning: {selection['reasoning']}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# CLI orchestration -- wires (a)-(e) together. Setup mirrors
# c_graphrag.py::main()'s own setup steps (same functions, same
# defaults) so the dev-split sample/candidate pool/FAISS cache/embed
# model are built exactly the way every other C run builds them.
# ---------------------------------------------------------------------

def _build_dev_sample(questions_parquet: str, answers_parquet: str, n_dev: int, seed: int, dev_offset: int):
    import duckdb

    import c_graphrag as cg

    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")

    oversample_pool = 384 * 4  # MAX_PLANNED_N_SAMPLE * 4, same pool as every other condition
    candidates = cg.get_candidate_questions(con, questions_parquet, answers_parquet, oversample_pool, seed)
    candidates = cg.filter_by_token_limit(candidates)
    sample_df = cg.sample_questions_split(candidates, n_dev, seed, split="dev", dev_offset=dev_offset)

    accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
    answers_df = cg.get_accepted_answers(con, answers_parquet, accepted_ids)
    sample_df = sample_df.merge(answers_df, on="AcceptedAnswerId", how="left")
    sample_df = sample_df.dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)

    eval_question_ids = sample_df["Id"].astype(int).tolist()
    all_answer_ids_map = cg.get_all_answer_ids_for_questions(con, answers_parquet, eval_question_ids)
    con.close()
    return sample_df, all_answer_ids_map, oversample_pool, candidates


TEST_SAMPLE_N = 384  # docs/DECISION_C_SCORING.md: test sample = 0-based positions 0-383


def exclude_test_sample(candidates, seed: int = 42):
    """The candidate pool without the test sample (0-based positions
    0..TEST_SAMPLE_N-1 of the seed's permutation)."""
    import c_graphrag as cg

    test_ids = set(cg.sample_questions_split(candidates, TEST_SAMPLE_N, seed, split="test")["Id"].astype(int))
    return candidates[~candidates["Id"].astype(int).isin(test_ids)]


def _build_hard_negative_pool(answers_parquet: str, candidates, seed: int = 42):
    """Amendment 3 donor pool: the candidate pool (same pool/filter as the
    test and dev samples) MINUS the test sample (0-based positions 0-383 of
    the same permutation, via sample_questions_split(split="test")), joined
    with each question's accepted answer body."""
    import duckdb

    import c_graphrag as cg

    candidates = exclude_test_sample(candidates, seed)
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    pool = candidates[["Id", "Title", "Tags", "AcceptedAnswerId"]].dropna(subset=["AcceptedAnswerId"])
    answers_df = cg.get_accepted_answers(con, answers_parquet, pool["AcceptedAnswerId"].unique().tolist())
    con.close()
    return pool.merge(answers_df, on="AcceptedAnswerId", how="inner").reset_index(drop=True)


def load_question_vectors(kg_workspace_dir: str, question_ids) -> dict:
    """{question_id: vector} from the FAISS-cache embeddings
    (question_embeddings.f32 memmap + question_embeddings_ivf_ids.npy,
    written L2-normalized by 11_densify_embedding_similarity.py) -- the
    SAME vectors C's anchoring uses. Reads only the requested rows; the
    FAISS index itself is not loaded. Ids absent from the cache are
    simply missing from the result."""
    import numpy as np

    import c_graphrag as cg

    ws = Path(kg_workspace_dir)
    n_total = json.loads((ws / "embed_checkpoint.json").read_text())["next_row"]
    ids_arr = np.load(ws / "question_embeddings_ivf_ids.npy")
    embeddings = np.memmap(ws / "question_embeddings.f32", dtype="float32", mode="r", shape=(n_total, cg.EMBED_DIM))
    wanted = {int(q) for q in question_ids}
    out = {}
    for row, qid in enumerate(ids_arr):
        qid = int(qid)
        if qid in wanted:
            out[qid] = np.array(embeddings[row], dtype=np.float32)
    return out


def _prepare_control_inputs(answers_parquet: str, candidates, sample_df, control_sets,
                            kg_workspace_dir: str | None = None, seed: int = 42):
    """(hard_pool_df, related_ids, question_vectors) -- all None unless
    hard negatives are requested."""
    if "hard_negative" not in control_sets:
        return None, None, None
    import c_graphrag as cg

    hard_pool_df = _build_hard_negative_pool(answers_parquet, candidates, seed)
    driver, database = cg.connect_neo4j(print)
    try:
        related_ids = fetch_is_related_to_neighbors(driver, database, sample_df["Id"].astype(int).tolist())
    finally:
        driver.close()
    question_vectors = load_question_vectors(
        kg_workspace_dir, set(hard_pool_df["Id"].astype(int)) | set(sample_df["Id"].astype(int)))
    print(f"      hard-negative pool (non-test): {len(hard_pool_df)} questions; dev questions with IS_RELATED_TO "
          f"neighbors: {sum(1 for v in related_ids.values() if v)}; cached vectors found: {len(question_vectors)}")
    return hard_pool_df, related_ids, question_vectors


def _build_selected_answer_ids_by_alpha(retrieval_output_paths: dict[float, str]) -> dict[float, dict[int, set]]:
    result: dict[float, dict[int, set]] = {}
    for alpha, path in retrieval_output_paths.items():
        per_question: dict[int, set] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                qid = record["question_id"]
                per_question[qid] = {item.get("answer_id") for item in record.get("retrieved_context") or []}
        result[alpha] = per_question
    return result


def run_stage1_sweep(
    questions_parquet: str, answers_parquet: str, kg_workspace_dir: str, embed_model_name: str,
    embedding_cache_path: str, n_dev: int, dev_offset: int, seed: int, judge_id: str, workers: int,
    plan_only: bool, yes: bool, controls_only: bool, ctxrel_version: str,
    control_sets=CONTROL_SETS, control_judge_ids=CONTROL_JUDGE_IDS, allow_dirty: bool = False,
) -> int:
    """Returns a process exit code (0 success, 1 controls failed/aborted)."""
    import c_graphrag as cg
    from embedding_cache import EmbeddingCache
    from llm_judge_context_relevance_v1 import load_ctxrel_records, load_items, run_batch

    from ctxrel_sweep import build_stage1_report, select_stage1_candidates

    print(f"[1/6] Building the dev sample (n={n_dev}, dev_offset={dev_offset} [1-based] -> "
          f"0-based iloc[{dev_offset - 1}:{dev_offset - 1 + n_dev}], seed={seed})...")
    sample_df, all_answer_ids_map, oversample_pool, candidates = _build_dev_sample(
        questions_parquet, answers_parquet, n_dev, seed, dev_offset,
    )
    print(f"      {len(sample_df)} dev questions ready. ctxrel_version={ctxrel_version}")

    if controls_only:
        print(f"[controls-only] Running {ctxrel_version} controls {list(control_sets)} with judges "
              f"{list(control_judge_ids)} on the dev split...")
        hard_pool_df, related_ids, question_vectors = _prepare_control_inputs(
            answers_parquet, candidates, sample_df, control_sets, kg_workspace_dir, seed)
        controls = run_controls(sample_df, ctxrel_version, control_judge_ids, questions_parquet, workers=workers,
                                deciding_judge_id=judge_id, control_sets=control_sets,
                                hard_pool_df=hard_pool_df, related_ids=related_ids,
                                question_vectors=question_vectors, require_clean_tree=not allow_dirty)
        _print_controls_result(controls)
        return 0 if controls["passed"] else 1

    git_at_start = git_state()
    print(f"      git_commit={git_at_start['git_commit']} git_tree_clean={git_at_start['git_tree_clean']}")
    if not git_at_start["git_tree_clean"] and not plan_only and not allow_dirty:
        dirty = [e for e in git_at_start["git_status_porcelain"] if e[3:] not in git_at_start["git_exempted"]]
        print(f"[STOP] Working tree is not clean -- commit first (or --plan-only / --allow-dirty). Dirty: {dirty}")
        return 1

    print("[2/6] Connecting to Neo4j + loading the FAISS cache (shared, read-only)...")
    driver, database = cg.connect_neo4j(print)
    faiss_index, faiss_ids, faiss_embeddings, id_to_row = cg.load_faiss_cache(Path(kg_workspace_dir), print)

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(embed_model_name, device="cpu")
    embedding_cache = EmbeddingCache(embedding_cache_path)

    try:
        print(f"[3/6] Retrieval-only C v3 for alpha in {DEFAULT_ALPHA_VALUES} (no API cost)...")
        retrieval_output_paths: dict[float, str] = {}
        own_outputs: list[str] = []
        for alpha in DEFAULT_ALPHA_VALUES:
            info = run_retrieval_only_for_alpha(
                sample_df, alpha, driver, database, faiss_index, faiss_ids, faiss_embeddings,
                id_to_row, all_answer_ids_map, embed_model, embedding_cache, questions_parquet,
                output_dir=str(RESULTS_DIR), seed=seed, n_sample=n_dev, oversample_pool=oversample_pool,
                ctxrel_version=ctxrel_version, git_commit=git_at_start["git_commit"],
            )
            retrieval_output_paths[alpha] = info["output_path"]
            own_outputs += [info["output_path"], info["history_path"]]
            print(f"      alpha={alpha} -> {info['output_path']} ({info['n_questions']} questions complete, "
                  f"{info['n_resumed']} resumed)")
    finally:
        embedding_cache.close()
    if embedding_cache.disabled:
        print(f"      [note] embedding cache was disabled during this run ({embedding_cache.disabled_reason}); "
              f"embeddings were computed directly -- results unaffected.")

    hard_pool_df, related_ids, question_vectors = _prepare_control_inputs(
        answers_parquet, candidates, sample_df, CONTROL_SETS, kg_workspace_dir, seed)
    control_items, _ = build_control_items(sample_df, hard_pool_df=hard_pool_df, related_ids=related_ids,
                                           question_vectors=question_vectors)
    dedup_items = load_items(list(retrieval_output_paths.values()), limit=None)
    plan = build_plan(len(dedup_items), len(control_items) * len(control_judge_ids), judge_id, workers)
    plan["ctxrel_version"] = ctxrel_version
    print()
    print(format_plan(plan))
    print()

    if plan_only:
        print("[plan-only] Stopping before any API call, as requested.")
        return 0

    if not yes:
        answer = input(f"Proceed with the controls + {ctxrel_version} judging above? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted -- no API calls made.")
            return 1

    print(f"[4/6] Running {ctxrel_version} controls (all three sets)...")
    controls = run_controls(sample_df, ctxrel_version, control_judge_ids, questions_parquet, workers=workers,
                            deciding_judge_id=judge_id, control_sets=CONTROL_SETS,
                            hard_pool_df=hard_pool_df, related_ids=related_ids,
                            question_vectors=question_vectors, require_clean_tree=not allow_dirty,
                            exempt_paths=own_outputs)
    _print_controls_result(controls)
    if not controls["passed"]:
        print(f"[STOP] Controls failed -- {ctxrel_version} cannot be used for the main sweep judging. "
              "Retrieval-only outputs above are complete and will be reused on the next attempt "
              "at the same commit.")
        return 1

    print(f"[5/6] Judging {len(dedup_items)} deduplicated context items with judge_id={judge_id}, "
          f"{ctxrel_version}...")
    vtag = ctxrel_version.replace("-", "_")
    ctxrel_output_path = RESULTS_DIR / f"{vtag}_stage1_sweep_{judge_id}.jsonl"
    failures_path = RESULTS_DIR / f"{vtag}_stage1_sweep_{judge_id}_failures.jsonl"
    judging_started = datetime.now(timezone.utc)
    batch_result = run_batch(dedup_items, [judge_id], questions_parquet, ctxrel_output_path, failures_path, workers,
                             prompt_version=ctxrel_version)
    from llm.manifest import write_manifest
    write_manifest(
        ctxrel_output_path, run_label=None,
        config={"ctxrel_version": ctxrel_version, "judge_id": judge_id, "n_dev": n_dev, "dev_offset": dev_offset,
                "seed": seed, "retrieval_output_paths": {str(a): p for a, p in retrieval_output_paths.items()},
                "controls_output_path": controls.get("output_path"),
                "git_commit": git_at_start["git_commit"], "git_tree_clean_at_start": git_at_start["git_tree_clean"],
                "git_state_at_judging": controls.get("git")},
        started_at_utc=judging_started.isoformat(), finished_at_utc=datetime.now(timezone.utc).isoformat(),
        item_counts={"attempted": batch_result["judged"] + batch_result["failed"],
                     "succeeded": batch_result["judged"], "failed": batch_result["failed"]},
        prompt_version=ctxrel_version, judge_version=ctxrel_version,
        total_tokens=batch_result["total_tokens_per_judge"], extra_output_files=[failures_path],
    )
    print(f"      judged={batch_result['judged']} skipped={batch_result['skipped']} "
          f"failed={batch_result['failed']} label_distribution={batch_result['label_distribution']}")

    print("[6/6] Building the stage-1 report...")
    ctxrel_by_key = load_ctxrel_records([ctxrel_output_path], prompt_version=ctxrel_version)
    selected_by_alpha = _build_selected_answer_ids_by_alpha(retrieval_output_paths)
    per_alpha_metrics = {
        alpha: compute_alpha_metrics(path, ctxrel_by_key, selected_by_alpha, alpha)
        for alpha, path in retrieval_output_paths.items()
    }
    stage1_report = build_stage1_report({
        alpha: {k: v for k, v in m.items() if k not in ("by_hop", "by_edge_type")}
        for alpha, m in per_alpha_metrics.items()
    })
    selection = select_stage1_candidates(stage1_report)

    report_path = RESULTS_DIR / f"stage1_report_{vtag}_{judge_id}_n{n_dev}_seed{seed}.md"
    report_path.write_text(build_stage1_markdown(per_alpha_metrics, selection, ctxrel_version=ctxrel_version))
    print(f"      report written -> {report_path}")
    print()
    print(f"keep_current: {selection['keep_current']}")
    if not selection["keep_current"]:
        print(f"top_two (stage 2 candidates): {selection['top_two']}")
    print(f"reasoning: {selection['reasoning']}")
    return 0


def _print_controls_result(controls: dict) -> None:
    if controls.get("error"):
        print(f"      [ERROR] {controls['error']}")
        return
    info = controls.get("info", {})
    if info.get("no_hard_negative"):
        print(f"      dev questions with NO eligible hard negative (excluded): {info['no_hard_negative']}")
    if info.get("hard_negative_missing_vector") or info.get("n_eligible_donors_without_vector"):
        print(f"      hard negatives: dev questions without a cached vector {info.get('hard_negative_missing_vector')}; "
              f"eligible donors skipped for missing vector: {info.get('n_eligible_donors_without_vector')}")
    if info.get("no_easy_negative"):
        print(f"      dev questions with NO eligible easy negative (excluded): {info['no_easy_negative']}")
    for jid, j in controls["summary"]["per_judge"].items():
        for cset in CONTROL_SETS:
            if cset not in j:
                continue
            t = j[cset]
            print(f"      {jid:9s} {cset:13s}: n={t['n']} %R/%P/%I/%err = {t['pct']['RELEVANT']}/{t['pct']['PARTIAL']}/"
                  f"{t['pct']['IRRELEVANT']}/{t['pct']['ERROR']}  match={t['match_rate']:.0%} "
                  f"(>= {t['threshold']:.0%}) pass={t['pass']}")
    git = controls.get("git", {})
    print(f"      git_commit={git.get('git_commit')} git_tree_clean={git.get('git_tree_clean')}")
    print(f"      controls passed (decided by {controls['summary']['deciding_judge_id']}; None = not all three "
          f"sets judged): {controls['passed']} (output: {controls.get('output_path')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    parser.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"))
    parser.add_argument("--kg-workspace-dir", default=os.getenv(
        "KG_WORKSPACE_DIR", str(Path(__file__).resolve().parents[2] / "01_data_cleaning" / "_kg_workspace")))
    parser.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    from embedding_cache import default_cache_path
    parser.add_argument("--embedding-cache-path", default=str(default_cache_path()),
                         help="Candidate-embedding cache (speed-up only; fail-safe). Default: "
                              "C_EMBEDDING_CACHE_PATH, else a local non-synced cache dir (see embedding_cache.py).")
    parser.add_argument("--n-dev", type=int, default=50)
    parser.add_argument("--dev-offset", type=int, default=385)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-id", default="primary",
                         help="Deciding judge for controls, and the judge used for the main sweep.")
    parser.add_argument("--ctxrel-version", required=True, choices=("ctxrel-v1", "ctxrel-v2"),
                         help="Context-relevance prompt version (REQUIRED, no default -- see "
                              "docs/DECISION_C_SCORING.md Amendments 1/2). Recorded in every output and manifest.")
    parser.add_argument("--control-sets", default=",".join(CONTROL_SETS),
                         help="--controls-only: comma-separated subset of " + ",".join(CONTROL_SETS) + ".")
    parser.add_argument("--control-judges", default=",".join(CONTROL_JUDGE_IDS),
                         help="Comma-separated judge ids for the controls (deciding judge = --judge-id).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true",
                         help="Build the dev sample and run retrieval-only (free), show the plan, then stop -- "
                              "no API call is made.")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive y/N confirmation.")
    parser.add_argument("--allow-dirty", action="store_true",
                         help="Run the controls even if the working tree has uncommitted changes outside "
                              "results/controls/ (recorded as git_tree_clean=false). Off by default.")
    parser.add_argument("--controls-only", action="store_true",
                         help="Run ONLY the ctxrel-v1 positive/negative controls on the dev split and exit -- "
                              "does not touch retrieval or run the main sweep judging.")
    args = parser.parse_args()

    if not args.questions_parquet or not args.answers_parquet:
        print("[ERROR] --questions-parquet/--answers-parquet (or QUESTIONS_PARQUET/ANSWERS_PARQUET in .env) required.")
        sys.exit(1)

    control_sets = tuple(c.strip() for c in args.control_sets.split(",") if c.strip())
    unknown = [c for c in control_sets if c not in CONTROL_SETS]
    if unknown or not control_sets:
        print(f"[ERROR] unknown --control-sets {unknown} (choose from {CONTROL_SETS})")
        sys.exit(1)
    if not args.controls_only and control_sets != CONTROL_SETS:
        print("[ERROR] --control-sets subsets are only allowed with --controls-only "
              "(the full sweep always runs all three sets).")
        sys.exit(1)

    exit_code = run_stage1_sweep(
        args.questions_parquet, args.answers_parquet, args.kg_workspace_dir, args.embed_model,
        args.embedding_cache_path, args.n_dev, args.dev_offset, args.seed, args.judge_id, args.workers,
        args.plan_only, args.yes, args.controls_only, args.ctxrel_version,
        control_sets=control_sets,
        control_judge_ids=tuple(j.strip() for j in args.control_judges.split(",") if j.strip()),
        allow_dirty=args.allow_dirty,
    )
    sys.exit(exit_code)
