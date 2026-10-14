"""
logscore_top5_check.py
=====================================
Read-only check (no LLM calls, Neo4j is only READ): does Amendment 6
(log-normalised upvote score in trustScore, docs/DECISION_C_SCORING.md)
change Condition C's top-5 contexts?

WHAT IS COMPARED
------------------------------------------------------------
For each DEV question (positions 385-434, same as stage1_sweep.py; the test
sample 1-384 is never loaded), candidates are retrieved ONCE (anchoring ->
traversal -> semantic expansion) and ranked with fuse_and_rank() v3 under
two Answer trust versions, at the same alpha:

  v2 : trustScore "v2-reputation" (Amendment 5). Read from the node while the
       graph is still v2; after 14_apply_logscore_trust.py --apply it is read
       from the backup parquet (--backup).
  v3 : trustScore "v3-logscore", computed in memory with trust_formulas.py
       from the node's score / isAccepted and its AUTHOR_TRUST weight, with
       P99 taken over all Answer nodes. For nodes already tagged v3 the
       stored value must equal the computed one exactly, or the script stops.

So the script gives the same comparison before and after --apply. Because
the candidate pool is identical, any top-5 difference comes from the trust
version alone. alpha = 0 (similarity only) must give 0 changes; this is
asserted.

OUTPUT (llm/c_graphrag/results/logscore_check/)
------------------------------------------------------------
  logscore_top5_<ts>.md / .json / .csv
Descriptive only: this script does not judge relevance and is not a
selection result.

USAGE (from the repo root; same .env as c_graphrag.py)
------------------------------------------------------------
    python3 llm/c_graphrag/logscore_top5_check.py
    python3 llm/c_graphrag/logscore_top5_check.py --backup 01_data_cleaning/_kg_workspace/<backup>.parquet
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "01_data_cleaning"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import c_graphrag as cg  # noqa: E402
import reputation_ablation_check as rac  # noqa: E402
import trust_formulas as tf  # noqa: E402

ALPHAS = (0.0, 0.25, 0.5)
V2 = "v2-reputation"
V3 = tf.TRUST_SCORE_VERSION
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "logscore_check"


# ---------------------------------------------------------------------
# Pure logic (no Neo4j/FAISS) -- tested in tests/test_logscore_trust.py
# ---------------------------------------------------------------------

def resolve_trust(inputs: dict[int, dict], p99_answer: float,
                  backup_trust: dict[int, float] | None = None) -> tuple[dict[int, float], dict[int, float]]:
    """(v2 trust, v3 trust) per answer_id.

    `inputs`: {answer_id: {"score", "is_accepted", "author_weight", "trust", "version"}} as
    stored in Neo4j. v3 is always computed with trust_formulas. v2 is the stored value for a
    node tagged v2; for a node already tagged v3 it comes from `backup_trust`, and the stored
    value must equal the computed v3 exactly."""
    ids = sorted(inputs)
    if not ids:
        return {}, {}
    computed = tf.trust_answer(
        [inputs[i]["score"] for i in ids],
        [bool(inputs[i]["is_accepted"]) for i in ids],
        [np.nan if inputs[i]["author_weight"] is None else inputs[i]["author_weight"] for i in ids],
        p99_answer,
    )
    v3 = {i: float(t) for i, t in zip(ids, computed)}
    v2 = {}
    for i in ids:
        version, stored = inputs[i]["version"], inputs[i]["trust"]
        if version == V2:
            v2[i] = float(stored)
        elif version == V3:
            if stored != v3[i]:
                raise ValueError(f"answer {i}: stored {V3} trustScore {stored!r} != computed {v3[i]!r}")
            if backup_trust is None or i not in backup_trust:
                raise ValueError(f"answer {i} is already {V3}; its {V2} value needs --backup "
                                 f"<backup parquet written by 14_apply_logscore_trust.py --apply>")
            v2[i] = float(backup_trust[i])
        else:
            raise ValueError(f"answer {i}: unexpected trustScoreVersion {version!r}")
    return v2, v3


def with_trust(candidates: list[dict], trust_by_answer: dict[int, float]) -> list[dict]:
    """Copy of the candidates with answer_trust_score replaced."""
    out = []
    for c in candidates:
        c2 = copy.deepcopy(c)
        c2["answer_trust_score"] = trust_by_answer[int(c2["answer_id"])]
        out.append(c2)
    return out


def rank_with_trust(alpha: float, graph_c: list[dict], exp_c: list[dict], trust_by_answer: dict[int, float],
                    embed_model, query_emb, embedding_cache, top_k: int = 5,
                    token_chunk_limit: int = 400) -> list[dict]:
    """Top-k at one alpha through the real fuse_and_rank() v3."""
    final, _ = cg.fuse_and_rank(
        with_trust(graph_c, trust_by_answer), with_trust(exp_c, trust_by_answer), top_k, token_chunk_limit,
        c_retrieval_version="v3", alpha=alpha,
        embed_model=embed_model, query_emb=query_emb, embedding_cache=embedding_cache,
    )
    return final


def compare_alpha(question_id: int, alpha: float, top_v2: list[dict], top_v3: list[dict]) -> dict:
    """One result row. Raises if alpha = 0 changes anything (must be pure similarity)."""
    cmp = rac.compare_topk(top_v2, top_v3, {})
    if alpha == 0.0 and cmp["order_changed"]:
        raise RuntimeError(f"Sanity check failed: alpha=0 changed the top-k of q{question_id}.")
    return {"question_id": question_id, "alpha": alpha,
            **{k: cmp[k] for k in ("set_changed", "rank1_changed", "order_changed", "jaccard", "n_swapped",
                                   "entered_ids", "left_ids", "entered_accepted", "left_accepted")}}


def summarize(rows: list[dict], alphas=ALPHAS, n_examples: int = 5) -> dict:
    summary = {}
    for alpha in alphas:
        sel = [r for r in rows if r["alpha"] == alpha]
        if not sel:
            continue
        changed = [r for r in sel if r["set_changed"]]
        summary[str(alpha)] = {
            "alpha": alpha, "n_questions": len(sel),
            "n_set_changed": len(changed),
            "pct_set_changed": round(100 * len(changed) / len(sel), 1),
            "mean_items_swapped": round(float(np.mean([r["n_swapped"] for r in sel])), 4),
            "mean_jaccard": round(float(np.mean([r["jaccard"] for r in sel])), 4),
            "n_rank1_changed": sum(r["rank1_changed"] for r in sel),
            "n_order_changed": sum(r["order_changed"] for r in sel),
            "entered_accepted": sum(r["entered_accepted"] for r in sel),
            "left_accepted": sum(r["left_accepted"] for r in sel),
            "example_question_ids": [r["question_id"] for r in changed[:n_examples]],
        }
    return summary


# ---------------------------------------------------------------------
# Data access (Neo4j read-only)
# ---------------------------------------------------------------------

def answer_score_p99(driver, database) -> float:
    """P99 of `score` over ALL Answer nodes, same function as the build and the apply script."""
    with driver.session(database=database) as s:
        scores = [rec["score"] for rec in s.run("MATCH (a:Answer) RETURN a.score AS score")]
    return tf.compute_p99(np.array([np.nan if x is None else x for x in scores], dtype=np.float64))


def fetch_answer_inputs(driver, database, answer_ids: list[int]) -> dict[int, dict]:
    if not answer_ids:
        return {}
    with driver.session(database=database) as s:
        result = s.run(
            """
            UNWIND $answer_ids AS aid
            MATCH (a:Answer {id: aid})
            OPTIONAL MATCH (a)-[r:AUTHOR_TRUST]->(:User)
            WITH a, max(r.weight) AS w
            RETURN a.id AS answer_id, a.score AS score, a.isAccepted AS is_accepted, w AS author_weight,
                   a.trustScore AS trust, a.trustScoreVersion AS version
            """,
            answer_ids=list(answer_ids),
        )
        return {int(r["answer_id"]): {k: r[k] for k in ("score", "is_accepted", "author_weight", "trust", "version")}
                for r in result}


def load_backup_trust(path) -> dict[int, float]:
    """{answer_id: v2 trustScore} from the backup written by 14_apply_logscore_trust.py --apply."""
    import pandas as pd
    bk = pd.read_parquet(path)
    bk = bk[bk["label"] == "Answer"]
    if not (bk["version"] == V2).all():
        raise ValueError(f"{path}: not every Answer row is {V2}; this is not a pre-Amendment-6 backup.")
    return dict(zip(bk["id"].astype(int), bk["trust"].astype(float)))


def build_markdown(summary: dict, meta: dict) -> str:
    lines = [
        "# Amendment 6 top-5 check: trustScore v2-reputation vs v3-logscore (Condition C, dev set)",
        "",
        f"- Time: {meta['finished_at']}  |  git commit: `{meta['git_commit']}`",
        f"- Dev set: {meta['n_questions']} questions (positions {meta['dev_offset']}-"
        f"{meta['dev_offset'] + meta['n_questions'] - 1}), seed {meta['seed']}, top_k {meta['top_k']}",
        f"- P99 Answer score: {meta['p99_answer']} (numpy '{tf.QUANTILE_METHOD}')  |  graph state: "
        f"{meta['candidate_versions']}  |  backup: {meta['backup']}",
        f"- No LLM calls. Neo4j read only. Duration {meta['duration_sec']} s.",
        "",
        "| alpha | Top-5 set changed | Mean items swapped | Mean Jaccard | Rank-1 changed | Order changed "
        "| Accepted in/out | Example question IDs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in summary.values():
        lines.append(
            f"| {v['alpha']} | {v['n_set_changed']}/{v['n_questions']} ({v['pct_set_changed']}%) | "
            f"{v['mean_items_swapped']} | {v['mean_jaccard']} | {v['n_rank1_changed']} | {v['n_order_changed']} | "
            f"{v['entered_accepted']}/{v['left_accepted']} | {', '.join(map(str, v['example_question_ids'])) or '-'} |"
        )
    lines += ["", "alpha = 0 sanity check (0 changes expected): passed.",
              "A change is not an improvement: this script does not judge relevance."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main() -> int:
    import stage1_sweep as s1
    from embedding_cache import EmbeddingCache, default_cache_path

    p = argparse.ArgumentParser(description="Amendment 6 top-5 impact check (read-only, no API calls).")
    p.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    p.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"))
    p.add_argument("--kg-workspace-dir", default=os.getenv(
        "KG_WORKSPACE_DIR", str(Path(__file__).resolve().parents[2] / "01_data_cleaning" / "_kg_workspace")))
    p.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    p.add_argument("--embedding-cache-path", default=str(default_cache_path()))
    p.add_argument("--backup", default=None,
                   help="Backup parquet from 14_apply_logscore_trust.py --apply (needed once the graph is v3).")
    p.add_argument("--n-dev", type=int, default=50)
    p.add_argument("--dev-offset", type=int, default=385)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-anchor", type=int, default=int(os.getenv("N_ANCHOR", 3)))
    p.add_argument("--n-semantic-expansion", type=int, default=int(os.getenv("N_SEMANTIC_EXPANSION", 3)))
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = p.parse_args()

    if not args.questions_parquet or not args.answers_parquet:
        print("[ERROR] QUESTIONS_PARQUET/ANSWERS_PARQUET are required (flag or .env).")
        return 1

    t0 = time.time()
    print(f"[1/5] Building the dev set (n={args.n_dev}, dev_offset={args.dev_offset}, seed={args.seed})...")
    sample_df, all_answer_ids_map, _, _ = s1._build_dev_sample(
        args.questions_parquet, args.answers_parquet, args.n_dev, args.seed, args.dev_offset)
    print(f"      {len(sample_df)} dev questions.")

    print("[2/5] Neo4j (read-only): P99 of the Answer score...")
    driver, database = cg.connect_neo4j(print)
    p99_answer = answer_score_p99(driver, database)
    backup_trust = load_backup_trust(args.backup) if args.backup else None
    print(f"      P99_A = {p99_answer} (numpy '{tf.QUANTILE_METHOD}'); backup: {args.backup or '-'}")

    print("[3/5] Loading the FAISS cache + embedding model (CPU)...")
    faiss_index, faiss_ids, faiss_embeddings, id_to_row = cg.load_faiss_cache(Path(args.kg_workspace_dir), print)
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(args.embed_model, device="cpu")
    embedding_cache = EmbeddingCache(args.embedding_cache_path)

    rows, versions = [], {}
    try:
        print(f"[4/5] One retrieval per question, then ranking v2 vs v3 at alpha {list(ALPHAS)}...")
        for i, row in sample_df.iterrows():
            qid = int(row["Id"])
            query_emb, graph_c, exp_c = rac.retrieve_candidates(
                row, faiss_index, faiss_ids, faiss_embeddings, id_to_row, all_answer_ids_map,
                driver, database, embed_model, args.n_anchor, args.n_semantic_expansion)
            aids = sorted({int(c["answer_id"]) for c in graph_c + exp_c})
            inputs = fetch_answer_inputs(driver, database, aids)
            for v in inputs.values():
                versions[v["version"]] = versions.get(v["version"], 0) + 1
            trust_v2, trust_v3 = resolve_trust(inputs, p99_answer, backup_trust)
            n_changed = 0
            for alpha in ALPHAS:
                top_v2 = rank_with_trust(alpha, graph_c, exp_c, trust_v2, embed_model, query_emb,
                                         embedding_cache, args.top_k)
                top_v3 = rank_with_trust(alpha, graph_c, exp_c, trust_v3, embed_model, query_emb,
                                         embedding_cache, args.top_k)
                r = compare_alpha(qid, alpha, top_v2, top_v3)
                r["candidate_pool_size"] = len(aids)
                n_changed += r["set_changed"]
                rows.append(r)
            print(f"      [{i + 1}/{len(sample_df)}] q{qid}: {len(aids)} candidates, "
                  f"top-5 set changed at {n_changed} of {len(ALPHAS)} alpha")
    except (ValueError, RuntimeError) as e:
        print(f"[STOP] {e}")
        return 1
    finally:
        embedding_cache.close()
        driver.close()

    print("[5/5] Writing the report...")
    summary = summarize(rows)
    finished = datetime.now(timezone.utc)
    meta = {
        "finished_at": finished.isoformat(timespec="seconds"),
        "git_commit": rac._git_commit(Path(__file__).resolve().parents[2]),
        "n_questions": len(sample_df), "dev_offset": args.dev_offset, "seed": args.seed, "top_k": args.top_k,
        "n_anchor": args.n_anchor, "n_semantic_expansion": args.n_semantic_expansion, "alphas": list(ALPHAS),
        "p99_answer": p99_answer, "quantile_method": tf.QUANTILE_METHOD, "trust_versions": [V2, V3],
        "candidate_versions": versions, "backup": args.backup, "duration_sec": round(time.time() - t0, 1),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"logscore_top5_{finished.strftime('%Y%m%dT%H%M%SZ')}"
    Path(f"{stem}.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2, default=str))
    with open(f"{stem}.csv", "w", newline="") as f:
        fields = ["question_id", "alpha", "candidate_pool_size", "set_changed", "rank1_changed", "order_changed",
                  "jaccard", "n_swapped", "entered_ids", "left_ids", "entered_accepted", "left_accepted"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], list) else r[k]) for k in fields})
    md = build_markdown(summary, meta)
    Path(f"{stem}.md").write_text(md)
    print(md)
    print(f"Report: {stem}.md / .json / .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
