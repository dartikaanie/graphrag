"""
audit.py -- retrieval-only leakage audit (no LLM), reports/leakage_audit.md
=====================================
Dev set only (positions 385..434, n = 50). Two steps, run one condition per
process (RAM):

    python llm/leakage/audit.py retrieve --condition B
    python llm/leakage/audit.py retrieve --condition C --alpha 0.5
    python llm/leakage/audit.py retrieve --condition C --alpha 0.25 --modes thread dup+neardup0.95
    python llm/leakage/audit.py retrieve --condition D
    python llm/leakage/audit.py analyze

`retrieve` writes results/leakage/audit/<cond>[_a<alpha>]_<mode>.jsonl with
each condition's own retrieval code (B: retrieve_context_kg; C/D:
process_sample(retrieval_only=True)) under every mode, including "off"
(LEAKAGE_FILTER=False, no exclusion at all) and "thread" (the exclusion of
every run before Amendment 8).

`analyze` computes, per condition x retrieval mode, against every
exclusion definition:
  - ID-level self-retrieval: % questions whose context holds an id in the
    exclusion set (q, its answers, or a related question / its answers);
  - text-level (metric only, never a filter): a context chunk shares an
    identical >= 50-character substring with q's accepted answer, or
    cosine(accepted answer, chunk) >= 0.95 (all-MiniLM-L6-v2, CPU);
  - final_k < k / = 0, latency, empty expansion slots (C), and the Linked
    sensitivity check; written to results/leakage/audit/summary.json.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "llm" / "c_graphrag", REPO_ROOT / "llm" / "d_lightrag"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from llm.common.leakage import (DEFAULT_EXCLUSION_SETS_PATH, MODE_OFF, MODE_THREAD,  # noqa: E402
                                LeakageContext, leakage_mode)

AUDIT_DIR = REPO_ROOT / "results" / "leakage" / "audit"
N_DEV, DEV_OFFSET, SEED, OVERSAMPLE_POOL = 50, 385, 42, 384 * 4
TOP_K = 5
DEFAULT_MODE = leakage_mode(True, False, 0.95)
LINKED_MODE = leakage_mode(True, True, 0.95)
MODES = (MODE_OFF, MODE_THREAD, "dup", DEFAULT_MODE, LINKED_MODE)
SUBSTRING_CHARS = 50
TEXT_COSINE = 0.95


def _env():
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    return os.environ["QUESTIONS_PARQUET"], os.environ["ANSWERS_PARQUET"]


def load_dev(n=N_DEV):
    """Dev questions (positions 385..), accepted answer text, thread answer
    map from the answers parquet (C/D's source) -- C's own loaders."""
    import duckdb

    import c_graphrag as cg

    cg.log = lambda *a, **k: None
    qp, ap = _env()
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    cand = cg.filter_by_token_limit(cg.get_candidate_questions(con, qp, ap, OVERSAMPLE_POOL, SEED))
    df = cg.sample_questions_split(cand, n, SEED, split="dev", dev_offset=DEV_OFFSET)
    acc = cg.get_accepted_answers(con, ap, df["AcceptedAnswerId"].dropna().unique().tolist())
    df = df.merge(acc, on="AcceptedAnswerId", how="left").dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)
    thread = cg.get_all_answer_ids_for_questions(con, ap, df["Id"].astype(int).tolist())
    con.close()
    return df, thread


def out_path(cond: str, mode: str, alpha=None) -> Path:
    tag = cond if alpha is None else f"{cond}_a{alpha:g}"
    return AUDIT_DIR / f"{tag}_{mode}.jsonl"


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


# ---------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------

def retrieve_b(df, modes):
    from sentence_transformers import SentenceTransformer

    from llm.b_rag import b_condition_b_rag as b
    from llm.b_rag.kg_corpus import encode_query, load_kg_answer_index
    from llm.common.text import build_query_text

    kg = load_kg_answer_index(REPO_ROOT / "llm" / "b_rag" / "index_cache", 400, "all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    threads = kg.thread_answer_ids(df["Id"].astype(int).tolist())
    for mode in modes:
        ctx = LeakageContext.from_mode(mode)
        recs = []
        for r in df.itertuples():
            qid = int(r.Id)
            excl = ctx.exclusion_for(qid, threads.get(qid, []), r.AcceptedAnswerId)
            stats = {"n_hits_filtered": 0}
            t0 = time.time()
            text = build_query_text(r.Title, r.Body)
            out = b.retrieve_context_kg(model, kg, text, TOP_K, excl, encode_query(model, text), stats=stats)
            recs.append({"question_id": qid, "accepted_answer_id": int(r.AcceptedAnswerId), "retrieved_context": out,
                         "retrieval_latency_sec": round(time.time() - t0, 3),
                         "leakage": ctx.query_log(excl, stats["n_hits_filtered"], len(out))})
        _write(out_path("B", mode), recs)
        print(f"[B/{mode}] {len(recs)} -> {out_path('B', mode).name}")


def retrieve_cd(cond, df, thread, modes, alpha):
    from sentence_transformers import SentenceTransformer

    mod = __import__("c_graphrag" if cond == "C" else "d_lightrag")
    mod.log = lambda *a, **k: None
    driver, database = mod.connect_neo4j(lambda *a, **k: None)
    index, ids_arr, emb, id_to_row = mod.load_faiss_cache(Path(os.environ["KG_WORKSPACE_DIR"]), lambda *a, **k: None)
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    cache = None
    if cond == "C":
        from embedding_cache import EmbeddingCache, default_cache_path

        cache = EmbeddingCache(default_cache_path())
    try:
        for mode in modes:
            ctx = LeakageContext.from_mode(mode)
            with tempfile.TemporaryDirectory() as tmp:
                if cond == "C":
                    recs, _ = mod.process_sample(
                        df, None, None, model, driver, database, index, ids_arr, emb, id_to_row, thread,
                        TOP_K, 3, 3, 400, "none", Path(tmp) / "o.jsonl", retrieval_only=True,
                        c_retrieval_version="v3", alpha=alpha, embedding_cache=cache, leakage=ctx)
                else:
                    recs, _ = mod.process_sample(
                        df, None, None, model, driver, database, index, ids_arr, emb, id_to_row, thread,
                        TOP_K, 3, 3, 400, "none", Path(tmp) / "o.jsonl", retrieval_only=True, leakage=ctx)
            path = out_path(cond, mode, alpha if cond == "C" else None)
            _write(path, recs)
            print(f"[{cond}/{mode}] {len(recs)} -> {path.name}")
    finally:
        driver.close()
        if cache is not None:
            cache.close()


# ---------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------

def _shingles(text: str, n=SUBSTRING_CHARS) -> set:
    return {text[i:i + n] for i in range(0, max(0, len(text) - n + 1))}


def shares_substring(a: str, b: str, n=SUBSTRING_CHARS) -> bool:
    """True iff a and b share an identical substring of >= n characters."""
    if len(a) < n or len(b) < n:
        return False
    sa = _shingles(a, n)
    return any(b[i:i + n] in sa for i in range(len(b) - n + 1))


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)] if path.exists() else []


def analyze():
    import duckdb
    from sentence_transformers import SentenceTransformer

    df, thread = load_dev()
    accepted = {int(r.Id): str(r.AcceptedAnswerBody) for r in df.itertuples()}
    acc_id = {int(r.Id): int(r.AcceptedAnswerId) for r in df.itertuples()}
    tables = {m: LeakageContext.from_mode(m) for m in MODES if m != MODE_OFF}
    # exclusion definitions as C/D build them (parquet thread map + table); B's
    # thread map differs only by answers that cannot be retrieved anyway (q excluded)
    excl = {m: {q: tables[m].exclusion_for(q, thread.get(q, []), acc_id[q]) for q in accepted} for m in tables}

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    acc_vec = {q: v for q, v in zip(accepted, model.encode(list(accepted.values()), convert_to_numpy=True,
                                                           normalize_embeddings=True, batch_size=1))}
    chunk_cache: dict[str, np.ndarray] = {}

    def chunk_vec(t):
        if t not in chunk_cache:
            chunk_cache[t] = model.encode([t], convert_to_numpy=True, normalize_embeddings=True)[0]
        return chunk_cache[t]

    def text_leak(q, items):
        sub = any(shares_substring(accepted[q], it["chunk_text"]) for it in items)
        cos = max((float(np.dot(acc_vec[q], chunk_vec(it["chunk_text"]))) for it in items), default=0.0)
        return sub, cos >= TEXT_COSINE, cos

    files = sorted(AUDIT_DIR.glob("*.jsonl"))
    summary = {"n_dev": len(accepted), "files": {}}
    for f in files:
        recs = [r for r in _read(f) if r["question_id"] in accepted]
        stem_cond, stem_mode = _split_stem(f.stem)
        row = {"condition": stem_cond, "retrieval_mode": stem_mode, "n": len(recs)}
        for m, ex in excl.items():
            hit = [r["question_id"] for r in recs if any(ex[r["question_id"]].excludes(i["question_id"], i["answer_id"])
                                                         for i in r["retrieved_context"])]
            row[f"id_self_retrieval_vs_{m}"] = {"n": len(hit), "pct": round(100 * len(hit) / len(recs), 1) if recs else None,
                                                "first35_n": sum(1 for q in hit if q in list(accepted)[:35])}
        tl = [text_leak(r["question_id"], r["retrieved_context"]) for r in recs]
        row["text_substring50"] = {"n": sum(t[0] for t in tl), "pct": round(100 * sum(t[0] for t in tl) / len(recs), 1)}
        row["text_cosine095"] = {"n": sum(t[1] for t in tl), "pct": round(100 * sum(t[1] for t in tl) / len(recs), 1)}
        row["text_any"] = {"n": sum(t[0] or t[1] for t in tl), "pct": round(100 * sum(t[0] or t[1] for t in tl) / len(recs), 1)}
        row["max_cos_accepted_mean"] = round(float(np.mean([t[2] for t in tl])), 4)
        ks = [len(r["retrieved_context"]) for r in recs]
        row["final_k_lt_k"] = sum(k < TOP_K for k in ks)
        row["final_k_0"] = sum(k == 0 for k in ks)
        lat = [r.get("retrieval_latency_sec") or 0 for r in recs]
        row["latency_mean"], row["latency_p95"] = round(float(np.mean(lat)), 3), round(float(np.percentile(lat, 95)), 3)
        row["n_hits_filtered_mean"] = round(float(np.mean([r.get("leakage", {}).get("n_hits_filtered", 0) for r in recs])), 2)
        if "expansion_qids" in (recs[0] if recs else {}):
            row["expansion_q_total"] = sum(len(r["expansion_qids"]) for r in recs)
            row["expansion_q_empty"] = sum(len(r["expansion_qids_empty"]) for r in recs)
            row["expansion_q_empty_ids"] = sorted({q for r in recs for q in r["expansion_qids_empty"]})
        summary["files"][f.stem] = row

    # C retrieval identical thread vs default (4c)
    ident = {}
    for alpha_tag in sorted({_split_stem(f.stem)[0] for f in files if f.stem.startswith("C")}):
        a = {r["question_id"]: r["retrieved_context"] for r in _read(AUDIT_DIR / f"{alpha_tag}_{MODE_THREAD}.jsonl")}
        b = {r["question_id"]: r["retrieved_context"] for r in _read(AUDIT_DIR / f"{alpha_tag}_{DEFAULT_MODE}.jsonl")}
        if a and b:
            diff = [q for q in a if a[q] != b.get(q)]
            ident[alpha_tag] = {"n_compared": len(a), "n_different": len(diff), "different": diff}
    summary["c_thread_vs_default_identical"] = ident
    summary["dev_exclusion_set_changes"] = {
        m: sorted(q for q in accepted if excl[m][q] != excl[MODE_THREAD][q]) for m in excl if m != MODE_THREAD}

    # empty expansion slots: has answers in the KG (so: all excluded) vs none
    kg_parquet = REPO_ROOT / "llm" / "b_rag" / "index_cache" / "kg_answers_v1_tok400" / "kg_answers.parquet"
    empty_ids = sorted({q for row in summary["files"].values() for q in row.get("expansion_q_empty_ids", [])})
    if empty_ids and kg_parquet.exists():
        n_ans = dict(duckdb.connect().execute(
            f"SELECT question_id, count(*) FROM read_parquet('{kg_parquet}') WHERE question_id IN "
            f"({','.join(map(str, empty_ids))}) GROUP BY 1").fetchall())
        for row in summary["files"].values():
            if "expansion_q_empty_ids" in row:
                row["expansion_q_empty_has_answers"] = sum(1 for q in row["expansion_q_empty_ids"] if n_ans.get(q, 0) > 0)

    summary["linked_sensitivity"] = linked_sensitivity(df, excl, accepted, model)
    (AUDIT_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"wrote {AUDIT_DIR / 'summary.json'}")


def _split_stem(stem: str) -> tuple[str, str]:
    """'C_a0.5_dup+neardup0.95' -> ('C_a0.5', 'dup+neardup0.95'); 'B_off' -> ('B', 'off')."""
    for m in sorted(MODES, key=len, reverse=True):
        if stem.endswith("_" + m):
            return stem[: -len(m) - 1], m
    raise ValueError(stem)


def linked_sensitivity(df, excl, accepted, model) -> dict:
    """(i) dev, retrieved: context items whose question is a Linked neighbour
    of q (in the default mode's retrieval, where Linked is NOT excluded), and
    whether they pass the text-level check. (ii) direct, no retrieval: for
    every NON-TEST pool question (pool positions > 384) with answered Linked
    neighbours, does any neighbour answer share >= 50 chars with / have
    cosine >= 0.95 to q's accepted answer?"""
    import duckdb
    import pyarrow.parquet as pq

    import c_graphrag as cg

    t = pq.read_table(DEFAULT_EXCLUSION_SETS_PATH).to_pandas()
    lk = t[t["mode"] == LINKED_MODE].set_index("question_id")
    df_ = t[t["mode"] == DEFAULT_MODE].set_index("question_id")
    linked_q = {int(q): set(map(int, lk.loc[q, "excluded_q_ids"])) - set(map(int, df_.loc[q, "excluded_q_ids"]))
                for q in lk.index}
    out = {"retrieved": {}}
    for cond_file in sorted(AUDIT_DIR.glob(f"*_{DEFAULT_MODE}.jsonl")):
        hits = []
        for r in _read(cond_file):
            q = r["question_id"]
            for it in r["retrieved_context"]:
                if it["question_id"] in linked_q.get(q, set()):
                    v = model.encode([it["chunk_text"], accepted[q]], convert_to_numpy=True, normalize_embeddings=True)
                    hits.append({"question_id": q, "linked_question_id": it["question_id"], "answer_id": it["answer_id"],
                                 "substring50": shares_substring(accepted[q], it["chunk_text"]),
                                 "cosine": round(float(np.dot(v[0], v[1])), 4)})
        out["retrieved"][_split_stem(cond_file.stem)[0]] = hits

    # (ii) direct check over non-test pool questions
    qp, ap = _env()
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    cg.log = lambda *a, **k: None
    cand = cg.filter_by_token_limit(cg.get_candidate_questions(con, qp, ap, OVERSAMPLE_POOL, SEED))
    perm = cand.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    non_test = perm.iloc[384:]
    rows = [(int(r.Id), int(r.AcceptedAnswerId)) for r in non_test.itertuples() if linked_q.get(int(r.Id))]
    kg_parquet = REPO_ROOT / "llm" / "b_rag" / "index_cache" / "kg_answers_v1_tok400" / "kg_answers.parquet"
    neigh = sorted({o for q, _ in rows for o in linked_q[q]})
    nb = con.execute(f"SELECT question_id, answer_id, answer_body FROM read_parquet('{kg_parquet}') "
                     f"WHERE question_id IN ({','.join(map(str, neigh)) or '-1'})").df() if neigh else None
    accs = cg.get_accepted_answers(con, ap, [a for _, a in rows]) if rows else None
    con.close()
    acc_body = dict(zip(accs["AcceptedAnswerId"].astype(int), accs["AcceptedAnswerBody"])) if rows else {}
    direct = []
    for q, a in rows:
        sub = nb[nb["question_id"].isin(linked_q[q])]
        if sub.empty or a not in acc_body:
            continue
        av = model.encode([acc_body[a]], convert_to_numpy=True, normalize_embeddings=True)[0]
        bodies = sub["answer_body"].astype(str).tolist()
        vecs = model.encode(bodies, convert_to_numpy=True, normalize_embeddings=True, batch_size=1)
        direct.append({"question_id": q, "n_linked_answers": len(bodies),
                       "substring50": any(shares_substring(acc_body[a], b) for b in bodies),
                       "max_cosine": round(float(max(vecs @ av)), 4)})
    out["direct_non_test_pool"] = {
        "n_questions": len(direct),
        "n_substring50": sum(d["substring50"] for d in direct),
        "n_cosine095": sum(d["max_cosine"] >= TEXT_COSINE for d in direct),
        "max_cosine_values": sorted((d["max_cosine"] for d in direct), reverse=True)[:10],
        "flagged": [d for d in direct if d["substring50"] or d["max_cosine"] >= TEXT_COSINE],
    }
    return out


def devruns(labelled: list[str]) -> None:
    """Step-5 comparison of generation dev runs: LABEL=path.jsonl ... ->
    results/leakage/audit/devruns.json (self-retrieval vs the default and the
    thread exclusion, text-level, cosine, final_k, latency, NF2)."""
    from sentence_transformers import SentenceTransformer

    df, thread = load_dev(35)
    accepted = {int(r.Id): str(r.AcceptedAnswerBody) for r in df.itertuples()}
    acc_id = {int(r.Id): int(r.AcceptedAnswerId) for r in df.itertuples()}
    ctxs = {m: LeakageContext.from_mode(m) for m in (MODE_THREAD, DEFAULT_MODE)}
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    out = {}
    for item in labelled:
        label, path = item.split("=", 1)
        recs = [r for r in _read(Path(path)) if r["question_id"] in accepted]
        row = {"file": Path(path).name, "n": len(recs)}
        for m, ctx in ctxs.items():
            ex = {q: ctx.exclusion_for(q, thread.get(q, []), acc_id[q]) for q in accepted}
            row[f"id_self_retrieval_vs_{m}"] = sum(any(ex[r["question_id"]].excludes(i["question_id"], i["answer_id"])
                                                       for i in r["retrieved_context"]) for r in recs)
        sub = cos = 0
        for r in recs:
            a = accepted[r["question_id"]]
            items = r["retrieved_context"]
            sub += any(shares_substring(a, i["chunk_text"]) for i in items)
            if items:
                v = model.encode([a] + [i["chunk_text"] for i in items], convert_to_numpy=True,
                                 normalize_embeddings=True, batch_size=1)
                cos += float((v[1:] @ v[0]).max()) >= TEXT_COSINE
        cs = [r["cosine_similarity"] for r in recs]
        lat = [r["retrieval_latency_sec"] for r in recs]
        ks = [len(r["retrieved_context"]) for r in recs]
        row.update({
            "text_substring50": sub, "text_cosine095": cos,
            "cosine_mean": round(float(np.mean(cs)), 4), "cosine_median": round(float(np.median(cs)), 4),
            "final_k_lt_k": sum(k < TOP_K for k in ks), "final_k_0": sum(k == 0 for k in ks),
            "latency_mean": round(float(np.mean(lat)), 3), "latency_p95": round(float(np.percentile(lat, 95)), 3),
            "pct_valid_citation": round(100 * float(np.mean([bool(r.get("has_valid_citation")) for r in recs])), 1),
            "leakage_modes": sorted({(r.get("leakage") or {}).get("mode", "thread (pre-Amendment 8 record)") for r in recs}),
        })
        out[label] = row
        print(label, row)
    (AUDIT_DIR / "devruns.json").write_text(json.dumps(out, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieve")
    r.add_argument("--condition", choices=["B", "C", "D"], required=True)
    r.add_argument("--modes", nargs="+", default=list(MODES))
    r.add_argument("--alpha", type=float, default=0.5, help="C only (v3)")
    sub.add_parser("analyze")
    d = sub.add_parser("devruns")
    d.add_argument("runs", nargs="+", help="LABEL=path/to/run.jsonl")
    args = parser.parse_args()
    if args.cmd == "retrieve":
        df, thread = load_dev()
        print(f"[dev] {len(df)} questions (positions {DEV_OFFSET}..{DEV_OFFSET + len(df) - 1})")
        if args.condition == "B":
            retrieve_b(df, args.modes)
        else:
            retrieve_cd(args.condition, df, thread, args.modes, args.alpha)
    elif args.cmd == "devruns":
        devruns(args.runs)
    else:
        analyze()


if __name__ == "__main__":
    main()
