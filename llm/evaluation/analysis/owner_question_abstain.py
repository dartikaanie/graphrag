"""Read-only test: does D abstain more because its top-5 answers are chosen by similarity to the owning
Question rather than to the answer text?

Set S = questions ABSTAIN in D (final panel label) but not in C; R = the other questions of the n=384 set.
Per question and condition (C, D), from the STORED top-5 retrieved_context only:
  1. overlap       shared answer_ids between C and D top-5 (one value per question)
  2. ans_sim       cosine(query, final chunk_text) per item -> mean and max
  3. q_sim         cosine(query, owning Question embedding) per item -> mean
  4. n_owner_q     distinct owning Questions among the top-5

Query vector = the cached embedding of the evaluation question, exactly what C and D used at run time
(c_graphrag.py:1186, d_lightrag.py:903). Owning Question vectors come from the same cache. Chunk texts are
encoded with all-MiniLM-L6-v2, one text per call, L2-normalised (as c_graphrag.compute_candidate_similarities).
No generation, judge or external API call; existing run files are only read.

Usage: python -m llm.evaluation.analysis.owner_question_abstain [--out-dir results/followup_owner_question]
"""

import argparse
import csv
import glob
import json
import random
import statistics as st
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EMBED_DIM = 384
EXPECT = {  # final panel label counts: FAKTUAL / SEBAGIAN / PENUH / ABSTAIN
    "C": (242, 4, 66, 72),
    "D": (241, 9, 43, 91),
}
LABELS = ("FAKTUAL", "HALUSINASI_SEBAGIAN", "HALUSINASI_PENUH", "ABSTAIN")
PANEL = {"C": "panel_official_R5_judge-v2.3_panel.jsonl", "D": "panel_official_R9_judge-v2.3_panel.jsonl"}
RUN_GLOB = {
    "C": "llm/c_graphrag/results/condition_c_*n384*alpha0-5_v3_95d2f39d2c.jsonl",
    "D": "llm/d_lightrag/results/condition_d_*n384*_5ae879c9c0.jsonl",
}
RUN_ID = {"C": "C-c522f523c9", "D": "D-e9c4125742"}  # panel run_id (== History id)


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def env_value(key):
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"{key} not in .env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "results" / "followup_owner_question"))
    ap.add_argument("--kg-workspace-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- labels + verification --------------------------------------------------------------
    label = {}
    for c in "CD":
        recs = [r for r in read_jsonl(REPO / "llm/evaluation/results" / PANEL[c]) if r["run_id"] == RUN_ID[c]]
        label[c] = {int(r["question_id"]): r["panel_label"] for r in recs}
        counts = tuple(sum(1 for v in label[c].values() if v == lab) for lab in LABELS)
        print(f"{c}: n={len(recs)} counts={counts} expected={EXPECT[c]}")
        if counts != EXPECT[c] or len(recs) != 384:
            raise SystemExit(f"STOP: label counts for {c} differ from expected {EXPECT[c]} (got {counts}, n={len(recs)})")
    qids = sorted(set(label["C"]) & set(label["D"]))
    S = [q for q in qids if label["D"][q] == "ABSTAIN" and label["C"][q] != "ABSTAIN"]
    S_set = set(S)
    print(f"|S|={len(S)} |R|={len(qids) - len(S)}")

    # ---- stored retrieval ---------------------------------------------------------------------
    runs = {}
    for c in "CD":
        files = glob.glob(str(REPO / RUN_GLOB[c]))
        if len(files) != 1:
            raise SystemExit(f"expected one run file for {c}, got {files}")
        runs[c] = {int(r["question_id"]): r for r in read_jsonl(files[0])}
        print(f"{c}: {Path(files[0]).name} ({len(runs[c])} records)")

    # ---- cached Question embeddings (read-only memmap) -------------------------------------------
    ws = Path(args.kg_workspace_dir or env_value("KG_WORKSPACE_DIR"))
    ids_arr = np.load(ws / "question_embeddings_ivf_ids.npy")
    n_total = json.loads((ws / "embed_checkpoint.json").read_text())["next_row"]
    emb = np.memmap(ws / "question_embeddings.f32", dtype="float32", mode="r", shape=(n_total, EMBED_DIM))
    id_to_row = {int(q): i for i, q in enumerate(ids_arr)}

    def qvec(qid):
        row = id_to_row.get(int(qid))
        if row is None:
            return None
        v = np.array(emb[row], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def text_vec(text):
        v = np.asarray(model.encode([text], convert_to_numpy=True, device="cpu", batch_size=1)[0], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / (n if n > 0 else 1e-9)

    # ---- per question ---------------------------------------------------------------------------
    rows, missing = [], {"query": 0, "owner": 0}
    top5 = {}
    for q in qids:
        qv = qvec(q)
        if qv is None:
            missing["query"] += 1
            qv = text_vec(runs["C"][q]["title"])  # fallback: title only (not expected; counted)
        row = {"question_id": q, "set": "S" if q in S_set else "R", "label_C": label["C"][q], "label_D": label["D"][q]}
        ids = {}
        for c in "CD":
            items = (runs[c][q].get("retrieved_context") or [])[:5]
            ids[c] = [int(it["answer_id"]) for it in items]
            asims, qsims, owners = [], [], set()
            for it in items:
                asims.append(float(np.dot(qv, text_vec(it["chunk_text"]))))
                oq = int(it.get("owner_question_id") or it.get("via_question_id"))
                owners.add(oq)
                ov = qvec(oq)
                if ov is None:
                    missing["owner"] += 1
                else:
                    qsims.append(float(np.dot(qv, ov)))
            row[f"n_items_{c}"] = len(items)
            row[f"ans_sim_mean_{c}"] = np.mean(asims) if asims else None
            row[f"ans_sim_max_{c}"] = max(asims) if asims else None
            row[f"q_sim_mean_{c}"] = np.mean(qsims) if qsims else None
            row[f"n_owner_q_{c}"] = len(owners)
        row["overlap"] = len(set(ids["C"]) & set(ids["D"]))
        top5[q] = ids
        rows.append(row)
    print("missing embeddings:", missing)

    fields = list(rows[0])
    with open(out / "per_question.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- tables ---------------------------------------------------------------------------------
    def stats(vals):
        vals = [v for v in vals if v is not None]
        return (len(vals), st.mean(vals), st.median(vals)) if vals else (0, None, None)

    metrics = [("overlap", None), ("ans_sim_mean", "CD"), ("ans_sim_max", "CD"), ("q_sim_mean", "CD"), ("n_owner_q", "CD")]
    table = []
    for s in ("S", "R"):
        sub = [r for r in rows if r["set"] == s]
        for m, conds in metrics:
            for c in (conds or [""]):
                key = m if not c else f"{m}_{c}"
                n, mean, med = stats([r[key] for r in sub])
                table.append({"set": s, "metric": m, "condition": c or "C∩D", "n": n, "mean": mean, "median": med})
    with open(out / "summary_by_set.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "metric", "condition", "n", "mean", "median"])
        w.writeheader()
        w.writerows(table)

    S_rows = [r for r in rows if r["set"] == "S"]
    lower = sum(1 for r in S_rows if r["ans_sim_mean_D"] < r["ans_sim_mean_C"])
    lower_max = sum(1 for r in S_rows if r["ans_sim_max_D"] < r["ans_sim_max_C"])
    with open(out / "S_D_vs_C_answer_text_sim.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_S", "D_mean_ans_sim_lower_than_C", "D_max_ans_sim_lower_than_C"])
        w.writerow([len(S_rows), lower, lower_max])

    # ---- examples -------------------------------------------------------------------------------
    sample = random.Random(42).sample(sorted(S), 10)
    by_q = {r["question_id"]: r for r in rows}
    ex_rows = []
    for q in sample:
        r = by_q[q]
        ex_rows.append({"question_id": q, "query": runs["C"][q]["title"][:200],
                        "C_top5_answer_ids": " ".join(map(str, top5[q]["C"])), "D_top5_answer_ids": " ".join(map(str, top5[q]["D"])),
                        "overlap": r["overlap"], "ans_sim_mean_C": r["ans_sim_mean_C"], "ans_sim_mean_D": r["ans_sim_mean_D"]})
    with open(out / "examples_S.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ex_rows[0]))
        w.writeheader()
        w.writerows(ex_rows)

    # ---- report ---------------------------------------------------------------------------------
    fmt = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
    md = ["# Owner-Question vs answer-text similarity: D abstentions", "",
          f"Label counts verified (C {EXPECT['C']}, D {EXPECT['D']}). |S| = {len(S)} (D ABSTAIN, C not ABSTAIN), |R| = {len(qids) - len(S)}. "
          "Raw numbers only, no significance tests.", "",
          "## Summary (mean / median)", "", "| Set | Metric | Cond. | n | mean | median |", "|---|---|---|---|---|---|"]
    for t in table:
        md.append(f"| {t['set']} | {t['metric']} | {t['condition']} | {t['n']} | {fmt(t['mean'])} | {fmt(t['median'])} |")
    md += ["", f"**D in S, answer-text similarity lower than C for the same question:** mean-sim {lower}/{len(S_rows)}, "
           f"max-sim {lower_max}/{len(S_rows)}.", "",
           "## 10 random examples from S (`random.Random(42).sample(sorted(S), 10)`)", "",
           "| question_id | query (first 200 chars) | C top-5 | D top-5 | overlap | ans_sim mean C | ans_sim mean D |", "|---|---|---|---|---|---|---|"]
    for e in ex_rows:
        md.append(f"| {e['question_id']} | {e['query'].replace('|', '/')} | {e['C_top5_answer_ids']} | {e['D_top5_answer_ids']} | "
                  f"{e['overlap']} | {fmt(e['ans_sim_mean_C'])} | {fmt(e['ans_sim_mean_D'])} |")
    md += ["", "## Assumptions and columns", "",
           "- Labels: `panel_label` in `panel_official_R5_*_panel.jsonl` (C, run C-c522f523c9) and `panel_official_R9_*_panel.jsonl` (D, run D-e9c4125742).",
           "- Retrieval: `retrieved_context[:5]` of the n=384 run files (C `95d2f39d2c`, D `5ae879c9c0`); columns `answer_id`, `chunk_text`, `owner_question_id` (fallback `via_question_id`).",
           "- Query = cached all-MiniLM-L6-v2 embedding of the evaluation question (title + first 1000 body chars), the vector both conditions used at run time; "
           "the query text shown is the stored `title` only (the body is not stored in the records).",
           "- Answer-text similarity = cosine(query, `chunk_text`), the final truncated `Q: title / A: body` text shown to the LLM, encoded one text per call and L2-normalised.",
           "- Question similarity = cosine(query, cached embedding of the owning Question); an item's owner is `owner_question_id`.",
           "- Overlap is one value per question (C∩D), so it is reported once per set. Items fewer than 5 are averaged over what exists (`n_items_*` in per_question.csv).",
           f"- Missing cache embeddings: query {missing['query']}, owner {missing['owner']}.",
           "- Nothing was generated, judged, or sent to an API; run files were only read.", ""]
    (out / "REPORT.md").write_text("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
