"""
build_blind_judge_files.py
--------------------------
Builds the blinded human-judging files from the official n=384 runs A, B, C, D.
Read-only on run data; no API call, no generation, no judge run.

Outputs (default: results/blind_judging/):
  judge_pass1_halluc.csv   1,536 rows  row_id, question, reference_answer, generated_answer
  judge_pass2_ansrel.csv   1,536 rows  row_id, question, generated_answer      (different row_ids/order)
  judge_pass3_ctxrel.csv   1,152 rows  row_id, question, chunk_1..chunk_5       (B/C/D only)
  judge_batches/pass{1,2,3}_batch_XXX.csv   same columns; 20 / 20 / 10 rows per batch
  judge_output_templates/pass{1,2,3}_output_template.csv
  private/judge_key.csv    row_id, pass, run_label, question_id   (DO NOT SHARE)

Text is what the existing panel judges saw:
  question   html_to_text(body) + title (+ tags, as each judge's prompt had them):
             pass1 title/tags/body, pass2 title/body (ansrel prompt has no tags; body cut at 3000),
             pass3 title/tags/body (ctxrel prompt)
  reference  html_to_text(ground_truth_answer)
  generated  blind_candidate_v3(llm_answer)  (citation markers and Sources block removed)
  chunks     retrieved_context[i].chunk_text, rank order

Independent shuffles per pass: random.Random(42), (43), (44).
Rerun:  python llm/evaluation/build_blind_judge_files.py [--out-dir DIR]
"""
import argparse
from collections import Counter
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

SEEDS = {1: 42, 2: 43, 3: 44}
BATCH = {1: 20, 2: 20, 3: 10}
N_CHUNKS = 5
RUN_LABELS = ["A", "B", "C", "D"]
# strings that would reveal a run/condition; checked case-insensitively in every pass-1..3 file and batch
FORBIDDEN = [r"condition[_ ]?[a-d]\b", r"\bcondition\b", r"run[_ ]?label", r"config_hash", r"\bpure[ _-]?llm\b",
             r"\bgraph-?rag\b", r"\blight-?rag\b", r"\bb_rag\b", r"\bc_graphrag\b", r"\bd_lightrag\b",
             r"\bgrounded\b", r"\bseed42\b"]


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def official_runs() -> dict:
    reg = json.load(open(ROOT / "results" / "run_registry.json"))
    runs = {}
    for r in reg["runs"]:
        if r["status"] == "official" and r["kind"] == "generation":
            assert r["condition"] not in runs, f"duplicate official run for {r['condition']}"
            runs[r["condition"]] = ROOT / r["files"]["output"]
    assert sorted(runs) == RUN_LABELS, f"expected official runs {RUN_LABELS}, found {sorted(runs)}"
    return runs


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def format_tags(raw):
    if not raw:
        return ""
    return ", ".join(re.findall(r"<([^>]+)>", raw)) or raw


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "blind_judging"))
    args = ap.parse_args(argv)
    load_env()
    from _judge_common import html_to_text, load_question_bodies
    from judge_blinding import blind_candidate_v3

    qp = os.getenv("QUESTIONS_PARQUET")
    assert qp and Path(qp).exists(), f"QUESTIONS_PARQUET not available: {qp!r}"

    runs = official_runs()
    recs = {c: read_jsonl(p) for c, p in runs.items()}
    qids = [r["question_id"] for r in recs["A"]]
    for c in RUN_LABELS:
        got = [r["question_id"] for r in recs[c]]
        assert len(got) == 384 and len(set(got)) == 384, f"run {c}: {len(got)} rows, {len(set(got))} distinct question_id"
        assert set(got) == set(qids), f"run {c}: question_id set differs from run A"

    bodies = load_question_bodies(qp, sorted({int(q) for q in qids}))

    def question_text(r, tags: bool, max_body: int):
        qid = int(r["question_id"])
        body = html_to_text(bodies[qid], max_body) if bodies.get(qid) else ""
        s = f"Question title: {r.get('title', '')}\n"
        if tags and format_tags(r.get("tags")):
            s += f"Tags: {format_tags(r.get('tags'))}\n"
        if body:
            s += f"Question body: {body}\n"
        return s.rstrip("\n")

    # master rows (unshuffled): one per (run, question)
    master = []
    for c in RUN_LABELS:
        for r in recs[c]:
            m = {"run_label": c, "question_id": r["question_id"],
                 "q1": question_text(r, True, 4000), "q2": question_text(r, False, 3000),
                 "q3": question_text(r, True, 4000),
                 "reference": html_to_text(r.get("ground_truth_answer", "") or ""),
                 "answer": blind_candidate_v3(r.get("llm_answer", "") or ""), "chunks": None}
            if c != "A":
                ctx = r.get("retrieved_context") or []
                ranks = [it["rank"] for it in ctx if "rank" in it]
                assert ranks == sorted(ranks), f"run {c} q {r['question_id']}: retrieved_context not in rank order"
                assert len(ctx) <= N_CHUNKS, f"run {c} q {r['question_id']}: {len(ctx)} chunks > {N_CHUNKS}"
                # fewer than 5 chunks happens (short pools): keep the real ones, leave the rest blank
                m["chunks"] = [it.get("chunk_text", "") or "" for it in ctx] + [""] * (N_CHUNKS - len(ctx))
                m["n_chunks"] = len(ctx)
            master.append(m)

    out = Path(args.out_dir)
    (out / "judge_batches").mkdir(parents=True, exist_ok=True)
    (out / "judge_output_templates").mkdir(exist_ok=True)
    (out / "private").mkdir(exist_ok=True)

    cols = {1: ["row_id", "question", "reference_answer", "generated_answer"],
            2: ["row_id", "question", "generated_answer"],
            3: ["row_id", "question"] + [f"chunk_{i}" for i in range(1, N_CHUNKS + 1)]}
    names = {1: "judge_pass1_halluc.csv", 2: "judge_pass2_ansrel.csv", 3: "judge_pass3_ctxrel.csv"}
    templ = {1: ["row_id", "label", "reason"], 2: ["row_id", "label"],
             3: ["row_id"] + [f"label_{i}" for i in range(1, N_CHUNKS + 1)]}
    expect_n = {1: 1536, 2: 1536, 3: 1152}
    key_rows, written = [], {}

    for p in (1, 2, 3):
        pool = [m for m in master if p != 3 or m["chunks"] is not None]
        assert len(pool) == expect_n[p], f"pass {p}: {len(pool)} rows, expected {expect_n[p]}"
        rng = random.Random(SEEDS[p])
        order = list(range(len(pool)))
        rng.shuffle(order)
        ids = rng.sample(range(100000, 1000000), len(pool))  # random, unrelated to order
        rows = []
        for pos, (idx, rid) in enumerate(zip(order, ids)):
            m = pool[idx]
            row_id = f"P{p}-{rid}"
            if p == 1:
                row = [row_id, m["q1"], m["reference"], m["answer"]]
            elif p == 2:
                row = [row_id, m["q2"], m["answer"]]
            else:
                row = [row_id, m["q3"]] + m["chunks"]
            rows.append(row)
            key_rows.append([row_id, p, m["run_label"], m["question_id"]])
        assert len({r[0] for r in rows}) == len(rows), f"pass {p}: duplicate row_id"
        written[p] = rows

        def dump(path, header, data):
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(data)

        dump(out / names[p], cols[p], rows)
        for b in range(0, len(rows), BATCH[p]):
            dump(out / "judge_batches" / f"pass{p}_batch_{b // BATCH[p] + 1:03d}.csv", cols[p], rows[b:b + BATCH[p]])
        dump(out / "judge_output_templates" / f"pass{p}_output_template.csv", templ[p],
             [[r[0]] + [""] * (len(templ[p]) - 1) for r in rows])

    with open(out / "private" / "judge_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "pass", "run_label", "question_id"])
        w.writerows(key_rows)

    # ---------------- verification ----------------
    problems = []
    for p in (1, 2, 3):
        kp = [k for k in key_rows if k[1] == p]
        if len(kp) != expect_n[p]:
            problems.append(f"pass {p}: key has {len(kp)} rows")
        seen = {}
        for _, _, c, q in kp:
            seen.setdefault(c, []).append(q)
        want = RUN_LABELS if p != 3 else ["B", "C", "D"]
        if sorted(seen) != want:
            problems.append(f"pass {p}: runs in key {sorted(seen)}")
        for c, qs in seen.items():
            if len(qs) != 384 or len(set(qs)) != 384:
                problems.append(f"pass {p} run {c}: {len(qs)} rows / {len(set(qs))} distinct question_id")
        nb = len(list((out / "judge_batches").glob(f"pass{p}_batch_*.csv")))
        if nb != -(-expect_n[p] // BATCH[p]):
            problems.append(f"pass {p}: {nb} batches")
    # no row can be matched across passes: row_id sets disjoint, order differs
    all_ids = [k[0] for k in key_rows]
    if len(set(all_ids)) != len(all_ids):
        problems.append("row_id reused across passes")
    # no run label / condition name anywhere in passes 1-3, batches, templates (key lives in private/)
    pat = re.compile("|".join(FORBIDDEN), re.I)
    run_ids = [p.stem for p in runs.values()] + [p.name.split("_")[-1].split(".")[0] for p in runs.values()]
    pub_files = list(out.glob("judge_pass*.csv")) + list((out / "judge_batches").glob("*.csv")) \
        + list((out / "judge_output_templates").glob("*.csv"))
    hits = []
    for pf in pub_files:
        hdr = open(pf, encoding="utf-8").readline().lower()
        if "run_label" in hdr or "condition" in hdr:
            problems.append(f"{pf.name}: header carries a run label column")
        text = open(pf, encoding="utf-8").read()
        for m in pat.finditer(text):
            hits.append((pf.name, m.group(0), text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")))
        for rid in set(run_ids):
            if rid and rid in text:
                problems.append(f"{pf.name}: contains run identifier {rid}")
    short = Counter((m["run_label"], m["n_chunks"]) for m in master if m["chunks"] is not None and m["n_chunks"] < N_CHUNKS)
    print(f"wrote {out}")
    if short:
        print(f"  rows with <{N_CHUNKS} chunks (blank chunk_N columns, nothing to label): {dict(short)}")
    for p in (1, 2, 3):
        print(f"  pass {p}: {len(written[p])} rows, {len(list((out / 'judge_batches').glob(f'pass{p}_batch_*.csv')))} batches")
    if hits:
        by = {}
        for f, w, ctx in hits:
            by.setdefault((w.lower()), []).append((f, ctx))
        print(f"WARNING: {len(hits)} content hit(s) on condition-like words (inside question/answer/chunk text):")
        for w, lst in by.items():
            print(f"  '{w}': {len(lst)} hit(s); e.g. {lst[0][0]}: ...{lst[0][1]}...")
    if problems:
        print("VERIFICATION FAILED:")
        for x in problems:
            print("  -", x)
        return 1
    print("verification OK: counts, one row per question_id per run, no run label/identifier in passes 1-3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
