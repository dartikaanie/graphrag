"""
build_claude_judge_inputs.py
----------------------------
Blinded input for an additional judge (Claude, run by hand in chat) on the official n=384 runs A-D, judge-v2.3
instrument. READ-ONLY on every official result; no API call; nothing official is modified.

Writes (default: <repo>/claude_judge/):
  system_prompt_v2_3.txt      judge_prompt_v2_3.SYSTEM_PROMPT (after replacements)
  user_template_v2_3.txt      judge_prompt_v2_2.USER_TEMPLATE (also printed)
  batches/batch_XXX.jsonl     5 items each: {"row_id", "user_message"} only
  key.csv                     NOT SHARED: row_id, item_id, run_label, run_id, question_id, has_context, n_context_items

user_message = messages[1]["content"] of judge_prompt_v2_3.build_messages with the SAME preparation the panel used
(prepare_judge_input, blind-v3, [K1]..[K5], citation labels). Verified: the rebuilt [system, user] messages hash to the
messages_sha256 stored in the official per-judge records, for every item.

The A-D <-> R1/R3/R5/R9 mapping is read from the official judge files (run_label of their records), not assumed.

Rerun:  python llm/evaluation/build_claude_judge_inputs.py [--out-dir DIR]
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

SEED = 42
BATCH_SIZE = 5
RUN_LABELS = ["A", "B", "C", "D"]
PROMPT_VERSION = "judge-v2.3"
# nothing in a batch file may reveal run/condition/provenance (the instrument's own "no retrieved context" text for A is
# part of the judge input and is not a label)
FORBIDDEN = [r"\bcondition\b", r"run[_ ]?label", r"run_id", r"item_id", r"source_file", r"config_hash",
             r"\bpure[ _-]?llm\b", r"\bgraph-?rag\b", r"\blight-?rag\b", r"\bb_rag\b", r"\bc_graphrag\b",
             r"\bd_lightrag\b", r"seed42", r"panel_official", r"\bdeepseek\b", r"\bllama\b", r"\bqwen"]


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def official_judge_files() -> dict:
    """{run_label: (cell id, per-judge records file)} read from the registry AND the files' own records."""
    reg = json.load(open(ROOT / "results" / "run_registry.json"))
    found = {}
    for r in reg["runs"]:
        jc = r.get("judge_config") or {}
        if not (r["status"] == "official" and r["kind"] == "judge" and jc.get("prompt_version") == PROMPT_VERSION
                and jc.get("panel_id") == "panel-v2" and "metric" not in jc):
            continue
        path = ROOT / r["files"]["output"]
        recs = [x for x in read_jsonl(path) if x.get("prompt_version") == PROMPT_VERSION and x.get("panel_id") == "panel-v2"]
        labels = {x["condition"] for x in recs}
        assert len(labels) == 1, f"{path.name}: records of several conditions {labels}"
        label = labels.pop()
        cell = re.search(r"_(R\d+)_", path.name).group(1)
        assert label not in found, f"two official judge files for run {label}"
        found[label] = (cell, path, recs)  # keyed by condition letter (A-D); records' run_label is e.g. "B-grounded"
    assert sorted(found) == RUN_LABELS, f"official judge files cover conditions {sorted(found)}, expected {RUN_LABELS}"
    return found


def build_items():
    """All 1,536 items with the exact messages the official judges got. Returns (items, mapping) where each item
    carries item_id, run_label, run_id, question_id, has_context, n_context_items, offered_items, citation_map,
    messages, messages_sha256, source_file, config_hash, condition."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from _judge_common import html_to_text, load_question_bodies
    from judge_panel import _messages_sha256, judge_version_spec, load_panel_items
    from llm_judge_hallucination_v2_2 import has_context_for

    spec = judge_version_spec(PROMPT_VERSION)
    judge_files = official_judge_files()
    official_by_item = {}
    for label, (cell, path, recs) in judge_files.items():
        for x in recs:
            official_by_item.setdefault(x["item_id"], x)
    # generation files of the official runs = the source_file the official judge records point to
    run_files = []
    for label in RUN_LABELS:
        srcs = {x["source_file"] for x in judge_files[label][2]}
        assert len(srcs) == 1, f"run {label}: judge records point to several source files"
        run_files.append(srcs.pop())
    items = load_panel_items(run_files)
    qp = os.getenv("QUESTIONS_PARQUET")
    assert qp and Path(qp).exists(), f"QUESTIONS_PARQUET not available: {qp!r}"
    bodies = {q: html_to_text(b) for q, b in
              load_question_bodies(qp, sorted({it["question_id"] for it in items})).items()}
    out = []
    for it in items:
        prepared = spec.prepare_judge_input(it["candidate"], it["retrieved_context"], has_context_for(it["condition"]))
        body = bodies.get(it["question_id"]) or "[question body not found in parquet]"
        messages = spec.build_messages(it["title"], body, it["tags"], html_to_text(it["ground_truth_answer_html"]),
                                       it["candidate"], prepared=prepared)
        out.append({"item_id": it["item_id"], "run_label": it["run_label"], "run_id": it["run_id"],
                    "condition": it["condition"], "question_id": it["question_id"],
                    "config_hash": it.get("config_hash"), "source_file": it["source_file"],
                    "has_context": prepared["has_context"], "n_context_items": prepared["n_context_items"],
                    "offered_items": prepared["offered_items"], "citation_map": prepared["citation_map"],
                    "messages": messages, "messages_sha256": _messages_sha256(messages)})
    return out, judge_files, official_by_item, spec


def verify(items, judge_files, official_by_item):
    problems = []
    if len(items) != 1536:
        problems.append(f"{len(items)} items, expected 1,536")
    ids = [it["item_id"] for it in items]
    if len(set(ids)) != len(ids):
        problems.append("duplicate item_id")
    for c in RUN_LABELS:
        qs = [it["question_id"] for it in items if it["condition"] == c]
        if len(qs) != 384 or len(set(qs)) != 384:
            problems.append(f"run {c}: {len(qs)} items / {len(set(qs))} distinct question_id")
    if set(ids) != set(official_by_item):
        problems.append(f"item set differs from the official judge records ({len(set(ids) ^ set(official_by_item))} items)")
    n_bad = 0
    for it in items:
        o = official_by_item.get(it["item_id"])
        if o is None:
            continue
        if o["messages_sha256"] != it["messages_sha256"] or o["run_label"] != it["run_label"] \
                or o["has_context"] != it["has_context"] or o["n_context_items"] != it["n_context_items"]:
            n_bad += 1
    if n_bad:
        problems.append(f"{n_bad} item(s) whose rebuilt messages/has_context differ from the official records")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "claude_judge"))
    args = ap.parse_args(argv)
    items, judge_files, official_by_item, spec = build_items()
    problems = verify(items, judge_files, official_by_item)

    import judge_prompt_v2_2 as v22
    import judge_prompt_v2_3 as v23
    out = Path(args.out_dir)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    for old in (out / "batches").glob("batch_*.jsonl"):
        old.unlink()
    (out / "system_prompt_v2_3.txt").write_text(v23.SYSTEM_PROMPT, encoding="utf-8")
    (out / "user_template_v2_3.txt").write_text(v22.USER_TEMPLATE, encoding="utf-8")

    rng = random.Random(SEED)
    order = list(range(len(items)))
    rng.shuffle(order)
    row_ids = [f"CJ-{n}" for n in rng.sample(range(100000, 1000000), len(items))]  # random, unrelated to order
    rows = []
    with open(out / "key.csv", "w", newline="", encoding="utf-8") as kf:
        kw = csv.writer(kf)
        kw.writerow(["row_id", "item_id", "run_label", "run_id", "question_id", "has_context", "n_context_items"])
        for pos, (idx, rid) in enumerate(zip(order, row_ids)):
            it = items[idx]
            rows.append({"row_id": rid, "user_message": it["messages"][1]["content"]})
            kw.writerow([rid, it["item_id"], it["run_label"], it["run_id"], it["question_id"],
                         it["has_context"], it["n_context_items"]])
    n_batches = 0
    for b in range(0, len(rows), BATCH_SIZE):
        n_batches += 1
        with open(out / "batches" / f"batch_{n_batches:03d}.jsonl", "w", encoding="utf-8") as f:
            for r in rows[b:b + BATCH_SIZE]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # blinding check on everything that may be shared
    pat = re.compile("|".join(FORBIDDEN), re.I)
    hits = Counter()
    run_ids = {it["run_id"] for it in items} | {it["config_hash"] for it in items if it["config_hash"]}
    for pf in list((out / "batches").glob("*.jsonl")) + [out / "system_prompt_v2_3.txt", out / "user_template_v2_3.txt"]:
        text = pf.read_text(encoding="utf-8")
        for m in pat.finditer(text):
            hits[(pf.parent.name if pf.parent.name == "batches" else pf.name, m.group(0).lower())] += 1
        for rid in run_ids:
            if rid and rid in text:
                problems.append(f"{pf.name}: contains run identifier {rid}")
    for pf in (out / "batches").glob("*.jsonl"):
        for line in open(pf, encoding="utf-8"):
            if set(json.loads(line)) != {"row_id", "user_message"}:
                problems.append(f"{pf.name}: keys other than row_id/user_message")
    print(f"wrote {out}: {len(rows)} items in {n_batches} batches of {BATCH_SIZE}")
    print("mapping read from the official judge files: " +
          ", ".join(f"{c} = {judge_files[c][0]}" for c in RUN_LABELS))
    print("\n=== USER_TEMPLATE (judge-v2.3 = judge-v2.2's) ===\n" + v22.USER_TEMPLATE)
    if hits:
        print("content hits on provenance-like words (inside question/answer/context text; review):")
        for (f, w), n in sorted(hits.items()):
            print(f"  {f}: '{w}' x{n}")
    if problems:
        print("VERIFICATION FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("verification OK: 1,536 items, 384 per run, item set and messages_sha256 identical to the official records, "
          "batch files hold only row_id + user_message")
    return 0


if __name__ == "__main__":
    sys.exit(main())
