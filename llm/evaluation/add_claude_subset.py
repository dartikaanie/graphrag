"""
add_claude_subset.py
--------------------
Pre-specified subset of the Claude judge input: 40 question_ids sampled with random.Random(42) from the 384 official
questions (the same questions for A, B, C, D) -> 160 items. Adds files only; existing claude_judge files are not changed.

  claude_judge/subset_q40_row_ids.txt   160 row_ids, in the shuffled subset order (random.Random(42))
  claude_judge/pilot_row_ids.txt        the first 20 row_ids of that order (must cover A, B, C, D)
  claude_judge/batches_subset/batch_XXX.jsonl   the subset's {"row_id","user_message"} lines, 5 per batch, same order

Rerun:  python llm/evaluation/add_claude_subset.py
"""
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CJ = ROOT / "claude_judge"
SEED, N_Q, BATCH, N_PILOT = 42, 40, 5, 20


def main() -> int:
    key = list(csv.DictReader(open(CJ / "key.csv", encoding="utf-8")))
    msgs = {}
    for p in sorted((CJ / "batches").glob("batch_*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            msgs[r["row_id"]] = r["user_message"]
    assert len(key) == len(msgs) == 1536
    qids = sorted({int(k["question_id"]) for k in key})
    assert len(qids) == 384
    rng = random.Random(SEED)
    sample = sorted(rng.sample(qids, N_Q))
    sset = set(sample)
    rows = [k for k in key if int(k["question_id"]) in sset]  # key order = unrelated to condition
    assert len(rows) == 4 * N_Q
    per_q = Counter(int(k["question_id"]) for k in rows)
    assert all(v == 4 for v in per_q.values()), "each sampled question must appear once per run"
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    rows = [rows[i] for i in order]
    ids = [r["row_id"] for r in rows]
    (CJ / "subset_q40_row_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (CJ / "pilot_row_ids.txt").write_text("\n".join(ids[:N_PILOT]) + "\n", encoding="utf-8")
    out = CJ / "batches_subset"
    out.mkdir(exist_ok=True)
    for old in out.glob("batch_*.jsonl"):
        old.unlink()
    for b in range(0, len(ids), BATCH):
        with open(out / f"batch_{b // BATCH + 1:03d}.jsonl", "w", encoding="utf-8") as f:
            for rid in ids[b:b + BATCH]:
                f.write(json.dumps({"row_id": rid, "user_message": msgs[rid]}, ensure_ascii=False) + "\n")
    cond = {k["row_id"]: k["item_id"].split("-")[0] for k in key}  # item_id = "<cond>-<hash>:<qid>"
    print(f"sampled question_ids ({N_Q}): {sample}")
    print("subset per run:", dict(sorted(Counter(cond[i] for i in ids).items())))
    print("pilot per run:", dict(sorted(Counter(cond[i] for i in ids[:N_PILOT]).items())))
    print(f"{len(ids)} row_ids, {len(ids) // BATCH} subset batches of {BATCH}")
    if set(cond[i] for i in ids[:N_PILOT]) != {"A", "B", "C", "D"}:
        print("PILOT DOES NOT COVER ALL FOUR RUNS")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
