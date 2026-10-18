"""
build_sample_ids.py
=====================================
Freezes the evaluation samples (docs/DECISION_C_SCORING.md, "Development
set"; reports/p0_run_matrix_lock.md) into

    llm/sample_ids/test_ids_v1.json   positions 1..384   (test sample)
    llm/sample_ids/dev_ids_v1.json    positions 385..434 (dev set, n = 50)

Read-only on the data: it runs the EXISTING sampler -- the candidate pool
(get_candidate_questions: accepted answer present in the answers parquet,
reservoir sample of 1536 with seed 42) and filter_by_token_limit (<= 2048
tokens) -- once with DuckDB's default thread count (A/B) and once with
threads=2 (C/D), and refuses to write unless both pools are identical and
the test/dev slices of every condition's own sampler (A, B, C, D and
llm/sample_split.py) agree. No Neo4j, no LLM, no network.

    python llm/sample_ids/build_sample_ids.py            # writes both files
    python llm/sample_ids/build_sample_ids.py --check    # compare with the existing files, write nothing

The files are never regenerated for an official run: every condition
checks its sample against them (llm/sample_split.assert_frozen_sample).
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "llm" / "c_graphrag"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from llm.sample_split import (DEFAULT_DEV_OFFSET, FROZEN_FILES, MAX_TEST_N, ids_sha256,  # noqa: E402
                              sample_questions_split)

SEED = 42
OVERSAMPLE_POOL = MAX_TEST_N * 4
N_DEV = 50
VERSION = "v1"


def _pool(questions_parquet: str, answers_parquet: str, threads: int | None):
    import duckdb

    from llm.b_rag import b_condition_b_rag as b

    b.log = lambda *a, **k: None
    con = duckdb.connect()
    if threads is not None:
        con.execute("SET memory_limit='2GB'")
        con.execute(f"SET threads={threads}")
    pool = b.filter_by_token_limit(b.get_candidate_questions(con, questions_parquet, answers_parquet,
                                                             OVERSAMPLE_POOL, SEED))
    con.close()
    return pool


def build(questions_parquet: str, answers_parquet: str) -> dict:
    import c_graphrag as cg

    from llm.a_pure_llm import a_baseline_replication as a
    from llm.b_rag import b_condition_b_rag as b
    from llm.d_lightrag import d_lightrag as d

    for mod in (a, cg, d):
        mod.log = lambda *x, **k: None
    pool_default = _pool(questions_parquet, answers_parquet, None)
    pool_t2 = _pool(questions_parquet, answers_parquet, 2)
    if list(pool_default["Id"]) != list(pool_t2["Id"]):
        raise SystemExit("[STOP] candidate pool differs between DuckDB default threads and threads=2")

    test = [int(i) for i in sample_questions_split(pool_default, MAX_TEST_N, SEED, "test")["Id"]]
    dev = [int(i) for i in sample_questions_split(pool_default, N_DEV, SEED, "dev", DEFAULT_DEV_OFFSET)["Id"]]
    own_test = {
        "A": a.sample_questions(pool_default, MAX_TEST_N, SEED),
        "B": b.sample_questions(pool_default, MAX_TEST_N, SEED),
        "C": cg.sample_questions_split(pool_t2, MAX_TEST_N, SEED, split="test"),
        "D": d.sample_questions(pool_t2, MAX_TEST_N, SEED),
    }
    for cond, df in own_test.items():
        if [int(i) for i in df["Id"]] != test:
            raise SystemExit(f"[STOP] condition {cond}'s own sampler gives a different test sample")
    if [int(i) for i in cg.sample_questions_split(pool_t2, N_DEV, SEED, split="dev")["Id"]] != dev:
        raise SystemExit("[STOP] Condition C's own dev sampler gives a different dev set")
    if set(test) & set(dev) or len(set(test)) != MAX_TEST_N or len(set(dev)) != N_DEV:
        raise SystemExit("[STOP] test/dev not disjoint or not unique")

    from llm.manifest import _git_info

    common = {
        "version": VERSION, "seed": SEED, "oversample_pool": OVERSAMPLE_POOL, "token_limit": 2048,
        "n_pool_after_token_filter": len(pool_default), "pool_ids_sha256": ids_sha256(pool_default["Id"]),
        "sampler": "get_candidate_questions + filter_by_token_limit + llm/sample_split.sample_questions_split "
                   "(df.sample(frac=1.0, random_state=seed)); identical for A, B, C, D and for DuckDB "
                   "default threads vs threads=2",
        "data": {k: {"name": Path(v).name, "size_bytes": os.path.getsize(v)}
                 for k, v in (("questions_parquet", questions_parquet), ("answers_parquet", answers_parquet))},
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_info().get("commit_sha"),
    }
    return {
        "test": {"split": "test", "positions": f"1..{MAX_TEST_N}", "dev_offset": None, "n": len(test),
                 "ids": test, "ids_sha256": ids_sha256(test), **common},
        "dev": {"split": "dev", "positions": f"{DEFAULT_DEV_OFFSET}..{DEFAULT_DEV_OFFSET + N_DEV - 1}",
                "dev_offset": DEFAULT_DEV_OFFSET, "n": len(dev), "ids": dev, "ids_sha256": ids_sha256(dev), **common},
    }


def main():
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare with the existing files; write nothing")
    args = parser.parse_args()
    built = build(os.environ["QUESTIONS_PARQUET"], os.environ["ANSWERS_PARQUET"])
    for split, data in built.items():
        path = FROZEN_FILES[split]
        if args.check:
            old = json.loads(path.read_text())
            same = old["ids"] == data["ids"] and old["ids_sha256"] == data["ids_sha256"]
            print(f"[{split}] {path.name}: {'identical' if same else 'DIFFERENT'} (ids_sha256 {data['ids_sha256'][:16]})")
            continue
        if path.exists():
            raise SystemExit(f"[STOP] {path} exists -- frozen lists are never overwritten; use --check")
        path.write_text(json.dumps(data, indent=1) + "\n")
        print(f"[{split}] wrote {path} (n={data['n']}, ids_sha256 {data['ids_sha256']})")


if __name__ == "__main__":
    main()
