"""
calibration_generate.py
=====================================
Generates the judge CALIBRATION set (docs/DECISION_C_SCORING.md, Amendment
4): answers for the dev split (positions 385-434, n = 50) under conditions
A, B-plain and B-grounded -- none depends on C's alpha -- with the SAME
generation model, prompt version and retrieval settings as the pilot. 150
answers, run one condition after another (8 GB machine).

Each condition is run through its own existing CLI (a_baseline_
replication.py / b_condition_b_rag.py), from that script's own folder, so
the generation code path is exactly the pilot's.

Rules enforced here:
  - starts only from a CLEAN working tree; the commit is recorded in the
    calibration registry and (by llm.manifest) in every run manifest;
  - every run is written to the calibration registry BEFORE it starts, so
    the judge guard (_judge_common.assert_calibration_judging_allowed)
    refuses judge-v2.x on these outputs -- even partial ones -- until the
    adjudicated human reference exists;
  - NO judge is ever run by this script.

CARA PAKAI (from the repo root)
----------------------------------------------------------------------------
    python3 llm/evaluation/calibration_generate.py --dry-run   # plan only, no API call
    python3 llm/evaluation/calibration_generate.py             # asks y/N, then generates
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))

from _judge_common import CALIBRATION_REGISTRY_PATH  # noqa: E402
from llm import archive_paths  # noqa: E402
from llm.repo_state import repo_relative, repo_state  # noqa: E402

# Generation settings -- identical to the 2026-10-05 pilot (see each
# condition's logs/run_history.jsonl); assert_matches_pilot() checks this.
PROVIDER, MODEL, SEED = "openai", "gpt-4o-mini", 42
N_DEV, DEV_OFFSET = 50, 385
OVERSAMPLE_POOL = 1536
B_TOP_K, B_INDEX_POOL, B_TOKEN_CHUNK_LIMIT = 5, 8000, 400
N_ANNOTATION_ITEMS = 60
PRICE_PER_M_INPUT, PRICE_PER_M_OUTPUT = 0.15, 0.60  # gpt-4o-mini, USD (judge_clients.py registry, 2026-10-05)

A_DIR, B_DIR = REPO_ROOT / "llm" / "a_pure_llm", REPO_ROOT / "llm" / "b_rag"


def calibration_runs() -> list[dict]:
    """The three runs, with their deterministic output paths."""
    from llm.a_pure_llm import a_baseline_replication as a
    from llm.b_rag import b_condition_b_rag as b
    from llm.manifest import compute_config_hash

    common = ["--provider", PROVIDER, "--model", MODEL, "--n-sample", str(N_DEV), "--seed", str(SEED),
              "--oversample-pool", str(OVERSAMPLE_POOL), "--sample-split", "dev", "--dev-offset", str(DEV_OFFSET)]
    # The calibration batch ran on B's legacy 8,000-answer index (before the KG
    # corpus, reports/b_corpus_parity.md) -- pinned so this keeps describing it.
    b_common = common + ["--corpus", "legacy_sample", "--top-k", str(B_TOP_K), "--index-pool", str(B_INDEX_POOL),
                         "--token-chunk-limit", str(B_TOKEN_CHUNK_LIMIT), "--require-citation"]
    runs = [{
        "run_label": "A [dev]", "condition": "A", "cwd": A_DIR, "script": "a_baseline_replication.py", "args": common,
        "output_path": A_DIR / a.build_output_path("results", PROVIDER, MODEL, N_DEV, SEED,
                                                   oversample_pool=OVERSAMPLE_POOL, sample_split="dev"),
        "config": a.build_config(PROVIDER, MODEL, N_DEV, SEED, OVERSAMPLE_POOL, "dev"),
        "pilot_file": A_DIR / a.build_output_path("results", PROVIDER, MODEL, 10, SEED, oversample_pool=OVERSAMPLE_POOL),
    }]
    for label, grounding in (("B-plain [dev]", False), ("B-grounded [dev]", True)):
        runs.append({
            "run_label": label, "condition": "B", "cwd": B_DIR, "script": "b_condition_b_rag.py",
            "args": b_common + ["--require-grounding" if grounding else "--no-require-grounding"],
            "output_path": B_DIR / b.build_output_path("results", PROVIDER, MODEL, N_DEV, SEED, True, grounding,
                                                       OVERSAMPLE_POOL, B_TOP_K, "dev", corpus="legacy_sample"),
            "config": b.build_config(PROVIDER, MODEL, N_DEV, SEED, OVERSAMPLE_POOL, B_TOP_K, True, grounding, "dev",
                                     corpus="legacy_sample"),
            "pilot_file": B_DIR / b.build_output_path("results", PROVIDER, MODEL, 10, SEED, True, grounding,
                                                      OVERSAMPLE_POOL, B_TOP_K, corpus="legacy_sample"),
        })
    for r in runs:
        r["config_hash"] = compute_config_hash(r["config"])
    return runs


def _sha256_ids(ids) -> str:
    import hashlib

    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode("utf-8")).hexdigest()


def data_fingerprint() -> dict:
    """What data this batch was generated from, recorded in the registry so
    a later run can be compared against it (the stage-1 manifests did not
    record this): the parquet paths with size and mtime, a SHA-256 of the
    filtered candidate pool's question IDs in pool order (the order the
    fixed permutation consumes), and a SHA-256 of the 50 dev question IDs
    in order. Built with condition A's own loader; condition B's loader is
    checked to give the identical pool and dev IDs."""
    import os

    import duckdb
    from dotenv import load_dotenv

    from llm.a_pure_llm import a_baseline_replication as a
    from llm.b_rag import b_condition_b_rag as b
    from llm.sample_split import sample_questions_split

    load_dotenv(REPO_ROOT / ".env")
    qp, ap = os.environ["QUESTIONS_PARQUET"], os.environ["ANSWERS_PARQUET"]
    files = {}
    for name, path in (("questions_parquet", qp), ("answers_parquet", ap)):
        st = os.stat(path)
        files[name] = {"path": path, "size_bytes": st.st_size,
                       "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
    con = duckdb.connect()
    pools = {}
    for name, mod in (("A", a), ("B", b)):
        pools[name] = mod.filter_by_token_limit(mod.get_candidate_questions(con, qp, ap, OVERSAMPLE_POOL, SEED))
    con.close()
    pool_ids = pools["A"]["Id"].astype(int).tolist()
    dev_ids = sample_questions_split(pools["A"], N_DEV, SEED, "dev", DEV_OFFSET)["Id"].astype(int).tolist()
    b_dev = sample_questions_split(pools["B"], N_DEV, SEED, "dev", DEV_OFFSET)["Id"].astype(int).tolist()
    if pools["B"]["Id"].astype(int).tolist() != pool_ids or b_dev != dev_ids:
        raise RuntimeError("conditions A and B do not produce the same candidate pool / dev questions")
    return {**files, "n_pool_after_token_filter": len(pool_ids), "pool_question_ids_sha256": _sha256_ids(pool_ids),
            "pool_question_ids_order": "candidate pool order after the token filter (input of the fixed permutation)",
            "n_dev": len(dev_ids), "dev_question_ids_sha256": _sha256_ids(dev_ids), "dev_question_ids": dev_ids}


def pilot_stats(run: dict) -> dict:
    """Mean tokens per answer and seconds per answer from the matching
    pilot run (same condition/grounding, n=10) -- the basis of the dry-run
    estimate. Raises if the pilot file is missing (no silent default)."""
    pilot = run["pilot_file"]
    # pilot files and run histories are in the pre-official archive (2026-10-19)
    records = [json.loads(line) for line in open(archive_paths.resolve(pilot)) if line.strip()]
    usage = [r.get("usage_full") or {} for r in records]
    history = REPO_ROOT / "llm" / pilot.parent.parent.name / "logs" / "run_history.jsonl"
    duration, prompt_version = None, None
    for line in open(archive_paths.resolve(history)):
        h = json.loads(line)
        if Path(str(h.get("output_path", ""))).name == pilot.name and h.get("status") == "success":
            duration, prompt_version = h.get("duration_sec"), h.get("prompt_version")
    return {
        "pilot_file": pilot.name, "n": len(records),
        "prompt_version": prompt_version, "model": records[0].get("llm_model"),
        "mean_prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage) / len(usage),
        "mean_completion_tokens": sum(u.get("completion_tokens", 0) for u in usage) / len(usage),
        "sec_per_item": (duration / len(records)) if duration else None,
    }


def build_plan(runs: list[dict]) -> dict:
    rows, tot_in, tot_out, tot_sec = [], 0, 0, 0.0
    for r in runs:
        st = pilot_stats(r)
        est_in, est_out = st["mean_prompt_tokens"] * N_DEV, st["mean_completion_tokens"] * N_DEV
        est_sec = (st["sec_per_item"] or 0) * N_DEV
        tot_in, tot_out, tot_sec = tot_in + est_in, tot_out + est_out, tot_sec + est_sec
        rows.append({"run_label": r["run_label"], "items": N_DEV, "output": repo_relative(r["output_path"]),
                     "config_hash": r["config_hash"], "exists": archive_paths.resolve(r["output_path"]).exists(),
                     "est_tokens_in": round(est_in), "est_tokens_out": round(est_out), "est_sec": round(est_sec),
                     "pilot": st})
    return {"rows": rows, "items": N_DEV * len(runs), "est_tokens_in": round(tot_in), "est_tokens_out": round(tot_out),
            "est_cost_usd": round(tot_in / 1e6 * PRICE_PER_M_INPUT + tot_out / 1e6 * PRICE_PER_M_OUTPUT, 4),
            "est_time_sec": round(tot_sec)}


def format_plan(plan: dict, git: dict) -> str:
    from llm.prompts import PROMPT_VERSION

    lines = [
        "=== CALIBRATION GENERATION -- PLAN (no API call made) ===",
        f"Split: dev, positions {DEV_OFFSET}-{DEV_OFFSET + N_DEV - 1} (1-based), seed {SEED}, oversample pool {OVERSAMPLE_POOL}",
        f"Generation: {PROVIDER} / {MODEL}, generation prompt {PROMPT_VERSION}; B: top_k={B_TOP_K}, "
        f"index_pool={B_INDEX_POOL}, token_chunk_limit={B_TOKEN_CHUNK_LIMIT} (same as the pilot)",
        f"git_commit={git['git_commit']} git_tree_clean={git['git_tree_clean']}",
        "",
        "| run | items | est. tokens in / out | est. time | output |",
        "|---|---|---|---|---|",
    ]
    for r in plan["rows"]:
        lines.append(f"| {r['run_label']} | {r['items']} | {r['est_tokens_in']} / {r['est_tokens_out']} | "
                     f"~{r['est_sec']}s | {r['output']}{' (EXISTS -- will resume)' if r['exists'] else ''} |")
    lines += [
        "",
        f"Total items: {plan['items']} answers (1 generation call each; no judge call)",
        f"Estimated tokens (in/out): {plan['est_tokens_in']} / {plan['est_tokens_out']} "
        f"(source: mean per answer in the matching n=10 pilot runs)",
        f"Estimated cost: ${plan['est_cost_usd']:.4f} (gpt-4o-mini ${PRICE_PER_M_INPUT}/M in, ${PRICE_PER_M_OUTPUT}/M out)",
        f"Estimated time: ~{plan['est_time_sec']}s (~{plan['est_time_sec'] / 60:.0f} min), runs sequential "
        f"(source: pilot run durations per answer; includes each run's model/index loading)",
    ]
    return "\n".join(lines)


def write_registry(runs: list[dict], git: dict, registry_path: Path, started_at: str,
                   data: dict | None = None) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    notes = []
    if registry_path.exists():  # keep hand-recorded notes across rewrites
        try:
            notes = json.loads(registry_path.read_text()).get("notes", [])
        except (OSError, json.JSONDecodeError):
            notes = []
    registry_path.write_text(json.dumps({
        "notes": notes,
        "purpose": "judge calibration set (docs/DECISION_C_SCORING.md, Amendment 4)",
        "created_at_utc": started_at, "git_commit": git["git_commit"],
        "git_tree_clean_at_start": git["git_tree_clean"],
        "sample_split": "dev", "n_dev": N_DEV, "dev_offset": DEV_OFFSET, "seed": SEED,
        "n_annotation_items": N_ANNOTATION_ITEMS,
        "data": data,
        "runs": [{"run_label": r["run_label"], "condition": r["condition"],
                  "output_path": repo_relative(r["output_path"]), "config_hash": r["config_hash"],
                  "n_records": r.get("n_records"),
                  "question_ids_match_dev_ids_in_order": r.get("question_ids_match_dev_ids_in_order"),
                  # tree state just before this run started: any changed path must be one of this
                  # batch's own outputs (`changed_paths_not_own_output` must stay empty)
                  "changed_paths_at_run_start": r.get("changed_paths_at_run_start"),
                  "changed_paths_not_own_output": r.get("changed_paths_not_own_output")} for r in runs],
    }, indent=2) + "\n")


def write_calibration_run_history(runs: list[dict], history_path: Path | None = None) -> Path:
    """Copies each calibration run's LATEST successful run_history entry,
    verbatim, from the condition's own (git-ignored) logs/run_history.jsonl
    into calibration/calibration_run_history.jsonl, which IS committed and
    is read by _run_metadata.resolve_run_metadata(). Verbatim, so the
    run_id (a hash of run_started_at + output_path) is unchanged."""
    from _run_metadata import CALIBRATION_HISTORY_PATH

    history_path = history_path or CALIBRATION_HISTORY_PATH
    lines = []
    for r in runs:
        source = r["cwd"] / "logs" / "run_history.jsonl"
        matching = [line.strip() for line in open(source) if line.strip()
                    and Path(str(json.loads(line).get("output_path", ""))).name == r["output_path"].name
                    and json.loads(line).get("status") == "success"]
        if not matching:
            raise RuntimeError(f"no successful run_history entry for {r['output_path'].name} in {source}")
        lines.append(max(matching, key=lambda line: json.loads(line).get("run_started_at") or ""))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("\n".join(lines) + "\n")
    return history_path


def own_output_paths(runs: list[dict], registry_path: Path) -> list[Path]:
    """Every file this batch itself writes that git can see: the three
    outputs, each condition's run_history.jsonl, and the registry.
    (Manifests, per-run logs and resume sidecars are git-ignored.)"""
    paths = [registry_path, registry_path.with_name("calibration_run_history.jsonl")]
    for r in runs:
        paths += [r["output_path"], r["cwd"] / "logs" / "run_history.jsonl"]
    return paths


def output_question_ids(path: Path) -> list[int]:
    return [json.loads(line)["question_id"] for line in open(path) if line.strip()] if path.exists() else []


def count_records(path: Path, config_hash: str) -> int:
    if not path.exists():
        return 0
    qids = set()
    for line in open(path):
        if line.strip():
            rec = json.loads(line)
            if rec.get("config_hash") == config_hash and rec.get("llm_answer") is not None:
                qids.add(rec["question_id"])
    return len(qids)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Show the plan (items, est. cost, time) and stop.")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive y/N confirmation.")
    parser.add_argument("--registry", default=str(CALIBRATION_REGISTRY_PATH))
    args = parser.parse_args(argv)

    runs = calibration_runs()
    git = repo_state()
    print(format_plan(build_plan(runs), git))
    print()
    if args.dry_run:
        print("[dry-run] Stopping before any API call.")
        return 0
    if not git["git_tree_clean"]:
        print(f"[STOP] Working tree is not clean -- commit first. Dirty: {git['git_status']}")
        return 1
    if not args.yes and input("Generate the 150 calibration answers above? [y/N] ").strip().lower() != "y":
        print("Aborted -- no API call made.")
        return 1

    started_at = datetime.now(timezone.utc).isoformat()
    registry_path = Path(args.registry)
    print("[data] fingerprinting the parquet files, candidate pool and dev question IDs...", flush=True)
    data = data_fingerprint()
    print(f"[data] pool {data['n_pool_after_token_filter']} questions (sha256 {data['pool_question_ids_sha256'][:16]}...), "
          f"dev {data['n_dev']} questions (sha256 {data['dev_question_ids_sha256'][:16]}...)")
    write_registry(runs, git, registry_path, started_at, data)  # BEFORE any output exists -> guard is active from now
    print(f"[registry] calibration runs registered -> {registry_path} (judge-v2.x is now refused on them)")

    for r in runs:
        own = own_output_paths(runs, registry_path)
        before = repo_state(own)
        r["changed_paths_at_run_start"] = before["git_status"]
        r["changed_paths_not_own_output"] = [e for e in before["git_status"] if e[3:] not in before["git_exempted"]]
        if not before["git_tree_clean"]:
            write_registry(runs, git, registry_path, started_at, data)
            print(f"[STOP] before {r['run_label']}: changes outside this batch's own outputs: "
                  f"{r['changed_paths_not_own_output']}")
            return 1
        print(f"\n=== {r['run_label']} -> {repo_relative(r['output_path'])} ===", flush=True)
        proc = subprocess.run([sys.executable, r["script"], *r["args"]], cwd=r["cwd"])
        r["n_records"] = count_records(r["output_path"], r["config_hash"])
        r["question_ids_match_dev_ids_in_order"] = output_question_ids(r["output_path"]) == data["dev_question_ids"]
        write_registry(runs, git, registry_path, started_at, data)
        if proc.returncode != 0 or r["n_records"] != N_DEV or not r["question_ids_match_dev_ids_in_order"]:
            print(f"[STOP] {r['run_label']}: exit code {proc.returncode}, {r['n_records']}/{N_DEV} answers, "
                  f"question ids match the dev ids in order: {r['question_ids_match_dev_ids_in_order']}. "
                  f"Fix and re-run this command (completed answers are resumed).")
            return 1
        print(f"[ok] {r['run_label']}: {r['n_records']}/{N_DEV} answers")

    print(f"[history] calibration run history entries copied -> {write_calibration_run_history(runs)}")
    print(f"\nAll {N_DEV * len(runs)} calibration answers generated at commit {git['git_commit']}. No judge was run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
