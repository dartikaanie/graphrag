"""
run_registry.py
===============
results/run_registry.json: one entry per generation / retrieval / judge run,
with a status that decides where it may be read:

    "official"    official evaluation runs (Amendment 14); the default view of
                  the dashboard and of every summary script
    "dev-stage2"  alpha stage-2 runs on the dev split (stage2_report.py)
    "archived"    pre-official data (archive of 2026-10-19); reachable only
                  through an explicit "show archive" / --include-archived

Entry fields: run_id, status, kind, condition, grounding, alpha, n, seed,
generator, judge_config, commit, created_at, files (role -> ORIGINAL repo-
relative path; open through llm/archive_paths.resolve), cost_usd, cost_source,
notes. The top-level "budget" and "deepinfra_spend" ledger are used by the
judging budget guard.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from llm import archive_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "results" / "run_registry.json"

STATUS_OFFICIAL, STATUS_DEV_STAGE2, STATUS_ARCHIVED = "official", "dev-stage2", "archived"
STATUSES = (STATUS_OFFICIAL, STATUS_DEV_STAGE2, STATUS_ARCHIVED)
DEFAULT_VIEW = (STATUS_OFFICIAL,)
ENTRY_FIELDS = ("run_id", "status", "kind", "condition", "grounding", "alpha", "n", "seed", "generator",
                "judge_config", "commit", "created_at", "files", "cost_usd", "cost_source", "notes")


class RegistryError(RuntimeError):
    pass


def empty_registry() -> dict:
    return {"schema_version": 1, "statuses": list(STATUSES),
            "budget": {"deepinfra_budget_usd": None, "set_by": None, "set_at": None},
            "deepinfra_spend": [], "runs": []}


def load(path: Path | None = None) -> dict:
    path = Path(path or REGISTRY_PATH)
    if not path.exists():
        return empty_registry()
    return json.loads(path.read_text())


def save(reg: dict, path: Path | None = None) -> Path:
    path = Path(path or REGISTRY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def validate_entry(entry: dict) -> None:
    missing = [f for f in ENTRY_FIELDS if f not in entry]
    if missing:
        raise RegistryError(f"registry entry {entry.get('run_id')!r} lacks fields {missing}")
    if entry["status"] not in STATUSES:
        raise RegistryError(f"status {entry['status']!r} not in {STATUSES}")
    if not isinstance(entry["files"], dict) or not entry["files"]:
        raise RegistryError(f"registry entry {entry['run_id']!r} has no files")


def register(entry: dict, path: Path | None = None) -> dict:
    """Adds one entry (run_id must be new). Returns the saved registry."""
    validate_entry(entry)
    reg = load(path)
    if any(r["run_id"] == entry["run_id"] for r in reg["runs"]):
        raise RegistryError(f"run_id {entry['run_id']!r} is already registered")
    reg["runs"].append(entry)
    save(reg, path)
    return reg


def runs(statuses=DEFAULT_VIEW, include_archived: bool = False, reg: dict | None = None) -> list[dict]:
    """Entries in `statuses` (default: official only); include_archived adds
    the archived ones -- the explicit "show archive" toggle."""
    reg = reg if reg is not None else load()
    wanted = set(statuses) | ({STATUS_ARCHIVED} if include_archived else set())
    return [r for r in reg["runs"] if r["status"] in wanted]


def _rel(p) -> str | None:
    rel = archive_paths.repo_rel(archive_paths.canonical(p))
    return rel


def find_by_path(p, reg: dict | None = None) -> dict | None:
    """The entry that lists file `p` (original or archived path) among its files."""
    rel = _rel(p)
    reg = reg if reg is not None else load()
    for r in reg["runs"]:
        if rel in (r.get("files") or {}).values():
            return r
    return None


def assert_reportable(paths, allowed=DEFAULT_VIEW, include_archived: bool = False, reg: dict | None = None) -> list[dict]:
    """Summary-script guard: every input file must belong to a registered run
    whose status is in `allowed` (archived runs only with include_archived).
    Raises RegistryError naming each refused file."""
    reg = reg if reg is not None else load()
    ok = set(allowed) | ({STATUS_ARCHIVED} if include_archived else set())
    found, problems = [], []
    for p in paths:
        if p is None:
            continue
        r = find_by_path(p, reg)
        if r is None:
            problems.append(f"{p}: not in {REGISTRY_PATH.name}")
        elif r["status"] not in ok:
            problems.append(f"{p}: status {r['status']!r} (allowed: {sorted(ok)})")
        else:
            found.append(r)
    if problems:
        raise RegistryError("refusing non-official input (pass --include-archived to read archived runs "
                            "explicitly): " + "; ".join(problems))
    return found


def add_cli_flag(parser) -> None:
    parser.add_argument("--include-archived", action="store_true",
                        help="Also accept archived (pre-official) runs as input; official runs only by default")


def guard_cli(paths, include_archived: bool, allowed=DEFAULT_VIEW, log=print) -> list[dict]:
    """assert_reportable for a summary script's CLI: prints the refusal and exits with status 2."""
    try:
        return assert_reportable([p for p in paths if p], allowed, include_archived)
    except RegistryError as e:
        log(f"[REFUSED] {e}")
        raise SystemExit(2)


def deepinfra_spent(reg: dict | None = None) -> float:
    reg = reg if reg is not None else load()
    return round(sum(float(e.get("cost_usd") or 0) for e in reg.get("deepinfra_spend") or []), 6)


def deepinfra_remaining(reg: dict | None = None) -> float | None:
    reg = reg if reg is not None else load()
    budget = (reg.get("budget") or {}).get("deepinfra_budget_usd")
    return None if budget is None else round(float(budget) - deepinfra_spent(reg), 6)


def record_deepinfra_spend(label: str, cost_usd: float, run_id: str | None = None, path: Path | None = None) -> dict:
    reg = load(path)
    reg.setdefault("deepinfra_spend", []).append({
        "label": label, "run_id": run_id, "cost_usd": round(float(cost_usd), 6),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    save(reg, path)
    return reg


# ---------------------------------------------------------------------------
# Registration of new runs (Amendment 14 draft, plan step 7a)
# ---------------------------------------------------------------------------

FROZEN_PARAMS_JSON = REPO_ROOT / "results" / "frozen_parameters.json"
GEN_PRICE_PER_M = (0.15, 0.60)  # gpt-4o-mini input / output, USD


def frozen_params_hash() -> str:
    """Hash of the frozen parameter table (llm/frozen_parameters.py) the run is checked against."""
    return json.loads(FROZEN_PARAMS_JSON.read_text())["table_hash"]


def expected_generation_config(status: str, condition: str) -> dict:
    """The frozen values a generation run's config must carry (official_run_matrix + Amendments 8/9)."""
    from llm import official_run_matrix as orm
    exp = {"provider": orm.PROVIDER, "model": orm.MODEL, "seed": orm.SEED, "oversample_pool": orm.OVERSAMPLE_POOL,
           "sample_split": "test" if status == STATUS_OFFICIAL else "dev",
           "n_sample": orm.N_SAMPLE if status == STATUS_OFFICIAL else 50}
    if condition in ("B", "C", "D"):
        exp.update(top_k=orm.TOP_K, leakage=orm.LEAKAGE_MODE)
    if status == STATUS_OFFICIAL and condition in ("B", "C", "D"):
        exp.update(require_grounding=True)  # Amendment 16: the four official runs are grounded; plain cells are legacy
    if condition == "B" and status == STATUS_OFFICIAL:
        exp.update(require_citation=True)
    if condition == "C":
        exp.update(c_retrieval_version=orm.C_RETRIEVAL_VERSION, n_anchor=orm.N_ANCHOR, n_semantic_expansion=orm.N_SEMANTIC_EXPANSION,
                   expansion_require_answer=True, fusion_mode="trust_weighted")
    if condition == "D":
        exp.update(d_retrieval_version=orm.D_RETRIEVAL_VERSION, n_low_level=orm.N_LOW_LEVEL,
                   n_high_level=orm.N_HIGH_LEVEL)
    return exp


def generation_entry(run_file, status: str) -> dict:
    """A registry entry for a generation run, after checking its config against the frozen values."""
    if status not in (STATUS_OFFICIAL, STATUS_DEV_STAGE2):
        raise RegistryError(f"new runs are registered as {STATUS_OFFICIAL!r} or {STATUS_DEV_STAGE2!r}")
    path = archive_paths.resolve(run_file)
    recs = [json.loads(line) for line in open(path) if line.strip()]
    if not recs:
        raise RegistryError(f"{path}: empty run file")
    cfg = recs[0].get("config") or {}
    if len({json.dumps(r.get("config"), sort_keys=True) for r in recs}) != 1:
        raise RegistryError(f"{path.name}: records carry more than one config")
    cond = cfg.get("condition")
    exp = expected_generation_config(status, cond)
    defaults = {"sample_split": "test"}  # keys left out of a config when at their default (config_hash stability)
    wrong = {k: (cfg.get(k, defaults.get(k)), v) for k, v in exp.items() if cfg.get(k, defaults.get(k)) != v}
    if cond == "B" and not str(cfg.get("corpus", "")).startswith("kg_answers"):
        wrong["corpus"] = (cfg.get("corpus"), "kg_answers_*")
    if status == STATUS_OFFICIAL and cond == "C":  # the one alpha of the official runs (Amendment 16)
        from llm.c_graphrag.official_params import load_official_params
        official = load_official_params() or {}
        if official.get("status") != "selected" or cfg.get("alpha") != official.get("alpha_primary"):
            wrong["alpha"] = (cfg.get("alpha"), f"alpha_primary of a 'selected' official_params.json "
                                                f"({official.get('status')}, {official.get('alpha_primary')})")
    if status == STATUS_DEV_STAGE2:
        from llm.c_graphrag.stage2_candidates import EXPLORATORY_ALPHAS, STAGE2_ALPHAS
        allowed = (*STAGE2_ALPHAS, *EXPLORATORY_ALPHAS)
        if cond != "C" or cfg.get("alpha") not in allowed:
            wrong["condition/alpha"] = ((cond, cfg.get("alpha")), f"C with alpha in {list(allowed)} (stage-2 candidates "
                                        f"{list(STAGE2_ALPHAS)} + exploratory {list(EXPLORATORY_ALPHAS)})")
    if wrong:
        raise RegistryError(f"{path.name}: config differs from the frozen values (got, expected): {wrong}")
    no_timing = sum(1 for r in recs if r.get("end_to_end_latency_sec") is None or r.get("answer_started_at_utc") is None)
    if no_timing:
        raise RegistryError(f"{path.name}: {no_timing} records without per-answer end-to-end timing (record schema "
                            "r3; NF3 cannot be computed) -- generate with the current code")
    if len(recs) != exp["n_sample"]:
        raise RegistryError(f"{path.name}: {len(recs)} records, expected {exp['n_sample']}")
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    tin = sum((r.get("usage_full") or {}).get("prompt_tokens", 0) for r in recs)
    tout = sum((r.get("usage_full") or {}).get("completion_tokens", 0) for r in recs)
    rel = archive_paths.repo_rel(archive_paths.canonical(path))
    files = {"output": rel}
    if manifest_path.exists():
        files["manifest"] = rel + ".manifest.json"
    return {
        "run_id": f"{path.stem}", "status": status, "kind": "generation", "condition": cond,
        "grounding": cfg.get("require_grounding"), "alpha": cfg.get("alpha"), "n": len(recs), "seed": cfg.get("seed"),
        "generator": cfg.get("model"), "judge_config": None,
        "commit": ((manifest.get("git") or {}).get("commit_sha") or "")[:7] or None,
        "created_at": manifest.get("started_at_utc"), "files": files,
        "cost_usd": round((tin * GEN_PRICE_PER_M[0] + tout * GEN_PRICE_PER_M[1]) / 1e6, 6),
        "cost_source": f"usage_full ({tin} in, {tout} out) at gpt-4o-mini list price",
        "notes": "", "config_hash": recs[0].get("config_hash"), "frozen_params_hash": frozen_params_hash(),
        "git_dirty": (manifest.get("git") or {}).get("dirty"),
    }


def set_budget(usd: float, who: str = "author", path: Path | None = None) -> dict:
    reg = load(path)
    reg["budget"] = {"deepinfra_budget_usd": float(usd), "set_by": who,
                     "set_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    save(reg, path)
    return reg


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="results/run_registry.json: budget, registration, overview")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("set-budget", help="Set the DeepInfra budget (USD) the judging guard enforces")
    b.add_argument("usd", type=float)
    r = sub.add_parser("register", help="Register generation run file(s) after checking them against the frozen values")
    r.add_argument("--status", choices=[STATUS_OFFICIAL, STATUS_DEV_STAGE2], required=True)
    r.add_argument("run_files", nargs="+")
    sub.add_parser("show", help="Counts, budget and spend")
    args = ap.parse_args(argv)
    if args.cmd == "set-budget":
        reg = set_budget(args.usd)
        print(f"DeepInfra budget ${args.usd:.2f}; spent ${deepinfra_spent(reg):.4f}; remaining ${deepinfra_remaining(reg):.4f}")
    elif args.cmd == "register":
        for f in args.run_files:
            try:
                e = generation_entry(f, args.status)
                register(e)
                print(f"[registered] {e['run_id']} ({e['status']}, {e['condition']}, grounding {e['grounding']}, "
                      f"alpha {e['alpha']}, n {e['n']}, ${e['cost_usd']:.4f})")
            except RegistryError as err:
                print(f"[REFUSED] {err}")
                return 2
    else:
        reg = load()
        print({s: sum(1 for x in reg["runs"] if x["status"] == s) for s in STATUSES}, "budget", reg.get("budget"),
              "spent", deepinfra_spent(reg), "remaining", deepinfra_remaining(reg))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
