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
