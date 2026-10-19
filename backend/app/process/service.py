"""
Assembles the Process page payloads from the registry, the decision record,
logs, reports and manifests. Read-only throughout; every value carries the
file (and line, where it applies) it was read from, and every payload an
"as_of" timestamp for its sources.
"""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timezone
from fnmatch import fnmatch

from app.process import guard, live, logs, runs
from app.process.decisions import DECISION_DOC, parse_decisions
from app.process.registry import PARAMETERS, STEPS, Param, Quote
from app.process.sources import (REPO_ROOT, SourceError, file_stamp, last_change, newest_first, quote, read_text, repo_path,
                                 resolve, stamp_to_iso)

KG_LOGS = "01_data_cleaning/logs"
LOGSCORE_DIR = "llm/c_graphrag/results/logscore_check"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mtime_iso(rel: str) -> str | None:
    p = repo_path(rel)
    if not p.exists():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()


def _quote_or_error(q: Quote | None) -> dict | None:
    if q is None:
        return None
    try:
        return quote(q.file, q.needle)
    except SourceError as exc:
        return {"text": None, "file": q.file, "line": None, "error": str(exc)}


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _decisions_text() -> str:
    return read_text(DECISION_DOC)


def _judge_frozen() -> bool:
    return guard.calibration_status(_decisions_text())["judge_frozen"]


def _resolve_status(p: Param, judge_frozen: bool) -> str:
    if p.status == "judge":
        return "frozen" if judge_frozen else "pending"
    return p.status


# ---------------------------------------------------------------- parameters

def parameter_row(p: Param, judge_frozen: bool, with_git: bool = True) -> dict:
    row = {
        "key": p.key, "name": p.name, "step": p.step, "unit": p.unit, "kind": p.kind,
        "used_by": list(p.used_by), "set_by": p.set_by, "status": _resolve_status(p, judge_frozen),
        "note": p.note, "value": None, "defined_in": None, "source_note": None, "error": None,
        "last_changed": None,
    }
    try:
        r = resolve(p.source)
        row["value"] = _jsonable(r.value)
        row["defined_in"] = {"file": r.file, "line": r.line}
        row["source_note"] = r.note
        if with_git:
            row["last_changed"] = last_change(r.file, r.line)
    except SourceError as exc:
        row["error"] = str(exc)
        row["defined_in"] = {"file": p.source.file, "line": None}
    row["rationale"] = _quote_or_error(p.rationale)
    row["rationale_documented"] = p.rationale is not None
    row["spec"] = _quote_or_error(p.spec)
    return row


def parameters() -> dict:
    frozen = _judge_frozen()
    rows = [parameter_row(p, frozen) for p in PARAMETERS]
    return {
        "as_of": _now(),
        "parameters": rows,
        "undocumented": [r["key"] for r in rows if r["kind"] == "parameter" and not r["rationale_documented"]],
    }


# ---------------------------------------------------------------- log helpers

def _kg_logs(glob: str) -> list[str]:
    directory = repo_path(KG_LOGS)
    if not directory.is_dir():
        return []
    return [f"{KG_LOGS}/{p.name}" for p in newest_first([p for p in directory.iterdir() if fnmatch(p.name, glob)])]


def _first_line(text: str, pattern: str) -> re.Match | None:
    return re.search(pattern, text, re.MULTILINE)


def _log_ref(rel: str) -> dict:
    name = rel.rsplit("/", 1)[-1]
    return {"log_id": f"kg_logs/{name}", "file": rel, "timestamp_utc": stamp_to_iso(file_stamp(name))}


def trust_history() -> list[dict]:
    """trustScoreVersion history read from the build/apply/verify logs."""
    history = []

    for rel in _kg_logs("*_build_knowledge_graph.log"):
        text = read_text(rel)
        m = _first_line(text, r"trustScore Answer\s+--\s+(min=.*)$")
        if m:
            history.append({
                "version": "(untagged, KG build)", "event": "KG build",
                "summary": f"Answer trustScore {m.group(1)}", **_log_ref(rel),
                "applied_at": stamp_to_iso(file_stamp(rel)), "backup": None, "verify": None,
            })
            break

    for rel in _kg_logs("*_apply_reputation_trust.log"):
        text = read_text(rel)
        if not _first_line(text, r"Mode: apply"):
            continue
        backup = _first_line(text, r"Backup ditulis: (\S+)")
        verify = _first_line(text, r"(Verifikasi: .*)$")
        new = _first_line(text, r"trustScore baru\s*: (.*)$")
        history.append({
            "version": "v2-reputation", "event": "Amendment 5 apply",
            "summary": f"Answer trustScore after: {new.group(1)}" if new else None, **_log_ref(rel),
            "applied_at": stamp_to_iso(file_stamp(rel)), "backup": backup.group(1) if backup else None,
            "verify": verify.group(1) if verify else None,
        })
        break

    for rel in _kg_logs("*_apply_logscore_trust.log"):
        text = read_text(rel)
        applied = _first_line(text, r"APPLIED_AT \(local\): (\S+)")
        if not applied:
            continue
        backup = _first_line(text, r"Backup written: (\S+)")
        verify = [m.group(1) for m in re.finditer(r"^\S+ \| (Verify (?:Answer|Question): .*)$", text, re.MULTILINE)]
        new = _first_line(text, r"Answer\s+new: (.*)$")
        history.append({
            "version": "v3-logscore", "event": "Amendment 6 apply",
            "summary": f"Answer trustScore after: {new.group(1)}" if new else None, **_log_ref(rel),
            "applied_at": applied.group(1), "backup": backup.group(1) if backup else None,
            "verify": "; ".join(verify) or None,
        })
        break

    return history


def latest_verify() -> dict | None:
    """Latest `--verify-applied` log: the page's default source for the
    graph's trustScoreVersion state (no Neo4j query on page load)."""
    for rel in _kg_logs("*_apply_logscore_trust.log"):
        text = read_text(rel)
        if not _first_line(text, r"Mode: verify-applied"):
            continue
        counts = {}
        for m in re.finditer(r"^\S+ \| (Answer|Question): ([\d,]+) nodes read \| trustScoreVersion: (\{.*\})$",
                             text, re.MULTILINE):
            try:
                counts[m.group(1)] = {"nodes": int(m.group(2).replace(",", "")),
                                      "versions": ast.literal_eval(m.group(3))}
            except (ValueError, SyntaxError):
                continue
        results = [m.group(1) for m in re.finditer(r"^\S+ \| (Verify-applied .*)$", text, re.MULTILINE)]
        return {**_log_ref(rel), "counts": counts, "results": results}
    return None


def stage1_result() -> dict | None:
    """Stage-1 result of record: the newest logscore stage-1 report (dev set,
    aggregates only). Read from the pre-official archive (2026-10-19) when it
    is no longer in results/; the row then carries from_archive=True."""
    from app.archive import archive_paths

    directory = repo_path(LOGSCORE_DIR)
    from_archive = not directory.is_dir()
    if from_archive:
        directory = archive_paths.archived_dir(directory)
        if directory is None or not directory.is_dir():
            return None
    reports = newest_first([p for p in directory.iterdir() if fnmatch(p.name, "stage1_logscore_report_*.json")])
    if not reports:
        return None
    rel = reports[0].relative_to(REPO_ROOT).as_posix()
    data = json.loads(read_text(rel))
    table = []
    for alpha in sorted(data.get("v3", {}), key=float):
        v2, v3 = data.get("v2", {}).get(alpha, {}), data["v3"][alpha]
        table.append({
            "alpha": float(alpha),
            "pct_questions_with_relevant": {"v2": v2.get("pct_questions_with_relevant"),
                                            "v3": v3.get("pct_questions_with_relevant")},
            "mean_relevance_score": {"v2": v2.get("mean_relevance_score"), "v3": v3.get("mean_relevance_score")},
            "pct_accepted_selected": {"v2": v2.get("pct_accepted_selected"), "v3": v3.get("pct_accepted_selected")},
            "n_questions": v3.get("n_questions"),
        })
    sel = data.get("selection_v3", {})
    meta = data.get("meta", {})
    md_name = reports[0].name.replace(".json", ".md")
    return {
        "file": rel,
        "from_archive": from_archive,
        "log_id": (f"{logs.ARCHIVE_KEY_PREFIX if from_archive else ''}logscore_check/{md_name}"
                   if (directory / md_name).exists() else None),
        "timestamp_utc": stamp_to_iso(file_stamp(reports[0].name)),
        "git_commit": (meta.get("git_commit") or "")[:7] or None,
        "judge_model": meta.get("judge_model"),
        "table": table,
        "ranked": sel.get("ranked"),
        "top_two": sel.get("top_two"),
        "keep_current": sel.get("keep_current"),
        "reasoning": sel.get("reasoning"),
    }


def _official_params() -> dict | None:
    try:
        return json.loads(read_text("llm/c_graphrag/official_params.json"))
    except (SourceError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- steps

def _step_status(key: str, ctx: dict) -> str:
    """Status rule per step, from artifacts (see each branch)."""
    if key in ("data", "sampling"):
        return "done" if ctx["build_log"] else "not started"  # KG build log proves the merged data exist
    if key == "kg_build":
        return "done" if ctx["build_log"] and ctx["densify_validated"] else "in progress"
    if key == "trust":
        v = ctx["verify"]
        ok = v and v["results"] and all(r.endswith("-> OK") for r in v["results"])
        return "done" if ok else "in progress"
    if key == "conditions":
        return "implemented"
    if key == "c_ranking":
        official = ctx["official"] or {}
        return "alpha pending (stage 2)" if official.get("alpha_primary") is None else "frozen"
    if key == "alpha_selection":
        s1 = "stage 1 done" if ctx["stage1"] else "stage 1 not done"
        return f"{s1}; stage 2: {ctx['stage2']['status']}"
    if key == "judge":
        return f"calibration: {ctx['calibration']['status']}"
    if key == "official":
        official = ctx["official"] or {}
        return "ready" if official.get("status") == "selected" else "not started"
    if key == "metrics":
        return "not started (needs the official run)"
    return "unknown"


def steps() -> dict:
    decisions_text = _decisions_text()
    calibration = guard.calibration_status(decisions_text)
    frozen = calibration["judge_frozen"]
    all_logs = logs.list_logs()
    verify = latest_verify()
    stage1 = stage1_result()
    ctx = {
        "build_log": bool(_kg_logs("*_build_knowledge_graph.log")),
        "densify_validated": any(lg["name"].endswith("_validate_densification.log") for lg in all_logs),
        "verify": verify,
        "stage1": stage1,
        "stage2": guard.stage2_status(frozen),
        "calibration": calibration,
        "official": _official_params(),
    }

    out = []
    for s in STEPS:
        step_logs = [lg for lg in all_logs if lg["step"] == s.key]
        dates = [lg["timestamp_utc"] for lg in step_logs if lg["timestamp_utc"]]
        item = {
            "key": s.key, "number": s.number, "title": s.title, "purpose": s.purpose,
            "modules": [{"file": m, "exists": repo_path(m).exists()} for m in s.modules],
            "inputs": s.inputs, "outputs": s.outputs,
            "quotes": [_quote_or_error(q) for q in s.quotes],
            "parameters": [p.key for p in PARAMETERS if p.step == s.key],
            "status": _step_status(s.key, ctx),
            "last_evidence_utc": max(dates) if dates else None,
            "logs": [lg["id"] for lg in step_logs],
        }
        if s.key == "trust":
            item["trust_history"] = trust_history()
            item["verify"] = verify
        if s.key == "alpha_selection":
            item["stage1"] = stage1
            item["stage2"] = ctx["stage2"]
        if s.key == "judge":
            item["calibration"] = calibration
        if s.key == "official":
            item["official_params"] = ctx["official"]
        out.append(item)

    return {
        "as_of": _now(),
        "sources_as_of": {
            DECISION_DOC: _mtime_iso(DECISION_DOC),
            "llm/c_graphrag/official_params.json": _mtime_iso("llm/c_graphrag/official_params.json"),
            "latest_log": max((lg["timestamp_utc"] for lg in all_logs if lg["timestamp_utc"]), default=None),
        },
        "headline": {
            "stage1": "done" if stage1 else "not done",
            "stage2": ctx["stage2"]["status"],
            "calibration": calibration["status"],
            "official_run": _step_status("official", ctx),
            "trust_score_version": (verify or {}).get("counts", {}).get("Answer", {}).get("versions"),
        },
        "steps": out,
    }


# ---------------------------------------------------------------- other endpoints

def decisions() -> dict:
    return {"as_of": _mtime_iso(DECISION_DOC), "source": DECISION_DOC, "decisions": parse_decisions()}


def run_list(include_archive: bool = False) -> dict:
    rows = runs.list_runs(include_archive)
    return {"as_of": _now(), "dirs": [d for _, d in runs.RUN_DIRS], "include_archive": include_archive, "runs": rows}


def live_values() -> dict:
    return {"default": latest_verify(), "live": live.get_cached(), "queries": live.QUERIES}


def log_list(include_archive: bool = False) -> dict:
    return {"as_of": _now(), "roots": [r.rel_dir for r in logs.LOG_ROOTS], "include_archive": include_archive,
            "logs": logs.list_logs(include_archive)}

