"""
build_archive_index.py
======================
Writes, for the pre-official archive of 2026-10-19 (archive_map.json):
  results/_archive/<tag>/ARCHIVE_INDEX.md   what each file is, original path,
                                            producing commit, citing documents
  results/run_registry.json                 one "archived" entry per archived run
                                            output (entries of other statuses are kept)
Read-only on the archived files; no API call. Re-runnable: archived entries
are rebuilt, official / dev-stage2 entries and the spend ledger are kept.

    python3 llm/build_archive_index.py [--tag 2026-10-19_pre-official]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from llm import run_registry  # noqa: E402

TAG_DEFAULT = "2026-10-19_pre-official"
GPT4OMINI_IN, GPT4OMINI_CACHED_IN, GPT4OMINI_OUT = 0.15, 0.075, 0.60  # USD per 1M tokens (list price)
CITING_DOCS = sorted([*REPO_ROOT.glob("docs/*.md"), *REPO_ROOT.glob("reports/*.md")])

# (regex on the original path, group, description) -- first match wins
GROUPS = [
    (r"/results_old(_2)?/", "G1 legacy results_old", "Legacy pilot outputs (Ollama phi3/qwen2.5 and early gpt-4o-mini, n1-n100); superseded before prompt v3"),
    (r"^results/leakage/audit/", "G11a leakage audit", "Leakage audit retrieval dumps and summaries (Amendment 8; reports/leakage_audit.md)"),
    (r"/logs/run_history", "G9 run histories", "Per-condition / per-judge run history (one JSON line per run)"),
    (r"^logs/", "G10 root audit logs", "Dashboard audit records: invalid / superseded runs and outputs"),
    (r"/logs/dashboard_", "G9 dashboard logs", "Per-dashboard-job log file"),
    (r"/logs/panel_calibration", "G8 panel console logs", "Console log of a judge-panel calibration run"),
    (r"/logs/", "G9 run logs", "Per-run generation log"),
    (r"_n10_seed42", "G2 n10 pilots", "n=10 pilot run on the test sample (2026-10-05 pilot; prompt v1-v3)"),
    (r"_dev_v3_", "G3 dev runs", "Dev-split run (positions 385+): calibration generation (A/B n50), B/C/D n35 parity runs, or C retrieval-only stage-1 runs (n50)"),
    (r"/controls/|ctxrel_v2_stage1|stage1_report_", "G4 C ctxrel controls/sweep", "Context-relevance control runs (Amendments 2-3) and the stage-1 sweep (ctxrel-v2)"),
    (r"/expansion_check/", "G5 C expansion check", "Stage-1 recheck with answerable-only expansion (Amendment 9); stage-1 result of record, top_two = [0.25, 0.0]"),
    (r"/logscore_check/", "G5 C logscore check", "Stage-1 recheck with trustScore v3-logscore (Amendment 6)"),
    (r"/reputation_check/", "G5 C reputation check", "Reputation ablation and stage-1 recheck (Amendment 5)"),
    (r"interjudge_|review_hard_cases_", "G7 inter-judge diagnostics", "Inter-judge calibration diagnostics (judge-v2.2 / v2.3): reports, category-C patterns, hard-case review sheets"),
    (r"panel_", "G7 panel calibration", "Judge-panel (panel-v1) calibration / smoke runs and agreement reports vs the frozen v1 labels"),
    (r"jv1_|judged__", "G6 judge-v1", "judge-v1 hallucination judging of the n10 pilots (dashboard)"),
    (r"jv2_", "G6 judge-v2", "judge-v2 hallucination judging of the n10 pilots, with reparsed versions (pv2)"),
    (r"ctxrel", "G6 ctxrel-v1", "Context-relevance judge v1 runs and smoke test"),
    (r"ansrel", "G6 answer relevance", "Answer-relevance smoke test (n1)"),
]


def group_of(rel: str) -> tuple[str, str]:
    for pattern, group, desc in GROUPS:
        if re.search(pattern, rel):
            return group, desc
    return "G0 other", "Other pre-official file"


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


_COMMITS: dict[str, str] = {}


def producing_commit(orig: str, manifest: dict | None) -> str:
    """The commit that added the file at its original path (git history keeps it after the
    archive rename), else the manifest commit of an untracked output."""
    if orig not in _COMMITS:
        _COMMITS[orig] = git("log", "--diff-filter=A", "--format=%h", "-1", "--", orig)
    if _COMMITS[orig]:
        return _COMMITS[orig]
    sha = ((manifest or {}).get("git") or {}).get("commit_sha")
    return f"untracked; manifest commit {sha[:7]}" if sha else "untracked (git-ignored)"


def doc_index() -> dict[Path, list[str]]:
    return {d: d.read_text(errors="ignore").splitlines() for d in CITING_DOCS}


def citations(name: str, docs: dict[Path, list[str]]) -> list[str]:
    """'<doc> (<nearest Amendment heading or section>)' for every document mentioning `name`."""
    out = []
    for doc, lines in docs.items():
        sections = []
        heading = None
        for line in lines:
            if line.startswith("#"):
                m = re.search(r"Amendment\s+(\d+)", line)
                heading = f"Amendment {m.group(1)}" if m else line.lstrip("#").strip()[:60]
            if name in line:
                sections.append(heading or "top")
        if sections:
            uniq = list(dict.fromkeys(sections))
            out.append(f"{doc.relative_to(REPO_ROOT).as_posix()} ({'; '.join(uniq[:4])})")
    return out


def load_histories(archive_logs: Path) -> list[dict]:
    rows = []
    for p in sorted(archive_logs.rglob("*run_history*.jsonl")):
        for line in p.read_text(errors="ignore").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r["_history_file"] = p.name
            rows.append(r)
    return rows


def history_for(orig: str, histories: list[dict]) -> dict | None:
    name = Path(orig).name
    hits = [h for h in histories if Path(str(h.get("output_path") or h.get("out") or "")).name == name]
    return max(hits, key=lambda h: h.get("run_started_at") or "") if hits else None


def first_record(path: Path) -> dict:
    try:
        with open(path) as f:
            line = f.readline()
        return json.loads(line) if line.strip() else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def generation_tokens(path: Path) -> tuple[int, int, int, int]:
    n = tin = tcached = tout = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = r.get("usage_full") or {}
            n += 1
            tin += int(u.get("prompt_tokens") or 0)
            tcached += int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
            tout += int(u.get("completion_tokens") or 0)
    return n, tin, tcached, tout


SIDE_SUFFIXES = (".manifest.json", ".resume.json", "_failures.jsonl", "_failures_resolved.jsonl", "_resolved.jsonl",
                 "_panel.jsonl")


def is_primary_output(orig: str) -> bool:
    return orig.endswith(".jsonl") and not any(orig.endswith(s) for s in SIDE_SUFFIXES) \
        and "/logs/" not in orig and not orig.startswith("logs/")


def kind_of(orig: str, rec: dict) -> str:
    name = Path(orig).name
    if orig.startswith("results/leakage/"):
        return "retrieval-only"
    if name.startswith(("condition_", "dev_ollama_", "write_test")):
        return "generation" if "llm_answer" in rec else "retrieval-only"
    if name.startswith("stage1_"):
        return "retrieval-only"
    return "judge"


def build_entry(orig: str, new: str, files_map: dict, histories: list[dict]) -> dict:
    path = REPO_ROOT / new
    rec = first_record(path)
    cfg = rec.get("config") or {}
    kind = kind_of(orig, rec)
    manifest_orig = orig + ".manifest.json"
    manifest = json.loads((REPO_ROOT / files_map[manifest_orig]).read_text()) if manifest_orig in files_map else None
    hist = history_for(orig, histories)
    files = {"output": orig}
    for suffix, role in ((".manifest.json", "manifest"), (".resume.json", "resume")):
        if orig + suffix in files_map:
            files[role] = orig + suffix
    stem = orig[: -len(".jsonl")]
    for suffix, role in (("_failures.jsonl", "failures"), ("_failures_resolved.jsonl", "failures_resolved"),
                         ("_resolved.jsonl", "resolved"), ("_panel.jsonl", "panel")):
        if stem + suffix in files_map:
            files[role] = stem + suffix

    grounding = cfg.get("require_grounding", (hist or {}).get("require_grounding"))
    m_alpha = re.search(r"alpha(\d)-(\d+)", orig)
    alpha = cfg.get("alpha", (hist or {}).get("alpha"))
    if alpha is None and m_alpha and cfg.get("condition", "C") == "C":
        alpha = float(f"{m_alpha.group(1)}.{m_alpha.group(2)}")
    cost, cost_source, judge_config = None, None, None
    if kind == "generation":
        n, tin, tcached, tout = generation_tokens(path)
        if tin or tout:
            cost = round(((tin - tcached) * GPT4OMINI_IN + tcached * GPT4OMINI_CACHED_IN + tout * GPT4OMINI_OUT) / 1e6, 6)
            cost_source = (f"estimate from usage_full ({tin} in incl. {tcached} cached, {tout} out) "
                           f"at gpt-4o-mini list price")
    elif kind == "judge" and hist:
        if hist.get("actual_cost_usd") is not None:
            cost, cost_source = hist["actual_cost_usd"], f"actual_cost_usd in {hist['_history_file']}"
        judge_config = {k: hist.get(k) for k in ("prompt_version", "judge", "panel", "source") if hist.get(k) is not None}
        if isinstance(judge_config.get("panel"), dict):
            p = judge_config["panel"]
            judge_config["panel"] = {k: p.get(k) for k in ("panel_id", "prompt_version", "parse_version", "temperature",
                                                           "max_tokens", "response_format")}
            judge_config["panel"]["judges"] = [j.get("model") for j in hist["panel"].get("judges") or []]
    created = ((manifest or {}).get("started_at_utc") or (hist or {}).get("run_started_at") or rec.get("timestamp_utc"))
    model = cfg.get("model") or (hist or {}).get("model") or rec.get("llm_model")
    n = cfg.get("n_sample") or (hist or {}).get("n_sample_target") or (hist or {}).get("n_items")
    return {
        "run_id": f"{Path(orig).stem}@{hashlib.sha1(orig.encode()).hexdigest()[:6]}",
        "status": run_registry.STATUS_ARCHIVED,
        "kind": kind,
        "condition": cfg.get("condition") or (hist or {}).get("condition") or _condition_from_name(orig),
        "grounding": grounding,
        "alpha": alpha,
        "n": n,
        "seed": cfg.get("seed", (hist or {}).get("seed")),
        "generator": model if kind == "generation" else None,
        "judge_config": judge_config,
        "commit": producing_commit(orig, manifest),
        "created_at": created,
        "files": files,
        "cost_usd": cost,
        "cost_source": cost_source,
        "notes": group_of(orig)[0],
    }


def _condition_from_name(orig: str) -> str | None:
    m = re.search(r"condition_([abcd])_", Path(orig).name)
    if m:
        return m.group(1).upper()
    m = re.search(r"llm/(a|b|c|d)_", orig)
    return m.group(1).upper() if m else None


T7 = [
    ("/Volumes/T7 Shield/S2/THESIS GRAPHRAG/project/graphrag_old/00_datasource/archive/",
     "archive_results.py copy of 2026-10-05 (122 files + SHA256SUMS, 36 MB); older copy of llm/*/results and logs. Not touched."),
    ("/Volumes/T7 Shield/S2/THESIS GRAPHRAG/project/graphrag/backend/.runs/",
     "older dashboard job state (3.1 MB). Not touched."),
    ("/Volumes/T7 Shield/S2/THESIS GRAPHRAG/project/graphrag_old/00_datasource/merged/questions_raw_union.parquet",
     "source data (input of every run; 3.3 GB). Stays on T7."),
    ("/Volumes/T7 Shield/S2/THESIS GRAPHRAG/project/graphrag_old/00_datasource/merged/answers_raw_union.parquet",
     "source data (input of every run; 2.7 GB). Stays on T7."),
]

KEPT_IN_PLACE = [
    ("results/leakage/exclusion_sets.parquet, results/leakage/exclusion_sets.meta.json", "input of the official runs (leakage exclusion, Amendment 8)"),
    ("llm/evaluation/annotation/calibration/*", "annotation sheets, freeze / completion records, adjudicated reference -- frozen, unchanged"),
    ("llm/evaluation/annotation/calibration/annotator1_v2.xlsx, annotator1_v2_export.json",
     "v2 sheet exported on 2026-10-19, not filled in, not used (annotation v2 paused; Amendment 14 reactivation plan)"),
    ("llm/evaluation/calibration/calibration_runs.json, calibration_run_history.jsonl", "calibration run registry (frozen; original paths resolve through archive_map.json)"),
    ("report/, reports/", "EDA reports and amendment drafts (documents; cited paths updated)"),
    ("logs/20261013T145605Z_validate_densification.log", "KG validation log (graph statistics read by the Process page), not run data"),
    ("01_data_cleaning/logs/", "KG build logs, not run data"),
]


def write_index(tag: str, amap: dict, entries: list[dict]) -> Path:
    base = REPO_ROOT / "results" / "_archive" / tag
    docs = doc_index()
    by_group = defaultdict(list)
    entry_by_output = {e["files"]["output"]: e for e in entries}
    manifests = {}
    for orig, new in amap["files"].items():
        if orig.endswith(".manifest.json"):
            try:
                manifests[orig[: -len(".manifest.json")]] = json.loads((REPO_ROOT / new).read_text())
            except (OSError, json.JSONDecodeError):
                pass
    for orig, new in sorted(amap["files"].items()):
        by_group[group_of(orig)[0]].append((orig, new))

    lines = [
        f"# Archive index: {tag}",
        "",
        "All pre-official result, log and diagnostic files were moved here on 2026-10-19 (Amendment 14, archive note), "
        "so that the dashboard and the summary scripts show official runs only. Nothing was deleted or edited: "
        "every file keeps its original repo-relative path below the archive root, and its SHA-256 was verified "
        "after the move.",
        "",
        f"- Run outputs, reports, job state: `results/_archive/{tag}/<original path>`",
        f"- Run histories and run logs: `logs/_archive/{tag}/<original path>`",
        f"- Map (original -> archived): `results/_archive/{tag}/archive_map.json`, read by `llm/archive_paths.py` "
        "(`resolve()` opens a moved file by its original path; `canonical()` maps an archived path back). "
        "Frozen records (calibration_runs.json, freeze/export JSON, run histories) keep the original paths byte-for-byte.",
        "- Run registry: `results/run_registry.json`, every archived run output registered with status `archived`.",
        "- Git: tracked files moved with `git mv`; git-ignored files moved as files and stay ignored (.gitignore rules "
        "under `results/_archive/*` and `logs/_archive/*`).",
        "",
        f"Files: {len(amap['files'])} (+ directory `backend/.runs/`, {sum(1 for _ in (REPO_ROOT / amap['dirs']['backend/.runs']).rglob('*') if _.is_file())} dashboard job-state files).",
        "",
        "## Stage-2 inputs (2.0 check)",
        "",
        "No archived run qualifies as an alpha stage-2 input. The Condition C dev runs with alpha 0.25 or 0 are all "
        "retrieval-only (no generated answer, no hallucination judging); stage 2 needs generated answers for "
        "alpha in {0.25, 0} x grounding on/off on the frozen dev list, started after TRUST_V3_APPLIED_AT "
        "with trustScore v3-logscore, answerable-only expansion and leakage mode dup+neardup0.95. "
        "All four stage-2 runs must therefore be generated. Table:",
        "",
        "| File | Date (UTC) | Kind | trustScore | Expansion | Leakage | Judged (hallucination) |",
        "|---|---|---|---|---|---|---|",
        "| `condition_c_..._fw0-7-0-3_v3-alpha0-25_dev_v3_a60969e85a.jsonl` | 2026-10-07 | retrieval-only | v2 (before Amendment 6, 2026-10-14) | original (pre-Amendment 9) | none | no (ctxrel-v2 stage-1 only) |",
        "| `condition_c_..._uniform_v3-alpha0-0_dev_v3_0a4af95a30.jsonl` | 2026-10-07 | retrieval-only | v2 | original | none | no (ctxrel-v2 stage-1 only) |",
        "| `reputation_check/stage1_repfix_alpha0-{0,25}.jsonl` | 2026-10-10 | retrieval-only | v2 + reputation (Amendment 5) | original | none | no |",
        "| `logscore_check/stage1_logscore_alpha0-{0,25}.jsonl` | 2026-10-14 | retrieval-only | v3-logscore (Amendment 6) | original | none | no |",
        "| `expansion_check/stage1_expansion_old_alpha0-{0,25}.jsonl` | 2026-10-18 | retrieval-only | v3-logscore | original | dup+neardup0.95 | no |",
        "| `expansion_check/stage1_expansion_new_alpha0-{0,25}.jsonl` | 2026-10-18 | retrieval-only | v3-logscore | answerable-only (Amendment 9) | dup+neardup0.95 | no; stage-1 result of record |",
        "",
        "Grounding does not apply to retrieval-only runs (it is a generation-prompt setting).",
        "",
        "## Kept in place (not archived)",
        "",
        *[f"- `{p}`: {why}" for p, why in KEPT_IN_PLACE],
        "",
        "## Older copies on the T7 drive (recorded, not touched)",
        "",
        *[f"- `{p}`: {why}" for p, why in T7],
        "",
        "## Files by group",
        "",
        "Columns: original path (archived under the same path below the archive root); producing commit "
        "(the commit that added a tracked file, or the manifest commit of an untracked one); documents citing "
        "the file by name (nearest Amendment / section heading).",
    ]
    for group in sorted(by_group):
        desc = next(d for _p, g, d in GROUPS if g == group) if group != "G0 other" else "Other"
        rows = by_group[group]
        lines += ["", f"### {group} ({len(rows)} files)", "", desc + ".", "",
                  "| Original path | Archived under | Commit | Cited by |", "|---|---|---|---|"]
        for orig, new in rows:
            root = new.split("/", 1)[0] + "/_archive"
            m = manifests.get(orig[: -len(".jsonl")] + ".jsonl") if orig.endswith(".jsonl") else None
            commit = (entry_by_output.get(orig) or {}).get("commit") or producing_commit(orig, m)
            cites = citations(Path(orig).name, docs)
            lines.append(f"| `{orig}` | `{root}` | {commit} | {'<br>'.join(cites) if cites else '--'} |")
    lines += ["", "### Dashboard job state", "",
              f"`backend/.runs/` -> `{amap['dirs']['backend/.runs']}/` (git-ignored). The dashboard starts with an "
              "empty job list; archived jobs open only with `include_archive=true`.", ""]
    out = base / "ARCHIVE_INDEX.md"
    out.write_text("\n".join(lines))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=TAG_DEFAULT)
    args = ap.parse_args(argv)
    base = REPO_ROOT / "results" / "_archive" / args.tag
    amap = json.loads((base / "archive_map.json").read_text())
    histories = load_histories(REPO_ROOT / "logs" / "_archive" / args.tag)
    entries = [build_entry(o, n, amap["files"], histories)
               for o, n in sorted(amap["files"].items()) if is_primary_output(o)]
    reg = run_registry.load()
    reg["runs"] = [r for r in reg["runs"] if r["status"] != run_registry.STATUS_ARCHIVED] + entries
    for e in entries:
        run_registry.validate_entry(e)
    run_registry.save(reg)
    idx = write_index(args.tag, amap, entries)
    print(f"[registry] {len(entries)} archived entries -> {run_registry.REGISTRY_PATH.relative_to(REPO_ROOT)}")
    print(f"[index] {idx.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
