"""
Run & config provenance: one metadata row per run manifest
(`*.manifest.json`) in the four condition results directories, read
non-recursively (results_old*/ and subfolders are never scanned). Rows carry
metadata only -- never output content -- and pass guard.check_run_metadata.
"""

from __future__ import annotations

import json

from app.process import guard
from app.process.sources import REPO_ROOT

RUN_DIRS = (
    ("A", "llm/a_pure_llm/results"),
    ("B", "llm/b_rag/results"),
    ("C", "llm/c_graphrag/results"),
    ("D", "llm/d_lightrag/results"),
)
MANIFEST_SUFFIX = ".manifest.json"
MAX_MANIFEST_BYTES = 1024 * 1024


def _row(condition_dir: str, name: str, m: dict) -> dict:
    cfg = m.get("config") if isinstance(m.get("config"), dict) else {}
    git = m.get("git") if isinstance(m.get("git"), dict) else {}
    graph = m.get("graph") if isinstance(m.get("graph"), dict) else {}
    counts = m.get("item_counts") if isinstance(m.get("item_counts"), dict) else {}
    run_file = name[: -len(MANIFEST_SUFFIX)]
    prompt_version = m.get("prompt_version")
    if prompt_version == "retrieval-only":
        kind = "retrieval-only"
    elif run_file.startswith("condition_"):
        kind = "generation"
    else:
        kind = "judge"
    sha = git.get("commit_sha") or ""
    return guard.check_run_metadata({
        "run_id": m.get("config_hash") or run_file,
        "file": run_file,
        "condition": cfg.get("condition") or condition_dir,
        "split": cfg.get("sample_split", "test") if kind != "judge" else cfg.get("sample_split"),
        "alpha": cfg.get("alpha"),
        "grounding": cfg.get("require_grounding"),
        "fusion_mode": cfg.get("fusion_mode"),
        "c_retrieval_version": cfg.get("c_retrieval_version"),
        "trust_score_version": graph.get("trustScoreVersion"),
        "config_hash": m.get("config_hash"),
        "prompt_version": prompt_version,
        "judge_version": m.get("judge_version"),
        "model": cfg.get("model"),
        "n_sample": cfg.get("n_sample"),
        "status": m.get("status"),
        "started_at_utc": m.get("started_at_utc"),
        "finished_at_utc": m.get("finished_at_utc"),
        "commit": sha[:7] or None,
        "git_dirty": git.get("dirty"),
        "items_succeeded": counts.get("succeeded"),
        "run_kind": kind,
    })


def list_runs() -> list[dict]:
    rows = []
    for condition, rel in RUN_DIRS:
        directory = REPO_ROOT / rel
        if not directory.is_dir():
            continue
        for p in directory.iterdir():
            if (not p.name.endswith(MANIFEST_SUFFIX) or not p.is_file() or p.is_symlink()
                    or guard.is_blocked_path(p) or p.stat().st_size > MAX_MANIFEST_BYTES):
                continue
            try:
                manifest = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(manifest, dict):
                rows.append(_row(condition, p.name, manifest))
    rows.sort(key=lambda r: r["started_at_utc"] or "", reverse=True)
    return rows
