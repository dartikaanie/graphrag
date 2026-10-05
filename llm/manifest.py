"""
llm/manifest.py
=====================================
Shared run-manifest writer (Step 5, docs/Agent prompt grounding factor
ui.md -- data retention/audit trail). ONE function, used by every
generation condition (A/B/C/D) AND the judge-v1 runner, so the manifest
schema never silently drifts between them.

A manifest is written NEXT TO its output file, at
"<output_path>.manifest.json" -- one manifest per output file, re-
written (not appended) each time that SAME output file is touched again
(e.g. a resumed run), since it describes metadata ABOUT that file's
current, complete state -- not a historical log of every run attempt
(run_history.jsonl / judge_v1_run_history.jsonl already serve that
purpose, append-only). Writing/overwriting a manifest is therefore not
a "never overwrite" violation -- what must never be overwritten is the
OUTPUT DATA itself (already enforced by each script's existing
append-only + "already_done" resumability).
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

_TRACKED_PACKAGES = ("openai", "anthropic", "sentence-transformers", "faiss-cpu", "neo4j", "duckdb", "pandas")


def _git_info() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        )
        return {"commit_sha": sha, "dirty": bool(status.strip())}
    except Exception:
        return {"commit_sha": None, "dirty": None}


def _package_versions() -> dict[str, str | None]:
    import importlib.metadata

    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def sha256_of_file(path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class MixedConfigHashError(ValueError):
    """Raised by assert_single_config_hash() -- a result file contains
    records from more than one config_hash, meaning it mixes answers from
    genuinely different configs (or prompt_versions) under one filename.
    Every reader that aggregates a file's records (judge runner, the
    comparison scripts, dashboard summaries) must raise this instead of
    silently averaging/reporting across the mix."""


def assert_single_config_hash(records: list[dict[str, Any]], source_label: str) -> str | None:
    """Returns the one config_hash shared by every record in `records` (or
    None if none have one -- pre-config_hash legacy data). Raises
    MixedConfigHashError if more than one distinct, non-null value is
    present. Records missing config_hash entirely (legacy, pre-this-fix)
    are NOT counted as a mismatch against a present hash -- they predate
    the field and can't be blamed for not having it; but 2+ DIFFERENT
    present hashes always is."""
    hashes = {r.get("config_hash") for r in records if r.get("config_hash")}
    if len(hashes) > 1:
        raise MixedConfigHashError(
            f"{source_label}: records have {len(hashes)} different config_hash values "
            f"{sorted(hashes)} -- this file mixes results from genuinely different configs "
            f"(or prompt_versions) and must not be read/aggregated as one run. Investigate "
            f"and split or regenerate it before reading."
        )
    return next(iter(hashes), None)


def compute_config_hash(config: dict[str, Any]) -> str:
    """Short (10 hex char) deterministic hash of every parameter that can
    change the ANSWERS a run produces -- condition, provider, model,
    n_sample, seed, oversample_pool, top_k, fusion_mode + its weights/trust
    cap, semantic expansion on/off, n_anchor/n_semantic_expansion,
    n_low_level/n_high_level, require_citation, grounding, and the C/D
    retrieval code versions. Folded into every output filename (alongside
    prompt_version) and stored in every record + the manifest.

    This exists because prompt_version ALONE isn't enough to tell two runs
    apart: two DIFFERENT configs (e.g. require_grounding True vs False)
    sharing the SAME prompt_version would otherwise compute the identical
    filename and silently mix into one file (2026-10-05 pilot, the second,
    structural half of the fix -- see read_already_done()'s docstring for
    the first half). Keys absent for a given condition (e.g. fusion_mode
    for B) are simply omitted by the caller, not passed as None -- a caller
    passing an irrelevant key with a dummy value would make two otherwise-
    identical configs hash differently for no reason.
    """
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:10]


def read_already_done(output_path, current_prompt_version: str | None = None,
                       current_config_hash: str | None = None) -> set[int]:
    """Question ids already present at `output_path` -- used by every
    condition's resume support (CLI and dashboard) to skip re-calling the
    LLM for questions an INTERRUPTED run of the SAME config already has an
    answer for.

    `current_prompt_version`: when given, a record only counts as "already
    done" if its OWN stored `prompt_version` matches -- including a record
    with no `prompt_version` at all (pre-dates the field, so it can never
    match a current, versioned run). Without this gate, an output file left
    over from an OLDER prompt_version (e.g. a stale pre-grounding-axis file
    that happens to share a later run's deterministic filename) looks
    "already fully done" to a brand-new run using a DIFFERENT prompt, and
    zero new generation happens at all -- silently, with no error, just an
    empty results list. This is exactly what caused every A/B/C-grounded/
    D-grounded run to no-op during the first dashboard pilot batch
    (2026-10-05): each one's deterministic output filename happened to
    already exist, fully populated, from an OLDER prompt_version (v2 or
    pre-versioning) test run.

    `current_config_hash`: same idea, but the STRONGER, preferred check --
    gates on compute_config_hash()'s full-parameter hash rather than just
    prompt_version, so two configs that happen to share a prompt_version
    but differ in any other answer-affecting parameter (e.g. grounding)
    are never conflated either. Every output filename now embeds its own
    config_hash (see each condition's build_output_path()), so in practice
    a hash mismatch at a given PATH should never occur post-fix -- this
    check is defense in depth for a manually-overridden `--output` path,
    and is intentionally the AND of both checks when both are given: either
    one mismatching is enough to treat a record as not-done.
    """
    already_done: set[int] = set()
    path = Path(output_path)
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if current_prompt_version is not None and record.get("prompt_version") != current_prompt_version:
                        continue
                    if current_config_hash is not None and record.get("config_hash") != current_config_hash:
                        continue
                    already_done.add(record["question_id"])
                except Exception:
                    continue
    return already_done


def sum_usage_tokens(records: list[dict]) -> int | None:
    """Sum prompt+completion (or input+output, for Anthropic's naming)
    tokens from each record's "usage_full" dict -- None (not 0) if NO
    record had usage info at all, so a provider that never returns usage
    (e.g. local HF models) doesn't misleadingly report "0 tokens used"."""
    total = 0
    any_usage = False
    for r in records:
        usage = r.get("usage_full")
        if isinstance(usage, dict):
            any_usage = True
            total += usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            total += usage.get("completion_tokens") or usage.get("output_tokens") or 0
    return total if any_usage else None


def write_manifest(
    output_path,
    *,
    run_label: str | None,
    config: dict[str, Any],
    started_at_utc: str,
    finished_at_utc: str,
    item_counts: dict[str, int],
    prompt_version: str | None = None,
    judge_version: str | None = None,
    blinding_version: str | None = None,
    total_tokens: int | dict | None = None,
    extra_output_files: list | None = None,
    status: str = "completed",
    config_hash: str | None = None,
    log_path: str | None = None,
) -> Path:
    """Writes "<output_path>.manifest.json". Returns the manifest path.

    `item_counts`: e.g. {"attempted": 10, "succeeded": 9, "failed": 1}.
    `extra_output_files`: additional file paths to checksum alongside
    `output_path` itself (e.g. a judge run's <out>_failures.jsonl) --
    each gets its own sha256 entry in "output_files".
    `status`: "completed" / "interrupted" / "failed" -- a manifest is now
    written on every exit path (not just a clean finish), so a reader can
    tell whether `item_counts` describes a finished run or a partial one.
    `config_hash`: compute_config_hash(config)'s value for this run --
    stored alongside `config` (not just derivable from it) so a reader can
    cheaply compare manifest-to-manifest, or manifest-to-record, without
    recomputing the hash itself.
    `log_path`: path to this run's own log file, if one was written (CLI
    runs always have one; dashboard-launched runs only since the
    per-run-log-file fix -- see docs/Agent prompt grounding factor ui.md
    follow-ups). None when no per-run log file exists.
    """
    output_path = Path(output_path)
    all_files = [output_path] + [Path(p) for p in (extra_output_files or [])]

    manifest = {
        "output_files": {str(p): sha256_of_file(p) for p in all_files},
        "run_label": run_label,
        "status": status,
        "config": config,
        "config_hash": config_hash,
        "prompt_version": prompt_version,
        "judge_version": judge_version,
        "blinding_version": blinding_version,
        "log_path": log_path,
        "git": _git_info(),
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "python_version": sys.version,
        "package_versions": _package_versions(),
        "item_counts": item_counts,
        "total_tokens": total_tokens,
    }

    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest_path
