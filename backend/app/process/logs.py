"""
Log viewer allowlist (Rule 7). Logs are listed only from the directories
below, non-recursively and only for the listed filename patterns -- no glob
over the repo (stray results_old*/ folders are never reached). A log is
addressed by `<root key>/<file name>`; `resolve_log` rejects anything else:
path separators, `..`, hidden files, symlinks, files outside the root after
resolution, patterns not in the allowlist, and guard-blocked paths.
Reads are capped: at most MAX_READ_BYTES from the end of the file and at
most MAX_TAIL_LINES lines. Text is redacted at the router boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from app.process import guard
from app.process.sources import REPO_ROOT, file_stamp, newest_first, stamp_to_iso

MAX_READ_BYTES = 256 * 1024
MAX_TAIL_LINES = 1000
DEFAULT_TAIL_LINES = 200


@dataclass(frozen=True)
class LogRoot:
    key: str
    rel_dir: str
    patterns: tuple[str, ...]


LOG_ROOTS: tuple[LogRoot, ...] = (
    LogRoot("kg_logs", "01_data_cleaning/logs", ("*.log",)),
    LogRoot("repo_logs", "logs", ("*.log",)),
    LogRoot("logscore_check", "llm/c_graphrag/results/logscore_check", ("*.md",)),
    LogRoot("reputation_check", "llm/c_graphrag/results/reputation_check", ("*.md",)),
    LogRoot("controls", "llm/c_graphrag/results/controls", ("*_summary.json",)),
    LogRoot("c_results", "llm/c_graphrag/results", ("stage1_report_*.md",)),
)
_ROOTS = {r.key: r for r in LOG_ROOTS}

# filename fragment -> pipeline step key (first match wins)
_STEP_BY_FRAGMENT = (
    ("postlinks", "data"),
    ("build_knowledge_graph", "kg_build"),
    ("load_author_trust", "kg_build"),
    ("densify", "kg_build"),
    ("validate_densification", "kg_build"),
    ("apply_reputation_trust", "trust"),
    ("apply_logscore_trust", "trust"),
    ("logscore_top5", "trust"),
    ("reputation_check", "trust"),
    ("stage1", "alpha_selection"),
    ("controls", "alpha_selection"),
)


class LogAccessError(Exception):
    pass


def step_for(name: str) -> str | None:
    for fragment, step in _STEP_BY_FRAGMENT:
        if fragment in name:
            return step
    return None


def _root_dir(root: LogRoot) -> Path:
    return (REPO_ROOT / root.rel_dir).resolve()


def _allowed_name(root: LogRoot, name: str) -> bool:
    return any(fnmatch(name, pat) for pat in root.patterns)


def list_logs() -> list[dict]:
    out = []
    for root in LOG_ROOTS:
        directory = REPO_ROOT / root.rel_dir
        if not directory.is_dir():
            continue
        files = [
            p for p in directory.iterdir()
            if p.is_file() and not p.is_symlink() and not p.name.startswith(".")
            and _allowed_name(root, p.name) and not guard.is_blocked_path(p)
        ]
        for p in files:
            stamp = file_stamp(p.name)
            out.append({
                "id": f"{root.key}/{p.name}",
                "root": root.rel_dir,
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "timestamp_utc": stamp_to_iso(stamp),
                "step": step_for(p.name),
                "_sort": stamp or "",
            })
    out.sort(key=lambda r: (r["_sort"], r["name"]), reverse=True)
    for r in out:
        r.pop("_sort")
    return out


def resolve_log(log_id: str) -> Path:
    if not isinstance(log_id, str) or log_id.count("/") != 1:
        raise LogAccessError("invalid log id")
    key, name = log_id.split("/", 1)
    root = _ROOTS.get(key)
    if root is None:
        raise LogAccessError("unknown log root")
    if not name or name.startswith(".") or "\\" in name or ".." in name or "\x00" in name:
        raise LogAccessError("invalid file name")
    if not _allowed_name(root, name):
        raise LogAccessError("file type not allowed")
    candidate = REPO_ROOT / root.rel_dir / name
    if candidate.is_symlink():
        raise LogAccessError("symlinks are not allowed")
    resolved = candidate.resolve()
    if resolved.parent != _root_dir(root):
        raise LogAccessError("path outside the allowed directory")
    if not resolved.is_file():
        raise LogAccessError("not found")
    if guard.is_blocked_path(resolved):
        raise LogAccessError("blocked by the blinding guard")
    return resolved


def read_tail(log_id: str, lines: int = DEFAULT_TAIL_LINES) -> dict:
    path = resolve_log(log_id)
    lines = max(1, min(int(lines), MAX_TAIL_LINES))
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > MAX_READ_BYTES:
            fh.seek(size - MAX_READ_BYTES)
        data = fh.read(MAX_READ_BYTES)
    text = data.decode("utf-8", errors="replace")
    all_lines = text.splitlines()
    if size > MAX_READ_BYTES and all_lines:
        all_lines = all_lines[1:]  # first line is probably cut mid-way
    tail = all_lines[-lines:]
    return {
        "id": log_id,
        "name": path.name,
        "step": step_for(path.name),
        "size_bytes": size,
        "lines_returned": len(tail),
        "truncated": size > MAX_READ_BYTES or len(all_lines) > len(tail),
        "text": "\n".join(tail),
    }
