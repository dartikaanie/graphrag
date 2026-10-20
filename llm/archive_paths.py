"""
archive_paths.py
================
Central path resolver for the pre-official archive (2026-10-19; see
results/_archive/2026-10-19_pre-official/ARCHIVE_INDEX.md).

Every pre-official result/log file was moved, keeping its repo-relative path,
under one of the archive roots:
    results/_archive/<tag>/<original path>   (run outputs, reports, job state)
    logs/_archive/<tag>/<original path>      (run histories and run logs)
The move is recorded in results/_archive/<tag>/archive_map.json:
    {"files": {"<original rel>": "<archived rel>", ...},
     "dirs":  {"<original rel dir>": "<archived rel dir>", ...}}   # whole-directory moves

Frozen records (calibration_runs.json, freeze/export JSON, run histories)
keep their ORIGINAL paths byte-for-byte; code that opens such a path goes
through resolve(), code that compares identities goes through canonical().

    resolve(p)    -> where the file is now (the original path if it still
                     exists, else its archived location, else p unchanged)
    canonical(p)  -> the ORIGINAL absolute path (an archived path is mapped
                     back), so guards and registries keep matching
    archived(p)   -> archived location or None
    archived_dir(d) -> archived counterpart of a directory, or None
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOTS = (REPO_ROOT / "results" / "_archive", REPO_ROOT / "logs" / "_archive", REPO_ROOT / "docs" / "_archive")
MAP_GLOB = "*/archive_map.json"


@lru_cache(maxsize=1)
def _maps() -> tuple[dict[str, str], dict[str, str]]:
    files: dict[str, str] = {}
    dirs: dict[str, str] = {}
    for root in ARCHIVE_ROOTS:
        if not root.is_dir():
            continue
        for map_path in sorted(root.glob(MAP_GLOB)):
            m = json.loads(map_path.read_text())
            files.update(m.get("files") or {})
            dirs.update(m.get("dirs") or {})
    return files, dirs


def reload() -> None:
    """Drop the cached maps (tests, or after writing a new archive map)."""
    _maps.cache_clear()
    _inverse_files.cache_clear()


def _abs(p) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p)


def _rel_candidates(p) -> list[str]:
    """Repo-relative POSIX readings of `p`: against the cwd, then (for a
    relative path) against the repo root."""
    candidates = [_abs(p)]
    if not Path(p).is_absolute():
        candidates.append(REPO_ROOT / p)
    out = []
    for c in candidates:
        try:
            out.append(Path(*_normalise(c).relative_to(REPO_ROOT).parts).as_posix())
        except ValueError:
            continue
    return out


def repo_rel(p) -> str | None:
    """Repo-relative POSIX path of `p` (relative paths are taken against the
    cwd, then against the repo root), or None if outside the repo."""
    rels = _rel_candidates(p)
    return rels[0] if rels else None


def _normalise(p: Path) -> Path:
    # resolve ".." without requiring the file to exist (and without following
    # symlinks outside the repo for missing files)
    try:
        return p.resolve()
    except OSError:
        return Path(*p.parts)


def archived(p) -> Path | None:
    files, dirs = _maps()
    for rel in _rel_candidates(p):
        if rel in files:
            return REPO_ROOT / files[rel]
        for orig, new in dirs.items():
            if rel == orig or rel.startswith(orig + "/"):
                return REPO_ROOT / new / rel[len(orig):].lstrip("/") if rel != orig else REPO_ROOT / new
    return None


def resolve(p) -> Path:
    """The path to open: `p` if it exists (as given, or repo-relative),
    else its archived location if one is recorded, else `p` unchanged."""
    path = Path(p)
    if path.exists():
        return path
    if not path.is_absolute() and (REPO_ROOT / path).exists():
        return REPO_ROOT / path
    a = archived(p)
    if a is not None and a.exists():
        return a
    return path


def canonical(p) -> Path:
    """ORIGINAL absolute path of `p`: an archived location is mapped back to
    where the file lived before the archive; any other path is returned
    absolute and normalised."""
    rel = repo_rel(p)
    if rel is None:
        return _normalise(_abs(p))
    dirs = _maps()[1]
    inverse = _inverse_files()
    if rel in inverse:
        return REPO_ROOT / inverse[rel]
    for orig, new in dirs.items():
        if rel == new or rel.startswith(new + "/"):
            return REPO_ROOT / orig / rel[len(new):].lstrip("/") if rel != new else REPO_ROOT / orig
    return REPO_ROOT / rel


@lru_cache(maxsize=1)
def _inverse_files() -> dict[str, str]:
    return {new: orig for orig, new in _maps()[0].items()}


def is_archived(p) -> bool:
    rel = repo_rel(p)
    return rel is not None and any(rel.startswith(Path(r).relative_to(REPO_ROOT).as_posix() + "/")
                                   for r in ARCHIVE_ROOTS)


def archived_dir(d) -> Path | None:
    """Archived counterpart of directory `d` (for 'show archive' scans), or
    None when nothing under `d` was archived. The folder structure is kept,
    so this is <archive base>/<original rel dir> of any archived file below it."""
    rel = repo_rel(d)
    if rel is None:
        return None
    files, dirs = _maps()
    if rel in dirs:
        return REPO_ROOT / dirs[rel]
    prefix = rel.rstrip("/") + "/"
    for orig, new in files.items():
        if orig.startswith(prefix):
            return REPO_ROOT / new[: len(new) - len(orig)] / rel
    return None
