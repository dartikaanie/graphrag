"""
repo_state.py
-------------
Commit + working-tree state of this repository, for runs that must start
from a clean tree and record their commit (same rule as the ctxrel control
runs; independent implementation -- stage1_sweep.py is not imported).

`clean` ignores only the exact files in `exempt_paths` (a run's own
outputs) and git-ignored files (never listed by git status). Every other
modified, staged, deleted or untracked file is dirty.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_relative(path) -> str | None:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def parse_porcelain_z(raw: str) -> list[tuple[str, str]]:
    """`git status --porcelain -z` -> [(XY, path)]; a rename/copy entry's
    extra NUL-separated source path is skipped."""
    tokens, entries, i = raw.split("\0"), [], 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if not tok:
            continue
        xy, path = tok[:2], tok[3:]
        entries.append((xy, path))
        if "R" in xy or "C" in xy:
            i += 1
    return entries


def repo_state(exempt_paths=()) -> dict:
    """{"git_commit", "git_tree_clean", "git_status": ["XY path", ...],
    "git_exempted": [path, ...]}."""
    def _git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout

    exempt = {r for r in (repo_relative(p) for p in exempt_paths) if r}
    try:
        commit = _git("rev-parse", "HEAD").strip()
        entries = parse_porcelain_z(_git("status", "--porcelain", "-z", "--untracked-files=all"))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return {"git_commit": None, "git_tree_clean": False, "git_status": [f"<git error: {e}>"], "git_exempted": []}
    exempted = [p for _, p in entries if p in exempt]
    dirty = [p for _, p in entries if p not in exempt]
    return {"git_commit": commit, "git_tree_clean": not dirty,
            "git_status": [f"{xy} {p}" for xy, p in entries], "git_exempted": exempted}
