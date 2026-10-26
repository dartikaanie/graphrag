"""Resume of a stopped generation run: one place used by the four CLIs and the dashboard engine
(2026-10-26 pre-run fix of the resume problems found in the check of the same day).

prepare_resume(output_path, config_hash, sample_split, log) -> already_done (question ids)

  1. Official cells only (sample_split == "test"): the first invocation of a run writes its code commit and the frozen
     parameter hash to <output>.invocations.jsonl (append-only, one line per invocation; the manifest lists them). A
     later invocation with a different commit or frozen hash is REFUSED (ResumeRefused). An output file that already has
     records but no invocation record is refused too: its starting commit cannot be verified.
  2. A torn last line (the process died while writing; no trailing newline) is cut off before anything is appended, and
     what was removed is logged. A last line that is complete JSON but lacks its newline gets the newline.
  3. Done = question ids of parseable records with the current config_hash. prompt_version is NOT part of the gate: since
     record schema r2 (commit 3c0378a) the records no longer carry it, so gating on it matched nothing and every resume
     regenerated and re-appended all questions. The filename already carries both prompt_version and config_hash.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class ResumeRefused(RuntimeError):
    pass


def invocations_path(output_path) -> Path:
    p = Path(output_path)
    return p.with_name(p.name + ".invocations.jsonl")


def read_invocations(output_path) -> list[dict]:
    path = invocations_path(output_path)
    out = []
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def repair_torn_tail(path, log=print) -> dict:
    """Make a JSONL file end at a line boundary. Returns {"removed_bytes", "removed_text", "added_newline"}."""
    path = Path(path)
    res = {"removed_bytes": 0, "removed_text": "", "added_newline": False}
    if not path.exists() or path.stat().st_size == 0:
        return res
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return res
    cut = data.rfind(b"\n") + 1            # 0 when the file has no newline at all
    tail = data[cut:]
    try:
        json.loads(tail.decode("utf-8"))
        with open(path, "ab") as f:         # a complete record that only lacks its newline
            f.write(b"\n")
        res["added_newline"] = True
        log(f"      [resume] last line of {path.name} was complete but had no newline: newline added")
        return res
    except Exception:
        pass
    with open(path, "r+b") as f:
        f.truncate(cut)
        f.flush()
        os.fsync(f.fileno())
    res.update(removed_bytes=len(tail), removed_text=tail[:80].decode("utf-8", "replace"))
    log(f"      [resume] torn last line of {path.name} removed ({len(tail)} bytes, starts: {res['removed_text']!r})")
    return res


def _current_state() -> dict:
    from llm.repo_state import repo_state
    from llm.run_registry import frozen_params_hash
    st = repo_state()
    return {"git_commit": st["git_commit"], "git_tree_clean": st["git_tree_clean"],
            "git_status": st["git_status"][:50], "frozen_params_hash": frozen_params_hash()}


def start_invocation(output_path, *, sample_split: str, config_hash: str, log=print) -> None:
    """Official (test-sample) cells only; see the module docstring, point 1."""
    if sample_split != "test":
        return
    path = Path(output_path)
    entries = read_invocations(path)
    cur = _current_state()
    has_records = path.exists() and path.stat().st_size > 0
    if entries:
        first = entries[0]
        diff = [f"{k}: started with {first.get(k)!r}, now {cur[k]!r}" for k in ("git_commit", "frozen_params_hash")
                if first.get(k) != cur[k]]
        if diff:
            raise ResumeRefused(
                f"cannot resume the official cell {path.name}: it was started under a different code or frozen "
                f"state ({'; '.join(diff)}). Resume at the starting commit, or archive the partial file and start "
                "a new run.")
    elif has_records:
        raise ResumeRefused(
            f"cannot resume {path.name}: it has records but no invocation record "
            f"({invocations_path(path).name}), so the commit and frozen hash it was started under are unknown.")
    entry = {"started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config_hash": config_hash,
             "pid": os.getpid(), "n_invocation": len(entries) + 1, **cur}
    with open(invocations_path(path), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())
    log(f"      [resume] invocation {entry['n_invocation']} of {path.name} at {str(cur['git_commit'])[:7]}, "
        f"frozen hash {cur['frozen_params_hash']}")


def prepare_resume(output_path, config_hash: str, sample_split: str = "test", log=print) -> set[int]:
    from llm.manifest import read_already_done
    start_invocation(output_path, sample_split=sample_split, config_hash=config_hash, log=log)
    repair_torn_tail(output_path, log)
    return read_already_done(output_path, None, config_hash)
