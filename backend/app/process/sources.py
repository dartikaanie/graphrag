"""
Value resolvers: read a parameter from the file that actually defines or
records it, WITHOUT importing (executing) that file. Pipeline scripts call
load_dotenv() and pull in duckdb/faiss/sentence-transformers at import time;
parsing them with `ast` keeps the backend light and never touches `.env`.

Every resolver returns a `Resolved` with the value plus the file:line it came
from. A resolver that cannot find exactly one match raises `SourceError`
instead of guessing, so a moved or renamed constant shows up as an error on
the page (and in backend/tests/test_process_drift.py), never as a stale value.

Kinds (`Src.kind`):
  const           top-level `NAME = <expr>` (literals, other constants, + - * /)
  cli_default     `parser.add_argument("--flag", default=...)`; for
                  `int(os.getenv("X", 5))` the code default 5 is returned with
                  a note that the environment variable X overrides it
  call_kwarg      `VAR = {"key": Call(kwarg=<literal>)}` (e.g. JUDGE_REGISTRY)
  source_literal  one regex match (group 1) in the file, optionally restricted
                  to one function body; for literals inside function bodies or
                  Cypher strings that are not module constants
  json_field      a field of a JSON file
  log_field       group 1 of a regex in the newest matching log/report file
"""

from __future__ import annotations

import ast
import json
import operator
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

_STAMP_RE = re.compile(r"(\d{8}T\d{6}Z)")


class SourceError(Exception):
    """The value could not be located unambiguously in its source."""


@dataclass(frozen=True)
class Src:
    kind: str
    file: str  # repo-relative path; for log_field the directory
    name: str = ""  # constant name, CLI flag, dict variable, or regex
    path: tuple = ()  # subscript path (const/json_field) or (key, kwarg) for call_kwarg
    function: str = ""  # source_literal: restrict the search to this function body
    glob: str = ""  # log_field: filename pattern inside `file`


@dataclass
class Resolved:
    value: Any
    file: str
    line: int | None
    note: str | None = None


# ---------------------------------------------------------------- file access

def repo_path(rel: str) -> Path:
    return REPO_ROOT / rel


@lru_cache(maxsize=256)
def _read_cached(path_str: str, mtime_ns: int) -> str:
    return Path(path_str).read_text(encoding="utf-8")


def read_text(rel: str) -> str:
    p = repo_path(rel)
    if not p.is_file():
        raise SourceError(f"{rel}: file not found")
    return _read_cached(str(p), p.stat().st_mtime_ns)


@lru_cache(maxsize=128)
def _parse_cached(path_str: str, mtime_ns: int) -> ast.Module:
    return ast.parse(_read_cached(path_str, mtime_ns))


def _parse(rel: str) -> ast.Module:
    p = repo_path(rel)
    if not p.is_file():
        raise SourceError(f"{rel}: file not found")
    return _parse_cached(str(p), p.stat().st_mtime_ns)


def file_stamp(name: str) -> str | None:
    m = _STAMP_RE.search(name)
    return m.group(1) if m else None


def stamp_to_iso(stamp: str | None) -> str | None:
    if not stamp:
        return None
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()


def newest_first(paths: list[Path]) -> list[Path]:
    """Sort by the UTC stamp in the filename (falls back to mtime)."""
    def key(p: Path):
        s = file_stamp(p.name)
        return (s or datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime("%Y%m%dT%H%M%SZ"), p.name)
    return sorted(paths, key=key, reverse=True)


# ---------------------------------------------------------------- ast helpers

def _module_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for node in tree.body:  # later assignments win, as at runtime
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            out[node.target.id] = node
    return out


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return f"{_call_name(func.value)}.{func.attr}"
    return ""


_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_CASTS = {"int": int, "float": float, "str": str}


def _eval(node: ast.AST, consts: dict[str, ast.AST], depth: int = 0) -> tuple[Any, str | None]:
    if depth > 20:
        raise SourceError("expression too deep")
    try:
        return ast.literal_eval(node), None
    except (ValueError, TypeError, SyntaxError):
        pass
    if isinstance(node, ast.Name) and node.id in consts:
        return _eval(consts[node.id].value, consts, depth + 1)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, n1 = _eval(node.left, consts, depth + 1)
        right, n2 = _eval(node.right, consts, depth + 1)
        return _BINOPS[type(node.op)](left, right), n1 or n2
    if isinstance(node, ast.Call):
        fname = _call_name(node.func)
        if fname in _CASTS and len(node.args) == 1 and not node.keywords:
            value, note = _eval(node.args[0], consts, depth + 1)
            return _CASTS[fname](value), note
        if fname in ("os.getenv", "os.environ.get") and len(node.args) == 2:
            env_name = ast.literal_eval(node.args[0])
            value, _ = _eval(node.args[1], consts, depth + 1)
            return value, f"code default; the environment variable {env_name} overrides it"
    raise SourceError(f"unsupported expression: {ast.unparse(node)}")


def _subscript(value: Any, path: tuple) -> Any:
    for key in path:
        value = value[key]
    return value


# ---------------------------------------------------------------- resolvers

def _const(src: Src) -> Resolved:
    consts = _module_assignments(_parse(src.file))
    if src.name not in consts:
        raise SourceError(f"{src.file}: no module-level constant {src.name}")
    node = consts[src.name]
    value, note = _eval(node.value, consts)
    return Resolved(_subscript(value, src.path), src.file, node.lineno, note)


def _cli_default(src: Src) -> Resolved:
    tree = _parse(src.file)
    consts = _module_assignments(tree)
    hits = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
        and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == src.name
    ]
    if len(hits) != 1:
        raise SourceError(f"{src.file}: expected one add_argument({src.name!r}), found {len(hits)}")
    kw = next((k for k in hits[0].keywords if k.arg == "default"), None)
    if kw is None:
        raise SourceError(f"{src.file}: {src.name} has no default")
    value, note = _eval(kw.value, consts)
    return Resolved(value, src.file, hits[0].lineno, note or "CLI default")


def _call_kwarg(src: Src) -> Resolved:
    consts = _module_assignments(_parse(src.file))
    node = consts.get(src.name)
    if node is None or not isinstance(node.value, ast.Dict):
        raise SourceError(f"{src.file}: {src.name} is not a module-level dict")
    key, kwarg = src.path
    for k, v in zip(node.value.keys, node.value.values):
        if isinstance(k, ast.Constant) and k.value == key and isinstance(v, ast.Call):
            for kw in v.keywords:
                if kw.arg == kwarg:
                    return Resolved(ast.literal_eval(kw.value), src.file, kw.value.lineno)
    raise SourceError(f"{src.file}: {src.name}[{key!r}].{kwarg} not found")


def _source_literal(src: Src) -> Resolved:
    text = read_text(src.file)
    first_line = 1
    if src.function:
        funcs = [n for n in ast.walk(_parse(src.file))
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == src.function]
        if len(funcs) != 1:
            raise SourceError(f"{src.file}: expected one function {src.function}, found {len(funcs)}")
        lines = text.splitlines()
        text = "\n".join(lines[funcs[0].lineno - 1:funcs[0].end_lineno])
        first_line = funcs[0].lineno
    matches = list(re.finditer(src.name, text, re.MULTILINE))
    if len(matches) != 1:
        raise SourceError(f"{src.file}: pattern {src.name!r} matched {len(matches)} times (expected 1)")
    m = matches[0]
    raw = m.group(1)
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw
    line = first_line + text[:m.start(1)].count("\n")
    return Resolved(value, src.file, line, "literal in code (not a named constant)")


def _json_field(src: Src) -> Resolved:
    try:
        data = json.loads(read_text(src.file))
        return Resolved(_subscript(data, src.path), src.file, None)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise SourceError(f"{src.file}: field {src.path} not readable ({exc})") from exc


def _parse_number(raw: str) -> Any:
    cleaned = raw.replace(",", "")
    try:
        return int(cleaned)
    except ValueError:
        try:
            return float(cleaned)
        except ValueError:
            return raw


def _log_field(src: Src) -> Resolved:
    directory = repo_path(src.file)
    if not directory.is_dir():
        raise SourceError(f"{src.file}: directory not found")
    files = newest_first([p for p in directory.iterdir() if p.is_file() and fnmatch(p.name, src.glob)])
    pattern = re.compile(src.name, re.MULTILINE)
    for p in files:
        rel = f"{src.file}/{p.name}"
        text = read_text(rel)
        m = pattern.search(text)
        if m:
            return Resolved(_parse_number(m.group(1)), rel, text[:m.start(1)].count("\n") + 1)
    raise SourceError(f"{src.file}/{src.glob}: no file matches {src.name!r}")


_RESOLVERS = {
    "const": _const,
    "cli_default": _cli_default,
    "call_kwarg": _call_kwarg,
    "source_literal": _source_literal,
    "json_field": _json_field,
    "log_field": _log_field,
}


def resolve(src: Src) -> Resolved:
    try:
        resolver = _RESOLVERS[src.kind]
    except KeyError as exc:
        raise SourceError(f"unknown source kind {src.kind}") from exc
    return resolver(src)


# ---------------------------------------------------------------- quotes

def quote(rel: str, needle: str) -> dict:
    """Verbatim excerpt of `rel` around the ONE line containing `needle`.

    Markdown: the bullet/table row/paragraph containing the line. Python:
    the contiguous `#` comment block, an inline comment, or the docstring
    paragraph. Raises SourceError unless the needle occurs exactly once, so
    a rewritten document never silently yields the wrong text."""
    lines = read_text(rel).splitlines()
    hits = [i for i, line in enumerate(lines) if needle in line]
    if len(hits) != 1:
        raise SourceError(f"{rel}: quote needle found {len(hits)} times: {needle!r}")
    i = hits[0]
    start = end = i

    if rel.endswith(".py"):
        stripped = lines[i].lstrip()
        if stripped.startswith("#"):
            while start > 0 and lines[start - 1].lstrip().startswith("#"):
                start -= 1
            while end + 1 < len(lines) and lines[end + 1].lstrip().startswith("#"):
                end += 1
            text = "\n".join(re.sub(r"^\s*#\s?", "", line) for line in lines[start:end + 1])
        elif "  # " in lines[i]:
            text = lines[i].split("  # ", 1)[1]
        else:  # docstring paragraph, bounded by blank lines and the docstring quotes
            while start > 0 and lines[start - 1].strip() and '"""' not in lines[start]:
                start -= 1
            while end + 1 < len(lines) and lines[end + 1].strip() and '"""' not in lines[end]:
                end += 1
            text = "\n".join(line.strip().replace('"""', "") for line in lines[start:end + 1])
    else:
        block_start = re.compile(r"^\s*([-*]|\d+\.|\||#{1,6})\s")
        if block_start.match(lines[i]):
            while end + 1 < len(lines) and lines[end + 1].startswith("  ") and not block_start.match(lines[end + 1]):
                end += 1
        else:
            while start > 0 and lines[start - 1].strip() and not block_start.match(lines[start - 1]):
                start -= 1
            while end + 1 < len(lines) and lines[end + 1].strip() and not block_start.match(lines[end + 1]):
                end += 1
        text = "\n".join(line.strip() for line in lines[start:end + 1])

    return {"text": text.strip(), "file": rel, "line": start + 1, "line_end": end + 1}


# ---------------------------------------------------------------- git provenance

def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


@lru_cache(maxsize=1024)
def _last_change_cached(rel: str, line: int | None, mtime_ns: int) -> dict | None:
    if line:
        out = _git(["blame", "--porcelain", "-L", f"{line},{line}", "--", rel])
        if not out:
            return None
        rows = out.splitlines()
        sha = rows[0].split()[0]
        if set(sha) == {"0"}:
            return {"commit": None, "date": None, "uncommitted": True}
        ts = next((int(r.split()[1]) for r in rows if r.startswith("committer-time ")), None)
        date = datetime.fromtimestamp(ts, timezone.utc).date().isoformat() if ts else None
        return {"commit": sha[:7], "date": date, "uncommitted": False}
    out = _git(["log", "-1", "--format=%h|%cs", "--", rel])
    if not out or "|" not in out:
        return None
    commit, date = out.strip().split("|", 1)
    return {"commit": commit, "date": date, "uncommitted": False}


def last_change(rel: str, line: int | None) -> dict | None:
    """Commit + date that last touched `rel` (at `line` when given), or None
    when git is unavailable. Read-only (`git blame` / `git log`)."""
    p = repo_path(rel)
    if not p.exists():
        return None
    return _last_change_cached(rel, line, p.stat().st_mtime_ns)
