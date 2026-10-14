"""
Decision timeline parsed from docs/DECISION_C_SCORING.md (nothing retyped).
Each entry is the original decision (dated by its "Date written:" line) or
an "## Amendment N (YYYY-MM-DD): title" section, with its subsections quoted
verbatim, the commit hashes it cites, and its "applied" record lines.
"""

from __future__ import annotations

import re

from app.process.sources import read_text

DECISION_DOC = "docs/DECISION_C_SCORING.md"

_AMENDMENT_RE = re.compile(r"^## Amendment (\d+) \((\d{4}-\d{2}-\d{2})\): (.+)$")
_COMMIT_RE = re.compile(r"`([0-9a-f]{7,40})`")
_APPLIED_RE = re.compile(r"^\s*-\s*(DB update applied:|Apply timestamp|Recheck run:)")
_WHY_HEADINGS = ("Why", "Finding", "What happened", "Problem")


def _split_sections(lines: list[str], offset: int) -> list[dict]:
    sections, current = [], None
    for i, line in enumerate(lines):
        m = re.match(r"^#{2,3} (.+)$", line)
        if m and line.startswith("### "):
            current = {"heading": m.group(1).strip(), "line": offset + i + 1, "lines": []}
            sections.append(current)
        elif line.startswith("## "):
            current = {"heading": line[3:].strip(), "line": offset + i + 1, "lines": []}
            sections.append(current)
        elif current is not None:
            current["lines"].append(line)
        else:
            current = {"heading": "", "line": offset + i + 1, "lines": [line]}
            sections.append(current)
    return [
        {"heading": s["heading"], "line": s["line"], "text": "\n".join(s["lines"]).strip()}
        for s in sections if "\n".join(s["lines"]).strip()
    ]


def _commits(text: str) -> list[str]:
    seen: list[str] = []
    for c in _COMMIT_RE.findall(text):
        if c not in seen:
            seen.append(c)
    return seen


def parse_decisions(text: str | None = None) -> list[dict]:
    text = text if text is not None else read_text(DECISION_DOC)
    lines = text.splitlines()

    starts = [(i, _AMENDMENT_RE.match(line)) for i, line in enumerate(lines)]
    starts = [(i, m) for i, m in starts if m]
    end_markers = [i for i, line in enumerate(lines) if line.startswith("## Outcome")]
    doc_end = end_markers[0] if end_markers else len(lines)

    entries = []
    base_end = starts[0][0] if starts else doc_end
    date_line = next((line for line in lines[:base_end] if line.startswith("Date written:")), "")
    base_date = re.search(r"\d{4}-\d{2}-\d{2}", date_line)
    base_body = lines[1:base_end]
    entries.append({
        "id": "decision",
        "number": 0,
        "date": base_date.group(0) if base_date else None,
        "title": lines[0].lstrip("# ").strip() if lines else "",
        "line": 1,
        "line_end": base_end,
        "sections": _split_sections(base_body, 1),
        "commits": _commits("\n".join(base_body)),
        "applied": [],
    })

    for idx, (i, m) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else doc_end
        body = lines[i + 1:end]
        while body and body[-1].strip() in ("", "---"):
            body = body[:-1]
        body_text = "\n".join(body)
        entries.append({
            "id": f"amendment-{m.group(1)}",
            "number": int(m.group(1)),
            "date": m.group(2),
            "title": m.group(3).strip(),
            "line": i + 1,
            "line_end": i + len(body) + 1,
            "sections": _split_sections(body, i + 1),
            "commits": _commits(body_text),
            "applied": [line.strip().lstrip("- ").strip() for line in body if _APPLIED_RE.match(line)],
        })

    for e in entries:
        why = next((s for s in e["sections"] if s["heading"].split(" (")[0] in _WHY_HEADINGS), None)
        intro = next((s for s in e["sections"] if s["heading"] == ""), None)
        e["why"] = why
        e["intro"] = intro["text"] if intro else None
    entries.sort(key=lambda e: (e["date"] or "", e["number"]))
    return entries


def amendment_record_line(text: str, needle: str) -> str | None:
    """First line of the decision document containing `needle` (verbatim)."""
    for line in text.splitlines():
        if needle in line:
            return line.strip()
    return None
