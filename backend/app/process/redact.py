"""
Secret redaction for every string the Process page sends to the browser
(Rule 6). Applied once, at the router boundary, to the whole response
(`redact_obj`), so no endpoint can forget it. `.env` itself is never read.

Covers: provider API keys (OpenAI/Anthropic `sk-...`, Hugging Face `hf_...`,
GitHub `gh?_...`, AWS `AKIA...`, Google `AIza...`), `Authorization` headers
and bearer tokens, `key=value` / `"key": "value"` pairs whose key names a
secret (api_key, secret, token, password, ...), and credentials embedded in
connection strings (`scheme://user:pass@host`).
"""

from __future__ import annotations

import re
from typing import Any

MASK = "[REDACTED]"

# A key name that denotes a secret: optional prefix ending in _ or -, then one
# of the secret words, then NOT another identifier character (so
# "max_tokens", "TOKEN_CHUNK_LIMIT" and "total_tokens" are left alone).
_SECRET_KEY = (
    r"(?<![A-Za-z0-9_\-])(?:[A-Za-z0-9]+[_\-])*"
    r"(?:api[_\-]?key|apikey|secret(?:[_\-]?key)?|password|passwd|pwd|token|"
    r"access[_\-]?key|private[_\-]?key|client[_\-]?secret)"
    r"(?![A-Za-z0-9_])"
)

_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Authorization: Bearer xxx / Authorization = Basic xxx
    (re.compile(r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?)(?:(bearer|basic|token)\s+)?[^\s,;\"'}]+"),
     r"\1\2 " + MASK),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/]+=*"), "Bearer " + MASK),
    # scheme://user:password@host
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^/\s:@]+:[^@\s/]+@"), r"\1" + MASK + "@"),
    # key="quoted value" / "key": 'quoted value'
    (re.compile(r"(?i)([\"']?" + _SECRET_KEY + r"[\"']?\s*[:=]\s*)([\"'])(.*?)\2"), r"\1\2" + MASK + r"\2"),
    # key=value / key: value (unquoted)
    (re.compile(r"(?i)([\"']?" + _SECRET_KEY + r"[\"']?\s*[:=]\s*)(?![\"'\[])([^\s,;\"'}]+)"), r"\1" + MASK),
    # bare provider keys anywhere in the text
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"), MASK),
    (re.compile(r"\bhf_[A-Za-z0-9]{8,}"), MASK),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"), MASK),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), MASK),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"), MASK),
]


def redact(text: str) -> str:
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text.replace("  " + MASK, " " + MASK)


def redact_obj(obj: Any) -> Any:
    """Recursively redact every string (dict keys are left as they are)."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj
