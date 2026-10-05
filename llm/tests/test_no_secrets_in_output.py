"""Step 5 (docs/Agent prompt grounding factor ui.md): "never write API
keys, auth headers, or full request headers into any log or record. Add
a test that scans new log/record output for key-like strings."

Two things, in one file:
  1. A detector (KEY_LIKE_PATTERN) for common real-world API-key shapes
     (OpenAI "sk-...", Anthropic "sk-ant-...", generic long bearer-style
     tokens), verified against both true positives (synthetic fake keys)
     and true negatives (real content that merely LOOKS key-ish, e.g. a
     sha256 hex digest or a UUID, which must NOT false-positive).
  2. A scan of every EXISTING generation/judge result, log, and manifest
     file actually on disk in this repo -- this is a real regression
     check against current data, not just a synthetic unit test.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Deliberately pattern-matches the SHAPE of real provider key formats
# (prefix + a long run of key-charset characters), not a specific vendor
# string, so it also catches an accidentally-pasted real key regardless
# of which provider issued it.
KEY_LIKE_PATTERN = re.compile(
    r"sk-[A-Za-z0-9_-]{20,}"          # OpenAI-style (sk-..., sk-proj-..., sk-ant-...)
    r"|AKIA[0-9A-Z]{16}"               # AWS access key id shape
    r"|Bearer\s+[A-Za-z0-9._-]{20,}",  # a raw bearer-token header value
    re.IGNORECASE,
)

SCAN_DIRS = [
    REPO_ROOT / "llm" / "a_pure_llm" / "results",
    REPO_ROOT / "llm" / "a_pure_llm" / "logs",
    REPO_ROOT / "llm" / "b_rag" / "results",
    REPO_ROOT / "llm" / "b_rag" / "logs",
    REPO_ROOT / "llm" / "c_graphrag" / "results",
    REPO_ROOT / "llm" / "c_graphrag" / "logs",
    REPO_ROOT / "llm" / "d_lightrag" / "results",
    REPO_ROOT / "llm" / "d_lightrag" / "logs",
    REPO_ROOT / "llm" / "evaluation" / "results",
    REPO_ROOT / "llm" / "evaluation" / "logs",
]


def _iter_jsonl_and_manifest_files():
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        for path in d.glob("*.jsonl"):
            yield path
        for path in d.glob("*.manifest.json"):
            yield path


def test_key_like_pattern_detects_synthetic_fake_keys():
    assert KEY_LIKE_PATTERN.search("sk-" + "a" * 40)
    assert KEY_LIKE_PATTERN.search("sk-proj-" + "B1c2D3e4" * 4)
    assert KEY_LIKE_PATTERN.search("AKIA1234567890ABCDEF")
    assert KEY_LIKE_PATTERN.search("Authorization: Bearer " + "x" * 30)


def test_key_like_pattern_does_not_false_positive_on_benign_hash_like_strings():
    """A sha256 hex digest, a UUID, or a short model-name-ish string must
    NOT be flagged -- the detector targets specific KEY PREFIXES/SHAPES,
    not "any long string of hex/alnum characters"."""
    sha256_hex = "a97690f182231d922bab5ed2b4fc5c3aecd3d464c67c23584acb52bf3ca07ec"
    uuid_ = "550e8400-e29b-41d4-a716-446655440000"
    assert not KEY_LIKE_PATTERN.search(sha256_hex)
    assert not KEY_LIKE_PATTERN.search(uuid_)
    assert not KEY_LIKE_PATTERN.search("gpt-4o-mini-2024-07-18")
    assert not KEY_LIKE_PATTERN.search("meta-llama/Llama-3.3-70B-Instruct")


def test_no_key_like_strings_in_any_existing_result_log_or_manifest_file():
    """Real regression check: scan every actual .jsonl result/log file and
    every .manifest.json currently on disk for anything key-shaped. A
    failure here means a real secret may have been written to output --
    this must be investigated immediately, never silently allowed."""
    offenders = []
    for path in _iter_jsonl_and_manifest_files():
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        for match in KEY_LIKE_PATTERN.finditer(text):
            offenders.append((str(path), match.group(0)[:12] + "..."))  # never print the full match

    assert offenders == [], f"key-like strings found in output files: {offenders}"


def test_no_key_like_strings_in_request_params_field_specifically():
    """request_params (added for Step 5 audit trail) stores only
    model/temperature/max_tokens -- it must NEVER include api_key,
    Authorization, or any header. Checked on every record that has this
    field, across every existing result file."""
    for path in _iter_jsonl_and_manifest_files():
        if path.suffix == ".json":
            continue  # manifest files, handled by the generic scan above
        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            request_params = record.get("request_params")
            if not isinstance(request_params, dict):
                continue
            forbidden_keys = {"api_key", "authorization", "auth", "headers", "key", "token", "secret"}
            found = forbidden_keys & {k.lower() for k in request_params}
            assert not found, f"{path}: request_params contains forbidden key(s) {found}"
