"""
reparse_judge_v2.py
=====================================
Offline re-parse of EXISTING judge-v2 output files against the fixed
parse_judgment_v2() (narrow "null"/"none" severity and "true"/"false"
reference_conflict string normalization -- see docs/JUDGE_V2_CHANGES.md,
2026-10-07 follow-up). Makes NO API calls: every record's already-stored
`raw_response` is simply re-parsed.

For each input file, every record is copied verbatim EXCEPT the
parse-derived fields (label, derived_label, judge_reported_label,
consistent, answer_attempted, completeness, claims, reasoning,
attempted_claims_conflict, parse_error, parse_version, normalized_fields),
which are replaced by what the fixed parser produces from the SAME
raw_response. `max_tokens` (read from request_params, None for every
existing record -- no cap was ever sent) and `truncated`
(finish_reason == "length") are added for parity with judge_one()'s
live-run record shape, going forward.

Writes to a NEW file per input (suffix "_reparsed.jsonl") -- the
original file is opened read-only and is never modified.

CARA PAKAI
----------------------------------------------------------------------------
    cd llm/evaluation
    python3 reparse_judge_v2.py \\
        results/jv2_dashboard_primary_judge-v2.jsonl \\
        results/jv2_dashboard_secondary_judge-v2.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_prompt_v2 import parse_judgment_v2  # noqa: E402

PARSE_DERIVED_FIELDS = (
    "label", "derived_label", "judge_reported_label", "consistent",
    "answer_attempted", "completeness", "claims", "reasoning",
    "attempted_claims_conflict", "parse_error", "parse_version", "normalized_fields",
)


def reparse_record(record: dict) -> dict:
    """Returns a NEW dict -- never mutates `record`."""
    out = dict(record)
    if record.get("call_failed") or not record.get("raw_response"):
        # Nothing to re-parse (the API call itself failed) -- leave as is,
        # just add the two new parity fields with their existing semantics.
        out["max_tokens"] = out.get("max_tokens", record.get("request_params", {}).get("max_tokens"))
        out["truncated"] = out.get("truncated", record.get("finish_reason") == "length")
        return out

    parsed = parse_judgment_v2(record["raw_response"])
    for field in PARSE_DERIVED_FIELDS:
        out[field] = parsed.get(field)
    out["max_tokens"] = record.get("request_params", {}).get("max_tokens")
    out["truncated"] = record.get("finish_reason") == "length"
    return out


def reparse_file(input_path: Path) -> dict:
    """Reads `input_path`, writes `<stem>_reparsed.jsonl` next to it
    (never overwriting an existing file of that name on a re-run --
    callers that want a fresh pass should remove the old reparsed file
    themselves first, same resumability stance as every other script
    here). Returns a stats dict for reporting."""
    output_path = input_path.with_name(input_path.stem + "_reparsed" + input_path.suffix)

    before_errors_by_label: Counter = Counter()
    before_errors_by_condition: Counter = Counter()
    after_errors_by_label: Counter = Counter()
    after_errors_by_condition: Counter = Counter()
    normalized_counts: Counter = Counter()
    n_total = 0
    n_before_errors = 0
    n_after_errors = 0

    reparsed_records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n_total += 1
            run_label = record.get("run_label")
            condition = record.get("condition")

            if record.get("parse_error"):
                n_before_errors += 1
                before_errors_by_label[run_label] += 1
                before_errors_by_condition[condition] += 1

            new_record = reparse_record(record)
            reparsed_records.append(new_record)

            if new_record.get("parse_error"):
                n_after_errors += 1
                after_errors_by_label[run_label] += 1
                after_errors_by_condition[condition] += 1
            for field in new_record.get("normalized_fields") or []:
                normalized_counts[field] += 1

    with open(output_path, "w") as f:
        for r in reparsed_records:
            f.write(json.dumps(r, default=str) + "\n")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "n_total": n_total,
        "n_before_errors": n_before_errors,
        "n_after_errors": n_after_errors,
        "before_errors_by_run_label": dict(before_errors_by_label),
        "after_errors_by_run_label": dict(after_errors_by_label),
        "before_errors_by_condition": dict(before_errors_by_condition),
        "after_errors_by_condition": dict(after_errors_by_condition),
        "normalized_field_counts": dict(normalized_counts),
    }


def format_report(stats: dict) -> str:
    lines = [
        f"=== {stats['input_path']} -> {stats['output_path']} ===",
        f"n={stats['n_total']}  parse_error: {stats['n_before_errors']} -> {stats['n_after_errors']}",
        f"normalized field occurrences: {stats['normalized_field_counts']}",
        "parse_error by run_label (before -> after):",
    ]
    labels = sorted(set(stats["before_errors_by_run_label"]) | set(stats["after_errors_by_run_label"]))
    for label in labels:
        b = stats["before_errors_by_run_label"].get(label, 0)
        a = stats["after_errors_by_run_label"].get(label, 0)
        lines.append(f"  {label}: {b} -> {a}")
    lines.append("parse_error by condition (before -> after):")
    conditions = sorted(set(stats["before_errors_by_condition"]) | set(stats["after_errors_by_condition"]))
    for condition in conditions:
        b = stats["before_errors_by_condition"].get(condition, 0)
        a = stats["after_errors_by_condition"].get(condition, 0)
        lines.append(f"  {condition}: {b} -> {a}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 reparse_judge_v2.py <input1.jsonl> [input2.jsonl ...]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        stats = reparse_file(Path(arg))
        print(format_report(stats))
        print()
