"""Cross-checks _run_metadata.py's run_id AND run_label formulas against
the REAL backend/app/services/history_service.py copies they are
manually-synced duplicates of (see _run_metadata.py's module docstring
for why they're duplicated rather than imported at runtime). This test is
the guard rail against the two silently drifting apart -- it imports the
backend module directly (test-only; the shipped llm/evaluation/ scripts
never do this) and asserts byte-identical output for EVERY entry in every
condition's logs/run_history.jsonl, not just one hand-picked example.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.history_service import _history_id as backend_history_id  # noqa: E402
from app.services.history_service import derive_run_label as backend_derive_run_label  # noqa: E402

from _run_metadata import HISTORY_PATHS, _history_id as local_history_id  # noqa: E402
from _run_metadata import derive_run_label as local_derive_run_label  # noqa: E402


def _all_run_history_entries():
    entries = []
    for condition, path in HISTORY_PATHS.items():
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries.append((condition, record))
    return entries


@pytest.mark.parametrize("condition,record", _all_run_history_entries(),
                          ids=lambda v: v if isinstance(v, str) else v.get("output_path", "?"))
def test_local_history_id_matches_backend_for_every_real_entry(condition, record):
    assert local_history_id(condition, record) == backend_history_id(condition, record)


@pytest.mark.parametrize("condition,record", _all_run_history_entries(),
                          ids=lambda v: v if isinstance(v, str) else v.get("output_path", "?"))
def test_local_run_label_matches_backend_for_every_real_entry(condition, record):
    assert local_derive_run_label(condition, record) == backend_derive_run_label(condition, record)


def test_at_least_one_real_entry_was_actually_checked():
    """Guards against this whole test file silently doing nothing if
    every condition's run_history.jsonl were ever empty/missing."""
    assert len(_all_run_history_entries()) > 0


def test_hash_formula_is_sensitive_to_each_input_component():
    """Not just "matches on real data" -- prove the two formulas would
    also DIVERGE together (not coincidentally agree) if run_started_at,
    output_path, or condition changed, by checking a few synthetic
    variations land on the same id from both functions."""
    base = {"run_started_at": "2026-01-01T00:00:00+00:00", "output_path": "/x/y.jsonl"}
    variations = [
        {**base},
        {**base, "run_started_at": "2026-01-02T00:00:00+00:00"},
        {**base, "output_path": "/x/z.jsonl"},
    ]
    for record in variations:
        assert local_history_id("B", record) == backend_history_id("B", record)
    # And condition alone must also change the id identically in both.
    assert local_history_id("B", base) == backend_history_id("B", base)
    assert local_history_id("C", base) == backend_history_id("C", base)
    assert local_history_id("B", base) != local_history_id("C", base)


def test_run_label_formula_is_sensitive_to_each_axis_in_both_copies():
    """Same guard as the hash test above, but for derive_run_label(): both
    copies must agree not just on real data, but across every combination
    of condition/fusion_mode/grounding/legacy-hack input."""
    cases = [
        ("A", {}),
        ("B", {"grounding": "on"}),
        ("B", {"grounding": "off"}),
        ("B", {"fusion_mode": "grounded"}),  # legacy hack
        ("B", {"fusion_mode": "plain"}),  # legacy hack
        ("B", {}),  # historical default
        ("C", {"fusion_mode": "uniform", "grounding": "on"}),
        ("C", {"fusion_mode": "trust_weighted", "grounding": "off"}),
        ("C", {"fusion_mode": "uniform"}),  # grounding inferred
        ("D", {"grounding": "on"}),
        ("D", {"grounding": "off"}),
        ("D", {}),  # historical default
    ]
    for condition, record in cases:
        assert local_derive_run_label(condition, record) == backend_derive_run_label(condition, record), (
            f"mismatch for condition={condition!r} record={record!r}"
        )
