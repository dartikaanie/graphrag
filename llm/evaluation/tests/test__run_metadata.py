"""Offline tests for _run_metadata.py's filename->condition inference and
the run_history.jsonl -> run parameters resolution, including the real
repo data (C's uniform vs trust_weighted ablation variants must resolve
to distinct run_label/fusion_mode -- this is the whole point of this
module, per the user's explicit "Condition labeling (blocking)" request).
"""

from pathlib import Path

import pytest

from _run_metadata import (
    RunMetadataNotFoundError,
    derive_run_label,
    infer_condition_from_filename,
    resolve_run_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("filename,expected", [
    ("condition_a_openai_gpt-4o-mini_n10_seed42.jsonl", "A"),
    ("condition_b_openai_gpt-4o-mini_n10_seed42.jsonl", "B"),
    ("condition_c_openai_gpt-4o-mini_n10_seed42_fw0-7-0-3.jsonl", "C"),
    ("condition_c_openai_gpt-4o-mini_n10_seed42_uniform.jsonl", "C"),
    ("condition_d_openai_gpt-4o-mini_n10_seed42_grounded.jsonl", "D"),
])
def test_infer_condition_from_filename(filename, expected):
    assert infer_condition_from_filename(f"/some/dir/{filename}") == expected


def test_infer_condition_rejects_unknown_prefix():
    with pytest.raises(RunMetadataNotFoundError):
        infer_condition_from_filename("/some/dir/results_old/weird_file.jsonl")


def test_resolve_run_metadata_raises_for_file_with_no_run_history_entry(tmp_path):
    fake_file = tmp_path / "condition_b_totally_made_up.jsonl"
    fake_file.write_text("")
    with pytest.raises(RunMetadataNotFoundError):
        resolve_run_metadata(str(fake_file))


def test_resolve_run_metadata_distinguishes_c_fusion_mode_variants():
    """THE regression this module exists for: C's two result files must
    resolve to DIFFERENT run_label/fusion_mode, never both collapsing to
    a bare "C" -- that would silently merge the trust_weighted and
    uniform ablation arms. fusion_mode itself stays in FULL
    ("trust_weighted"), only run_label shortens it ("trust")."""
    trust_weighted = resolve_run_metadata(
        str(REPO_ROOT / "llm/c_graphrag/results/condition_c_openai_gpt-4o-mini_n10_seed42_fw0-7-0-3.jsonl")
    )
    uniform = resolve_run_metadata(
        str(REPO_ROOT / "llm/c_graphrag/results/condition_c_openai_gpt-4o-mini_n10_seed42_uniform.jsonl")
    )

    assert trust_weighted["fusion_mode"] == "trust_weighted"
    assert trust_weighted["run_label"] == "C-trust-grounded"
    assert uniform["fusion_mode"] == "uniform"
    assert uniform["run_label"] == "C-uniform-grounded"
    assert trust_weighted["run_label"] != uniform["run_label"]
    assert trust_weighted["run_id"] != uniform["run_id"]
    # Both real C runs predate the dedicated "grounding" field -- grounding
    # is correctly inferred "on" (C's historical default) from require_grounding.
    assert trust_weighted["grounding"] == "on"
    assert uniform["grounding"] == "on"


def test_resolve_run_metadata_known_run_id_matches_dashboard_hash(tmp_path, monkeypatch):
    """Pinned regression: this exact run_id is reproducible from the
    SAME (condition, run_started_at, output_path) formula the dashboard's
    history_service.py::_history_id() uses -- if this ever breaks, judge
    output run_ids will stop lining up with the dashboard's History page.

    Reads a FIXED fixture copy of the relevant run_history.jsonl entry
    (not the live file) -- this test used to read the real
    llm/c_graphrag/logs/run_history.jsonl directly, which made it flaky:
    any new real run appended to that file (e.g. a pilot batch) can shift
    which record resolve_run_metadata() picks up for a given output_path,
    since it intentionally returns the LAST matching record (a legitimate
    "re-run overwrites this path's metadata" semantic, not a bug) --
    breaking this pin for a reason unrelated to the formula it's meant to
    guard. A fixed fixture removes that source of flakiness entirely.
    """
    import _run_metadata

    # The run_id hash is derived from (condition, run_started_at,
    # output_path) -- output_path must be this EXACT original string (not
    # a tmp_path, which would change every test run) to reproduce the
    # pinned hash. The file doesn't need to physically exist at this path:
    # resolve_run_metadata() only matches path strings against
    # run_history.jsonl records, it never opens output_path itself.
    output_path = str(REPO_ROOT / "llm/c_graphrag/results/condition_c_openai_gpt-4o-mini_n10_seed42_uniform.jsonl")
    history_path = tmp_path / "run_history.jsonl"
    history_path.write_text(
        '{"run_started_at": "2026-09-29T01:47:57.541326+00:00", "condition": "C", '
        f'"status": "success", "output_path": "{output_path}", "fusion_mode": "uniform", '
        '"require_grounding": true}\n'
    )
    monkeypatch.setitem(_run_metadata.HISTORY_PATHS, "C", history_path)

    uniform = resolve_run_metadata(output_path)
    assert uniform["run_id"] == "C-958c05cc40"


def test_resolve_run_metadata_condition_without_fusion_mode_has_bare_run_label():
    a = resolve_run_metadata(str(REPO_ROOT / "llm/a_pure_llm/results/condition_a_openai_gpt-4o-mini_n10_seed42.jsonl"))
    assert a["fusion_mode"] is None
    assert a["run_label"] == "A"


def test_resolve_run_metadata_d_has_retrieval_version_not_fusion_mode():
    d = resolve_run_metadata(
        str(REPO_ROOT / "llm/d_lightrag/results/condition_d_openai_gpt-4o-mini_n10_seed42_grounded.jsonl")
    )
    assert d["fusion_mode"] is None
    assert d["run_label"] == "D-grounded"
    assert d["grounding"] == "on"
    assert d["retrieval_version"] == "v3"


def test_resolve_run_metadata_distinguishes_b_grounded_vs_b_plain_dedicated_field(tmp_path, monkeypatch):
    """Condition B's CURRENT (dedicated "grounding" field) runs must
    resolve to distinct, correctly-labeled run_label values."""
    import _run_metadata

    plain_output = tmp_path / "condition_b_openai_gpt-4o-mini_n10_seed42_plain.jsonl"
    grounded_output = tmp_path / "condition_b_openai_gpt-4o-mini_n10_seed42_grounded.jsonl"
    plain_output.write_text("")
    grounded_output.write_text("")

    history_path = tmp_path / "run_history.jsonl"
    history_path.write_text(
        '{"run_started_at": "t1", "output_path": "' + str(plain_output) + '", "grounding": "off"}\n'
        '{"run_started_at": "t2", "output_path": "' + str(grounded_output) + '", "grounding": "on"}\n'
    )
    monkeypatch.setitem(_run_metadata.HISTORY_PATHS, "B", history_path)

    plain = resolve_run_metadata(str(plain_output))
    grounded = resolve_run_metadata(str(grounded_output))

    assert plain["run_label"] == "B-plain"
    assert plain["grounding_inferred"] is False
    assert grounded["run_label"] == "B-grounded"
    assert grounded["grounding_inferred"] is False
    assert plain["run_label"] != grounded["run_label"]
    assert plain["run_id"] != grounded["run_id"]


def test_resolve_run_metadata_b_legacy_fusion_mode_hack_reads_correctly(tmp_path, monkeypatch):
    """B briefly wrote its grounding state INTO fusion_mode ("plain"/
    "grounded") before the dedicated "grounding" field existed (see
    docs/GROUNDING_FACTOR_UI.md) -- any run_history entry from that
    window must still resolve to the correct run_label, marked inferred,
    and must NEVER leak "plain"/"grounded" back out as if it were a real
    fusion_mode value."""
    import _run_metadata

    plain_output = tmp_path / "condition_b_legacy_plain.jsonl"
    grounded_output = tmp_path / "condition_b_legacy_grounded.jsonl"
    plain_output.write_text("")
    grounded_output.write_text("")

    history_path = tmp_path / "run_history.jsonl"
    history_path.write_text(
        '{"run_started_at": "t1", "output_path": "' + str(plain_output) + '", "fusion_mode": "plain"}\n'
        '{"run_started_at": "t2", "output_path": "' + str(grounded_output) + '", "fusion_mode": "grounded"}\n'
    )
    monkeypatch.setitem(_run_metadata.HISTORY_PATHS, "B", history_path)

    plain = resolve_run_metadata(str(plain_output))
    grounded = resolve_run_metadata(str(grounded_output))

    assert plain["run_label"] == "B-plain"
    assert plain["fusion_mode"] is None  # never leaked as a real fusion_mode
    assert plain["grounding_inferred"] is True
    assert grounded["run_label"] == "B-grounded"
    assert grounded["fusion_mode"] is None
    assert grounded["grounding_inferred"] is True


def test_resolve_run_metadata_b_with_no_grounding_concept_at_all_infers_historical_plain(tmp_path, monkeypatch):
    """A B run_history entry from before ANY of this existed (no
    "grounding", no "require_grounding", no legacy fusion_mode hack) must
    infer "plain" -- B's historical default -- marked inferred."""
    import _run_metadata

    output = tmp_path / "condition_b_ancient.jsonl"
    output.write_text("")
    history_path = tmp_path / "run_history.jsonl"
    history_path.write_text('{"run_started_at": "t0", "output_path": "' + str(output) + '"}\n')
    monkeypatch.setitem(_run_metadata.HISTORY_PATHS, "B", history_path)

    result = resolve_run_metadata(str(output))
    assert result["run_label"] == "B-plain"
    assert result["grounding_inferred"] is True


# ---------------------------------------------------------------------------
# derive_run_label() -- ALL 9 factorial labels from docs/GROUNDING_FACTOR_UI.md's
# table, plus the legacy/inferred paths. Pure function, no files/filesystem.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("condition,record,expected_label", [
    ("A", {}, "A"),
    ("B", {"grounding": "off"}, "B-plain"),
    ("B", {"grounding": "on"}, "B-grounded"),
    ("C", {"fusion_mode": "uniform", "grounding": "off"}, "C-uniform-plain"),
    ("C", {"fusion_mode": "uniform", "grounding": "on"}, "C-uniform-grounded"),
    ("C", {"fusion_mode": "trust_weighted", "grounding": "off"}, "C-trust-plain"),
    ("C", {"fusion_mode": "trust_weighted", "grounding": "on"}, "C-trust-grounded"),
    ("D", {"grounding": "off"}, "D-plain"),
    ("D", {"grounding": "on"}, "D-grounded"),
])
def test_derive_run_label_all_9_factorial_labels(condition, record, expected_label):
    result = derive_run_label(condition, record)
    assert result["run_label"] == expected_label
    assert result["grounding_inferred"] is False


def test_derive_run_label_fusion_mode_always_kept_in_full_not_shortened_in_the_field():
    result = derive_run_label("C", {"fusion_mode": "trust_weighted", "grounding": "on"})
    assert result["fusion_mode"] == "trust_weighted"  # full value
    assert "trust-" in result["run_label"]  # only the LABEL is shortened


def test_derive_run_label_legacy_require_grounding_bool_still_works():
    """Before the "grounding" string field existed, "require_grounding"
    (bool) was the only signal -- must still resolve correctly, not
    inferred (it's an explicit, unambiguous field, just an older one)."""
    on = derive_run_label("D", {"require_grounding": True})
    off = derive_run_label("D", {"require_grounding": False})
    assert on == {"run_label": "D-grounded", "fusion_mode": None, "grounding": "on", "grounding_inferred": False}
    assert off == {"run_label": "D-plain", "fusion_mode": None, "grounding": "off", "grounding_inferred": False}


def test_derive_run_label_historical_defaults_per_condition_when_everything_missing():
    assert derive_run_label("B", {})["run_label"] == "B-plain"
    assert derive_run_label("C", {"fusion_mode": "uniform"})["run_label"] == "C-uniform-grounded"
    assert derive_run_label("D", {})["run_label"] == "D-grounded"
    for condition in ("B", "C", "D"):
        record = {} if condition != "C" else {"fusion_mode": "uniform"}
        assert derive_run_label(condition, record)["grounding_inferred"] is True
