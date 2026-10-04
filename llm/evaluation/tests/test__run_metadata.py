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
    uniform ablation arms."""
    trust_weighted = resolve_run_metadata(
        str(REPO_ROOT / "llm/c_graphrag/results/condition_c_openai_gpt-4o-mini_n10_seed42_fw0-7-0-3.jsonl")
    )
    uniform = resolve_run_metadata(
        str(REPO_ROOT / "llm/c_graphrag/results/condition_c_openai_gpt-4o-mini_n10_seed42_uniform.jsonl")
    )

    assert trust_weighted["fusion_mode"] == "trust_weighted"
    assert trust_weighted["run_label"] == "C-trust_weighted"
    assert uniform["fusion_mode"] == "uniform"
    assert uniform["run_label"] == "C-uniform"
    assert trust_weighted["run_label"] != uniform["run_label"]
    assert trust_weighted["run_id"] != uniform["run_id"]


def test_resolve_run_metadata_known_run_id_matches_dashboard_hash():
    """Pinned regression: this exact run_id is reproducible from the
    SAME (condition, run_started_at, output_path) formula the dashboard's
    history_service.py::_history_id() uses -- if this ever breaks, judge
    output run_ids will stop lining up with the dashboard's History page."""
    uniform = resolve_run_metadata(
        str(REPO_ROOT / "llm/c_graphrag/results/condition_c_openai_gpt-4o-mini_n10_seed42_uniform.jsonl")
    )
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
    assert d["run_label"] == "D"
    assert d["retrieval_version"] == "v3"
