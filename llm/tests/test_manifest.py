"""Tests for llm/manifest.py -- the shared run-manifest writer (Step 5).
No LLM calls; only file I/O + a `git rev-parse`/`git status` subprocess
call against this real repo (read-only)."""

import json

import pytest

from llm.manifest import (
    MixedConfigHashError,
    assert_single_config_hash,
    compute_config_hash,
    sha256_of_file,
    write_manifest,
)


def test_compute_config_hash_is_deterministic_and_order_independent():
    a = compute_config_hash({"condition": "C", "seed": 42, "top_k": 5})
    b = compute_config_hash({"top_k": 5, "seed": 42, "condition": "C"})
    assert a == b
    assert len(a) == 10


def test_compute_config_hash_differs_for_different_configs():
    a = compute_config_hash({"condition": "B", "require_grounding": True})
    b = compute_config_hash({"condition": "B", "require_grounding": False})
    assert a != b


def test_assert_single_config_hash_returns_the_shared_hash():
    records = [{"config_hash": "abc123"}, {"config_hash": "abc123"}]
    assert assert_single_config_hash(records, "test.jsonl") == "abc123"


def test_assert_single_config_hash_none_for_legacy_records_without_the_field():
    records = [{"question_id": 1}, {"question_id": 2}]
    assert assert_single_config_hash(records, "test.jsonl") is None


def test_assert_single_config_hash_raises_on_a_deliberately_mixed_file():
    """The exact scenario the 2026-10-05 pilot incident fix must prevent:
    a resumed/re-run file ending up with records from two DIFFERENT
    configs (e.g. an old prompt_version's rows plus a new run's rows)."""
    records = [
        {"question_id": 1, "config_hash": "aaaa111111"},
        {"question_id": 2, "config_hash": "aaaa111111"},
        {"question_id": 3, "config_hash": "bbbb222222"},
    ]
    with pytest.raises(MixedConfigHashError):
        assert_single_config_hash(records, "condition_c_mixed.jsonl")


def test_sha256_of_file_matches_known_content(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("hello world")
    import hashlib
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert sha256_of_file(path) == expected


def test_sha256_of_file_none_for_missing_file(tmp_path):
    assert sha256_of_file(tmp_path / "does_not_exist.jsonl") is None


def test_write_manifest_creates_file_next_to_output(tmp_path):
    output_path = tmp_path / "condition_c_test.jsonl"
    output_path.write_text('{"question_id": 1}\n')

    manifest_path = write_manifest(
        output_path,
        run_label="C-uniform-grounded",
        config={"seed": 42, "n_sample": 10},
        started_at_utc="2026-10-05T00:00:00Z",
        finished_at_utc="2026-10-05T00:01:00Z",
        item_counts={"attempted": 1, "succeeded": 1, "failed": 0},
        prompt_version="v3",
        total_tokens=100,
    )

    assert manifest_path == output_path.with_name(output_path.name + ".manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_label"] == "C-uniform-grounded"
    assert manifest["config"] == {"seed": 42, "n_sample": 10}
    assert manifest["prompt_version"] == "v3"
    assert manifest["item_counts"] == {"attempted": 1, "succeeded": 1, "failed": 0}
    assert manifest["total_tokens"] == 100
    assert manifest["output_files"][str(output_path)] == sha256_of_file(output_path)
    assert "python_version" in manifest
    assert "package_versions" in manifest
    assert "git" in manifest and "commit_sha" in manifest["git"]
    assert manifest["status"] == "completed"


def test_write_manifest_status_defaults_to_completed_but_is_overridable(tmp_path):
    output_path = tmp_path / "out.jsonl"
    output_path.write_text("a\n")
    manifest_path = write_manifest(
        output_path, run_label="D-plain", status="failed", config={}, started_at_utc="t1", finished_at_utc="t2",
        item_counts={"attempted": 1, "succeeded": 0, "failed": 1},
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"


def test_write_manifest_includes_extra_output_files(tmp_path):
    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"
    output_path.write_text("a\n")
    failures_path.write_text("b\n")

    manifest_path = write_manifest(
        output_path, run_label="B-plain", config={}, started_at_utc="t1", finished_at_utc="t2",
        item_counts={"attempted": 1, "succeeded": 1, "failed": 0},
        extra_output_files=[failures_path],
    )
    manifest = json.loads(manifest_path.read_text())
    assert str(output_path) in manifest["output_files"]
    assert str(failures_path) in manifest["output_files"]
    assert manifest["output_files"][str(failures_path)] == sha256_of_file(failures_path)


def test_write_manifest_output_file_missing_gives_null_checksum_not_crash(tmp_path):
    output_path = tmp_path / "never_written.jsonl"
    manifest_path = write_manifest(
        output_path, run_label=None, config={}, started_at_utc="t1", finished_at_utc="t2",
        item_counts={"attempted": 0, "succeeded": 0, "failed": 0},
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["output_files"][str(output_path)] is None


def test_write_manifest_rerun_overwrites_manifest_for_same_output_path(tmp_path):
    """The manifest describes the CURRENT state of one output file -- a
    second write_manifest() call for the same output_path (e.g. after a
    resumed run finishes) legitimately overwrites the manifest (not the
    output DATA, which is a separate, already-enforced guarantee)."""
    output_path = tmp_path / "out.jsonl"
    output_path.write_text("a\n")
    write_manifest(output_path, run_label=None, config={}, started_at_utc="t1", finished_at_utc="t2",
                    item_counts={"attempted": 1, "succeeded": 1, "failed": 0})

    output_path.write_text("a\nb\n")  # simulate a resumed run appending more data
    manifest_path = write_manifest(output_path, run_label=None, config={}, started_at_utc="t1", finished_at_utc="t3",
                                    item_counts={"attempted": 2, "succeeded": 2, "failed": 0})
    manifest = json.loads(manifest_path.read_text())
    assert manifest["item_counts"]["attempted"] == 2
    assert manifest["finished_at_utc"] == "t3"
