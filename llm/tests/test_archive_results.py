"""Tests for archive_results.py (Step 5 -- data retention/audit trail).
Uses monkeypatched SOURCE_DIRS/REPO_ROOT against a throwaway tmp_path tree
-- never touches the real repo's actual results/logs."""

import hashlib

import archive_results


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _make_fake_repo(tmp_path):
    repo_root = tmp_path / "fake_repo"
    (repo_root / "llm/a_pure_llm/results").mkdir(parents=True)
    (repo_root / "llm/a_pure_llm/logs").mkdir(parents=True)
    (repo_root / "llm/a_pure_llm/results/condition_a_test.jsonl").write_text('{"question_id": 1}\n')
    (repo_root / "llm/a_pure_llm/logs/run_history.jsonl").write_text('{"run_started_at": "t1"}\n')
    return repo_root


def test_dry_run_copies_nothing(tmp_path, monkeypatch, capsys):
    repo_root = _make_fake_repo(tmp_path)
    monkeypatch.setattr(archive_results, "REPO_ROOT", repo_root)
    monkeypatch.setattr(archive_results, "SOURCE_DIRS", ["llm/a_pure_llm/results", "llm/a_pure_llm/logs"])

    dest = tmp_path / "archive_dest"
    result = archive_results.archive(dest, dry_run=True)

    assert result["dry_run"] is True
    assert result["would_copy"] == 2
    assert not dest.exists()  # dry-run creates nothing


def test_real_run_copies_files_preserving_structure_and_writes_checksums(tmp_path, monkeypatch):
    repo_root = _make_fake_repo(tmp_path)
    monkeypatch.setattr(archive_results, "REPO_ROOT", repo_root)
    monkeypatch.setattr(archive_results, "SOURCE_DIRS", ["llm/a_pure_llm/results", "llm/a_pure_llm/logs"])

    dest = tmp_path / "archive_dest"
    result = archive_results.archive(dest, dry_run=False)

    assert result["copied"] == 2
    assert result["mismatches"] == []

    copied_result = dest / "llm/a_pure_llm/results/condition_a_test.jsonl"
    copied_history = dest / "llm/a_pure_llm/logs/run_history.jsonl"
    assert copied_result.exists()
    assert copied_history.exists()
    assert copied_result.read_text() == '{"question_id": 1}\n'

    sums_path = dest / "SHA256SUMS"
    assert sums_path.exists()
    sums_content = sums_path.read_text()
    assert "llm/a_pure_llm/results/condition_a_test.jsonl" in sums_content
    assert _sha256(copied_result) in sums_content


def test_source_files_are_never_modified_or_deleted(tmp_path, monkeypatch):
    repo_root = _make_fake_repo(tmp_path)
    monkeypatch.setattr(archive_results, "REPO_ROOT", repo_root)
    monkeypatch.setattr(archive_results, "SOURCE_DIRS", ["llm/a_pure_llm/results", "llm/a_pure_llm/logs"])

    source_file = repo_root / "llm/a_pure_llm/results/condition_a_test.jsonl"
    original_content = source_file.read_text()
    original_mtime = source_file.stat().st_mtime

    archive_results.archive(tmp_path / "archive_dest", dry_run=False)

    assert source_file.exists()  # never deleted
    assert source_file.read_text() == original_content  # never modified
    assert source_file.stat().st_mtime == original_mtime


def test_checksum_mismatch_is_detected_if_dest_file_is_corrupted_after_copy(tmp_path, monkeypatch):
    """Simulates a corrupted copy (e.g. a disk error during transfer to
    an external drive) -- archive() must report it as a mismatch, not
    silently report success."""
    repo_root = _make_fake_repo(tmp_path)
    monkeypatch.setattr(archive_results, "REPO_ROOT", repo_root)
    monkeypatch.setattr(archive_results, "SOURCE_DIRS", ["llm/a_pure_llm/results"])

    dest = tmp_path / "archive_dest"

    original_sha256_of_file = archive_results.sha256_of_file
    call_count = {"n": 0}

    def flaky_sha256(path):
        call_count["n"] += 1
        # First call (source) returns the real hash; verification call
        # (after "copy") gets a corrupted/mismatched hash simulated here.
        if call_count["n"] > 1:
            return "0" * 64
        return original_sha256_of_file(path)

    monkeypatch.setattr(archive_results, "sha256_of_file", flaky_sha256)
    result = archive_results.archive(dest, dry_run=False)
    assert result["mismatches"] == ["llm/a_pure_llm/results/condition_a_test.jsonl"]
