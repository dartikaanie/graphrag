"""Tests for the dashboard's "Archive all results" button
(POST /api/settings/archive, GET /api/settings/archive-dest) -- must call
the SAME archive_results.archive() the CLI menu item uses, never a
reimplementation. Monkeypatches archive_results.REPO_ROOT/SOURCE_DIRS
against a throwaway tmp_path tree -- never touches the real repo.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import archive_results  # noqa: E402

from app.services import settings_service  # noqa: E402


def _make_fake_repo(tmp_path):
    repo_root = tmp_path / "fake_repo"
    (repo_root / "llm/a_pure_llm/results").mkdir(parents=True)
    (repo_root / "llm/a_pure_llm/results/condition_a_test.jsonl").write_text('{"question_id": 1}\n')
    return repo_root


def test_resolved_archive_dest_reads_env_var(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_ARCHIVE_DIR", "/tmp/somewhere")
    assert settings_service.resolved_archive_dest() == "/tmp/somewhere"


def test_run_archive_raises_without_dest(monkeypatch):
    monkeypatch.delenv("GRAPHRAG_ARCHIVE_DIR", raising=False)
    try:
        settings_service.run_archive(None, dry_run=True)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_run_archive_calls_the_same_archive_results_function(tmp_path, monkeypatch):
    repo_root = _make_fake_repo(tmp_path)
    monkeypatch.setattr(archive_results, "REPO_ROOT", repo_root)
    monkeypatch.setattr(archive_results, "SOURCE_DIRS", ["llm/a_pure_llm/results"])

    dest = tmp_path / "archive_dest"
    result = settings_service.run_archive(str(dest), dry_run=False)

    assert result["copied"] == 1
    assert result["dest"] == str(dest)
    assert result["mismatches"] == []
    assert (dest / "llm/a_pure_llm/results/condition_a_test.jsonl").exists()


def test_archive_endpoint_reports_copied_files_and_checksum_status(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    repo_root = _make_fake_repo(tmp_path)
    monkeypatch.setattr(archive_results, "REPO_ROOT", repo_root)
    monkeypatch.setattr(archive_results, "SOURCE_DIRS", ["llm/a_pure_llm/results"])

    dest = tmp_path / "archive_dest"
    client = TestClient(app)
    resp = client.post("/api/settings/archive", json={"dest": str(dest), "dry_run": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["copied"] == 1
    assert body["mismatches"] == []
    assert body["dest"] == str(dest)


def test_archive_endpoint_dry_run_copies_nothing(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    repo_root = _make_fake_repo(tmp_path)
    monkeypatch.setattr(archive_results, "REPO_ROOT", repo_root)
    monkeypatch.setattr(archive_results, "SOURCE_DIRS", ["llm/a_pure_llm/results"])

    dest = tmp_path / "archive_dest_dry"
    client = TestClient(app)
    resp = client.post("/api/settings/archive", json={"dest": str(dest), "dry_run": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    assert body["would_copy"] == 1
    assert not dest.exists()


def test_archive_endpoint_400_when_no_dest_available(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.delenv("GRAPHRAG_ARCHIVE_DIR", raising=False)
    client = TestClient(app)
    resp = client.post("/api/settings/archive", json={"dry_run": True})
    assert resp.status_code == 400
