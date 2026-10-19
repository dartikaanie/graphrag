"""Dashboard default view: no pre-official (archived) run, job or log is listed
unless the request asks for the archive explicitly (include_archive=true)."""

import pytest
from fastapi.testclient import TestClient

from app.archive import archive_paths
from app.main import app

client = TestClient(app)
HAS_ARCHIVE = archive_paths.archived_dir(archive_paths.REPO_ROOT / "llm" / "c_graphrag" / "results") is not None
needs_archive = pytest.mark.skipif(not HAS_ARCHIVE, reason="no pre-official archive present")


def test_history_lists_no_archived_run_by_default():
    items = client.get("/api/history", params={"page_size": 200}).json()["items"]
    assert not any(i.get("archived") for i in items)
    batches = client.get("/api/history/batches").json()["batches"]
    assert not any(run.get("archived") for b in batches for run in b.get("runs", []))


@needs_archive
def test_history_archive_toggle_lists_archived_runs_and_their_detail_opens():
    items = client.get("/api/history", params={"page_size": 200, "include_archive": "true"}).json()["items"]
    archived = [i for i in items if i.get("archived")]
    assert archived
    assert client.get(f"/api/history/{archived[0]['history_id']}").status_code == 200


def test_process_runs_and_logs_default_exclude_archive():
    assert not any(r.get("archived") for r in client.get("/api/process/runs").json()["runs"])
    assert not any(r.get("archived") for r in client.get("/api/process/logs").json()["logs"])


@needs_archive
def test_process_archive_toggle():
    runs = client.get("/api/process/runs", params={"include_archive": "true"}).json()["runs"]
    assert any(r["archived"] for r in runs)
    logs = client.get("/api/process/logs", params={"include_archive": "true"}).json()["logs"]
    archived = [log for log in logs if log["archived"]]
    assert archived and all(log["id"].startswith("archived~") for log in archived)
    assert client.get(f"/api/process/logs/{archived[0]['id']}").status_code == 200


def test_registry_default_view_is_official_only():
    body = client.get("/api/registry").json()
    assert all(r["status"] == "official" for r in body["runs"])
    assert body["statuses"] == ["official"]


@needs_archive
def test_archived_job_state_needs_the_toggle():
    runs_dir = archive_paths.archived(archive_paths.REPO_ROOT / "backend" / ".runs")
    job = next(runs_dir.glob("*.json")).stem
    assert client.get(f"/api/runs/{job}").status_code == 404
    r = client.get(f"/api/runs/{job}", params={"include_archive": "true"})
    assert r.status_code == 200 and r.json()["archived"] is True
