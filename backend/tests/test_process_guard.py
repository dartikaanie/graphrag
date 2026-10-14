"""Rules 1, 5, 7, 8 for the Process page: read-only API, blinding guard,
log allowlist / path traversal, Neo4j-unavailable behaviour. Fakes only --
no Neo4j, no network."""

import json
import os
import re

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.process import guard, live, logs, runs

client = TestClient(app)

ENDPOINTS = ["/api/process/steps", "/api/process/parameters", "/api/process/decisions", "/api/process/runs",
             "/api/process/logs", "/api/process/live"]

# keys that would carry item content (questions, answers, contexts, judge outputs)
CONTENT_KEYS = {"question", "question_title", "question_body", "answer", "accepted_answer", "candidate_answer",
                "generated_answer", "llm_answer", "retrieved_context", "context_items", "chunk_text", "claims",
                "judge_output", "raw_output", "judge_label", "human_label", "messages", "reasoning_text"}


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


@pytest.fixture(scope="module")
def all_bodies():
    return {path: client.get(path) for path in ENDPOINTS}


# ---------------------------------------------------------------- Rule 1: read-only

def test_only_get_routes_plus_the_live_refresh():
    methods = {}
    for r in app.routes:
        if getattr(r, "path", "").startswith("/api/process"):
            methods[r.path] = r.methods
    posts = [p for p, m in methods.items() if m & {"POST", "PUT", "PATCH", "DELETE"}]
    assert posts == ["/api/process/live/refresh"]


def test_live_queries_are_read_only():
    write = re.compile(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|CALL|LOAD\s+CSV|FOREACH)\b", re.I)
    for name, q in live.QUERIES.items():
        assert not write.search(q), name
        assert q.strip().upper().startswith("MATCH"), name


def test_process_package_has_no_write_or_llm_calls():
    from pathlib import Path
    pkg = Path(guard.__file__).parent
    text = "\n".join(p.read_text() for p in pkg.glob("*.py"))
    for forbidden in ("--apply", "--restore", "subprocess.Popen", "import openai", "import anthropic",
                      "get_llm_client", "write_transaction", "execute_write"):
        assert forbidden not in text, forbidden
    assert not re.search(r"^\s*(from|import)\s+\S*(engine_service|client_factory)", text, re.M)


# ---------------------------------------------------------------- Rule 5: blinding

@pytest.mark.parametrize("path", [
    guard.ANNOTATION_DIR / "calibration" / "annotator1.xlsx",
    guard.ANNOTATION_DIR / "calibration" / "export_manifest.json",
    guard.CALIBRATION_DIR / "calibration_runs.json",
    guard.ANNOTATION_KEY_DIR / "key.json",
    guard.STAGE2_DIR / "anything.jsonl",
    guard.REPO_ROOT / "llm" / "evaluation" / "results" / "jv21_calibration_primary.jsonl",
    guard.REPO_ROOT / "logs" / "confirmation_set.log",
])
def test_blocked_paths(path):
    assert guard.is_blocked_path(path)


def test_ordinary_log_is_not_blocked():
    assert not guard.is_blocked_path(guard.REPO_ROOT / "01_data_cleaning" / "logs" / "x_apply_logscore_trust.log")


def test_endpoints_never_return_content_keys(all_bodies):
    for path, resp in all_bodies.items():
        assert resp.status_code == 200, path
        assert CONTENT_KEYS.isdisjoint(set(_walk_keys(resp.json()))), path


def test_endpoints_never_return_annotation_material(all_bodies):
    """No annotation file name, key location or key hash in any response."""
    export = guard.ANNOTATION_DIR / "calibration" / "export_manifest.json"
    forbidden = ["annotator1.xlsx", "annotator2.xlsx", "graphrag_annotation_keys", "reference_adjudicated",
                 "calibration_runs.json"]
    if export.is_file():
        meta = json.loads(export.read_text())
        forbidden += [str(meta.get("key_sha256")), str(meta.get("key_file_name"))]
    for path, resp in all_bodies.items():
        for s in forbidden:
            if s and s != "None":
                assert s not in resp.text, (path, s)


def test_run_rows_are_metadata_only(tmp_path, monkeypatch):
    results = tmp_path / "llm" / "c_graphrag" / "results"
    results.mkdir(parents=True)
    manifest = {
        "config": {"condition": "C", "sample_split": "test", "alpha": 0.5, "require_grounding": True},
        "config_hash": "abc123", "status": "completed", "git": {"commit_sha": "6a42942deadbeef", "dirty": False},
        "started_at_utc": "2026-10-14T00:00:00+00:00",
        "question": "SECRET QUESTION TEXT", "answers": ["SECRET ANSWER"], "judge_output": {"label": "X"},
    }
    (results / "condition_c_x_n384.jsonl.manifest.json").write_text(json.dumps(manifest))
    (results / "condition_c_x_n384.jsonl").write_text(json.dumps({"question": "SECRET QUESTION TEXT"}))
    old = tmp_path / "llm" / "c_graphrag" / "results_old"
    old.mkdir()
    (old / "condition_c_old.jsonl.manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runs, "REPO_ROOT", tmp_path)

    rows = runs.list_runs()
    assert len(rows) == 1
    assert set(rows[0]) <= guard.RUN_METADATA_FIELDS
    assert rows[0]["split"] == "test" and rows[0]["commit"] == "6a42942"
    assert "SECRET" not in json.dumps(rows)


def test_check_run_metadata_rejects_extra_fields():
    with pytest.raises(guard.GuardViolation):
        guard.check_run_metadata({"run_id": "x", "question": "text"})


def test_stage2_outputs_are_hidden(tmp_path, monkeypatch):
    stage2 = tmp_path / "stage2"
    stage2.mkdir()
    (stage2 / "stage2_alpha0-5_SECRETNAME.jsonl").write_text("SECRET_STAGE2_CONTENT")
    monkeypatch.setattr(guard, "STAGE2_DIR", stage2)
    monkeypatch.setattr(guard, "BLOCKED_DIRS", guard.BLOCKED_DIRS[:-1] + (stage2,))
    status = guard.stage2_status(judge_frozen=False)
    assert status == {"status": "outputs exist (hidden until the confirmation set is annotated)", "outputs_exist": True}
    body = client.get("/api/process/steps").text
    assert "SECRETNAME" not in body and "SECRET_STAGE2_CONTENT" not in body
    assert "outputs exist (hidden" in body


def test_calibration_status_reads_counts_only(tmp_path, monkeypatch):
    ann = tmp_path / "annotation"
    (ann / "calibration").mkdir(parents=True)
    (ann / "calibration" / "export_manifest.json").write_text(json.dumps({
        "created_at_utc": "2026-10-13T13:56:00Z", "n_items": 60, "n_items_annotator2": 30,
        "key_sha256": "deadbeefcafe", "key_file_name": "calibration_key.json", "columns": ["question"],
    }))
    monkeypatch.setattr(guard, "ANNOTATION_DIR", ann)
    status = guard.calibration_status("- Judge version frozen: [ ]  commit: [ ]")
    assert status["status"] == "in progress" and status["judge_frozen"] is False
    assert "deadbeefcafe" not in json.dumps(status) and "calibration_key" not in json.dumps(status)
    assert guard.calibration_status("- Judge version frozen: judge-v2.2  commit: abc1234")["status"] == "done"


# ---------------------------------------------------------------- Rule 7: log allowlist

@pytest.fixture
def fake_logs(tmp_path, monkeypatch):
    kg = tmp_path / "01_data_cleaning" / "logs"
    kg.mkdir(parents=True)
    (kg / "20261014T085801Z_apply_logscore_trust.log").write_text("line1\nline2\nline3\n")
    (kg / "20260829T075916Z_build_knowledge_graph.log").write_text("built\n")
    (kg / "20261013T000000Z_calibration_judge.log").write_text("blocked by name\n")
    (kg / "notes.txt").write_text("wrong suffix\n")
    (kg / ".hidden.log").write_text("hidden\n")
    (kg / "sub").mkdir()
    (kg / "sub" / "nested.log").write_text("nested\n")
    outside = tmp_path / "outside.log"
    outside.write_text("OUTSIDE SECRET\n")
    os.symlink(outside, kg / "20261014T000000Z_link.log")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-should-never-be-read\n")
    old = tmp_path / "llm" / "c_graphrag" / "results_old"
    old.mkdir(parents=True)
    (old / "stage1_report_old.md").write_text("old\n")
    monkeypatch.setattr(logs, "REPO_ROOT", tmp_path)
    return kg


def test_only_allowlisted_logs_are_listed(fake_logs):
    names = {lg["name"] for lg in logs.list_logs()}
    assert names == {"20261014T085801Z_apply_logscore_trust.log", "20260829T075916Z_build_knowledge_graph.log"}


def test_logs_are_newest_first_and_mapped_to_steps(fake_logs):
    listed = logs.list_logs()
    assert listed[0]["name"].startswith("20261014T085801Z")
    assert listed[0]["step"] == "trust" and listed[1]["step"] == "kg_build"


@pytest.mark.parametrize("log_id", [
    "kg_logs/../../.env",
    "kg_logs/..",
    "kg_logs/sub/nested.log",
    "kg_logs/.hidden.log",
    "kg_logs/notes.txt",
    "kg_logs/20261014T000000Z_link.log",
    "kg_logs/20261013T000000Z_calibration_judge.log",
    "unknown_root/x.log",
    "kg_logs/missing.log",
    "../.env",
    "kg_logs\\..\\.env",
    "",
])
def test_resolve_log_rejects(fake_logs, log_id):
    with pytest.raises(logs.LogAccessError):
        logs.resolve_log(log_id)


def test_read_tail_and_caps(fake_logs):
    out = logs.read_tail("kg_logs/20261014T085801Z_apply_logscore_trust.log", lines=2)
    assert out["text"] == "line2\nline3" and out["lines_returned"] == 2
    big = fake_logs / "20261015T000000Z_build_knowledge_graph.log"
    big.write_text("".join(f"row {i} " + "x" * 100 + "\n" for i in range(10_000)))  # > 1 MB
    out = logs.read_tail(f"kg_logs/{big.name}", lines=10_000)
    assert out["lines_returned"] <= logs.MAX_TAIL_LINES
    assert out["truncated"] is True
    assert out["text"].endswith("row 9999 " + "x" * 100)


@pytest.mark.parametrize("url", [
    "/api/process/logs/kg_logs/..%2F..%2F.env",
    "/api/process/logs/kg_logs/%2E%2E",
    "/api/process/logs/nope/x.log",
    "/api/process/logs/kg_logs/20261014T000000Z_link.log",
])
def test_router_blocks_traversal(fake_logs, url):
    resp = client.get(url)
    assert resp.status_code == 404
    assert "OUTSIDE SECRET" not in resp.text and "sk-should-never-be-read" not in resp.text


def test_router_tail_is_capped():
    assert client.get("/api/process/logs/kg_logs/x.log?tail=100000").status_code == 422


# ---------------------------------------------------------------- Rule 8: Neo4j off

def test_live_refresh_reports_unavailable_when_neo4j_is_off(monkeypatch):
    live.clear_cache()

    def down(query):
        raise ConnectionRefusedError("bolt://localhost:7687")

    monkeypatch.setattr(live, "_default_runner", down)
    resp = client.post("/api/process/live/refresh")
    assert resp.status_code == 200
    body = resp.json()
    assert body["live"]["available"] is False and "unavailable" in body["live"]["message"]
    live.clear_cache()


def test_live_refresh_uses_fake_runner_and_caches():
    live.clear_cache()
    calls = []

    def fake(query):
        calls.append(query)
        if "trustScoreVersion" in query:
            return [{"version": "v3-logscore", "n": 3}]
        return [{"n": 7}]

    first = live.refresh(runner=fake)
    assert first["available"] is True
    assert first["values"]["nodes"]["Answer"] == 7 and first["values"]["edges"]["EMBED_SIM"] == 7
    assert first["values"]["answer_trust_versions"] == {"v3-logscore": 3}
    n = len(calls)
    live.refresh(runner=fake)
    assert len(calls) == n  # cached
    live.clear_cache()


def test_page_endpoints_do_not_touch_neo4j(monkeypatch):
    import app.db.neo4j_client as neo

    def boom(*a, **k):
        raise AssertionError("Neo4j must not be queried on page load")

    monkeypatch.setattr(neo, "run_query", boom)
    monkeypatch.setattr(neo, "get_driver", boom)
    for path in ENDPOINTS:
        assert client.get(path).status_code == 200, path
