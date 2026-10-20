"""GET /api/report (read-only, Amendment 14 draft step 9): real view lists only official runs; mock view is labelled."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_real_report_has_only_official_runs_and_is_not_mock():
    body = client.get("/api/report").json()
    assert body["meta"]["mock"] is False
    assert all(r["status"] == "official" for r in body["runs"])
    assert body["configuration"]["table_hash"] and body["configuration"]["frozen_parameters"]
    assert body["limitations"]["wording"] == "panel-detected hallucination rate"


def test_mock_report_items_and_export_are_labelled():
    body = client.get("/api/report", params={"mock": "true"}).json()
    assert body["meta"]["mock"] is True and [r["label"] for r in body["runs"]] == ["R1", "R3", "R5", "R7", "R9"]
    items = client.get("/api/report/items", params={"run_id": "MOCK-R5", "mock": "true"}).json()["items"]
    item = client.get("/api/report/item", params={"run_id": "MOCK-R5", "question_id": items[0]["question_id"],
                                                  "mock": "true"}).json()
    assert item["mock"] is True and set(item["judges"]) == {"llama", "deepseek", "qwen3-235b"}
    assert "MOCK DATA" in client.get("/api/report/export.md", params={"mock": "true"}).text
    assert client.get("/api/report/items", params={"run_id": "nope"}).status_code == 404
