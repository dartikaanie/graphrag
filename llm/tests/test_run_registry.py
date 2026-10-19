"""results/run_registry.json logic on a temporary registry (no repo data)."""

import pytest

from llm import run_registry as rr


def _entry(run_id, status, output, **kw):
    e = {"run_id": run_id, "status": status, "kind": "generation", "condition": "C", "grounding": True,
         "alpha": 0.25, "n": 50, "seed": 42, "generator": "gpt-4o-mini", "judge_config": None, "commit": "abc1234",
         "created_at": "2026-10-20T00:00:00+00:00", "files": {"output": output}, "cost_usd": None,
         "cost_source": None, "notes": ""}
    e.update(kw)
    return e


@pytest.fixture
def reg_path(tmp_path, monkeypatch):
    path = tmp_path / "run_registry.json"
    monkeypatch.setattr(rr, "REGISTRY_PATH", path)
    return path


def test_register_validates_and_rejects_duplicates(reg_path):
    rr.register(_entry("r1", "official", "llm/c_graphrag/results/a.jsonl"))
    with pytest.raises(rr.RegistryError, match="already registered"):
        rr.register(_entry("r1", "official", "llm/c_graphrag/results/b.jsonl"))
    with pytest.raises(rr.RegistryError, match="status"):
        rr.register(_entry("r2", "pilot", "x.jsonl"))
    bad = _entry("r3", "official", "x.jsonl")
    del bad["seed"]
    with pytest.raises(rr.RegistryError, match="lacks fields"):
        rr.register(bad)


def test_default_view_is_official_only(reg_path):
    for i, s in enumerate(("official", "dev-stage2", "archived")):
        rr.register(_entry(f"r{i}", s, f"llm/c_graphrag/results/{s}.jsonl"))
    assert [r["status"] for r in rr.runs()] == ["official"]
    assert {r["status"] for r in rr.runs(include_archived=True)} == {"official", "archived"}
    assert {r["status"] for r in rr.runs(("official", "dev-stage2"))} == {"official", "dev-stage2"}


def test_assert_reportable_refuses_unregistered_and_archived(reg_path):
    rr.register(_entry("off", "official", "llm/c_graphrag/results/off.jsonl"))
    rr.register(_entry("arc", "archived", "llm/c_graphrag/results/arc.jsonl"))
    rr.register(_entry("s2", "dev-stage2", "llm/c_graphrag/results/s2.jsonl"))
    root = rr.REPO_ROOT
    assert [r["run_id"] for r in rr.assert_reportable([root / "llm/c_graphrag/results/off.jsonl"])] == ["off"]
    with pytest.raises(rr.RegistryError, match="status 'archived'"):
        rr.assert_reportable([root / "llm/c_graphrag/results/arc.jsonl"])
    assert rr.assert_reportable([root / "llm/c_graphrag/results/arc.jsonl"], include_archived=True)
    with pytest.raises(rr.RegistryError, match="not in run_registry.json"):
        rr.assert_reportable([root / "llm/c_graphrag/results/unknown.jsonl"])
    with pytest.raises(rr.RegistryError):
        rr.assert_reportable([root / "llm/c_graphrag/results/s2.jsonl"])
    assert rr.assert_reportable([root / "llm/c_graphrag/results/s2.jsonl"], allowed=("dev-stage2",))


def test_guard_cli_exits_with_status_2(reg_path):
    with pytest.raises(SystemExit) as e:
        rr.guard_cli(["nowhere.jsonl"], include_archived=False, log=lambda *_: None)
    assert e.value.code == 2


def test_deepinfra_spend_and_remaining_budget(reg_path):
    assert rr.deepinfra_spent() == 0 and rr.deepinfra_remaining() is None
    reg = rr.load()
    reg["budget"]["deepinfra_budget_usd"] = 6.0
    rr.save(reg)
    rr.record_deepinfra_spend("screening qwen3", 0.041)
    rr.record_deepinfra_spend("alpha stage 2", 1.2, run_id="x")
    assert rr.deepinfra_spent() == pytest.approx(1.241)
    assert rr.deepinfra_remaining() == pytest.approx(4.759)
