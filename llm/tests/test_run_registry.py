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


def _gen_file(tmp_path, cfg, n):
    import json
    p = tmp_path / "condition_c_x.jsonl"
    p.write_text("".join(json.dumps({"question_id": i, "config": cfg, "config_hash": "h",
                                     "usage_full": {"prompt_tokens": 1000, "completion_tokens": 300},
                                     "answer_started_at_utc": "2026-10-20T00:00:00.000+00:00",
                                     "answer_finished_at_utc": "2026-10-20T00:00:05.000+00:00",
                                     "end_to_end_latency_sec": 5.0}) + "\n"
                         for i in range(n)))
    return p


STAGE2_CFG = {"condition": "C", "provider": "openai", "model": "gpt-4o-mini", "seed": 42, "oversample_pool": 1536,
              "sample_split": "dev", "n_sample": 50, "top_k": 5, "leakage": "dup+neardup0.95",
              "c_retrieval_version": "v3", "n_anchor": 3, "n_semantic_expansion": 3, "expansion_require_answer": True,
              "fusion_mode": "trust_weighted", "alpha": 0.25, "require_grounding": True}


def test_generation_entry_checks_the_frozen_config(tmp_path, reg_path):
    e = rr.generation_entry(_gen_file(tmp_path, STAGE2_CFG, 50), "dev-stage2")
    assert e["status"] == "dev-stage2" and e["alpha"] == 0.25 and e["n"] == 50
    assert e["cost_usd"] == pytest.approx(50 * (1000 * 0.15 + 300 * 0.60) / 1e6)
    assert e["frozen_params_hash"]
    with pytest.raises(rr.RegistryError, match="leakage"):
        rr.generation_entry(_gen_file(tmp_path, {**STAGE2_CFG, "leakage": "thread"}, 50), "dev-stage2")
    with pytest.raises(rr.RegistryError, match="condition/alpha"):
        rr.generation_entry(_gen_file(tmp_path, {**STAGE2_CFG, "alpha": 0.5}, 50), "dev-stage2")
    with pytest.raises(rr.RegistryError, match="sample_split"):  # a dev run cannot be registered as official
        rr.generation_entry(_gen_file(tmp_path, STAGE2_CFG, 50), "official")
    official = {k: v for k, v in STAGE2_CFG.items() if k != "sample_split"} | {"n_sample": 384}
    assert rr.generation_entry(_gen_file(tmp_path, official, 384), "official")["status"] == "official"


def test_runs_without_r3_timing_are_refused(tmp_path, reg_path):
    import json
    p = _gen_file(tmp_path, STAGE2_CFG, 50)
    lines = [json.loads(x) for x in p.read_text().splitlines()]
    del lines[3]["end_to_end_latency_sec"]
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    with pytest.raises(rr.RegistryError, match="end-to-end timing"):
        rr.generation_entry(p, "dev-stage2")
