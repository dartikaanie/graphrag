"""Tests for judge_clients.py's registry -- exactly one primary/secondary,
role-change logging, no secrets leaked by registry_public_view()."""

import json

import judge_clients


def test_validate_registry_passes_for_the_real_registry():
    judge_clients.validate_registry()  # must not raise


def test_validate_registry_raises_if_two_primaries(monkeypatch):
    bad = dict(judge_clients.JUDGE_REGISTRY)
    bad["extra_primary"] = judge_clients.JudgeConfig(
        judge_id="extra_primary", model="x", base_url=None, api_key_env="X",
        role="primary", precision="?", price_per_m_input=0, price_per_m_output=0, price_date="2026-01-01",
    )
    monkeypatch.setattr(judge_clients, "JUDGE_REGISTRY", bad)
    try:
        judge_clients.validate_registry()
        assert False, "expected ValueError"
    except ValueError as e:
        assert "primary" in str(e)


def test_validate_registry_raises_if_no_secondary(monkeypatch):
    bad = {k: v for k, v in judge_clients.JUDGE_REGISTRY.items() if v.role != "secondary"}
    monkeypatch.setattr(judge_clients, "JUDGE_REGISTRY", bad)
    try:
        judge_clients.validate_registry()
        assert False, "expected ValueError"
    except ValueError as e:
        assert "secondary" in str(e)


def test_registry_has_exactly_four_roles_covered_by_four_judges():
    roles = {c.role for c in judge_clients.JUDGE_REGISTRY.values()}
    assert roles == {"primary", "secondary", "fallback"}  # no exploratory registered yet, by design


def test_registry_public_view_never_includes_raw_api_key(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "sk-super-secret-value")
    view = judge_clients.registry_public_view()
    dumped = json.dumps(view)
    assert "sk-super-secret-value" not in dumped
    primary_entry = next(e for e in view if e["judge_id"] == "primary")
    assert primary_entry["api_key_configured"] is True


def test_registry_public_view_reports_key_not_configured(monkeypatch):
    monkeypatch.delenv("SOME_UNSET_KEY_ENV", raising=False)
    bad = {"x": judge_clients.JudgeConfig(
        judge_id="x", model="m", base_url=None, api_key_env="SOME_UNSET_KEY_ENV",
        role="exploratory", precision="?", price_per_m_input=0, price_per_m_output=0, price_date="2026-01-01",
    )}
    monkeypatch.setattr(judge_clients, "JUDGE_REGISTRY", bad)
    view = judge_clients.registry_public_view()
    assert view[0]["api_key_configured"] is False


def test_record_role_change_appends_only(tmp_path, monkeypatch):
    path = tmp_path / "judge_role_changes.jsonl"
    monkeypatch.setattr(judge_clients, "JUDGE_ROLE_CHANGES_PATH", path)

    judge_clients.record_role_change(None, "primary", "primary", "initial registration")
    judge_clients.record_role_change("primary", "fallback_promoted", "primary", "primary failed human-kappa validation")

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["new_judge_id"] == "primary"
    second = json.loads(lines[1])
    assert second["old_judge_id"] == "primary"
    assert second["reason"] == "primary failed human-kappa validation"


def test_primary_precision_is_labeled_fp8_not_bf16():
    """Regression guard for the 2026-10-05 correction: the registry must
    never again claim BF16 for a model DeepInfra only serves as Turbo/FP8."""
    primary = judge_clients.JUDGE_REGISTRY["primary"]
    assert "FP8" in primary.precision or "Turbo" in primary.precision
    assert "BF16" not in primary.precision.split("--")[0]
