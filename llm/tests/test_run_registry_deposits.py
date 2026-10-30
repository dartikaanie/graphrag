"""Budget deposits: a top-up raises the budget and is not negative spend. Temporary registry only."""

import pytest

from llm import run_registry as rr


@pytest.fixture
def reg_path(tmp_path, monkeypatch):
    path = tmp_path / "run_registry.json"
    monkeypatch.setattr(rr, "REGISTRY_PATH", path)
    return path


def test_deposit_raises_the_budget_not_the_spend(reg_path):
    rr.set_budget(9.0)
    rr.record_deepinfra_spend("judging", 7.376118)
    rr.record_deepinfra_spend("estimate reconciliation", -1.106118)
    assert rr.deepinfra_remaining() == 2.73
    rr.record_deepinfra_deposit(5.0, "top-up", confirmed_by="author")
    reg = rr.load()
    assert rr.deepinfra_spent(reg) == 6.27                        # unchanged by the deposit
    assert rr.deepinfra_deposited(reg) == 5.0 and rr.deepinfra_budget_total(reg) == 14.0
    assert rr.deepinfra_remaining(reg) == 7.73
    assert reg["budget"]["deepinfra_budget_usd"] == 9.0           # the budget that was set is not rewritten
    assert reg["deepinfra_deposits"][0]["confirmed_by"] == "author"


def test_without_deposits_nothing_changes_and_a_deposit_must_be_positive(reg_path):
    assert rr.deepinfra_remaining() is None and rr.deepinfra_deposited() == 0
    rr.set_budget(9.0)
    rr.record_deepinfra_spend("x", 1.5)
    assert rr.deepinfra_remaining() == 7.5
    with pytest.raises(rr.RegistryError):
        rr.record_deepinfra_deposit(-1.0, "bad")
    with pytest.raises(rr.RegistryError):
        rr.record_deepinfra_deposit(0.0, "bad")
