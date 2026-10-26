"""Resume of a stopped judging run (2026-10-26 pre-run check), offline: fake clients, no API call.

Unit of resume: one (item, judge model, prompt version) call. Usable records go to <out>.jsonl (the cache); failed or
unparseable calls go to <out>_failures.jsonl and are therefore retried on resume. The kill is a BaseException raised
by the fake client after N calls (nothing in run_tasks catches it)."""
import json
import threading
from types import SimpleNamespace

import pytest

import judge_panel as jp
from llm import run_registry as rr
from llm.common.resume import repair_torn_tail
from judge_panel_config import PANEL_JUDGES
from test_judge_panel import GOOD

ITEMS = [{"item_id": f"run-a:{i}", "run_id": "run-a", "question_id": i, "condition": "A", "run_label": "A",
          "config_hash": "h", "source_file": "a.jsonl", "title": "t", "tags": "", "ground_truth_answer_html": "ref",
          "candidate": "c", "retrieved_context": []} for i in range(1, 41)]   # 40 items x 3 judges = 120 calls


class Killed(BaseException):
    pass


class KillingClient:
    """Always answers GOOD; raises Killed once `limit` calls have been made across all judges."""
    lock, count = threading.Lock(), {"n": 0}

    def __init__(self, shared, limit):
        self.shared, self.limit = shared, limit
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        with self.lock:
            if self.limit is not None and self.shared["n"] >= self.limit:
                raise Killed
            self.shared["n"] += 1
        msg = SimpleNamespace(model_dump=lambda: {"content": GOOD})
        usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 1000, "completion_tokens": 100})
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage,
                               model=kw["model"], id="r")


@pytest.fixture(autouse=True)
def _text(monkeypatch):
    monkeypatch.setattr("_judge_common.html_to_text", lambda s, *a, **k: s or "")


def _plan(out):
    done = {jp.cache_key(r) for r in jp.read_jsonl(out)}
    return jp.plan_tasks(ITEMS, {i["question_id"]: "body" for i in ITEMS}, done)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "REGISTRY_PATH", tmp_path / "run_registry.json")
    rr.save(rr.load(), None)
    return tmp_path / "run_registry.json"


def _run(out, limit=None, finish=True):
    """One invocation as main() does it: repair torn tails, read the cache, reconcile the ledger from the stored
    records, call the missing judges, reconcile again at the end (only if the run finished)."""
    repair_torn_tail(out, lambda *a: None)
    repair_torn_tail(jp.failures_path_of(out), lambda *a: None)
    tasks, prepared, n_cached = _plan(out)
    jp.reconcile_ledger(out, "dev-stage2", lambda *a: None)
    shared = {"n": 0}
    clients = {j.judge_id: KillingClient(shared, limit) for j in PANEL_JUDGES}
    try:
        jp.run_tasks(tasks, prepared, clients, out, concurrency=1, sleep=lambda s: None)
    except Killed:
        return len(tasks), n_cached, shared["n"]
    if finish:
        jp.reconcile_ledger(out, "dev-stage2", lambda *a: None)
    return len(tasks), n_cached, shared["n"]


def _ledger(out):
    return round(sum(e["cost_usd"] for e in rr.load()["deepinfra_spend"] if e.get("file") == out.name), 6)


def _stored_cost(out):
    return round(sum(r["estimated_cost_usd"] for r in jp.read_jsonl(out) + jp.read_jsonl(jp.failures_path_of(out))), 6)


def test_killed_after_n_calls_and_resumed_gives_every_call_exactly_once(tmp_path):
    out = tmp_path / "p.jsonl"
    n_tasks, cached, calls = _run(out, limit=50)
    assert (n_tasks, cached, calls) == (120, 0, 50)
    keys = [jp.cache_key(r) for r in jp.read_jsonl(out)]
    assert len(keys) == 50 and len(set(keys)) == 50            # the 50 paid calls are on disk, once each
    n_tasks, cached, calls = _run(out)
    assert (n_tasks, cached, calls) == (70, 50, 70)            # only the missing 70 are called; none re-paid
    recs = jp.read_jsonl(out)
    assert len(recs) == 120 and len({jp.cache_key(r) for r in recs}) == 120
    assert {r["item_id"] for r in recs} == {i["item_id"] for i in ITEMS}
    assert _run(out)[0] == 0                                    # a finished run resumed again is a no-op


def test_a_failed_call_is_not_cached_and_is_retried(tmp_path):
    out = tmp_path / "p.jsonl"
    _run(out)
    recs = jp.read_jsonl(out)
    out.write_text("".join(json.dumps(r) + "\n" for r in recs[:-1]))      # one record "failed": not in the cache
    n_tasks, cached, _ = _run(out)
    assert (n_tasks, cached) == (1, 119)


def test_a_torn_last_line_no_longer_crashes_the_resume(tmp_path):
    out = tmp_path / "p.jsonl"
    _run(out, limit=5, finish=False)
    out.write_text(out.read_text()[:-30])                       # died in the middle of the 5th record
    assert len(jp.read_jsonl(out)) == 4                         # the reader skips the torn tail ...
    logs = []
    assert repair_torn_tail(out, logs.append)["removed_bytes"] > 0 and logs   # ... the start cuts it off and logs it
    assert all(line.endswith("\n") for line in out.open())
    _run(out)
    recs = jp.read_jsonl(out)
    assert len(recs) == 120 and len({jp.cache_key(r) for r in recs}) == 120 and out.read_text().count("\n") == 120


def test_corruption_in_the_middle_of_the_file_still_raises(tmp_path):
    out = tmp_path / "p.jsonl"
    _run(out, limit=5, finish=False)
    lines = out.read_text().splitlines(True)
    lines[1] = "{broken\n"
    out.write_text("".join(lines))
    with pytest.raises(json.JSONDecodeError):
        jp.read_jsonl(out)


def test_ledger_is_booked_from_stored_records_also_for_a_killed_run_and_never_twice(tmp_path, registry):
    out = tmp_path / "p.jsonl"
    _run(out, limit=50, finish=False)                           # killed: 50 paid records, no ledger entry yet
    assert _ledger(out) == 0 and _stored_cost(out) > 0
    killed_cost = _stored_cost(out)
    _run(out, finish=False)                                     # next start books the killed run's records first;
    assert _ledger(out) == killed_cost                          # this run finished but did not reconcile at its end
    assert jp.reconcile_ledger(out, "dev-stage2", lambda *a: None) > 0      # so the remainder is booked now ...
    assert jp.reconcile_ledger(out, "dev-stage2", lambda *a: None) == 0.0   # ... and a second call books nothing
    _run(out)                                                   # (a finished run resumed again books nothing new)
    assert _ledger(out) == _stored_cost(out) and _stored_cost(out) > killed_cost
    entries = [e for e in rr.load()["deepinfra_spend"] if e.get("file") == out.name]
    assert len(entries) == 2 and entries[0]["cost_usd"] == killed_cost    # killed run, then the remainder: once each


def test_a_failed_calls_cost_is_booked_too_and_legacy_entries_are_left_alone(tmp_path, registry):
    out = tmp_path / "p.jsonl"
    _run(out, limit=3, finish=False)
    jp.failures_path_of(out).write_text(json.dumps({"item_id": "x", "estimated_cost_usd": 0.01}) + "\n")
    jp.reconcile_ledger(out, "dev-stage2", lambda *a: None)
    assert _ledger(out) == _stored_cost(out)
    legacy = tmp_path / "old.jsonl"
    legacy.write_text(json.dumps({"item_id": "y", "estimated_cost_usd": 0.5}) + "\n")
    rr.record_deepinfra_spend(f"dev-stage2 judging {legacy.name}", 0.2)       # untagged, as before 2026-10-26
    assert jp.reconcile_ledger(legacy, "dev-stage2", lambda *a: None) == 0.0
    assert not any(e.get("file") == legacy.name for e in rr.load()["deepinfra_spend"])


def test_end_to_end_kill_resume_gives_every_call_once_and_the_cost_once(tmp_path, registry):
    out = tmp_path / "p.jsonl"
    _run(out, limit=37, finish=False)
    out.write_text(out.read_text()[:-25])                       # and the kill tore the last line
    _run(out, limit=80, finish=False)                           # a second kill
    _run(out)
    recs = jp.read_jsonl(out)
    assert len(recs) == 120 and len({jp.cache_key(r) for r in recs}) == 120
    assert _ledger(out) == _stored_cost(out) == round(sum(r["estimated_cost_usd"] for r in recs), 6)


def test_main_reconciles_the_ledger_at_start_and_register_judge_run_uses_it():
    import inspect
    src = inspect.getsource(jp)
    assert src.count("record_deepinfra_spend(") == 1                      # only inside reconcile_ledger
    assert src.index("record_deepinfra_spend(") < src.index("def register_judge_run")
    main = src[src.index("def main("):]
    assert "reconcile_ledger(out, registry_gate" in main and "repair_torn_tail(out" in main
