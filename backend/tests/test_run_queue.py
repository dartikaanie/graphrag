"""Phase 4 -- the memory-safety queueing machinery in engine_service.py.
No LLM calls, no real Neo4j/DuckDB/FAISS -- RUNNERS entries are stubbed
with fast, sleeping functions so the tests exercise the real locking code
(_HEAVY_RUN_LOCK, start_run(), the queue bookkeeping) without touching any
actual condition logic.
"""

import threading
import time

from app.services import engine_service, run_registry


def _make_ordering_stub(name: str, order: list, order_lock: threading.Lock, sleep_s: float = 0.05):
    def stub(run_id, params):
        with order_lock:
            order.append(name)
        time.sleep(sleep_s)
    return stub


def test_heavy_queue_ordering_is_fifo(monkeypatch):
    """B starts first and holds the lock; C and D queue up behind it in the
    order they arrived -- the stub records execution order, which must
    match arrival order (B, then C, then D)."""
    order: list[str] = []
    order_lock = threading.Lock()

    monkeypatch.setitem(engine_service.RUNNERS, "B", _make_ordering_stub("B", order, order_lock))
    monkeypatch.setitem(engine_service.RUNNERS, "C", _make_ordering_stub("C", order, order_lock))
    monkeypatch.setitem(engine_service.RUNNERS, "D", _make_ordering_stub("D", order, order_lock))

    run_b = run_registry.create_run("B", "batch", {})
    run_c = run_registry.create_run("C", "batch", {})
    run_d = run_registry.create_run("D", "batch", {})

    t_b = threading.Thread(target=engine_service.start_run, args=(run_b, "B", {}))
    t_b.start()
    time.sleep(0.05)  # let B acquire _HEAVY_RUN_LOCK before C/D even try
    t_c = threading.Thread(target=engine_service.start_run, args=(run_c, "C", {}))
    t_c.start()
    time.sleep(0.02)  # let C enter the queue (and start blocking on the lock) before D
    t_d = threading.Thread(target=engine_service.start_run, args=(run_d, "D", {}))
    t_d.start()

    for t in (t_b, t_c, t_d):
        t.join(timeout=5)

    assert order == ["B", "C", "D"]


def test_only_one_heavy_run_executes_at_a_time(monkeypatch):
    concurrent = 0
    max_concurrent = 0
    lock = threading.Lock()

    def make_stub(sleep_s: float):
        def stub(run_id, params):
            nonlocal concurrent, max_concurrent
            with lock:
                concurrent += 1
                max_concurrent = max(max_concurrent, concurrent)
            time.sleep(sleep_s)
            with lock:
                concurrent -= 1
        return stub

    monkeypatch.setitem(engine_service.RUNNERS, "B", make_stub(0.05))
    monkeypatch.setitem(engine_service.RUNNERS, "C", make_stub(0.05))
    monkeypatch.setitem(engine_service.RUNNERS, "D", make_stub(0.05))

    run_ids = {c: run_registry.create_run(c, "batch", {}) for c in ("B", "C", "D")}
    threads = [
        threading.Thread(target=engine_service.start_run, args=(rid, c, {}))
        for c, rid in run_ids.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert max_concurrent == 1


def test_condition_a_also_queues_behind_a_heavy_run(monkeypatch):
    """Condition A shares _HEAVY_RUN_LOCK with B/C/D (it still runs a DuckDB
    sampling query and must not overlap C/D's ~4GB FAISS footprint on an
    8GB machine) -- it must show "queued" while B holds the lock, and only
    start after B finishes."""
    order: list[str] = []
    order_lock = threading.Lock()

    monkeypatch.setitem(engine_service.RUNNERS, "B", _make_ordering_stub("B", order, order_lock, sleep_s=0.15))
    monkeypatch.setitem(engine_service.RUNNERS, "A", _make_ordering_stub("A", order, order_lock, sleep_s=0.02))

    run_b = run_registry.create_run("B", "batch", {})
    run_a = run_registry.create_run("A", "batch", {})

    t_b = threading.Thread(target=engine_service.start_run, args=(run_b, "B", {}))
    t_b.start()
    time.sleep(0.02)  # let B acquire _HEAVY_RUN_LOCK first

    t_a = threading.Thread(target=engine_service.start_run, args=(run_a, "A", {}))
    t_a.start()
    time.sleep(0.05)  # give A's thread time to enter the queue

    assert run_registry.get_run(run_a)["status"] == "queued"

    t_b.join(timeout=5)
    t_a.join(timeout=5)

    assert order == ["B", "A"]


def test_a_b_c_d_all_serialize_one_at_a_time(monkeypatch):
    concurrent = 0
    max_concurrent = 0
    lock = threading.Lock()

    def make_stub(sleep_s: float):
        def stub(run_id, params):
            nonlocal concurrent, max_concurrent
            with lock:
                concurrent += 1
                max_concurrent = max(max_concurrent, concurrent)
            time.sleep(sleep_s)
            with lock:
                concurrent -= 1
        return stub

    for c in ("A", "B", "C", "D"):
        monkeypatch.setitem(engine_service.RUNNERS, c, make_stub(0.05))

    run_ids = {c: run_registry.create_run(c, "batch", {}) for c in ("A", "B", "C", "D")}
    threads = [
        threading.Thread(target=engine_service.start_run, args=(rid, c, {}))
        for c, rid in run_ids.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert max_concurrent == 1


def test_cancelling_a_queued_run_skips_its_runner(monkeypatch):
    calls: list[str] = []

    def blocking_b(run_id, params):
        calls.append(run_id)
        time.sleep(0.3)

    def should_never_run(run_id, params):
        calls.append(run_id)

    monkeypatch.setitem(engine_service.RUNNERS, "B", blocking_b)
    monkeypatch.setitem(engine_service.RUNNERS, "C", should_never_run)

    run_b = run_registry.create_run("B", "batch", {})
    run_c = run_registry.create_run("C", "batch", {})

    t_b = threading.Thread(target=engine_service.start_run, args=(run_b, "B", {}))
    t_b.start()
    time.sleep(0.05)  # let B acquire the lock

    t_c = threading.Thread(target=engine_service.start_run, args=(run_c, "C", {}))
    t_c.start()
    time.sleep(0.05)  # let C enter the queue

    assert run_registry.get_run(run_c)["status"] == "queued"
    run_registry.request_cancel(run_c)

    t_b.join(timeout=5)
    t_c.join(timeout=5)

    assert run_c not in calls  # C's runner was never invoked
    assert run_registry.get_run(run_c)["status"] == "cancelled"


def test_heavy_lock_released_even_if_runner_raises(monkeypatch):
    """If one queued run's runner raises (an unhandled exception inside
    run_condition_x itself would be a bug there, but the QUEUE must not
    deadlock because of it) -- the next queued run still gets to execute.
    This is the backend-side half of "Run All continuing after one
    failure": execution-level independence (each condition already has its
    own thread/queue slot; failing one, not the others)."""
    executed: list[str] = []

    def raising_b(run_id, params):
        raise RuntimeError("boom")

    def fine_c(run_id, params):
        executed.append("C")

    monkeypatch.setitem(engine_service.RUNNERS, "B", raising_b)
    monkeypatch.setitem(engine_service.RUNNERS, "C", fine_c)

    run_b = run_registry.create_run("B", "batch", {})
    run_c = run_registry.create_run("C", "batch", {})

    t_b = threading.Thread(target=engine_service.start_run, args=(run_b, "B", {}))
    # start_run() doesn't catch runner exceptions itself (run_condition_x()
    # does, internally) -- a raising stub simulates a runner that escapes
    # cleanly (thread just dies), proving the LOCK is still released via
    # `finally` regardless.
    t_b.start()
    t_b.join(timeout=5)

    t_c = threading.Thread(target=engine_service.start_run, args=(run_c, "C", {}))
    t_c.start()
    t_c.join(timeout=5)

    assert executed == ["C"]
    assert not engine_service._HEAVY_RUN_LOCK.locked()


def test_faiss_cache_loaded_once_for_c_then_d(monkeypatch):
    """C and D share ONE _faiss_cache() call for the same kg_workspace_dir
    -- the underlying loader must be invoked exactly once, not once per
    condition."""
    call_count = {"n": 0}

    class StubCondModule:
        @staticmethod
        def load_faiss_cache(path, logger):
            call_count["n"] += 1
            return ("stub-index", "stub-ids", "stub-embeddings", {"stub": 0})

    engine_service._faiss_cache.cache_clear()
    monkeypatch.setattr(engine_service, "_condition_c_module", lambda: StubCondModule)

    result_for_c = engine_service._faiss_cache("/fake/kg_workspace")
    result_for_d = engine_service._faiss_cache("/fake/kg_workspace")

    assert call_count["n"] == 1
    assert result_for_c == result_for_d

    engine_service._faiss_cache.cache_clear()


def test_release_faiss_cache_forces_reload(monkeypatch):
    call_count = {"n": 0}

    class StubCondModule:
        @staticmethod
        def load_faiss_cache(path, logger):
            call_count["n"] += 1
            return ("stub-index", "stub-ids", "stub-embeddings", {"stub": 0})

    engine_service._faiss_cache.cache_clear()
    monkeypatch.setattr(engine_service, "_condition_c_module", lambda: StubCondModule)

    engine_service._faiss_cache("/fake/kg_workspace")
    assert call_count["n"] == 1

    result = engine_service.release_faiss_cache()
    assert result["released"] is True

    engine_service._faiss_cache("/fake/kg_workspace")
    assert call_count["n"] == 2  # reloaded after release

    engine_service._faiss_cache.cache_clear()


def test_judge_queue_bounded_by_semaphore(monkeypatch):
    """At most _JUDGE_CONCURRENCY_LIMIT (2) judge runs execute at once --
    a 3rd must show status "queued" until a slot frees up."""
    concurrent = 0
    max_concurrent = 0
    lock = threading.Lock()

    def stub_judge_batch(run_id, params):
        nonlocal concurrent, max_concurrent
        with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        time.sleep(0.08)
        with lock:
            concurrent -= 1
        run_registry.update_run(run_id, status="completed")

    monkeypatch.setattr(engine_service, "run_judge_batch_dashboard", stub_judge_batch)
    monkeypatch.setattr(engine_service.settings_service, "apply_to_environment", lambda: None)
    monkeypatch.setattr(
        engine_service.settings_service, "get_raw_settings", lambda: {"judges_wait_for_heavy_run": False}
    )

    run_ids = [run_registry.create_run("JUDGE", "batch", {}) for _ in range(3)]
    threads = [threading.Thread(target=engine_service.start_judge_run, args=(rid, {})) for rid in run_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert max_concurrent <= engine_service._JUDGE_CONCURRENCY_LIMIT


def test_judge_waits_for_heavy_run_when_toggle_on(monkeypatch):
    """With "judges_wait_for_heavy_run" ON (the default), a judge run must
    not execute while B holds _HEAVY_RUN_LOCK -- it should stay "queued"
    until B finishes, THEN run."""
    order: list[str] = []
    order_lock = threading.Lock()

    def slow_b(run_id, params):
        with order_lock:
            order.append("B-start")
        time.sleep(0.15)
        with order_lock:
            order.append("B-end")

    def judge_stub(run_id, params):
        with order_lock:
            order.append("JUDGE")
        run_registry.update_run(run_id, status="completed")

    monkeypatch.setitem(engine_service.RUNNERS, "B", slow_b)
    monkeypatch.setattr(engine_service, "run_judge_batch_dashboard", judge_stub)
    monkeypatch.setattr(engine_service.settings_service, "apply_to_environment", lambda: None)
    monkeypatch.setattr(
        engine_service.settings_service, "get_raw_settings", lambda: {"judges_wait_for_heavy_run": True}
    )

    run_b = run_registry.create_run("B", "batch", {})
    run_judge = run_registry.create_run("JUDGE", "batch", {})

    t_b = threading.Thread(target=engine_service.start_run, args=(run_b, "B", {}))
    t_b.start()
    time.sleep(0.02)  # let B acquire _HEAVY_RUN_LOCK first

    t_judge = threading.Thread(target=engine_service.start_judge_run, args=(run_judge, {}))
    t_judge.start()
    time.sleep(0.05)

    assert run_registry.get_run(run_judge)["status"] == "queued"

    t_b.join(timeout=5)
    t_judge.join(timeout=5)

    assert order == ["B-start", "B-end", "JUDGE"]


def test_judge_does_not_wait_for_heavy_run_when_toggle_off(monkeypatch):
    """With "judges_wait_for_heavy_run" OFF, a judge run must still execute
    immediately even while B holds _HEAVY_RUN_LOCK (the pre-toggle
    behavior), bounded only by _JUDGE_SEMAPHORE."""
    started_judge = threading.Event()

    def slow_b(run_id, params):
        time.sleep(0.2)

    def judge_stub(run_id, params):
        started_judge.set()
        run_registry.update_run(run_id, status="completed")

    monkeypatch.setitem(engine_service.RUNNERS, "B", slow_b)
    monkeypatch.setattr(engine_service, "run_judge_batch_dashboard", judge_stub)
    monkeypatch.setattr(engine_service.settings_service, "apply_to_environment", lambda: None)
    monkeypatch.setattr(
        engine_service.settings_service, "get_raw_settings", lambda: {"judges_wait_for_heavy_run": False}
    )

    run_b = run_registry.create_run("B", "batch", {})
    run_judge = run_registry.create_run("JUDGE", "batch", {})

    t_b = threading.Thread(target=engine_service.start_run, args=(run_b, "B", {}))
    t_b.start()
    time.sleep(0.02)

    t_judge = threading.Thread(target=engine_service.start_judge_run, args=(run_judge, {}))
    t_judge.start()
    t_judge.join(timeout=5)

    assert started_judge.is_set()
    assert run_registry.get_run(run_judge)["status"] == "completed"

    t_b.join(timeout=5)
