"""secondary_metrics.py (Amendment 16): plan/dry run makes no call and counts exactly; the judge is the single
registry judge "primary" at temperature 0; the answer is blinded; official files stay behind the Amendment 14 gate."""

import json

import pytest

import judge_versions as jv
import secondary_metrics as sm


def _run_file(tmp_path, name, records, split="dev"):
    p = tmp_path / name
    p.write_text("".join(json.dumps({**r, "config": {"sample_split": split}}) + "\n" for r in records))
    return str(p)


def _item(aid, text):
    return {"answer_id": aid, "chunk_text": text}


@pytest.fixture
def no_client(monkeypatch):
    import judge_clients

    def boom(*a, **k):
        raise AssertionError("a judge client was created in a dry run")

    monkeypatch.setattr(judge_clients, "get_judge_client", boom)


def test_plan_counts_calls_exactly_and_creates_no_client(tmp_path, no_client):
    b = _run_file(tmp_path, "b.jsonl", [
        {"question_id": 1, "llm_answer": "a", "retrieved_context": [_item(10, "x"), _item(11, "y")]},
        {"question_id": 2, "llm_answer": "b", "retrieved_context": [_item(10, "x")]}])
    c = _run_file(tmp_path, "c.jsonl", [
        {"question_id": 1, "llm_answer": "a", "retrieved_context": [_item(10, "x"), _item(12, "z")]}])
    a = _run_file(tmp_path, "a.jsonl", [{"question_id": 1, "llm_answer": "a"}, {"question_id": 2, "llm_answer": "b"}])
    p = sm.plan({"A": ("A", a), "B": ("B", b), "C": ("C", c)})
    # context relevance: unique (question, answer, text) items over B and C: (1,10), (1,11), (2,10), (1,12) = 4
    # answer relevance: one call per answer of every file: 2 + 2 + 1 = 5
    assert p["calls"] == {"ctxrel": 4, "ansrel": 5} and p["total_calls"] == 9
    assert p["judge"]["model"] == "meta-llama/Llama-3.3-70B-Instruct" and p["judge"]["temperature"] == 0.0
    assert "NO CALL WAS MADE" in sm.plan_text(p)


def test_projection_for_the_four_runs(no_client):
    p = sm.project(384, 5)
    assert p["calls"] == {"ctxrel": 3 * 384 * 5, "ansrel": 4 * 384} and p["total_calls"] == 7296
    assert set(p["per_run"]) == {"A", "B-grounded", "C-primary-grounded", "D-grounded"}
    assert p["per_run"]["A"]["ctxrel_items_before_dedup"] == 0  # A has no context
    assert p["estimated_cost_usd"] > 0


def test_one_judge_temperature_zero_existing_prompts():
    import llm_judge_context_relevance_v1 as cx
    from judge_clients import JUDGE_REGISTRY

    assert sm.JUDGE_ID == "primary" and JUDGE_REGISTRY["primary"].model == "meta-llama/Llama-3.3-70B-Instruct"
    assert cx.REQUEST_TEMPERATURE == 0.0
    assert sm.CTXREL_PROMPT in ("ctxrel-v1", "ctxrel-v2") and cx.get_prompt_builder(sm.CTXREL_PROMPT)


def test_official_files_are_refused_before_amendment14_approval(tmp_path, monkeypatch):
    f = _run_file(tmp_path, "c.jsonl", [{"question_id": 1}], split="test")
    monkeypatch.setattr(jv, "AMENDMENT14_APPROVED", False)
    with pytest.raises(jv.JudgeVersionError, match="Amendment 14"):
        sm._gate([f])
    sm._gate([_run_file(tmp_path, "dev.jsonl", [{"question_id": 1}], split="dev")])  # dev files need no approval
    monkeypatch.setattr(jv, "AMENDMENT14_APPROVED", True)
    sm._gate([f])


def test_answer_relevance_prompt_is_blinded_and_sees_only_question_and_answer(tmp_path, monkeypatch):
    import _judge_common
    import llm_judge_context_relevance_v1 as cx

    seen = []

    def fake_call(client, config, messages):
        seen.append(messages)
        return {"raw": json.dumps({"answer_relevance_label": "MENJAWAB", "justification": "ok"}),
                "response_model": "m", "prompt_tokens": 1, "completion_tokens": 1}, []

    class Cfg:
        model = "meta-llama/Llama-3.3-70B-Instruct"

    import judge_clients
    monkeypatch.setattr(judge_clients, "get_judge_client", lambda jid: (object(), Cfg()))
    monkeypatch.setattr(cx, "call_judge_with_retry", fake_call)
    monkeypatch.setattr(_judge_common, "load_question_bodies", lambda parquet, ids: {i: "<p>question body</p>" for i in ids})
    monkeypatch.setattr(sm, "OUT_DIR", tmp_path)
    monkeypatch.setattr(sm, "ANSREL_FILE", tmp_path / "ansrel.jsonl")
    answer = "Use foo() [SO-123]. The retrieved sources do not cover bar. Sources: [SO-123]"
    f = _run_file(tmp_path, "c.jsonl", [{"question_id": 7, "title": "T", "llm_answer": answer,
                                         "ground_truth_answer": "REFERENCE", "retrieved_context": [_item(1, "CTX")],
                                         "condition": "C", "trust_weight": 0.9}])
    out = sm.run_ansrel([f], workers=1)
    assert out["answer_relevance_judged"] == 1 and len(seen) == 1
    text = json.dumps(seen[0])
    for leak in ("[SO-123]", "retrieved sources", "REFERENCE", "CTX", "trust", "GraphRAG", "Condition"):
        assert leak not in text, leak
    assert "Use foo()" in text and "question body" in text
    rec = json.loads((tmp_path / "ansrel.jsonl").read_text())
    assert rec["label"] == "MENJAWAB" and rec["score"] == 1.0 and rec["temperature"] == 0.0
    # resume: a second call judges nothing new
    assert sm.run_ansrel([f], workers=1)["answer_relevance_judged"] == 0


def test_report_section_reads_stored_judgments(tmp_path, monkeypatch):
    f = _run_file(tmp_path, "a.jsonl", [{"question_id": 1}, {"question_id": 2}])
    canon = str(sm.archive_paths.canonical(f))
    rows = [{"source_file": canon, "question_id": 1, "prompt_version": sm.ANSREL_PROMPT, "label": "MENJAWAB", "score": 1.0},
            {"source_file": canon, "question_id": 2, "prompt_version": sm.ANSREL_PROMPT, "label": "TIDAK_MENJAWAB", "score": 0.0}]
    (tmp_path / "ansrel.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(sm, "ANSREL_FILE", tmp_path / "ansrel.jsonl")
    monkeypatch.setattr(sm, "CTXREL_FILE", tmp_path / "missing.jsonl")
    out = sm.report_section(f, "A")
    assert out["answer_relevance"]["mean_score"] == 0.5 and out["answer_relevance"]["n_judged"] == 2
    assert out["context_relevance"] is None  # condition A has no context


# --- operational safeguards of --run (2026-10-28): cap, ledger, resume, torn tail, identity, registration -------------

import judge_clients  # noqa: E402
import llm_judge_context_relevance_v1 as cx  # noqa: E402
from llm import run_registry as rr  # noqa: E402


@pytest.fixture(autouse=True)   # reconcile/register must NEVER reach the real results/run_registry.json
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "REGISTRY_PATH", tmp_path / "run_registry.json")
    rr.save(rr.load(), None)
    return tmp_path / "run_registry.json"


class _Cfg:
    model = "meta-llama/Llama-3.3-70B-Instruct"


def _fake_ansrel(monkeypatch, tmp_path, fail_ids=(), tokens=(1000, 50)):
    import _judge_common

    calls = []

    def fake_call(client, config, messages):
        calls.append(messages)
        qid = int(next(m["content"] for m in messages if m["role"] == "user").split("QID=")[1].split()[0]) \
            if "QID=" in json.dumps(messages) else None
        if qid in fail_ids:
            return None, [{"attempt": 1, "error": "boom", "status_code": 500}]
        return {"raw": json.dumps({"answer_relevance_label": "MENJAWAB", "justification": "ok"}),
                "response_model": "m", "prompt_tokens": tokens[0], "completion_tokens": tokens[1]}, []

    monkeypatch.setattr(judge_clients, "get_judge_client", lambda jid: (object(), _Cfg()))
    monkeypatch.setattr(cx, "call_judge_with_retry", fake_call)
    monkeypatch.setattr(_judge_common, "load_question_bodies", lambda parquet, ids: {i: "QID=%d" % i for i in ids})
    monkeypatch.setattr(sm, "ANSREL_FILE", tmp_path / "ansrel.jsonl")
    return calls


def _answers(tmp_path, n=6):
    return _run_file(tmp_path, "c.jsonl", [{"question_id": i, "title": "T", "llm_answer": "use foo()"} for i in range(1, n + 1)])


def test_spend_guard_stops_before_the_cap_and_reports_skipped(tmp_path, monkeypatch):
    calls = _fake_ansrel(monkeypatch, tmp_path, tokens=(3000, 400))   # each call costs exactly the reserved worst case
    f = _answers(tmp_path, 6)
    one = sm.SpendGuard(10).worst_case("ansrel")
    guard = sm.SpendGuard(2.5 * one)          # room for 2 calls in flight/spent, not 3
    out = sm.run_ansrel([f], workers=1, guard=guard, ans_file=tmp_path / "ansrel.jsonl")
    assert out["answer_relevance_judged"] == 2 and len(calls) == 2
    assert out["budget_skipped"] == guard.n_skipped == 4
    assert guard.spent <= guard.max_cost_usd and guard.reserved == pytest.approx(0)
    assert guard.spent == pytest.approx(2 * (3000 * 0.10 + 400 * 0.32) / 1e6)  # actual tokens x registry price


def test_failed_call_goes_to_the_failures_file_and_is_retried_on_resume(tmp_path, monkeypatch):
    _fake_ansrel(monkeypatch, tmp_path, fail_ids=(3,))
    f = _answers(tmp_path, 4)
    out_file = tmp_path / "ansrel.jsonl"
    out = sm.run_ansrel([f], workers=1, ans_file=out_file)
    assert out["answer_relevance_judged"] == 3 and out["answer_relevance_failed"] == 1
    assert len(sm._read(out_file)) == 3 and len(sm._read(sm.failures_path_of(out_file))) == 1
    _fake_ansrel(monkeypatch, tmp_path)       # the judge works now: only the failed one is called again
    again = sm.run_ansrel([f], workers=1, ans_file=out_file)
    assert again["answer_relevance_judged"] == 1 and again["skipped"] == 3
    assert len(sm._read(out_file)) == 4


def test_guarded_ctxrel_runs_run_batch_in_chunks_and_stops_at_the_cap(tmp_path, monkeypatch):
    items = [{"question_id": i, "answer_id": i, "text_hash": f"h{i}", "chunk_text": "x", "title": "t"} for i in range(1, 8)]
    batches = []

    def fake_batch(chunk, judge_ids, qp, output_path, failures_path, workers, prompt_version=None, **k):
        batches.append(len(chunk))
        with open(output_path, "a") as f:
            for it in chunk:
                f.write(json.dumps({"question_id": it["question_id"], "answer_id": it["answer_id"], "text_hash": it["text_hash"],
                                    "judge_id": sm.JUDGE_ID, "judge_model": _Cfg.model, "prompt_version": prompt_version,
                                    "parse_error": False, "prompt_tokens": 3000, "completion_tokens": 400}) + "\n")
        return {"judged": len(chunk), "skipped": 0, "failed": 0, "parse_errors": 0}

    monkeypatch.setattr(cx, "load_items", lambda files, limit: items)
    monkeypatch.setattr(cx, "run_batch", fake_batch)
    monkeypatch.setattr(sm, "CTXREL_CHUNK", 3)
    one = sm.SpendGuard(10).worst_case("ctxrel", 3)
    guard = sm.SpendGuard(1.2 * one)          # one chunk of 3 fits (each costs the reserved worst case); the second would not
    out = tmp_path / "ctx.jsonl"
    res = sm.run_ctxrel(["x"], 1, guard, out)
    assert batches == [3] and res["judged"] == 3 and res["budget_skipped"] == 4 and guard.n_skipped == 4
    assert guard.spent == pytest.approx(3 * (3000 * 0.10 + 400 * 0.32) / 1e6)
    # resume with a larger cap: the 3 stored items are skipped, the other 4 are judged (3 + 1)
    guard2 = sm.SpendGuard(100)
    res2 = sm.run_ctxrel(["x"], 1, guard2, out)
    assert batches == [3, 3, 1] and res2["judged"] == 4 and res2["skipped"] == 3
    # without a guard the unguarded path is the plain run_batch call (one batch of everything)
    sm.run_ctxrel(["x"], 1, None, tmp_path / "ctx2.jsonl")
    assert batches[-1] == 7


def test_ledger_is_booked_once_from_the_stored_records_and_is_idempotent(tmp_path):
    out = tmp_path / "ansrel_official.jsonl"
    out.write_text("".join(json.dumps({"prompt_tokens": 1_000_000, "completion_tokens": 0, "label": "MENJAWAB"}) + "\n"
                           for _ in range(2)))
    assert sm.reconcile_ledger(out, "official", lambda *a: None) == pytest.approx(0.2)   # 2 x 1M tokens x $0.10
    assert sm.reconcile_ledger(out, "official", lambda *a: None) == 0.0                   # second call: nothing new
    entries = rr.load()["deepinfra_spend"]
    assert len(entries) == 1 and entries[0]["file"] == out.name and entries[0]["cost_usd"] == pytest.approx(0.2)
    with open(out, "a") as f:                                                              # a later run adds one record
        f.write(json.dumps({"prompt_tokens": 1_000_000, "completion_tokens": 0}) + "\n")
    assert sm.reconcile_ledger(out, "official", lambda *a: None) == pytest.approx(0.1)
    assert sum(e["cost_usd"] for e in rr.load()["deepinfra_spend"]) == pytest.approx(0.3)


def test_torn_last_line_is_tolerated_and_repaired(tmp_path):
    from llm.common.resume import repair_torn_tail

    out = tmp_path / "x.jsonl"
    out.write_text(json.dumps({"label": "MENJAWAB", "prompt_tokens": 10}) + "\n" + '{"label": "MENJA')
    assert len(sm._read_tolerant(out)) == 1                       # readers skip the torn tail
    repair_torn_tail(out, lambda *a: None)
    assert out.read_text().endswith("\n") and len(sm._read_tolerant(out)) == 1
    out.write_text('{"a": 1}\nnot json\n{"b": 2}\n')              # corruption elsewhere is NOT hidden
    with pytest.raises(json.JSONDecodeError):
        sm._read_tolerant(out)


def test_identity_report_ok_and_mismatch(tmp_path, monkeypatch):
    import llm.sample_split as ss

    monkeypatch.setattr(ss, "load_frozen_ids", lambda split, path=None: {"ids": [1, 2] if split == "test" else [99]})
    b = _run_file(tmp_path, "b.jsonl", [{"question_id": 1, "retrieved_context": [_item(10, "x")]},
                                        {"question_id": 2, "retrieved_context": [_item(11, "y")]}])
    a = _run_file(tmp_path, "a.jsonl", [{"question_id": 1}, {"question_id": 2}])
    runs = {"A": ("A", a), "B": ("B", b)}
    monkeypatch.setattr(cx, "load_items", lambda files, limit: [
        {"question_id": r["question_id"], "answer_id": it["answer_id"], "text_hash": sm._text_hash(it["chunk_text"])}
        for f in files for r in sm._read(f) for it in r.get("retrieved_context") or []])
    ctx, ans = tmp_path / "ctx.jsonl", tmp_path / "ans.jsonl"
    canon = lambda p: str(sm.archive_paths.canonical(p))  # noqa: E731
    ctx.write_text("".join(json.dumps({"question_id": q, "answer_id": a_, "text_hash": sm._text_hash(t), "parse_error": False}) + "\n"
                           for q, a_, t in ((1, 10, "x"), (2, 11, "y"))))
    ans.write_text("".join(json.dumps({"source_file": canon(p), "question_id": q, "label": "MENJAWAB"}) + "\n"
                           for p in (a, b) for q in (1, 2)))
    res = sm.identity_report(runs, ctx, ans)
    assert res["ok"] and res["ctxrel"]["expected"] == 2 and res["ansrel"]["expected"] == 4
    with open(ans, "a") as f:                                       # a duplicate record and a dev id break it
        f.write(json.dumps({"source_file": canon(a), "question_id": 99, "label": "MENJAWAB"}) + "\n")
    bad = sm.identity_report(runs, ctx, ans)
    assert not bad["ok"] and bad["dev_ids"] == 1
    with pytest.raises(sm.SecondaryGuardError, match="not among the current inputs"):
        sm.assert_no_foreign_records(runs, ctx, ans)


def test_run_needs_a_cap_and_official_runs_refuse_a_foreign_output_dir(tmp_path, monkeypatch, capsys):
    f = _run_file(tmp_path, "c.jsonl", [{"question_id": 1, "llm_answer": "a"}], split="test")

    class Args:
        max_cost_usd, out_dir, workers = None, None, 1

    runs = {"C": ("C", f)}
    assert sm.run_guarded(runs, [f], Args) == 2 and "--max-cost-usd" in capsys.readouterr().out
    Args.max_cost_usd, Args.out_dir = 1.0, str(tmp_path / "elsewhere")
    assert sm.run_guarded(runs, [f], Args) == 2 and "no --out-dir" in capsys.readouterr().out


def test_register_secondary_run_records_an_official_judge_entry(tmp_path):
    ctx, ans = tmp_path / "ctx.jsonl", tmp_path / "ans.jsonl"
    sm.register_secondary_run("official", {"A": ("A", "a.jsonl")}, ctx, ans, 0.5, "2026-10-28T00:00:00+00:00", 10, lambda *a: None)
    e = rr.load()["runs"][-1]
    assert e["kind"] == "judge" and e["status"] == "official" and e["cost_usd"] == 0.5
    assert e["judge_config"]["ctxrel_prompt"] == "ctxrel-v2" and e["judge_config"]["temperature"] == 0.0
