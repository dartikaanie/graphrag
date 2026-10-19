"""judge_panel.py runner pieces (Amendment 12), offline: a fake OpenAI client, no API call."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import judge_panel as jp
from judge_panel_config import JUDGES_BY_ID, MAX_RETRIES, MAX_TOKENS, PANEL_ID, PANEL_JUDGES, TEMPERATURE
from judge_prompt_v2_2 import PROMPT_VERSION

GOOD = json.dumps({
    "answer_attempted": True, "completeness": "FULL", "approach_match": "SAME",
    "claims": [{"claim": "x", "verdict": "CONTRADICTED", "severity": "CORE", "reference_conflict": False,
                "supported_by_context": "N/A", "cited_items": [], "cited_item_supports": None, "evidence": "e"}],
    "reasoning": "r", "label": "HALUSINASI_PENUH"})


class FakeClient:
    """Returns the queued outcomes in order: a str is the message content, an Exception is raised."""

    def __init__(self, outcomes, model="served", reasoning_content=None):
        self.outcomes, self.calls = list(outcomes), []
        self.model, self.reasoning_content = model, reasoning_content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        msg = SimpleNamespace(model_dump=lambda: {"content": out, "reasoning_content": self.reasoning_content})
        usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 1000, "completion_tokens": 100,
                                                    "estimated_cost": 0.0001})
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage,
                               model=self.model, id="resp-1")


def _conn_error():
    import httpx
    import openai
    return openai.APIConnectionError(request=httpx.Request("POST", "https://x"))


def _bad_request():
    import httpx
    import openai
    return openai.BadRequestError("bad", response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
                                  body=None)


ITEM = {"item_id": "run-a:1", "run_id": "run-a", "question_id": 1, "condition": "A", "run_label": "A",
        "config_hash": "h", "source_file": "a.jsonl"}
PREPARED = {"has_context": False, "n_context_items": 0, "citation_map": {}, "offered_items": []}
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u" * 300}]
NOSLEEP = lambda s: None  # noqa: E731


def _rec(judge_id, client):
    return jp.judge_record(ITEM, JUDGES_BY_ID[judge_id], MESSAGES, PREPARED, client, sleep=NOSLEEP)


def test_panel_config():
    assert PANEL_ID == "panel-v1" and len(PANEL_JUDGES) == 3
    assert [j.model for j in PANEL_JUDGES] == ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "deepseek-ai/DeepSeek-V3.2",
                                                "Qwen/Qwen2.5-72B-Instruct"]
    assert all(j.base_url == "https://api.deepinfra.com/v1/openai" for j in PANEL_JUDGES)
    assert "gpt-4o-mini" not in {j.model for j in PANEL_JUDGES}
    assert JUDGES_BY_ID["deepseek"].extra_body == {"reasoning_effort": "none"}
    assert all(j.expected_served_model == j.model for j in PANEL_JUDGES)
    assert TEMPERATURE == 0.0 and MAX_TOKENS == 2000 and PROMPT_VERSION == "judge-v2.2"


def test_request_params_and_record_fields():
    c = FakeClient([GOOD], model="deepseek-ai/DeepSeek-V3.2")
    r = _rec("deepseek", c)
    call = c.calls[0]
    assert call["model"] == "deepseek-ai/DeepSeek-V3.2" and call["temperature"] == 0.0 and call["max_tokens"] == 2000
    assert call["extra_body"] == {"reasoning_effort": "none"}
    for f in ("item_id", "condition", "run_label", "judge_model", "prompt_version", "temperature", "max_tokens",
              "thinking_disabled", "raw_response", "parsed_json", "label", "parse_ok", "normalized_fields",
              "finish_reason", "prompt_tokens", "completion_tokens", "estimated_cost_usd", "latency_s",
              "timestamp_utc"):
        assert f in r, f
    assert r["label"] == "HALUSINASI_PENUH" and r["parse_ok"] and r["hallucinated"] is True
    assert r["thinking_disabled"] is True and r["served_model_mismatch"] is False
    assert r["estimated_cost_usd"] == pytest.approx(1000 / 1e6 * 0.26 + 100 / 1e6 * 0.38)


def test_no_extra_body_for_llama_and_qwen():
    for jid in ("llama", "qwen"):
        c = FakeClient([GOOD])
        _rec(jid, c)
        assert "extra_body" not in c.calls[0]


def test_retry_on_parse_failure_then_success():
    c = FakeClient(["not json", GOOD])
    r = _rec("qwen", c)
    assert len(c.calls) == 2 and r["parse_ok"] and len(r["attempts"]) == 2
    assert r["n_paid_attempts"] == 2
    assert r["estimated_cost_usd"] == pytest.approx(2 * (1000 / 1e6 * 0.36 + 100 / 1e6 * 0.40))


def test_parse_failure_exhausts_retries():
    c = FakeClient(["nope"] * 5)
    r = _rec("qwen", c)
    assert len(c.calls) == 1 + MAX_RETRIES and r["parse_ok"] is False and r["call_failed"] is False
    assert r["label"] is None and r["raw_response"] == "nope"


def test_retry_on_network_error():
    c = FakeClient([_conn_error(), GOOD])
    r = _rec("llama", c)
    assert len(c.calls) == 2 and r["parse_ok"] and r["attempts"][0]["error"].startswith("APIConnectionError")


def test_network_error_exhausts_retries():
    c = FakeClient([_conn_error()] * 5)
    r = _rec("llama", c)
    assert len(c.calls) == 1 + MAX_RETRIES and r["call_failed"] and r["label"] is None


def test_non_retryable_fails_fast():
    c = FakeClient([_bad_request(), GOOD])
    r = _rec("llama", c)
    assert len(c.calls) == 1 and r["call_failed"]


def test_reasoning_detected():
    r = _rec("deepseek", FakeClient([GOOD], model="deepseek-ai/DeepSeek-V3.2", reasoning_content="thinking..."))
    assert r["reasoning_present"] is True and r["thinking_disabled"] is False
    assert jp.reasoning_present({"content": "<think>x</think>{}"})
    assert not jp.reasoning_present({"content": "{}", "reasoning_content": None})


def test_served_model_mismatch_flag():
    assert _rec("llama", FakeClient([GOOD], model="meta-llama/Llama-3.3-70B-Instruct-Turbo"))[
        "served_model_mismatch"] is False
    assert _rec("llama", FakeClient([GOOD], model="something-else"))["served_model_mismatch"] is True


def test_cache_skips_completed_calls(tmp_path, monkeypatch):
    monkeypatch.setattr("_judge_common.html_to_text", lambda s, *a, **k: s or "")
    items = [{**ITEM, "title": "t", "tags": "", "ground_truth_answer_html": "ref", "candidate": "c",
              "retrieved_context": []}]
    done = {(ITEM["item_id"], JUDGES_BY_ID["llama"].model, PROMPT_VERSION)}
    tasks, _, n_cached = jp.plan_tasks(items, {1: "body"}, done)
    assert n_cached == 1 and [t[1].judge_id for t in tasks] == ["deepseek", "qwen"]


def test_run_tasks_writes_cache_and_failures(tmp_path):
    out = tmp_path / "p.jsonl"
    tasks = [(ITEM, j, MESSAGES) for j in PANEL_JUDGES]
    clients = {"llama": FakeClient([GOOD]), "deepseek": FakeClient(["bad"] * 3), "qwen": FakeClient([GOOD])}
    new = jp.run_tasks(tasks, {ITEM["item_id"]: PREPARED}, clients, out, concurrency=3, sleep=NOSLEEP)
    assert len(new) == 3
    cached = jp.read_jsonl(out)
    assert {r["judge_id"] for r in cached} == {"llama", "qwen"}
    assert [r["judge_id"] for r in jp.read_jsonl(jp.failures_path_of(out))] == ["deepseek"]
    panel = jp.build_panel_records([ITEM], cached)[0]
    assert panel["panel_label"] == "HALUSINASI_PENUH" and panel["panel_incomplete"] is True
    assert panel["judge_labels"] == {"llama": "HALUSINASI_PENUH", "deepseek": None, "qwen": "HALUSINASI_PENUH"}


def test_project_cost_and_guard():
    tasks = [(ITEM, j, MESSAGES) for j in PANEL_JUDGES]
    p = jp.project_cost(tasks, [])
    tin = sum(len(m["content"]) for m in MESSAGES) / jp.CHARS_PER_TOKEN_EST
    expected = sum(tin / 1e6 * j.price_per_m_input + MAX_TOKENS / 1e6 * j.price_per_m_output for j in PANEL_JUDGES)
    assert p["total_usd"] == pytest.approx(expected, abs=1e-5)
    assert p["worst_case_with_retries_usd"] == pytest.approx(p["total_usd"] * 3, abs=1e-4)
    # completed records lower the output estimate to their mean
    done = [{"judge_model": j.model, "completion_tokens": 500} for j in PANEL_JUDGES]
    assert jp.project_cost(tasks, done)["total_usd"] < p["total_usd"]


# --- annotation exclusion ----------------------------------------------------

def _excl(tmp_path, cal_items, conf_items=None, cal_file=True):
    from annotation_tool import AnnotationError

    def load_key(purpose, key_dir):
        items = cal_items if purpose == "calibration" else conf_items
        if items is None:
            raise AnnotationError("no key")
        return {"items": items}
    run_file = tmp_path / "run.jsonl"
    run_file.write_text("")
    cal_paths = {str(run_file.resolve())} if cal_file else set()
    return jp.annotation_exclusions([str(run_file)], tmp_path, cal_paths, load_key), str(run_file)


def test_exclusion_drops_sample_items_and_same_questions(tmp_path):
    excl, run_file = _excl(tmp_path, [{"run_id": "r1", "question_id": 10}])
    items = [{"run_id": r, "question_id": q, "source_file": run_file} for r, q in
             (("r1", 10), ("r2", 10), ("r2", 11))]
    kept, counts = jp.apply_exclusions(items, excl, explicit_question_ids=False)
    assert [(i["run_id"], i["question_id"]) for i in kept] == [("r2", 11)]
    assert counts == {"excluded_annotation_items": 1, "excluded_same_question_as_calibration": 1}


def test_exclusion_explicit_request_is_an_error(tmp_path):
    excl, run_file = _excl(tmp_path, [{"run_id": "r1", "question_id": 10}])
    with pytest.raises(jp.PanelGuardError):
        jp.apply_exclusions([{"run_id": "r2", "question_id": 10, "source_file": run_file}], excl, True)


def test_calibration_runs_need_the_key(tmp_path):
    with pytest.raises(jp.PanelGuardError):
        _excl(tmp_path, None)
    excl, _ = _excl(tmp_path, None, cal_file=False)  # not a calibration run file: key optional
    assert excl["pairs"] == set()


def test_confirmation_items_excluded(tmp_path):
    excl, run_file = _excl(tmp_path, [], [{"run_id": "t1", "question_id": 5}], cal_file=False)
    kept, counts = jp.apply_exclusions([{"run_id": "t1", "question_id": 5, "source_file": run_file}], excl, False)
    assert kept == [] and counts["excluded_annotation_items"] == 1


def test_summary_counts_unspecified_and_failures():
    items = [{**ITEM, "item_id": f"r:{q}", "question_id": q, "run_label": lab} for q, lab in ((1, "A"), (2, "B-plain"))]
    base = {"panel_id": PANEL_ID, "prompt_version": PROMPT_VERSION, "parse_ok": True, "call_failed": False}
    recs = [{**base, "item_id": "r:1", "judge_model": j.model, "label": "HALUSINASI_SEVERITY_UNSPECIFIED",
             "judge_id": j.judge_id} for j in PANEL_JUDGES[:2]]
    recs.append({**base, "item_id": "r:1", "judge_model": PANEL_JUDGES[2].model, "label": "HALUSINASI_PENUH",
                 "judge_id": "qwen"})
    recs.append({**base, "item_id": "r:2", "judge_model": PANEL_JUDGES[0].model, "label": "FAKTUAL",
                 "judge_id": "llama"})
    panel = jp.build_panel_records(items, recs)
    s = jp.summarize(recs, panel, [])
    assert s["per_judge"]["llama"]["unspecified_votes"] == 1 and s["per_judge"]["qwen"]["unspecified_votes"] == 0
    assert s["unspecified_by_run_label"] == {"A": 1, "B-plain": 0}
    assert s["unresolved_by_run_label"] == {"A": 0, "B-plain": 1}
    assert {k: v["final_failures"] for k, v in s["per_judge"].items()} == {"llama": 0, "deepseek": 1, "qwen": 1}


def test_parse_item_specs():
    assert jp.parse_item_specs(None, 3) is None
    assert jp.parse_item_specs(["0:12", "2:7", "0:13"], 3) == {0: {12, 13}, 2: {7}}
    for bad in (["3:1"], ["x:1"], ["1"]):
        with pytest.raises(ValueError):
            jp.parse_item_specs(bad, 3)


def test_select_calibration_set_exact():
    key = {"items": [{"run_id": "r1", "question_id": 1}, {"run_id": "r2", "question_id": 2}]}
    items = [{"run_id": r, "question_id": q} for r, q in (("r1", 1), ("r1", 2), ("r2", 2), ("r2", 3))]
    assert [(i["run_id"], i["question_id"]) for i in jp.select_calibration_set(items, key)] == [("r1", 1), ("r2", 2)]
    with pytest.raises(jp.PanelGuardError):
        jp.select_calibration_set(items[:1], key)
