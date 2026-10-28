"""secondary_panel.py (operational amendment 12j): the secondary metrics judged by the 3-judge panel, one stage per metric.
Offline: fake clients, isolated registry. Nothing here calls an API or touches the real run_registry.json."""

import json
import types

import pytest

import judge_panel as jp
import secondary_metrics as sm
import secondary_panel as sp
from llm import run_registry as rr

IDS = [j.judge_id for j in sp.JUDGES]


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "REGISTRY_PATH", tmp_path / "run_registry.json")
    rr.save(rr.load(), None)
    return tmp_path / "run_registry.json"


# --- fake client -----------------------------------------------------------------------------------------------------

class _Msg:
    def __init__(self, content):
        self.content = content

    def model_dump(self):
        return {"content": self.content}


class _Usage:
    def __init__(self, pt, ct):
        self.pt, self.ct = pt, ct

    def model_dump(self):
        return {"prompt_tokens": self.pt, "completion_tokens": self.ct}


class FakeClient:
    """label_of(call_index, messages) -> raw JSON text for ansrel; the same client serves any judge."""

    def __init__(self, judge, behaviour, tokens=(1000, 50)):
        self.judge, self.behaviour, self.tokens, self.n = judge, behaviour, tokens, 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))
        self.requests = []

    def _create(self, **request):
        self.n += 1
        self.requests.append(request)
        raw = self.behaviour(self.judge.judge_id, request["messages"], self.n)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=_Msg(raw), finish_reason="stop")],
                                     usage=_Usage(*self.tokens), model=self.judge.expected_served_model, id=f"r{self.n}")


def _ans(label):
    return json.dumps({"answer_relevance_label": label, "justification": "ok"})


def _clients(behaviour, tokens=(1000, 50)):
    return {j.judge_id: FakeClient(j, behaviour, tokens) for j in sp.JUDGES}


def _items(n, metric="ansrel"):
    return [{"source_file": "/x/a.jsonl", "question_id": i, "messages": [{"role": "user", "content": f"Q{i}"}]}
            for i in range(1, n + 1)]


def _run(items, clients, out, guard=None):
    return sp.run_stage("ansrel", items, clients, out, guard, concurrency=1, sleep=lambda s: None)


def _recs(path):
    return sm._read_tolerant(path)


# --- request settings, judges, unchanged definitions ---------------------------------------------------------------------

def test_judges_prompts_labels_and_request_are_the_panel_ones():
    from judge_panel_config import PANEL_V2_JUDGES
    import judge_prompt_v2_3 as v23

    assert sp.JUDGES == PANEL_V2_JUDGES and IDS == ["llama", "deepseek", "qwen3-235b"]
    spec = sp.RequestSpec("ansrel")
    assert (spec.temperature, spec.max_tokens, spec.response_format) == (0.0, v23.REQUEST_MAX_TOKENS, v23.REQUEST_RESPONSE_FORMAT)
    assert sp.METRICS["ansrel"]["prompt"] == "ansrel-v1" and sp.METRICS["ctxrel"]["prompt"] == "ctxrel-v2"
    assert sp.METRICS["ansrel"]["score"] == {"MENJAWAB": 1.0, "MENJAWAB_SEBAGIAN": 0.5, "TIDAK_MENJAWAB": 0.0}
    assert sp.METRICS["ctxrel"]["score"] == {"RELEVANT": 1.0, "PARTIAL": 0.5, "IRRELEVANT": 0.0}
    assert spec.parse(_ans("MENJAWAB"))["label"] == "MENJAWAB" and spec.parse("nonsense")["parse_error"]
    assert sp.RequestSpec("ctxrel").parse(json.dumps({"label": "PARTIAL", "reason": "r"}))["label"] == "PARTIAL"
    assert sp.RequestSpec("ctxrel").parse(json.dumps({"label": "MAYBE"}))["parse_error"]  # the parsers are not loosened


def test_each_judge_gets_the_panel_request_settings(tmp_path):
    clients = _clients(lambda j, m, n: _ans("MENJAWAB"))
    _run(_items(1), clients, tmp_path / "o.jsonl")
    req = {jid: c.requests[0] for jid, c in clients.items()}
    assert all(r["temperature"] == 0.0 and r["response_format"] == {"type": "json_object"} for r in req.values())
    assert req["deepseek"]["extra_body"] == {"reasoning_effort": "none"} and "extra_body" not in req["llama"]
    assert req["llama"]["model"] == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    assert req["qwen3-235b"]["model"] == "Qwen/Qwen3-235B-A22B-Instruct-2507"


# --- aggregation -------------------------------------------------------------------------------------------------------------

def test_aggregate_majority_median_incomplete_unresolved():
    a = lambda **kw: sp.aggregate_key(dict(zip(IDS, kw["l"])), "ansrel")  # noqa: E731
    assert a(l=["MENJAWAB", "MENJAWAB", "TIDAK_MENJAWAB"]) == {"panel_label": "MENJAWAB", "decided_by": "majority", "n_valid": 3}
    assert a(l=["MENJAWAB", "TIDAK_MENJAWAB", "MENJAWAB_SEBAGIAN"])["panel_label"] == "MENJAWAB_SEBAGIAN"   # median fallback
    assert a(l=["MENJAWAB", "TIDAK_MENJAWAB", "MENJAWAB_SEBAGIAN"])["decided_by"] == "median"
    assert a(l=["PARTIAL", None, None])["panel_label"] == sp.INCOMPLETE                                        # < 2 usable
    assert a(l=[None, None, None])["panel_label"] == sp.INCOMPLETE
    assert a(l=["MENJAWAB", "MENJAWAB", None]) == {"panel_label": "MENJAWAB", "decided_by": "majority", "n_valid": 2}
    assert a(l=["MENJAWAB", None, "TIDAK_MENJAWAB"])["panel_label"] == sp.UNRESOLVED                           # 2 usable, disagree
    m = sp.aggregate_key({"llama": "IRRELEVANT", "deepseek": "RELEVANT", "qwen3-235b": "PARTIAL"}, "ctxrel")
    assert m["panel_label"] == "PARTIAL" and m["decided_by"] == "median"


def test_agreement_statistics_on_hand_checked_examples():
    # Fleiss: rows AAA, AAB, BBB -> P_bar = 7/9, P_e = 41/81, kappa = 22/40 = 0.55
    assert sp.fleiss_kappa([["A", "A", "A"], ["A", "A", "B"], ["B", "B", "B"]], ("A", "B")) == 0.55
    # Krippendorff ordinal: units (A,A), (B,B), (A,B), n_A = n_B = 3, delta^2_AB = 9, D_o = 3, D_e = 5.4 -> 1 - 3/5.4 = 4/9
    assert sp.krippendorff_alpha_ordinal([["A", "A"], ["B", "B"], ["A", "B"]], ("A", "B", "C")) == pytest.approx(0.4444, abs=1e-4)
    assert sp.krippendorff_alpha_ordinal([["A", "A"], ["B", "B"], ["C", "C"]], ("A", "B", "C")) == 1.0
    L = ("TIDAK_MENJAWAB", "MENJAWAB_SEBAGIAN", "MENJAWAB")
    rows = {1: dict(zip(IDS, [L[2]] * 3)), 2: dict(zip(IDS, [L[2], L[2], L[1]])), 3: dict(zip(IDS, [L[0], L[1], None]))}
    ag = sp.agreement(rows, "ansrel")
    assert ag["n_keys_three_usable"] == 2 and ag["pct_all_three_agree"] == 50.0
    assert ag["pairwise"]["llama_vs_deepseek"] == {"n": 3, "pct_agree": 66.67}     # (agree, agree, differ)
    assert ag["pairwise"]["llama_vs_qwen3-235b"]["n"] == 2 and ag["pairwise"]["llama_vs_qwen3-235b"]["pct_agree"] == 50.0
    assert ag["fleiss_kappa"] is not None and ag["krippendorff_alpha_ordinal"] is not None


# --- cap, retry, resume, torn tail --------------------------------------------------------------------------------------------

def test_cap_stops_before_exceeding_and_reports_skipped(tmp_path):
    out = tmp_path / "o.jsonl"
    clients = _clients(lambda j, m, n: _ans("MENJAWAB"), tokens=(3000, 400))
    guard = jp.BudgetGuard(0.012, 2000, lambda msgs: 3000)       # the worst case of one call (all retries) is ~0.005-0.007
    stats = _run(_items(10), clients, out, guard)
    made = sum(c.n for c in clients.values())
    assert 0 < made < 30 and stats["calls"] == made and guard.n_skipped == 30 - made
    assert guard.spent <= guard.max_cost_usd


def test_failed_and_unparseable_calls_go_to_the_failures_file_and_are_retried_on_resume(tmp_path):
    out = tmp_path / "o.jsonl"

    def flaky(judge, messages, n):                       # qwen never returns a parseable answer for question 2
        return "not json" if judge == "qwen3-235b" and "Q2" in messages[0]["content"] else _ans("MENJAWAB")

    s1 = _run(_items(3), _clients(flaky), out)
    assert s1["usable"] == 8 and s1["failed"] == 1
    assert len(_recs(out)) == 8 and len(_recs(sp.failures_path_of(out))) == 1
    bad = _recs(sp.failures_path_of(out))[0]
    assert bad["judge_id"] == "qwen3-235b" and bad["n_paid_attempts"] == 1 + sp.MAX_RETRIES and bad["label"] is None
    assert bad["estimated_cost_usd"] > 0                  # every parse-failure attempt was paid
    c2 = _clients(lambda j, m, n: _ans("MENJAWAB"))       # the judge works now: only the failed call is made again
    s2 = _run(_items(3), c2, out)
    assert s2["calls"] == 1 and s2["cached"] == 8 and sum(c.n for c in c2.values()) == 1
    assert len(_recs(out)) == 9


def test_torn_last_line_is_repaired_and_the_record_is_redone(tmp_path):
    from llm.common.resume import repair_torn_tail

    out = tmp_path / "o.jsonl"
    _run(_items(2), _clients(lambda j, m, n: _ans("MENJAWAB")), out)
    text = out.read_text()
    out.write_text(text[:-40])                            # the process died while writing the last record
    assert len(_recs(out)) == 5                           # readers skip the torn tail
    repair_torn_tail(out, lambda *a: None)
    assert out.read_text().endswith("\n")
    c = _clients(lambda j, m, n: _ans("MENJAWAB"))
    assert _run(_items(2), c, out)["calls"] == 1 and len(_recs(out)) == 6


# --- ledger, identity, registration ---------------------------------------------------------------------------------------------

def test_ledger_is_booked_once_tagged_to_the_output_file(tmp_path):
    out = tmp_path / "ansrel_panel_official.jsonl"
    _run(_items(2), _clients(lambda j, m, n: _ans("MENJAWAB"), tokens=(1_000_000, 0)), out)
    cost = sp._stored_cost(out)
    assert cost == pytest.approx(2 * 1.0 * (0.10 + 0.26 + 0.09))
    assert jp.reconcile_ledger(out, "official", lambda *a: None) == pytest.approx(cost)
    assert jp.reconcile_ledger(out, "official", lambda *a: None) == 0.0
    e = rr.load()["deepinfra_spend"]
    assert len(e) == 1 and e[0]["file"] == out.name


def _stage_files(tmp_path, n=3):
    f = tmp_path / "c.jsonl"
    f.write_text("".join(json.dumps({"question_id": i, "title": "T", "llm_answer": "a", "config": {"sample_split": "dev"}}) + "\n"
                         for i in range(1, n + 1)))
    return {"C": ("C", str(f))}


def test_identity_per_judge_ok_mismatch_and_unusable_limit(tmp_path, monkeypatch):
    import llm.sample_split as ss

    monkeypatch.setattr(ss, "load_frozen_ids", lambda split, path=None: {"ids": [1, 2, 3] if split == "test" else [99]})
    runs = _stage_files(tmp_path)
    out = tmp_path / "o.jsonl"
    key = str(sm.archive_paths.canonical(runs["C"][1]))
    items = [{"source_file": key, "question_id": i, "messages": [{"role": "user", "content": f"Q{i}"}]} for i in (1, 2, 3)]
    _run(items, _clients(lambda j, m, n: _ans("MENJAWAB")), out)
    ident = sp.identity_report("ansrel", runs, out)
    assert ident["ok"] and all(d["keys_equal_expected"] and d["unusable"] == 0 for d in ident["per_judge"].values())
    lines = out.read_text().splitlines()
    out.write_text("\n".join(l for l in lines if not ('"llama"' in l and '"question_id": 3' in l)) + "\n")
    bad = sp.identity_report("ansrel", runs, out)
    assert not bad["ok"] and bad["per_judge"]["llama"]["unusable"] == 1       # 1 of 3 > the 2% limit
    out.write_text("\n".join(lines) + "\n" + json.dumps({**json.loads(lines[0]), "question_id": 99}) + "\n")
    assert sp.identity_report("ansrel", runs, out)["per_judge"]["llama"]["dev_ids"] == 1


def test_registration_records_an_official_judge_entry(tmp_path):
    paths = sp.out_paths("ansrel", tmp_path)
    sp.register_stage("official", "ansrel", {"A": ("A", "a.jsonl")}, paths, 0.5, "2026-10-28T00:00:00+00:00", 10, lambda *a: None)
    e = rr.load()["runs"][-1]
    assert e["kind"] == "judge" and e["status"] == "official" and e["judge_config"]["metric"] == "ansrel"
    assert [j["judge_id"] for j in e["judge_config"]["judges"]] == IDS


# --- stage selection ---------------------------------------------------------------------------------------------------------------

def test_a_stage_needs_its_metric_and_its_conditions(tmp_path, capsys):
    assert sp.main(["--run", "--max-cost-usd", "1"]) == 2 and "--metric" in capsys.readouterr().out

    class Args:
        max_cost_usd, out_dir, workers = None, None, 1

    runs = {"A": ("A", str(tmp_path / "a.jsonl"))}
    assert sp.run_guarded("ansrel", runs, ["x"], Args) == 2 and "--max-cost-usd" in capsys.readouterr().out
    Args.max_cost_usd = 1.0
    assert sp.run_guarded("ctxrel", runs, [_stage_files(tmp_path)["C"][1]], Args) == 2   # ctxrel has no condition A
    assert "covers conditions BCD" in capsys.readouterr().out


def test_official_files_refuse_a_foreign_out_dir(tmp_path, capsys):
    f = tmp_path / "c.jsonl"
    f.write_text(json.dumps({"question_id": 1, "config": {}}) + "\n")           # absent sample_split = test = official

    class Args:
        max_cost_usd, out_dir, workers = 1.0, str(tmp_path / "elsewhere"), 1

    assert sp.run_guarded("ansrel", {"C": ("C", str(f))}, [str(f)], Args) == 2 and "no --out-dir" in capsys.readouterr().out


# --- end to end on a dev file (no registry / ledger), then the report ----------------------------------------------------------------

def test_dev_run_end_to_end_and_report_with_only_one_stage(tmp_path, monkeypatch):
    runs = _stage_files(tmp_path, 4)
    path = runs["C"][1]
    key = str(sm.archive_paths.canonical(path))
    labels = {1: ["MENJAWAB"] * 3, 2: ["MENJAWAB", "MENJAWAB", "TIDAK_MENJAWAB"],
              3: ["TIDAK_MENJAWAB", "MENJAWAB_SEBAGIAN", "MENJAWAB"], 4: ["TIDAK_MENJAWAB"] * 3}

    def behaviour(judge, messages, n):
        q = int(messages[0]["content"][1:])
        return _ans(labels[q][IDS.index(judge)])

    monkeypatch.setattr(sp, "build_items", lambda metric, runs_: [
        {"source_file": key, "question_id": q, "messages": [{"role": "user", "content": f"Q{q}"}]} for q in labels])
    ns = types.SimpleNamespace(max_cost_usd=5.0, out_dir=str(tmp_path / "out"), workers=1)
    assert sp.run_guarded("ansrel", runs, [path], ns, clients=_clients(behaviour)) == 0
    panel = [json.loads(l) for l in open(sp.out_paths("ansrel", ns.out_dir)["panel"])]
    by_q = {p["key"][1]: p for p in panel}
    assert by_q[1]["panel_label"] == "MENJAWAB" and by_q[2]["decided_by"] == "majority"
    assert by_q[3]["panel_label"] == "MENJAWAB_SEBAGIAN" and by_q[3]["decided_by"] == "median" and by_q[4]["panel_score"] == 0.0
    # nothing was booked or registered for a dev file
    assert rr.load().get("deepinfra_spend", []) == [] and not [r for r in rr.load()["runs"] if r["kind"] == "judge"]

    rep = sp.report(runs, ns.out_dir, results_dir=tmp_path)
    assert rep["ctxrel"] == {"status": "not run"}                                  # the missing stage is "not run", not zero
    a = rep["ansrel"]["all_cells"]["all_answers"]
    assert rep["ansrel"]["status"] == "run" and a["n_decided"] == 4 and a["decided_by_median"] == 1
    assert a["panel_labels"] == {"TIDAK_MENJAWAB": 1, "MENJAWAB_SEBAGIAN": 1, "MENJAWAB": 2}
    assert a["panel_mean_score"] == pytest.approx((1 + 1 + 0.5 + 0) / 4, abs=1e-3)
    assert a["per_judge"]["llama"]["mean_score"] == pytest.approx((1 + 1 + 0 + 0) / 4)
    assert a["agreement"]["pct_all_three_agree"] == 50.0
    assert "non_abstain" not in rep["ansrel"]["all_cells"] and "not available" in rep["ansrel"]["non_abstain_status"]
    # non-ABSTAIN version: split by the hallucination panel's ABSTAIN labels (question 4 abstained)
    (tmp_path / "panel_official_R5_judge-v2.3_panel.jsonl").write_text(
        json.dumps({"condition": "C", "question_id": 4, "panel_label": "ABSTAIN"}) + "\n"
        + json.dumps({"condition": "C", "question_id": 1, "panel_label": "FAKTUAL"}) + "\n")
    rep2 = sp.report(runs, ns.out_dir, results_dir=tmp_path)["ansrel"]["all_cells"]
    assert rep2["non_abstain"]["n_keys"] == 3 and rep2["all_answers"]["n_keys"] == 4
    assert rep2["non_abstain"]["panel_mean_score"] == pytest.approx((1 + 1 + 0.5) / 3, abs=1e-3)


def test_report_with_no_stage_at_all(tmp_path):
    assert sp.report(None, tmp_path) == {"ansrel": {"status": "not run"}, "ctxrel": {"status": "not run"}}
