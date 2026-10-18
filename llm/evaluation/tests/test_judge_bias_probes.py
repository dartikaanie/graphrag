"""judge_bias_probes.py (Amendment 11) with a fake judge: sampling, flip rates, point-biserial, dev-only."""

import json

import pytest

import judge_bias_probes as bp
import llm_judge_hallucination_v2_2 as runner

CTX = [{"question_id": 11, "chunk_text": "first item"}, {"question_id": 22, "chunk_text": "second item"}]


def _items(n_per=20):
    out = []
    for cond in ("A", "B", "C", "D"):
        for q in range(n_per):
            out.append({"run_id": f"{cond}-run", "condition": cond, "run_label": cond, "question_id": q,
                        "title": "t", "tags": "", "source_file": f"/{cond}.jsonl", "ground_truth_answer_html": "r",
                        "candidate": ("x " * (q + 1)) + "[SO-11]", "retrieved_context": [] if cond == "A" else CTX})
    return out


class _Cfg:
    model = "fake-model"
    base_url = None


def _payload(halu):
    claims = [{"claim": "c", "verdict": "FABRICATED" if halu else "SUPPORTED", "severity": "CORE" if halu else None,
               "reference_conflict": False, "supported_by_context": "NO", "cited_items": [],
               "cited_item_supports": None, "evidence": "e"}]
    return {"raw": json.dumps({"answer_attempted": True, "completeness": "FULL", "approach_match": "SAME",
                               "claims": claims, "reasoning": "r", "label": "FAKTUAL"}),
            "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.1, "response_id": "x",
            "response_model": "fake-model", "response_created": 1, "system_fingerprint": None,
            "finish_reason": "stop", "usage_full": {}, "request_params": {}}


def test_sample_is_40_spread_over_b_c_d_and_deterministic():
    s = bp.sample_probe_items(_items())
    assert len(s) == 40 and {i["condition"] for i in s} == {"B", "C", "D"}
    assert sorted(sum(1 for i in s if i["condition"] == c) for c in "BCD") == [13, 13, 14]
    assert [(i["condition"], i["question_id"]) for i in s] == \
        [(i["condition"], i["question_id"]) for i in bp.sample_probe_items(list(reversed(_items())))]
    with pytest.raises(bp.ProbeError):
        bp.sample_probe_items([i for i in _items() if i["condition"] == "A"])


def test_shuffled_context_is_fixed_per_item():
    it = {**_items()[30], "retrieved_context": [{"question_id": i, "chunk_text": str(i)} for i in range(5)]}
    assert bp.shuffled_context(it) == bp.shuffled_context(it) and bp.shuffled_context(it) != it["retrieved_context"]


def test_probes_with_fake_judge(monkeypatch):
    """The fake judge flags HALU when the candidate shows a [K label (authority) or when the second context
    item comes first (order); otherwise not hallucinated."""
    def fake(client, config, messages, temperature, log, max_tokens):
        user = messages[1]["content"]
        cand = user.split("### CANDIDATE ANSWER")[1]
        ctx = user.split("### RETRIEVED CONTEXT")[1]
        halu = "[K" in cand or ctx.index("second item") < ctx.index("first item")
        return _payload(halu), []

    monkeypatch.setattr(runner, "call_judge_llm_with_retry", fake)
    sample = bp.sample_probe_items(_items())
    rep = bp.run_probes(sample, "primary", None, _Cfg(), {}, {})
    a, o = rep["authority"], rep["order"]
    assert a["n_pairs"] == 40 and a["n_unusable"] == 0
    n_base_halu = sum(1 for r in rep["records"]["blind"] if r["hallucinated"])
    assert a["flips_halu"] == 40 - n_base_halu == a["to_hallucinated"] and a["to_not_hallucinated"] == 0
    expected_order_flips = sum(1 for it, r in zip(sample, rep["records"]["blind"])
                               if not r["hallucinated"] and bp.shuffled_context(it)[0]["chunk_text"] == "second item")
    assert o["flips_halu"] == expected_order_flips
    assert all(r["probe_citation_labels_kept"] for r in rep["records"]["citation_labels"])
    assert set(rep["length"]) == {"B", "C", "D"}
    md = bp.probe_markdown(rep)
    assert "authority: [Kn] kept vs blind-v3" in md and "point-biserial" in md


def test_point_biserial_known_value():
    assert bp.point_biserial([1, 2, 3, 4], [0, 0, 1, 1]) == pytest.approx(0.8944, abs=1e-4)
    assert bp.point_biserial([1, 2, 3], [1, 1, 1]) is None


def test_flip_stats_counts():
    H = {"answer_attempted": True, "claims": [{"verdict": "FABRICATED"}]}
    N = {"answer_attempted": True, "claims": [{"verdict": "SUPPORTED"}]}
    A = {"answer_attempted": False, "claims": []}
    U = {"parse_error": True}
    s = bp.flip_stats([H, N, A, N, U], [N, N, N, H, N])
    assert (s["n_usable"], s["flips_3cat"], s["n_both_non_abstain"], s["flips_halu"]) == (4, 3, 3, 2)
    assert (s["to_hallucinated"], s["to_not_hallucinated"]) == (1, 1)


def test_cli_plan_makes_no_call_and_refuses_test_runs(tmp_path, monkeypatch, capsys):
    run = tmp_path / "test_run.jsonl"
    run.write_text(json.dumps({"question_id": 1, "config": {"sample_split": "test"}}) + "\n")
    assert bp.main(["--run-files", str(run)]) == 1 and "dev set only" in capsys.readouterr().out
    monkeypatch.setattr(runner, "call_judge_llm_with_retry", lambda *a: pytest.fail("no call in plan mode"))
    monkeypatch.setattr(runner, "load_items", lambda files, limit: _items())
    dev = tmp_path / "dev_run.jsonl"
    dev.write_text(json.dumps({"question_id": 1, "config": {"sample_split": "dev"}}) + "\n")
    assert bp.main(["--run-files", str(dev)]) == 0
    assert "120 judge calls" in capsys.readouterr().out
