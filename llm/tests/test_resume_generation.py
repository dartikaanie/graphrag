"""Resume of a stopped generation run, end to end with a mocked LLM (no network, no embedding model).

The path under test is the one the four CLIs and the dashboard use: llm.common.resume.prepare_resume (official-cell
commit/hash check, torn-tail repair, config_hash-only done-gate) followed by the condition's process_sample. Condition A
stands for all four: B, C and D share the same open(output, "a") + prepare_resume mechanism.

The kill is a BaseException raised from the fake LLM call (the loop's `except Exception` does not catch it), i.e. the
process dying between two questions; a torn record is simulated by cutting the file in the middle of the last line."""
import json
from pathlib import Path

import pandas as pd
import pytest

from llm.a_pure_llm import a_baseline_replication as A
from llm.common import resume
from llm.manifest import compute_config_hash, read_already_done
from llm.prompts import PROMPT_VERSION
from llm.sample_split import load_frozen_ids

FROZEN = [int(i) for i in load_frozen_ids("test")["ids"]]
assert len(FROZEN) == 384


class Killed(BaseException):
    pass


@pytest.fixture(autouse=True)
def _no_embeddings(monkeypatch):
    monkeypatch.setattr(A, "compute_similarity", lambda *a, **k: 0.5)
    monkeypatch.setattr(A, "log", lambda *a, **k: None)
    monkeypatch.setattr(A.time, "sleep", lambda *_: None)   # the loop sleeps 0.3 s per question (rate-limit guard)
    state = {"git_commit": "c0ffee", "git_tree_clean": True, "git_status": [], "frozen_params_hash": "abc"}
    monkeypatch.setattr(resume, "_current_state", lambda: dict(state))
    return state


def _df():
    return pd.DataFrame({"Id": FROZEN, "Title": [f"t{i}" for i in FROZEN], "Body": "b", "Tags": "<python>",
                         "n_tokens": 10, "AcceptedAnswerId": [i + 1 for i in FROZEN], "AcceptedAnswerBody": "gt"})


def _llm(kill_after=None):
    calls = {"n": 0}

    def call(client, messages, model):
        if kill_after is not None and calls["n"] >= kill_after:
            raise Killed
        calls["n"] += 1
        return {"content": "answer", "usage_full": {"prompt_tokens": 100, "completion_tokens": 10}}
    call.calls = calls
    return call


def _config(split="test"):
    cfg = A.build_config("openai", "gpt-4o-mini", 384, 42, 1536, split)
    return cfg, compute_config_hash(cfg)


def _run(path, kill_after=None, split="dev"):
    """One CLI invocation: prepare_resume, then process_sample (the fake LLM dies after `kill_after` calls)."""
    cfg, h = _config(split)
    done = resume.prepare_resume(path, h, split, lambda *a: None)
    fake = _llm(kill_after)
    try:
        A.process_sample(_df(), None, fake, None, "gpt-4o-mini", path, done, config=cfg, config_hash=h)
    except Killed:
        pass
    return fake.calls["n"], done


def _ids(path):
    return [json.loads(line)["question_id"] for line in open(path) if line.strip()]


def _parseable_ids(path):
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line)["question_id"])
        except Exception:
            pass
    return out


def test_killed_after_n_questions_and_resumed_gives_384_unique_in_frozen_order(tmp_path):
    out = tmp_path / "condition_a.jsonl"
    n1, _ = _run(out, kill_after=137)
    assert n1 == 137 and _ids(out) == FROZEN[:137]            # unit of resume = one question
    n2, done = _run(out)
    assert done == set(FROZEN[:137]) and n2 == 384 - 137      # only the missing 247 are generated, nothing re-paid
    assert _ids(out) == FROZEN and len(set(_ids(out))) == 384
    n3, _ = _run(out)
    assert n3 == 0 and _ids(out) == FROZEN                    # a finished run resumed again is a no-op


def test_resume_as_the_cli_calls_it_continues_instead_of_restarting(tmp_path):
    """Was a strict xfail: the CLIs gated on prompt_version, which the records no longer carry (schema r2)."""
    out = tmp_path / "f.jsonl"
    _run(out, kill_after=137)
    rec = json.loads(open(out).readline())
    assert "prompt_version" not in rec                        # still true: that is why the gate must not use it
    cfg, h = _config("dev")
    assert read_already_done(out, PROMPT_VERSION, h) == set()  # the old call matches nothing ...
    assert resume.prepare_resume(out, h, "dev", lambda *a: None) == set(FROZEN[:137])   # ... the new one resumes


def test_a_torn_last_line_is_cut_off_and_the_next_record_is_not_lost(tmp_path):
    """Was a strict xfail: the next record used to be glued to the torn fragment."""
    out = tmp_path / "f.jsonl"
    _run(out, kill_after=5)
    text = out.read_text()
    out.write_text(text[: len(text) - 40])                    # the process died in the middle of the 5th record
    logs = []
    cfg, h = _config("dev")
    done = resume.prepare_resume(out, h, "dev", logs.append)
    assert done == set(FROZEN[:4]) and any("torn last line" in m for m in logs)   # what was removed is logged
    _run(out)
    assert _parseable_ids(out) == FROZEN and len(open(out).readlines()) == 384    # every line parses, frozen order


def test_a_complete_record_that_only_lacks_its_newline_is_kept(tmp_path):
    out = tmp_path / "f.jsonl"
    _run(out, kill_after=3)
    out.write_text(out.read_text().rstrip("\n"))
    assert resume.repair_torn_tail(out)["added_newline"] is True
    assert _ids(out) == FROZEN[:3]


def test_repair_torn_tail_edge_cases(tmp_path):
    p = tmp_path / "x.jsonl"
    assert resume.repair_torn_tail(p)["removed_bytes"] == 0                   # missing file
    p.write_text('{"a": 1}\n{"a": 2')                                          # torn tail
    assert resume.repair_torn_tail(p)["removed_bytes"] == len('{"a": 2') and p.read_text() == '{"a": 1}\n'
    p.write_text('{"a": 1')                                                    # torn, only line
    resume.repair_torn_tail(p)
    assert p.read_text() == ""
    p.write_text('{"a": 1}\n')                                                 # clean file untouched
    assert resume.repair_torn_tail(p)["removed_bytes"] == 0 and p.read_text() == '{"a": 1}\n'


def test_a_different_config_hash_is_not_counted_as_done(tmp_path):
    out = tmp_path / "f.jsonl"
    cfg, h = _config()
    rows = [{"question_id": 1, "config_hash": h}, {"question_id": 2, "config_hash": "other"}, {"question_id": 3}]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert read_already_done(out, None, h) == {1}


def test_the_output_name_carries_the_config_hash_so_a_changed_config_starts_a_new_file(tmp_path):
    a = A.build_output_path(str(tmp_path), "openai", "gpt-4o-mini", 384, 42, 1536)
    b = A.build_output_path(str(tmp_path), "openai", "gpt-4o-mini", 384, 42, 1000)
    assert a != b and compute_config_hash(A.build_config("openai", "gpt-4o-mini", 384, 42, 1536)) in a.name


def test_a_failed_call_is_filled_in_by_the_resume_and_ids_pair_regardless_of_position(tmp_path):
    """A question whose API call failed has no record; the resume generates it at the END of the file. The file order is
    then not the frozen order, but the 384 ids are complete and unique, and everything downstream (registry, report)
    pairs questions by id."""
    out = tmp_path / "f.jsonl"
    cfg, h = _config("dev")
    state = {"n": 0}

    def flaky(client, messages, model):
        state["n"] += 1
        if state["n"] == 11:
            raise RuntimeError("API error")
        return {"content": "a", "usage_full": {"prompt_tokens": 1, "completion_tokens": 1}}
    A.process_sample(_df(), None, flaky, None, "m", out, set(), config=cfg, config_hash=h)
    assert len(_ids(out)) == 383
    _run(out)
    ids = _ids(out)
    assert sorted(ids) == sorted(FROZEN) and len(set(ids)) == 384 and ids[-1] == FROZEN[10] and ids != FROZEN


# --- official cells: commit / frozen hash of the first invocation --------------------------------------------------

def test_official_cell_resumes_under_the_same_commit_and_frozen_hash(tmp_path):
    out = tmp_path / "off.jsonl"
    _run(out, kill_after=10, split="test")
    n, done = _run(out, split="test")
    assert done == set(FROZEN[:10]) and n == 374 and _ids(out) == FROZEN
    inv = resume.read_invocations(out)
    assert [e["n_invocation"] for e in inv] == [1, 2] and inv[0]["git_commit"] == "c0ffee"   # appended, not overwritten


@pytest.mark.parametrize("key, value", [("git_commit", "deadbe"), ("frozen_params_hash", "zzz")])
def test_official_cell_refuses_to_resume_under_a_different_commit_or_frozen_hash(tmp_path, _no_embeddings, key, value):
    out = tmp_path / "off.jsonl"
    _run(out, kill_after=10, split="test")
    before = out.read_text()
    _no_embeddings[key] = value
    with pytest.raises(resume.ResumeRefused, match=key):
        _run(out, split="test")
    assert out.read_text() == before and len(resume.read_invocations(out)) == 1       # nothing appended, nothing cut


def test_official_cell_refuses_a_file_with_records_but_no_invocation_record(tmp_path):
    out = tmp_path / "off.jsonl"
    _run(out, kill_after=3, split="dev")                      # a dev-style file (no invocation record) ...
    with pytest.raises(resume.ResumeRefused, match="no invocation record"):
        _run(out, split="test")                               # ... cannot be continued as an official cell


def test_dev_runs_are_never_checked(tmp_path, _no_embeddings):
    out = tmp_path / "dev.jsonl"
    _run(out, kill_after=3, split="dev")
    _no_embeddings["git_commit"] = "other"
    _run(out, split="dev")
    assert _ids(out) == FROZEN and not resume.invocations_path(out).exists()


def test_the_manifest_lists_every_invocation():
    import inspect
    from llm import manifest
    assert "read_invocations" in inspect.getsource(manifest.write_manifest) and "invocations" in inspect.getsource(manifest.write_manifest)


@pytest.mark.parametrize("script", ["llm/a_pure_llm/a_baseline_replication.py", "llm/b_rag/b_condition_b_rag.py",
                                    "llm/c_graphrag/c_graphrag.py", "llm/d_lightrag/d_lightrag.py",
                                    "backend/app/services/engine_service.py"])
def test_every_entry_point_uses_prepare_resume_and_none_gates_on_prompt_version(script):
    src = (Path(__file__).resolve().parents[2] / script).read_text()
    assert "prepare_resume" in src
    assert "read_already_done(output_path, PROMPT_VERSION" not in src and "PROMPT_VERSION, config_hash)" not in src
