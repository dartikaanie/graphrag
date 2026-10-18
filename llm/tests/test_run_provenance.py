"""Run provenance in manifests (llm/common/provenance.py): leakage
exclusion-set hash + mode, question-index fingerprint (C, D), B corpus
index, frozen sample-ID check, and D's graph trustScoreVersion. Fake files
in tmp_path only."""

import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import llm.common.provenance as pv
from llm.a_pure_llm import a_baseline_replication as a
from llm.b_rag import b_condition_b_rag as b
from llm.d_lightrag import d_lightrag as d
from llm.manifest import write_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "llm" / "c_graphrag"))
import c_graphrag as cg  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_leakage_provenance(tmp_path):
    parquet = tmp_path / "exclusion_sets.parquet"
    parquet.write_bytes(b"fake-parquet")
    parquet.with_suffix(".meta.json").write_text(json.dumps({"sha256": _sha(b"fake-parquet")}))
    out = pv.leakage_provenance("dup+neardup0.95", parquet)
    assert out["exclusion_sets_sha256"] == _sha(b"fake-parquet")
    assert out["exclusion_sets_sha256_matches_builder"] is True and out["table_used"] is True
    assert out["mode"] == "dup+neardup0.95"
    assert pv.leakage_provenance("thread", parquet)["table_used"] is False
    parquet.write_bytes(b"changed")
    assert pv.leakage_provenance("dup", parquet)["exclusion_sets_sha256_matches_builder"] is False


def test_real_exclusion_sets_match_their_builder_record():
    out = pv.leakage_provenance("dup+neardup0.95")
    assert out["exclusion_sets_sha256_matches_builder"] is True


def test_question_index_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "_SMALL_FILE_BYTES", 16)
    monkeypatch.setattr(pv, "_PARTIAL_BYTES", 4)
    (tmp_path / "question_embeddings_ivf.faiss").write_bytes(b"0123456789abcdefghijklmnop")  # > 16: partial
    (tmp_path / "question_embeddings_ivf_ids.npy").write_bytes(b"ids")
    (tmp_path / "embed_checkpoint.json").write_text("{}")
    first = pv.question_index_provenance(tmp_path)
    files = first["files"]
    assert "partial_sha256" in files["question_embeddings_ivf.faiss"]
    assert files["question_embeddings_ivf_ids.npy"]["sha256"] == _sha(b"ids")
    assert files["question_embeddings.f32"] is None
    (tmp_path / "question_embeddings_ivf_ids.npy").write_bytes(b"ids2")
    assert pv.question_index_provenance(tmp_path)["fingerprint"] != first["fingerprint"]


def test_b_corpus_provenance(tmp_path):
    man = tmp_path / "idx.manifest.json"
    man.write_text('{"n_vectors": 3}')
    index = SimpleNamespace(cache_key="kg_answers_v1_tok400_x", manifest={"ids_sha256": "abc"}, ntotal=3,
                            paths=SimpleNamespace(index_manifest=lambda ids_sha: man))
    out = pv.b_corpus_provenance(index)
    assert out == {"corpus_cache_key": "kg_answers_v1_tok400_x", "corpus_ids_sha256": "abc", "corpus_n_vectors": 3,
                   "index_manifest_sha256": _sha(man.read_bytes())}


def test_manifest_carries_provenance_and_graph(tmp_path):
    out = tmp_path / "run.jsonl"
    out.write_text(json.dumps({"question_id": 1, "run_id": "r-1"}) + "\n")
    sample_check = {"split": "test", "file": "llm/sample_ids/test_ids_v1.json", "file_ids_sha256": "f" * 64,
                    "n_checked": 1, "checked_ids_sha256": "c" * 64}
    prov = pv.run_provenance(sample_check=sample_check, leakage_mode="dup+neardup0.95", kg_workspace_dir=tmp_path)
    m = json.loads(write_manifest(out, run_id="r-1", run_label="D-grounded", config={}, started_at_utc="t0",
                                  finished_at_utc="t1", item_counts={}, provenance=prov,
                                  graph={"trustScoreVersion": "v3-logscore"}).read_text())
    assert m["provenance"]["sample_ids"] == sample_check
    assert len(m["provenance"]["leakage"]["exclusion_sets_sha256"]) == 64
    assert "fingerprint" in m["provenance"]["question_index"]
    assert m["graph"] == {"trustScoreVersion": "v3-logscore"} and m["run_id"] == "r-1"


def test_fetch_trust_score_version_is_shared_by_c_and_d():
    assert cg.fetch_trust_score_version is pv.fetch_trust_score_version is d.fetch_trust_score_version


def test_every_cli_manifest_gets_provenance_and_d_gets_graph():
    for mod in (a, b, cg, d):
        src = inspect.getsource(mod.main)
        n_calls = src.count("write_manifest(")
        assert n_calls and src.count("run_id=run_id, provenance=provenance,") == n_calls, mod.__name__
    d_src = inspect.getsource(d.main)
    assert d_src.count("provenance=provenance, graph=graph_state,") == d_src.count("write_manifest(")
    for mod in (b, cg, d):
        assert "leakage_mode=leakage_mode" in inspect.getsource(mod.main).split("provenance = run_provenance(")[1][:200]
    for mod in (cg, d):
        assert "kg_workspace_dir=kg_workspace_dir" in inspect.getsource(mod.main).split("provenance = run_provenance(")[1][:200]
