"""The question index (~4.15 GB, read fully into RAM) is read at most once per
process in an integration session; query vectors are owned float32 copies,
not memmap views; a stored nprobe other than 32 STOPs the load."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from llm.common.question_index import EXPECTED_NPROBE, query_vector, read_question_index
from question_index_spy import QuestionIndexReadSpy


def test_spy_allows_one_question_index_read_and_fails_on_the_second():
    fake = SimpleNamespace(read_index=lambda path, *a: f"index:{path}")
    spy = QuestionIndexReadSpy(fake).install()
    try:
        assert fake.read_index("/ws/question_embeddings_ivf.faiss") == "index:/ws/question_embeddings_ivf.faiss"
        assert fake.read_index("/other/kg_answers.faiss") == "index:/other/kg_answers.faiss"  # not counted
        assert fake.read_index("/other/kg_answers.faiss")
        with pytest.raises(AssertionError, match="second time"):
            fake.read_index(Path("/ws/question_embeddings_ivf.faiss"))
        assert spy.reads == ["/ws/question_embeddings_ivf.faiss"]
    finally:
        spy.uninstall()
    assert fake.read_index("/ws/question_embeddings_ivf.faiss")  # original restored


def test_query_vector_is_an_owned_float32_copy_with_the_same_values(tmp_path):
    path = tmp_path / "emb.f32"
    data = np.random.default_rng(0).standard_normal((5, 384)).astype("float32")
    data.tofile(path)
    emb = np.memmap(path, dtype="float32", mode="r", shape=(5, 384))
    view = np.ascontiguousarray(emb[2:3])
    assert isinstance(view, np.memmap) or view.base is not None  # the old way: still backed by the file
    q = query_vector(emb, 2)
    assert type(q) is np.ndarray and q.flags.owndata and q.flags.c_contiguous
    assert q.dtype == np.float32 and q.shape == (1, 384)
    assert np.array_equal(q, data[2:3])


@pytest.mark.parametrize("nprobe,ok", [(EXPECTED_NPROBE, True), (7, False)])
def test_read_question_index_checks_stored_nprobe_and_never_sets_it(tmp_path, nprobe, ok):
    faiss = pytest.importorskip("faiss")
    xb = np.random.default_rng(1).standard_normal((200, 8)).astype("float32")
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(8), 8, 4, faiss.METRIC_INNER_PRODUCT)
    index.train(xb)
    index.add(xb)
    index.nprobe = nprobe
    path = tmp_path / "tiny_ivf.faiss"  # not the spied name: this is a 200-vector stand-in
    faiss.write_index(index, str(path))
    if ok:
        assert read_question_index(path).nprobe == EXPECTED_NPROBE
    else:
        with pytest.raises(SystemExit, match="nprobe=7"):
            read_question_index(path)


@pytest.mark.skipif(os.getenv("GRAPHRAG_INTEGRATION") != "1", reason="reads the KG workspace; set GRAPHRAG_INTEGRATION=1")
def test_integration_session_reads_the_question_index_exactly_once(question_faiss, question_index_read_spy):
    import c_graphrag as cg

    assert question_faiss[0].nprobe == EXPECTED_NPROBE
    assert len(question_index_read_spy.reads) == 1
    with pytest.raises(AssertionError, match="second time"):  # guard is live: a 2nd load fails before reading
        cg.load_faiss_cache(Path(os.environ["KG_WORKSPACE_DIR"]), lambda *a, **k: None)
    assert len(question_index_read_spy.reads) == 1
