"""The question FAISS index shared by C, D and the near-dup precompute
(question_embeddings_ivf.faiss, built by 01_data_cleaning/11_densify_embedding_similarity.py).

- read_question_index(): faiss.read_index() (whole index in RAM, ~4.15 GB) plus
  a check that the nprobe stored in the file is the one script 11 wrote (32).
  A different value STOPS the run -- it is never set here, because nprobe
  changes which IVF lists are searched and so the retrieval itself.
- query_vector(): one row of the question-embedding memmap as an owned
  float32 copy. A slice of the memmap is a view, so FAISS would read its pages
  from the external drive during search(); a failed page read there is a
  SIGBUS. The copy holds the same 384 floats, so search results are identical.
"""

from pathlib import Path

import numpy as np

EXPECTED_NPROBE = 32


def read_question_index(index_path: Path):
    import faiss

    index = faiss.read_index(str(index_path))
    nprobe = faiss.extract_index_ivf(index).nprobe
    if nprobe != EXPECTED_NPROBE:
        raise SystemExit(f"[STOP] {index_path} has nprobe={nprobe}, expected {EXPECTED_NPROBE} "
                         f"(script 11). Not changing it here -- check the index build.")
    return index


def query_vector(embeddings, row: int) -> np.ndarray:
    return np.array(embeddings[row:row + 1], dtype=np.float32, copy=True)
