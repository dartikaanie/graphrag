"""Condition B's KG-answer corpus: file layout, cache key, and the RUNTIME
side (load the FAISS index, search with per-query leakage exclusion, look
the hits' text up in kg_answers.parquet via DuckDB).

This module never imports neo4j. The only code that talks to Neo4j is the
offline export stage of build_kg_answer_index.py.

Layout under <index_cache_dir>/<corpus_dir_name()>/, e.g.
llm/b_rag/index_cache/kg_answers_v1_tok400/:

    kg_answers.parquet          answer_id, question_id, question_title,
                                answer_body, is_accepted, rel_type,
                                doc_text (= truncate_doc_text(answer_doc_text(title, body))),
                                truncated -- sorted by answer_id
    export_manifest.json        n_rows, ids_sha256, export stats
    <model>/embeddings.f32      float32 memmap (n_rows x 384), L2-normalized, row i = parquet row i
    <model>/ids.i64             int64 memmap, answer_id per row
    <model>/encode_checkpoint.json
    <cache_key>.faiss           IndexFlatIP over embeddings.f32
    <cache_key>_meta.parquet    answer_id, question_id, is_accepted (row i = vector i)
    <cache_key>.manifest.json

cache_key = kg_answers_v1_tok<limit>_<model>_<first 12 hex of ids_sha256>,
so it changes whenever the corpus version, truncation, embedding model or
the exported answer id list changes.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CORPUS_VERSION = "kg_answers_v1"
EMBED_DIM = 384  # all-MiniLM-L6-v2
CORPUS_KG = "kg_answers"
CORPUS_LEGACY = "legacy_sample"
CORPUS_CHOICES = (CORPUS_KG, CORPUS_LEGACY)


def corpus_dir_name(token_chunk_limit: int) -> str:
    return f"{CORPUS_VERSION}_tok{token_chunk_limit}"


def safe_model_name(embed_model_name: str) -> str:
    return embed_model_name.replace("/", "_")


def cache_key(token_chunk_limit: int, embed_model_name: str, ids_sha256: str) -> str:
    return f"{corpus_dir_name(token_chunk_limit)}_{safe_model_name(embed_model_name)}_{ids_sha256[:12]}"


class IdsDigest:
    """sha256 over the answer ids as little-endian int64, in row order --
    streamable (update per batch == hashing the whole array at once)."""

    def __init__(self):
        self._h = hashlib.sha256()

    def update(self, ids) -> None:
        self._h.update(np.asarray(ids, dtype="<i8").tobytes())

    def hexdigest(self) -> str:
        return self._h.hexdigest()


def ids_sha256(ids) -> str:
    d = IdsDigest()
    d.update(ids)
    return d.hexdigest()


@dataclass(frozen=True)
class CorpusPaths:
    root: Path
    token_chunk_limit: int
    embed_model_name: str

    @property
    def parquet(self) -> Path:
        return self.root / "kg_answers.parquet"

    @property
    def export_manifest(self) -> Path:
        return self.root / "export_manifest.json"

    @property
    def embed_dir(self) -> Path:
        return self.root / safe_model_name(self.embed_model_name)

    @property
    def embeddings(self) -> Path:
        return self.embed_dir / "embeddings.f32"

    @property
    def embed_ids(self) -> Path:
        return self.embed_dir / "ids.i64"

    @property
    def encode_checkpoint(self) -> Path:
        return self.embed_dir / "encode_checkpoint.json"

    def key(self, ids_sha: str) -> str:
        return cache_key(self.token_chunk_limit, self.embed_model_name, ids_sha)

    def index(self, ids_sha: str) -> Path:
        return self.root / f"{self.key(ids_sha)}.faiss"

    def index_meta(self, ids_sha: str) -> Path:
        return self.root / f"{self.key(ids_sha)}_meta.parquet"

    def index_manifest(self, ids_sha: str) -> Path:
        return self.root / f"{self.key(ids_sha)}.manifest.json"


def corpus_paths(index_cache_dir, token_chunk_limit: int, embed_model_name: str) -> CorpusPaths:
    return CorpusPaths(Path(index_cache_dir) / corpus_dir_name(token_chunk_limit), token_chunk_limit, embed_model_name)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json_atomic(path: Path, data: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def encode_query(embed_model, query_text: str) -> np.ndarray:
    """Same computation as c_graphrag.embed_query(): one text per encode
    call, float32, numpy L2-normalization -- on CPU, as C loads its model
    (device pinned here because the dashboard shares a default-device
    model between A and B)."""
    emb = embed_model.encode([query_text], convert_to_numpy=True, device="cpu").astype("float32")
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1e-9
    return emb / norm


class KGAnswerIndex:
    """Runtime view of the built corpus: FAISS IndexFlatIP + id arrays +
    kg_answers.parquet for text. No Neo4j."""

    def __init__(self, paths: CorpusPaths, index, answer_ids: np.ndarray, question_ids: np.ndarray,
                 manifest: dict, con=None):
        import duckdb

        self.paths = paths
        self.index = index
        self.answer_ids = answer_ids
        self.question_ids = question_ids
        self.manifest = manifest
        self.cache_key = paths.key(manifest["ids_sha256"])
        self._con = con or duckdb.connect()
        self._parquet_sql = str(paths.parquet).replace("'", "''")

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def thread_answer_ids(self, question_ids) -> dict:
        """question_id -> every answer_id of that thread in kg_answers.parquet
        (the per-query exclusion set's answer half)."""
        qids = [int(q) for q in question_ids]
        if not qids:
            return {}
        df = self._con.execute(
            f"SELECT question_id, answer_id FROM read_parquet('{self._parquet_sql}') "
            f"WHERE question_id IN ({','.join(map(str, qids))}) ORDER BY answer_id"
        ).df()
        out = {q: [] for q in qids}
        for q, a in zip(df["question_id"], df["answer_id"]):
            out[int(q)].append(int(a))
        return out

    def search(self, query_emb: np.ndarray, top_k: int, exclusion=None, stats: dict | None = None) -> list[tuple[int, float]]:
        """Top-k (row, score) with the exclusion applied BEFORE the cut:
        over-fetch top_k + |excluded answers| + 5 (every answer of every
        excluded question is in exclusion.answer_ids, so at most that many
        hits can be dropped), widen further if needed, filter, cut to top_k.
        Fewer than top_k only if the whole corpus runs out. `stats`
        (optional): stats["n_hits_filtered"] = hits dropped by the exclusion
        in the pass that produced the result."""
        n_excl = len(exclusion.answer_ids) if exclusion is not None else 0
        fetch = min(self.ntotal, top_k + n_excl + 5)
        while True:
            scores, rows = self.index.search(np.ascontiguousarray(query_emb, dtype="float32"), fetch)
            kept, dropped = [], 0
            for score, row in zip(scores[0], rows[0]):
                if row == -1:
                    continue
                if exclusion is not None and exclusion.excludes(self.question_ids[row], self.answer_ids[row]):
                    dropped += 1
                    continue
                kept.append((int(row), float(score)))
                if len(kept) >= top_k:
                    break
            if stats is not None:
                stats["n_hits_filtered"] = dropped
            if len(kept) >= top_k or fetch >= self.ntotal:
                return kept
            fetch = min(self.ntotal, fetch * 2)

    def fetch_docs(self, answer_ids: list[int]) -> dict:
        """answer_id -> row dict (question_id, question_title, doc_text,
        is_accepted) from kg_answers.parquet."""
        if not answer_ids:
            return {}
        df = self._con.execute(
            f"SELECT answer_id, question_id, question_title, doc_text, is_accepted "
            f"FROM read_parquet('{self._parquet_sql}') "
            f"WHERE answer_id IN ({','.join(str(int(a)) for a in answer_ids)})"
        ).df()
        return {int(r["answer_id"]): r for r in df.to_dict("records")}


def load_kg_answer_index(index_cache_dir, token_chunk_limit: int, embed_model_name: str, log=print) -> KGAnswerIndex:
    """Load the index built by build_kg_answer_index.py and check that the
    parquet, the meta and the index agree (row counts and id order)."""
    import faiss
    import pyarrow.parquet as pq

    paths = corpus_paths(index_cache_dir, token_chunk_limit, embed_model_name)
    if not paths.export_manifest.exists():
        raise SystemExit(
            f"[ABORT] KG answer corpus not found at {paths.root}. Build it first:\n"
            f"    python llm/b_rag/build_kg_answer_index.py --stage all --token-chunk-limit {token_chunk_limit}\n"
            f"(or run with --corpus legacy_sample to reproduce the old 8,000-answer pilot index)."
        )
    export = read_json(paths.export_manifest)
    sha = export["ids_sha256"]
    index_path, meta_path, manifest_path = paths.index(sha), paths.index_meta(sha), paths.index_manifest(sha)
    if not (index_path.exists() and meta_path.exists() and manifest_path.exists()):
        raise SystemExit(
            f"[ABORT] FAISS index {index_path.name} missing for the current kg_answers.parquet "
            f"(ids_sha256={sha[:12]}). Run build_kg_answer_index.py --stage encode and --stage index."
        )
    manifest = read_json(manifest_path)
    meta = pq.read_table(meta_path, columns=["answer_id", "question_id"])
    answer_ids = meta.column("answer_id").to_numpy().astype(np.int64)
    question_ids = meta.column("question_id").to_numpy().astype(np.int64)
    index = faiss.read_index(str(index_path))
    n_parquet = pq.ParquetFile(paths.parquet).metadata.num_rows
    if not (index.ntotal == len(answer_ids) == n_parquet == export["n_rows"] == manifest["n_vectors"]):
        raise SystemExit(
            f"[ABORT] KG corpus mismatch: index={index.ntotal} meta={len(answer_ids)} parquet={n_parquet} "
            f"export_manifest={export['n_rows']} index_manifest={manifest['n_vectors']} -- rebuild the index."
        )
    if ids_sha256(answer_ids) != sha or manifest["ids_sha256"] != sha:
        raise SystemExit("[ABORT] KG corpus id order differs between parquet and index meta -- rebuild the index.")
    log(f"[6/8] KG answer index dimuat: {index.ntotal:,} vektor ({paths.key(sha)}) dari {paths.root}")
    manifest = {**manifest, "ids_sha256": sha}
    return KGAnswerIndex(paths, index, answer_ids, question_ids, manifest)
