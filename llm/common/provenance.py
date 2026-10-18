"""Run provenance: inputs a run reads that are not part of its config /
config_hash, recorded in the manifest (llm/manifest.write_manifest(...,
provenance=...)) so an official run can be audited afterwards.

    leakage         mode + sha256 of results/leakage/exclusion_sets.parquet
                    (and whether it equals the sha256 its builder recorded)
    question_index  fingerprint of C/D's FAISS question index (KG workspace)
    b_corpus        Condition B's KG-answer index (cache key, ids sha256,
                    sha256 of its build manifest)
    sample_ids      the frozen sample-ID check (llm/sample_split.verify_sample)

The question index files are ~4 GB each, too large to hash on every run:
the fingerprint is size + mtime of every file, the full sha256 of the small
ones (ids .npy, checkpoint) and a partial sha256 (first and last 8 MiB +
size) of the two large ones. A changed index changes the ids file, the
checkpoint or the sizes; a byte-identical copy on another disk gives the
same fingerprint.

fetch_trust_score_version() (moved here from c_graphrag.py, unchanged) is
the graph state C and D record as manifest["graph"].
"""

import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUSION_SETS_PATH = REPO_ROOT / "results" / "leakage" / "exclusion_sets.parquet"
QUESTION_INDEX_FILES = ("question_embeddings_ivf.faiss", "question_embeddings_ivf_ids.npy",
                        "question_embeddings.f32", "embed_checkpoint.json")
_SMALL_FILE_BYTES = 64 * 1024 * 1024
_PARTIAL_BYTES = 8 * 1024 * 1024


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return Path(path).name


@lru_cache(maxsize=64)
def _sha256_cached(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    st = path.stat()
    return _sha256_cached(str(path), st.st_size, st.st_mtime_ns)


@lru_cache(maxsize=64)
def _partial_sha256_cached(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(_PARTIAL_BYTES))
        if size > _PARTIAL_BYTES:
            f.seek(max(size - _PARTIAL_BYTES, _PARTIAL_BYTES))
            h.update(f.read(_PARTIAL_BYTES))
    return h.hexdigest()


def leakage_provenance(mode: str, path=EXCLUSION_SETS_PATH) -> dict:
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    recorded = json.loads(meta_path.read_text()).get("sha256") if meta_path.exists() else None
    sha = sha256_file(path)
    return {"mode": mode, "exclusion_sets_path": _rel(path), "exclusion_sets_sha256": sha,
            "exclusion_sets_sha256_matches_builder": (sha == recorded) if sha and recorded else None,
            "table_used": mode not in ("off", "thread")}


def question_index_provenance(kg_workspace_dir) -> dict:
    root = Path(kg_workspace_dir)
    files = {}
    for name in QUESTION_INDEX_FILES:
        p = root / name
        if not p.exists():
            files[name] = None
            continue
        st = p.stat()
        entry = {"size_bytes": st.st_size,
                 "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
        if st.st_size <= _SMALL_FILE_BYTES:
            entry["sha256"] = _sha256_cached(str(p), st.st_size, st.st_mtime_ns)
        else:
            entry["partial_sha256"] = _partial_sha256_cached(str(p), st.st_size, st.st_mtime_ns)
        files[name] = entry
    fingerprint = hashlib.sha256(json.dumps(
        {k: (None if v is None else {kk: vv for kk, vv in v.items() if kk != "mtime_utc"}) for k, v in files.items()},
        sort_keys=True).encode()).hexdigest()
    return {"fingerprint": fingerprint, "files": files,
            "method": "size + sha256 (files <= 64 MiB) or partial sha256 (first/last 8 MiB + size)"}


def b_corpus_provenance(index) -> dict:
    """Condition B's KGAnswerIndex (llm/b_rag/kg_corpus.py)."""
    ids_sha = index.manifest.get("ids_sha256")
    return {"corpus_cache_key": index.cache_key, "corpus_ids_sha256": ids_sha, "corpus_n_vectors": index.ntotal,
            "index_manifest_sha256": sha256_file(index.paths.index_manifest(ids_sha)) if hasattr(index, "paths") else None}


def run_provenance(*, sample_check: dict | None = None, leakage_mode: str | None = None,
                   kg_workspace_dir=None, b_index=None) -> dict:
    out = {}
    if sample_check is not None:
        out["sample_ids"] = sample_check
    if leakage_mode is not None:
        out["leakage"] = leakage_provenance(leakage_mode)
    if kg_workspace_dir is not None:
        out["question_index"] = question_index_provenance(kg_workspace_dir)
    if b_index is not None and hasattr(b_index, "cache_key"):
        out["b_corpus"] = b_corpus_provenance(b_index)
    return out


def fetch_trust_score_version(driver, database) -> str | None:
    """trustScoreVersion of the Answer nodes (docs/DECISION_C_SCORING.md
    Amendments 5-6), recorded in the run manifest. One version on every
    node -> that version; no node tagged -> None; anything else -> a
    "mixed(...)" string, which no check accepts. Read-only."""
    with driver.session(database=database) as session:
        counts = {r["version"]: r["n"] for r in session.run(
            "MATCH (a:Answer) RETURN a.trustScoreVersion AS version, count(*) AS n")}
    if not counts or set(counts) == {None}:
        return None
    if len(counts) == 1:
        return next(iter(counts))
    return "mixed(" + ", ".join(f"{v}={n}" for v, n in sorted(counts.items(), key=lambda kv: str(kv[0]))) + ")"

