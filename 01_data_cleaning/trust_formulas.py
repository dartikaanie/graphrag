"""
trust_formulas.py
=================
Single source of truth for the node trustScore math (Tabel II.2,
docs/DECISION_C_SCORING.md Amendments 5 and 6). Pure, vectorised functions
with no I/O: the KG build (7_build_knowledge_graph.py), the Neo4j update
(14_apply_logscore_trust.py) and the read-only checks in llm/c_graphrag/
all import from here, so the same inputs always give the same float64
values.

    norm_log(score; P99) = clip(log1p(max(score, 0)) / log1p(P99), 0, 1)
    w_auth               = clip(log1p(rep) / log1p(P99_rep), 0, 1)

    trust_Q = 0.7 * norm_log(score_Q; P99_Q) + 0.3 * minmax(log1p(views))
    trust_A = 0.5 * norm_log(score_A; P99_A) + 0.3 * is_accepted + 0.2 * w_auth

P99 is the 99th percentile of the raw score column (negatives included) over
ALL nodes of the same type, computed with numpy's "linear" method (the numpy
and pandas default; the same definition as DuckDB's quantile_cont).

Missing values: a missing (NaN/None) score or view count is treated as 0 and
a missing w_auth as 0 (answer without an AUTHOR_TRUST edge), exactly as the
KG build has always done with fillna(0).
"""

import numpy as np

TRUST_SCORE_VERSION = "v3-logscore"

P99_QUANTILE = 0.99
QUANTILE_METHOD = "linear"

W_Q_SCORE = 0.7
W_Q_VIEWS = 0.3
W_A_SCORE = 0.5
W_A_ACCEPTED = 0.3
W_A_AUTHOR = 0.2


def _filled(values) -> np.ndarray:
    """float64 array with missing values replaced by 0."""
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)


def compute_p99(scores, method: str = QUANTILE_METHOD) -> float:
    """99th percentile of the raw scores (negatives included). Raises if the
    input is empty or the result is <= 0 (norm_log would be undefined)."""
    arr = _filled(scores)
    if arr.size == 0:
        raise ValueError("compute_p99: no scores")
    p99 = float(np.quantile(arr, P99_QUANTILE, method=method))
    if p99 <= 0:
        raise ValueError(f"compute_p99: P99 = {p99} <= 0, cannot normalise")
    return p99


def norm_log(score, p99: float) -> np.ndarray:
    """clip(log1p(max(score, 0)) / log1p(p99), 0, 1). Negative and missing
    scores give 0; scores at or above p99 give 1."""
    if not p99 > 0:
        raise ValueError(f"norm_log: p99 must be > 0, got {p99}")
    return np.clip(np.log1p(np.maximum(_filled(score), 0.0)) / np.log1p(p99), 0.0, 1.0)


def minmax(values) -> np.ndarray:
    """Global min-max to [0, 1]; a constant input gives 0.5 everywhere."""
    arr = _filled(values)
    if arr.size == 0:
        return arr
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.full(arr.shape, 0.5)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def author_weight(reputation, p99_reputation: float) -> np.ndarray:
    """w_auth, the AUTHOR_TRUST edge weight (Amendment 5, unchanged)."""
    return np.clip(np.log1p(np.maximum(_filled(reputation), 0.0)) / np.log1p(p99_reputation), 0.0, 1.0)


def trust_question(score, views, p99_score: float) -> np.ndarray:
    return W_Q_SCORE * norm_log(score, p99_score) + W_Q_VIEWS * minmax(np.log1p(_filled(views)))


def trust_answer(score, is_accepted, w_auth, p99_score: float) -> np.ndarray:
    accepted = np.asarray(is_accepted, dtype=bool).astype(np.float64)
    return W_A_SCORE * norm_log(score, p99_score) + W_A_ACCEPTED * accepted + W_A_AUTHOR * _filled(w_auth)
