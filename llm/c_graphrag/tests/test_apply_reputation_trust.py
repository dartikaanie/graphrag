"""The Neo4j update (01_data_cleaning/13_apply_reputation_trust.py) must use
exactly the repfix formula evaluated in Amendment 5."""

import importlib.util
from pathlib import Path

import numpy as np

import reputation_ablation_check as rac

_SPEC = importlib.util.spec_from_file_location(
    "apply_reputation_trust",
    Path(__file__).resolve().parents[3] / "01_data_cleaning" / "13_apply_reputation_trust.py")
art = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(art)


def test_db_update_formula_matches_repfix():
    old = np.array([0.10, 0.35, 0.892, 0.60])
    w = np.array([np.nan, 0.0, 1.0, 0.55])
    cands = [{"answer_id": i, "answer_trust_score": t} for i, t in enumerate(old)]
    weights = {i: x for i, x in enumerate(w) if not np.isnan(x)}
    expected = [c["answer_trust_score"] for c in rac.apply_reputation_fix(cands, weights)]
    assert np.allclose(art.compute_new_trust(old, w), expected)


def test_bounds():
    out = art.compute_new_trust(np.array([0.1, 0.95]), np.array([0.0, 1.0]))
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isclose(out[0], 0.0)
