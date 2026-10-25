"""Amendment 10 wiring: C-primary / C-ablation labels come from the stage-2
result, and the two official C runs differ only in alpha."""

import json

import pytest

import c_graphrag as cg
import stage2_report as s2
from llm.c_graphrag.official_params import (ROLE_ABLATION, ROLE_PRIMARY, OfficialParamsError, check_c_test_run,
                                           role_for_alpha)
from llm.manifest import compute_config_hash, write_manifest
from llm.official_run_matrix import LEAKAGE_MODE, build_matrix

CANDIDATES = s2.STAGE2_ALPHAS  # (0.25, 0.0), Amendment 9


def _official_from_stage2(winner: float) -> dict:
    """What stage2_report drafts for official_params.json from a stage-2 selection."""
    return s2.draft_official_params({"winner": winner, "keep_current": False}, alphas=CANDIDATES)


@pytest.mark.parametrize("winner", CANDIDATES)
def test_roles_follow_the_stage2_winner(winner):
    other = next(a for a in CANDIDATES if a != winner)
    official = _official_from_stage2(winner)
    assert (official["alpha_primary"], official["alpha_ablation"]) == (winner, other)
    assert role_for_alpha(official, winner) == ROLE_PRIMARY and role_for_alpha(official, other) == ROLE_ABLATION
    assert role_for_alpha(official, 0.5) is None
    locked = dict(c_retrieval_version="v3", max_hops=2, edge_types=tuple(official["edge_types"]),
                  use_author_trust=False, accepted_only=False)
    assert check_c_test_run(official, alpha=winner, **locked) == ROLE_PRIMARY
    with pytest.raises(OfficialParamsError, match="legacy"):  # Amendment 16: the ablation alpha is a legacy cell
        check_c_test_run(official, alpha=other, **locked)
    assert check_c_test_run(official, alpha=other, allow_legacy=True, **locked) == ROLE_ABLATION
    roles = {(r["role"], r["alpha"]) for r in build_matrix(official, include_legacy=True) if r["condition"] == "C"}
    assert roles == {(ROLE_PRIMARY, winner), (ROLE_ABLATION, other)}


def test_draft_refuses_a_winner_outside_the_candidates():
    with pytest.raises(s2.Stage2Error):
        s2.draft_official_params({"winner": 0.5, "keep_current": False}, alphas=CANDIDATES)


@pytest.mark.parametrize("grounding", (True, False))
def test_matrix_c_primary_and_ablation_differ_only_in_alpha(grounding):
    rows = {r["role"]: r for r in build_matrix(_official_from_stage2(0.25), include_legacy=True)
            if r["condition"] == "C" and r["grounding"] is grounding}
    p, a = rows[ROLE_PRIMARY], rows[ROLE_ABLATION]
    a = {**a, "argv": [x for x in a["argv"] if x != "--allow-legacy-cells"],
         "dashboard": {k: v for k, v in a["dashboard"].items() if k != "allow_legacy_cells"}}
    p = {**p, "argv": [x for x in p["argv"] if x != "--allow-legacy-cells"],
         "dashboard": {k: v for k, v in p["dashboard"].items() if k != "allow_legacy_cells"}}
    i = p["argv"].index("--alpha")
    assert p["argv"][:i + 1] + p["argv"][i + 2:] == a["argv"][:i + 1] + a["argv"][i + 2:]
    assert (p["argv"][i + 1], a["argv"][i + 1]) == ("0.25", "0")
    assert {k for k in p["dashboard"] if p["dashboard"][k] != a["dashboard"].get(k)} == {"alpha"}
    assert (p["cwd"], p["script"]) == (a["cwd"], a["script"])


@pytest.mark.parametrize("grounding", (True, False))
def test_c_configs_differ_only_in_alpha_and_have_their_own_hash(grounding):
    common = dict(c_retrieval_version="v3", sample_split="test", leakage_mode=LEAKAGE_MODE,
                  expansion_require_answer=True)
    args = ("openai", "gpt-4o-mini", 384, 42, 1536, 5, 3, 3, "trust_weighted", 0.7, 0.3, 0.4, grounding, True)
    primary = cg.build_config(*args, alpha=0.25, **common)
    ablation = cg.build_config(*args, alpha=0.0, **common)
    assert {k for k in set(primary) | set(ablation) if primary.get(k) != ablation.get(k)} == {"alpha"}
    assert compute_config_hash(primary) != compute_config_hash(ablation)
    assert primary["expansion_require_answer"] is True and primary["leakage"] == LEAKAGE_MODE


def test_manifest_records_the_role_outside_config(tmp_path):
    out = tmp_path / "run.jsonl"
    out.write_text("")
    kw = dict(run_label="C-trust", config={"alpha": 0.0}, started_at_utc="t", finished_at_utc="t", item_counts={})
    with_role = json.loads(write_manifest(out, official_role=ROLE_ABLATION, **kw).read_text())
    assert with_role["official_role"] == ROLE_ABLATION and "official_role" not in with_role["config"]
    assert "official_role" not in json.loads(write_manifest(out, **kw).read_text())
