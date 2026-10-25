"""Amendment 10 launch rule for Condition C on the test sample
(llm/c_graphrag/official_params.py, used by c_graphrag.main() and the
dashboard's POST /api/runs). Fake official_params dicts only."""

import inspect
from argparse import Namespace

import pytest

import c_graphrag as cg
from llm.c_graphrag.official_params import OfficialParamsError, check_c_test_run

EDGES = ["EMBED_SIM", "HAS_ACCEPTED_ANSWER", "HAS_ANSWER", "IS_RELATED_TO", "TAG_COOCCUR"]
PENDING = {"status": "pending_selection", "c_retrieval_version": "v4", "alpha_primary": None,
           "alpha_ablation": None, "selection_record": {"path": "x", "commit": None}, "sample_split": "test",
           "max_hops": 2, "edge_types": EDGES, "use_author_trust": False, "accepted_only": False}
SELECTED = {**PENDING, "status": "selected", "alpha_primary": 0.25, "alpha_ablation": 0.0,
            "selection_record": {"path": "x", "commit": "abc1234"}}


def _args(**kw):
    base = dict(sample_split="test", c_retrieval_version="v4", alpha=0.25, max_hops=2, edge_types=None,
                use_author_trust=False, accepted_only=False)
    return Namespace(**{**base, **kw})


def test_official_alphas_give_their_roles():
    assert cg.official_role_for_args(_args(alpha=0.25), SELECTED) == "c_primary"
    # Amendment 16: the second candidate (alpha_ablation) is a LEGACY cell -- refused unless explicitly allowed
    for alpha in (0.0, 0):
        with pytest.raises(OfficialParamsError, match="legacy"):
            cg.official_role_for_args(_args(alpha=alpha), SELECTED)
        assert cg.official_role_for_args(_args(alpha=alpha, allow_legacy_cells=True), SELECTED) == "c_ablation"


def test_test_split_refuses_v2():
    with pytest.raises(OfficialParamsError, match="requires --c-retrieval-version v4"):
        cg.official_role_for_args(_args(c_retrieval_version="v2", alpha=None), SELECTED)


def test_test_split_refuses_pending_selection():
    with pytest.raises(OfficialParamsError, match="pending_selection"):
        cg.official_role_for_args(_args(alpha=0.25), PENDING)


def test_test_split_refuses_unlisted_alpha():
    for alpha in (0.5, 1.0, None):  # 1.0 = main()'s default for v3 without --alpha
        with pytest.raises(OfficialParamsError, match="not an official alpha"):
            cg.official_role_for_args(_args(alpha=alpha), SELECTED)


def test_test_split_refuses_other_locked_param_changes():
    with pytest.raises(OfficialParamsError, match="max_hops"):
        cg.official_role_for_args(_args(max_hops=1), SELECTED)
    with pytest.raises(OfficialParamsError, match="edge_types"):
        cg.official_role_for_args(_args(edge_types=["HAS_ANSWER"]), SELECTED)
    with pytest.raises(OfficialParamsError, match="use_author_trust"):
        cg.official_role_for_args(_args(use_author_trust=True), SELECTED)
    # same edge types in another order are the same set
    assert cg.official_role_for_args(_args(edge_types=list(reversed(EDGES))), SELECTED) == "c_primary"


def test_selected_params_need_two_distinct_alphas():
    with pytest.raises(OfficialParamsError, match="two distinct"):
        check_c_test_run({**SELECTED, "alpha_ablation": 0.25}, c_retrieval_version="v4", alpha=0.25, max_hops=2,
                         edge_types=EDGES, use_author_trust=False, accepted_only=False)


def test_dev_split_is_never_checked():
    """Dev runs keep the old behaviour: v2, any alpha, pending params."""
    assert cg.official_role_for_args(_args(sample_split="dev", c_retrieval_version="v2", alpha=None), PENDING) is None
    assert cg.official_role_for_args(_args(sample_split="dev", alpha=0.75, max_hops=1), PENDING) is None


def test_main_calls_the_check_before_any_data_or_neo4j_work():
    src = inspect.getsource(cg.main)
    assert src.index("official_role_for_args(args)") < src.index("get_candidate_questions(")
    assert src.index("official_role_for_args(args)") < src.index("connect_neo4j(")


def test_real_official_params_refuse_every_test_run_while_pending():
    from llm.c_graphrag.official_params import load_official_params

    official = load_official_params()
    if official["status"] != "pending_selection":
        pytest.skip("official_params.json is selected")
    for alpha in (0.25, 0.0):
        with pytest.raises(OfficialParamsError):
            cg.official_role_for_args(_args(alpha=alpha))
