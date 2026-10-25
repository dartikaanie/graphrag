"""Official run matrix R1-R9 (llm/official_run_matrix.py) and its document
docs/OFFICIAL_RUN_MATRIX.md. Fake official_params dicts only."""

from pathlib import Path

from llm.official_run_matrix import build_matrix, cli_command

REPO_ROOT = Path(__file__).resolve().parents[2]
EDGES = ["EMBED_SIM", "HAS_ACCEPTED_ANSWER", "HAS_ANSWER", "IS_RELATED_TO", "TAG_COOCCUR"]
PENDING = {"status": "pending_selection", "c_retrieval_version": "v3", "alpha_primary": None,
           "alpha_ablation": None, "selection_record": {"path": "x", "commit": None}, "sample_split": "test",
           "max_hops": 2, "edge_types": EDGES, "use_author_trust": False, "accepted_only": False}
SELECTED = {**PENDING, "status": "selected", "alpha_primary": 0.0, "alpha_ablation": 0.25,
            "selection_record": {"path": "x", "commit": "abc1234"}}


def test_nine_cells_in_thesis_order():
    m = build_matrix(SELECTED)
    assert [(r["run"], r["label"]) for r in m] == [
        ("R1", "A"), ("R2", "B-plain"), ("R3", "B-grounded"),
        ("R4", "C-primary-plain"), ("R5", "C-primary-grounded"),
        ("R6", "C-ablation-plain"), ("R7", "C-ablation-grounded"),
        ("R8", "D-plain"), ("R9", "D-grounded")]


def test_c_cells_use_two_different_official_alphas():
    c = [r for r in build_matrix(SELECTED) if r["condition"] == "C"]
    assert [(r["role"], r["alpha"], r["grounding"]) for r in c] == [
        ("c_primary", 0.0, False), ("c_primary", 0.0, True), ("c_ablation", 0.25, False), ("c_ablation", 0.25, True)]
    for r in c:
        assert r["dashboard"]["c_retrieval_version"] == "v3"
        assert r["dashboard"]["fusion_mode"] == "trust_weighted"
        assert r["argv"][r["argv"].index("--alpha") + 1] == f"{r['alpha']:g}"
        assert r["dashboard"]["edge_types"] == EDGES and r["dashboard"]["max_hops"] == 2


def test_pending_matrix_has_placeholders_not_values():
    c = [r for r in build_matrix(PENDING) if r["condition"] == "C"]
    assert [r["alpha"] for r in c] == [None] * 4
    assert {r["argv"][r["argv"].index("--alpha") + 1] for r in c} == {"<alpha_primary>", "<alpha_ablation>"}


def test_every_run_pins_the_shared_sample_and_grounding():
    for r in build_matrix(SELECTED):
        a = r["argv"]
        assert a[a.index("--n-sample") + 1] == "384" and a[a.index("--seed") + 1] == "42"
        assert a[a.index("--oversample-pool") + 1] == "1536" and a[a.index("--sample-split") + 1] == "test"
        assert a[a.index("--model") + 1] == "gpt-4o-mini"
        if r["condition"] == "A":
            assert r["grounding"] is None and "--require-grounding" not in a
        else:
            assert ("--require-grounding" if r["grounding"] else "--no-require-grounding") in a
            assert a[a.index("--leakage-mode") + 1] == "thread"  # Amendment 15
            assert a[a.index("--top-k") + 1] == "5" and a[a.index("--token-chunk-limit") + 1] == "400"
        assert r["dashboard"]["require_grounding" if r["condition"] != "A" else "condition"] is not None


def test_cli_flags_exist_in_each_script():
    """Every flag the matrix passes is a real argparse option of that script."""
    import re

    for r in build_matrix(SELECTED):
        src = (REPO_ROOT / r["cwd"] / r["script"]).read_text()
        src += (REPO_ROOT / "llm" / "sample_split.py").read_text() + (REPO_ROOT / "llm" / "common" / "leakage.py").read_text()
        for flag in [a for a in r["argv"] if a.startswith("--")]:
            base = flag.replace("--no-", "--")
            assert re.search(rf'add_argument\(\s*"{re.escape(base)}"', src), (r["run"], flag)


def test_document_lists_exactly_the_matrix_commands():
    doc = (REPO_ROOT / "docs" / "OFFICIAL_RUN_MATRIX.md").read_text()
    for r in build_matrix(PENDING):
        assert f"`{cli_command(r)}`" in doc, r["run"]
