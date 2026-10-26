"""llm/c_graphrag/official_params.json must stay byte-identical (as
parsed JSON) to the ```json official_params fenced block in
docs/DECISION_C_SCORING.md -- the decision record is the human-readable
source of truth and the .json file is what the dashboard's "Official
(n=384)" preset actually reads, so a drift between them would silently
let the dashboard launch an official run with parameters the decision
record doesn't actually specify.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _extract_official_params_block(markdown: str) -> dict:
    match = re.search(r"```json official_params\n(.*?)\n```", markdown, re.DOTALL)
    assert match, "docs/DECISION_C_SCORING.md must contain a ```json official_params fenced block"
    return json.loads(match.group(1))


def test_official_params_json_matches_decision_record_block():
    markdown = (REPO_ROOT / "docs" / "DECISION_C_SCORING.md").read_text()
    from_doc = _extract_official_params_block(markdown)
    from_file = json.loads((REPO_ROOT / "llm" / "c_graphrag" / "official_params.json").read_text())
    assert from_doc == from_file


def test_official_params_has_required_keys():
    from_file = json.loads((REPO_ROOT / "llm" / "c_graphrag" / "official_params.json").read_text())
    assert set(from_file) == {
        "status", "c_retrieval_version", "alpha_primary", "alpha_ablation", "selection_record",
        "sample_split", "max_hops", "edge_types", "use_author_trust", "accepted_only",
    }
    assert set(from_file["selection_record"]) == {"path", "commit"}


def test_official_params_status_is_valid():
    from_file = json.loads((REPO_ROOT / "llm" / "c_graphrag" / "official_params.json").read_text())
    assert from_file["status"] in ("pending_selection", "selected")


def test_official_params_alphas_null_until_selected():
    """Amendment 10: both stage-2 candidates are v3 values, so
    c_retrieval_version is "v4" (Amendment 17) from the start; alpha_primary,
    alpha_ablation and selection_record.commit stay null until the Outcome
    section and the Amendment 10 Record are filled in and status flips to
    "selected". Once selected, the two alphas are the two distinct stage-2
    candidates."""
    from_file = json.loads((REPO_ROOT / "llm" / "c_graphrag" / "official_params.json").read_text())
    assert from_file["c_retrieval_version"] == "v4"
    if from_file["status"] == "pending_selection":
        assert from_file["alpha_primary"] is None
        assert from_file["alpha_ablation"] is None
        assert from_file["selection_record"]["commit"] is None
    else:
        from stage2_report import STAGE2_ALPHAS

        alphas = {from_file["alpha_primary"], from_file["alpha_ablation"]}
        assert alphas == set(STAGE2_ALPHAS), "C-primary and C-ablation must be the two stage-2 candidates"
        # the commit field is filled in after the author's commit (Amendment 10 Record); until then it stays null
        assert from_file["selection_record"]["commit"] is None or isinstance(from_file["selection_record"]["commit"], str)
        # stage 2 of 2026-10-25 (docs/STAGE2_RESULT_AND_RUNTIME_NOTE.md): 0.5 won at the tie-break step
        assert (from_file["alpha_primary"], from_file["alpha_ablation"]) == (0.5, 0.25)
