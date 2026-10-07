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
        "status", "c_retrieval_version", "alpha", "sample_split", "max_hops", "edge_types",
        "use_author_trust", "accepted_only",
    }


def test_official_params_status_is_valid():
    from_file = json.loads((REPO_ROOT / "llm" / "c_graphrag" / "official_params.json").read_text())
    assert from_file["status"] in ("pending_selection", "locked")


def test_official_params_alpha_and_version_null_until_outcome_filled_in():
    """While status is "pending_selection", the dashboard must refuse to
    launch ANY Official run (not just v3) -- see
    test_config_router.py's check_official_c_params_lock tests for the
    launch-refusal logic itself. This test documents and locks the
    underlying data invariant: alpha and c_retrieval_version both stay
    null until the Outcome section is actually filled in and status
    flips to "locked" (the selection procedure hasn't run yet, so there
    is no known winning version, v2 or v3)."""
    from_file = json.loads((REPO_ROOT / "llm" / "c_graphrag" / "official_params.json").read_text())
    if from_file["status"] == "pending_selection":
        assert from_file["alpha"] is None
        assert from_file["c_retrieval_version"] is None
    if from_file["c_retrieval_version"] == "v3":
        assert from_file["alpha"] is not None, "v3 official params must have a concrete alpha once chosen"
