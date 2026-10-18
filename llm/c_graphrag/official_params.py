"""Official Condition C parameters (docs/DECISION_C_SCORING.md, Amendment 10).

official_params.json holds the frozen C parameters for the test sample:
C-primary runs at `alpha_primary` (the stage-2 winner, α*) and C-ablation at
`alpha_ablation` (the other stage-2 candidate). Both are null and `status` is
"pending_selection" until the Outcome section and the Amendment 10 Record are
filled in; then `status` becomes "selected".

check_c_test_run() is the one launch rule for a Condition C run on the test
sample, used by the CLI (c_graphrag.py) and the dashboard (engine_service.py):
c_retrieval_version must be v3, the params must be selected, alpha must be
alpha_primary or alpha_ablation, and the other locked keys must match. Dev
runs are never checked here.

No heavy imports: the dashboard backend imports this module directly.
"""

import json
from pathlib import Path

OFFICIAL_PARAMS_PATH = Path(__file__).resolve().parent / "official_params.json"

STATUS_PENDING = "pending_selection"
STATUS_SELECTED = "selected"
STATUSES = (STATUS_PENDING, STATUS_SELECTED)

ROLE_PRIMARY = "c_primary"
ROLE_ABLATION = "c_ablation"

# Locked alongside alpha; compared as given (edge_types as a sorted list).
LOCKED_KEYS = ("c_retrieval_version", "sample_split", "max_hops", "edge_types", "use_author_trust", "accepted_only")


class OfficialParamsError(ValueError):
    """A test-sample Condition C run does not match the official parameters."""


def load_official_params(path=OFFICIAL_PARAMS_PATH) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def official_alphas(official: dict) -> dict:
    """{role: alpha} for the two official C runs; empty while pending."""
    if not official or official.get("status") != STATUS_SELECTED:
        return {}
    return {ROLE_PRIMARY: official["alpha_primary"], ROLE_ABLATION: official["alpha_ablation"]}


def _norm(key, value):
    if key == "edge_types" and value is not None:
        return sorted(value)
    return value


def check_c_test_run(official: dict | None, *, c_retrieval_version: str, alpha, max_hops, edge_types,
                     use_author_trust: bool, accepted_only: bool, sample_split: str = "test") -> str:
    """Role ("c_primary" / "c_ablation") of a test-sample C run, or
    OfficialParamsError explaining why it may not run."""
    if sample_split != "test":
        raise ValueError("check_c_test_run is for --sample-split test only")
    if official is None:
        raise OfficialParamsError(f"{OFFICIAL_PARAMS_PATH} not found -- an official C run needs it.")
    if c_retrieval_version != "v3":
        raise OfficialParamsError(
            f"Condition C on the test sample requires --c-retrieval-version v3 (got {c_retrieval_version!r}); "
            "see docs/DECISION_C_SCORING.md, Amendment 10.")
    if official.get("status") != STATUS_SELECTED:
        raise OfficialParamsError(
            f"official_params.json status is {official.get('status')!r}, not {STATUS_SELECTED!r}: alpha_primary/"
            "alpha_ablation are not selected yet (stage 2 pending). Test-sample C runs are refused until the "
            "Outcome section and the Amendment 10 Record are filled in.")
    requested = {"c_retrieval_version": c_retrieval_version, "sample_split": sample_split, "max_hops": max_hops,
                 "edge_types": edge_types, "use_author_trust": use_author_trust, "accepted_only": accepted_only}
    mismatched = [k for k in LOCKED_KEYS if _norm(k, requested[k]) != _norm(k, official.get(k))]
    if mismatched:
        raise OfficialParamsError(
            "Test-sample C run differs from official_params.json on: "
            + ", ".join(f"{k} ({requested[k]!r} != {official.get(k)!r})" for k in mismatched))
    roles = official_alphas(official)
    if None in roles.values() or roles[ROLE_PRIMARY] == roles[ROLE_ABLATION]:
        raise OfficialParamsError(
            f"official_params.json is 'selected' but alpha_primary={roles[ROLE_PRIMARY]!r} and "
            f"alpha_ablation={roles[ROLE_ABLATION]!r} are not two distinct values (Amendment 10).")
    for role, value in roles.items():
        if alpha is not None and float(alpha) == float(value):
            return role
    raise OfficialParamsError(
        f"alpha={alpha!r} is not an official alpha: alpha_primary={official['alpha_primary']!r}, "
        f"alpha_ablation={official['alpha_ablation']!r} (official_params.json).")
