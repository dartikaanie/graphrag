"""
judge_panel_aggregate.py
------------------------
Deterministic, tie-free aggregation of the three panel judges' labels
(docs/DECISION_C_SCORING.md, Amendment 12 (d)). Pure; no I/O.

Ordinal score of a judge label:
    ABSTAIN -> 0, FAKTUAL -> 0, HALUSINASI_SEBAGIAN -> 1, HALUSINASI_PENUH -> 2

All three judges valid:
  1. >= 2 judges ABSTAIN                     -> ABSTAIN
  2. otherwise the median of the three scores: 0 -> FAKTUAL,
     1 -> HALUSINASI_SEBAGIAN, 2 -> HALUSINASI_PENUH
  3. hallucinated = label in {SEBAGIAN, PENUH}; this equals a 2-of-3
     majority vote on hallucinated yes/no.

Exactly two judges valid (the third still failed after retries):
  - both agree on ABSTAIN status and on binary hallucinated -> their common
    result; severity = the lower of the two if they differ;
    panel_incomplete = True.
  - otherwise UNRESOLVED.
Fewer than two valid -> UNRESOLVED. An item is never dropped.

HALUSINASI_SEVERITY_UNSPECIFIED: judge-v2.1/v2.2 give this label when an
error claim has no severity and no CORE error settles it (Amendment 4
Addendum: a missing severity is never imputed). It is a valid vote for
"hallucinated" with unknown severity 1 or 2. The panel tries both values: if
the panel label is the same either way it is used; if not, the panel label
is HALUSINASI_SEVERITY_UNSPECIFIED. The binary outcome never depends on it.
"""

from itertools import product

ABSTAIN = "ABSTAIN"
FAKTUAL = "FAKTUAL"
SEBAGIAN = "HALUSINASI_SEBAGIAN"
PENUH = "HALUSINASI_PENUH"
UNSPECIFIED = "HALUSINASI_SEVERITY_UNSPECIFIED"
UNRESOLVED = "UNRESOLVED"

JUDGE_LABELS = (ABSTAIN, FAKTUAL, SEBAGIAN, PENUH)        # the 4 labels with a fixed ordinal score
VALID_JUDGE_LABELS = JUDGE_LABELS + (UNSPECIFIED,)
HALLUCINATED_LABELS = (SEBAGIAN, PENUH, UNSPECIFIED)
PANEL_LABELS = (ABSTAIN, FAKTUAL, SEBAGIAN, PENUH, UNSPECIFIED, UNRESOLVED)

SCORE = {ABSTAIN: 0, FAKTUAL: 0, SEBAGIAN: 1, PENUH: 2}
LABEL_OF_SCORE = {0: FAKTUAL, 1: SEBAGIAN, 2: PENUH}


def _scores(label: str) -> tuple[int, ...]:
    """Possible ordinal scores of one judge label (two for UNSPECIFIED)."""
    return (1, 2) if label == UNSPECIFIED else (SCORE[label],)


def _resolve(candidates: set[str]) -> str:
    return candidates.pop() if len(candidates) == 1 else UNSPECIFIED


def _aggregate_three(labels: list[str]) -> str:
    if sum(1 for lab in labels if lab == ABSTAIN) >= 2:
        return ABSTAIN
    return _resolve({LABEL_OF_SCORE[sorted(s)[1]] for s in product(*(_scores(lab) for lab in labels))})


def _aggregate_two(labels: list[str]) -> str:
    a, b = labels
    if (a == ABSTAIN) != (b == ABSTAIN):
        return UNRESOLVED
    if a == ABSTAIN:
        return ABSTAIN
    if (a in HALLUCINATED_LABELS) != (b in HALLUCINATED_LABELS):
        return UNRESOLVED
    if a not in HALLUCINATED_LABELS:
        return FAKTUAL
    return _resolve({LABEL_OF_SCORE[min(s)] for s in product(_scores(a), _scores(b))})


def aggregate_labels(judge_labels: dict[str, str | None], panel_id: str = "panel-v1") -> dict:
    """`judge_labels`: {judge_id: label, or None for a judge with no valid result (failed call or parse error
    after retries)}. Returns the panel part of the panel record (see module docstring)."""
    for jid, lab in judge_labels.items():
        if lab is not None and lab not in VALID_JUDGE_LABELS:
            raise ValueError(f"judge {jid!r}: unknown label {lab!r}")
    valid = {jid: lab for jid, lab in judge_labels.items() if lab is not None}
    labels = list(valid.values())

    if len(judge_labels) == 3 and len(valid) == 3:
        panel_label, incomplete = _aggregate_three(labels), False
    elif len(valid) == 2:
        panel_label = _aggregate_two(labels)
        incomplete = panel_label != UNRESOLVED
    else:
        panel_label, incomplete = UNRESOLVED, False

    hallucinated = None if panel_label in (UNRESOLVED, ABSTAIN) else panel_label in HALLUCINATED_LABELS
    votes = {lab: labels.count(lab) for lab in VALID_JUDGE_LABELS if lab in labels}
    return {
        "panel_id": panel_id,
        "judge_labels": dict(judge_labels),
        "vote_breakdown": {
            "labels": votes,
            "n_valid": len(valid),
            "n_missing": len(judge_labels) - len(valid),
            "n_abstain": labels.count(ABSTAIN),
            "n_hallucinated": sum(1 for lab in labels if lab in HALLUCINATED_LABELS),
            "n_not_hallucinated": labels.count(FAKTUAL),
        },
        "panel_label": panel_label,
        "panel_hallucinated": hallucinated,
        "panel_abstain": panel_label == ABSTAIN if panel_label != UNRESOLVED else None,
        "panel_incomplete": incomplete,
        "unanimous": len(valid) == 3 and len(judge_labels) == 3 and len(set(labels)) == 1,
    }


def panel_category(panel_label: str) -> str | None:
    """Panel label -> judge_binary's 3-category scale (ABSTAIN / NOT_HALLUCINATED / HALLUCINATED);
    None for UNRESOLVED."""
    if panel_label == UNRESOLVED:
        return None
    if panel_label == ABSTAIN:
        return "ABSTAIN"
    return "HALLUCINATED" if panel_label in HALLUCINATED_LABELS else "NOT_HALLUCINATED"
