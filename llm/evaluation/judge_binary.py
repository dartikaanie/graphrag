"""
judge_binary.py
---------------
The binary hallucination label -- the PRIMARY hallucination metric from
docs/DECISION_C_SCORING.md, Amendment 4 -- derived deterministically from
a judge-v2 / judge-v2.1 record's claims (like derive_label_v2, never from
the judge's self-reported label):

    hallucinated = True   >= 1 claim is CONTRADICTED or FABRICATED (any
                          severity, including a missing one)
                   False  attempted answer with no such claim
                   None   ABSTAIN (answer_attempted is false), a parse
                          error, or a failed call

judge-v2.1 records carry the field (written by the v2.1 runner). judge-v2
records do not; readers call hallucinated_of(), which derives it on the
fly, so existing judge-v2 output files are never rewritten.
"""

ERROR_VERDICTS = ("CONTRADICTED", "FABRICATED")
CATEGORY_ABSTAIN = "ABSTAIN"
CATEGORY_NOT_HALLUCINATED = "NOT_HALLUCINATED"
CATEGORY_HALLUCINATED = "HALLUCINATED"
CATEGORIES = (CATEGORY_ABSTAIN, CATEGORY_NOT_HALLUCINATED, CATEGORY_HALLUCINATED)


def is_unusable(record: dict) -> bool:
    """A record with no judgment at all: failed call or parse error."""
    return bool(record.get("call_failed")) or bool(record.get("parse_error")) \
        or not isinstance(record.get("answer_attempted"), bool)


def derive_hallucinated(record: dict) -> bool | None:
    if is_unusable(record) or record["answer_attempted"] is False:
        return None
    return any(str(c.get("verdict", "")).upper() in ERROR_VERDICTS for c in (record.get("claims") or []))


def hallucinated_of(record: dict) -> bool | None:
    """The record's own `hallucinated` field when present (v2.1), else derived (v2)."""
    if "hallucinated" in record:
        return record["hallucinated"]
    return derive_hallucinated(record)


def category_of(record: dict) -> str | None:
    """3-category label used for judge-vs-human agreement: ABSTAIN /
    NOT_HALLUCINATED / HALLUCINATED; None for an unusable record."""
    if is_unusable(record):
        return None
    if record["answer_attempted"] is False:
        return CATEGORY_ABSTAIN
    return CATEGORY_HALLUCINATED if hallucinated_of(record) else CATEGORY_NOT_HALLUCINATED


def severity_of(record: dict) -> str | None:
    """Descriptive severity of a HALLUCINATED record: "PENUH" (any CORE
    error), "SEBAGIAN" (every error claim MINOR), or "UNSPECIFIED" (an
    error claim has no severity and no CORE error settles it -- never
    imputed). None when the record is not hallucinated."""
    if category_of(record) != CATEGORY_HALLUCINATED:
        return None
    errors = [c for c in (record.get("claims") or []) if str(c.get("verdict", "")).upper() in ERROR_VERDICTS]
    severities = [str(c.get("severity") or "").upper() for c in errors]
    if "CORE" in severities:
        return "PENUH"
    if errors and all(s == "MINOR" for s in severities):
        return "SEBAGIAN"
    return "UNSPECIFIED"


def binary_summary(records: list[dict]) -> dict:
    """Binary hallucination summary over one group of records (one judge,
    one run_label). Unusable records (parse error / failed call) are
    counted and excluded from every rate."""
    n_total = len(records)
    usable = [r for r in records if not is_unusable(r)]
    n = len(usable)
    cats = [category_of(r) for r in usable]
    n_abstain = cats.count(CATEGORY_ABSTAIN)
    n_hall = cats.count(CATEGORY_HALLUCINATED)
    n_attempted = n - n_abstain
    sev = [severity_of(r) for r in usable if category_of(r) == CATEGORY_HALLUCINATED]
    claims = [c for r in usable if r["answer_attempted"] for c in (r.get("claims") or [])]
    n_unverifiable = sum(1 for c in claims if str(c.get("verdict", "")).upper() == "UNVERIFIABLE")

    def rate(a, b):
        return round(a / b, 4) if b else None

    return {
        "n_records": n_total, "n_unusable": n_total - n, "n": n,
        "n_abstain": n_abstain, "n_hallucinated": n_hall, "n_not_hallucinated": n_attempted - n_hall,
        "abstain_rate": rate(n_abstain, n),
        "hallucination_rate_excl_abstain": rate(n_hall, n_attempted),
        "hallucination_rate_abstain_counted": rate(n_hall + n_abstain, n),
        "severity_distribution": {"PENUH": sev.count("PENUH"), "SEBAGIAN": sev.count("SEBAGIAN"),
                                  "UNSPECIFIED": sev.count("UNSPECIFIED")},
        "n_claims": len(claims), "n_unverifiable_claims": n_unverifiable,
        "unverifiable_rate": rate(n_unverifiable, len(claims)),
    }


# ---------------------------------------------------------------------------
# Agreement (unweighted Cohen's kappa) on the 3-category and binary labels
# ---------------------------------------------------------------------------

def confusion_matrix(labels_a: list, labels_b: list, categories) -> dict:
    """{row category (rater A): {column category (rater B): count}}."""
    matrix = {r: {c: 0 for c in categories} for r in categories}
    for a, b in zip(labels_a, labels_b):
        matrix[a][b] += 1
    return matrix


def cohen_kappa(labels_a: list, labels_b: list, categories) -> float | None:
    """Unweighted Cohen's kappa. None when undefined: no items, or
    expected agreement == 1 (both raters used one and the same single
    category for every item)."""
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        return None
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    expected = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def agreement_report(labels_a: list, labels_b: list) -> dict:
    """Agreement between two raters given paired 3-category labels (each
    in CATEGORIES). Returns the 3-category kappa + confusion matrix, and
    the BINARY kappa (hallucinated vs not) restricted to items that BOTH
    raters label non-ABSTAIN (Amendment 4 Addendum), each with its n."""
    k3 = cohen_kappa(labels_a, labels_b, CATEGORIES)
    both = [(a, b) for a, b in zip(labels_a, labels_b) if a != CATEGORY_ABSTAIN and b != CATEGORY_ABSTAIN]
    bin_cats = (CATEGORY_NOT_HALLUCINATED, CATEGORY_HALLUCINATED)
    ba, bb = [a for a, _ in both], [b for _, b in both]
    kb = cohen_kappa(ba, bb, bin_cats)
    return {
        "n": len(labels_a),
        "kappa_3cat": round(k3, 4) if k3 is not None else None,
        "observed_agreement_3cat": round(sum(1 for a, b in zip(labels_a, labels_b) if a == b) / len(labels_a), 4)
        if labels_a else None,
        "confusion_3cat": confusion_matrix(labels_a, labels_b, CATEGORIES),
        "n_both_non_abstain": len(both),
        "kappa_binary_non_abstain": round(kb, 4) if kb is not None else None,
        "confusion_binary_non_abstain": confusion_matrix(ba, bb, bin_cats),
    }
