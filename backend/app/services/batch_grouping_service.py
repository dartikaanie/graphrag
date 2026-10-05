"""Groups run_history records (across A/B/C/D) into "batches" -- the set
of runs launched together from one Factorial Batch "Confirm & Launch"
click -- for the Hallucination Judge (judge-v1) page's run picker.

Two sources of truth, in priority order:
  1. A RECORDED `batch_id` (written by FactorialBatchPage.tsx into every
     run's params since this field existed -- see schemas.py's
     RunCreateRequest.batch_id). Authoritative, never inferred away.
  2. For OLDER records with no `batch_id` at all: INFERRED groups, by
     clustering records that share the same "batch-wide" params
     (provider, model, n_sample_target, seed, oversample_pool, top_k --
     deliberately excluding anything that's supposed to VARY within one
     batch, like fusion_mode/require_grounding/n_low_level/n_anchor) and
     whose run_started_at falls within INFER_WINDOW_MINUTES of each
     other (chained: each record just needs to be within the window of
     the PREVIOUS record in the same cluster, not the cluster's first
     member -- so a batch whose 9 runs take 8 minutes end-to-end, each a
     few seconds after the last, still clusters as one group even though
     the first and last are further apart than the window alone would
     allow). Inferred groups are marked `"inferred": True` and ALWAYS
     shown as such -- never presented as if they were recorded.
"""

from datetime import datetime, timedelta
from typing import Any

INFER_WINDOW_MINUTES = 10

# Deliberately EXCLUDES top_k -- Condition A has no retrieval step at all
# (no top_k concept, the field is simply absent from its records), while
# B/C/D all share the SAME top_k within one factorial batch. Including it
# would wrongly split A into its own single-run cluster instead of
# joining the other 8 runs from the same "Confirm & Launch" click.
_SHARED_SIGNATURE_FIELDS = ("provider", "model", "n_sample_target", "seed", "oversample_pool")


def _signature(record: dict[str, Any]) -> tuple:
    return tuple(record.get(f) for f in _SHARED_SIGNATURE_FIELDS)


def _parse_time(iso: str | None):
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


def group_into_batches(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Returns batches sorted newest-launched-first, each:
    {"batch_id": str, "inferred": bool, "launched_at": str | None,
     "runs": [record, ...]}. `records` should already have invalid/
     superseded entries filtered out (same as the run picker's other
     filtering) -- this function doesn't re-check that."""
    recorded: dict[str, list[dict]] = {}
    unrecorded: list[dict] = []
    for r in records:
        bid = r.get("batch_id")
        if bid:
            recorded.setdefault(bid, []).append(r)
        else:
            unrecorded.append(r)

    batches: list[dict[str, Any]] = []
    for bid, runs in recorded.items():
        launched_at = runs[0].get("batch_launched_at") or min(
            (r.get("run_started_at") for r in runs if r.get("run_started_at")), default=None,
        )
        batches.append({"batch_id": bid, "inferred": False, "launched_at": launched_at, "runs": runs})

    # Inferred clustering: sort by run_started_at (missing timestamps sort
    # last, deterministically, rather than crashing on None comparisons).
    unrecorded_sorted = sorted(unrecorded, key=lambda r: r.get("run_started_at") or "9999")
    clusters: list[list[dict]] = []
    cluster_last_time: list[datetime | None] = []
    cluster_signature: list[tuple] = []

    for r in unrecorded_sorted:
        sig = _signature(r)
        t = _parse_time(r.get("run_started_at"))
        placed = False
        for i, (last_t, last_sig) in enumerate(zip(cluster_last_time, cluster_signature)):
            if sig != last_sig:
                continue
            if last_t is not None and t is not None and (t - last_t) > timedelta(minutes=INFER_WINDOW_MINUTES):
                continue
            clusters[i].append(r)
            cluster_last_time[i] = t
            placed = True
            break
        if not placed:
            clusters.append([r])
            cluster_last_time.append(t)
            cluster_signature.append(sig)

    for i, runs in enumerate(clusters):
        launched_at = min((r.get("run_started_at") for r in runs if r.get("run_started_at")), default=None)
        inferred_id = f"inferred-{launched_at or i}-{len(runs)}"
        batches.append({"batch_id": inferred_id, "inferred": True, "launched_at": launched_at, "runs": runs})

    batches.sort(key=lambda b: b["launched_at"] or "", reverse=True)
    return batches
