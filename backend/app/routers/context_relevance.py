"""Context-relevance-v1 dashboard endpoints (Step 3 of docs/
agent_prompt_judge_results_export.md) -- fully separate from
routers/judge_v1.py and routers/judge.py. Never modifies the pre-
existing llm_judge_context_relevance.py (no suffix) or its files.
backend/ -> llm/ only, never the reverse.
"""

import threading

from fastapi import APIRouter, HTTPException

from app.models.schemas import ContextRelevanceV1LaunchRequest, ContextRelevanceV1PlanRequest
from app.services import ctxrel_v1_lookup_service, engine_service, run_registry

router = APIRouter(prefix="/api/context-relevance")

# Same canonical factorial order used throughout the dashboard (history_
# service.derive_run_label()'s 9 labels) -- duplicated here rather than
# imported since there's no single shared constants module yet; see
# FactorialResultsPage.tsx's FACTORIAL_ORDER for the frontend's own copy.
FACTORIAL_ORDER = [
    "A", "B-plain", "B-grounded",
    "C-uniform-plain", "C-uniform-grounded", "C-trust-plain", "C-trust-grounded",
    "D-plain", "D-grounded",
]


@router.get("/jobs")
def list_jobs(show_invalid: bool = False):
    """Past context-relevance-v1 jobs -- mirrors GET /api/judge-v1/jobs,
    separate job log. Previously there was no way to see these jobs in
    the dashboard at all."""
    return {"jobs": ctxrel_v1_lookup_service.load_ctxrel_v1_jobs(show_invalid=show_invalid)}


@router.post("/plan")
def plan(body: ContextRelevanceV1PlanRequest):
    if not body.run_ids:
        raise HTTPException(status_code=400, detail="run_ids must not be empty")
    if not body.judge_ids:
        raise HTTPException(status_code=400, detail="judge_ids must not be empty")
    return engine_service.plan_context_relevance_v1(body.run_ids, body.judge_ids, body.workers)


@router.post("/launch")
def launch(body: ContextRelevanceV1LaunchRequest):
    if not body.run_ids:
        raise HTTPException(status_code=400, detail="run_ids must not be empty")
    if not body.judge_ids:
        raise HTTPException(status_code=400, detail="judge_ids must not be empty")

    params = {"run_ids": body.run_ids, "judge_ids": body.judge_ids, "workers": body.workers, "limit": body.limit}
    job_run_id = run_registry.create_run("CONTEXT_RELEVANCE_V1", "batch", params)
    thread = threading.Thread(
        target=engine_service.start_context_relevance_v1_run, args=(job_run_id, params), daemon=True,
    )
    thread.start()
    return {"run_id": job_run_id}


@router.get("/status/{run_id}")
def status(run_id: str):
    run = run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Context-relevance job not found")
    return run


@router.post("/cancel/{run_id}")
def cancel(run_id: str):
    run = run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Context-relevance job not found")
    run_registry.request_cancel(run_id)
    return {"ok": True}


@router.get("/results/summary")
def results_summary(run_id: str, judge_id: str = "primary"):
    """Context-relevance metrics (Step 3.3) for ONE run, joined against
    the given judge_id's canonical output file at read time (rank
    position is derived from the run's OWN retrieved_context order, not
    stored on the deduplicated judged record -- see
    llm_judge_context_relevance_v1.compute_run_context_relevance())."""
    from app.services import history_service

    detail = history_service.get_history_detail(run_id)
    if not detail or not detail.get("output_path"):
        raise HTTPException(status_code=404, detail="run not found or has no output file")

    mod = engine_service._context_relevance_v1_module()
    out_path, _, _ = engine_service.ctxrel_v1_output_paths(judge_id)
    ctxrel_by_key = mod.load_ctxrel_records([out_path]) if out_path.exists() else {}
    metrics = mod.compute_run_context_relevance(detail["output_path"], ctxrel_by_key)
    return {"run_id": run_id, "run_label": detail.get("run_label"), "judge_id": judge_id, "metrics": metrics}


def _compute_link_to_outcomes_rows(output_path: str, ctxrel_by_key: dict, jv1_by_qid: dict) -> list[dict]:
    import json as _json

    ctxrel_mod = engine_service._context_relevance_v1_module()
    rows = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = _json.loads(line)
            qid = record.get("question_id")
            retrieved = record.get("retrieved_context") or []
            has_relevant = any(
                (ctxrel_by_key.get((qid, item.get("answer_id"),
                                    ctxrel_mod._text_hash(item.get("chunk_text", "") or ""))) or {}).get("label") == "RELEVANT"
                for item in retrieved
            )
            jv1_record = jv1_by_qid.get(qid)
            rows.append({
                "question_id": qid,
                "has_relevant_context": has_relevant,
                "hallucination_label": jv1_record.get("label") if jv1_record else None,
            })
    return rows


def _aggregate_link_to_outcomes(rows: list[dict], grounded: bool) -> dict:
    """Descriptive-only contingency counts (n=10 pilot, no statistical
    test) -- hallucination label x has_relevant_context, and (grounded
    runs only) ABSTAIN x has_relevant_context."""
    label_table: dict[str, dict[str, int]] = {}
    for r in rows:
        label = r["hallucination_label"] or "UNKNOWN"
        bucket = "has_relevant" if r["has_relevant_context"] else "no_relevant"
        label_table.setdefault(label, {"has_relevant": 0, "no_relevant": 0})
        label_table[label][bucket] += 1

    abstain_table = None
    if grounded:
        abstain_table = {"has_relevant": {"abstain": 0, "not_abstain": 0}, "no_relevant": {"abstain": 0, "not_abstain": 0}}
        for r in rows:
            bucket = "has_relevant" if r["has_relevant_context"] else "no_relevant"
            is_abstain = r["hallucination_label"] == "ABSTAIN"
            abstain_table[bucket]["abstain" if is_abstain else "not_abstain"] += 1

    return {"label_x_relevant": label_table, "abstain_x_relevant": abstain_table}


@router.get("/results/link-to-outcomes")
def results_link_to_outcomes(run_id: str, context_judge_id: str = "primary", hallucination_judge_id: str = "primary"):
    """Step 3.4: per run_label, hallucination label x "has >=1 RELEVANT
    context item", and (for grounded runs) ABSTAIN x "has >=1 RELEVANT
    context item" -- descriptive only, no statistical test (n=10 pilot)."""
    from app.services import history_service

    detail = history_service.get_history_detail(run_id)
    if not detail or not detail.get("output_path"):
        raise HTTPException(status_code=404, detail="run not found or has no output file")

    ctxrel_mod = engine_service._context_relevance_v1_module()
    out_path, _, _ = engine_service.ctxrel_v1_output_paths(context_judge_id)
    ctxrel_by_key = ctxrel_mod.load_ctxrel_records([out_path]) if out_path.exists() else {}

    jv1_mod = engine_service._judge_agreement_module()
    jv1_out, _, _ = engine_service.judge_v1_output_paths(hallucination_judge_id)
    jv1_records = jv1_mod.load_judge_records([str(jv1_out)]) if jv1_out.exists() else []
    jv1_by_qid = {r["question_id"]: r for r in jv1_records if r.get("run_id") == run_id}

    rows = _compute_link_to_outcomes_rows(detail["output_path"], ctxrel_by_key, jv1_by_qid)
    run_label = detail.get("run_label") or ""
    grounded = run_label.endswith("-grounded")
    tables = _aggregate_link_to_outcomes(rows, grounded)
    return {"run_id": run_id, "run_label": detail.get("run_label"), "rows": rows, "tables": tables}


@router.get("/results/batch-summary")
def results_batch_summary(batch_id: str, judge_id: str = "primary"):
    """Per-run_label context-relevance metrics for a whole batch, in
    factorial order, with condition A shown as a no-context row --
    resolves runs the SAME way GET /api/history/batches does (recorded
    batch_id first, inferred chained-window clustering otherwise), so a
    batch_id taken from that endpoint always resolves here too. Returns
    `has_any_results: false` (with every row's metrics null) rather than
    a bare empty response when no ctxrel output exists yet for ANY run
    in this batch -- callers must show that explicitly, never render
    nothing silently."""
    from app.services import batch_grouping_service, history_service

    records, _ = history_service.list_history(None, None, None, page=1, page_size=100000, show_superseded=True)
    batches = batch_grouping_service.group_into_batches(records)
    batch = next((b for b in batches if b["batch_id"] == batch_id), None)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"batch '{batch_id}' not found")

    mod = engine_service._context_relevance_v1_module()
    out_path, _, _ = engine_service.ctxrel_v1_output_paths(judge_id)
    ctxrel_by_key = mod.load_ctxrel_records([out_path]) if out_path.exists() else {}

    by_label = {r["run_label"]: r for r in batch["runs"] if r.get("run_label") and r.get("status") == "success"}
    rows = []
    has_any_results = False
    for label in FACTORIAL_ORDER:
        run = by_label.get(label)
        if run is None:
            continue
        if run.get("condition") == "A":
            rows.append({"run_label": label, "run_id": run.get("history_id"), "no_context": True, "metrics": None})
            continue
        metrics = mod.compute_run_context_relevance(run["output_path"], ctxrel_by_key)
        if metrics["n_items_judged"] > 0:
            has_any_results = True
        rows.append({"run_label": label, "run_id": run.get("history_id"), "no_context": False, "metrics": metrics})

    return {"batch_id": batch_id, "judge_id": judge_id, "rows": rows, "has_any_results": has_any_results}
