"""Judge-v2 (hallucination, reference-based, fixed prompt -- see docs/
JUDGE_V2_CHANGES.md) dashboard endpoints -- mirrors routers/judge_v1.py,
for a SEPARATE job log/output files. judge-v1's router/files/outputs
are never touched or depended on by this module. backend/ -> llm/
only, never the reverse.
"""

import threading

from fastapi import APIRouter, HTTPException

from app.models.schemas import JudgeV2LaunchRequest, JudgeV2PlanRequest
from app.services import engine_service, judge_v2_lookup_service, run_registry

router = APIRouter(prefix="/api/judge-v2")


@router.get("/registry")
def get_registry():
    """Same judge model registry as judge-v1 (same registry, same
    models/roles) -- judge-v2 is a different PROMPT, not a different
    set of judge models."""
    return {"judges": engine_service._judge_clients_module().registry_public_view()}


@router.get("/jobs")
def list_jobs(show_invalid: bool = False):
    return {"jobs": judge_v2_lookup_service.load_judge_v2_jobs(show_invalid=show_invalid)}


@router.post("/plan")
def plan(body: JudgeV2PlanRequest):
    if not body.run_ids:
        raise HTTPException(status_code=400, detail="run_ids must not be empty")
    if not body.judge_ids:
        raise HTTPException(status_code=400, detail="judge_ids must not be empty")
    return engine_service.plan_judge_v2(body.run_ids, body.judge_ids, body.workers)


@router.post("/launch")
def launch(body: JudgeV2LaunchRequest):
    if not body.run_ids:
        raise HTTPException(status_code=400, detail="run_ids must not be empty")
    if not body.judge_ids:
        raise HTTPException(status_code=400, detail="judge_ids must not be empty")

    params = {"run_ids": body.run_ids, "judge_ids": body.judge_ids, "workers": body.workers}
    job_run_id = run_registry.create_run("JUDGE_V2", "batch", params)
    thread = threading.Thread(
        target=engine_service.start_judge_v2_run, args=(job_run_id, params), daemon=True,
    )
    thread.start()
    return {"run_id": job_run_id}


@router.get("/status/{run_id}")
def status(run_id: str):
    run = run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Judge-v2 job not found")
    return run


@router.post("/cancel/{run_id}")
def cancel(run_id: str):
    run = run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Judge-v2 job not found")
    run_registry.request_cancel(run_id)
    return {"ok": True}


def _existing_canonical_outputs(judge_ids: list[str]) -> list[str]:
    paths = []
    for jid in judge_ids:
        out_path, _, _ = engine_service.judge_v2_output_paths(jid)
        if out_path.exists():
            paths.append(str(out_path))
    return paths


@router.get("/results/summary")
def results_summary(judge_id: str = "primary"):
    """Per-run_label hallucination (reference-based) rates for judge-v2,
    PLUS v2-only metrics (completeness distribution, factual-and-
    complete rate, % with reference_conflict, attempted_claims_conflict
    rate) -- see judge_agreement.compute_run_label_summary_v2_extra()."""
    mod = engine_service._judge_agreement_module()
    records = mod.load_judge_records(_existing_canonical_outputs([judge_id]))
    summary = mod.compute_run_label_summary(records, judge_id)
    extra = mod.compute_run_label_summary_v2_extra(records, judge_id)
    for run_label, extra_metrics in extra.items():
        summary.setdefault(run_label, {}).update(extra_metrics)
    return {"summary": summary, "prompt_version": engine_service._judge_v2_module().PROMPT_VERSION}


@router.get("/results/agreement")
def results_agreement(judge_id_a: str = "primary", judge_id_b: str = "secondary"):
    mod = engine_service._judge_agreement_module()
    records = mod.load_judge_records(_existing_canonical_outputs([judge_id_a, judge_id_b]))
    pairs = mod.pair_two_judges(records, judge_id_a, judge_id_b)
    if not pairs:
        return {"n_pairs": 0, "confusion_matrix": None, "kappas": None, "insufficient_data": True}
    confusion_matrix = mod.build_confusion_matrix(pairs)
    kappas = mod.compute_kappas(pairs)
    return {
        "n_pairs": len(pairs), "confusion_matrix": confusion_matrix, "kappas": kappas,
        "insufficient_data": kappas["weighted"] is None,
    }


@router.get("/results/disagreements")
def results_disagreements(judge_id_a: str = "primary", judge_id_b: str = "secondary",
                           run_label: str | None = None, label_a: str | None = None, label_b: str | None = None):
    mod = engine_service._judge_agreement_module()
    records = mod.load_judge_records(_existing_canonical_outputs([judge_id_a, judge_id_b]))
    pairs = mod.pair_two_judges(records, judge_id_a, judge_id_b)
    disagreements = []
    for a, b in pairs:
        if a.get("label") == b.get("label"):
            continue
        if run_label and a.get("run_label") != run_label:
            continue
        if label_a and a.get("label") != label_a:
            continue
        if label_b and b.get("label") != label_b:
            continue
        disagreements.append({
            "run_id": a.get("run_id"), "run_label": a.get("run_label"), "question_id": a.get("question_id"),
            f"{judge_id_a}_label": a.get("label"), f"{judge_id_b}_label": b.get("label"),
            f"{judge_id_a}_reasoning": a.get("reasoning"), f"{judge_id_b}_reasoning": b.get("reasoning"),
        })
    return {"disagreements": disagreements}


@router.get("/results/version-comparison")
def results_version_comparison(run_ids: str, judge_id_a: str = "primary", judge_id_b: str = "secondary"):
    """Step 5: v1 vs v2 for the SAME runs/judge model -- a label
    transition matrix (v1 label -> v2 label, per run_label, for
    judge_id_a) and kappa(judge_id_a, judge_id_b) under v1 and under v2
    side by side. `run_ids` is a comma-separated list."""
    ids = [r for r in run_ids.split(",") if r.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="run_ids must not be empty")
    return engine_service.build_version_comparison(ids, judge_id_a, judge_id_b)


@router.get("/results/reference-conflicts")
def results_reference_conflicts(run_ids: str, judge_ids: str = "primary,secondary"):
    """Step 5: "Reference conflicts" list for human review -- every v2
    claim flagged reference_conflict=True. `run_ids`/`judge_ids` are
    comma-separated lists."""
    ids = [r for r in run_ids.split(",") if r.strip()]
    jids = [j for j in judge_ids.split(",") if j.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="run_ids must not be empty")
    return {"conflicts": engine_service.reference_conflicts(ids, jids)}
