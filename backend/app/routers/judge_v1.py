"""Judge-v1 (hallucination, reference-based) dashboard endpoints --
fully separate from routers/judge.py (the legacy, faithfulness/context-
grounded judge). Never modifies the legacy judge's files/manifest/
lookup service. backend/ -> llm/ only, never the reverse.
"""

import threading

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    JudgeV1ExportHumanCsvRequest,
    JudgeV1ImportHumanCsvRequest,
    JudgeV1LaunchRequest,
    JudgeV1PlanRequest,
)
from app.services import engine_service, invalid_judge_runs_service, judge_v1_lookup_service, run_registry

router = APIRouter(prefix="/api/judge-v1")


@router.get("/registry")
def get_registry():
    """Judge model registry (role/precision/price, never raw API keys) --
    used by the launch UI's judge multi-select + role badges."""
    return {"judges": engine_service._judge_clients_module().registry_public_view()}


@router.get("/jobs")
def list_jobs(show_invalid: bool = False):
    """Past judge-v1 jobs (from judge_v1_run_history.jsonl), each flagged
    with output_file_missing / is_other_config -- never silently hidden
    for either reason, only invalid (pre-dashboard smoke test) jobs are
    hidden, and only when show_invalid=False (the default)."""
    return {"jobs": judge_v1_lookup_service.load_judge_v1_jobs(show_invalid=show_invalid)}


@router.post("/plan")
def plan(body: JudgeV1PlanRequest):
    if not body.run_ids:
        raise HTTPException(status_code=400, detail="run_ids must not be empty")
    if not body.judge_ids:
        raise HTTPException(status_code=400, detail="judge_ids must not be empty")
    return engine_service.plan_judge_v1(body.run_ids, body.judge_ids, body.workers)


@router.post("/launch")
def launch(body: JudgeV1LaunchRequest):
    if not body.run_ids:
        raise HTTPException(status_code=400, detail="run_ids must not be empty")
    if not body.judge_ids:
        raise HTTPException(status_code=400, detail="judge_ids must not be empty")

    params = {"run_ids": body.run_ids, "judge_ids": body.judge_ids, "workers": body.workers}
    job_run_id = run_registry.create_run("JUDGE_V1", "batch", params)
    thread = threading.Thread(
        target=engine_service.start_judge_v1_run, args=(job_run_id, params), daemon=True,
    )
    thread.start()
    return {"run_id": job_run_id}


@router.get("/status/{run_id}")
def status(run_id: str):
    run = run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Judge-v1 job not found")
    return run


@router.post("/cancel/{run_id}")
def cancel(run_id: str):
    run = run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Judge-v1 job not found")
    run_registry.request_cancel(run_id)
    return {"ok": True}


def _existing_canonical_outputs(judge_ids: list[str]) -> list[str]:
    """The canonical per-judge_id dashboard output files that actually
    exist on disk -- the frontend never needs to know/pass absolute
    paths, it just names judge_ids. CLI/manual judge-v1 runs with custom
    --out paths aren't covered here by design (this is the DASHBOARD's
    own canonical file per judge_id, see engine_service.judge_v1_output_paths)."""
    paths = []
    for jid in judge_ids:
        out_path, _, _ = engine_service.judge_v1_output_paths(jid)
        if out_path.exists():
            paths.append(str(out_path))
    return paths


@router.get("/results/summary")
def results_summary(judge_id: str = "primary"):
    """Per-run_label hallucination (reference-based) rates for the given
    judge_id, read from its canonical dashboard output file."""
    mod = engine_service._judge_agreement_module()
    records = mod.load_judge_records(_existing_canonical_outputs([judge_id]))
    return {"summary": mod.compute_run_label_summary(records, judge_id)}


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


@router.post("/human-export")
def human_export(body: JudgeV1ExportHumanCsvRequest):
    from app.services import history_service

    mod = engine_service._judge_agreement_module()
    run_files = []
    for run_id in body.run_ids:
        detail = history_service.get_history_detail(run_id)
        if not detail or not detail.get("output_path"):
            raise HTTPException(status_code=400, detail=f"{run_id}: not found or no output file")
        run_files.append(detail["output_path"])

    out_dir = engine_service.REPO_ROOT / "llm" / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "jv1_dashboard_human_export.csv"
    primary_out, _, _ = engine_service.judge_v1_output_paths("primary")

    import argparse

    args = argparse.Namespace(
        run_files=run_files, judge_output=[str(primary_out)] if primary_out.exists() else [],
        out=str(out_path), mapping=None, n=body.n, seed=body.seed, min_per_label=body.min_per_label,
        questions_parquet=str(engine_service._questions_parquet()),
    )
    mod.cmd_export_human_csv(args)
    mapping_path = out_path.with_name(out_path.stem + "_mapping.jsonl")
    metadata_path = out_path.with_name(out_path.stem + "_metadata.json")
    return {"csv_path": str(out_path), "mapping_path": str(mapping_path), "metadata_path": str(metadata_path)}


@router.post("/human-kappa")
def human_kappa(body: JudgeV1ImportHumanCsvRequest):
    mod = engine_service._judge_agreement_module()
    primary_out, _, _ = engine_service.judge_v1_output_paths("primary")
    secondary_out, _, _ = engine_service.judge_v1_output_paths("secondary")
    judge_outputs = [str(p) for p in (primary_out, secondary_out) if p.exists()]
    if not judge_outputs:
        raise HTTPException(status_code=400, detail="No judge-v1 output files found yet")
    return {"results": mod.compute_human_kappa(body.csv_path, body.mapping_path, judge_outputs)}
