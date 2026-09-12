"""Joins LLM-as-judge results (llm/evaluation/) onto the existing History
pages -- read-only lookups against judge_run_history.jsonl (the manifest
llm_judge_hallucination.py appends to) and the individual judge output
files it produces. No write path here; running a judge evaluation goes
through engine_service.run_judge_batch_dashboard()/routers/judge.py.
"""

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
JUDGE_HISTORY_PATH = REPO_ROOT / "llm" / "evaluation" / "logs" / "judge_run_history.jsonl"


def _load_judge_run_history() -> list[dict[str, Any]]:
    if not JUDGE_HISTORY_PATH.exists():
        return []
    records = []
    with open(JUDGE_HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def get_judge_evaluations_for_input(input_path: str) -> list[dict[str, Any]]:
    """All judge_run_history.jsonl entries whose `input_path` matches the
    given condition result file (a condition's own `output_path`) --
    possibly more than one if that file has been judged with different
    configs (provider/model/temperature/majority_rounds). Returns a
    summary per entry (not the full per-question judge file) suitable for
    the History Detail page's "Hasil LLM-as-Judge" table.
    """
    if not input_path:
        return []
    matches = [r for r in _load_judge_run_history() if r.get("input_path") == input_path]
    matches.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return [
        {
            "judge_provider": r.get("judge_provider"),
            "judge_model": r.get("judge_model"),
            "judge_temperature": r.get("judge_temperature"),
            "majority_rounds": r.get("majority_rounds"),
            "n_total": r.get("n_total"),
            "pct_faktual": r.get("pct_faktual"),
            "pct_sebagian": r.get("pct_sebagian"),
            "pct_penuh": r.get("pct_penuh"),
            "mean_faithfulness": r.get("mean_faithfulness"),
            "mean_answer_relevance": r.get("mean_answer_relevance"),
            "kappa_value": r.get("kappa_value"),
            "kappa_interpretation": r.get("kappa_interpretation"),
            "evaluated_at": r.get("timestamp"),
            "output_path": r.get("output_path"),
        }
        for r in matches
    ]


def get_judge_results_for_question(input_path: str, question_id: int) -> list[dict[str, Any]]:
    """For every judge run that evaluated `input_path`, open ITS OWN output
    file (a separate path from `input_path`) and find the row for this
    question_id. Returns a list because more than one judge config may
    have evaluated the same source file."""
    evaluations = get_judge_evaluations_for_input(input_path)
    results = []
    for ev in evaluations:
        judge_output_path = ev.get("output_path")
        if not judge_output_path or not Path(judge_output_path).exists():
            continue
        with open(judge_output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("question_id") == question_id:
                    results.append({
                        "judge_provider": ev.get("judge_provider"),
                        "judge_model": ev.get("judge_model"),
                        "judge_temperature": ev.get("judge_temperature"),
                        "majority_rounds": ev.get("majority_rounds"),
                        "hallucination_label": row.get("hallucination_label"),
                        "faithfulness_score": row.get("faithfulness_score"),
                        "answer_relevance_score": row.get("answer_relevance_score"),
                        "justification": row.get("justification"),
                        "evaluated_at": row.get("evaluated_at"),
                    })
                    break
    return results
