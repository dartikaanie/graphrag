"""
frozen_parameters.py
====================
The frozen parameter table of the official evaluation (Amendment 14 draft,
step 4): every value is read from the file that defines it -- module
constants through `ast` (no heavy imports, no Neo4j/FAISS), the official
parameter file, the run matrix and the judge panel config -- and recorded
with its source location. Writes:
    docs/FROZEN_PARAMETERS.md            (Markdown table)
    results/frozen_parameters.json       (for the dashboard's report page)

    python3 llm/frozen_parameters.py [--check]   # --check: exit 1 if the files are stale
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "llm" / "evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MD_PATH = REPO_ROOT / "docs" / "FROZEN_PARAMETERS.md"
JSON_PATH = REPO_ROOT / "results" / "frozen_parameters.json"
OFFICIAL_PARAMS = "llm/c_graphrag/official_params.json"


def const(rel: str, name: str):
    """(value, 'file:line') of a module-level `NAME = <literal>` assignment."""
    tree = ast.parse((REPO_ROOT / rel).read_text())
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name and node.value is not None:
                return ast.literal_eval(node.value), f"{rel}:{node.lineno}"
    raise KeyError(f"{name} not found in {rel}")


def _line_of(rel: str, needle: str) -> str:
    for i, line in enumerate((REPO_ROOT / rel).read_text().splitlines(), 1):
        if needle in line:
            return f"{rel}:{i}"
    raise KeyError(f"{needle!r} not found in {rel}")


def build() -> list[dict]:
    from llm import official_run_matrix as orm
    from judge_panel_config import (MAX_RETRIES, PANEL_V2_ID, PANEL_V2_JUDGES)
    import judge_prompt_v2_3 as jp

    official = json.loads((REPO_ROOT / OFFICIAL_PARAMS).read_text())
    out: list[dict] = []

    def add(group, name, value, source, status="frozen", decision=""):
        out.append({"group": group, "parameter": name, "value": value, "source": source, "status": status,
                    "decision": decision})

    # --- alpha selection ---
    v, s = const("llm/c_graphrag/stage1_sweep.py", "DEFAULT_ALPHA_VALUES")
    add("Alpha selection", "stage-1 alpha grid", list(v), s, "done", "Selection procedure; stage 1 of record: Amendment 17 (C v4 re-run)")
    from llm.c_graphrag.stage2_candidates import EXPLORATORY_ALPHAS, STAGE1_REPORT_OF_RECORD, STAGE2_ALPHAS
    add("Alpha selection", "stage-2 candidates", list(STAGE2_ALPHAS),
        _line_of("llm/c_graphrag/stage2_candidates.py", "STAGE2_ALPHAS = stage1_top_two()"), "candidate (stage 2 not run)",
        f"derived from the top_two of {STAGE1_REPORT_OF_RECORD} (Amendment 17); never written by hand")
    add("Alpha selection", "exploratory reference alphas (not eligible for selection)", list(EXPLORATORY_ALPHAS),
        _line_of("llm/c_graphrag/stage2_candidates.py", "EXPLORATORY_ALPHAS ="), "frozen (pre-declared)",
        "run like the candidates, reported separately, never an input of the selection rule")
    add("Alpha selection", "alpha* (C-primary; the only alpha of the official runs)", official["alpha_primary"], f"{OFFICIAL_PARAMS}#alpha_primary",
        official["status"], "set by stage2_report.py (Amendment 10)")
    add("Alpha selection", "alpha_abl (second stage-2 candidate)", official["alpha_ablation"],
        f"{OFFICIAL_PARAMS}#alpha_ablation", official["status"],
        "stage 2 still produces it (Amendment 10); NO official run uses it since Amendment 16 (legacy cell R6/R7)")
    v, s = const("llm/c_graphrag/ctxrel_sweep.py", "STAGE2_ANSWER_COUNT_TIE_MARGIN")
    add("Alpha selection", "stage-2 tie margin (answers)", v, s, "frozen",
        "Selection procedure; tie-breaks: secondary criterion, then larger alpha")
    add("Alpha selection", "stage-2 metric", "binary hallucination rate excl. ABSTAIN, pooled over grounding on/off, "
        "PANEL label (Amendment 14 draft (g))", _line_of("llm/c_graphrag/stage2_report.py", "def stage2_metrics"),
        "frozen (Amendment 14 draft)", "Amendment 4 binary metric")

    # --- run matrix (Amendment 16) ---
    rm = "llm/official_run_matrix.py"
    add("Run matrix", "official runs (thesis R1-R4 = cell id)", {t: [r["run"], r["label"]] for r in orm.build_matrix(official)
                                                              for t in [r["thesis_run"]]},
        _line_of(rm, "OFFICIAL_CELLS ="), "frozen (Amendment 16)",
        "A, B-grounded, C-primary-grounded, D-grounded; grounding + citation instructions ON for B, C, D")
    add("Run matrix", "legacy cells (refused on the test sample without --allow-legacy-cells)",
        [r["label"] for r in orm.build_matrix(official, include_legacy=True) if r["legacy"]], _line_of(rm, "LEGACY_FLAG ="),
        "frozen (Amendment 16)", "plain and C-ablation cells of Amendment 10; kept reproducible, never started by default")
    add("Run matrix", "pre-registered comparisons", ["C vs A (R5 vs R1)", "C vs B (R5 vs R3)", "C vs D (R5 vs R9)"],
        _line_of("llm/evaluation/official_report.py", "COMPARISONS ="), "frozen (Amendment 16)",
        "hallucination rate, panel label, ABSTAIN excluded")

    # --- trust score ---
    tf = "01_data_cleaning/trust_formulas.py"
    for name in ("TRUST_SCORE_VERSION", "W_A_SCORE", "W_A_ACCEPTED", "W_A_AUTHOR", "W_Q_SCORE", "W_Q_VIEWS",
                 "P99_QUANTILE", "QUANTILE_METHOD"):
        v, s = const(tf, name)
        add("trustScore", name, v, s, "frozen", "Amendments 5-6")
    v, s = const("llm/c_graphrag/stage2_report.py", "TRUST_V3_APPLIED_AT")
    add("trustScore", "v3-logscore applied at", v, s, "frozen", "Amendment 6")

    # --- C retrieval ---
    cg = "llm/c_graphrag/c_graphrag.py"
    add("C retrieval", "C retrieval version (official)", official["c_retrieval_version"],
        f"{OFFICIAL_PARAMS}#c_retrieval_version", "frozen (Amendment 17)",
        "v4 = v3 ranking + answerable-only anchors; v3 (Amendment 16) stays reproducible")
    v, s = const("llm/common/anchors.py", "ANCHOR_SCAN_DEPTH")
    add("C retrieval", "entity anchoring (C v4 and D v5, one shared function)",
        f"first n_anchor questions with >= 1 non-excluded answer, scanning {v} candidates of the same FAISS list", s,
        "frozen (Amendment 17)", "llm/common/anchors.py::select_answerable_anchors; fewer found -> use what was found, recorded")
    add("C retrieval", "traversal depth (max_hops)", official["max_hops"], f"{OFFICIAL_PARAMS}#max_hops")
    add("C retrieval", "edge types (5)", official["edge_types"], f"{OFFICIAL_PARAMS}#edge_types")
    add("C retrieval", "use_author_trust", official["use_author_trust"], f"{OFFICIAL_PARAMS}#use_author_trust",
        "frozen", "Amendment 5 (off: reputation already in trustScore)")
    add("C retrieval", "accepted_only", official["accepted_only"], f"{OFFICIAL_PARAMS}#accepted_only")
    add("C retrieval", "semantic expansion mode", "answerable-only (--expansion-require-answer)",
        _line_of("llm/official_run_matrix.py", '"--expansion-require-answer"'), "frozen", "Amendment 9")
    for name in ("DEFAULT_SEMANTIC_EXPANSION_TRUST_CAP", "EXPANSION_OVERFETCH", "DEFAULT_FUSION_W_PATH_TRUST",
                 "DEFAULT_FUSION_W_ANSWER_INTRINSIC_TRUST"):
        v, s = const(cg, name)
        add("C retrieval", name, v, s)
    add("C retrieval", "LIMIT for two-hop candidates", 50, _line_of(cg, "LIMIT 50"), "frozen",
        "per query, ordered by edge weight then answer id")
    add("C retrieval", "N_anchor", orm.N_ANCHOR, _line_of("llm/official_run_matrix.py", "N_ANCHOR ="))
    add("C retrieval", "n_semantic_expansion", orm.N_SEMANTIC_EXPANSION,
        _line_of("llm/official_run_matrix.py", "N_SEMANTIC_EXPANSION ="))
    add("C retrieval", "fusion_mode (part of config_hash; v3 ranks by alpha)", "trust_weighted",
        _line_of("llm/official_run_matrix.py", '"--fusion-mode", "trust_weighted"'))

    # --- retrieval, all conditions ---
    add("Retrieval (B, C, D)", "k (top_k) per condition", {"A": None, "B": orm.TOP_K, "C": orm.TOP_K, "D": orm.TOP_K},
        _line_of("llm/official_run_matrix.py", "TOP_K ="))
    add("Retrieval (B, C, D)", "candidate text limit (tokens per context item)", orm.TOKEN_CHUNK_LIMIT,
        _line_of("llm/official_run_matrix.py", "TOKEN_CHUNK_LIMIT ="))
    add("Retrieval (B, C, D)", "B corpus", "kg_answers", _line_of("llm/official_run_matrix.py", '"--corpus", "kg_answers"'),
        "frozen", "B corpus parity amendment")
    add("Retrieval (B, C, D)", "D n_low_level / n_high_level", [orm.N_LOW_LEVEL, orm.N_HIGH_LEVEL],
        _line_of("llm/official_run_matrix.py", "N_LOW_LEVEL ="))
    v, s = const("llm/d_lightrag/d_lightrag.py", "D_RETRIEVAL_VERSION")
    add("Retrieval (B, C, D)", "D retrieval version (official)", v, s, "frozen (Amendment 17)",
        "v5 = v4 (one cosine score on both levels) + answerable-only anchors + the high level fills up to k answers; "
        "v4 and v3 (edge-weight scores) stay reproducible")
    add("Retrieval (B, C, D)", "D v5 high level: fill rule", "top n_high_level neighbours by cosine, then more neighbours in "
        "score order until >= k distinct answers (low-level answers count) or none left; top-k answers kept",
        _line_of("llm/d_lightrag/d_lightrag.py", "def select_high_level_fill"), "frozen (Amendment 17)", "k = top_k = 5")
    add("Retrieval (B, C, D)", "D retrieval version pinned in the run matrix", orm.D_RETRIEVAL_VERSION,
        _line_of("llm/official_run_matrix.py", "D_RETRIEVAL_VERSION ="), "frozen (Amendment 17)",
        "test_d_retrieval_v5 checks it equals d_lightrag.D_RETRIEVAL_VERSION")
    add("Retrieval (B, C, D)", "C retrieval version pinned in the run matrix", orm.C_RETRIEVAL_VERSION,
        _line_of("llm/official_run_matrix.py", "C_RETRIEVAL_VERSION ="), "frozen (Amendment 17)",
        "test_c_retrieval_v4 checks it equals c_graphrag.C_RETRIEVAL_VERSION")
    v, s = const("llm/d_lightrag/d_lightrag.py", "HIGH_LEVEL_CANDIDATE_CAP")
    add("Retrieval (B, C, D)", "D v4 cap on neighbour Questions scored per query", v, s, "frozen (Amendment 16)",
        "smallest question ids kept when exceeded; recorded as high_level_cap_hit")
    v, s = const(cg, "EMBED_DIM")
    add("Retrieval (B, C, D)", "embedding model", f"all-MiniLM-L6-v2 ({v}-dim)", _line_of(cg, '"all-MiniLM-L6-v2"'))

    # --- leakage ---
    lk = "llm/common/leakage.py"
    add("Leakage exclusion", "mode (official runs)", orm.LEAKAGE_MODE, _line_of("llm/official_run_matrix.py", "LEAKAGE_MODE ="),
        "frozen", "Amendment 15")
    add("Leakage exclusion", "rules", "exclude q and every answer in its thread (incl. accepted) and nothing else; "
        "Duplicate, Linked and near-duplicate Questions are NOT excluded (Amendment 8's dup+neardup0.95 superseded)",
        _line_of(lk, "def build_exclusion"), "frozen", "Amendment 15")
    v, s = const(lk, "DEFAULT_NEAR_DUP_COSINE")
    add("Leakage exclusion", "near-duplicate cosine threshold (sensitivity mode only, not applied in `thread`)", v, s,
        "informational", "Amendment 8")
    add("Leakage exclusion", "precomputed exclusion sets (read only by the sensitivity modes; `thread` needs none)",
        "results/leakage/exclusion_sets.parquet", "results/leakage/exclusion_sets.meta.json", "informational",
        "Amendment 8")

    # --- generation and sampling ---
    v, s = const("llm/prompts.py", "PROMPT_VERSION")
    add("Generation", "generator", f"{orm.PROVIDER} {orm.MODEL}", _line_of("llm/official_run_matrix.py", "MODEL ="),
        "frozen", "Amendment 4 (generator only, never a judge)")
    add("Generation", "generator prompt version", v, s)
    add("Generation", "temperature / max_tokens", "not sent: provider defaults (recorded per answer as "
        "request_params.temperature = null)", _line_of("llm/client_factory.py", "if temperature is not None"),
        "frozen", "as in every pilot, dev and calibration run")
    add("Sampling", "n (test sample)", orm.N_SAMPLE, _line_of("llm/official_run_matrix.py", "N_SAMPLE ="))
    add("Sampling", "seed", orm.SEED, _line_of("llm/official_run_matrix.py", "SEED ="))
    add("Sampling", "oversample pool", orm.OVERSAMPLE_POOL, _line_of("llm/official_run_matrix.py", "OVERSAMPLE_POOL ="))
    v, s = const(cg, "TOKEN_LIMIT")
    add("Sampling", "question token limit (exclusion)", v, s)
    v, s = const("llm/sample_split.py", "DEFAULT_DEV_OFFSET")
    add("Sampling", "dev positions", f"{v}-{v + 49} (n = 50; frozen list llm/sample_ids/dev_ids_v1.json)", s)
    add("Sampling", "test positions", "1-384 (frozen list llm/sample_ids/test_ids_v1.json)",
        _line_of("llm/sample_split.py", "MAX_TEST_N ="))

    # --- reporting ---
    v, s = const("llm/record_schema.py", "RECORD_SCHEMA_VERSION")
    add("Reporting", "answer record schema", f"{v} (per-answer answer_started_at_utc / answer_finished_at_utc / "
        "end_to_end_latency_sec around retrieval + generation; generation_latency_sec)", s, "frozen (Amendment 14 draft)")
    v, s = const("llm/evaluation/official_report.py", "NF3_TARGET_SEC")
    add("Reporting", "NF3 latency", f"p95 of end_to_end_latency_sec over a Condition C run <= {v} s "
        "(fallback for archived runs: run duration / answers, labelled)", s, "frozen (Amendment 14 draft)")

    # --- judge ---
    jv = "llm/evaluation/judge_prompt_v2_3.py"
    for label, name in (("prompt", "PROMPT_VERSION"), ("parser", "PARSE_VERSION"), ("temperature", "REQUEST_TEMPERATURE"),
                        ("max_tokens", "REQUEST_MAX_TOKENS"), ("JSON mode", "REQUEST_RESPONSE_FORMAT")):
        v, s = const(jv, name)
        add("Judge", label, v, s, "frozen (Amendment 14 draft)")
    add("Judge", "blinding", jp.BLINDING_VERSION, _line_of("llm/evaluation/judge_blinding.py", "BLINDING_VERSION ="),
        "frozen (Amendment 14 draft)")
    add("Judge", "retries", MAX_RETRIES, _line_of("llm/evaluation/judge_panel_config.py", "MAX_RETRIES ="),
        "frozen (Amendment 14 draft)")
    add("Judge", "panel", PANEL_V2_ID, _line_of("llm/evaluation/judge_panel_config.py", "PANEL_V2_ID ="),
        "frozen (Amendment 14 draft)")
    for j in PANEL_V2_JUDGES:
        add("Judge", f"panel member: {j.judge_id}", {"model": j.model, "extra_body": j.extra_body,
                                                     "price_per_m": [j.price_per_m_input, j.price_per_m_output]},
            _line_of("llm/evaluation/judge_panel_config.py", f'"{j.model}"'), "frozen (Amendment 14 draft)")
    add("Judge", "aggregation", "2-of-3 majority (judge_panel_aggregate.aggregate_labels)",
        _line_of("llm/evaluation/judge_panel_aggregate.py", "def aggregate_labels"), "frozen (Amendment 14 draft)")
    # --- secondary metrics (Amendment 16) ---
    sm = "llm/evaluation/secondary_metrics.py"
    add("Secondary metrics", "judge (one, not the panel)", "primary = meta-llama/Llama-3.3-70B-Instruct, temperature 0.0",
        _line_of(sm, "JUDGE_ID ="), "frozen (Amendment 16)", "context relevance B/C/D, answer relevance A-D; secondary only")
    for label, name in (("context-relevance prompt", "CTXREL_PROMPT"), ("answer-relevance prompt", "ANSREL_PROMPT")):
        v, s = const(sm, name)
        add("Secondary metrics", label, v, s, "frozen (Amendment 16)")
    return out


def _fmt(v) -> str:
    if isinstance(v, (dict, list, tuple)):
        return "`" + json.dumps(v, ensure_ascii=False) + "`"
    return "--" if v is None else f"`{v}`"


def render(table: list[dict]) -> tuple[str, dict]:
    # the hash covers the VALUES only (group, parameter, value): moving a line in a source file does not change it
    body = json.dumps([[r["group"], r["parameter"], r["value"]] for r in table], sort_keys=True, ensure_ascii=False)
    config_hash = hashlib.sha256(body.encode()).hexdigest()[:12]
    lines = ["# Frozen parameters of the official evaluation", "",
             "Generated by `llm/frozen_parameters.py` from the files that define each value (do not edit by hand). "
             f"Parameter-table hash: `{config_hash}`. Status \"frozen (Amendment 14 draft)\" becomes binding when "
             "Amendment 14 is approved; \"frozen (Amendment 16)\" is the four-run design. alpha\\* is filled in by stage 2 "
             "(the second candidate alpha_abl is used by no official run).", ""]
    group = None
    for r in table:
        if r["group"] != group:
            group = r["group"]
            lines += ["", f"## {group}", "", "| Parameter | Value | Status | Source | Decision |", "|---|---|---|---|---|"]
        lines.append(f"| {r['parameter']} | {_fmt(r['value'])} | {r['status']} | `{r['source']}` | {r['decision'] or '--'} |")
    return "\n".join(lines) + "\n", {"table_hash": config_hash, "parameters": table}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    md, js = render(build())
    js_text = json.dumps(js, indent=1, ensure_ascii=False) + "\n"
    if args.check:
        stale = [p for p, t in ((MD_PATH, md), (JSON_PATH, js_text)) if not p.exists() or p.read_text() != t]
        print("stale: " + ", ".join(map(str, stale)) if stale else "up to date")
        return 1 if stale else 0
    MD_PATH.write_text(md)
    JSON_PATH.write_text(js_text)
    print(f"{len(js['parameters'])} parameters, hash {js['table_hash']} -> {MD_PATH.relative_to(REPO_ROOT)}, "
          f"{JSON_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
