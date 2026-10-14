"""Drift tests for the Process page: every value the page shows must equal
the value the pipeline actually uses.

* Constants: app/process/sources.py reads them with `ast` (no import). Here
  each module is really imported -- in a subprocess, with `dotenv` stubbed so
  `.env` is never read and sys.path changes don't leak into other tests --
  and the imported value is compared with the /api/process/parameters value.
* CLI defaults: compared with the config the stage-1 runs actually recorded
  in their manifests.
* Log/report values: cross-checked against an independent artifact.

No Neo4j, no network.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.process import sources
from app.process.registry import PARAMETERS

REPO_ROOT = Path(__file__).resolve().parents[2]
client = TestClient(app)

# registry key -> (module file, expression evaluated in that module's namespace)
RUNTIME = {
    "data.valid_link_types": ("01_data_cleaning/6_postlinks_to_sord.py", "VALID_LINK_TYPES"),
    "sampling.oversample_pool": ("backend/app/services/engine_service.py", "DEFAULT_OVERSAMPLE_POOL"),
    "sampling.oversample_pool_calibration": ("llm/evaluation/calibration_generate.py", "OVERSAMPLE_POOL"),
    "sampling.token_limit": ("llm/c_graphrag/c_graphrag.py", "TOKEN_LIMIT"),
    "sampling.dev_offset": ("llm/sample_split.py", "DEFAULT_DEV_OFFSET"),
    "sampling.n_dev": ("llm/c_graphrag/stage2_report.py", "N_DEV"),
    "kg.tag_max_df": ("01_data_cleaning/10_densify_tag_cooccurrence.py", "DEFAULT_MAX_DF"),
    "kg.tag_jaccard": ("01_data_cleaning/10_densify_tag_cooccurrence.py", "DEFAULT_JACCARD_THRESHOLD"),
    "kg.tag_top_k": ("01_data_cleaning/10_densify_tag_cooccurrence.py", "DEFAULT_TOP_K"),
    "kg.tag_weight": ("01_data_cleaning/10_densify_tag_cooccurrence.py", "DEFAULT_TRUST_WEIGHT"),
    "kg.emb_top_k": ("01_data_cleaning/11_densify_embedding_similarity.py", "DEFAULT_TOP_K"),
    "kg.emb_cosine": ("01_data_cleaning/11_densify_embedding_similarity.py", "DEFAULT_COSINE_THRESHOLD"),
    "kg.emb_weight": ("01_data_cleaning/11_densify_embedding_similarity.py", "DEFAULT_TRUST_WEIGHT"),
    "kg.target_isolated": ("01_data_cleaning/12_validate_densification.py", "TARGET_ISOLATED_PCT"),
    "kg.target_mean_degree": ("01_data_cleaning/12_validate_densification.py", "TARGET_MEAN_DEGREE"),
    "kg.target_median_degree": ("01_data_cleaning/12_validate_densification.py", "TARGET_MEDIAN_DEGREE"),
    "trust.w_a_score": ("01_data_cleaning/trust_formulas.py", "W_A_SCORE"),
    "trust.w_a_accepted": ("01_data_cleaning/trust_formulas.py", "W_A_ACCEPTED"),
    "trust.w_a_author": ("01_data_cleaning/trust_formulas.py", "W_A_AUTHOR"),
    "trust.w_q_score": ("01_data_cleaning/trust_formulas.py", "W_Q_SCORE"),
    "trust.w_q_views": ("01_data_cleaning/trust_formulas.py", "W_Q_VIEWS"),
    "trust.version": ("01_data_cleaning/trust_formulas.py", "TRUST_SCORE_VERSION"),
    "trust.p99_quantile": ("01_data_cleaning/trust_formulas.py", "P99_QUANTILE"),
    "trust.quantile_method": ("01_data_cleaning/trust_formulas.py", "QUANTILE_METHOD"),
    "trust.applied_at": ("llm/c_graphrag/stage2_report.py", "TRUST_V3_APPLIED_AT"),
    "cond.prompt_version": ("llm/prompts.py", "PROMPT_VERSION"),
    "cond.c_retrieval_version": ("llm/c_graphrag/c_graphrag.py", "C_RETRIEVAL_VERSION"),
    "cond.d_retrieval_version": ("llm/d_lightrag/d_lightrag.py", "D_RETRIEVAL_VERSION"),
    "crank.alpha_candidates": ("llm/c_graphrag/stage1_sweep.py", "list(DEFAULT_ALPHA_VALUES)"),
    "crank.fusion_w_path": ("llm/c_graphrag/c_graphrag.py", "DEFAULT_FUSION_W_PATH_TRUST"),
    "crank.fusion_w_intrinsic": ("llm/c_graphrag/c_graphrag.py", "DEFAULT_FUSION_W_ANSWER_INTRINSIC_TRUST"),
    "crank.semantic_cap": ("llm/c_graphrag/c_graphrag.py", "DEFAULT_SEMANTIC_EXPANSION_TRUST_CAP"),
    "crank.max_hops": ("llm/c_graphrag/c_graphrag.py", "DEFAULT_MAX_HOPS"),
    "crank.edge_types": ("llm/c_graphrag/c_graphrag.py", "list(DEFAULT_EDGE_TYPES)"),
    "crank.use_author_trust": ("llm/c_graphrag/c_graphrag.py", "DEFAULT_USE_AUTHOR_TRUST"),
    "crank.author_trust_beta": ("llm/c_graphrag/c_graphrag.py", "AUTHOR_TRUST_BETA"),
    "crank.accepted_only": ("llm/c_graphrag/c_graphrag.py", "DEFAULT_ACCEPTED_ONLY"),
    "sel.ctxrel_version": ("llm/evaluation/ctxrel_prompt_v2.py", "PROMPT_VERSION"),
    "sel.control_positive": ("llm/c_graphrag/stage1_sweep.py", "CONTROL_THRESHOLDS['positive'][1]"),
    "sel.control_easy": ("llm/c_graphrag/stage1_sweep.py", "CONTROL_THRESHOLDS['easy_negative'][1]"),
    "sel.control_hard": ("llm/c_graphrag/stage1_sweep.py", "CONTROL_THRESHOLDS['hard_negative'][1]"),
    "sel.tie_pp": ("llm/c_graphrag/ctxrel_sweep.py", "TIE_THRESHOLD_PERCENTAGE_POINTS"),
    "sel.tie_score": ("llm/c_graphrag/ctxrel_sweep.py", "RELEVANCE_SCORE_TIE_THRESHOLD"),
    "sel.stage2_tie": ("llm/c_graphrag/ctxrel_sweep.py", "STAGE2_ANSWER_COUNT_TIE_MARGIN"),
    "sel.stage2_alphas": ("llm/c_graphrag/stage2_report.py", "list(STAGE2_ALPHAS)"),
    "judge.primary_model": ("llm/evaluation/judge_clients.py", "JUDGE_REGISTRY['primary'].model"),
    "judge.secondary_model": ("llm/evaluation/judge_clients.py", "JUDGE_REGISTRY['secondary'].model"),
    "judge.version": ("llm/evaluation/judge_prompt_v2_1.py", "PROMPT_VERSION"),
    "judge.parse_version": ("llm/evaluation/judge_prompt_v2_1.py", "PARSE_VERSION"),
    "judge.temperature": ("llm/evaluation/judge_prompt_v2_1.py", "REQUEST_TEMPERATURE"),
    "judge.max_tokens": ("llm/evaluation/judge_prompt_v2_1.py", "REQUEST_MAX_TOKENS"),
    "judge.kappa_target": ("llm/evaluation/annotation_tool.py", "ACCEPTANCE_KAPPA"),
    "judge.n_annotation": ("llm/evaluation/calibration_generate.py", "N_ANNOTATION_ITEMS"),
    "off.n": ("backend/app/services/engine_service.py", "MAX_PLANNED_N_SAMPLE"),
    "met.latency_target": ("01_data_cleaning/12_validate_densification.py", "TARGET_LATENCY_SEC"),
    "met.hallucination_metric_stats": ("llm/evaluation/compare_conditions_stats.py", "list(METRICS_HALLUCINATION)"),
}

# registry key -> config field recorded by the C runs of stage 1 (token_chunk_limit is
# not recorded in those manifests, so crank.token_chunk_limit is only checked by
# the ast reader itself)
CLI_IN_MANIFESTS = {
    "sampling.seed": "seed",
    "crank.top_k": "top_k",
    "crank.n_anchor": "n_anchor",
    "crank.n_semantic_expansion": "n_semantic_expansion",
    "cond.generation_model": "model",
}

_LOADER = r"""
import importlib.util, json, sys, types
from pathlib import Path
repo = Path(sys.argv[1])
fake = types.ModuleType("dotenv")
fake.load_dotenv = lambda *a, **k: False          # never read .env
fake.find_dotenv = lambda *a, **k: ""
fake.dotenv_values = lambda *a, **k: {}         # pydantic-settings reads env files through this
sys.modules["dotenv"] = fake
sys.path[:0] = [str(repo), str(repo / "backend")]
out, loaded = {}, {}
for key, (rel, expr) in json.loads(sys.argv[2]).items():
    path = repo / rel
    sys.path.insert(0, str(path.parent))
    if rel in loaded:
        mod = loaded[rel]
    elif rel.startswith("backend/"):
        mod = importlib.import_module(rel[len("backend/"):-3].replace("/", "."))
    else:
        spec = importlib.util.spec_from_file_location("drift_" + path.stem.replace("-", "_"), path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    loaded[rel] = mod
    value = eval(expr, vars(mod))
    out[key] = {str(k): v for k, v in value.items()} if isinstance(value, dict) else value
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def runtime_values():
    proc = subprocess.run(
        [sys.executable, "-c", _LOADER, str(REPO_ROOT), json.dumps(RUNTIME)],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def page_values():
    body = client.get("/api/process/parameters").json()
    return {row["key"]: row for row in body["parameters"]}


def test_every_parameter_resolves(page_values):
    errors = {k: r["error"] for k, r in page_values.items() if r["error"]}
    assert errors == {}


@pytest.mark.parametrize("key", sorted(RUNTIME))
def test_displayed_value_equals_runtime_value(key, runtime_values, page_values):
    assert page_values[key]["value"] == runtime_values[key], key


def test_runtime_map_covers_every_module_constant():
    consts = {p.key for p in PARAMETERS if p.source.kind in ("const", "call_kwarg")}
    assert consts - set(RUNTIME) == set()


def _stage1_c_configs():
    configs = []
    for p in (REPO_ROOT / "llm" / "c_graphrag" / "results").glob("condition_c_*_dev_v3_*.jsonl.manifest.json"):
        configs.append(json.loads(p.read_text())["config"])
    return configs


@pytest.mark.parametrize("key", sorted(CLI_IN_MANIFESTS))
def test_cli_default_equals_value_recorded_by_stage1_runs(key, page_values):
    configs = _stage1_c_configs()
    assert configs, "no stage-1 C manifests found"
    recorded = {c[CLI_IN_MANIFESTS[key]] for c in configs}
    assert recorded == {page_values[key]["value"]}, key


def test_source_literals_point_at_their_line(page_values):
    for p in PARAMETERS:
        if p.source.kind != "source_literal":
            continue
        row = page_values[p.key]
        line = sources.read_text(row["defined_in"]["file"]).splitlines()[row["defined_in"]["line"] - 1]
        assert repr(row["value"]) in line or str(row["value"]) in line, p.key


def test_p99_matches_the_stage1_report(page_values):
    reports = sorted((REPO_ROOT / "llm/c_graphrag/results/logscore_check").glob("stage1_logscore_report_*.json"))
    meta = json.loads(reports[-1].read_text())["meta"]
    assert page_values["trust.p99_answer"]["value"] == meta["p99_answer"]


def test_official_fields_equal_official_params_json(page_values):
    official = json.loads((REPO_ROOT / "llm/c_graphrag/official_params.json").read_text())
    for key, field in (("off.status", "status"), ("crank.alpha_official", "alpha"),
                       ("off.max_hops", "max_hops"), ("off.edge_types", "edge_types"),
                       ("off.use_author_trust", "use_author_trust"), ("off.accepted_only", "accepted_only"),
                       ("off.sample_split", "sample_split")):
        assert page_values[key]["value"] == official[field], key


def test_calibration_pool_copy_equals_dashboard_pool(page_values):
    assert page_values["sampling.oversample_pool_calibration"]["value"] == page_values["sampling.oversample_pool"]["value"]


def test_ast_reader_detects_a_changed_constant(tmp_path, monkeypatch):
    """A changed source value is picked up on the next read (no stale cache)."""
    monkeypatch.setattr(sources, "REPO_ROOT", tmp_path)
    f = tmp_path / "m.py"
    f.write_text("X = 1\n")
    assert sources.resolve(sources.Src("const", "m.py", "X")).value == 1
    f.write_text("X = 2  # changed\n")
    import os
    os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 1_000_000))
    assert sources.resolve(sources.Src("const", "m.py", "X")).value == 2


def test_ambiguous_literal_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "REPO_ROOT", tmp_path)
    (tmp_path / "m.py").write_text("a = f(0.5)\nb = f(0.5)\n")
    with pytest.raises(sources.SourceError):
        sources.resolve(sources.Src("source_literal", "m.py", r"f\(([0-9.]+)\)"))
