"""The official run matrix R1-R9 on the test sample (n = 384).

docs/DECISION_C_SCORING.md (Amendment 10) and docs/OFFICIAL_RUN_MATRIX.md:

    R1 A            R2 B-plain        R3 B-grounded
    R4 C-primary-plain     R5 C-primary-grounded     (alpha_primary  = alpha*)
    R6 C-ablation-plain    R7 C-ablation-grounded    (alpha_ablation = the other stage-2 candidate)
    R8 D-plain      R9 D-grounded

C's alphas come from llm/c_graphrag/official_params.json only; while it is
pending they are None and the CLI strings carry a placeholder. Every C cell
uses C retrieval v3 with fusion_mode "trust_weighted" (the CLI default; v3
ignores fusion_mode for ranking, but it is part of config_hash, so CLI and
dashboard runs of the same cell share one config_hash and output file).

Leakage mode and the Amendment 9 expansion switch are pinned explicitly on
the CLI so a stray .env value cannot change an official run. The dashboard
takes both from LeakageSettings / EXPANSION_REQUIRE_ANSWER (env), whose
defaults are the same values.

Pure data: no heavy imports, safe for the dashboard backend.
"""

from llm.c_graphrag.official_params import ROLE_ABLATION, ROLE_PRIMARY, STATUS_SELECTED

N_SAMPLE = 384
SEED = 42
PROVIDER = "openai"
MODEL = "gpt-4o-mini"
TOP_K = 5
OVERSAMPLE_POOL = N_SAMPLE * 4  # 1536, the pool every condition samples from
TOKEN_CHUNK_LIMIT = 400
N_ANCHOR = 3
N_SEMANTIC_EXPANSION = 3
N_LOW_LEVEL = 3
N_HIGH_LEVEL = 3
LEAKAGE_MODE = "dup+neardup0.95"

_COMMON_ARGS = ["--n-sample", str(N_SAMPLE), "--seed", str(SEED), "--oversample-pool", str(OVERSAMPLE_POOL),
                "--sample-split", "test", "--provider", PROVIDER, "--model", MODEL]

CELLS = (
    ("R1", "A", "A", None, None),
    ("R2", "B-plain", "B", False, None),
    ("R3", "B-grounded", "B", True, None),
    ("R4", "C-primary-plain", "C", False, ROLE_PRIMARY),
    ("R5", "C-primary-grounded", "C", True, ROLE_PRIMARY),
    ("R6", "C-ablation-plain", "C", False, ROLE_ABLATION),
    ("R7", "C-ablation-grounded", "C", True, ROLE_ABLATION),
    ("R8", "D-plain", "D", False, None),
    ("R9", "D-grounded", "D", True, None),
)

SCRIPTS = {
    "A": ("llm/a_pure_llm", "a_baseline_replication.py"),
    "B": ("llm/b_rag", "b_condition_b_rag.py"),
    "C": ("llm/c_graphrag", "c_graphrag.py"),
    "D": ("llm/d_lightrag", "d_lightrag.py"),
}

_ALPHA_KEY = {ROLE_PRIMARY: "alpha_primary", ROLE_ABLATION: "alpha_ablation"}


def _grounding_flag(grounding: bool) -> str:
    return "--require-grounding" if grounding else "--no-require-grounding"


def _alpha_value(official: dict | None, role: str):
    if not official or official.get("status") != STATUS_SELECTED:
        return None
    return official[_ALPHA_KEY[role]]


def build_matrix(official: dict | None) -> list[dict]:
    """One dict per run: run, label, condition, grounding (None for A),
    role (C only), alpha (C only; None while pending), cwd, script, argv
    (CLI arguments after the script), dashboard (RunCreateRequest fields)."""
    rows = []
    for run, label, condition, grounding, role in CELLS:
        cwd, script = SCRIPTS[condition]
        argv = list(_COMMON_ARGS)
        dashboard = {"condition": condition, "mode": "batch", "provider": PROVIDER, "model": MODEL,
                     "n_sample": N_SAMPLE, "seed": SEED, "oversample_pool": OVERSAMPLE_POOL, "sample_split": "test"}
        alpha = None
        if condition == "B":
            argv += ["--top-k", str(TOP_K), "--token-chunk-limit", str(TOKEN_CHUNK_LIMIT), "--corpus", "kg_answers",
                     "--require-citation",
                     _grounding_flag(grounding), "--leakage-mode", LEAKAGE_MODE]
            dashboard.update(top_k=TOP_K, require_citation=True, require_grounding=grounding)
        elif condition == "C":
            alpha = _alpha_value(official, role)
            alpha_arg = f"{alpha:g}" if alpha is not None else f"<{_ALPHA_KEY[role]}>"
            argv += ["--top-k", str(TOP_K), "--token-chunk-limit", str(TOKEN_CHUNK_LIMIT),
                     "--n-anchor", str(N_ANCHOR), "--n-semantic-expansion", str(N_SEMANTIC_EXPANSION),
                     "--c-retrieval-version", "v3", "--alpha", alpha_arg,
                     "--fusion-mode", "trust_weighted", _grounding_flag(grounding),
                     "--expansion-require-answer", "--leakage-mode", LEAKAGE_MODE]
            dashboard.update(top_k=TOP_K, n_anchor=N_ANCHOR, n_semantic_expansion=N_SEMANTIC_EXPANSION,
                             c_retrieval_version="v3", alpha=alpha, fusion_mode="trust_weighted",
                             require_grounding=grounding)
            for key in ("max_hops", "edge_types", "use_author_trust", "accepted_only"):
                if official and key in official:
                    dashboard[key] = official[key]
        elif condition == "D":
            argv += ["--top-k", str(TOP_K), "--token-chunk-limit", str(TOKEN_CHUNK_LIMIT),
                     "--n-low-level", str(N_LOW_LEVEL), "--n-high-level", str(N_HIGH_LEVEL),
                     _grounding_flag(grounding), "--leakage-mode", LEAKAGE_MODE]
            dashboard.update(top_k=TOP_K, n_low_level=N_LOW_LEVEL, n_high_level=N_HIGH_LEVEL,
                             require_grounding=grounding)
        rows.append({"run": run, "label": label, "condition": condition, "grounding": grounding, "role": role,
                     "alpha": alpha, "cwd": cwd, "script": script, "argv": argv, "dashboard": dashboard})
    return rows


def cli_command(row: dict) -> str:
    """Shell command for one run, from the repo root."""
    return f"cd {row['cwd']} && python {row['script']} " + " ".join(row["argv"])
