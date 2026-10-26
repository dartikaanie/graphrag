"""The official run matrix on the test sample (n = 384): FOUR runs.

docs/DECISION_C_SCORING.md (Amendment 16) and docs/OFFICIAL_RUN_MATRIX.md. Thesis run -> cell id:

    thesis R1 = cell R1  A (plain LLM)
    thesis R2 = cell R3  B-grounded
    thesis R3 = cell R5  C-primary-grounded   (alpha_primary = alpha*)
    thesis R4 = cell R9  D-grounded

Grounding and citation instructions are ON for B, C and D. The other five cells (R2 B-plain, R4 C-primary-plain,
R6/R7 C-ablation, R8 D-plain) are LEGACY: still defined (Amendment 10 history stays reproducible) but refused on the
test sample unless --allow-legacy-cells (CLI) / allow_legacy_cells (API) is passed explicitly; see
legacy_cell_error(). build_matrix() returns the four official cells unless include_legacy=True.

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
C_RETRIEVAL_VERSION = "v4"  # Amendment 17: v3 ranking + answerable anchors (must equal c_graphrag.C_RETRIEVAL_VERSION)
D_RETRIEVAL_VERSION = "v5"  # Amendment 17: answerable anchors + fill up to k (must equal d_lightrag.D_RETRIEVAL_VERSION)
# Pinned in every official command and checked at start-up (pin_error), so a changed .env value cannot reach a test run.
EMBED_MODEL = "all-MiniLM-L6-v2"
FUSION_W_PATH_TRUST = 0.7           # = c_graphrag.DEFAULT_FUSION_W_PATH_TRUST
FUSION_W_INTRINSIC = 0.3            # = c_graphrag.DEFAULT_FUSION_W_ANSWER_INTRINSIC_TRUST
SEMANTIC_EXPANSION_TRUST_CAP = 0.4  # = c_graphrag.DEFAULT_SEMANTIC_EXPANSION_TRUST_CAP
LEAKAGE_MODE = "thread"  # Amendment 15: q + its thread only (supersedes Amendment 8's "dup+neardup0.95")

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

# The four official cells -> the run number the thesis uses (Bab III, Tabel matriks run).
OFFICIAL_CELLS = {"R1": "R1", "R3": "R2", "R5": "R3", "R9": "R4"}
LEGACY_FLAG = "--allow-legacy-cells"


def is_legacy_cell(run: str) -> bool:
    return run not in OFFICIAL_CELLS


def add_legacy_argument(parser) -> None:
    """--allow-legacy-cells for the B, C and D CLIs (see legacy_cell_error)."""
    parser.add_argument("--allow-legacy-cells", action="store_true", default=False,
                        help="Allow a test-sample run of a LEGACY matrix cell (plain, C-ablation). Without it such "
                             "a run is refused (Amendment 16: the official evaluation has four runs).")


def legacy_cell_error(condition: str, *, sample_split: str = "test", require_citation: bool | None = True,
                      require_grounding: bool | None = None, c_role: str | None = None,
                      allow_legacy: bool = False) -> str | None:
    """Reason a test-sample run is a LEGACY cell (and so refused), or None when it may start. Official cells:
    A; B with citation AND grounding; C at alpha_primary with grounding; D with grounding. Dev runs and
    allow_legacy=True are never refused."""
    if allow_legacy or sample_split != "test" or condition == "A":
        return None
    why = None
    if condition == "B" and not (require_citation and require_grounding):
        why = "B without citation+grounding (B-plain / plain RAG)"
    elif condition in ("C", "D") and not require_grounding:
        why = f"{condition} without grounding ({condition}-plain)"
    elif condition == "C" and c_role == ROLE_ABLATION:
        why = "C-ablation (alpha_ablation)"
    if why is None:
        return None
    return (f"{why} is a legacy cell: the official evaluation has four runs (A, B-grounded, C-primary-grounded, "
            f"D-grounded). Pass {LEGACY_FLAG} (API: allow_legacy_cells=true) to run it deliberately.")

def pin_error(condition: str, *, sample_split: str = "test", embed_model=None, fusion_w_path_trust=None,
              fusion_w_intrinsic=None, semantic_expansion_trust_cap=None) -> str | None:
    """Reason a TEST-sample B/C/D run must not start because a value that the command pins was changed (typically a
    .env value: EMBED_MODEL, FUSION_W_PATH_TRUST, FUSION_W_ANSWER_INTRINSIC_TRUST, SEMANTIC_EXPANSION_TRUST_CAP),
    or None. Dev runs are never checked. No legacy bypass: the pinned values are the same for every cell."""
    if sample_split != "test" or condition == "A":
        return None
    checks = [("embed_model (--embed-model / EMBED_MODEL)", embed_model, EMBED_MODEL)]
    if condition == "C":
        checks += [("fusion_w_path_trust", fusion_w_path_trust, FUSION_W_PATH_TRUST),
                   ("fusion_w_intrinsic", fusion_w_intrinsic, FUSION_W_INTRINSIC),
                   ("semantic_expansion_trust_cap", semantic_expansion_trust_cap, SEMANTIC_EXPANSION_TRUST_CAP)]
    bad = [f"{name} = {got!r} (frozen: {want!r})" for name, got, want in checks if got != want]
    if not bad:
        return None
    return ("Test-sample run refused: a pinned official parameter differs from the frozen value: " + "; ".join(bad)
            + ". Check .env and the command line (docs/OFFICIAL_RUN_MATRIX.md).")


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


def build_matrix(official: dict | None, include_legacy: bool = False) -> list[dict]:
    """One dict per run: run (cell id), thesis_run, legacy, label, condition, grounding (None for A),
    role (C only), alpha (C only; None while pending), cwd, script, argv
    (CLI arguments after the script), dashboard (RunCreateRequest fields). The four official cells only,
    unless include_legacy=True; legacy rows carry --allow-legacy-cells / allow_legacy_cells."""
    rows = []
    for run, label, condition, grounding, role in CELLS:
        legacy = is_legacy_cell(run)
        if legacy and not include_legacy:
            continue
        cwd, script = SCRIPTS[condition]
        argv = list(_COMMON_ARGS)
        dashboard = {"condition": condition, "mode": "batch", "provider": PROVIDER, "model": MODEL,
                     "n_sample": N_SAMPLE, "seed": SEED, "oversample_pool": OVERSAMPLE_POOL, "sample_split": "test"}
        alpha = None
        if condition == "B":
            argv += ["--top-k", str(TOP_K), "--token-chunk-limit", str(TOKEN_CHUNK_LIMIT), "--corpus", "kg_answers",
                     "--require-citation",
                     _grounding_flag(grounding), "--leakage-mode", LEAKAGE_MODE, "--embed-model", EMBED_MODEL]
            dashboard.update(top_k=TOP_K, require_citation=True, require_grounding=grounding)
        elif condition == "C":
            alpha = _alpha_value(official, role)
            alpha_arg = f"{alpha:g}" if alpha is not None else f"<{_ALPHA_KEY[role]}>"
            argv += ["--top-k", str(TOP_K), "--token-chunk-limit", str(TOKEN_CHUNK_LIMIT),
                     "--n-anchor", str(N_ANCHOR), "--n-semantic-expansion", str(N_SEMANTIC_EXPANSION),
                     "--c-retrieval-version", C_RETRIEVAL_VERSION, "--alpha", alpha_arg,
                     "--fusion-mode", "trust_weighted", _grounding_flag(grounding),
                     "--expansion-require-answer", "--leakage-mode", LEAKAGE_MODE, "--embed-model", EMBED_MODEL,
                     "--fusion-w-path-trust", f"{FUSION_W_PATH_TRUST:g}", "--fusion-w-intrinsic", f"{FUSION_W_INTRINSIC:g}",
                     "--semantic-expansion-trust-cap", f"{SEMANTIC_EXPANSION_TRUST_CAP:g}"]
            dashboard.update(top_k=TOP_K, n_anchor=N_ANCHOR, n_semantic_expansion=N_SEMANTIC_EXPANSION,
                             c_retrieval_version=C_RETRIEVAL_VERSION, alpha=alpha, fusion_mode="trust_weighted",
                             require_grounding=grounding)
            for key in ("max_hops", "edge_types", "use_author_trust", "accepted_only"):
                if official and key in official:
                    dashboard[key] = official[key]
        elif condition == "D":
            argv += ["--top-k", str(TOP_K), "--token-chunk-limit", str(TOKEN_CHUNK_LIMIT),
                     "--n-low-level", str(N_LOW_LEVEL), "--n-high-level", str(N_HIGH_LEVEL),
                     "--d-retrieval-version", D_RETRIEVAL_VERSION, _grounding_flag(grounding), "--leakage-mode", LEAKAGE_MODE,
                     "--embed-model", EMBED_MODEL]
            dashboard.update(top_k=TOP_K, n_low_level=N_LOW_LEVEL, n_high_level=N_HIGH_LEVEL,
                             require_grounding=grounding)
        if legacy:
            argv.append(LEGACY_FLAG)
            dashboard["allow_legacy_cells"] = True
        rows.append({"run": run, "thesis_run": OFFICIAL_CELLS.get(run), "legacy": legacy, "label": label, "condition": condition, "grounding": grounding, "role": role,
                     "alpha": alpha, "cwd": cwd, "script": script, "argv": argv, "dashboard": dashboard})
    return rows


def cli_command(row: dict) -> str:
    """Shell command for one run, from the repo root."""
    return f"cd {row['cwd']} && python {row['script']} " + " ".join(row["argv"])
