"""
judge_clients.py
=====================================
Registry terpusat untuk judge-model LLM-as-Judge (judge-v1, lihat
judge_prompt_v1.py), dengan exactly satu model per ROLE:
  - "primary"   -- judge_id yg dipakai utk angka hallucination resmi.
  - "secondary" -- judge_id dipakai HANYA utk Cohen's kappa agreement
                    dengan primary, bukan sumber label resmi.
  - "fallback"  -- pre-registered, dipakai HANYA kalau primary gagal
                    validasi human-kappa -- lihat docs/
                    agent_prompt_phase2_judge_ui.md.
  - "exploratory" -- model manapun yg ditambahkan belakangan, TIDAK
                    PERNAH masuk angka resmi (primary/secondary).

PRECISION CORRECTION (2026-10-05): modul ini SEBELUMNYA mengklaim
primary = "SENGAJA versi BF16, BUKAN varian -Turbo/FP8" untuk
meta-llama/Llama-3.3-70B-Instruct di DeepInfra. Klaim itu SALAH --
dikonfirmasi via `client.models.list()` DAN satu live chat completion
call: DeepInfra HANYA meng-host varian -Turbo (FP8) utk model ini
("meta-llama/Llama-3.3-70B-Instruct-Turbo") -- meminta ID tanpa suffix
"-Turbo" tetap di-route ke situ (response.model kembali dgn suffix
-Turbo). Tidak ada endpoint BF16 terpisah yang tersedia di DeepInfra utk
model ini (atau utk Llama-3.1-70B-Instruct, dicek sama). `precision`
field di bawah sekarang mencatat kenyataan ini (FP8/Turbo), bukan klaim
yang tidak terverifikasi. Harga $0.10/$0.32 per 1M token itu SENDIRI
sudah benar (itu harga Turbo yang memang dipakai) -- yang salah hanya
LABEL presisinya, bukan harganya.

DeepInfra expose endpoint OpenAI-compatible (base_url berbeda, format
request/response sama) -- jadi client `openai.OpenAI` yang sama dipakai
untuk keduanya, hanya base_url + api_key yang beda. Ini BUKAN
llm.client_factory.get_llm_client() (yang tidak punya parameter base_url
dan dipakai generator A/B/C/D) -- modul ini terpisah khusus untuk judge.

Key dimuat dari .env di root repo via python-dotenv (DEEPINFRA_API_KEY,
OPENAI_API_KEY) -- TIDAK PERNAH diprint, dan .env sendiri sudah
dikonfirmasi untracked + diignore oleh git.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGE_ROLE_CHANGES_PATH = REPO_ROOT / "logs" / "judge_role_changes.jsonl"

ROLES = ("primary", "secondary", "fallback", "exploratory")


@dataclass(frozen=True)
class JudgeConfig:
    judge_id: str          # stored on every output record
    model: str
    base_url: str | None   # None == OpenAI's own default base URL
    api_key_env: str
    role: str              # one of ROLES -- exactly one "primary", one "secondary"
    precision: str         # what's ACTUALLY served, not an assumption -- see module docstring
    price_per_m_input: float
    price_per_m_output: float
    price_date: str        # ISO date the prices below were last checked/set
    expected_served_model: str | None = None  # the model id the PROVIDER actually
    # reports back via response.model for THIS judge_id's requests -- may
    # differ from `model` above (e.g. DeepInfra silently routes a bare
    # "Llama-3.3-70B-Instruct" request to the "-Turbo" variant). judge_one()
    # compares every response.model against this and warns (logs + a
    # `served_model_mismatch` field on the record) if the provider ever
    # starts serving something else -- a silent provider-side model swap
    # would otherwise go completely unnoticed.
    # Last-resort token/latency estimate for the plan table, ONLY used
    # when no real judge-v1 records exist yet for this judge_model
    # anywhere (see engine_service._judge_token_latency_stats's fallback
    # chain: real records for this exact judge+version -> real records
    # for this judge_model from ANY judge-v1 output file -> these
    # defaults). ~1200 in / ~600 out matches the 2026-10-05 smoke test's
    # actual observed tokens for this prompt (judge_prompt_v1.py's
    # messages are long: full question + reference answer + candidate).
    default_input_tokens_per_item: int = 1200
    default_output_tokens_per_item: int = 600
    default_latency_sec_per_item: float = 30.0


JUDGE_REGISTRY: dict[str, JudgeConfig] = {
    "primary": JudgeConfig(
        judge_id="primary",
        model="meta-llama/Llama-3.3-70B-Instruct",
        base_url="https://api.deepinfra.com/v1/openai",
        api_key_env="DEEPINFRA_API_KEY",
        role="primary",
        precision="FP8 (Turbo), as served by DeepInfra, verified via response.model on 2026-10-05",
        price_per_m_input=0.10,
        price_per_m_output=0.32,
        price_date="2026-10-05",
        expected_served_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ),
    "secondary": JudgeConfig(
        judge_id="secondary",
        model="gpt-4o-mini",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        role="secondary",
        precision="unspecified (OpenAI-hosted, not disclosed by provider)",
        price_per_m_input=0.15,
        price_per_m_output=0.60,
        price_date="2026-10-05",
        # OpenAI echoes back the pinned dated snapshot for the "gpt-4o-mini"
        # alias, NOT the bare alias string -- verified via a live call on
        # 2026-10-05. Using the bare alias here would make every single
        # record falsely report served_model_mismatch=True.
        expected_served_model="gpt-4o-mini-2024-07-18",
    ),
    "fallback": JudgeConfig(
        judge_id="fallback",
        model="meta-llama/Llama-3.1-70B-Instruct",
        base_url="https://api.deepinfra.com/v1/openai",
        api_key_env="DEEPINFRA_API_KEY",
        role="fallback",
        precision="FP8 (Turbo), as served by DeepInfra, verified via response.model on 2026-10-05 -- accepted "
                   "(2026-10-05) for both primary and fallback; no BF16 provider was integrated, see "
                   "docs/agent_prompt_phase2_judge_ui.md's final report for the verification detail.",
        price_per_m_input=0.40,
        price_per_m_output=0.40,
        price_date="2026-10-05",
        expected_served_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    ),
}


def validate_registry() -> None:
    """Exactly one primary, exactly one secondary -- raises ValueError
    (not an assertion) so a caller (e.g. a startup check or a test) gets
    a clear message rather than a bare AssertionError."""
    by_role: dict[str, list[str]] = {role: [] for role in ROLES}
    for judge_id, config in JUDGE_REGISTRY.items():
        if config.role not in ROLES:
            raise ValueError(f"Judge '{judge_id}' has unknown role '{config.role}'")
        by_role[config.role].append(judge_id)
    if len(by_role["primary"]) != 1:
        raise ValueError(f"Expected exactly 1 'primary' judge, found {by_role['primary']}")
    if len(by_role["secondary"]) != 1:
        raise ValueError(f"Expected exactly 1 'secondary' judge, found {by_role['secondary']}")


def record_role_change(old_judge_id: str | None, new_judge_id: str, role: str, reason: str) -> None:
    """Appends to logs/judge_role_changes.jsonl -- append-only, never
    overwritten. This is the ONLY sanctioned way to change which judge_id
    holds an official role (primary/secondary); the UI must never let a
    user flip JUDGE_REGISTRY's roles without going through this (and, in
    practice, a code review of the registry change itself, since
    JUDGE_REGISTRY is a Python module constant, not a runtime-editable
    setting)."""
    JUDGE_ROLE_CHANGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "old_judge_id": old_judge_id,
        "new_judge_id": new_judge_id,
        "role": role,
        "reason": reason,
    }
    with open(JUDGE_ROLE_CHANGES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def registry_public_view() -> list[dict[str, Any]]:
    """Registry entries WITHOUT api_key_env's actual value (never read
    the env var here) -- safe to return from an API endpoint. Includes
    whether each judge's key is currently configured (bool only)."""
    return [
        {
            "judge_id": c.judge_id, "model": c.model, "role": c.role,
            "precision": c.precision, "price_per_m_input": c.price_per_m_input,
            "price_per_m_output": c.price_per_m_output, "price_date": c.price_date,
            "base_url": c.base_url, "api_key_configured": bool(os.getenv(c.api_key_env)),
        }
        for c in JUDGE_REGISTRY.values()
    ]


def get_judge_client(judge_id: str):
    """Return (OpenAI client, JudgeConfig) for any registered judge_id.
    Raises a clear error (never a bare KeyError/openai SDK traceback) if
    judge_id is unknown or its API key env var isn't set -- callers (the
    runner CLI) print this and exit cleanly rather than crash mid-batch.
    """
    if judge_id not in JUDGE_REGISTRY:
        raise ValueError(f"Unknown judge_id '{judge_id}' -- must be one of {list(JUDGE_REGISTRY)}")

    config = JUDGE_REGISTRY[judge_id]
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{config.api_key_env} not set -- required for judge '{judge_id}' ({config.model}). "
            f"Add it to the repo-root .env file."
        )

    from openai import OpenAI

    kwargs = {"api_key": api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs), config
