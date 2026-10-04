"""
judge_clients.py
=====================================
Registry terpusat untuk DUA judge LLM-as-Judge (judge-v1, lihat
judge_prompt_v1.py):
  - "primary"   -- meta-llama/Llama-3.3-70B-Instruct via DeepInfra, model
                    keluarga BERBEDA dari generator (gpt-4o-mini) supaya
                    mengurangi self-preference bias. SENGAJA versi BF16,
                    BUKAN varian -Turbo/FP8 manapun -- DeepInfra meng-host
                    beberapa varian kuantisasi model yang sama dengan nama
                    mirip; versi BF16 dipilih di sini sebagai yang paling
                    dekat dengan bobot asli Meta, bukan yang paling cepat
                    atau paling murah.
  - "secondary" -- gpt-4o-mini via OpenAI, HANYA dipakai untuk mengukur
                    inter-judge agreement (Cohen's kappa) dengan primary,
                    bukan sumber label utama.

DeepInfra expose endpoint OpenAI-compatible (base_url berbeda, format
request/response sama) -- jadi client `openai.OpenAI` yang sama dipakai
untuk keduanya, hanya base_url + api_key yang beda. Ini BUKAN
llm.client_factory.get_llm_client() (yang tidak punya parameter base_url
dan dipakai generator A/B/C/D) -- modul ini terpisah khusus untuk judge.

Key dimuat dari .env di root repo via python-dotenv (DEEPINFRA_API_KEY,
OPENAI_API_KEY) -- TIDAK PERNAH diprint, dan .env sendiri sudah
dikonfirmasi untracked + diignore oleh git (lihat docs/Agent prompt llm
judge.md Step 2 -- dicek manual sebelum modul ini ditulis).
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class JudgeConfig:
    judge_id: str          # "primary" | "secondary" -- stored on every output record
    model: str
    base_url: str | None   # None == OpenAI's own default base URL
    api_key_env: str


JUDGE_REGISTRY: dict[str, JudgeConfig] = {
    "primary": JudgeConfig(
        judge_id="primary",
        model="meta-llama/Llama-3.3-70B-Instruct",
        base_url="https://api.deepinfra.com/v1/openai",
        api_key_env="DEEPINFRA_API_KEY",
    ),
    "secondary": JudgeConfig(
        judge_id="secondary",
        model="gpt-4o-mini",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
    ),
}


def get_judge_client(judge_id: str):
    """Return (OpenAI client, JudgeConfig) for "primary"/"secondary".
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
