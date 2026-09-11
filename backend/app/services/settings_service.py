"""Dashboard configuration: provider/model/API keys/Neo4j credentials/data
paths, per PLAN_UI_UX.md §5.6 / §1.5 / §7.9.

SECURITY: this repo already had a real API key committed to git via a
repo-root .env (found during the initial audit, still not fully remediated
in git history). To not reopen that exact leak path, Settings state is
stored OUTSIDE the repo entirely, in the user's home directory, encrypted
at rest with Fernet (symmetric encryption; the key file is chmod 600 and
lives next to the encrypted blob, both outside git's reach by construction
since neither path is inside this repo). Secrets are never returned raw
from get_masked_settings() -- only a masked preview + an is_set flag -- and
are never logged.
"""

import json
import os
import stat
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

CONFIG_DIR = Path.home() / ".graphrag-dashboard"
CONFIG_PATH = CONFIG_DIR / "config.json"
KEY_PATH = CONFIG_DIR / "key"

SECRET_FIELDS = {"api_key", "anthropic_api_key", "neo4j_password"}

DEFAULTS: dict[str, Any] = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": "",
    "anthropic_api_key": "",
    "ollama_host": "http://localhost:11434",
    "num_ctx": 8192,
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "",
    "neo4j_database": "graphrag",
    "questions_parquet": "",
    "answers_parquet": "",
}


def _get_or_create_key() -> bytes:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if KEY_PATH.exists():
        return KEY_PATH.read_bytes()
    key = Fernet.generate_key()
    KEY_PATH.write_bytes(key)
    os.chmod(KEY_PATH, stat.S_IRUSR | stat.S_IWUSR)
    return key


def _seed_from_existing_env() -> dict[str, Any]:
    """First-run convenience only: copies whatever the repo-root .env
    already has into the new encrypted store, ONE TIME. After this the
    encrypted store is authoritative -- .env is never read or written
    again by this service.

    Reads the .env FILE directly (via dotenv_values) rather than
    os.getenv/os.environ: this runs on the very first GET/PUT /api/settings
    call in a fresh backend process, before any condition module has been
    imported (imports are what trigger those scripts' own load_dotenv()
    calls) -- os.environ may not have these vars populated yet at that
    point, which would otherwise seed empty keys even though .env has them.
    """
    from dotenv import dotenv_values

    from app.config import get_settings as get_app_settings

    env_file = dotenv_values(str(Path(__file__).resolve().parents[3] / ".env"))
    app_settings = get_app_settings()
    seeded = dict(DEFAULTS)
    seeded.update({
        "provider": env_file.get("LLM_PROVIDER") or DEFAULTS["provider"],
        "model": env_file.get("LLM_MODEL") or DEFAULTS["model"],
        "api_key": env_file.get("OPENAI_API_KEY") or "",
        "anthropic_api_key": env_file.get("ANTHROPIC_API_KEY") or "",
        "neo4j_uri": app_settings.neo4j_uri,
        "neo4j_user": app_settings.neo4j_user,
        "neo4j_password": app_settings.neo4j_password,
        "neo4j_database": app_settings.neo4j_database,
        "questions_parquet": app_settings.questions_parquet,
        "answers_parquet": app_settings.answers_parquet,
    })
    return seeded


def _load() -> dict[str, Any] | None:
    if not CONFIG_PATH.exists():
        return None
    try:
        fernet = Fernet(_get_or_create_key())
        return json.loads(fernet.decrypt(CONFIG_PATH.read_bytes()))
    except (InvalidToken, ValueError, json.JSONDecodeError):
        return None


def _save(data: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    fernet = Fernet(_get_or_create_key())
    CONFIG_PATH.write_bytes(fernet.encrypt(json.dumps(data).encode()))
    os.chmod(CONFIG_PATH, stat.S_IRUSR | stat.S_IWUSR)


def get_raw_settings() -> dict[str, Any]:
    """Full values including secrets in the clear -- for internal backend
    use only (applying to a run, testing a connection). Never return this
    directly from an API response.
    """
    data = _load()
    if data is None:
        data = _seed_from_existing_env()
        _save(data)
    merged = dict(DEFAULTS)
    merged.update(data)
    return merged


def _mask(value: str) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}...{value[-4:]}"


def get_masked_settings() -> dict[str, Any]:
    data = get_raw_settings()
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in SECRET_FIELDS:
            out[key] = {"is_set": bool(value), "masked": _mask(value)}
        else:
            out[key] = value
    return out


def update_settings(patch: dict[str, Any]) -> dict[str, Any]:
    data = get_raw_settings()
    for key, value in patch.items():
        if key not in DEFAULTS or value is None:
            continue
        # An empty string for a secret field means "leave it as-is" (the
        # frontend never round-trips the real secret back to us to re-save
        # it) -- only a non-empty value actually replaces it.
        if key in SECRET_FIELDS and value == "":
            continue
        data[key] = value
    _save(data)
    return get_masked_settings()


def api_key_for_provider(data: dict[str, Any], provider: str) -> str:
    return data.get("anthropic_api_key", "") if provider == "anthropic" else data.get("api_key", "")


def apply_to_environment() -> None:
    """Push saved settings into process env vars right before a run, so the
    existing condition scripts (which read os.getenv(...) directly, e.g.
    llm/client_factory.py, c_graphrag.py's connect_neo4j) pick them up
    without needing their own settings-awareness. Only overwrites a var
    when the saved value is actually non-empty, so an unconfigured Settings
    field falls back to whatever the repo-root .env already provides.
    """
    data = get_raw_settings()
    if data["api_key"]:
        os.environ["OPENAI_API_KEY"] = data["api_key"]
    if data["anthropic_api_key"]:
        os.environ["ANTHROPIC_API_KEY"] = data["anthropic_api_key"]
    if data["neo4j_uri"]:
        os.environ["NEO4J_URI"] = data["neo4j_uri"]
    if data["neo4j_user"]:
        os.environ["NEO4J_USER"] = data["neo4j_user"]
    if data["neo4j_password"]:
        os.environ["NEO4J_PASSWORD"] = data["neo4j_password"]
    if data["neo4j_database"]:
        os.environ["NEO4J_DATABASE"] = data["neo4j_database"]


def test_neo4j(uri: str, user: str, password: str) -> tuple[bool, str]:
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=5)
        try:
            driver.verify_connectivity()
        finally:
            driver.close()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def test_llm_provider(provider: str, api_key: str, model: str, ollama_host: str) -> tuple[bool, str]:
    try:
        if provider == "openai":
            from openai import OpenAI

            OpenAI(api_key=api_key).models.list()
            return True, "Connected"
        if provider == "anthropic":
            import anthropic

            anthropic.Anthropic(api_key=api_key).messages.create(
                model=model, max_tokens=1, messages=[{"role": "user", "content": "hi"}]
            )
            return True, "Connected"
        if provider == "ollama":
            import ollama

            ollama.Client(host=ollama_host).list()
            return True, "Connected"
        return False, f"Unknown provider '{provider}'"
    except Exception as e:
        return False, str(e)
