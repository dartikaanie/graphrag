"""
app/config.py
=====================================
Central settings loader for Phase 1 (backend skeleton).

NOTE (Phase 7 TODO, see PLAN_UI_UX.md §5.6 / §1.5): once the Settings page
exists, provider/model/API-key/Neo4j credentials must move to an encrypted
file OUTSIDE the git repo (e.g. ~/.graphrag-dashboard/config.json) so we
don't repeat the leaked-.env pattern already found in this repo's git
history. For now (Phase 1: master-data + stats endpoints only, no API keys
touched), we read the existing root .env — it is already gitignored and
contains no new secrets beyond what the CLI scripts already use.
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    questions_parquet: str = "00_datasource/merged/questions_raw_union.parquet"
    answers_parquet: str = "00_datasource/merged/answers_raw_union.parquet"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "graphrag"

    def resolved_questions_parquet(self) -> str:
        p = Path(self.questions_parquet)
        return str(p if p.is_absolute() else REPO_ROOT / p)

    def resolved_answers_parquet(self) -> str:
        p = Path(self.answers_parquet)
        return str(p if p.is_absolute() else REPO_ROOT / p)


@lru_cache
def get_settings() -> Settings:
    return Settings()
