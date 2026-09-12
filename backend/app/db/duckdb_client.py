"""DuckDB helper for reading the SORD Parquet files (questions/answers).

Mirrors the dedup pattern already used by llm/a_pure_llm/a_baseline_replication.py
(ROW_NUMBER() OVER (PARTITION BY Id ...) rn=1) so counts/detail here stay
consistent with what the experiment scripts actually see.
"""

from functools import lru_cache

import duckdb

from app.config import get_settings


@lru_cache
def get_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    return con


def questions_parquet() -> str:
    # Settings-page override wins if set (same precedence as
    # engine_service._questions_parquet()) -- otherwise this and the run
    # engine can silently disagree about which Parquet file is "current"
    # whenever the path changes, since app.config.Settings is a process-
    # lifetime @lru_cache that never re-reads .env after backend startup.
    from app.services import settings_service

    override = settings_service.get_raw_settings().get("questions_parquet")
    return override or get_settings().resolved_questions_parquet()


def answers_parquet() -> str:
    from app.services import settings_service

    override = settings_service.get_raw_settings().get("answers_parquet")
    return override or get_settings().resolved_answers_parquet()


def dedup_questions_cte() -> str:
    return f"""
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{questions_parquet()}')
        )
        WHERE rn = 1
    """


def dedup_answers_cte() -> str:
    return f"""
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{answers_parquet()}')
        )
        WHERE rn = 1
    """
