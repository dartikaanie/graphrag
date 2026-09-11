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
    return get_settings().resolved_questions_parquet()


def answers_parquet() -> str:
    return get_settings().resolved_answers_parquet()


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
