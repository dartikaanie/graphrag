"""Singleton Neo4j driver + session helper, shared across services."""

from functools import lru_cache

from neo4j import GraphDatabase, Driver

from app.config import get_settings


@lru_cache
def get_driver() -> Driver:
    settings = get_settings()
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def run_query(query: str, parameters: dict | None = None) -> list[dict]:
    settings = get_settings()
    driver = get_driver()
    with driver.session(database=settings.neo4j_database) as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]


def close_driver():
    get_driver.cache_clear()
