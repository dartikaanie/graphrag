"""Pieces shared by the retrieval conditions (B, C, D) so that the parts that
must be identical across conditions -- query text, answer-document text and
its truncation, leakage exclusion -- live in exactly one place. See
reports/b_corpus_parity.md.

Nothing in this package imports neo4j: Condition B imports it at runtime and
must never touch Neo4j.
"""
