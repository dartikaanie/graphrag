"""Per-question output record: fields shared by Conditions A-D.

RECORD_SCHEMA_VERSION
    "r1" (implicit): every record written before this module existed -- no
         run_id / condition / latency fields beyond retrieval_latency_sec
         (B/C/D only).
    "r2": adds, at the top level of every record, run_id (the run's UUID,
         the same value as the manifest's run_id), condition, and
         record_schema_version, plus generation_latency_sec (the generator
         LLM call only), total_latency_sec (end to end per question: from
         the start of retrieval -- or of prompt building for A -- until
         the record is complete, before it is written) and, for A,
         retrieval_latency_sec = None. Condition D's context items also
         carry is_accepted. No field was removed or renamed.
    "r3": adds per-answer end-to-end timing (Amendment 14 draft; NF3):
         answer_started_at_utc / answer_finished_at_utc (wall clock, ISO 8601
         UTC) and end_to_end_latency_sec = finished - started, spanning
         retrieval (prompt building for A) + generation and ending when the
         generator call returns -- evaluation-only steps (citation
         extraction, cosine similarity) are excluded, unlike
         total_latency_sec. generation_latency_sec is unchanged. NF3 = p95 of
         end_to_end_latency_sec over a Condition C run. Records without these
         fields (r1/r2, the archived runs) fall back to run duration / answers,
         labelled as a fallback.

The record fields are not part of config / config_hash, so resume and
existing output paths are unchanged.

Per-item retrieval provenance is versioned separately: B/C/D records carry
provenance_version ("v2") and their retrieved_context items carry path /
anchor_id / alt_paths / rank (llm/common/item_provenance.py); record_schema_version
is unchanged. Records without provenance_version predate it.
"""

import uuid
from datetime import datetime, timezone

RECORD_SCHEMA_VERSION = "r3"


def new_run_id() -> str:
    return str(uuid.uuid4())


def record_header(run_id: str, condition: str) -> dict:
    return {"run_id": run_id, "condition": condition, "record_schema_version": RECORD_SCHEMA_VERSION}


def latency_fields(retrieval_sec: float | None, generation_sec: float | None, total_sec: float | None) -> dict:
    def r(v):
        return None if v is None else round(float(v), 3)

    return {"retrieval_latency_sec": r(retrieval_sec), "generation_latency_sec": r(generation_sec),
            "total_latency_sec": r(total_sec)}


def answer_timing(started: float, finished: float) -> dict:
    """Per-answer end-to-end timing (schema r3): `started` / `finished` are time.time() values taken at the start
    of retrieval (prompt building for A) and right after the generator call returned."""
    def iso(t):
        return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="milliseconds")

    return {"answer_started_at_utc": iso(started), "answer_finished_at_utc": iso(finished),
            "end_to_end_latency_sec": round(float(finished - started), 3)}
