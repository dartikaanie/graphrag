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

The record fields are not part of config / config_hash, so resume and
existing output paths are unchanged.
"""

import uuid

RECORD_SCHEMA_VERSION = "r2"


def new_run_id() -> str:
    return str(uuid.uuid4())


def record_header(run_id: str, condition: str) -> dict:
    return {"run_id": run_id, "condition": condition, "record_schema_version": RECORD_SCHEMA_VERSION}


def latency_fields(retrieval_sec: float | None, generation_sec: float | None, total_sec: float | None) -> dict:
    def r(v):
        return None if v is None else round(float(v), 3)

    return {"retrieval_latency_sec": r(retrieval_sec), "generation_latency_sec": r(generation_sec),
            "total_latency_sec": r(total_sec)}
