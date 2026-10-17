"""Per-query leakage exclusion, shared by Conditions B, C and D.

For the evaluation question being answered, retrieval must never return
its own thread: not the question itself (as a context item's
question_id / via-question) and not any of its answers (the accepted
answer being predicted, nor any other answer of that thread).

Inputs are plain ids -- no Neo4j, no parquet. Each condition supplies the
thread's answer ids from wherever it already gets them (C/D: the answers
parquet; B: kg_answers.parquet). When duplicate-question exclusion is added
later, it goes into build_exclusion() and reaches all three conditions.
"""

from dataclasses import dataclass
from typing import Iterable


class LeakageError(RuntimeError):
    """A retrieval result contains the evaluation question's own thread."""


@dataclass(frozen=True)
class LeakageExclusion:
    question_ids: frozenset
    answer_ids: frozenset

    def excludes(self, question_id, answer_id) -> bool:
        return int(question_id) in self.question_ids or int(answer_id) in self.answer_ids


def build_exclusion(question_id, thread_answer_ids: Iterable = (), accepted_answer_id=None) -> LeakageExclusion:
    """The exclusion set for one evaluation question: {question_id} and
    every answer id of its thread, plus `accepted_answer_id` (added
    explicitly in case the thread lookup missed it)."""
    answer_ids = {int(a) for a in thread_answer_ids}
    if accepted_answer_id is not None:
        answer_ids.add(int(accepted_answer_id))
    return LeakageExclusion(frozenset({int(question_id)}), frozenset(answer_ids))


def filter_excluded(items: list, exclusion: LeakageExclusion, limit: int | None = None) -> list:
    """Drop items (dicts with question_id/answer_id) the exclusion covers,
    keeping order; cut to `limit` if given."""
    kept = [it for it in items if not exclusion.excludes(it["question_id"], it["answer_id"])]
    return kept if limit is None else kept[:limit]


def assert_no_leakage(items: list, exclusion: LeakageExclusion) -> None:
    """Per-query check: retrieved items ∩ exclusion set = ∅."""
    leaked = [(int(it["question_id"]), int(it["answer_id"])) for it in items
              if exclusion.excludes(it["question_id"], it["answer_id"])]
    if leaked:
        raise LeakageError(
            f"retrieval returned {len(leaked)} item(s) from the evaluation question's own thread "
            f"(question_id, answer_id): {leaked[:5]}"
        )
