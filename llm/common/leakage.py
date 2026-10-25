"""Per-query leakage exclusion, shared by Conditions B, C and D.

For the evaluation question q being answered, retrieval must never return
q itself or anything that is q under another id:

  thread   q, every answer of q's thread, q's accepted answer. NOTHING ELSE:
             Duplicate, Linked and near-duplicate questions stay retrievable.
             The official mode since Amendment 15 (it supersedes Amendment 8's
             dup+neardup0.95 for the official evaluation).
  dup      + Questions linked to q by IS_RELATED_TO {link_type: 'Duplicate'},
             either direction, with all their answers          (always, when on)
  linked   + Questions linked by IS_RELATED_TO {link_type: 'Linked'}, either
             direction, with all their answers                  (LEAKAGE_EXCLUDE_LINKED)
  neardup  + Questions whose Title+Body embedding has cosine >= threshold
             with q's, with all their answers                   (LEAKAGE_NEAR_DUP_COSINE)

Exclusion is by question identity / similarity only -- never by the content
of q's accepted answer (that is only measured, in the audit).

This module never imports neo4j: Condition B imports it at runtime. The
related-question part (dup / linked / neardup) is precomputed offline by
llm/leakage/build_exclusion_sets.py into results/leakage/exclusion_sets.parquet
(one row per pool question per mode) and only READ here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXCLUSION_SETS_PATH = REPO_ROOT / "results" / "leakage" / "exclusion_sets.parquet"

MODE_OFF = "off"        # LEAKAGE_FILTER=False -- audit only
MODE_THREAD = "thread"  # q + its thread only; the official mode (Amendment 15), also what every run before Amendment 8 used
DEFAULT_NEAR_DUP_COSINE = 0.95


class LeakageError(RuntimeError):
    """A retrieval result contains the evaluation question's own thread."""


class LeakageSettings(BaseSettings):
    """LEAKAGE_FILTER / LEAKAGE_EXCLUDE_DUP / LEAKAGE_EXCLUDE_LINKED /
    LEAKAGE_NEAR_DUP_COSINE, from the environment or the repo-root .env.
    Folded into each condition's config (and so its config_hash) as
    "leakage": <mode>. With nothing but LEAKAGE_FILTER on (the default) the
    mode is "thread" (Amendment 15); the dup / linked / neardup modes are
    opt-in sensitivity modes."""

    model_config = SettingsConfigDict(env_file=str(REPO_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore")

    leakage_filter: bool = True
    leakage_exclude_dup: bool = False
    leakage_exclude_linked: bool = False
    leakage_near_dup_cosine: Optional[float] = None

    @field_validator("leakage_near_dup_cosine", mode="before")
    @classmethod
    def _none_strings(cls, v):
        if isinstance(v, str) and v.strip().lower() in ("", "none", "null", "off"):
            return None
        return v

    @property
    def mode(self) -> str:
        return leakage_mode(self.leakage_filter, self.leakage_exclude_linked, self.leakage_near_dup_cosine,
                            exclude_dup=self.leakage_exclude_dup)


def leakage_mode(leakage_filter: bool = True, exclude_linked: bool = False,
                 near_dup_cosine: Optional[float] = DEFAULT_NEAR_DUP_COSINE, exclude_dup: bool = True) -> str:
    """Canonical mode string: "thread" (no dup, linked or near-dup exclusion;
    the settings default), "dup", "dup+neardup0.95", "dup+linked+neardup0.95",
    "off". Linked and near-dup exclusion imply dup. The function's own
    defaults (exclude_dup=True, 0.95) keep returning "dup+neardup0.95", the
    Amendment 8 mode; LeakageSettings passes its own (thread) defaults."""
    if not leakage_filter:
        return MODE_OFF
    if not exclude_dup and not exclude_linked and near_dup_cosine is None:
        return MODE_THREAD
    mode = "dup"
    if exclude_linked:
        mode += "+linked"
    if near_dup_cosine is not None:
        mode += f"+neardup{float(near_dup_cosine):g}"
    return mode


def default_mode() -> str:
    return LeakageSettings().mode


def config_leakage_key(mode: Optional[str]) -> dict:
    """The config entry for a mode: {} for None (the legacy / unspecified
    mode of every run made before Amendment 8 -- their config_hash must not
    change), {"leakage": mode} for every named mode, "thread" included
    (Amendment 15: a new thread run must not share a config_hash with a
    pre-Amendment-8 run)."""
    return {} if mode is None else {"leakage": mode}


@dataclass(frozen=True)
class LeakageExclusion:
    question_ids: frozenset
    answer_ids: frozenset

    def excludes(self, question_id, answer_id) -> bool:
        return int(question_id) in self.question_ids or int(answer_id) in self.answer_ids


NO_EXCLUSION = LeakageExclusion(frozenset(), frozenset())


def build_exclusion(question_id, thread_answer_ids: Iterable = (), accepted_answer_id=None,
                    related_question_ids: Iterable = (), related_answer_ids: Iterable = ()) -> LeakageExclusion:
    """The exclusion set for one evaluation question: {question_id} ∪
    related_question_ids, and thread_answer_ids ∪ {accepted_answer_id} ∪
    related_answer_ids. Ids only -- callers pass the related part from
    ExclusionTable (or nothing, for MODE_THREAD)."""
    q_ids = {int(question_id)} | {int(q) for q in related_question_ids}
    a_ids = {int(a) for a in thread_answer_ids} | {int(a) for a in related_answer_ids}
    if accepted_answer_id is not None:
        a_ids.add(int(accepted_answer_id))
    return LeakageExclusion(frozenset(q_ids), frozenset(a_ids))


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


class ExclusionTable:
    """Rows of exclusion_sets.parquet for one mode: question_id ->
    (excluded_q_ids, excluded_a_ids, n_dup, n_linked, n_neardup)."""

    def __init__(self, mode: str, rows: dict):
        self.mode = mode
        self._rows = rows

    @classmethod
    def load(cls, mode: str, path=DEFAULT_EXCLUSION_SETS_PATH) -> "ExclusionTable":
        import pyarrow.parquet as pq

        path = Path(path)
        if not path.exists():
            raise LeakageError(f"{path} not found -- run llm/leakage/build_exclusion_sets.py first.")
        table = pq.read_table(path, filters=[("mode", "=", mode)])
        if table.num_rows == 0:
            raise LeakageError(f"mode '{mode}' is not in {path} -- rerun llm/leakage/build_exclusion_sets.py "
                               f"with that mode.")
        d = table.to_pydict()
        rows = {int(q): (frozenset(int(x) for x in eq), frozenset(int(x) for x in ea), nd, nl, nn)
                for q, eq, ea, nd, nl, nn in zip(d["question_id"], d["excluded_q_ids"], d["excluded_a_ids"],
                                                 d["n_dup"], d["n_linked"], d["n_neardup"])}
        return cls(mode, rows)

    def __contains__(self, question_id) -> bool:
        return int(question_id) in self._rows

    def related(self, question_id) -> tuple[frozenset, frozenset]:
        try:
            q_ids, a_ids, *_ = self._rows[int(question_id)]
        except KeyError:
            raise LeakageError(
                f"question {question_id} has no precomputed exclusion set (mode '{self.mode}') -- run "
                f"llm/leakage/build_exclusion_sets.py --question-ids {question_id}"
            ) from None
        return q_ids, a_ids

    def counts(self, question_id) -> dict:
        _, _, nd, nl, nn = self._rows[int(question_id)]
        return {"n_dup": nd, "n_linked": nl, "n_neardup": nn}


class LeakageContext:
    """What a condition needs per query: the mode and (for dup modes) the
    precomputed table. MODE_THREAD needs no table; MODE_OFF excludes nothing."""

    def __init__(self, mode: str, table: ExclusionTable | None = None):
        if mode not in (MODE_OFF, MODE_THREAD) and table is None:
            raise ValueError(f"mode '{mode}' needs an ExclusionTable")
        self.mode = mode
        self.table = table

    @classmethod
    def from_mode(cls, mode: str, path=DEFAULT_EXCLUSION_SETS_PATH) -> "LeakageContext":
        return cls(mode, None if mode in (MODE_OFF, MODE_THREAD) else ExclusionTable.load(mode, path))

    @classmethod
    def from_settings(cls, settings: LeakageSettings | None = None, path=DEFAULT_EXCLUSION_SETS_PATH):
        return cls.from_mode((settings or LeakageSettings()).mode, path)

    @property
    def active(self) -> bool:
        return self.mode != MODE_OFF

    def exclusion_for(self, question_id, thread_answer_ids: Iterable = (), accepted_answer_id=None) -> LeakageExclusion:
        if self.mode == MODE_OFF:
            return NO_EXCLUSION
        if self.mode == MODE_THREAD:
            return build_exclusion(question_id, thread_answer_ids, accepted_answer_id)
        rq, ra = self.table.related(question_id)
        return build_exclusion(question_id, thread_answer_ids, accepted_answer_id, rq, ra)

    def query_log(self, exclusion: LeakageExclusion, n_hits_filtered: int, final_k: int) -> dict:
        """The per-query record/log entry every condition writes."""
        return {"mode": self.mode, "n_excluded_q": len(exclusion.question_ids),
                "n_excluded_a": len(exclusion.answer_ids), "n_hits_filtered": int(n_hits_filtered),
                "final_k": int(final_k)}


LEGACY_CONTEXT = LeakageContext(MODE_THREAD)


def add_leakage_arguments(parser) -> None:
    """--leakage-filter / --leakage-exclude-linked / --leakage-near-dup-cosine,
    defaulting to LeakageSettings (env / .env). Same flags for B, C, D."""
    import argparse

    s = LeakageSettings()
    parser.add_argument("--leakage-filter", action=argparse.BooleanOptionalAction, default=s.leakage_filter,
                        help="Per-query leakage exclusion (default from LEAKAGE_FILTER, True). "
                             "--no-leakage-filter is for the leakage audit only.")
    parser.add_argument("--leakage-exclude-dup", action=argparse.BooleanOptionalAction,
                        default=s.leakage_exclude_dup,
                        help="Also exclude IS_RELATED_TO Duplicate questions (default from LEAKAGE_EXCLUDE_DUP, "
                             "False: mode 'thread'). A sensitivity mode; implied by linked / near-dup exclusion.")
    parser.add_argument("--leakage-exclude-linked", action=argparse.BooleanOptionalAction,
                        default=s.leakage_exclude_linked,
                        help="Also exclude IS_RELATED_TO Linked questions (default from LEAKAGE_EXCLUDE_LINKED, False); "
                             "implies Duplicate exclusion.")
    parser.add_argument("--leakage-near-dup-cosine", type=lambda v: LeakageSettings._none_strings(v),
                        default=s.leakage_near_dup_cosine,
                        help="Near-duplicate question threshold, or 'none' (default from LEAKAGE_NEAR_DUP_COSINE, none); "
                             "implies Duplicate exclusion.")
    parser.add_argument("--leakage-mode", default=None,
                        help="Explicit mode string, overriding the flags above -- 'thread' (the official mode, "
                             "Amendment 15), 'dup+neardup0.95' (Amendment 8) etc. for a sensitivity run.")


def leakage_mode_from_args(args) -> str:
    if getattr(args, "leakage_mode", None):
        return args.leakage_mode
    near = args.leakage_near_dup_cosine
    return leakage_mode(args.leakage_filter, args.leakage_exclude_linked, None if near is None else float(near),
                        exclude_dup=args.leakage_exclude_dup)
