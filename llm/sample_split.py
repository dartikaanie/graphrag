"""
sample_split.py
---------------
Test/dev split of the evaluation questions for conditions A and B -- the
SAME rule as c_graphrag.sample_questions_split (docs/DECISION_C_SCORING.md,
"Development set"): one fixed permutation `df.sample(frac=1.0,
random_state=seed)`; test = positions 1..n (iloc[0:n]), dev = positions
dev_offset..dev_offset+n-1, 1-based (default 385 -> iloc[384:384+n]).

`sample_split` is only added to a run's config / filename when it is not
the default "test", so every existing (test) run keeps its config_hash and
output path.
"""

import pandas as pd

DEFAULT_SAMPLE_SPLIT = "test"
DEFAULT_DEV_OFFSET = 385  # 1-based position of the first dev question
DEVLARGE_OFFSET = 435  # 1-based first position of dev-large (Amendment 19): 150 questions after the dev list
DEVLARGE_N = 150
SAMPLE_SPLITS = ("test", "dev", "devlarge")


def sample_questions_split(df: pd.DataFrame, n: int, seed: int, split: str = DEFAULT_SAMPLE_SPLIT,
                           dev_offset: int = DEFAULT_DEV_OFFSET) -> pd.DataFrame:
    if split not in SAMPLE_SPLITS:
        raise ValueError(f"Unknown sample_split '{split}' -- must be one of {SAMPLE_SPLITS}")
    df_permuted = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    start = 0 if split == "test" else (DEVLARGE_OFFSET - 1 if split == "devlarge" else dev_offset - 1)
    return df_permuted.iloc[start:start + n].reset_index(drop=True)


def split_filename_tag(sample_split: str) -> str:
    return "" if sample_split == DEFAULT_SAMPLE_SPLIT else f"_{sample_split}"


def add_split_to_config(config: dict, sample_split: str) -> dict:
    if sample_split != DEFAULT_SAMPLE_SPLIT:
        config["sample_split"] = sample_split
    return config


def add_split_arguments(parser) -> None:
    parser.add_argument("--sample-split", choices=SAMPLE_SPLITS, default=DEFAULT_SAMPLE_SPLIT,
                        help="'test' (default): positions 1..n of the fixed permutation. 'dev': positions "
                             "--dev-offset..+n-1 -- the development set, disjoint from the test sample.")
    parser.add_argument("--dev-offset", type=int, default=DEFAULT_DEV_OFFSET,
                        help="1-based first position of the dev split (default 385 = 0-based row 384).")


# ---------------------------------------------------------------------
# Frozen sample-ID lists (llm/sample_ids/, built by build_sample_ids.py)
# ---------------------------------------------------------------------
# Every condition checks the questions it actually sampled against these
# lists before any retrieval or LLM call: a run on n questions must use
# exactly the first n ids of its split's list. The lists are built ONCE
# from the existing sampler (get_candidate_questions + filter_by_token_limit
# + sample_questions_split) and never regenerated for an official run.

import hashlib
import json
from pathlib import Path

MAX_TEST_N = 384  # the official test sample; positions 1..384
FROZEN_DIR = Path(__file__).resolve().parent / "sample_ids"
FROZEN_FILES = {"test": FROZEN_DIR / "test_ids_v1.json", "dev": FROZEN_DIR / "dev_ids_v1.json",
                "devlarge": FROZEN_DIR / "devlarge_ids_v1.json"}


class SampleMismatchError(RuntimeError):
    """The sampled questions differ from the frozen sample-ID list."""


def ids_sha256(ids) -> str:
    """sha256 of the ids in order, as ASCII "id1,id2,...". """
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def assert_split_bounds(split: str, n_sample: int, dev_offset: int = DEFAULT_DEV_OFFSET) -> None:
    """Runtime guard of the test/dev disjointness: the test split never goes
    past position MAX_TEST_N, and the dev split always starts after it."""
    if dev_offset <= MAX_TEST_N:
        raise SampleMismatchError(f"dev_offset={dev_offset} must be > {MAX_TEST_N} (the test sample is positions "
                                  f"1..{MAX_TEST_N}); otherwise dev and test overlap.")
    if split == "test" and n_sample > MAX_TEST_N:
        raise SampleMismatchError(f"n_sample={n_sample} > {MAX_TEST_N}: the test split is positions 1..{MAX_TEST_N}; "
                                  f"more would reach into the dev set (positions {DEFAULT_DEV_OFFSET}..).")


def load_frozen_ids(split: str, path=None) -> dict:
    path = Path(path) if path else FROZEN_FILES[split]
    if not path.exists():
        raise SampleMismatchError(f"{path} not found -- run llm/sample_ids/build_sample_ids.py first.")
    data = json.loads(path.read_text())
    if ids_sha256(data["ids"]) != data["ids_sha256"]:
        raise SampleMismatchError(f"{path}: ids do not match the stored ids_sha256 -- the file was edited.")
    return data


def assert_frozen_sample(sample_ids, split: str, dev_offset: int = DEFAULT_DEV_OFFSET, path=None) -> dict:
    """Raise SampleMismatchError unless `sample_ids` (in sampled order) are
    exactly the first len(sample_ids) ids of the frozen list for `split`.
    Returns what a manifest records about the check."""
    ids = [int(i) for i in sample_ids]
    frozen = load_frozen_ids(split, path)
    if split == "devlarge":
        dev_offset = DEVLARGE_OFFSET  # fixed position, --dev-offset does not apply
    if split in ("dev", "devlarge") and dev_offset != frozen["dev_offset"]:
        raise SampleMismatchError(f"dev_offset={dev_offset}, but the frozen dev list starts at position "
                                  f"{frozen['dev_offset']}.")
    if len(ids) > len(frozen["ids"]):
        raise SampleMismatchError(f"{len(ids)} {split} questions requested, the frozen list has {len(frozen['ids'])}.")
    expected = frozen["ids"][:len(ids)]
    if ids != expected:
        first = next(i for i, (x, y) in enumerate(zip(ids, expected)) if x != y)
        raise SampleMismatchError(
            f"sampled {split} questions differ from {FROZEN_FILES.get(split, path)} at position {first + 1}: "
            f"got {ids[first]}, frozen {expected[first]} -- the pool, filter, seed or sampler changed.")
    used = Path(path) if path else FROZEN_FILES[split]
    try:
        used = used.resolve().relative_to(FROZEN_DIR.parents[1])
    except ValueError:  # a list outside the repo (tests)
        pass
    return {"split": split, "file": str(used),
            "file_ids_sha256": frozen["ids_sha256"], "n_checked": len(ids),
            "checked_ids_sha256": ids_sha256(ids)}


def verify_sample(sample_ids, split: str, n_sample: int, dev_offset: int = DEFAULT_DEV_OFFSET) -> dict:
    """assert_split_bounds + assert_frozen_sample, the check every
    condition runs right after sampling (CLI and dashboard)."""
    assert_split_bounds(split, n_sample, dev_offset)
    return assert_frozen_sample(sample_ids, split, dev_offset)
