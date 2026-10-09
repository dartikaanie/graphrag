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
SAMPLE_SPLITS = ("test", "dev")


def sample_questions_split(df: pd.DataFrame, n: int, seed: int, split: str = DEFAULT_SAMPLE_SPLIT,
                           dev_offset: int = DEFAULT_DEV_OFFSET) -> pd.DataFrame:
    if split not in SAMPLE_SPLITS:
        raise ValueError(f"Unknown sample_split '{split}' -- must be one of {SAMPLE_SPLITS}")
    df_permuted = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    start = 0 if split == "test" else dev_offset - 1
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
