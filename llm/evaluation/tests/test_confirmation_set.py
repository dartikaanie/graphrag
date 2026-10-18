"""Amendment 11 confirmation set: refused until the judge is frozen and all nine official test runs exist."""

import json

import pytest

import annotation_tool as at

TEST_IDS = set(range(1, 21))
OFFICIAL = {"status": "selected", "alpha_primary": 0.25, "alpha_ablation": 0.0}
RECORD = """## Amendment 11 (2026-10-19): judge-v2.2 ...

### Record (fill in)
- Judge version frozen: {value}  commit: {commit}
- Date frozen: [YYYY-MM-DD]
"""
CELLS = [("A", None, None), ("B", False, None), ("B", True, None), ("C", False, 0.25), ("C", True, 0.25),
         ("C", False, 0.0), ("C", True, 0.0), ("D", False, None), ("D", True, None)]


def _decision(tmp_path, value="[ ]", commit="[ ]"):
    path = tmp_path / "DECISION.md"
    path.write_text(RECORD.format(value=value, commit=commit))
    return path


def _runs(tmp_path, cells=CELLS, split="test", ids=TEST_IDS, role=None):
    files = []
    for n, (cond, grounded, alpha) in enumerate(cells):
        cfg = {"condition": cond, "sample_split": split, "require_grounding": grounded, "alpha": alpha}
        path = tmp_path / f"run{n}.jsonl"
        path.write_text("".join(json.dumps({"question_id": q, "config": cfg}) + "\n" for q in sorted(ids)))
        if role and cond == "C":
            path.with_name(path.name + ".manifest.json").write_text(json.dumps({"official_role": role}))
        files.append(str(path))
    return files


def test_frozen_judge_version_is_read_from_the_record(tmp_path):
    assert at.frozen_judge_version(_decision(tmp_path)) is None
    assert at.frozen_judge_version(_decision(tmp_path, "judge-v2.3", "abc1234")) == "judge-v2.3"
    assert at.frozen_judge_version(at.DECISION_RECORD) is None      # the real record is still empty


def test_refused_while_the_judge_is_not_frozen(tmp_path):
    with pytest.raises(at.AnnotationError, match="not frozen yet"):
        at.check_confirmation_allowed(_runs(tmp_path), _decision(tmp_path), TEST_IDS, OFFICIAL)


def test_nine_official_runs_map_to_their_cells(tmp_path):
    out = at.check_confirmation_allowed(_runs(tmp_path), _decision(tmp_path, "judge-v2.2", "abc"), TEST_IDS, OFFICIAL)
    assert out["judge_version"] == "judge-v2.2"
    assert sorted(out["cells"].values()) == sorted(["A", "B-plain", "B-grounded", "C-primary-plain",
                                                    "C-primary-grounded", "C-ablation-plain", "C-ablation-grounded",
                                                    "D-plain", "D-grounded"])


@pytest.mark.parametrize("kw, needle", [
    ({"cells": CELLS[:-1]}, "official runs missing: ['D-grounded']"),
    ({"split": "dev"}, "not 'test'"),
    ({"ids": set(range(1, 20))}, "frozen test sample"),
    ({"cells": CELLS[:3] + [("C", False, 0.5)] + CELLS[4:]}, "not one of the official runs"),
    ({"cells": CELLS + [("A", None, None)]}, "a second run for A"),
])
def test_refused_without_exactly_the_nine_official_runs(tmp_path, kw, needle):
    with pytest.raises(at.AnnotationError, match=needle.replace("[", r"\[").replace("]", r"\]")):
        at.check_confirmation_allowed(_runs(tmp_path, **kw), _decision(tmp_path, "judge-v2.2", "abc"), TEST_IDS, OFFICIAL)


def test_manifest_role_is_used_for_c(tmp_path):
    files = _runs(tmp_path, role="c_ablation")  # every C manifest says ablation -> duplicates
    with pytest.raises(at.AnnotationError, match="a second run for C-ablation"):
        at.check_confirmation_allowed(files, _decision(tmp_path, "judge-v2.2", "abc"), TEST_IDS, OFFICIAL)
