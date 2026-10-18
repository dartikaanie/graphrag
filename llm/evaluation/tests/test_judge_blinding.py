"""blind-v3 (judge_blinding.py, Amendment 11): meta phrases normalised, nothing else."""

import judge_blinding as jb
from judge_prompt_v1 import BLINDING_VERSION as V2_VERSION
from judge_prompt_v1 import blind_candidate

M = jb.NEUTRAL_MARKER


def test_version_and_v2_untouched():
    assert jb.BLINDING_VERSION == "blind-v3" and V2_VERSION == "blind-v2"
    assert blind_candidate("Use X [SO-1]. The retrieved sources do not cover Y.") == \
        "Use X . The retrieved sources do not cover Y."  # blind-v2 keeps prose as before


def test_negative_coverage_becomes_the_neutral_marker_and_keeps_the_rest():
    assert jb.blind_candidate_v3("The retrieved sources do not cover the Gradle error. Use `clean` [SO-123].") == \
        f"{M} the Gradle error. Use `clean`."
    assert jb.blind_candidate_v3("Unfortunately, the context provided does not specifically address Python 2.") == \
        f"{M} Python 2."
    assert jb.blind_candidate_v3("The sources, however, do not explicitly provide code.") == f"{M} code."
    out = jb.blind_candidate_v3("The retrieved sources do not cover this, but `pip install -U x` fixes it.")
    assert out == f"{M} this, but `pip install -U x` fixes it."  # the claim after the meta phrase survives


def test_attributions_are_removed():
    assert jb.blind_candidate_v3("Based on the provided context, call `foo()`.") == "Call `foo()`."
    assert jb.blind_candidate_v3("According to the retrieved sources, the flag is `-x`.") == "The flag is `-x`."
    assert jb.blind_candidate_v3("Here are some steps based on the provided context: run it.") == \
        "Here are some steps: run it."
    assert jb.blind_candidate_v3("This is not covered in the retrieved sources, but it works.") == \
        "This is not covered, but it works."
    assert jb.blind_candidate_v3("More details might be needed beyond the context provided.") == \
        "More details might be needed."


def test_ordinary_prose_is_not_touched():
    for text in ("In the context of Android 12, use WorkManager.",
                 "Based on the code snippet you provided, the bug is in line 3.",
                 "Download the source tarball and build it.",
                 "Set the MP4 as the source in your `<video>` tag."):
        assert jb.blind_candidate_v3(text) == blind_candidate(text) == text


def test_condition_a_answer_changes_only_by_citation_removal():
    a = ("I don't have enough information to be certain, but in the context of Django 4 you can use "
         "`QuerySet.bulk_update()`. See the docs [SO-42].")
    assert jb.blind_candidate_v3(a) == blind_candidate(a)
    assert "[SO-42]" not in jb.blind_candidate_v3(a)


def test_meta_phrase_counts():
    counts = jb.meta_phrase_counts(["The retrieved sources do not cover X.", "the retrieved sources do not cover Y",
                                    "Based on the provided context, Z."])
    assert counts[("sources_do_not_cover", "the retrieved sources do not cover")] == 2
    assert counts[("attribution_prefix", "based on the provided context,")] == 1
