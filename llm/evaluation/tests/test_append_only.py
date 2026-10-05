"""Step 5 (docs/Agent prompt grounding factor ui.md): "outputs, manifests,
and logs are append-only or written to new files. A re-run creates a new
file or appends; it never truncates existing data." This test proves
that guarantee directly -- by writing, then writing again, and checking
the first write's bytes are still there as a PREFIX of the final file --
rather than just inspecting source code for `open(..., "a")` calls.
"""

import json

from llm.manifest import write_manifest
from llm_judge_hallucination_v1 import write_record  # noqa: E402 (sys.path set up by conftest)


def test_judge_write_record_never_truncates_existing_lines(tmp_path):
    output_path = tmp_path / "out.jsonl"
    write_record(output_path, {"question_id": 1})
    first_write_content = output_path.read_text()
    assert first_write_content == '{"question_id": 1}\n'

    write_record(output_path, {"question_id": 2})
    second_write_content = output_path.read_text()

    # The first write's exact bytes are still there, as a PREFIX --
    # never rewritten/truncated, only appended to.
    assert second_write_content.startswith(first_write_content)
    assert second_write_content == first_write_content + '{"question_id": 2}\n'


def test_judge_write_record_survives_many_appends_in_order(tmp_path):
    output_path = tmp_path / "out.jsonl"
    for i in range(5):
        write_record(output_path, {"question_id": i})

    lines = output_path.read_text().strip().split("\n")
    assert len(lines) == 5
    for i, line in enumerate(lines):
        assert json.loads(line)["question_id"] == i


def test_manifest_write_never_modifies_the_data_output_file(tmp_path):
    """write_manifest() creates/overwrites a SEPARATE "<out>.manifest.json"
    sidecar file -- it must never touch the actual data file's content
    (the file the manifest is describing)."""
    output_path = tmp_path / "out.jsonl"
    output_path.write_text('{"question_id": 1}\n{"question_id": 2}\n')
    original_content = output_path.read_text()

    write_manifest(
        output_path, run_label="A", config={}, started_at_utc="t1", finished_at_utc="t2",
        item_counts={"attempted": 2, "succeeded": 2, "failed": 0},
    )
    # Re-run (e.g. resumed run appended a third line, THEN manifest rewritten)
    output_path.write_text(original_content + '{"question_id": 3}\n')
    write_manifest(
        output_path, run_label="A", config={}, started_at_utc="t1", finished_at_utc="t3",
        item_counts={"attempted": 3, "succeeded": 3, "failed": 0},
    )

    final_content = output_path.read_text()
    assert final_content.startswith(original_content)  # never truncated by the manifest writer
    assert final_content == original_content + '{"question_id": 3}\n'


def test_manifest_file_itself_is_a_separate_sidecar_not_mixed_into_the_data_file(tmp_path):
    output_path = tmp_path / "out.jsonl"
    output_path.write_text('{"question_id": 1}\n')
    manifest_path = write_manifest(
        output_path, run_label="A", config={}, started_at_utc="t1", finished_at_utc="t2",
        item_counts={"attempted": 1, "succeeded": 1, "failed": 0},
    )
    assert manifest_path != output_path
    assert manifest_path.name == output_path.name + ".manifest.json"
    # The data file itself was NOT touched by writing the manifest.
    assert output_path.read_text() == '{"question_id": 1}\n'
