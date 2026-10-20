"""docs/FROZEN_PARAMETERS.md and results/frozen_parameters.json match the code (regenerate with
python3 llm/frozen_parameters.py after a deliberate, amendment-backed change)."""

from llm import frozen_parameters as fp


def test_frozen_parameter_files_are_up_to_date():
    assert fp.main(["--check"]) == 0


def test_hash_ignores_source_line_numbers():
    table = fp.build()
    moved = [{**r, "source": r["source"] + "0"} for r in table]
    assert fp.render(table)[1]["table_hash"] == fp.render(moved)[1]["table_hash"]
