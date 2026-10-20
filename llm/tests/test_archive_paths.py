"""The pre-official archive (2026-10-19): the map is complete, moved files
resolve by their original path, frozen records stay unchanged, and the
step-3 screening input and the registry entries resolve to real files."""

import json
import subprocess

import pytest

from llm import archive_paths, run_registry

REPO_ROOT = archive_paths.REPO_ROOT
TAG = "2026-10-19_pre-official"
MAP = REPO_ROOT / "results" / "_archive" / TAG / "archive_map.json"

pytestmark = pytest.mark.skipif(not MAP.exists(), reason="archive map not present")


@pytest.fixture(scope="module")
def amap():
    return json.loads(MAP.read_text())


def test_every_archived_file_exists_and_its_original_is_gone(amap):
    for orig, new in amap["files"].items():
        assert (REPO_ROOT / new).is_file(), new
        # append-only run histories restart at the original path after the archive (new runs); nothing else may
        if "run_history" not in orig:
            assert not (REPO_ROOT / orig).exists(), f"{orig} is still at its original path"
        assert new.split("/")[1] == "_archive" and new.split("/")[2] == TAG and new.endswith("/" + orig)


def test_resolve_and_canonical_round_trip(amap):
    for orig, new in [(o, n) for o, n in amap["files"].items() if not (REPO_ROOT / o).exists()][:50]:
        assert archive_paths.resolve(orig) == REPO_ROOT / new
        assert archive_paths.resolve(REPO_ROOT / orig) == REPO_ROOT / new
        assert archive_paths.canonical(REPO_ROOT / new) == REPO_ROOT / orig
        assert archive_paths.canonical(REPO_ROOT / orig) == REPO_ROOT / orig
        assert archive_paths.is_archived(REPO_ROOT / new) and not archive_paths.is_archived(REPO_ROOT / orig)


def test_directory_moves_and_archived_dirs(amap):
    runs_dir = REPO_ROOT / "backend" / ".runs"
    assert archive_paths.archived(runs_dir) == REPO_ROOT / amap["dirs"]["backend/.runs"]
    c_results = archive_paths.archived_dir(REPO_ROOT / "llm" / "c_graphrag" / "results")
    assert c_results == REPO_ROOT / "results" / "_archive" / TAG / "llm" / "c_graphrag" / "results"
    assert archive_paths.archived_dir(REPO_ROOT / "llm" / "c_graphrag" / "logs") == \
        REPO_ROOT / "logs" / "_archive" / TAG / "llm" / "c_graphrag" / "logs"
    assert archive_paths.archived_dir(REPO_ROOT / "01_data_cleaning" / "logs") is None


def test_unknown_paths_pass_through(tmp_path):
    p = tmp_path / "x.jsonl"
    assert archive_paths.resolve(p) == p
    assert archive_paths.archived(p) is None
    assert archive_paths.canonical(p) == p.resolve()


def test_kept_in_place_inputs_were_not_archived(amap):
    for keep in ("results/leakage/exclusion_sets.parquet", "results/leakage/exclusion_sets.meta.json",
                 "logs/20261013T145605Z_validate_densification.log"):
        assert keep not in amap["files"] and (REPO_ROOT / keep).exists(), keep
    assert not any(o.startswith(("llm/evaluation/annotation/", "llm/evaluation/calibration/", "report/", "reports/",
                                 "docs/")) for o in amap["files"])


def test_calibration_runs_resolve_by_their_frozen_paths():
    """Step-3 screening input: the frozen registry keeps the original paths; each resolves."""
    reg = json.loads((REPO_ROOT / "llm" / "evaluation" / "calibration" / "calibration_runs.json").read_text())
    assert len(reg["runs"]) == 3
    for r in reg["runs"]:
        assert not (REPO_ROOT / r["output_path"]).exists()
        assert archive_paths.resolve(r["output_path"]).is_file(), r["output_path"]


def test_run_metadata_resolves_archived_calibration_runs():
    import sys
    sys.path.insert(0, str(REPO_ROOT / "llm" / "evaluation"))
    from _run_metadata import resolve_run_metadata
    reg = json.loads((REPO_ROOT / "llm" / "evaluation" / "calibration" / "calibration_runs.json").read_text())
    for r in reg["runs"]:
        meta_orig = resolve_run_metadata(str(REPO_ROOT / r["output_path"]))
        meta_arch = resolve_run_metadata(str(archive_paths.resolve(r["output_path"])))
        assert meta_orig == meta_arch and meta_orig["source_file"] == str(REPO_ROOT / r["output_path"])


def test_frozen_records_are_unchanged_since_head():
    """Frozen JSON / annotation records are byte-for-byte as committed."""
    paths = ["llm/evaluation/calibration", "llm/evaluation/annotation/calibration"]
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *paths], cwd=REPO_ROOT)
    if r.returncode not in (0, 1):
        pytest.skip("git not available")
    assert r.returncode == 0, "a frozen calibration/annotation record differs from HEAD"


def test_archived_registry_entries_point_at_real_files():
    reg = run_registry.load()
    archived = [r for r in reg["runs"] if r["status"] == run_registry.STATUS_ARCHIVED]
    assert archived
    for r in archived:
        run_registry.validate_entry(r)
        for path in r["files"].values():
            assert archive_paths.resolve(path).is_file(), (r["run_id"], path)
    assert run_registry.runs(reg=reg) == [r for r in reg["runs"] if r["status"] == run_registry.STATUS_OFFICIAL]


def test_repo_relative_path_resolves_from_a_subfolder(monkeypatch, amap):
    orig = next(o for o in amap["files"] if o.startswith("llm/evaluation/results/"))
    monkeypatch.chdir(REPO_ROOT / "llm" / "evaluation")
    assert archive_paths.resolve(orig) == REPO_ROOT / amap["files"][orig]
    assert archive_paths.resolve("results/" + orig.rsplit("/", 1)[1]) == REPO_ROOT / amap["files"][orig]
