"""
judge_versions.py
-----------------
Freeze guard for the hallucination judge (docs/DECISION_C_SCORING.md,
Amendment 11, change 3; audit P1-7). No heavy imports.

- judge-v2.2 is the BASE version before calibration; it is not a revision.
- At most MAX_REVISIONS revisions may follow (v2.3, v2.4, v2.5), counted from
  the first run against the human labels; a fourth version after v2.2 is
  refused by the calibration runner.
- Every version must be logged in docs/JUDGE_REVISION_LOG.md before it runs
  on calibration data (date, commit, reason, kappa before/after).
- Official runs (test sample) and stage 2 accept only OFFICIAL_ALLOWED, which
  stays empty until the version is frozen; the frozen version must also be
  the one in Amendment 11's Record.

Which rule applies is decided from the run files: a registered calibration
run -> calibration; a test-sample run -> official; anything else (dev) ->
development (calibration or official versions).
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REVISION_LOG = REPO_ROOT / "docs" / "JUDGE_REVISION_LOG.md"

BASE_VERSION = "judge-v2.2"
MAX_REVISIONS = 3
CALIBRATION_ALLOWED = ("judge-v2.2", "judge-v2.3", "judge-v2.4", "judge-v2.5")  # base + at most 3 revisions
# Amendment 14 (approved): official runs use the frozen panel without a human-agreement gate. Set in the commit
# that approves Amendment 14.
OFFICIAL_ALLOWED: tuple = ("judge-v2.3",)
AMENDMENT14_APPROVED: bool = True
AMENDMENT14_COMMIT = "approve amendments 14-17 and unlock official judging"
OFFICIAL_PANEL = "panel-v2"

PURPOSE_CALIBRATION, PURPOSE_OFFICIAL, PURPOSE_DEVELOPMENT = "calibration", "official", "development"


class JudgeVersionError(RuntimeError):
    pass


def logged_versions(log_path: Path = REVISION_LOG) -> list[str]:
    """Versions in the revision log table, in order ("| judge-v2.x | ..." rows)."""
    if not Path(log_path).exists():
        return []
    out = []
    for line in Path(log_path).read_text().splitlines():
        m = re.match(r"\|\s*(judge-v\d+(?:\.\d+)*)\s*\|", line)
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def revisions_after_base(versions: list[str]) -> list[str]:
    if BASE_VERSION not in versions:
        return versions
    return versions[versions.index(BASE_VERSION) + 1:]


def run_purpose(run_files: list[str], calibration_paths: set[str] | None = None) -> str:
    """calibration if any run file is a registered calibration run, official if any is a test-sample run,
    else development."""
    if calibration_paths is None:
        from _judge_common import load_calibration_registry
        calibration_paths = {str((REPO_ROOT / r["output_path"]).resolve())
                             for r in load_calibration_registry().get("runs", [])}
    if any(str(Path(f).resolve()) in calibration_paths for f in run_files):
        return PURPOSE_CALIBRATION
    from llm import archive_paths  # pre-official archive (2026-10-19): open wherever the file now is
    for f in run_files:
        with open(archive_paths.resolve(f)) as fh:
            for line in fh:
                if line.strip():
                    config = json.loads(line).get("config")
                    # A test-sample run does not store sample_split (it is only written when it differs from the
                    # default "test", so earlier config hashes stay unchanged): an ABSENT key in a generation
                    # config means "test". A record without any config (judge outputs, sweeps) is not a run file.
                    if isinstance(config, dict) and (config.get("sample_split") or "test") == "test":
                        return PURPOSE_OFFICIAL
                    break
    return PURPOSE_DEVELOPMENT


def check_version(version: str, purpose: str, log_path: Path = REVISION_LOG,
                  official_allowed: tuple | None = None, frozen_in_record: str | None = None) -> None:
    """Raises JudgeVersionError when `version` may not run for `purpose`."""
    official_allowed = OFFICIAL_ALLOWED if official_allowed is None else official_allowed
    if purpose == PURPOSE_OFFICIAL:
        if version not in official_allowed:
            raise JudgeVersionError(f"{version} is not an official judge version (OFFICIAL_ALLOWED = "
                                    f"{list(official_allowed)}); official runs need the frozen judge (Amendment 11)")
        if frozen_in_record is not None and frozen_in_record != version:
            raise JudgeVersionError(f"{version} differs from the version frozen in Amendment 11's Record "
                                    f"({frozen_in_record})")
        return
    if version in official_allowed and purpose == PURPOSE_DEVELOPMENT:
        return
    if version not in CALIBRATION_ALLOWED:
        raise JudgeVersionError(f"{version} is not in CALIBRATION_ALLOWED {list(CALIBRATION_ALLOWED)}: only "
                                f"{BASE_VERSION} and at most {MAX_REVISIONS} revisions after it may run")
    versions = logged_versions(log_path)
    if version not in versions:
        raise JudgeVersionError(f"{version} is not logged in {Path(log_path).name}: log it (date, commit, reason) "
                                f"before running it")
    revisions = revisions_after_base(versions)
    if len(revisions) > MAX_REVISIONS or (version in revisions and revisions.index(version) >= MAX_REVISIONS):
        raise JudgeVersionError(f"{len(revisions)} revisions after {BASE_VERSION} are logged ({revisions}); "
                                f"at most {MAX_REVISIONS} are allowed (Amendment 11)")


def check_run_allowed(version: str, run_files: list[str], log=print, official: bool = False,
                      panel_id: str | None = None) -> str:
    """The runner's guard: returns the purpose, or raises JudgeVersionError. `official` is the runner's explicit
    --official flag (Amendment 14): required for test-sample files, refused for any other files."""
    purpose = run_purpose(run_files)
    frozen = None
    if official and purpose != PURPOSE_OFFICIAL:
        raise JudgeVersionError(f"--official given, but the run files are a {purpose} run, not test-sample runs")
    if purpose == PURPOSE_OFFICIAL:
        if panel_id is not None:  # the panel runner: Amendment 14 path
            if not official:
                raise JudgeVersionError("test-sample run files need the explicit --official flag (Amendment 14)")
            if not AMENDMENT14_APPROVED:
                raise JudgeVersionError("Amendment 14 is not approved yet (judge_versions.AMENDMENT14_APPROVED); "
                                        "official judging is refused")
            if panel_id != OFFICIAL_PANEL:
                raise JudgeVersionError(f"official runs use {OFFICIAL_PANEL}, not {panel_id}")
        else:
            from annotation_tool import frozen_judge_version
            frozen = frozen_judge_version()
    check_version(version, purpose, frozen_in_record=frozen)
    log(f"[judge-version] {version} allowed for a {purpose} run")
    return purpose
