# Prepared change: approving Amendment 14 (NOT applied)

Nothing below has been applied. `llm/evaluation/judge_versions.py` still has `AMENDMENT14_APPROVED = False` and
`OFFICIAL_ALLOWED = ()`, so every official (test-sample) judging run is refused. Apply this change in the one commit
that approves `reports/amendment14_draft.md` (and, if you want, the Amendment 15 / 16 notes in the same commit).

## 1. The change (`llm/evaluation/judge_versions.py`)

```diff
-# Set to (<frozen version>,) in the same commit that fills in Amendment 11's Record -- never before.
-OFFICIAL_ALLOWED: tuple = ()
-# Amendment 14 (draft): official runs use the frozen panel without a human-agreement gate. Both values are set in
-# the commit that approves Amendment 14 (AMENDMENT14_COMMIT = that commit's subject line or id, OFFICIAL_ALLOWED =
-# ("judge-v2.3",)); until then every official judging run is refused.
-AMENDMENT14_APPROVED: bool = False
+# Amendment 14 (approved): official runs use the frozen panel without a human-agreement gate. Set in the commit
+# that approves Amendment 14.
+OFFICIAL_ALLOWED: tuple = ("judge-v2.3",)
+AMENDMENT14_APPROVED: bool = True
+AMENDMENT14_COMMIT = "<subject line or short id of the approval commit>"
```

`AMENDMENT14_COMMIT` is mentioned in the current comment but does not exist yet; it is documentation only (nothing
reads it). A commit cannot contain its own id, so use the subject line.

## 2. Test that must change in the same commit

`llm/evaluation/tests/test_judge_versions.py::test_constants_match_the_runner_and_the_real_log` asserts
`(frozen_judge_version() is None) == (jv.OFFICIAL_ALLOWED == ())`, an Amendment 11 invariant ("the human freeze and
the official list are filled in the same commit"). After approval `OFFICIAL_ALLOWED` is non-empty while the human
freeze stays empty (Amendment 14 removes the human gate), so the line must be replaced by:

```diff
-    from annotation_tool import frozen_judge_version
-    assert (frozen_judge_version() is None) == (jv.OFFICIAL_ALLOWED == ())   # both filled in the same commit
+    assert jv.OFFICIAL_ALLOWED in ((), ("judge-v2.3",))   # Amendment 14: no human freeze involved
+    assert jv.AMENDMENT14_APPROVED == bool(jv.OFFICIAL_ALLOWED)   # flag and list change together
```

Everything else already passes with the new values: `test_amendment14_gate.py` sets the two values itself with
`monkeypatch`.

## 3. What the panel path never asks for (test: `test_panel_path_needs_no_human_annotation.py`)

`check_run_allowed(..., panel_id="panel-v2")` never calls `annotation_tool.frozen_judge_version` (the human freeze of
the old single-judge path), and `judge_panel.main` calls `assert_annotation_frozen` only under `--calibration-set`.
The test poisons both functions and shows that dev and official panel runs pass without touching them.

## 4. Does stage-2 alpha judging (dev set) need this gate?

**No.** Stage-2 files are dev-split files, so `run_purpose()` returns `development`. For that purpose
`check_version()` accepts any version in `CALIBRATION_ALLOWED` that is logged in `docs/JUDGE_REVISION_LOG.md`, and
`judge-v2.3` is (`logged_versions() == ["judge-v2.2", "judge-v2.3"]`). `AMENDMENT14_APPROVED` and `OFFICIAL_ALLOWED`
are consulted only for `PURPOSE_OFFICIAL` (test-sample files). The stage-2 judging therefore **can run before
approval**, with `python3 llm/evaluation/judge_panel.py --panel panel-v2 --judge-version judge-v2.3 ...` (no
`--official`), provided the four stage-2 runs are registered as `dev-stage2` and the DeepInfra budget is set in
`results/run_registry.json` (`panel_v2_registry_gate`), which is a separate budget gate, not the approval flag.
`reports/amendment14_draft.md` section (g) says the same: stage 2 "uses the frozen panel-v2 configuration", which is
not validated against human labels. The pre-registration order still matters for the thesis: approve Amendment 14
(and 16) before stage 2 if you want the rule on the record before any selection result exists.

The secondary metrics (`llm/evaluation/secondary_metrics.py`) use the same gate: a run on official files is refused
until `AMENDMENT14_APPROVED` is `True`.
