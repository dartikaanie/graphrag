"""Amendment 6 (log-normalised upvote score in trustScore): trust_formulas.py,
the build/apply equivalence, 14_apply_logscore_trust.py on a fake Neo4j, the
stage-2 report's v3 cutoff, and the top-5 / stage-1 recheck helpers.
No Neo4j, no network."""

import importlib.util
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "01_data_cleaning"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import trust_formulas as tf  # noqa: E402

import logscore_stage1_recheck as lsr  # noqa: E402
import logscore_top5_check as ltc  # noqa: E402
import stage2_report as s2  # noqa: E402


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, DATA_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


apply_mod = _load("apply_logscore_trust", "14_apply_logscore_trust.py")
build_mod = _load("build_knowledge_graph", "7_build_knowledge_graph.py")

LOGGER = logging.getLogger("test_logscore_trust")
LOGGER.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------
# trust_formulas
# ---------------------------------------------------------------------

def test_constants():
    assert tf.TRUST_SCORE_VERSION == "v3-logscore"
    assert (tf.W_A_SCORE, tf.W_A_ACCEPTED, tf.W_A_AUTHOR) == (0.5, 0.3, 0.2)
    assert (tf.W_Q_SCORE, tf.W_Q_VIEWS) == (0.7, 0.3)
    assert (tf.P99_QUANTILE, tf.QUANTILE_METHOD) == (0.99, "linear")


def test_norm_log_edges():
    out = tf.norm_log([-26, -1, 0, 46, 47, 6095], 46)
    assert out[0] == 0.0 and out[1] == 0.0  # negative -> 0
    assert out[2] == 0.0                    # 0 -> 0
    assert out[3] == 1.0                    # score == P99 -> 1
    assert out[4] == 1.0 and out[5] == 1.0  # above P99 -> clipped to 1
    assert tf.norm_log([5], 46)[0] == pytest.approx(math.log1p(5) / math.log1p(46))


def test_norm_log_monotonic_and_bounded():
    scores = np.arange(-30, 200)
    out = tf.norm_log(scores, 46)
    assert np.all(np.diff(out) >= 0)
    assert out.min() == 0.0 and out.max() == 1.0


def test_norm_log_missing_score_is_zero():
    # Chosen behaviour: a missing score counts as 0, as the KG build's fillna(0) always did.
    out = tf.norm_log([np.nan, None, 3], 46)
    assert out[0] == 0.0 and out[1] == 0.0 and out[2] > 0


def test_norm_log_rejects_bad_p99():
    with pytest.raises(ValueError):
        tf.norm_log([1, 2], 0)


def test_compute_p99_matches_numpy_linear():
    scores = np.arange(1, 102)  # 1..101 -> position 0.99 * 100 = 99 -> value 100 exactly
    assert tf.compute_p99(scores) == 100.0
    mixed = np.array([-26, -3, 0, 0, 1, 1, 2, 5, 46, 6095])
    assert tf.compute_p99(mixed) == float(np.quantile(mixed.astype(float), 0.99, method="linear"))
    assert tf.compute_p99(mixed) == pytest.approx(46 + 0.91 * (6095 - 46))  # hand-computed
    assert tf.compute_p99(pd.Series(mixed)) == pd.Series(mixed).quantile(0.99)  # pandas default


def test_compute_p99_raises_when_not_positive():
    with pytest.raises(ValueError, match="<= 0"):
        tf.compute_p99([0, 0, 0, -5])
    with pytest.raises(ValueError):
        tf.compute_p99([])


def test_trust_answer_hand_computed():
    out = tf.trust_answer([5, -2, 100, 0], [True, False, True, False], [0.5, np.nan, 1.0, 0.25], 46)
    assert out[0] == pytest.approx(0.5 * math.log1p(5) / math.log1p(46) + 0.3 + 0.2 * 0.5)
    assert out[1] == 0.0                       # negative score, not accepted, no AUTHOR_TRUST edge
    assert out[2] == pytest.approx(1.0)        # 0.5 + 0.3 + 0.2
    assert out[3] == pytest.approx(0.2 * 0.25)
    assert out.dtype == np.float64


def test_trust_question_hand_computed():
    # views 0 / 9 / 99 -> log1p = 0 / ln 10 / ln 100 -> min-max = 0 / 0.5 / 1
    out = tf.trust_question([0, 3, 30], [0, 9, 99], 30)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(0.7 * math.log1p(3) / math.log1p(30) + 0.3 * 0.5)
    assert out[2] == pytest.approx(1.0)


def test_trust_stays_in_unit_interval():
    rng = np.random.default_rng(0)
    n = 5000
    score = rng.integers(-40, 12000, n)
    a = tf.trust_answer(score, rng.random(n) < 0.4, rng.random(n), tf.compute_p99(score))
    q = tf.trust_question(score, rng.integers(0, 8_000_000, n), tf.compute_p99(score))
    for out in (a, q):
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_views_term_is_global_minmax_of_log1p():
    views = np.array([5, 100, 7_998_997])
    expected = (np.log1p(views) - np.log1p(5)) / (np.log1p(7_998_997) - np.log1p(5))
    assert np.array_equal(tf.minmax(np.log1p(views)), expected)
    assert np.array_equal(tf.minmax([3, 3, 3]), [0.5, 0.5, 0.5])


def test_author_weight_unchanged_from_amendment5():
    rep = np.array([0.0, 1.0, 150.0, 20_000.0, 1_000_000.0])
    p99 = 20_000.0
    assert np.array_equal(tf.author_weight(rep, p99), (np.log1p(rep) / np.log1p(p99)).clip(0, 1))


# ---------------------------------------------------------------------
# Build path vs apply path
# ---------------------------------------------------------------------

def _fixture_values(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "a_score": rng.integers(-26, 300, n), "a_accepted": rng.random(n) < 0.37,
        "a_weight": np.where(rng.random(n) < 0.05, np.nan, rng.random(n)),
        "q_score": rng.integers(-42, 200, n), "q_views": rng.integers(5, 500_000, n),
    }


def test_build_and_apply_give_identical_arrays():
    v = _fixture_values()
    ids = np.arange(len(v["a_score"]))
    q_built, a_built = build_mod.compute_trust_scores(
        pd.DataFrame({"Id": ids, "Score": v["q_score"], "ViewCount": v["q_views"]}),
        pd.DataFrame({"answer_id": ids, "Score": v["a_score"], "is_accepted": v["a_accepted"],
                      "author_weight": v["a_weight"]}),
        LOGGER)
    frames = {
        "Answer": pd.DataFrame({"id": ids, "score": v["a_score"], "is_accepted": v["a_accepted"],
                                "author_weight": v["a_weight"], "trust": 0.0, "version": "v2-reputation"}),
        "Question": pd.DataFrame({"id": ids, "score": v["q_score"], "views": v["q_views"],
                                  "trust": 0.0, "version": None}),
    }
    apply_mod.compute_new_trust(frames)
    assert np.array_equal(a_built["trustScore"].to_numpy(), frames["Answer"]["trust_new"].to_numpy())
    assert np.array_equal(q_built["trustScore"].to_numpy(), frames["Question"]["trust_new"].to_numpy())
    assert a_built["trustScore"].to_numpy().dtype == np.float64


def test_build_without_reputation_uses_zero_author_weight():
    _, a = build_mod.compute_trust_scores(
        pd.DataFrame({"Id": [1, 2], "Score": [1, 30], "ViewCount": [5, 50]}),
        pd.DataFrame({"answer_id": [1, 2], "Score": [0, 10], "is_accepted": [False, True]}), LOGGER)
    assert a["trustScore"].iloc[0] == 0.0
    assert a["trustScore"].iloc[1] == pytest.approx(0.5 * 1.0 + 0.3)  # P99 of [0, 10] = 9.9 -> clipped


# ---------------------------------------------------------------------
# 14_apply_logscore_trust.py on a fake Neo4j
# ---------------------------------------------------------------------

class _FakeResult:
    def consume(self):
        return None


class _FakeTx:
    def __init__(self, db):
        self.db = db

    def run(self, cypher, rows=None, **params):
        label = next(lb for lb, c in apply_mod.WRITE.items() if c == cypher)
        if self.db.before_write:
            self.db.before_write()
        for row in rows:
            node = self.db.nodes[label][row["id"]]
            node["trust"], node["version"] = row["trust"], row["version"]
        self.db.writes.append((label, len(rows)))
        return _FakeResult()


class _FakeSession:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, **params):
        label = next(lb for lb, c in apply_mod.READ.items() if c == cypher)
        self.db.reads.append(label)
        return [{"id": i, **node} for i, node in self.db.nodes[label].items()]

    def execute_write(self, fn):
        return fn(_FakeTx(self.db))


class FakeNeo4j:
    """In-memory stand-in for the driver: only the two reads and two writes of script 14."""

    def __init__(self, n=40, seed=3):
        v = _fixture_values(n, seed)
        self.nodes = {
            "Answer": {int(i): {"score": int(v["a_score"][i]), "is_accepted": bool(v["a_accepted"][i]),
                                "author_weight": None if np.isnan(v["a_weight"][i]) else float(v["a_weight"][i]),
                                "trust": 0.1 + i / 1000, "version": "v2-reputation"} for i in range(n)},
            "Question": {int(i): {"score": int(v["q_score"][i]), "views": int(v["q_views"][i]),
                                  "trust": 0.2 + i / 1000, "version": None} for i in range(n)},
        }
        self.writes, self.reads, self.before_write = [], [], None

    def session(self, database=None):
        return _FakeSession(self)

    def snapshot(self):
        return {lb: {i: (n["trust"], n["version"]) for i, n in nodes.items()} for lb, nodes in self.nodes.items()}


def _args(tmp_path, *flags):
    return apply_mod.parse_args([*flags, "--kg-workspace-dir", str(tmp_path), "--min-free-gb", "0"])


def _backups(tmp_path):
    return sorted(tmp_path.glob(f"{apply_mod.BACKUP_PREFIX}_*.parquet"))


def _expected(db):
    frames = {"Answer": pd.DataFrame([{"id": i, **n} for i, n in db.nodes["Answer"].items()]),
              "Question": pd.DataFrame([{"id": i, **n} for i, n in db.nodes["Question"].items()])}
    apply_mod.compute_new_trust(frames)
    return {lb: dict(zip(f["id"], f["trust_new"])) for lb, f in frames.items()}


def test_default_mode_is_dry_run_and_writes_nothing(tmp_path):
    db = FakeNeo4j()
    before = db.snapshot()
    assert apply_mod.run(_args(tmp_path), db, "db", LOGGER) == 0
    assert apply_mod.run(_args(tmp_path, "--dry-run"), db, "db", LOGGER) == 0
    assert db.writes == [] and db.snapshot() == before and _backups(tmp_path) == []


def test_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        apply_mod.parse_args(["--apply", "--restore"])


def test_apply_backs_up_before_writing_then_verifies(tmp_path):
    db = FakeNeo4j()
    before, expected = db.snapshot(), _expected(db)
    seen = []
    db.before_write = lambda: seen.append(len(_backups(tmp_path)))
    assert apply_mod.run(_args(tmp_path, "--apply"), db, "db", LOGGER) == 0
    assert seen and all(n == 1 for n in seen)  # the backup existed before every write
    backup = pd.read_parquet(_backups(tmp_path)[0])
    assert list(backup.columns) == ["label", "id", "trust", "version"] and len(backup) == 80
    for label in ("Answer", "Question"):
        part = backup[backup["label"] == label]
        assert {i: t for i, t in zip(part["id"], part["trust"])} == {i: t for i, (t, _) in before[label].items()}
        for i, node in db.nodes[label].items():
            assert node["trust"] == expected[label][i] and node["version"] == "v3-logscore"
    assert set(backup[backup["label"] == "Answer"]["version"]) == {"v2-reputation"}
    assert backup[backup["label"] == "Question"]["version"].isna().all()
    assert apply_mod.run(_args(tmp_path, "--verify-applied"), db, "db", LOGGER) == 0


def test_apply_stops_on_unexpected_graph_state(tmp_path):
    db = FakeNeo4j()
    db.nodes["Answer"][0]["version"] = None  # not the Amendment 5 state
    assert apply_mod.run(_args(tmp_path, "--apply"), db, "db", LOGGER) == 1
    assert db.writes == [] and _backups(tmp_path) == []


def test_apply_refuses_new_backup_after_partial_apply(tmp_path):
    db = FakeNeo4j()
    before = db.snapshot()
    assert apply_mod.run(_args(tmp_path, "--apply"), db, "db", LOGGER) == 0
    backup = _backups(tmp_path)[0]
    n_writes = len(db.writes)
    assert apply_mod.run(_args(tmp_path, "--apply"), db, "db", LOGGER) == 1  # would back up v3 values
    assert len(db.writes) == n_writes and _backups(tmp_path) == [backup]
    assert apply_mod.run(_args(tmp_path, "--apply", "--backup", str(backup)), db, "db", LOGGER) == 0
    assert _backups(tmp_path) == [backup]  # resumed, backup not rewritten
    assert apply_mod.run(_args(tmp_path, "--restore", "--backup", str(backup)), db, "db", LOGGER) == 0
    assert db.snapshot() == before


def test_apply_stops_when_disk_is_low(tmp_path):
    db = FakeNeo4j()
    args = apply_mod.parse_args(["--apply", "--kg-workspace-dir", str(tmp_path), "--min-free-gb", "1e9"])
    assert apply_mod.run(args, db, "db", LOGGER) == 1
    assert db.writes == [] and _backups(tmp_path) == []


def test_restore_writes_back_exactly_the_backup(tmp_path):
    db = FakeNeo4j()
    before = db.snapshot()
    assert apply_mod.run(_args(tmp_path, "--apply"), db, "db", LOGGER) == 0
    assert db.snapshot() != before
    assert apply_mod.run(_args(tmp_path, "--restore"), db, "db", LOGGER) == 1  # no --backup given
    assert apply_mod.run(_args(tmp_path, "--restore", "--backup", str(_backups(tmp_path)[0])), db, "db", LOGGER) == 0
    assert db.snapshot() == before  # trustScore bit-identical, Question version removed again
    assert apply_mod.run(_args(tmp_path, "--verify-applied"), db, "db", LOGGER) == 1


def test_verify_detects_a_single_changed_value(tmp_path):
    db = FakeNeo4j()
    assert apply_mod.run(_args(tmp_path, "--apply"), db, "db", LOGGER) == 0
    n_writes = len(db.writes)
    db.nodes["Answer"][7]["trust"] += 1e-12
    assert apply_mod.run(_args(tmp_path, "--verify-applied"), db, "db", LOGGER) == 1
    assert len(db.writes) == n_writes  # verify never writes
    assert apply_mod.max_abs_diff([0.5, np.nan], [0.5, 0.25]) == np.inf


def test_only_answers_leaves_questions_untouched(tmp_path):
    db = FakeNeo4j()
    before = db.snapshot()
    assert apply_mod.run(_args(tmp_path, "--apply", "--only-answers"), db, "db", LOGGER) == 0
    assert db.snapshot()["Question"] == before["Question"]
    assert {lb for lb, _ in db.writes} == {"Answer"}


# ---------------------------------------------------------------------
# stage2_report: v3 cutoff
# ---------------------------------------------------------------------

QIDS = list(range(1000, 1050))
APPLIED = "2026-10-14T09:00:00+07:00"  # == 2026-10-14T02:00:00+00:00


@pytest.fixture
def _stage1_dev_ids_stub(monkeypatch):
    monkeypatch.setattr(s2, "stage1_dev_ids", lambda: QIDS)


def _run(tmp, alpha, grounded, started, **manifest_extra) -> Path:
    cfg = {"condition": "C", "c_retrieval_version": "v3", "sample_split": "dev", "alpha": alpha,
           "require_grounding": grounded, "expansion_require_answer": True, "leakage": "thread"}
    path = tmp / f"run_a{alpha}_{'g' if grounded else 'p'}.jsonl"
    path.write_text("".join(json.dumps({"question_id": q, "config": cfg}) + "\n" for q in QIDS))
    manifest = {"started_at_utc": started, "graph": {"trustScoreVersion": "v3-logscore"}, **manifest_extra}
    path.with_name(path.name + ".manifest.json").write_text(json.dumps(manifest))
    return path


# fixed stage-2 candidates for these checks, independent of the STAGE2_ALPHAS decision
STAGE2_TEST_ALPHAS = (0.5, 0.25)


def _four_runs(tmp, started, special=None, **manifest_extra):
    """4 stage-2 runs; `special` overrides the start of the (0.5, grounded) run."""
    return [_run(tmp, a, g, special if (special and (a, g) == (0.5, True)) else started, **manifest_extra)
            for a in (0.5, 0.25) for g in (True, False)]


@pytest.mark.usefixtures("_stage1_dev_ids_stub")
def test_stage2_placeholder_fails_closed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(s2, "TRUST_V3_APPLIED_AT", None)  # explicit: not masked by any fixture or a filled-in value
    runs = _four_runs(tmp_path, "2026-12-01T00:00:00+00:00")
    with pytest.raises(s2.Stage2Error, match="TRUST_V3_APPLIED_AT is not set"):
        s2.load_runs(runs, alphas=STAGE2_TEST_ALPHAS)
    monkeypatch.setattr(s2, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["stage2_report.py", "--run-files", *map(str, runs),
                                      "--judge-files", "none.jsonl", "--judge-version", "judge-v2.1"])
    assert s2.main() == 1
    assert "[STOP]" in capsys.readouterr().out and not (tmp_path / "out").exists()


@pytest.mark.usefixtures("_stage1_dev_ids_stub")
@pytest.mark.parametrize("bad, msg", [("2026-10-14T09:00:00", "no UTC offset"), ("yesterday", "not an ISO 8601")])
def test_stage2_rejects_unusable_cutoff(tmp_path, bad, msg):
    with pytest.raises(s2.Stage2Error, match=msg):
        s2.load_runs(_four_runs(tmp_path, "2026-12-01T00:00:00+00:00"), applied_at=bad, alphas=STAGE2_TEST_ALPHAS)


@pytest.mark.usefixtures("_stage1_dev_ids_stub")
def test_stage2_run_before_apply_excluded(tmp_path):
    runs = _four_runs(tmp_path, "2026-10-14T02:00:01+00:00", special="2026-10-14T01:59:59+00:00")
    with pytest.raises(s2.Stage2Error, match="Amendment 6"):
        s2.load_runs(runs, applied_at=APPLIED, alphas=STAGE2_TEST_ALPHAS)


@pytest.mark.usefixtures("_stage1_dev_ids_stub")
def test_stage2_run_after_apply_included(tmp_path, monkeypatch):
    runs = _four_runs(tmp_path, "2026-10-14T02:00:01+00:00")
    assert len(s2.load_runs(runs, applied_at=APPLIED, alphas=STAGE2_TEST_ALPHAS)) == 4
    monkeypatch.setattr(s2, "TRUST_V3_APPLIED_AT", APPLIED)  # the constant is read at call time
    assert len(s2.load_runs(runs, alphas=STAGE2_TEST_ALPHAS)) == 4


@pytest.mark.usefixtures("_stage1_dev_ids_stub")
def test_stage2_same_instant_in_two_timezones(tmp_path):
    # 03:00 UTC is 10:00 in Jakarta: after 09:00+07:00 although "03:00" < "09:00" as text.
    assert len(s2.load_runs(_four_runs(tmp_path, "2026-10-14T03:00:00+00:00"), applied_at=APPLIED, alphas=STAGE2_TEST_ALPHAS)) == 4
    # The same instant written two ways is not "after".
    for same in ("2026-10-14T02:00:00+00:00", "2026-10-14T09:00:00+07:00", "2026-10-14T02:00:00Z"):
        with pytest.raises(s2.Stage2Error, match="Amendment 6"):
            s2.load_runs(_four_runs(tmp_path, "2026-10-14T03:00:00+00:00", special=same), applied_at=APPLIED, alphas=STAGE2_TEST_ALPHAS)
    # 09:30 written with the Jakarta offset is after; a start without an offset is read as UTC.
    assert len(s2.load_runs(_four_runs(tmp_path, "2026-10-14T09:30:00+07:00"), applied_at=APPLIED, alphas=STAGE2_TEST_ALPHAS)) == 4
    with pytest.raises(s2.Stage2Error, match="Amendment 6"):
        s2.load_runs(_four_runs(tmp_path, "2026-10-14T03:00:00+00:00", special="2026-10-14T01:30:00"),
                     applied_at=APPLIED, alphas=STAGE2_TEST_ALPHAS)


@pytest.mark.usefixtures("_stage1_dev_ids_stub")
def test_stage2_trust_version_mismatch_or_missing_excluded(tmp_path):
    after = "2026-10-14T03:00:00+00:00"
    assert len(s2.load_runs(_four_runs(tmp_path, after), applied_at=APPLIED, alphas=STAGE2_TEST_ALPHAS)) == 4
    for bad in ("v2-reputation", "mixed(v2-reputation=5, v3-logscore=7)", None):
        with pytest.raises(s2.Stage2Error, match="manifest records trustScoreVersion"):
            s2.load_runs(_four_runs(tmp_path, after, graph={"trustScoreVersion": bad}), applied_at=APPLIED, alphas=STAGE2_TEST_ALPHAS)
    with pytest.raises(s2.Stage2Error, match="trustScoreVersion None"):
        s2.load_runs(_four_runs(tmp_path, after, graph=None), applied_at=APPLIED, alphas=STAGE2_TEST_ALPHAS)  # manifest without the field


class _VersionSession:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, **params):
        assert "SET" not in cypher and "trustScoreVersion" in cypher  # read-only
        return self.rows


class _VersionDriver:
    def __init__(self, rows):
        self.rows = rows

    def session(self, database=None):
        return _VersionSession(self.rows)


@pytest.mark.usefixtures("_stage1_dev_ids_stub")
def test_run_manifest_records_graph_trust_version(tmp_path):
    import c_graphrag as cg
    from llm.manifest import write_manifest

    def version(rows):
        return cg.fetch_trust_score_version(_VersionDriver(rows), "db")

    assert version([{"version": "v3-logscore", "n": 725795}]) == "v3-logscore"
    assert version([{"version": None, "n": 10}]) is None and version([]) is None
    mixed = version([{"version": "v3-logscore", "n": 7}, {"version": "v2-reputation", "n": 5}])
    assert mixed == "mixed(v2-reputation=5, v3-logscore=7)"

    # What c_graphrag.py / the dashboard write at the end of a run, then what stage 2 reads back.
    for (alpha, grounded), ver in zip([(a, g) for a in (0.5, 0.25) for g in (True, False)],
                                      ["v3-logscore", "v3-logscore", "v3-logscore", mixed]):
        path = _run(tmp_path, alpha, grounded, "unused")
        write_manifest(path, run_label=None, config={}, started_at_utc="2026-10-14T03:00:00+00:00",
                       finished_at_utc="2026-10-14T03:10:00+00:00", item_counts={},
                       graph={"trustScoreVersion": ver})
    manifest = json.loads((tmp_path / "run_a0.5_g.jsonl.manifest.json").read_text())
    assert manifest["graph"] == {"trustScoreVersion": "v3-logscore"}
    runs = [tmp_path / f"run_a{a}_{g}.jsonl" for a in (0.5, 0.25) for g in ("g", "p")]
    with pytest.raises(s2.Stage2Error, match="run_a0.25_p.jsonl: manifest records trustScoreVersion 'mixed"):
        s2.load_runs(runs, applied_at=APPLIED, alphas=STAGE2_TEST_ALPHAS)
    # A manifest written without `graph` (other conditions) has no such key.
    other = tmp_path / "other.jsonl"
    other.write_text("")
    assert "graph" not in json.loads(write_manifest(
        other, run_label=None, config={}, started_at_utc="t", finished_at_utc="t", item_counts={}).read_text())


def test_author_trust_loader_uses_the_shared_w_auth():
    loader = _load("load_author_trust", "9_load_author_trust.py")
    users = pd.DataFrame({"user_id": range(1, 301),
                          "reputation": np.r_[np.nan, -5.0, np.random.default_rng(4).integers(1, 900_000, 298)]})
    loaded, p99 = loader.author_trust_weights(users)
    built = build_mod.compute_author_trust_weight(users, LOGGER)
    rep = users["reputation"].fillna(0).clip(lower=0)
    assert p99 == rep.quantile(0.99)
    old_inline = (np.log1p(rep) / np.log1p(p99)).clip(0, 1).to_numpy()  # formula both scripts had before
    assert np.array_equal(loaded["weight"].to_numpy(), old_inline)
    assert np.array_equal(built["weight"].to_numpy(), old_inline)
    assert np.array_equal(loaded["weight"].to_numpy(), tf.author_weight(rep, p99))
    assert loaded["weight"].iloc[0] == 0.0 and loaded["weight"].iloc[1] == 0.0
    assert users["reputation"].isna().iloc[0]  # input not mutated


def test_build_tags_nodes_with_the_trust_version():
    q, a = build_mod.compute_trust_scores(
        pd.DataFrame({"Id": [1, 2], "Score": [1, 30], "ViewCount": [5, 50]}),
        pd.DataFrame({"answer_id": [1, 2], "Score": [0, 10], "is_accepted": [False, True]}), LOGGER)
    assert set(q["trustScoreVersion"]) == {"v3-logscore"} and set(a["trustScoreVersion"]) == {"v3-logscore"}
    assert "q.trustScoreVersion = row.trustScoreVersion" in build_mod.WRITE_QUESTION_NODES
    assert "a.trustScoreVersion = row.trustScoreVersion" in build_mod.WRITE_ANSWER_NODES
    for cypher in (build_mod.WRITE_QUESTION_NODES, build_mod.WRITE_ANSWER_NODES):
        assert "trustScore = row.trustScore," in cypher


# ---------------------------------------------------------------------
# Top-5 check and stage-1 recheck helpers
# ---------------------------------------------------------------------

class _FakeEnc:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


@pytest.fixture(autouse=True)
def _no_tiktoken_download(monkeypatch):
    import tiktoken
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: _FakeEnc())


class _FakeEmbedModel:
    def encode(self, texts, convert_to_numpy=True, device="cpu", batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        return np.stack([np.array([len(t) % 7 + 1, (len(t) % 3) + 1, 1.0], dtype=np.float32) for t in texts])


def _cand(aid, body="same body text", edge=1.0, accepted=False):
    return {"via_question_id": 10, "via_question_title": "Q", "answer_id": aid, "answer_body": body,
            "answer_trust_score": None, "is_accepted": accepted, "edge_weight": edge, "hop": 1,
            "rel_type": "HAS_ANSWER"}


def _inputs(scores, version="v2-reputation", stored=None):
    return {i + 1: {"score": s, "is_accepted": False, "author_weight": 0.5,
                    "trust": 0.1 + 0.5 * (s + 26) / 6121 if stored is None else stored[i], "version": version}
            for i, s in enumerate(scores)}


Q_EMB = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


def test_resolve_trust_before_apply_uses_stored_v2_and_computed_v3():
    inputs = _inputs([0, 3, 46])
    v2, v3 = ltc.resolve_trust(inputs, 46.0)
    assert v2 == {i: inputs[i]["trust"] for i in inputs}
    assert list(v3.values()) == list(tf.trust_answer([0, 3, 46], [False] * 3, [0.5] * 3, 46.0))


def test_resolve_trust_after_apply_needs_backup_and_exact_v3():
    stored = list(tf.trust_answer([0, 3, 46], [False] * 3, [0.5] * 3, 46.0))
    inputs = _inputs([0, 3, 46], version="v3-logscore", stored=stored)
    with pytest.raises(ValueError, match="--backup"):
        ltc.resolve_trust(inputs, 46.0)
    v2, v3 = ltc.resolve_trust(inputs, 46.0, backup_trust={1: 0.11, 2: 0.12, 3: 0.13})
    assert v2 == {1: 0.11, 2: 0.12, 3: 0.13} and list(v3.values()) == stored
    inputs[2]["trust"] += 1e-9
    with pytest.raises(ValueError, match="!= computed"):
        ltc.resolve_trust(inputs, 46.0, backup_trust={1: 0.11, 2: 0.12, 3: 0.13})


def test_top5_alpha_zero_gives_zero_changes():
    scores = [0, 1, 2, 3, 5, 8, 13, 46]
    graph = [_cand(i + 1, body="x " * (i + 1)) for i in range(len(scores))]
    v2, v3 = ltc.resolve_trust(_inputs(scores), 46.0)
    rows = []
    for alpha in ltc.ALPHAS:
        top_v2 = ltc.rank_with_trust(alpha, graph, [], v2, _FakeEmbedModel(), Q_EMB, None)
        top_v3 = ltc.rank_with_trust(alpha, graph, [], v3, _FakeEmbedModel(), Q_EMB, None)
        rows.append(ltc.compare_alpha(77, alpha, top_v2, top_v3))
    zero = ltc.summarize(rows)["0.0"]
    assert zero["n_set_changed"] == 0 and zero["n_order_changed"] == 0
    assert zero["mean_jaccard"] == 1.0 and zero["mean_items_swapped"] == 0.0
    assert graph[0]["answer_trust_score"] is None  # inputs are not mutated


def test_top5_trust_version_can_change_ranking_and_alpha_zero_is_asserted():
    # v2 (min-max on raw score) barely separates score 2 from score 40; v3 does.
    graph = [_cand(1), _cand(2)]
    v2, v3 = {1: 0.30, 2: 0.29}, {1: 0.10, 2: 0.60}
    top_v2 = ltc.rank_with_trust(1.0, graph, [], v2, _FakeEmbedModel(), Q_EMB, None, top_k=1)
    top_v3 = ltc.rank_with_trust(1.0, graph, [], v3, _FakeEmbedModel(), Q_EMB, None, top_k=1)
    row = ltc.compare_alpha(5, 1.0, top_v2, top_v3)
    assert row["set_changed"] and row["entered_ids"] == [2] and row["jaccard"] == 0.0
    s = ltc.summarize([row], alphas=(1.0,))["1.0"]
    assert s["n_set_changed"] == 1 and s["mean_items_swapped"] == 1.0 and s["example_question_ids"] == [5]
    with pytest.raises(RuntimeError, match="alpha=0"):
        ltc.compare_alpha(5, 0.0, top_v2, top_v3)


def _judgment(qid, aid, text, label="RELEVANT", model="m1", judge_id="primary", version="ctxrel-v2"):
    return {"question_id": qid, "answer_id": aid, "text_hash": f"h:{text}", "judge_id": judge_id,
            "judge_model": model, "prompt_version": version, "label": label}


def test_recheck_cache_key_and_new_item_counts(tmp_path):
    cache = tmp_path / "judged.jsonl"
    cache.write_text("".join(json.dumps(r) + "\n" for r in [
        _judgment(1, 10, "a"),
        _judgment(1, 11, "b", judge_id="secondary"),   # other judge: not a cache hit
        _judgment(1, 12, "c", version="ctxrel-v1"),    # other prompt version: not a cache hit
        _judgment(1, 13, "d", label=None),             # failed call: not a cache hit
        _judgment(2, 10, "a", model="m2"),             # other judge model
    ]))
    judged, models = lsr.load_judged([cache, tmp_path / "missing.jsonl"])
    assert models == {"m1", "m2"} and set(judged) == {(1, 10, "h:a"), (2, 10, "h:a")}
    judged, _ = lsr.load_judged([cache], judge_model="m1")
    assert set(judged) == {(1, 10, "h:a")}

    def ctx(*pairs):
        return [{"answer_id": aid, "chunk_text": text} for aid, text in pairs]

    records = {0.25: [{"question_id": 1, "title": "t", "tags": "<x>", "retrieved_context": ctx((10, "a"), (11, "b"))}],
               0.5: [{"question_id": 1, "title": "t", "tags": "<x>",
                      "retrieved_context": ctx((10, "a"), (11, "b"), (10, "changed text"))}]}
    text_hash = lambda t: f"h:{t}"  # noqa: E731
    assert lsr.new_items_per_alpha(records, set(judged), text_hash) == {0.25: 1, 0.5: 2}
    todo = lsr.rsr.unjudged_items(records, set(judged), text_hash)
    assert [(i["answer_id"], i["text_hash"]) for i in todo] == [(11, "h:b"), (10, "h:changed text")]


def test_recheck_reuses_the_stage1_rule_and_paths():
    from ctxrel_sweep import select_stage1_candidates
    assert lsr.ALPHAS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert lsr.v2_file_for_alpha(0.25, Path("d")) == Path("d/stage1_repfix_alpha0-25.jsonl")
    m = {a: {"pct_questions_with_relevant": p, "mean_relevance_score": s}
         for a, p, s in [(0.0, 48.0, 0.33), (0.25, 48.0, 0.32), (0.5, 50.0, 0.33), (0.75, 46.0, 0.29), (1.0, 36.0, 0.22)]}
    assert select_stage1_candidates(m)["top_two"][0] == 0.5
    deltas = lsr.compare_metrics(m, {a: {**v, "mean_relevance_score": v["mean_relevance_score"] + 0.01}
                                     for a, v in m.items()})
    assert [d["delta_mean_relevance_score"] for d in deltas] == [0.01] * 5
    assert all(d["delta_pct_questions_with_relevant"] == 0.0 for d in deltas)
