"""
14_apply_logscore_trust.py
==========================
Applies Amendment 6 (docs/DECISION_C_SCORING.md): the upvote score inside
trustScore is log-normalised against the P99 of its node type instead of
global min-max. Answer and Question nodes are both updated.

    norm_log(score; P99) = clip(log1p(max(score, 0)) / log1p(P99), 0, 1)
    trust_Q = 0.7 * norm_log(score_Q; P99_Q) + 0.3 * minmax(log1p(views))
    trust_A = 0.5 * norm_log(score_A; P99_A) + 0.3 * is_accepted + 0.2 * w_auth

The formulas are imported from trust_formulas.py, the same module the KG
build uses. Inputs are read from Neo4j: node `score`, `isAccepted`,
`viewCount`, and the AUTHOR_TRUST edge weight (w_auth; 0 when the answer has
no such edge). Only `trustScore` and `trustScoreVersion` are written; edge
weights (HAS_ANSWER etc.) and `authorReputation` are never touched.

MODES (exactly one; the default writes nothing)
------------------------------------------------------------
--dry-run          Default. Reads, computes in memory, reports P99, old vs
                   new distribution and how many nodes would change.
--apply            Checks the graph state and free disk space, backs up the
                   current trustScore + trustScoreVersion of every node to a
                   timestamped parquet, writes the new values with
                   trustScoreVersion = "v3-logscore", then reads back and
                   requires max |diff| = 0.
--restore          Writes back the values of the backup given with --backup,
                   then requires max |diff| = 0 against that backup.
--verify-applied   Read-only. Recomputes and compares with Neo4j.

New values depend only on score / accepted / views / w_auth, never on the
old trustScore, so repeating --apply gives the same result. If an earlier
--apply was interrupted, some nodes are already v3: the script then refuses
to take a new backup (it would mix old and new values) and asks for
--backup <the backup written by the interrupted run>.

Recommended before --apply: a database dump from Neo4j Desktop as a second
backup.

USAGE (from 01_data_cleaning/)
------------------------------------------------------------
    python 14_apply_logscore_trust.py
    python 14_apply_logscore_trust.py --apply
    python 14_apply_logscore_trust.py --verify-applied
    python 14_apply_logscore_trust.py --restore --backup _kg_workspace/<backup>.parquet
"""

import argparse
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trust_formulas as tf  # noqa: E402

load_dotenv()

LOG_DIR = Path("logs")
VERSION = tf.TRUST_SCORE_VERSION
# State the graph must be in before the first --apply (Amendment 5 applied
# to Answer nodes; Question nodes were never tagged).
PREVIOUS_VERSION = {"Answer": "v2-reputation", "Question": None}
BACKUP_PREFIX = "trust_backup_pre_amendment6"
BATCH_SIZE = 10_000
MIN_FREE_GB = 3.0

READ_ANSWERS = """
MATCH (a:Answer)
OPTIONAL MATCH (a)-[r:AUTHOR_TRUST]->(:User)
WITH a, max(r.weight) AS w
RETURN a.id AS id, a.score AS score, a.isAccepted AS is_accepted, w AS author_weight,
       a.trustScore AS trust, a.trustScoreVersion AS version
"""
READ_QUESTIONS = """
MATCH (q:Question)
RETURN q.id AS id, q.score AS score, q.viewCount AS views,
       q.trustScore AS trust, q.trustScoreVersion AS version
"""
READ = {"Answer": READ_ANSWERS, "Question": READ_QUESTIONS}
COLUMNS = {
    "Answer": ["id", "score", "is_accepted", "author_weight", "trust", "version"],
    "Question": ["id", "score", "views", "trust", "version"],
}
# A null trustScoreVersion in $rows removes the property (restore of nodes
# that had none before).
WRITE = {
    "Answer": "UNWIND $rows AS row MATCH (n:Answer {id: row.id}) "
              "SET n.trustScore = row.trust, n.trustScoreVersion = row.version",
    "Question": "UNWIND $rows AS row MATCH (n:Question {id: row.id}) "
                "SET n.trustScore = row.trust, n.trustScoreVersion = row.version",
}


class StopApply(RuntimeError):
    """A safety check failed; nothing further is written."""


# ---------------------------------------------------------------------
# Pure logic -- tested in llm/c_graphrag/tests/test_logscore_trust.py
# ---------------------------------------------------------------------

def compute_new_trust(frames: dict) -> dict:
    """Adds `trust_new` to each frame in place; returns the P99 used per label."""
    params = {"quantile_method": tf.QUANTILE_METHOD}
    if "Answer" in frames:
        a = frames["Answer"]
        if a["is_accepted"].isna().any():
            raise StopApply("Answer nodes with a missing isAccepted exist; cannot compute trust_A.")
        params["p99_answer"] = tf.compute_p99(a["score"])
        a["trust_new"] = tf.trust_answer(a["score"], a["is_accepted"], a["author_weight"], params["p99_answer"])
    if "Question" in frames:
        q = frames["Question"]
        params["p99_question"] = tf.compute_p99(q["score"])
        q["trust_new"] = tf.trust_question(q["score"], q["views"], params["p99_question"])
    return params


def describe(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "min": float(v.min()), "median": float(np.median(v)),
            "p90": float(np.quantile(v, 0.9)), "max": float(v.max()),
            "n_zero": int((v == 0.0).sum()), "n_one": int((v == 1.0).sum())}


def max_abs_diff(stored, expected) -> float:
    """Largest |stored - expected|; a value missing on one side only counts as inf."""
    s = np.asarray(stored, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if s.size == 0:
        return 0.0
    diff = np.abs(s - e)
    diff[np.isnan(s) & np.isnan(e)] = 0.0
    diff[np.isnan(diff)] = np.inf
    return float(diff.max())


def version_counts(frame: pd.DataFrame) -> dict:
    return {str(k): int(v) for k, v in frame["version"].fillna("(none)").value_counts().items()}


def build_backup(frames: dict) -> pd.DataFrame:
    """label, id, trust, version of every node, as currently stored."""
    return pd.concat(
        [f[["id", "trust", "version"]].assign(label=label) for label, f in frames.items()],
        ignore_index=True,
    )[["label", "id", "trust", "version"]]


def _rows(frame: pd.DataFrame, trust_col: str, version) -> list[dict]:
    """Batch rows with native Python types. `version`: a column name or a constant."""
    versions = frame[version] if version in frame.columns else [version] * len(frame)
    return [{"id": int(i), "trust": None if pd.isna(t) else float(t),
             "version": v if isinstance(v, str) else None}
            for i, t, v in zip(frame["id"], frame[trust_col], versions)]


# ---------------------------------------------------------------------
# Neo4j access
# ---------------------------------------------------------------------

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{ts}_apply_logscore_trust.log"
    logger = logging.getLogger("apply_logscore_trust")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger, log_path


def connect_neo4j(logger):
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
    user = os.getenv("NEO4J_USER", "neo4j").strip()
    password = (os.getenv("NEO4J_PASSWORD") or "").strip().strip('"').strip("'")
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip()
    if not password:
        raise ValueError("NEO4J_PASSWORD not found in .env (project root)")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    logger.info(f"Connected to Neo4j: {uri} | database: {database}")
    return driver, database


def read_nodes(driver, database, label: str) -> pd.DataFrame:
    columns = COLUMNS[label]
    data = {c: [] for c in columns}
    with driver.session(database=database) as s:
        for rec in s.run(READ[label]):
            for c in columns:
                data[c].append(rec[c])
    return pd.DataFrame(data, columns=columns)


def read_frames(driver, database, labels, logger) -> dict:
    frames = {}
    for label in labels:
        frames[label] = read_nodes(driver, database, label)
        logger.info(f"{label}: {len(frames[label]):,} nodes read | trustScoreVersion: "
                    f"{version_counts(frames[label])}")
    return frames


def write_batched(driver, database, label: str, frame: pd.DataFrame, trust_col: str, version,
                  logger, tag: str, batch_size: int = BATCH_SIZE):
    total, written, t0 = len(frame), 0, time.time()
    with driver.session(database=database) as s:
        for i in range(0, total, batch_size):
            batch = _rows(frame.iloc[i:i + batch_size], trust_col, version)
            s.execute_write(lambda tx, b=batch: tx.run(WRITE[label], rows=b).consume())
            written += len(batch)
            if (i // batch_size) % 10 == 0 or written == total:
                rate = written / max(time.time() - t0, 1e-9)
                logger.info(f"  [{tag} {label}] {written:,}/{total:,} ({rate:,.0f}/s)")


# ---------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------

def report(frames: dict, params: dict, logger) -> None:
    logger.info(f"P99 score -- Answer: {params.get('p99_answer')} | Question: {params.get('p99_question')} "
                f"| quantile method: numpy '{params['quantile_method']}' at {tf.P99_QUANTILE}")
    for label, f in frames.items():
        for name, col in (("old", "trust"), ("new", "trust_new")):
            d = describe(f[col])
            logger.info(f"{label:<8} {name}: min {d['min']:.4f} | median {d['median']:.4f} | p90 {d['p90']:.4f} "
                        f"| max {d['max']:.4f} | ==0: {d['n_zero']:,} | ==1: {d['n_one']:,}")
        changed = int((f["trust"].astype(float).to_numpy() != f["trust_new"].to_numpy()).sum())
        logger.info(f"{label:<8} nodes whose trustScore would change: {changed:,} of {len(f):,}")


def check_state_for_apply(frames: dict, backup_given: bool) -> None:
    """First apply: every node must carry its expected previous version.
    Resume (some nodes already v3): only allowed with the existing backup."""
    n_v3 = sum(int((f["version"] == VERSION).sum()) for f in frames.values())
    if n_v3 and not backup_given:
        raise StopApply(f"{n_v3:,} nodes already carry {VERSION} (earlier --apply). A new backup would mix old "
                        f"and new values. Re-run with --backup <backup of that run> to resume.")
    if backup_given:
        return
    for label, f in frames.items():
        want = PREVIOUS_VERSION[label]
        ok = f["version"].isna() if want is None else (f["version"] == want)
        if not bool(ok.all()):
            raise StopApply(f"{label}: expected trustScoreVersion {want!r} on every node before the first apply, "
                            f"found {version_counts(f)}.")


def run_apply(args, driver, database, frames: dict, logger) -> int:
    backup_dir = Path(args.kg_workspace_dir)
    check_state_for_apply(frames, backup_given=bool(args.backup))

    free_gb = shutil.disk_usage(backup_dir).free / 1e9
    logger.info(f"Free disk space: {free_gb:.1f} GB (minimum {args.min_free_gb} GB)")
    if free_gb < args.min_free_gb:
        raise StopApply("Not enough free disk space for the backup and the Neo4j transaction logs.")

    if args.backup:
        backup_path = Path(args.backup)
        if not backup_path.exists():
            raise StopApply(f"Backup not found: {backup_path}")
        n_backup = len(pd.read_parquet(backup_path, columns=["id"]))
        if n_backup != sum(len(f) for f in frames.values()):
            raise StopApply(f"Backup has {n_backup:,} rows, the graph has "
                            f"{sum(len(f) for f in frames.values()):,} nodes for the selected labels.")
        logger.info(f"Resuming with the existing backup (not rewritten): {backup_path}")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = backup_dir / f"{BACKUP_PREFIX}_{stamp}.parquet"
        backup = build_backup(frames)
        backup.to_parquet(backup_path, index=False)
        if len(pd.read_parquet(backup_path, columns=["id"])) != len(backup):
            raise StopApply(f"Backup could not be read back completely: {backup_path}")
        logger.info(f"Backup written: {backup_path} ({len(backup):,} rows)")

    for label, f in frames.items():
        write_batched(driver, database, label, f, "trust_new", VERSION, logger, "apply")

    ok = True
    for label, f in frames.items():
        after = read_nodes(driver, database, label).merge(f[["id", "trust_new"]], on="id", how="outer")
        diff = max_abs_diff(after["trust"], after["trust_new"])
        n_ver = int((after["version"] == VERSION).sum())
        label_ok = diff == 0.0 and n_ver == len(f) == len(after)
        ok = ok and label_ok
        logger.info(f"Verify {label}: max |diff| = {diff:.3e}; {n_ver:,}/{len(f):,} tagged {VERSION} -> "
                    f"{'OK' if label_ok else 'FAILED'}")
    if not ok:
        return 1
    done = datetime.now().astimezone()
    logger.info(f"APPLIED_AT (local): {done.isoformat(timespec='seconds')}")
    logger.info(f"APPLIED_AT (UTC)  : {done.astimezone(timezone.utc).isoformat(timespec='seconds')}")
    logger.info("Paste one of the two lines above (same instant) into the Amendment 6 Record and into "
                "TRUST_V3_APPLIED_AT in llm/c_graphrag/stage2_report.py.")
    logger.info(f"Backup path for the Record: {backup_path}")
    return 0


def run_restore(args, driver, database, labels, logger) -> int:
    if not args.backup:
        raise StopApply("--restore needs --backup <path to the backup parquet>.")
    backup_path = Path(args.backup)
    if not backup_path.exists():
        raise StopApply(f"Backup not found: {backup_path}")
    backup = pd.read_parquet(backup_path)
    logger.info(f"Backup: {backup_path} ({len(backup):,} rows)")
    ok = True
    for label in labels:
        part = backup[backup["label"] == label]
        if part.empty:
            logger.info(f"{label}: not in the backup, skipped.")
            continue
        write_batched(driver, database, label, part, "trust", "version", logger, "restore")
        after = read_nodes(driver, database, label).merge(
            part[["id", "trust", "version"]].rename(columns={"trust": "trust_bk", "version": "version_bk"}),
            on="id", how="outer")
        diff = max_abs_diff(after["trust"], after["trust_bk"])
        same_version = int((after["version"].fillna("") == after["version_bk"].fillna("")).sum())
        label_ok = diff == 0.0 and same_version == len(part) == len(after)
        ok = ok and label_ok
        logger.info(f"Verify {label}: max |diff| vs backup = {diff:.3e}; trustScoreVersion restored on "
                    f"{same_version:,}/{len(part):,} -> {'OK' if label_ok else 'FAILED'}")
    return 0 if ok else 1


def run_verify(frames: dict, logger) -> int:
    ok = True
    for label, f in frames.items():
        diff = max_abs_diff(f["trust"], f["trust_new"])
        n_diff = int((f["trust"].astype(float) != f["trust_new"]).sum())
        n_ver = int((f["version"] == VERSION).sum())
        label_ok = diff == 0.0 and n_ver == len(f)
        ok = ok and label_ok
        logger.info(f"Verify-applied {label}: max |diff| = {diff:.3e}; nodes differing: {n_diff:,}; "
                    f"trustScoreVersion: {version_counts(f)} -> {'OK' if label_ok else 'NOT APPLIED / DIFFERENT'}")
    return 0 if ok else 1


def run(args, driver, database, logger) -> int:
    labels = ["Answer"] if args.only_answers else ["Answer", "Question"]
    mode = "apply" if args.apply else "restore" if args.restore else \
        "verify-applied" if args.verify_applied else "dry-run"
    logger.info(f"Mode: {mode} | labels: {labels} | target version: {VERSION}")
    try:
        if args.restore:
            return run_restore(args, driver, database, labels, logger)
        frames = read_frames(driver, database, labels, logger)
        params = compute_new_trust(frames)
        if args.verify_applied:
            logger.info(f"P99 score -- Answer: {params.get('p99_answer')} | Question: {params.get('p99_question')} "
                        f"| quantile method: numpy '{params['quantile_method']}'")
            return run_verify(frames, logger)
        report(frames, params, logger)
        if not args.apply:
            logger.info("Dry-run: nothing was written. Use --apply to write.")
            return 0
        return run_apply(args, driver, database, frames, logger)
    except StopApply as e:
        logger.info(f"[STOP] {e}")
        return 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Amendment 6: log-normalised upvote score in trustScore.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="Default: read and report only.")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--restore", action="store_true")
    g.add_argument("--verify-applied", action="store_true")
    p.add_argument("--backup", default=None,
                   help="Backup parquet: required for --restore; with --apply, resumes an interrupted apply.")
    p.add_argument("--only-answers", action="store_true",
                   help="Leave Question nodes untouched (trust_Q is not used in Condition C ranking).")
    p.add_argument("--kg-workspace-dir", default="_kg_workspace")
    p.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logger, log_path = setup_logging()
    logger.info(f"Log: {log_path}")
    driver, database = connect_neo4j(logger)
    try:
        return run(args, driver, database, logger)
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
