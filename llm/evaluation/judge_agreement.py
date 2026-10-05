"""
judge_agreement.py
=====================================
Step 5 for judge-v1: inter-judge agreement, human-annotation export,
human-vs-judge agreement, and per-run_label summary. Reads the OUTPUT of
llm_judge_hallucination_v1.py (one record per (item, judge_id)), never
re-judges anything itself except where explicitly noted.

TERMINOLOGY (do not conflate): every rate/label this file reports is
judge-v1's **hallucination (reference-based)** metric -- candidate vs.
accepted SO answer + judge's own knowledge, retrieved_context never
shown to the judge. This is a DIFFERENT metric from llm_judge_
hallucination.py's (the old, un-renamed script) **faithfulness
(context-grounded)** -- candidate vs. the retrieved_context the
generator actually saw. The two scripts' outputs are not comparable
label-for-label even though both happen to use similarly-named classes.

Four subcommands (run independently, at different times):
    agreement        -- confusion matrix + Cohen's kappa, primary vs secondary
    export-human-csv -- blinded CSV for manual labeling + a SEPARATE id-mapping file
    human-kappa       -- once human_label is filled in, weighted kappa human-vs-judge
    summary           -- per-run_label hallucination (reference-based) rates, primary judge

CARA PAKAI
----------------------------------------------------------------------------
    cd llm/evaluation
    python3 judge_agreement.py agreement --judge-output results/jv1_*.jsonl
    python3 judge_agreement.py export-human-csv \\
        --run-files ../a_pure_llm/results/*.jsonl ../b_rag/results/*.jsonl ... \\
        --out results/jv1_human_annotation.csv --n 60 --seed 42
    python3 judge_agreement.py human-kappa \\
        --human-csv results/jv1_human_annotation.csv \\
        --mapping logs/jv1_human_export_mapping.jsonl \\
        --judge-output results/jv1_*.jsonl
    python3 judge_agreement.py summary --judge-output results/jv1_*.jsonl --judge-id primary
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _judge_common import html_to_text, load_question_bodies  # noqa: E402
from _run_metadata import RunMetadataNotFoundError, resolve_run_metadata  # noqa: E402
from judge_prompt_v1 import LABELS, blind_candidate  # noqa: E402
from llm_judge_hallucination_v1 import format_tags  # noqa: E402

log = print

ORDINAL_LABELS = ["FAKTUAL", "HALUSINASI_SEBAGIAN", "HALUSINASI_PENUH"]  # excludes ABSTAIN, by spec
ALL_LABELS = list(LABELS)  # FAKTUAL, HALUSINASI_SEBAGIAN, HALUSINASI_PENUH, ABSTAIN


# ---------------------------------------------------------------------------
# Shared loading
# ---------------------------------------------------------------------------

class MixedJudgeVersionError(ValueError):
    """Raised by load_judge_records() -- a judge output file (or set of
    files given together) contains more than one DISTINCT prompt_version
    or blinding_version. This is the chokepoint EVERY reader (summary,
    agreement, disagreements, human-export, human-kappa, the CLI
    subcommands) goes through, so the check only needs to live here
    once. A future judge-v2 writes to its own filename (see
    engine_service.judge_v1_output_paths's docstring) so this should
    only ever fire for a manually-overridden --out path that got reused
    across versions."""


def _assert_single_judge_version(records: list[dict], field: str, source_label: str) -> None:
    values = {r.get(field) for r in records if r.get(field)}
    if len(values) > 1:
        raise MixedJudgeVersionError(
            f"{source_label}: records have {len(values)} different {field} values {sorted(values)} -- "
            f"this file mixes judge outputs from different {field}s and must not be read/aggregated as "
            f"one judge run. Investigate and split or regenerate it before reading."
        )


def load_judge_records(paths: list[str]) -> list[dict]:
    records = []
    for path in paths:
        with open(path) as f:
            path_records = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path_records.append(json.loads(line))
        _assert_single_judge_version(path_records, "prompt_version", path)
        _assert_single_judge_version(path_records, "blinding_version", path)
        records.extend(path_records)
    return records


def index_by_key(records: list[dict]) -> dict:
    """(run_id, question_id, judge_id) -> record. If duplicates exist
    (e.g. a file re-judged after a partial failure), the LAST one in
    file order wins -- matches how an append-only resumable file should
    be read (newest write is authoritative for that key)."""
    index = {}
    for r in records:
        index[(r["run_id"], r["question_id"], r["judge_id"])] = r
    return index


# ---------------------------------------------------------------------------
# agreement -- pure logic (testable without touching sklearn/CLI/files)
# ---------------------------------------------------------------------------

def pair_two_judges(records: list[dict], judge_id_a: str, judge_id_b: str) -> list[tuple]:
    """(run_id, question_id) pairs that have records from BOTH given
    judge_ids, neither with a failed API call -- generalizes
    pair_primary_secondary() to any two judge_ids (the dashboard's
    Agreement panel lets a user pick any two that have judged the same
    items, defaulting to primary vs secondary)."""
    index = index_by_key(records)
    seen_keys = {(r["run_id"], r["question_id"]) for r in records}
    pairs = []
    for run_id, question_id in seen_keys:
        a = index.get((run_id, question_id, judge_id_a))
        b = index.get((run_id, question_id, judge_id_b))
        if a is None or b is None:
            continue
        if a.get("call_failed") or b.get("call_failed"):
            continue
        pairs.append((a, b))
    return pairs


def pair_primary_secondary(records: list[dict]) -> list[tuple]:
    """(run_id, question_id) pairs that have BOTH a primary and a
    secondary record, neither of which had a failed API call."""
    return pair_two_judges(records, "primary", "secondary")


def build_confusion_matrix(pairs: list[tuple]) -> dict:
    """4x4 confusion matrix (rows=primary, cols=secondary), including
    ABSTAIN. A pair where either side's label isn't one of ALL_LABELS
    (e.g. a parse error) is silently skipped -- it has no valid cell."""
    matrix = {a: {b: 0 for b in ALL_LABELS} for a in ALL_LABELS}
    for p, s in pairs:
        pl, sl = p.get("label"), s.get("label")
        if pl in ALL_LABELS and sl in ALL_LABELS:
            matrix[pl][sl] += 1
    return matrix


def compute_kappas(pairs: list[tuple]) -> dict:
    """Linear-weighted + unweighted Cohen's kappa over the 3 ordinal
    classes (FAKTUAL < HALUSINASI_SEBAGIAN < HALUSINASI_PENUH), EXCLUDING
    any pair where either judge's label is ABSTAIN (or missing/invalid)."""
    from sklearn.metrics import cohen_kappa_score

    ordinal_pairs = [(p["label"], s["label"]) for p, s in pairs
                     if p.get("label") in ORDINAL_LABELS and s.get("label") in ORDINAL_LABELS]
    excluded = len(pairs) - len(ordinal_pairs)
    if not ordinal_pairs:
        return {"weighted": None, "unweighted": None, "n_used": 0, "n_excluded": excluded}

    primary_labels = [p for p, _ in ordinal_pairs]
    secondary_labels = [s for _, s in ordinal_pairs]
    weighted = cohen_kappa_score(primary_labels, secondary_labels, labels=ORDINAL_LABELS, weights="linear")
    unweighted = cohen_kappa_score(primary_labels, secondary_labels, labels=ORDINAL_LABELS)
    return {"weighted": weighted, "unweighted": unweighted, "n_used": len(ordinal_pairs), "n_excluded": excluded}


def cmd_agreement(args):
    records = load_judge_records(args.judge_output)
    pairs = pair_primary_secondary(records)

    if not pairs:
        log("[ERROR] no (primary, secondary) pairs found for the same (run_id, question_id) in the given files")
        sys.exit(1)

    matrix = build_confusion_matrix(pairs)
    log(f"\nConfusion matrix (rows=primary, cols=secondary), n={len(pairs)} paired items:")
    header = "".join(f"{lbl[:12]:>14}" for lbl in ALL_LABELS)
    log(f"{'':>20}{header}")
    for a in ALL_LABELS:
        row = "".join(f"{matrix[a][b]:>14}" for b in ALL_LABELS)
        log(f"{a:>20}{row}")

    kappas = compute_kappas(pairs)
    log(f"\nPairs used for kappa (ABSTAIN excluded from either side): {kappas['n_used']} "
        f"({kappas['n_excluded']} pair(s) excluded for having an ABSTAIN on at least one side)")
    log(f"Linear-weighted Cohen's kappa (ordinal FAKTUAL < SEBAGIAN < PENUH): {kappas['weighted']}")
    log(f"Unweighted Cohen's kappa: {kappas['unweighted']}")

    # label vs derived_label consistency + parse-error rate, PER JUDGE
    log("\nPer-judge label/derived_label consistency & parse-error rate:")
    for judge_id in ("primary", "secondary"):
        judge_records = [r for r in records if r["judge_id"] == judge_id]
        if not judge_records:
            continue
        n = len(judge_records)
        n_parse_error = sum(1 for r in judge_records if r.get("parse_error"))
        consistent_records = [r for r in judge_records if not r.get("parse_error") and not r.get("call_failed")]
        n_consistent = sum(1 for r in consistent_records if r.get("consistent"))
        consistency_rate = round(n_consistent / len(consistent_records), 3) if consistent_records else None
        log(f"  {judge_id}: n={n}, parse_error_rate={round(n_parse_error / n, 3)}, "
            f"label==derived_label consistency rate={consistency_rate}")


# ---------------------------------------------------------------------------
# export-human-csv
# ---------------------------------------------------------------------------

def _split_evenly_by_group(grouped: dict, n_total: int, rng: random.Random) -> list:
    """Proportional-equal allocation across the dict's groups: n_total
    split as evenly as possible (base_share each, with the first
    `remainder` groups -- in sorted key order -- getting one extra), then
    the combined pick is reshuffled so item order doesn't betray group
    order. If a group's pool is smaller than its share, that group just
    contributes fewer items (no padding/repeat). `rng` is a caller-owned
    random.Random so a sequence of calls stays deterministic together."""
    groups = sorted(grouped)
    if not groups:
        return []

    base_share = n_total // len(groups)
    remainder = n_total % len(groups)

    picked = []
    for i, group in enumerate(groups):
        pool = list(grouped[group])
        rng.shuffle(pool)
        share = base_share + (1 if i < remainder else 0)
        picked.extend(pool[:share])

    rng.shuffle(picked)
    return picked


def stratified_sample(by_condition: dict, n_total: int, seed: int) -> list:
    """Thin, backwards-compatible wrapper: one-shot proportional-equal
    sample across a pre-grouped dict (originally keyed by condition
    letter; works for any grouping key). See _split_evenly_by_group()."""
    rng = random.Random(seed)
    return _split_evenly_by_group(by_condition, n_total, rng)


def _group_by(items: list[dict], key_fn) -> dict:
    grouped: dict = defaultdict(list)
    for item in items:
        grouped[key_fn(item)].append(item)
    return grouped


def stratified_sample_for_human_export(items: list[dict], n_total: int, seed: int,
                                        min_per_label: int = 10) -> tuple[list[dict], dict]:
    """Two-dimensional stratified sample for the human-annotation export.
    Each item must already carry `run_label` and `primary_label` keys
    (the latter from a join against the primary judge's existing output
    -- see cmd_export_human_csv()).

    Procedure (applied in this order, all draws share ONE
    random.Random(seed) so the whole thing is reproducible from
    (items, n_total, seed, min_per_label) alone):
      1. For each of the 3 ordinal primary-judge labels (FAKTUAL,
         HALUSINASI_SEBAGIAN, HALUSINASI_PENUH), guarantee at least
         `min_per_label` items carrying that label (or ALL available, if
         fewer than `min_per_label` exist for it) -- drawn evenly across
         run_label so no single run variant dominates a label's slice.
      2. Fill any remaining slots up to n_total from whatever's left
         (any label, including ones already at their guarantee),
         stratified by run_label the same way.
    If the 3 guarantees alone already reach or exceed n_total, step 2 is
    skipped and the final sample size is whatever the guarantees produced
    (never truncated -- a guarantee is never partially honored just to
    hit an exact n_total).

    Returns (sampled_items, procedure_metadata) -- the metadata dict is
    meant to be written to the export's own metadata file verbatim, so
    the exact sampling procedure actually applied is auditable later,
    not just described in a docstring.
    """
    rng = random.Random(seed)

    def item_key(item):
        return (item["run_id"], item["question_id"])

    selected_keys: set = set()
    selected: list[dict] = []
    label_guarantee_report = {}

    for label in ORDINAL_LABELS:
        pool = [it for it in items if it.get("primary_label") == label and item_key(it) not in selected_keys]
        target = min(min_per_label, len(pool))
        picked = _split_evenly_by_group(_group_by(pool, lambda it: it["run_label"]), target, rng)
        for it in picked:
            selected_keys.add(item_key(it))
            selected.append(it)
        label_guarantee_report[label] = {
            "min_requested": min_per_label, "available": len(pool), "picked": len(picked),
            "guarantee_met": len(picked) >= min_per_label,
        }

    remaining_slots = n_total - len(selected)
    if remaining_slots > 0:
        remaining_pool = [it for it in items if item_key(it) not in selected_keys]
        picked = _split_evenly_by_group(_group_by(remaining_pool, lambda it: it["run_label"]), remaining_slots, rng)
        for it in picked:
            selected_keys.add(item_key(it))
            selected.append(it)

    rng.shuffle(selected)

    run_label_distribution: dict[str, int] = defaultdict(int)
    for it in selected:
        run_label_distribution[it["run_label"]] += 1

    metadata = {
        "procedure": "stratified_sample_for_human_export_v1",
        "steps": [
            "1. For each of FAKTUAL/HALUSINASI_SEBAGIAN/HALUSINASI_PENUH (primary judge label), "
            "guarantee >= min_per_label items (or all available if fewer exist), drawn evenly "
            "across run_label.",
            "2. Fill remaining slots up to n_total from whatever's left, stratified by run_label.",
        ],
        "n_total_requested": n_total,
        "n_total_sampled": len(selected),
        "seed": seed,
        "min_per_primary_label": min_per_label,
        "label_guarantee_report": label_guarantee_report,
        "run_label_distribution": dict(run_label_distribution),
    }
    return selected, metadata


def cmd_export_human_csv(args):
    import os

    questions_parquet = args.questions_parquet or os.getenv("QUESTIONS_PARQUET")
    if not questions_parquet:
        log("[ERROR] --questions-parquet (or QUESTIONS_PARQUET in .env) is required")
        sys.exit(1)

    # Primary labels, to honor the >=10-per-label guarantee -- joined by
    # (run_id, question_id). An item with no primary judge record yet
    # (not judged, or judged only by secondary) simply has primary_label
    # = None and can still be picked during the "fill remaining slots"
    # step, just never counts toward a label guarantee.
    primary_label_by_key: dict[tuple, str] = {}
    if args.judge_output:
        for record in load_judge_records(args.judge_output):
            if record.get("judge_id") == "primary" and not record.get("call_failed"):
                primary_label_by_key[(record["run_id"], record["question_id"])] = record.get("label")

    items: list[dict] = []
    try:
        for path in args.run_files:
            metadata = resolve_run_metadata(path)
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    key = (metadata["run_id"], record["question_id"])
                    items.append({
                        **metadata,
                        "question_id": record["question_id"],
                        "title": record.get("title", ""),
                        "tags": format_tags(record.get("tags")),
                        "ground_truth_answer_html": record.get("ground_truth_answer", "") or "",
                        "candidate": record.get("llm_answer", "") or "",
                        "primary_label": primary_label_by_key.get(key),
                    })
    except RunMetadataNotFoundError as e:
        log(f"[ERROR] {e}")
        sys.exit(1)

    if not items:
        log("[ERROR] no items loaded from --run-files")
        sys.exit(1)

    sampled_items, procedure_metadata = stratified_sample_for_human_export(
        items, args.n, args.seed, min_per_label=args.min_per_label)

    question_ids = sorted({item["question_id"] for item in sampled_items})
    bodies = load_question_bodies(questions_parquet, question_ids)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path = Path(args.mapping) if args.mapping else out_path.with_name(
        out_path.stem + "_mapping.jsonl")
    metadata_path = out_path.with_name(out_path.stem + "_metadata.json")

    with open(out_path, "w", newline="", encoding="utf-8") as csv_f, \
         open(mapping_path, "w", encoding="utf-8") as map_f:
        writer = csv.writer(csv_f)
        writer.writerow(["item_id", "question_title", "question_body", "reference_answer",
                          "candidate_answer_blinded", "human_label"])
        for idx, item in enumerate(sampled_items, start=1):
            item_id = f"item_{idx:04d}"
            body_text = html_to_text(bodies.get(item["question_id"], ""))
            reference_text = html_to_text(item["ground_truth_answer_html"])
            blinded_candidate = blind_candidate(item["candidate"])
            writer.writerow([item_id, item["title"], body_text, reference_text, blinded_candidate, ""])
            map_f.write(json.dumps({
                "item_id": item_id,
                "run_id": item["run_id"],
                "condition": item["condition"],
                "run_label": item["run_label"],
                "question_id": item["question_id"],
            }) + "\n")

    procedure_metadata.update({
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_files": args.run_files,
        "judge_output_files_for_primary_label_join": args.judge_output or [],
        "out_csv": str(out_path),
        "mapping_file": str(mapping_path),
    })
    with open(metadata_path, "w", encoding="utf-8") as meta_f:
        json.dump(procedure_metadata, meta_f, indent=2)

    log(f"Wrote {len(sampled_items)} blinded item(s) to {out_path}")
    log(f"Wrote item_id -> (run_id, condition, run_label, question_id) mapping to {mapping_path} "
        f"(KEEP THIS SEPARATE from the CSV while labeling -- it reveals condition)")
    log(f"Wrote sampling procedure metadata to {metadata_path}")
    log(f"Run_label distribution: {procedure_metadata['run_label_distribution']}")
    log(f"Primary-label guarantee report: {json.dumps(procedure_metadata['label_guarantee_report'], indent=2)}")


# ---------------------------------------------------------------------------
# human-kappa
# ---------------------------------------------------------------------------

def compute_human_kappa(human_csv_path: str, mapping_path: str, judge_output_paths: list[str],
                         judge_ids: tuple[str, ...] = ("primary", "secondary")) -> dict[str, dict]:
    """Pure version of cmd_human_kappa()'s math -- returns a dict keyed by
    judge_id (only for judge_ids that have >=1 comparable pair; a judge_id
    with none is simply absent, with its n_missing_human/n_missing_judge
    counts still reported via a sibling "insufficient_data" dict) instead
    of printing, so the dashboard backend can call this directly."""
    from sklearn.metrics import cohen_kappa_score

    with open(human_csv_path, newline="", encoding="utf-8") as f:
        human_rows = list(csv.DictReader(f))

    mapping = {}
    with open(mapping_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            mapping[m["item_id"]] = (m["run_id"], m["question_id"])

    judge_records = load_judge_records(judge_output_paths)
    judge_index = index_by_key(judge_records)

    results: dict[str, dict] = {}
    for judge_id in judge_ids:
        human_labels, judge_labels = [], []
        n_missing_human, n_missing_judge = 0, 0
        for row in human_rows:
            human_label = (row.get("human_label") or "").strip().upper()
            if not human_label:
                n_missing_human += 1
                continue
            key = mapping.get(row["item_id"])
            if key is None:
                continue
            run_id, question_id = key
            judge_record = judge_index.get((run_id, question_id, judge_id))
            if judge_record is None or judge_record.get("label") not in ORDINAL_LABELS:
                n_missing_judge += 1
                continue
            if human_label not in ORDINAL_LABELS:
                continue  # ABSTAIN (either side) excluded from kappa, same rule as judge-vs-judge
            human_labels.append(human_label)
            judge_labels.append(judge_record["label"])

        if not human_labels:
            results[judge_id] = {
                "insufficient_data": True, "n_paired": 0,
                "n_missing_human": n_missing_human, "n_missing_judge": n_missing_judge,
            }
            continue

        weighted = cohen_kappa_score(human_labels, judge_labels, labels=ORDINAL_LABELS, weights="linear")
        unweighted = cohen_kappa_score(human_labels, judge_labels, labels=ORDINAL_LABELS)
        results[judge_id] = {
            "insufficient_data": False, "n_paired": len(human_labels),
            "weighted_kappa": round(float(weighted), 3), "unweighted_kappa": round(float(unweighted), 3),
        }
    return results


def cmd_human_kappa(args):
    results = compute_human_kappa(args.human_csv, args.mapping, args.judge_output)
    for judge_id, r in results.items():
        if r["insufficient_data"]:
            log(f"{judge_id}: no comparable (human, judge) pairs yet "
                f"(missing human_label: {r['n_missing_human']}, missing/ABSTAIN judge label: {r['n_missing_judge']})")
        else:
            log(f"{judge_id}: n={r['n_paired']} paired, weighted kappa={r['weighted_kappa']:.3f}, "
                f"unweighted kappa={r['unweighted_kappa']:.3f}")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def compute_run_label_summary(records: list[dict], judge_id: str) -> dict[str, dict]:
    """Pure version of cmd_summary()'s per-run_label math -- returns a
    dict keyed by run_label instead of printing, so the dashboard backend
    (judge-v1 Results UI) can call this directly rather than parsing CLI
    output. Each value: n, abstention_rate, hall_rate_excl_abstain,
    hall_rate_incl_abstain, label_distribution, parse_error_rate,
    consistency_rate. Already filters `records` to `judge_id` internally,
    same as cmd_summary()."""
    records = [r for r in records if r["judge_id"] == judge_id]
    by_run_label: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_run_label[r["run_label"]].append(r)

    out: dict[str, dict] = {}
    for run_label, recs in by_run_label.items():
        n = len(recs)
        n_abstain = sum(1 for r in recs if r.get("label") == "ABSTAIN")
        n_hallucinated = sum(1 for r in recs if r.get("label") in ("HALUSINASI_SEBAGIAN", "HALUSINASI_PENUH"))
        n_non_abstain = n - n_abstain
        n_parse_error = sum(1 for r in recs if r.get("parse_error"))
        # `consistent` (label vs derived_label agreement, see judge_prompt_v1)
        # is None for a record that never got that far (call_failed/parse
        # error) -- excluded from the denominator rather than counted as
        # "inconsistent", same exclusion style as kappa's ABSTAIN handling.
        consistent_values = [r.get("consistent") for r in recs if r.get("consistent") is not None]

        distribution: dict[str, int] = defaultdict(int)
        for r in recs:
            distribution[r.get("label") or "CALL_FAILED/PARSE_ERROR"] += 1

        out[run_label] = {
            "n": n,
            "abstention_rate": round(n_abstain / n, 3) if n else None,
            "hall_rate_excl_abstain": round(n_hallucinated / n_non_abstain, 3) if n_non_abstain else None,
            "hall_rate_incl_abstain": round((n_hallucinated + n_abstain) / n, 3) if n else None,
            "label_distribution": dict(distribution),
            "parse_error_rate": round(n_parse_error / n, 3) if n else None,
            "consistency_rate": round(sum(1 for v in consistent_values if v) / len(consistent_values), 3)
            if consistent_values else None,
        }
    return out


def cmd_summary(args):
    records = load_judge_records(args.judge_output)
    summary = compute_run_label_summary(records, args.judge_id)
    if not summary:
        log(f"[ERROR] no records found for judge_id='{args.judge_id}' in the given files")
        sys.exit(1)

    log(f"\nPer-run_label hallucination (reference-based) summary (judge_id={args.judge_id}):")
    for run_label in sorted(summary):
        s = summary[run_label]
        log(f"\n  {run_label} (n={s['n']}):")
        log(f"    Abstention rate: {s['abstention_rate']}")
        log(f"    Hallucination (reference-based) rate (excluding ABSTAIN): {s['hall_rate_excl_abstain']}")
        log(f"    Hallucination (reference-based) rate (ABSTAIN counted as non-factual): {s['hall_rate_incl_abstain']}")
        log(f"    Label distribution: {s['label_distribution']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_agree = sub.add_parser("agreement")
    p_agree.add_argument("--judge-output", nargs="+", required=True)
    p_agree.set_defaults(func=cmd_agreement)

    p_export = sub.add_parser("export-human-csv")
    p_export.add_argument("--run-files", nargs="+", required=True)
    p_export.add_argument("--out", required=True)
    p_export.add_argument("--mapping", default=None, help="Default: <out>_mapping.jsonl next to --out")
    p_export.add_argument("--n", type=int, default=60)
    p_export.add_argument("--seed", type=int, default=42)
    p_export.add_argument("--min-per-label", type=int, default=10,
                           help="Guarantee at least this many items per primary-judge ordinal label, "
                                "where available (default 10)")
    p_export.add_argument("--judge-output", nargs="*", default=[],
                           help="Existing judge output file(s) to join primary labels from, for the "
                                "per-label guarantee. Omit to sample without any label guarantee "
                                "(e.g. before any judging has run yet).")
    p_export.add_argument("--questions-parquet", default=None)
    p_export.set_defaults(func=cmd_export_human_csv)

    p_human = sub.add_parser("human-kappa")
    p_human.add_argument("--human-csv", required=True)
    p_human.add_argument("--mapping", required=True)
    p_human.add_argument("--judge-output", nargs="+", required=True)
    p_human.set_defaults(func=cmd_human_kappa)

    p_summary = sub.add_parser("summary")
    p_summary.add_argument("--judge-output", nargs="+", required=True)
    p_summary.add_argument("--judge-id", choices=["primary", "secondary"], default="primary")
    p_summary.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
