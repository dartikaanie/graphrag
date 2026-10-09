"""
annotation_tool.py
=====================================
Blind human annotation for judge calibration and confirmation
(docs/DECISION_C_SCORING.md, Amendment 4 + Addendum).

Subcommands
    export              stratified sample -> one blind XLSX per annotator + key (outside the repo)
    validate            check a filled XLSX (invalid cells are reported, nothing is written)
    human-kappa         annotator 1 vs annotator 2 on the overlap + adjudication template
    finalize-reference  build the adjudicated reference file (unlocks the calibration judge guard)
    judge-kappa         judge vs human reference, per judge and judge version (Markdown report)

Two purposes share one flow:
    calibration   60 items = 20 per condition (A, B-plain, B-grounded), seed 42; annotator 2 labels a
                  fixed 30-item subset (10 per condition, seed 42).
    confirmation  30 items stratified across the stage-2 run files given, seed 42; annotator 1 only.

BLINDING. The XLSX files contain no condition, run label, or judge output. Annotators see exactly what
the judge sees: question title + body, accepted answer, and the candidate answer with citation markers
(e.g. [SO-12345]) stripped by judge_prompt_v1.blind_candidate (recorded in the export manifest); the
text is otherwise unchanged (same HTML-to-text conversion and truncation as the judge's input).

KEY. item_id -> (question_id, run, condition, seed) is written OUTSIDE the repository (default
~/graphrag_annotation_keys/, see --key-dir / GRAPHRAG_ANNOTATION_KEY_DIR). Only its SHA-256 is stored in
the export manifest next to the annotation files; every later step verifies the key against that hash.

CARA PAKAI (from llm/evaluation)
----------------------------------------------------------------------------
    python3 annotation_tool.py export --purpose calibration
    python3 annotation_tool.py validate --xlsx annotation/calibration/annotator1.xlsx --purpose calibration
    python3 annotation_tool.py human-kappa --purpose calibration
    python3 annotation_tool.py finalize-reference --purpose calibration
    python3 annotation_tool.py judge-kappa --purpose calibration --judge-output results/jv2_1_*.jsonl
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from judge_binary import (  # noqa: E402
    CATEGORIES, CATEGORY_ABSTAIN, CATEGORY_HALLUCINATED, CATEGORY_NOT_HALLUCINATED, agreement_report, category_of,
)

log = print

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
ANNOTATION_ROOT = EVAL_DIR / "annotation"
DEFAULT_KEY_DIR = Path.home() / "graphrag_annotation_keys"
DEFAULT_SEED = 42
ACCEPTANCE_KAPPA = 0.6

PURPOSES = {
    # n_per_stratum / n_overlap_per_stratum: fixed per-condition counts (Amendment 4, calibration set).
    "calibration": {"prefix": "CAL", "n_per_stratum": 20, "n_overlap_per_stratum": 10, "n_total": 60},
    # n_total split evenly across the stage-2 run files given; one annotator.
    "confirmation": {"prefix": "CONF", "n_per_stratum": None, "n_overlap_per_stratum": 0, "n_total": 30},
}

COLUMNS = ["item_id", "question", "accepted_answer", "candidate_answer", "abstain", "hallucinated", "severity", "notes"]
TEXT_COLUMNS = ("question", "accepted_answer", "candidate_answer")
LABEL_COLUMNS = ("abstain", "hallucinated", "severity")
YES_NO = ("Y", "N")
SEVERITIES = ("PENUH", "SEBAGIAN")
MAX_CHARS_EACH = 6000  # same cut as judge_prompt_v2_1.build_messages
_ILLEGAL_XLSX_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_CITATION_MARKER = re.compile(r"\[SO-[^\]]*\]")


class AnnotationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Text shown to annotators == text shown to the judge
# ---------------------------------------------------------------------------

def _cut(s: str) -> str:
    s = s or ""
    return s if len(s) <= MAX_CHARS_EACH else s[:MAX_CHARS_EACH] + "\n[...truncated]"


def _xlsx_safe(s: str) -> str:
    return _ILLEGAL_XLSX_CHARS.sub("", s or "")


def shown_texts(title: str, body_html: str, reference_html: str, candidate: str) -> dict:
    """The three text cells. Same conversions as the judge input: html_to_text for the question body and
    the accepted answer, blind_candidate (citation markers stripped) for the candidate, cut at 6000 chars."""
    from _judge_common import html_to_text
    from judge_prompt_v1 import blind_candidate

    body = html_to_text(body_html) if body_html else "[question body not found in parquet]"
    return {
        "question": _xlsx_safe(f"{title}\n\n{_cut(body)}"),
        "accepted_answer": _xlsx_safe(_cut(html_to_text(reference_html))),
        "candidate_answer": _xlsx_safe(_cut(blind_candidate(candidate))),
    }


def text_sha256(text) -> str:
    """Hash of a text cell, whitespace-normalized so a spreadsheet round trip (line endings, trailing
    spaces) does not look like an edit, while any change of wording does."""
    return hashlib.sha256(" ".join(str(text or "").split()).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Sampling (pure, deterministic)
# ---------------------------------------------------------------------------

def _rng(seed: int, *parts) -> random.Random:
    return random.Random("|".join([str(seed), *map(str, parts)]))


def allocate_evenly(strata: list[str], n_total: int) -> dict[str, int]:
    """n_total split as evenly as possible over the sorted strata; the first `remainder` get one extra."""
    strata = sorted(strata)
    base, rem = divmod(n_total, len(strata))
    return {s: base + (1 if i < rem else 0) for i, s in enumerate(strata)}


def sample_items(items: list[dict], purpose: str, seed: int = DEFAULT_SEED,
                 n_per_stratum: int | None = None, n_overlap_per_stratum: int | None = None,
                 n_total: int | None = None) -> list[dict]:
    """Stratified sample by `run_label`. Within each stratum the pool is sorted by question_id and drawn
    with a random.Random seeded from (seed, purpose, stratum) -- independent of every other stratum and of
    any judge output. The annotator-2 subset is drawn from each stratum's sample the same way. Returns the
    sampled items in annotator-1 order (shuffled across strata), each with `item_id` and
    `in_annotator2_subset`; item ids are assigned AFTER the shuffle, so they carry no stratum information."""
    spec = PURPOSES[purpose]
    n_per_stratum = spec["n_per_stratum"] if n_per_stratum is None else n_per_stratum
    n_overlap = spec["n_overlap_per_stratum"] if n_overlap_per_stratum is None else n_overlap_per_stratum
    n_total = spec["n_total"] if n_total is None else n_total

    by_stratum: dict[str, list[dict]] = {}
    for it in items:
        by_stratum.setdefault(it["run_label"], []).append(it)
    quota = ({s: n_per_stratum for s in by_stratum} if n_per_stratum is not None
             else allocate_evenly(list(by_stratum), n_total))

    picked: list[dict] = []
    for stratum in sorted(by_stratum):
        pool = sorted(by_stratum[stratum], key=lambda it: it["question_id"])
        if len(pool) < quota[stratum]:
            raise AnnotationError(f"stratum '{stratum}' has {len(pool)} items, needs {quota[stratum]}")
        chosen = _rng(seed, purpose, "sample", stratum).sample(pool, quota[stratum])
        overlap_ids = {id(x) for x in _rng(seed, purpose, "overlap", stratum).sample(
            sorted(chosen, key=lambda it: it["question_id"]), min(n_overlap, len(chosen)))}
        picked += [{**it, "in_annotator2_subset": id(it) in overlap_ids} for it in chosen]

    picked.sort(key=lambda it: (it["run_label"], it["question_id"]))
    _rng(seed, purpose, "order", "annotator1").shuffle(picked)
    width = max(3, len(str(len(picked))))
    for i, it in enumerate(picked, start=1):
        it["item_id"] = f"{spec['prefix']}-{i:0{width}d}"
    return picked


def annotator2_order(sampled: list[dict], purpose: str, seed: int = DEFAULT_SEED) -> list[dict]:
    subset = sorted((it for it in sampled if it["in_annotator2_subset"]), key=lambda it: it["item_id"])
    _rng(seed, purpose, "order", "annotator2").shuffle(subset)
    return subset


# ---------------------------------------------------------------------------
# XLSX writing / reading
# ---------------------------------------------------------------------------

def write_annotation_xlsx(path: Path, rows: list[dict]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "anotasi"
    ws.append(COLUMNS)
    for r in rows:
        ws.append([r["item_id"], r["question"], r["accepted_answer"], r["candidate_answer"], None, None, None, None])

    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDDDDD")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"
    for col, width in zip("ABCDEFGH", (11, 60, 60, 60, 10, 13, 12, 32)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2, max_row=len(rows) + 1):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    last = len(rows) + 1
    for col, values in (("E", YES_NO), ("F", YES_NO), ("G", SEVERITIES)):
        dv = DataValidation(type="list", formula1='"' + ",".join(values) + '"', allow_blank=True, showErrorMessage=True)
        dv.error, dv.errorTitle = f"Isi dengan salah satu: {', '.join(values)}", "Nilai tidak valid"
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{last}")

    info = wb.create_sheet("petunjuk")
    for line in (
        "Baca docs/ANNOTATION_GUIDELINE.md sebelum mulai.",
        "abstain: Y jika jawaban kandidat tidak mencoba menjawab; N jika mencoba menjawab.",
        "hallucinated: isi HANYA jika abstain = N. Y jika ada >= 1 klaim yang bertentangan dengan referensi "
        "atau fakta mapan, atau yang dikarang (API/fungsi/opsi tidak ada). N jika tidak ada.",
        "severity: isi HANYA jika hallucinated = Y. PENUH jika kesalahan mengenai solusi utama; SEBAGIAN jika "
        "hanya detail sekunder.",
        "notes: catatan bebas. Tulis 'referensi usang' bila memakai aturan referensi usang.",
        "Jangan mengubah kolom item_id, question, accepted_answer, candidate_answer.",
    ):
        info.append([line])
    info.column_dimensions["A"].width = 120
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def _norm_label(value) -> str | None:
    if value is None:
        return None
    v = str(value).strip().upper()
    return v or None


def read_annotation_xlsx(path: Path) -> tuple[list[dict], list[str]]:
    """Reads a filled annotation sheet. Returns (rows, problems); every problem names its cell. Rules:
      abstain      required, Y/N
      hallucinated required iff abstain == N, Y/N; must be empty when abstain == Y
      severity     required iff hallucinated == Y, PENUH/SEBAGIAN; must be empty otherwise
    A row with any problem has category None."""
    from openpyxl import load_workbook

    ws = load_workbook(path, data_only=True)["anotasi"]
    header = [str(c.value).strip() if c.value is not None else "" for c in ws[1]]
    problems: list[str] = []
    if header[:len(COLUMNS)] != COLUMNS:
        return [], [f"{path.name}: header row is {header}, expected {COLUMNS}"]
    col = {name: i for i, name in enumerate(COLUMNS)}
    letter = {name: "ABCDEFGH"[i] for name, i in col.items()}

    rows, seen = [], set()
    for n, cells in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if all(v is None or str(v).strip() == "" for v in cells[:len(COLUMNS)]):
            continue
        item_id = str(cells[col["item_id"]] or "").strip()
        if not item_id:
            problems.append(f"{path.name}!A{n}: missing item_id")
            continue
        if item_id in seen:
            problems.append(f"{path.name}!A{n}: duplicate item_id {item_id}")
        seen.add(item_id)

        abstain, hall, sev = (_norm_label(cells[col[c]]) for c in LABEL_COLUMNS)
        row_problems = []
        if abstain not in YES_NO:
            row_problems.append(f"{path.name}!{letter['abstain']}{n} ({item_id}): abstain is {abstain!r}, must be Y or N")
        elif abstain == "Y":
            if hall is not None:
                row_problems.append(f"{path.name}!{letter['hallucinated']}{n} ({item_id}): hallucinated must be "
                                    f"empty when abstain = Y (is {hall!r})")
            if sev is not None:
                row_problems.append(f"{path.name}!{letter['severity']}{n} ({item_id}): severity must be empty "
                                    f"when abstain = Y (is {sev!r})")
        else:
            if hall not in YES_NO:
                row_problems.append(f"{path.name}!{letter['hallucinated']}{n} ({item_id}): hallucinated is "
                                    f"{hall!r}, must be Y or N when abstain = N")
            elif hall == "Y" and sev not in SEVERITIES:
                row_problems.append(f"{path.name}!{letter['severity']}{n} ({item_id}): severity is {sev!r}, must "
                                    f"be PENUH or SEBAGIAN when hallucinated = Y")
            elif hall == "N" and sev is not None:
                row_problems.append(f"{path.name}!{letter['severity']}{n} ({item_id}): severity must be empty "
                                    f"when hallucinated = N (is {sev!r})")
        problems += row_problems

        category = None
        if not row_problems:
            category = (CATEGORY_ABSTAIN if abstain == "Y"
                        else CATEGORY_HALLUCINATED if hall == "Y" else CATEGORY_NOT_HALLUCINATED)
        rows.append({
            "item_id": item_id, "row": n, "abstain": abstain, "hallucinated": hall, "severity": sev,
            "notes": str(cells[col["notes"]] or "").strip(), "category": category,
            "text_sha256": {c: text_sha256(cells[col[c]]) for c in TEXT_COLUMNS},
        })
    return rows, problems


# ---------------------------------------------------------------------------
# Key (outside the repo) + manifest (inside, hash only)
# ---------------------------------------------------------------------------

def key_dir_from(arg: str | None) -> Path:
    return Path(arg or os.getenv("GRAPHRAG_ANNOTATION_KEY_DIR") or DEFAULT_KEY_DIR).expanduser()


def purpose_dir(purpose: str, annotation_root: Path | None = None) -> Path:
    return (annotation_root or ANNOTATION_ROOT) / purpose


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(purpose: str, annotation_root: Path | None = None) -> dict:
    path = purpose_dir(purpose, annotation_root) / "export_manifest.json"
    if not path.exists():
        raise AnnotationError(f"no export manifest at {path} -- run `export --purpose {purpose}` first")
    return json.loads(path.read_text())


def load_verified_key(purpose: str, key_dir: Path, annotation_root: Path | None = None) -> dict:
    """Loads the key from OUTSIDE the repo and verifies its SHA-256 against the committed manifest."""
    manifest = load_manifest(purpose, annotation_root)
    key_path = key_dir / manifest["key_file_name"]
    if not key_path.exists():
        raise AnnotationError(f"key file not found at {key_path} (set --key-dir or GRAPHRAG_ANNOTATION_KEY_DIR)")
    actual = file_sha256(key_path)
    if actual != manifest["key_sha256"]:
        raise AnnotationError(f"key file {key_path} has sha256 {actual}, but the export manifest records "
                              f"{manifest['key_sha256']} -- this is not the key of this export")
    return json.loads(key_path.read_text())


def check_against_key(rows: list[dict], key: dict, annotator: str, file_name: str) -> list[str]:
    """Item-id coverage and text integrity of a filled sheet against the key."""
    problems = []
    expected = {it["item_id"]: it for it in key["items"] if annotator == "annotator1" or it["in_annotator2_subset"]}
    got = {r["item_id"] for r in rows}
    for missing in sorted(set(expected) - got):
        problems.append(f"{file_name}: item {missing} is missing")
    for extra in sorted(got - set(expected)):
        problems.append(f"{file_name}: unexpected item {extra}")
    for r in rows:
        k = expected.get(r["item_id"])
        if k is None:
            continue
        for c in TEXT_COLUMNS:
            if r["text_sha256"][c] != k["text_sha256"][c]:
                problems.append(f"{file_name} row {r['row']} ({r['item_id']}): column '{c}' was edited "
                                f"(text no longer matches the export)")
    return problems


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def load_run_items(run_files: list[str]) -> list[dict]:
    from _run_metadata import resolve_run_metadata

    items = []
    for path in run_files:
        meta = resolve_run_metadata(path)
        for line in open(path):
            if not line.strip():
                continue
            rec = json.loads(line)
            items.append({
                "run_id": meta["run_id"], "run_label": meta["run_label"], "condition": meta["condition"],
                "source_file": str(path), "config_hash": rec.get("config_hash"),
                "question_id": rec["question_id"], "title": rec.get("title", "") or "",
                "ground_truth_answer_html": rec.get("ground_truth_answer", "") or "",
                "candidate": rec.get("llm_answer", "") or "",
            })
    return items


def build_export(items: list[dict], bodies_html: dict, purpose: str, seed: int = DEFAULT_SEED, **sample_kw) -> dict:
    """Pure: returns {"annotator1": [rows], "annotator2": [rows], "key_items": [...], "n_markers_stripped"}.
    Rows hold ONLY the blind columns."""
    sampled = sample_items(items, purpose, seed, **sample_kw)
    rows_by_id, key_items, n_markers = {}, [], 0
    for it in sampled:
        texts = shown_texts(it["title"], bodies_html.get(it["question_id"], ""), it["ground_truth_answer_html"],
                            it["candidate"])
        markers = len(_CITATION_MARKER.findall(it["candidate"]))
        n_markers += markers
        rows_by_id[it["item_id"]] = {"item_id": it["item_id"], **texts}
        key_items.append({
            "item_id": it["item_id"], "question_id": it["question_id"], "run_id": it["run_id"],
            "run_label": it["run_label"], "condition": it["condition"], "config_hash": it["config_hash"],
            "source_file": Path(it["source_file"]).name, "in_annotator2_subset": it["in_annotator2_subset"],
            "citation_markers_stripped": markers,
            "text_sha256": {c: text_sha256(texts[c]) for c in TEXT_COLUMNS},
        })
    return {
        "annotator1": [rows_by_id[it["item_id"]] for it in sampled],
        "annotator2": [rows_by_id[it["item_id"]] for it in annotator2_order(sampled, purpose, seed)],
        "key_items": key_items, "n_markers_stripped": n_markers,
    }


def cmd_export(args) -> int:
    from _judge_common import load_calibration_registry, load_question_bodies
    from llm.repo_state import repo_state

    purpose, out_dir, key_dir = args.purpose, purpose_dir(args.purpose), key_dir_from(args.key_dir)
    run_files = args.run_files
    if not run_files and purpose == "calibration":
        run_files = [str(REPO_ROOT / r["output_path"]) for r in load_calibration_registry().get("runs", [])]
    if not run_files:
        raise AnnotationError("--run-files is required (no calibration registry found)")
    manifest_path = out_dir / "export_manifest.json"
    if manifest_path.exists() and not args.force:
        raise AnnotationError(f"{manifest_path} already exists -- an export for '{purpose}' was already made. "
                              f"Re-exporting would invalidate annotation in progress; pass --force to overwrite.")
    questions_parquet = args.questions_parquet or os.getenv("QUESTIONS_PARQUET")
    if not questions_parquet:
        raise AnnotationError("--questions-parquet (or QUESTIONS_PARQUET in .env) is required")

    items = load_run_items(run_files)
    sample_kw = {}
    if args.n_total is not None:
        sample_kw["n_total"] = args.n_total
    sampled_preview = sample_items(items, purpose, args.seed, **sample_kw)
    bodies = load_question_bodies(questions_parquet, sorted({it["question_id"] for it in sampled_preview}))
    export = build_export(items, bodies, purpose, args.seed, **sample_kw)

    spec, git = PURPOSES[purpose], repo_state()
    key = {
        "purpose": purpose, "seed": args.seed, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git["git_commit"],
        "sampling": {"strata": "run_label", "n_per_stratum": spec["n_per_stratum"],
                     "n_overlap_per_stratum": spec["n_overlap_per_stratum"],
                     "n_total": len(export["key_items"]), "run_files": [Path(f).name for f in run_files]},
        "items": export["key_items"],
    }
    key_dir.mkdir(parents=True, exist_ok=True)
    key_name = f"{purpose}_key_seed{args.seed}.json"
    key_path = key_dir / key_name
    key_path.write_text(json.dumps(key, indent=2) + "\n")

    files = {"annotator1": out_dir / "annotator1.xlsx"}
    write_annotation_xlsx(files["annotator1"], export["annotator1"])
    if export["annotator2"]:
        files["annotator2"] = out_dir / "annotator2.xlsx"
        write_annotation_xlsx(files["annotator2"], export["annotator2"])

    manifest_path.write_text(json.dumps({
        "purpose": purpose, "seed": args.seed, "created_at_utc": key["created_at_utc"],
        "git_commit": git["git_commit"],
        "n_items": len(export["annotator1"]), "n_items_annotator2": len(export["annotator2"]),
        "columns": COLUMNS,
        "key_file_name": key_name, "key_sha256": file_sha256(key_path),
        "key_location_note": "kept OUTSIDE the repository until adjudication is complete (Amendment 4 Addendum)",
        "blinding": {
            "hidden": ["condition", "run_label", "run_id", "judge output"],
            "citation_markers_stripped": True, "n_citation_markers_stripped": export["n_markers_stripped"],
            "citation_stripping": "judge_prompt_v1.blind_candidate (same as the judge input); text otherwise unchanged",
            "text_conversion": "html_to_text for question body and accepted answer; cut at 6000 chars, as the judge input",
        },
    }, indent=2) + "\n")

    log(f"Wrote {len(export['annotator1'])} items -> {files['annotator1']}")
    if "annotator2" in files:
        log(f"Wrote {len(export['annotator2'])} items -> {files['annotator2']}")
    log(f"Wrote key OUTSIDE the repo -> {key_path}")
    log(f"Wrote manifest (key sha256 only) -> {manifest_path}")
    return 0


# ---------------------------------------------------------------------------
# validate / human-kappa / finalize-reference
# ---------------------------------------------------------------------------

def load_annotator(purpose: str, annotator: str, key: dict, annotation_root: Path | None = None,
                   path: Path | None = None) -> tuple[dict, list[str]]:
    path = path or purpose_dir(purpose, annotation_root) / f"{annotator}.xlsx"
    if not path.exists():
        return {}, [f"{path} not found"]
    rows, problems = read_annotation_xlsx(path)
    problems += check_against_key(rows, key, annotator, path.name)
    return {r["item_id"]: r for r in rows}, problems


def cmd_validate(args) -> int:
    key = load_verified_key(args.purpose, key_dir_from(args.key_dir))
    annotator = args.annotator or ("annotator2" if "annotator2" in Path(args.xlsx).name else "annotator1")
    rows, problems = load_annotator(args.purpose, annotator, key, path=Path(args.xlsx))
    log(f"{args.xlsx}: {len(rows)} rows read as {annotator}; {len(problems)} problem(s)")
    for p in problems:
        log(f"  - {p}")
    return 1 if problems else 0


def human_human(a1: dict, a2: dict) -> dict:
    """Pre-adjudication agreement on the overlap items + the disagreement list."""
    overlap = sorted(set(a1) & set(a2))
    report = agreement_report([a1[i]["category"] for i in overlap], [a2[i]["category"] for i in overlap])
    report["item_ids"] = overlap
    report["disagreements"] = [
        {"item_id": i, "annotator1": a1[i]["category"], "annotator2": a2[i]["category"],
         "annotator1_severity": a1[i]["severity"], "annotator2_severity": a2[i]["severity"]}
        for i in overlap if a1[i]["category"] != a2[i]["category"]
    ]
    return report


def write_adjudication_xlsx(path: Path, disagreements: list[dict], a1: dict, a2: dict, sheet_rows: dict) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "adjudikasi"
    header = ["item_id", "question", "accepted_answer", "candidate_answer",
              "a1_abstain", "a1_hallucinated", "a1_severity", "a1_notes",
              "a2_abstain", "a2_hallucinated", "a2_severity", "a2_notes",
              "abstain", "hallucinated", "severity", "notes"]
    ws.append(header)
    for d in disagreements:
        i = d["item_id"]
        t = sheet_rows.get(i, {})
        ws.append([i, t.get("question"), t.get("accepted_answer"), t.get("candidate_answer"),
                   a1[i]["abstain"], a1[i]["hallucinated"], a1[i]["severity"], a1[i]["notes"],
                   a2[i]["abstain"], a2[i]["hallucinated"], a2[i]["severity"], a2[i]["notes"],
                   None, None, None, None])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "B2"
    for row in ws.iter_rows(min_row=1, max_row=len(disagreements) + 1):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for col, width in zip("ABCD", (11, 50, 50, 50)):
        ws.column_dimensions[col].width = width
    last = len(disagreements) + 1
    for col, values in (("M", YES_NO), ("N", YES_NO), ("O", SEVERITIES)):
        dv = DataValidation(type="list", formula1='"' + ",".join(values) + '"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{max(last, 2)}")
    wb.save(path)


def read_adjudication_xlsx(path: Path) -> tuple[dict, list[str]]:
    """Final labels of the disagreement items: same validation rules as an annotation sheet, on the
    final abstain/hallucinated/severity columns (M, N, O)."""
    from openpyxl import load_workbook

    ws = load_workbook(path, data_only=True)["adjudikasi"]
    out, problems = {}, []
    for n, cells in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if not cells or not cells[0]:
            continue
        item_id = str(cells[0]).strip()
        abstain, hall, sev = (_norm_label(v) for v in cells[12:15])
        ok = (abstain == "Y" and hall is None and sev is None) or \
             (abstain == "N" and hall == "N" and sev is None) or \
             (abstain == "N" and hall == "Y" and sev in SEVERITIES)
        if not ok:
            problems.append(f"{path.name} row {n} ({item_id}): final labels abstain={abstain!r} "
                            f"hallucinated={hall!r} severity={sev!r} are missing or inconsistent (columns M-O)")
            continue
        out[item_id] = {
            "abstain": abstain, "hallucinated": hall, "severity": sev,
            "category": CATEGORY_ABSTAIN if abstain == "Y" else
            CATEGORY_HALLUCINATED if hall == "Y" else CATEGORY_NOT_HALLUCINATED,
        }
    return out, problems


def _texts_from_sheet(path: Path) -> dict:
    from openpyxl import load_workbook

    out = {}
    for cells in load_workbook(path, data_only=True)["anotasi"].iter_rows(min_row=2, values_only=True):
        if cells and cells[0]:
            out[str(cells[0]).strip()] = {"question": cells[1], "accepted_answer": cells[2], "candidate_answer": cells[3]}
    return out


def cmd_human_kappa(args) -> int:
    purpose, pdir = args.purpose, purpose_dir(args.purpose)
    key = load_verified_key(purpose, key_dir_from(args.key_dir))
    a1, p1 = load_annotator(purpose, "annotator1", key)
    a2, p2 = load_annotator(purpose, "annotator2", key)
    problems = p1 + p2
    if problems:
        log(f"[STOP] {len(problems)} problem(s) in the annotation files -- fix them first:")
        for p in problems:
            log(f"  - {p}")
        return 1
    report = human_human(a1, a2)
    adjudication_path = pdir / "adjudication.xlsx"
    if not adjudication_path.exists() or args.force:
        write_adjudication_xlsx(adjudication_path, report["disagreements"], a1, a2,
                                _texts_from_sheet(pdir / "annotator1.xlsx"))
    (pdir / "human_human_agreement.json").write_text(json.dumps(report, indent=2) + "\n")
    log(f"Human-human agreement on {report['n']} overlap items (pre-adjudication):")
    log(f"  3-category kappa (unweighted): {report['kappa_3cat']}  observed agreement: {report['observed_agreement_3cat']}")
    log(f"  binary kappa on both-non-ABSTAIN (n={report['n_both_non_abstain']}): {report['kappa_binary_non_abstain']}")
    log(f"  disagreements: {len(report['disagreements'])} -> fill the final columns in {adjudication_path}")
    return 0


def build_reference(purpose: str, key: dict, a1: dict, a2: dict, adjudicated: dict) -> tuple[dict, list[str]]:
    """Reference label per item: annotator 1's label; on overlap items where the two annotators disagree,
    the adjudicated label (required). Returns (reference, problems)."""
    problems, items = [], []
    for k in key["items"]:
        i = k["item_id"]
        if i not in a1:
            problems.append(f"item {i}: no annotator-1 label")
            continue
        label, source = a1[i], "annotator1"
        if i in a2:
            if a2[i]["category"] == a1[i]["category"]:
                source = "agreed"
            elif i in adjudicated:
                label, source = adjudicated[i], "adjudicated"
            else:
                problems.append(f"item {i}: annotators disagree and no adjudicated label was given")
                continue
        items.append({"item_id": i, "category": label["category"], "severity": label.get("severity"),
                      "source": source})
    return {"purpose": purpose, "items": items}, problems


def cmd_finalize_reference(args) -> int:
    purpose, pdir = args.purpose, purpose_dir(args.purpose)
    manifest = load_manifest(purpose)
    key = load_verified_key(purpose, key_dir_from(args.key_dir))
    a1, problems = load_annotator(purpose, "annotator1", key)
    a2, adjudicated, hh = {}, {}, None
    if manifest.get("n_items_annotator2"):
        a2, p2 = load_annotator(purpose, "annotator2", key)
        problems += p2
        if not problems:
            hh = human_human(a1, a2)
            if hh["disagreements"]:
                adj_path = pdir / "adjudication.xlsx"
                if not adj_path.exists():
                    problems.append(f"{adj_path} not found -- run `human-kappa` and fill it")
                else:
                    adjudicated, p3 = read_adjudication_xlsx(adj_path)
                    problems += p3
    reference, p4 = ({}, []) if problems else build_reference(purpose, key, a1, a2, adjudicated)
    problems += p4
    if problems:
        log(f"[STOP] cannot finalize the reference -- {len(problems)} problem(s):")
        for p in problems:
            log(f"  - {p}")
        return 1
    reference.update({
        "status": "adjudicated", "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "key_sha256": manifest["key_sha256"], "n_items": len(reference["items"]),
        "human_human": None if hh is None else {k: hh[k] for k in (
            "n", "kappa_3cat", "observed_agreement_3cat", "n_both_non_abstain", "kappa_binary_non_abstain")},
        "n_adjudicated": sum(1 for it in reference["items"] if it["source"] == "adjudicated"),
    })
    out = pdir / "reference_adjudicated.json"
    out.write_text(json.dumps(reference, indent=2) + "\n")
    log(f"Reference with {reference['n_items']} items -> {out}")
    return 0


# ---------------------------------------------------------------------------
# judge-kappa
# ---------------------------------------------------------------------------

def judge_vs_human(reference: dict, key: dict, judge_records: list[dict]) -> dict:
    """{(prompt_version, judge_id): agreement} -- 3-category kappa on all reference items with a usable
    judge record, binary kappa on items BOTH label non-ABSTAIN. Items whose judge record is missing or
    unusable (parse error / failed call) are excluded and counted, never guessed."""
    human = {it["item_id"]: it["category"] for it in reference["items"]}
    key_by_item = {k["item_id"]: k for k in key["items"]}
    index: dict[tuple, dict] = {}
    for r in judge_records:
        index[(r.get("prompt_version"), r["judge_id"], r["run_id"], r["question_id"])] = r

    out = {}
    for version, judge_id in sorted({(k[0], k[1]) for k in index}, key=str):
        h_labels, j_labels, n_missing, n_unusable = [], [], 0, 0
        for item_id, h in sorted(human.items()):
            k = key_by_item[item_id]
            rec = index.get((version, judge_id, k["run_id"], k["question_id"]))
            if rec is None:
                n_missing += 1
                continue
            cat = category_of(rec)
            if cat is None:
                n_unusable += 1
                continue
            h_labels.append(h)
            j_labels.append(cat)
        rep = agreement_report(h_labels, j_labels) if h_labels else {"n": 0, "kappa_3cat": None}
        rep.update({"n_reference_items": len(human), "n_judge_missing": n_missing, "n_judge_unusable": n_unusable})
        out[(version, judge_id)] = rep
    return out


def _matrix_md(matrix: dict, cats) -> list[str]:
    lines = ["| human \\ judge | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for r in cats:
        lines.append(f"| {r} | " + " | ".join(str(matrix[r][c]) for c in cats) + " |")
    return lines


def judge_kappa_markdown(purpose: str, reference: dict, results: dict, deciding_judge: str = "primary") -> str:
    lines = [f"# Judge vs human agreement -- {purpose} set", "",
             f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
             f"Human reference: {reference['n_items']} items (status: {reference['status']}; "
             f"{reference.get('n_adjudicated', 0)} adjudicated).", ""]
    hh = reference.get("human_human")
    if hh:
        lines += [f"Human-human (pre-adjudication, n = {hh['n']}): 3-category kappa = {hh['kappa_3cat']}; "
                  f"binary kappa on both-non-ABSTAIN (n = {hh['n_both_non_abstain']}) = "
                  f"{hh['kappa_binary_non_abstain']}.", ""]
    lines += ["Acceptance (Amendment 4): unweighted 3-category Cohen's kappa (ABSTAIN / not hallucinated / "
              f"hallucinated), {deciding_judge} judge vs human reference, >= {ACCEPTANCE_KAPPA}. The binary kappa "
              "(hallucinated vs not, items both label non-ABSTAIN) is reported alongside and does not decide.", "",
              "| judge version | judge | n | 3-category kappa | acceptance | n both non-ABSTAIN | binary kappa | "
              "judge missing | judge unusable |", "|---|---|---|---|---|---|---|---|---|"]
    for (version, judge_id), r in results.items():
        k3 = r.get("kappa_3cat")
        accept = "-" if judge_id != deciding_judge else ("n/a" if k3 is None else "PASS" if k3 >= ACCEPTANCE_KAPPA else "below target")
        lines.append(f"| {version} | {judge_id} | {r['n']} | {k3} | {accept} | {r.get('n_both_non_abstain')} | "
                     f"{r.get('kappa_binary_non_abstain')} | {r['n_judge_missing']} | {r['n_judge_unusable']} |")
    for (version, judge_id), r in results.items():
        if not r["n"]:
            continue
        lines += ["", f"## {version} -- {judge_id} (n = {r['n']})", "", "3-category confusion matrix:", ""]
        lines += _matrix_md(r["confusion_3cat"], CATEGORIES)
        lines += ["", f"Binary confusion matrix, items both label non-ABSTAIN (n = {r['n_both_non_abstain']}):", ""]
        lines += _matrix_md(r["confusion_binary_non_abstain"], (CATEGORY_NOT_HALLUCINATED, CATEGORY_HALLUCINATED))
    return "\n".join(lines) + "\n"


def cmd_judge_kappa(args) -> int:
    from judge_agreement import load_judge_records

    purpose, pdir = args.purpose, purpose_dir(args.purpose)
    ref_path = pdir / "reference_adjudicated.json"
    if not ref_path.exists():
        raise AnnotationError(f"{ref_path} not found -- run `finalize-reference --purpose {purpose}` first")
    reference = json.loads(ref_path.read_text())
    key = load_verified_key(purpose, key_dir_from(args.key_dir))
    records = []
    for path in args.judge_output:  # one file at a time: a file never mixes judge versions, a set of files may
        records += load_judge_records([path])
    results = judge_vs_human(reference, key, records)
    out = Path(args.out) if args.out else pdir / "judge_vs_human_report.md"
    out.write_text(judge_kappa_markdown(purpose, reference, results))
    for (version, judge_id), r in results.items():
        log(f"{version} / {judge_id}: n={r['n']} kappa_3cat={r.get('kappa_3cat')} "
            f"binary(n={r.get('n_both_non_abstain')})={r.get('kappa_binary_non_abstain')}")
    log(f"Report -> {out}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--purpose", choices=sorted(PURPOSES), required=True)
        p.add_argument("--key-dir", default=None,
                       help=f"Directory of the key file, outside the repo (default: $GRAPHRAG_ANNOTATION_KEY_DIR "
                            f"or {DEFAULT_KEY_DIR})")

    p = sub.add_parser("export")
    common(p)
    p.add_argument("--run-files", nargs="*", default=[],
                   help="Run JSONL files to sample from (calibration: defaults to the calibration registry)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-total", type=int, default=None, help="confirmation only: total items (default 30)")
    p.add_argument("--questions-parquet", default=None)
    p.add_argument("--force", action="store_true", help="Overwrite an existing export (invalidates annotation in progress)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("validate")
    common(p)
    p.add_argument("--xlsx", required=True)
    p.add_argument("--annotator", choices=["annotator1", "annotator2"], default=None)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("human-kappa")
    common(p)
    p.add_argument("--force", action="store_true", help="Rewrite adjudication.xlsx even if it exists")
    p.set_defaults(func=cmd_human_kappa)

    p = sub.add_parser("finalize-reference")
    common(p)
    p.set_defaults(func=cmd_finalize_reference)

    p = sub.add_parser("judge-kappa")
    common(p)
    p.add_argument("--judge-output", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_judge_kappa)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except AnnotationError as e:
        log(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
