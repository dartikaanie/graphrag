"""
ingest_claude_judge.py
----------------------
Reads Claude's pasted judge-v2.3 responses (one JSON object per row_id), parses each with
judge_prompt_v2_3.parse_judgment_v2_3(raw, has_context, offered_items) -- has_context/offered_items rebuilt from the
ORIGINAL item exactly as the official judges got them -- and writes per-judge records in the schema of the official
records to results/sensitivity_claude/<condition>.jsonl (one file per condition A-D).

Accepted input (a file, or stdin with "-"; objects may be a JSON array, JSON lines, or simply concatenated):
  {"row_id": "CJ-123456", "response": {...judge JSON...}}          response may also be a JSON string
  {"row_id": "CJ-123456", "answer_attempted": ..., "claims": ...}  the judge JSON itself with row_id added
An object may carry "model" (or "judge_model") for its own model name; otherwise --model applies to the whole call.
The model name is required (one of the two), since you report it per batch.

Re-ingesting a row_id REPLACES its earlier record (corrected paste); row_ids not in claude_judge/key.csv are rejected.
Official results are never touched; output goes only to results/sensitivity_claude/.

    python llm/evaluation/ingest_claude_judge.py pasted_batch_001.txt --model "claude-opus-5-5"
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

import build_claude_judge_inputs as bci  # noqa: E402

JUDGE_ID = "claude"
PROVIDER = "claude.ai chat"
PANEL_ID = "panel-v2-claude-sensitivity"
REGISTRY_STATUS = "sensitivity"
OUT_DIR = ROOT / "results" / "sensitivity_claude"
KEY = ROOT / "claude_judge" / "key.csv"


def iter_json_objects(text: str):
    """Yield every top-level JSON object in `text` (array elements, JSON lines or concatenated objects)."""
    dec, i, n = json.JSONDecoder(), 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        if text[i] == "[":
            arr, j = dec.raw_decode(text, i)
            yield from (a for a in arr if isinstance(a, dict))
            i = j
        else:
            obj, j = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                yield obj
            i = j


def split_response(obj: dict):
    """-> (row_id, raw judge text, model or None)."""
    row_id = obj["row_id"]
    model = obj.get("model") or obj.get("judge_model")
    if "response" in obj:
        resp = obj["response"]
        raw = resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)
    else:
        raw = json.dumps({k: v for k, v in obj.items() if k not in ("row_id", "model", "judge_model")},
                         ensure_ascii=False)
    return row_id, raw, model


def make_record(item: dict, template_keys: list, raw: str, model: str, parse) -> dict:
    from judge_binary import derive_hallucinated
    from judge_panel import _PARSE_FIELDS
    from judge_prompt_v2_2 import _load_json

    parsed = parse(raw, item["has_context"], item["offered_items"])
    ok = not parsed.get("parse_error", False)
    rec = {k: None for k in template_keys}  # same field set as the official records; what a chat has not -> null
    rec.update({
        "panel_id": PANEL_ID, "item_id": item["item_id"], "run_id": item["run_id"],
        "question_id": item["question_id"], "condition": item["condition"], "run_label": item["run_label"],
        "config_hash": item["config_hash"], "source_file": item["source_file"],
        "judge_id": JUDGE_ID, "judge_model": model, "provider": PROVIDER,
        "prompt_version": bci.PROMPT_VERSION, "blinding_version": "blind-v3", "parse_version": "pv2.3-1",
        "temperature": None, "max_tokens": None, "request_response_format": None, "request_extra_body": None,
        "thinking_disabled_requested": None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "messages_sha256": item["messages_sha256"],
        "has_context": item["has_context"], "n_context_items": item["n_context_items"],
        "citation_map": item["citation_map"],
        "call_failed": False, "parse_ok": ok, "parse_error": not ok, "error": None if ok else "parse_error",
        "attempts": [], "raw_response": raw, "parsed_json": _load_json(raw),
        "normalized_fields": parsed.get("normalized_fields", []) if ok else [],
        **{f: (parsed.get(f) if ok else None) for f in _PARSE_FIELDS},
        "registry_status": REGISTRY_STATUS,
    })
    rec["hallucinated"] = derive_hallucinated({**rec, "parse_error": not ok})
    return rec


def load_key() -> dict:
    with open(KEY, encoding="utf-8") as f:
        return {r["row_id"]: r for r in csv.DictReader(f)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="file(s) with the pasted JSON responses, or - for stdin")
    ap.add_argument("--model", help="model name reported for this paste (objects may carry their own 'model')")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--dry-run", action="store_true", help="parse and report, write nothing")
    args = ap.parse_args(argv)

    import judge_prompt_v2_3 as v23
    key = load_key()
    items, judge_files, official_by_item, _ = bci.build_items()
    by_item = {it["item_id"]: it for it in items}
    template_keys = list(next(iter(official_by_item.values())).keys())

    new, problems, seen = {}, [], set()
    for src in args.inputs:
        text = sys.stdin.read() if src == "-" else Path(src).read_text(encoding="utf-8")
        for obj in iter_json_objects(text):
            if "row_id" not in obj:
                problems.append(f"{src}: object without row_id (keys {sorted(obj)[:4]})")
                continue
            row_id, raw, model = split_response(obj)
            model = model or args.model
            if not model:
                problems.append(f"{row_id}: no model name (pass --model or add 'model' to the object)")
                continue
            if row_id not in key:
                problems.append(f"{row_id}: not in key.csv")
                continue
            if row_id in seen:
                problems.append(f"{row_id}: appears twice in this paste; the later one is used")
            seen.add(row_id)
            new[row_id] = make_record(by_item[key[row_id]["item_id"]], template_keys, raw, model,
                                      v23.parse_judgment_v2_3)

    out = Path(args.out_dir)
    n_parse_err = sum(1 for r in new.values() if r["parse_error"])
    per_cond = {}
    for rid, rec in new.items():
        per_cond.setdefault(rec["condition"], {})[rec["item_id"]] = rec
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        for cond, recs in per_cond.items():
            path = out / f"{cond}.jsonl"
            existing = {}
            if path.exists():
                existing = {r["item_id"]: r for r in bci.read_jsonl(path)}
            existing.update(recs)  # re-ingest replaces
            with open(path, "w", encoding="utf-8") as f:
                for r in existing.values():
                    f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
    total = {c: len(bci.read_jsonl(out / f"{c}.jsonl")) if (out / f"{c}.jsonl").exists() else 0 for c in bci.RUN_LABELS}
    print(f"{'(dry run) ' if args.dry_run else ''}ingested {len(new)} response(s); parse errors: {n_parse_err}")
    print("records on disk per condition: " + ", ".join(f"{c}={n}/384" for c, n in total.items()))
    for p in problems:
        print("  WARNING:", p)
    for rid, rec in new.items():
        if rec["parse_error"]:
            print(f"  parse error: {rid} ({rec['item_id']})")
    return 1 if problems and not new else 0


if __name__ == "__main__":
    sys.exit(main())
