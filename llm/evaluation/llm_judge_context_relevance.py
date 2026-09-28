"""
llm_judge_context_relevance.py
=====================================
LLM-as-Judge post-hoc evaluation of RETRIEVAL RELEVANCE (reference-free,
per-item) dan CONTEXT SUFFICIENCY (reference-aware, per-question) untuk
Kondisi B/C/D. Mengisi dimensi "retrieval relevance" dari 4 dimensi
evaluasi RAG di Kamalipour, Asadi & Amiri Chimeh (2026, Computer Science
Review 61, 100925, §6.1) yang SEBELUMNYA hanya dicek lewat proxy tag-
overlap (Jaccard) di analyze_retrieval_quality.py -- bukan penilaian isi
teks yang sesungguhnya.

KENAPA LLM JUDGE, BUKAN COSINE SIMILARITY QUERY<->CONTEXT
------------------------------------------------------------
Untuk Kondisi B, cosine similarity antara query dan context akan SIRKULER
-- FAISS SUDAH me-ranking top-k berdasarkan cosine similarity itu sendiri,
jadi "mengevaluasi" retrieval B dengan metrik yang SAMA PERSIS dipakai B
untuk retrieve tidak memberi informasi independen apa pun (skornya akan
selalu tinggi by construction). Kondisi C/D juga tidak murni cosine-based
(ada graph traversal + trust weighting), jadi metrik yang sama untuk semua
kondisi harus independen dari mekanisme retrieval masing-masing -- itulah
LLM-as-judge yang menilai isi teks, bukan skor vektor.

DUA PANGGILAN LLM TERPISAH PER PERTANYAAN (SENGAJA, bukan 1 panggilan)
------------------------------------------------------------
1. Item relevance (REFERENCE-FREE): judge menilai tiap item HANYA dari
   apakah isinya membantu menjawab pertanyaan -- TIDAK diberi jawaban
   referensi sama sekali, supaya label per-item murni dari
   pertanyaan+konten, tidak terkontaminasi oleh mengetahui "jawaban yang
   benar" lebih dulu (yang bisa membuat judge menilai item relevan HANYA
   karena kebetulan menyebut istilah yang sama dengan jawaban referensi).
2. Context sufficiency (REFERENCE-AWARE): judge menilai APAKAH seluruh
   context (digabung) cukup untuk menghasilkan solusi inti jawaban
   referensi -- ini SENGAJA memakai jawaban referensi, karena pertanyaan
   yang dijawab beda ("apakah keseluruhan konteks cukup" bukan "apakah
   item ini relevan").
   Kalau digabung jadi satu panggilan, skor per-item bisa bocor dari
   melihat jawaban referensi terlebih dulu.

ISOLASI JUDGE DARI METADATA RETRIEVAL (WAJIB, integritas evaluasi)
------------------------------------------------------------
Prompt HANYA berisi chunk_text tiap item (+ index posisi [1]..[k]) --
TIDAK PERNAH trust_weight/combined_score/score/relevance_score/
source_stage/hop/rel_type/is_accepted, dan TIDAK PERNAH condition/
provider/model generator. Ini supaya judge tidak bisa "curang" menilai
item relevan hanya karena tahu itu di-rank tinggi oleh trust-weighting
Kondisi C, dan supaya perbandingan B vs C vs D independen dari mekanisme
ranking masing-masing.

KETERBATASAN METODOLOGIS (WAJIB dibaca sebelum dipakai di Bab V)
------------------------------------------------------------
- Question body dari QUESTIONS_PARQUET bisa TIDAK LENGKAP untuk sebagian
  question_id (lihat `body_available` per record) -- kalau tidak
  ditemukan, judge hanya diberi title, hasil tetap disimpan (bukan
  di-skip) tapi WAJIB dibaca dengan catatan ini.
- Item di dalam satu pertanyaan TIDAK independen satu sama lain (mereka
  semua respons terhadap pertanyaan yang sama) -- analisis statistik
  lanjutan (lihat compare_conditions_stats.py) beroperasi di level
  PERTANYAAN (agregat per question_id), bukan di level item mentah,
  justru karena alasan ini.
- Sama seperti judge hallucination: ini penilaian LLM, bukan ground
  truth manusia -- rawan bias judge (self-preference, sensitivitas
  posisi/redaksi). Validasi via --kappa-validation dianjurkan.
- "SEBAGIAN" (di kedua skema label item & sufficiency) sengaja dipakai
  sebagai kelas tengah eksplisit, bukan pemaksaan biner relevan/tidak --
  konsisten dengan skema 3-kelas hallucination_label di judge lama.

WAJIB DIPAKAI ULANG, TIDAK DIDUPLIKASI
------------------------------------------------------------
- get_llm_client()/call_llm_* dari llm/client_factory.py.
- Helper resume/output-path/HTML-stripping/kappa dari _judge_common.py.

SETUP .env (opsional -- semua override-able via CLI, sama nama dgn judge lama)
----------------------------------------------------------------------------
    JUDGE_PROVIDER=openai
    JUDGE_MODEL=gpt-4o-mini
    JUDGE_TEMPERATURE=0.1
    SECONDARY_JUDGE_PROVIDER=openai
    SECONDARY_JUDGE_MODEL=gpt-4o
    KAPPA_SAMPLE_SIZE=50
    QUESTIONS_PARQUET=...

CARA PAKAI
----------
    python llm_judge_context_relevance.py \\
        --input-path ../c_graphrag/results/condition_c_....jsonl --condition C

    # + validasi Cohen's Kappa (item-level DAN sufficiency-level)
    python llm_judge_context_relevance.py \\
        --input-path <file> --condition C --kappa-validation \\
        --secondary-judge-provider openai --secondary-judge-model gpt-4o \\
        --kappa-sample-size 5

    # skip panggilan sufficiency (hemat 1 panggilan/pertanyaan), hanya item relevance
    python llm_judge_context_relevance.py --input-path <file> --condition C --skip-sufficiency

Resumable secara alami (append-only), pola SAMA dengan llm_judge_hallucination.py.
"""

import argparse
import json
import os
import random
import statistics
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm.client_factory import get_llm_client  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _judge_common import (  # noqa: E402
    RESULTS_DIR,
    build_output_path,
    cohens_kappa,
    html_to_text,
    input_question_ids,
    is_complete,
    load_already_done,
    load_question_bodies,
    strip_fences,
    warn_if_self_judging,
)

log = print

ITEM_LABELS = ("RELEVAN", "SEBAGIAN", "TIDAK_RELEVAN")
SUFFICIENCY_LABELS = ("CUKUP", "SEBAGIAN", "TIDAK_CUKUP")
RETRIEVAL_CONDITIONS = ("B", "C", "D")


# ---------------------------------------------------------------------
# Prompt construction -- HANYA chunk_text per item, TIDAK PERNAH field
# metadata retrieval (lihat catatan isolasi di docstring modul).
# ---------------------------------------------------------------------

def _format_items_block(retrieved: list) -> str:
    lines = []
    for i, item in enumerate(retrieved, start=1):
        lines.append(f"[{i}] {item.get('chunk_text', '')}")
    return "\n\n".join(lines)


def build_item_relevance_prompt(title: str, body_text: str, retrieved: list) -> list[dict]:
    k = len(retrieved)
    items_block = _format_items_block(retrieved)
    system_content = (
        "You are an impartial evaluator (LLM-as-judge) assessing RETRIEVAL "
        "RELEVANCE for a software-engineering Q&A retrieval system. You will "
        "receive a question and a numbered list of retrieved text snippets. "
        "You do NOT know which system retrieved them, what order/score they "
        "were ranked by, or any metadata attached to them -- judge purely from "
        "whether each snippet's CONTENT would help answer the question. "
        "Respond with ONLY a single JSON object, no markdown fences, no extra text."
    )
    question_section = f"Question title: {title}\n"
    if body_text:
        question_section += f"Question body: {body_text}\n"

    user_content = (
        f"{question_section}\n"
        f"Retrieved items ({k} total, in their original order):\n{items_block}\n\n"
        "For EACH item, assign exactly one label:\n"
        "- \"RELEVAN\": contains information that helps answer this question.\n"
        "- \"SEBAGIAN\": same topic or technology as the question, but does not "
        "address the core of what is being asked.\n"
        "- \"TIDAK_RELEVAN\": unrelated to the question, or only superficially "
        "similar (shares keywords but is about a different problem).\n\n"
        f"Respond with ONLY this JSON object, with EXACTLY {k} entries in "
        f"\"items\" (idx must cover 1..{k}, each exactly once):\n"
        "{\n"
        '  "items": [\n'
        '    {"idx": 1, "label": "RELEVAN" | "SEBAGIAN" | "TIDAK_RELEVAN", "reason": "<=15 words"},\n'
        "    ...\n"
        "  ]\n"
        "}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_sufficiency_prompt(title: str, body_text: str, retrieved: list, ground_truth_text: str) -> list[dict]:
    items_block = "\n\n".join(item.get("chunk_text", "") for item in retrieved)
    system_content = (
        "You are an impartial evaluator (LLM-as-judge) assessing whether a set "
        "of retrieved text snippets, TAKEN TOGETHER, contains the information "
        "needed to produce a given reference answer's key solution. You do NOT "
        "know which system retrieved them or any ranking/score metadata. "
        "Respond with ONLY a single JSON object, no markdown fences, no extra text."
    )
    question_section = f"Question title: {title}\n"
    if body_text:
        question_section += f"Question body: {body_text}\n"

    user_content = (
        f"{question_section}\n"
        f"Retrieved context (all items concatenated):\n{items_block}\n\n"
        f"Reference (ground-truth) answer:\n{ground_truth_text}\n\n"
        "Does the retrieved context above, taken as a whole, contain the "
        "information needed to produce this reference answer's key solution? "
        "Assign exactly one label:\n"
        "- \"CUKUP\": the context contains what is needed for the key solution.\n"
        "- \"SEBAGIAN\": the context has some relevant information but misses an "
        "important part of the key solution.\n"
        "- \"TIDAK_CUKUP\": the context does not contain what is needed for the "
        "key solution.\n\n"
        "Respond with ONLY this JSON object:\n"
        "{\n"
        '  "sufficiency": "CUKUP" | "SEBAGIAN" | "TIDAK_CUKUP",\n'
        '  "justification": "<at most 2 short sentences>"\n'
        "}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------
# Output parsing -- item-count & label validation (retry sekali, lalu
# gagal -- TIDAK DITEBAK, sesuai instruksi eksplisit).
# ---------------------------------------------------------------------

def _parse_items_output(raw: str, k: int) -> list[dict] | None:
    try:
        data = json.loads(strip_fences(raw))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    items = data.get("items")
    if not isinstance(items, list) or len(items) != k:
        return None

    by_idx = {}
    for it in items:
        if not isinstance(it, dict):
            return None
        idx = it.get("idx")
        label = it.get("label")
        if not isinstance(idx, int) or idx in by_idx:
            return None
        if label not in ITEM_LABELS:
            return None
        reason = it.get("reason")
        reason = reason if isinstance(reason, str) else ""
        by_idx[idx] = {"idx": idx, "label": label, "reason": reason}

    if set(by_idx.keys()) != set(range(1, k + 1)):
        return None
    return [by_idx[i] for i in range(1, k + 1)]


def _parse_sufficiency_output(raw: str) -> dict | None:
    try:
        data = json.loads(strip_fences(raw))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    sufficiency = data.get("sufficiency")
    if sufficiency not in SUFFICIENCY_LABELS:
        return None
    justification = data.get("justification")
    justification = justification if isinstance(justification, str) else ""
    return {"sufficiency": sufficiency, "justification": justification}


# ---------------------------------------------------------------------
# Per-record judging
# ---------------------------------------------------------------------

def judge_context_relevance_record(llm_client, call_llm_fn, judge_model: str, judge_temperature: float,
                                    record: dict, question_bodies: dict[int, str], max_body_chars: int,
                                    skip_sufficiency: bool) -> dict:
    qid = record.get("question_id")
    retrieved = record.get("retrieved_context") or []
    now = datetime.now(timezone.utc).isoformat()

    body_html = question_bodies.get(qid)
    body_available = bool(body_html and str(body_html).strip())
    body_text = html_to_text(body_html, max_body_chars) if body_available else ""
    title = record.get("title", "")

    if not retrieved:
        return {
            "question_id": qid, "status": "no_context", "n_items": 0,
            "body_available": body_available, "item_labels": [],
            "context_precision_strict": None, "context_precision_lenient": None,
            "first_relevant_rank": None, "reciprocal_rank": None,
            "sufficiency": None, "sufficiency_justification": None,
            "evaluated_at": now,
        }

    k = len(retrieved)
    messages1 = build_item_relevance_prompt(title, body_text, retrieved)
    raw = call_llm_fn(llm_client, messages1, judge_model, temperature=judge_temperature)
    parsed_items = _parse_items_output(raw, k)
    if parsed_items is None:
        raw = call_llm_fn(llm_client, messages1, judge_model, temperature=judge_temperature)
        parsed_items = _parse_items_output(raw, k)
    if parsed_items is None:
        return {
            "question_id": qid, "status": "failed", "n_items": k,
            "body_available": body_available, "item_labels": [],
            "context_precision_strict": None, "context_precision_lenient": None,
            "first_relevant_rank": None, "reciprocal_rank": None,
            "sufficiency": None, "sufficiency_justification": None,
            "evaluated_at": now,
        }

    item_labels = []
    for it in parsed_items:
        src = retrieved[it["idx"] - 1]
        item_labels.append({
            "idx": it["idx"],
            "answer_id": src.get("answer_id"),
            "question_id": src.get("question_id"),
            "label": it["label"],
            "reason": it["reason"],
        })

    n_relevan = sum(1 for it in item_labels if it["label"] == "RELEVAN")
    n_sebagian = sum(1 for it in item_labels if it["label"] == "SEBAGIAN")
    context_precision_strict = n_relevan / k
    context_precision_lenient = (n_relevan + 0.5 * n_sebagian) / k
    first_relevant_rank = next((it["idx"] for it in item_labels if it["label"] == "RELEVAN"), None)
    reciprocal_rank = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0

    sufficiency = None
    sufficiency_justification = None
    if not skip_sufficiency:
        gt_text = html_to_text(record.get("ground_truth_answer", ""), max_body_chars)
        messages2 = build_sufficiency_prompt(title, body_text, retrieved, gt_text)
        raw2 = call_llm_fn(llm_client, messages2, judge_model, temperature=judge_temperature)
        parsed_suff = _parse_sufficiency_output(raw2)
        if parsed_suff is None:
            raw2 = call_llm_fn(llm_client, messages2, judge_model, temperature=judge_temperature)
            parsed_suff = _parse_sufficiency_output(raw2)
        if parsed_suff is not None:
            sufficiency = parsed_suff["sufficiency"]
            sufficiency_justification = parsed_suff["justification"]

    return {
        "question_id": qid, "status": "ok", "n_items": k,
        "body_available": body_available, "item_labels": item_labels,
        "context_precision_strict": round(context_precision_strict, 4),
        "context_precision_lenient": round(context_precision_lenient, 4),
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": round(reciprocal_rank, 4),
        "sufficiency": sufficiency,
        "sufficiency_justification": sufficiency_justification,
        "evaluated_at": now,
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def _summary_from_output(output_path) -> dict:
    n = 0
    n_no_context = 0
    n_failed = 0
    precisions_strict, precisions_lenient, rrs = [], [], []
    suff_counts: Counter = Counter()

    output_path = Path(output_path)
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                status = r.get("status")
                if status == "no_context":
                    n_no_context += 1
                    continue
                if status == "failed":
                    n_failed += 1
                    continue
                if r.get("context_precision_strict") is not None:
                    precisions_strict.append(r["context_precision_strict"])
                if r.get("context_precision_lenient") is not None:
                    precisions_lenient.append(r["context_precision_lenient"])
                if r.get("reciprocal_rank") is not None:
                    rrs.append(r["reciprocal_rank"])
                if r.get("sufficiency"):
                    suff_counts[r["sufficiency"]] += 1

    return {
        "n_total": n,
        "n_no_context": n_no_context,
        "n_failed": n_failed,
        "precision_strict_mean": round(statistics.mean(precisions_strict), 4) if precisions_strict else None,
        "precision_strict_median": round(statistics.median(precisions_strict), 4) if precisions_strict else None,
        "precision_lenient_mean": round(statistics.mean(precisions_lenient), 4) if precisions_lenient else None,
        "precision_lenient_median": round(statistics.median(precisions_lenient), 4) if precisions_lenient else None,
        "mrr_mean": round(statistics.mean(rrs), 4) if rrs else None,
        "mrr_median": round(statistics.median(rrs), 4) if rrs else None,
        "sufficiency_distribution": dict(suff_counts),
    }


# ---------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------

def run_ctxrel_batch(input_path, condition: str, judge_provider: str, judge_model: str, judge_temperature: float,
                      questions_parquet: str, max_body_chars: int, skip_sufficiency: bool, force: bool = False,
                      on_progress=None, check_cancel=None) -> dict:
    input_path = Path(input_path)
    output_path = build_output_path("ctxrel", input_path, judge_provider, judge_model, judge_temperature)

    if force and output_path.exists():
        output_path.unlink()

    if not force and is_complete(input_path, output_path):
        summary = _summary_from_output(output_path)
        summary["output_path"] = str(output_path)
        summary["n_evaluated_baru"] = 0
        summary["n_dari_cache"] = summary["n_total"]
        if on_progress:
            on_progress({"question_id": None, "status": "already_complete"})
        log(f"      [already_complete] {output_path} sudah lengkap ({summary['n_total']} pertanyaan) "
            f"-- 0 panggilan LLM baru.")
        return summary

    already_done = load_already_done(output_path)
    if already_done:
        log(f"      [resume] {len(already_done)} pertanyaan sudah dinilai sebelumnya -> {output_path}")

    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    todo_ids = [r.get("question_id") for r in records if r.get("question_id") not in already_done]
    question_bodies = load_question_bodies(questions_parquet, todo_ids)

    warn_if_self_judging(records[:20], judge_model)

    llm_client, call_llm_fn = get_llm_client(judge_provider, judge_model, log=log)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_evaluated_baru = 0
    n_dari_cache = 0

    with open(output_path, "a") as f_out:
        for record in records:
            if check_cancel is not None and check_cancel():
                log(f"      [cancelled] Dihentikan setelah {n_evaluated_baru} pertanyaan baru dievaluasi.")
                break

            qid = record.get("question_id")
            n_total += 1

            if qid in already_done:
                n_dari_cache += 1
                if on_progress:
                    on_progress({"question_id": qid, "status": "cached"})
                continue

            try:
                judged = judge_context_relevance_record(
                    llm_client, call_llm_fn, judge_model, judge_temperature, record,
                    question_bodies, max_body_chars, skip_sufficiency,
                )
            except Exception as e:
                log(f"      Id={qid} [FAIL] {e}")
                if on_progress:
                    on_progress({"question_id": qid, "status": "failed", "error": str(e)})
                continue

            try:
                line_out = json.dumps(judged, default=str)
            except Exception as e:
                log(f"      Id={qid} [FAIL-WRITE] {e}")
                if on_progress:
                    on_progress({"question_id": qid, "status": "failed", "error": str(e)})
                continue

            f_out.write(line_out + "\n")
            f_out.flush()
            already_done.add(qid)
            n_evaluated_baru += 1

            log(f"      Id={qid} status={judged['status']} n_items={judged['n_items']} "
                f"precision_strict={judged.get('context_precision_strict')} "
                f"sufficiency={judged.get('sufficiency')}")

            if on_progress:
                on_progress({"question_id": qid, "status": judged["status"]})

    summary = _summary_from_output(output_path)
    summary["n_total"] = n_total
    summary["output_path"] = str(output_path)
    summary["n_evaluated_baru"] = n_evaluated_baru
    summary["n_dari_cache"] = n_dari_cache
    return summary


# ---------------------------------------------------------------------
# Cohen's Kappa validation -- DUA nilai: item-level (flattened item
# labels, 3 kelas) dan question-level (sufficiency, 3 kelas berbeda set).
# ---------------------------------------------------------------------

def _read_jsonl(path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def run_ctxrel_kappa_validation(input_path, condition: str, judge_provider: str, judge_model: str,
                                 judge_temperature: float, secondary_judge_provider: str,
                                 secondary_judge_model: str, secondary_judge_temperature: float,
                                 questions_parquet: str, max_body_chars: int, skip_sufficiency: bool,
                                 kappa_sample_size: int, seed: int, force: bool = False) -> dict:
    input_path = Path(input_path)

    primary_summary = run_ctxrel_batch(
        input_path, condition, judge_provider, judge_model, judge_temperature,
        questions_parquet, max_body_chars, skip_sufficiency, force=force,
    )
    primary_output_path = Path(primary_summary["output_path"])

    all_ids = input_question_ids(input_path)
    rng = random.Random(seed)
    sample_ids = set(all_ids if len(all_ids) <= kappa_sample_size else rng.sample(all_ids, kappa_sample_size))

    judge_models_identical = (judge_provider, judge_model) == (secondary_judge_provider, secondary_judge_model)
    if judge_models_identical:
        print(
            f"PERINGATAN: judge utama dan judge kedua memakai model yang identik "
            f"({judge_provider}/{judge_model}) -- validasi Cohen's Kappa dengan judge "
            f"yang sama tidak mendeteksi self-enhancement bias, hanya mengukur "
            f"variance internal model itu sendiri.",
            file=sys.stderr,
        )

    tmp_path = input_path.parent / f"{input_path.stem}__ctxrel_kappa_subsample_{uuid.uuid4().hex[:8]}.jsonl"
    try:
        with open(input_path) as f_in, open(tmp_path, "w") as f_out:
            for line in f_in:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if rec.get("question_id") in sample_ids:
                    f_out.write(stripped + "\n")

        secondary_summary = run_ctxrel_batch(
            tmp_path, condition, secondary_judge_provider, secondary_judge_model, secondary_judge_temperature,
            questions_parquet, max_body_chars, skip_sufficiency, force=True,
        )
        secondary_output_path = Path(secondary_summary["output_path"])
    finally:
        tmp_path.unlink(missing_ok=True)

    primary_by_qid = {r["question_id"]: r for r in _read_jsonl(primary_output_path) if r.get("question_id") in sample_ids}
    secondary_by_qid = {r["question_id"]: r for r in _read_jsonl(secondary_output_path)}

    item_pairs: list[tuple[str, str]] = []
    suff_pairs: list[tuple[str, str]] = []
    for qid in sorted(sample_ids, key=lambda x: (x is None, x)):
        p = primary_by_qid.get(qid)
        s = secondary_by_qid.get(qid)
        if p is None or s is None:
            continue
        if p.get("status") == "ok" and s.get("status") == "ok":
            p_items = {it["idx"]: it["label"] for it in p.get("item_labels", [])}
            s_items = {it["idx"]: it["label"] for it in s.get("item_labels", [])}
            for idx in sorted(set(p_items) & set(s_items)):
                item_pairs.append((p_items[idx], s_items[idx]))
        if p.get("sufficiency") is not None and s.get("sufficiency") is not None:
            suff_pairs.append((p["sufficiency"], s["sufficiency"]))

    item_kappa, item_interp = (None, None)
    if item_pairs:
        item_kappa, item_interp = cohens_kappa([a for a, _ in item_pairs], [b for _, b in item_pairs], list(ITEM_LABELS))

    suff_kappa, suff_interp = (None, None)
    if suff_pairs:
        suff_kappa, suff_interp = cohens_kappa([a for a, _ in suff_pairs], [b for _, b in suff_pairs], list(SUFFICIENCY_LABELS))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kappa_out_path = RESULTS_DIR / f"ctxrel_kappa_validation_{condition}_{ts}.json"
    kappa_result = {
        "n_sample": len(sample_ids),
        "n_item_pairs": len(item_pairs),
        "item_level_kappa": round(item_kappa, 4) if item_kappa is not None else None,
        "item_level_interpretation": item_interp,
        "n_sufficiency_pairs": len(suff_pairs),
        "sufficiency_kappa": round(suff_kappa, 4) if suff_kappa is not None else None,
        "sufficiency_interpretation": suff_interp,
        "judge_models_identical": judge_models_identical,
        "primary_output_path": str(primary_output_path),
        "secondary_output_path": str(secondary_output_path),
    }
    kappa_out_path.write_text(json.dumps(kappa_result, indent=2, default=str))
    kappa_result["kappa_json_path"] = str(kappa_out_path)
    kappa_result["primary_summary"] = primary_summary
    return kappa_result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", required=True, help="File JSONL hasil Kondisi B/C/D yang sudah selesai")
    parser.add_argument("--condition", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--judge-provider", default=os.getenv("JUDGE_PROVIDER", "openai"),
                         choices=["openai", "anthropic", "local", "ollama"])
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--judge-temperature", type=float, default=float(os.getenv("JUDGE_TEMPERATURE", 0.1)))
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    parser.add_argument("--max-body-chars", type=int, default=3000,
                         help="Batas panjang (karakter, bukan token) question body/ground_truth setelah "
                              "HTML-stripping yang dikirim ke prompt judge (default 3000).")
    parser.add_argument("--skip-sufficiency", action="store_true",
                         help="Lewati panggilan ke-2 (context sufficiency) -- hemat 1 panggilan LLM/pertanyaan, "
                              "hanya menghasilkan item_labels + precision/MRR, sufficiency selalu null.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--kappa-validation", action="store_true")
    parser.add_argument("--secondary-judge-provider", default=os.getenv("SECONDARY_JUDGE_PROVIDER"))
    parser.add_argument("--secondary-judge-model", default=os.getenv("SECONDARY_JUDGE_MODEL"))
    parser.add_argument("--kappa-sample-size", type=int, default=int(os.getenv("KAPPA_SAMPLE_SIZE", 50)))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.condition == "A":
        log("[ERROR] Kondisi A tidak melakukan retrieval (tidak ada retrieved_context) -- "
            "context-relevance judge tidak berlaku untuk Kondisi A. Gunakan --condition B/C/D.")
        sys.exit(1)

    if not args.questions_parquet:
        log("[ERROR] QUESTIONS_PARQUET harus diset di .env atau lewat --questions-parquet")
        sys.exit(1)

    if args.kappa_validation and not (args.secondary_judge_provider and args.secondary_judge_model):
        log("[ERROR] --secondary-judge-provider dan --secondary-judge-model wajib diisi kalau "
            "--kappa-validation aktif (atau set SECONDARY_JUDGE_PROVIDER/SECONDARY_JUDGE_MODEL di .env).")
        sys.exit(1)

    input_path = Path(args.input_path)
    if not input_path.exists():
        log(f"[ERROR] --input-path tidak ditemukan: {input_path}")
        sys.exit(1)

    log(f"[config] condition={args.condition} judge={args.judge_provider}/{args.judge_model} "
        f"temperature={args.judge_temperature} skip_sufficiency={args.skip_sufficiency} force={args.force}")

    kappa_result = None
    if args.kappa_validation:
        kappa_result = run_ctxrel_kappa_validation(
            input_path, args.condition, args.judge_provider, args.judge_model, args.judge_temperature,
            args.secondary_judge_provider, args.secondary_judge_model, args.judge_temperature,
            args.questions_parquet, args.max_body_chars, args.skip_sufficiency,
            args.kappa_sample_size, args.seed, force=args.force,
        )
        summary = kappa_result["primary_summary"]
    else:
        summary = run_ctxrel_batch(
            input_path, args.condition, args.judge_provider, args.judge_model, args.judge_temperature,
            args.questions_parquet, args.max_body_chars, args.skip_sufficiency, force=args.force,
        )

    output_path = summary["output_path"]

    log("\n" + "=" * 70)
    log(f"RINGKASAN CONTEXT-RELEVANCE JUDGE -- Kondisi {args.condition}")
    log("=" * 70)
    log(f"Total pertanyaan        : {summary.get('n_total')}")
    log(f"Baru dievaluasi         : {summary.get('n_evaluated_baru')}")
    log(f"Dari cache (sudah ada)  : {summary.get('n_dari_cache')}")
    log(f"no_context (0 item)     : {summary.get('n_no_context')}")
    log(f"failed (parse/validasi) : {summary.get('n_failed')}")
    log(f"Precision@k strict  mean/median : {summary.get('precision_strict_mean')}/{summary.get('precision_strict_median')}")
    log(f"Precision@k lenient mean/median : {summary.get('precision_lenient_mean')}/{summary.get('precision_lenient_median')}")
    log(f"MRR                 mean/median : {summary.get('mrr_mean')}/{summary.get('mrr_median')}")
    log(f"Distribusi sufficiency  : {summary.get('sufficiency_distribution')}")
    log(f"\nHasil judge tersimpan -> {output_path}")

    if kappa_result is not None:
        log(f"\nItem-level Cohen's Kappa   : {kappa_result['item_level_kappa']} ({kappa_result['item_level_interpretation']}) "
            f"[n_pairs={kappa_result['n_item_pairs']}]")
        log(f"Sufficiency Cohen's Kappa  : {kappa_result['sufficiency_kappa']} ({kappa_result['sufficiency_interpretation']}) "
            f"[n_pairs={kappa_result['n_sufficiency_pairs']}]")
        log(f"[kappa] Hasil lengkap disimpan -> {kappa_result['kappa_json_path']}")

    manifest_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "condition": args.condition,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "judge_temperature": args.judge_temperature,
        "skip_sufficiency": args.skip_sufficiency,
        "n_total": summary.get("n_total"),
        "n_evaluated_baru": summary.get("n_evaluated_baru"),
        "n_dari_cache": summary.get("n_dari_cache"),
        "n_no_context": summary.get("n_no_context"),
        "n_failed": summary.get("n_failed"),
        "precision_strict_mean": summary.get("precision_strict_mean"),
        "mrr_mean": summary.get("mrr_mean"),
    }
    if kappa_result is not None:
        manifest_record["item_level_kappa"] = kappa_result.get("item_level_kappa")
        manifest_record["sufficiency_kappa"] = kappa_result.get("sufficiency_kappa")

    from _judge_common import append_manifest
    history_path = append_manifest("context_relevance_run_history.jsonl", manifest_record)
    log(f"[logging] Manifest run context-relevance dicatat -> {history_path}")


if __name__ == "__main__":
    sys.exit(main())
