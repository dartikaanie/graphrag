"""
llm_judge_answer_relevance.py
=====================================
LLM-as-Judge post-hoc evaluation: ANSWER RELEVANCE, REFERENCE-FREE dan
INDEPENDEN dari kebenaran faktual jawaban. Mengisi dimensi "answer
relevance" dari 4 dimensi evaluasi RAG di Kamalipour, Asadi & Amiri Chimeh
(2026, Computer Science Review 61, 100925, §6.1.3). Berlaku untuk KEEMPAT
kondisi (A/B/C/D) -- metrik ini tidak butuh retrieved_context sama sekali.

KENAPA METRIK INI, PADAHAL llm_judge_hallucination.py SUDAH PUNYA
answer_relevance_score
------------------------------------------------------------
`answer_relevance_score` di llm_judge_hallucination.py dihasilkan DI
PROMPT YANG SAMA yang juga menunjukkan jawaban referensi (ground_truth_
answer) dan rubrik hallucination -- artinya skor itu REFERENCE-CONDITIONED
(judge sudah tahu "jawaban yang benar" sebelum menilai relevansi), bukan
pengukuran independen. Judge yang sudah membaca ground truth bisa secara
tidak sengaja mencampur "apakah jawaban ini menjawab pertanyaan" dengan
"apakah jawaban ini konsisten dengan ground truth" -- dua hal yang secara
konseptual berbeda (jawaban bisa sangat relevan tapi salah, atau benar tapi
tidak menjawab yang ditanyakan).

Metrik BARU di sini murni REFERENCE-FREE: judge HANYA diberi pertanyaan
(title+body) dan jawaban yang dinilai -- TIDAK ADA jawaban referensi, TIDAK
ADA retrieved_context, TIDAK ADA rubrik hallucination di prompt yang sama.
Ini metrik answer-relevance UTAMA untuk tesis ini; `answer_relevance_score`
dari judge hallucination TETAP ADA (script itu TIDAK diubah) tapi dibaca
sebagai sinyal sekunder/pelengkap, bukan diganti.

`llm_judge_hallucination.py` TIDAK diubah perilakunya sama sekali oleh
script ini -- keduanya independen, boleh dijalankan dalam urutan apa pun.

ISOLASI JUDGE (WAJIB, integritas evaluasi)
------------------------------------------------------------
Prompt TIDAK PERNAH menyebutkan condition/provider/model generator, TIDAK
PERNAH menyertakan retrieved_context atau ground_truth_answer -- judge
menilai relevansi MURNI dari isi pertanyaan vs isi jawaban.

KETERBATASAN METODOLOGIS
------------------------------------------------------------
- Sama seperti context-relevance judge: question body bisa tidak lengkap
  untuk sebagian question_id (lihat `body_available`).
- Ini penilaian LLM, bukan ground truth manusia -- rawan bias judge.
  Validasi via --kappa-validation dianjurkan.
- Karena TIDAK melihat retrieved_context maupun ground truth, metrik ini
  TIDAK bisa membedakan jawaban yang relevan-tapi-salah dari jawaban yang
  relevan-dan-benar -- itu memang bukan tugasnya (lihat Hallucination Rate
  dan Faithfulness Score untuk itu).

WAJIB DIPAKAI ULANG, TIDAK DIDUPLIKASI
------------------------------------------------------------
- get_llm_client()/call_llm_* dari llm/client_factory.py.
- Helper resume/output-path/HTML-stripping/kappa dari _judge_common.py.

SETUP .env (sama nama dgn judge lain)
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
    python llm_judge_answer_relevance.py \\
        --input-path ../a_pure_llm/results/condition_a_....jsonl --condition A

    python llm_judge_answer_relevance.py \\
        --input-path <file> --condition C --kappa-validation \\
        --secondary-judge-provider openai --secondary-judge-model gpt-4o \\
        --kappa-sample-size 5

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
    append_manifest,
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

ANSWER_RELEVANCE_LABELS = ("MENJAWAB", "MENJAWAB_SEBAGIAN", "TIDAK_MENJAWAB")
LABEL_TO_SCORE = {"MENJAWAB": 1.0, "MENJAWAB_SEBAGIAN": 0.5, "TIDAK_MENJAWAB": 0.0}


# ---------------------------------------------------------------------
# Prompt construction -- HANYA pertanyaan + jawaban. TIDAK ADA reference
# answer, TIDAK ADA retrieved_context, TIDAK ADA condition/provider/model.
# ---------------------------------------------------------------------

def build_answer_relevance_prompt(title: str, body_text: str, llm_answer: str) -> list[dict]:
    system_content = (
        "You are an impartial evaluator (LLM-as-judge) assessing ANSWER "
        "RELEVANCE for a software-engineering Q&A system -- whether an answer "
        "addresses what was actually asked, INDEPENDENT of whether the answer "
        "is factually correct. You are given ONLY the question and the answer "
        "-- no reference answer, no additional context. Do NOT assume or "
        "mention which system/model/condition produced this answer. Respond "
        "with ONLY a single JSON object, no markdown fences, no extra text."
    )
    question_section = f"Question title: {title}\n"
    if body_text:
        question_section += f"Question body: {body_text}\n"

    user_content = (
        f"{question_section}\n"
        f"Answer to evaluate:\n{llm_answer}\n\n"
        "Judge ONLY whether this answer addresses what the question asked -- "
        "do NOT judge whether it is factually correct (you have no reference "
        "to check correctness against, so do not try). Assign exactly one label:\n"
        "- \"MENJAWAB\": directly addresses the core of the question.\n"
        "- \"MENJAWAB_SEBAGIAN\": on topic but misses an important part of the "
        "question, or is too generic to be actionable.\n"
        "- \"TIDAK_MENJAWAB\": off-target, a refusal, or answers a different "
        "question than the one asked.\n\n"
        "Respond with ONLY this JSON object:\n"
        "{\n"
        '  "answer_relevance_label": "MENJAWAB" | "MENJAWAB_SEBAGIAN" | "TIDAK_MENJAWAB",\n'
        '  "justification": "<at most 2 short sentences>"\n'
        "}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _parse_answer_relevance_output(raw: str) -> dict | None:
    try:
        data = json.loads(strip_fences(raw))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    label = data.get("answer_relevance_label")
    if label not in ANSWER_RELEVANCE_LABELS:
        return None
    justification = data.get("justification")
    justification = justification if isinstance(justification, str) else ""
    return {"answer_relevance_label": label, "justification": justification}


# ---------------------------------------------------------------------
# Per-record judging
# ---------------------------------------------------------------------

def judge_answer_relevance_record(llm_client, call_llm_fn, judge_model: str, judge_temperature: float,
                                   record: dict, question_bodies: dict[int, str], max_body_chars: int) -> dict:
    qid = record.get("question_id")
    body_html = question_bodies.get(qid)
    body_available = bool(body_html and str(body_html).strip())
    body_text = html_to_text(body_html, max_body_chars) if body_available else ""
    title = record.get("title", "")
    llm_answer = record.get("llm_answer", "")

    messages = build_answer_relevance_prompt(title, body_text, llm_answer)
    raw = call_llm_fn(llm_client, messages, judge_model, temperature=judge_temperature)
    parsed = _parse_answer_relevance_output(raw)
    if parsed is None:
        raw = call_llm_fn(llm_client, messages, judge_model, temperature=judge_temperature)
        parsed = _parse_answer_relevance_output(raw)
    if parsed is None:
        return {
            "question_id": qid, "status": "failed", "body_available": body_available,
            "answer_relevance_label": None, "answer_relevance_score": None,
            "justification": None, "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

    label = parsed["answer_relevance_label"]
    return {
        "question_id": qid, "status": "ok", "body_available": body_available,
        "answer_relevance_label": label, "answer_relevance_score": LABEL_TO_SCORE[label],
        "justification": parsed["justification"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def _summary_from_output(output_path) -> dict:
    n = 0
    n_failed = 0
    labels = []
    scores = []
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
                if r.get("status") == "failed":
                    n_failed += 1
                    continue
                labels.append(r.get("answer_relevance_label"))
                if r.get("answer_relevance_score") is not None:
                    scores.append(r["answer_relevance_score"])

    counts = Counter(labels)
    n_ok = len(labels)
    return {
        "n_total": n,
        "n_failed": n_failed,
        "pct_menjawab": round(counts.get("MENJAWAB", 0) / n_ok * 100, 1) if n_ok else None,
        "pct_menjawab_sebagian": round(counts.get("MENJAWAB_SEBAGIAN", 0) / n_ok * 100, 1) if n_ok else None,
        "pct_tidak_menjawab": round(counts.get("TIDAK_MENJAWAB", 0) / n_ok * 100, 1) if n_ok else None,
        "mean_answer_relevance_score": round(statistics.mean(scores), 4) if scores else None,
    }


# ---------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------

def run_ansrel_batch(input_path, condition: str, judge_provider: str, judge_model: str, judge_temperature: float,
                      questions_parquet: str, max_body_chars: int, force: bool = False,
                      on_progress=None, check_cancel=None) -> dict:
    input_path = Path(input_path)
    output_path = build_output_path("ansrel", input_path, judge_provider, judge_model, judge_temperature)

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
                judged = judge_answer_relevance_record(
                    llm_client, call_llm_fn, judge_model, judge_temperature, record,
                    question_bodies, max_body_chars,
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

            log(f"      Id={qid} label={judged['answer_relevance_label']} "
                f"score={judged.get('answer_relevance_score')}")

            if on_progress:
                on_progress({"question_id": qid, "status": judged["status"]})

    summary = _summary_from_output(output_path)
    summary["n_total"] = n_total
    summary["output_path"] = str(output_path)
    summary["n_evaluated_baru"] = n_evaluated_baru
    summary["n_dari_cache"] = n_dari_cache
    return summary


# ---------------------------------------------------------------------
# Cohen's Kappa validation
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


def run_ansrel_kappa_validation(input_path, condition: str, judge_provider: str, judge_model: str,
                                 judge_temperature: float, secondary_judge_provider: str,
                                 secondary_judge_model: str, secondary_judge_temperature: float,
                                 questions_parquet: str, max_body_chars: int, kappa_sample_size: int,
                                 seed: int, force: bool = False) -> dict:
    input_path = Path(input_path)

    primary_summary = run_ansrel_batch(
        input_path, condition, judge_provider, judge_model, judge_temperature,
        questions_parquet, max_body_chars, force=force,
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

    tmp_path = input_path.parent / f"{input_path.stem}__ansrel_kappa_subsample_{uuid.uuid4().hex[:8]}.jsonl"
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

        secondary_summary = run_ansrel_batch(
            tmp_path, condition, secondary_judge_provider, secondary_judge_model, secondary_judge_temperature,
            questions_parquet, max_body_chars, force=True,
        )
        secondary_output_path = Path(secondary_summary["output_path"])
    finally:
        tmp_path.unlink(missing_ok=True)

    primary_by_qid = {r["question_id"]: r for r in _read_jsonl(primary_output_path) if r.get("question_id") in sample_ids}
    secondary_by_qid = {r["question_id"]: r for r in _read_jsonl(secondary_output_path)}

    pairs = []
    for qid in sorted(sample_ids, key=lambda x: (x is None, x)):
        p = primary_by_qid.get(qid)
        s = secondary_by_qid.get(qid)
        if p is None or s is None or p.get("status") != "ok" or s.get("status") != "ok":
            continue
        pairs.append((p["answer_relevance_label"], s["answer_relevance_label"]))

    kappa_value, interpretation = (None, None)
    if pairs:
        kappa_value, interpretation = cohens_kappa(
            [a for a, _ in pairs], [b for _, b in pairs], list(ANSWER_RELEVANCE_LABELS),
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kappa_out_path = RESULTS_DIR / f"ansrel_kappa_validation_{condition}_{ts}.json"
    kappa_result = {
        "n_sample": len(sample_ids),
        "n_pairs": len(pairs),
        "kappa_value": round(kappa_value, 4) if kappa_value is not None else None,
        "interpretation": interpretation,
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
    parser.add_argument("--input-path", required=True, help="File JSONL hasil Kondisi A/B/C/D yang sudah selesai")
    parser.add_argument("--condition", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--judge-provider", default=os.getenv("JUDGE_PROVIDER", "openai"),
                         choices=["openai", "anthropic", "local", "ollama"])
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--judge-temperature", type=float, default=float(os.getenv("JUDGE_TEMPERATURE", 0.1)))
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    parser.add_argument("--max-body-chars", type=int, default=3000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--kappa-validation", action="store_true")
    parser.add_argument("--secondary-judge-provider", default=os.getenv("SECONDARY_JUDGE_PROVIDER"))
    parser.add_argument("--secondary-judge-model", default=os.getenv("SECONDARY_JUDGE_MODEL"))
    parser.add_argument("--kappa-sample-size", type=int, default=int(os.getenv("KAPPA_SAMPLE_SIZE", 50)))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
        f"temperature={args.judge_temperature} force={args.force}")

    kappa_result = None
    if args.kappa_validation:
        kappa_result = run_ansrel_kappa_validation(
            input_path, args.condition, args.judge_provider, args.judge_model, args.judge_temperature,
            args.secondary_judge_provider, args.secondary_judge_model, args.judge_temperature,
            args.questions_parquet, args.max_body_chars, args.kappa_sample_size, args.seed, force=args.force,
        )
        summary = kappa_result["primary_summary"]
    else:
        summary = run_ansrel_batch(
            input_path, args.condition, args.judge_provider, args.judge_model, args.judge_temperature,
            args.questions_parquet, args.max_body_chars, force=args.force,
        )

    output_path = summary["output_path"]

    log("\n" + "=" * 70)
    log(f"RINGKASAN ANSWER-RELEVANCE JUDGE (reference-free) -- Kondisi {args.condition}")
    log("=" * 70)
    log(f"Total pertanyaan        : {summary.get('n_total')}")
    log(f"Baru dievaluasi         : {summary.get('n_evaluated_baru')}")
    log(f"Dari cache (sudah ada)  : {summary.get('n_dari_cache')}")
    log(f"failed (parse/validasi) : {summary.get('n_failed')}")
    log(f"% MENJAWAB              : {summary.get('pct_menjawab', '-')}%")
    log(f"% MENJAWAB_SEBAGIAN     : {summary.get('pct_menjawab_sebagian', '-')}%")
    log(f"% TIDAK_MENJAWAB        : {summary.get('pct_tidak_menjawab', '-')}%")
    log(f"Mean answer_relevance_score : {summary.get('mean_answer_relevance_score')}")
    log(f"\nHasil judge tersimpan -> {output_path}")

    if kappa_result is not None:
        log(f"\nCohen's Kappa: {kappa_result['kappa_value']} ({kappa_result['interpretation']}) "
            f"[n_pairs={kappa_result['n_pairs']}]")
        log(f"[kappa] Hasil lengkap disimpan -> {kappa_result['kappa_json_path']}")

    manifest_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "condition": args.condition,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "judge_temperature": args.judge_temperature,
        "n_total": summary.get("n_total"),
        "n_evaluated_baru": summary.get("n_evaluated_baru"),
        "n_dari_cache": summary.get("n_dari_cache"),
        "n_failed": summary.get("n_failed"),
        "pct_menjawab": summary.get("pct_menjawab"),
        "mean_answer_relevance_score": summary.get("mean_answer_relevance_score"),
    }
    if kappa_result is not None:
        manifest_record["kappa_value"] = kappa_result.get("kappa_value")

    history_path = append_manifest("answer_relevance_run_history.jsonl", manifest_record)
    log(f"[logging] Manifest run answer-relevance dicatat -> {history_path}")


if __name__ == "__main__":
    sys.exit(main())
