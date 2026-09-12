"""
llm_judge_hallucination.py
=====================================
LLM-as-Judge post-hoc evaluation: Faithfulness, Answer Relevance, and a
3-class Hallucination Rate, plus inter-judge agreement validation via
Cohen's Kappa.

REVISI -- OUTPUT SEBAGAI FILE TERPISAH (bukan merge ke file kondisi asli)
------------------------------------------------------------
Versi sebelumnya menulis record GABUNGAN (field generator + field judge)
ke `judged_{nama_asli}.jsonl`. Versi ini menulis file BARU TERPISAH ke
llm/evaluation/results/, berisi HANYA identifier (question_id) + hasil
judge -- TIDAK menyalin ulang field record asli. Nama file deterministik
dari kombinasi (file sumber, provider judge, model judge, temperature,
majority_rounds) via build_judge_output_path() -- run dengan config
IDENTIK akan mendeteksi "sudah lengkap" dan TIDAK memanggil LLM sama
sekali; run dengan config BERBEDA (mis. --majority-rounds lain) otomatis
menulis ke file BARU, tidak pernah menimpa hasil judge yang lain.
llm/evaluation/logs/judge_run_history.jsonl adalah manifest APPEND-ONLY
yang mencatat SETIAP kali script ini dijalankan (termasuk saat hasilnya
"already_complete") -- dipakai backend utk join hasil judge ke halaman
History yang sudah ada (lihat backend/app/services/judge_lookup_service.py).

TUJUAN -- baca file JSONL hasil Kondisi A/B/C/D yang SUDAH SELESAI dijalankan
(question_id, title, llm_answer, ground_truth_answer, retrieved_context, ...)
dan tambahkan penilaian LLM-as-judge DI ATASNYA. Script ini TIDAK me-re-run
generator (a_baseline_replication.py/b_condition_b_rag.py/c_graphrag.py/
d_lightrag.py) sama sekali, dan TIDAK PERNAH menulis ke input_path -- murni
membaca record yang sudah ada.

DUA MODE PENILAIAN (dipilih otomatis dari --condition)
------------------------------------------------------------
- "no_context" (Kondisi A): tidak ada retrieval sama sekali, jadi
  faithfulness TIDAK APPLICABLE (selalu null). Judge menilai
  hallucination_label murni dari kesesuaian llm_answer vs
  ground_truth_answer, memakai PENGETAHUAN UMUM judge sendiri -- tidak ada
  konteks eksternal untuk mengecek groundedness-nya. Ini setara secara
  metodologis dengan "Fabricated Claim Rate" (pola sama dengan catatan NF2
  Kondisi A yang sudah ada di c_graphrag.py: "Fabricated Citation Rate").
- "context_grounded" (Kondisi B/C/D): faithfulness dinilai terhadap
  retrieved_context yang SAMA PERSIS sudah dipakai generator saat membuat
  jawaban (dibaca ulang dari field `retrieved_context` di record JSONL
  asal -- TIDAK di-retrieve ulang, sengaja, supaya judge menilai apa yang
  BENAR-BENAR dilihat generator, bukan retrieval baru yang bisa berbeda).

KETERBATASAN DATA YANG WAJIB DIKETAHUI: record JSONL kondisi mana pun
TIDAK menyimpan Body pertanyaan asli (hanya title, tags, ground_truth_answer,
llm_answer, retrieved_context) -- lihat skema di
a_baseline_replication.py/b_condition_b_rag.py/c_graphrag.py/d_lightrag.py.
Judge di sini karena itu menilai dari title + llm_answer + ground_truth +
(kalau ada) retrieved_context saja, TANPA body pertanyaan asli. Ini
keterbatasan data yang WAJIB didokumentasikan di Bab V, bukan bug di script
ini -- memperbaikinya butuh mengubah skema record kondisi generator (di luar
scope permintaan ini).

WAJIB DIPAKAI ULANG, TIDAK DIDUPLIKASI
------------------------------------------------------------
- get_llm_client()/call_llm_* dari llm/client_factory.py -- cara memanggil
  LLM SAMA PERSIS dengan kondisi A/B/C/D. call_llm_* menerima parameter
  opsional `temperature: float | None = None` (diteruskan ke API provider
  HANYA kalau bukan None) -- generator A/B/C/D yang tidak mengirim
  parameter ini SAMA SEKALI TIDAK BERUBAH PERILAKUNYA.

ISOLASI JUDGE DARI IDENTITAS GENERATOR (WAJIB, integritas evaluasi)
------------------------------------------------------------
Prompt yang dikirim ke judge TIDAK PERNAH menyebutkan condition/provider/
model generator -- judge hanya melihat pertanyaan, jawaban yang dinilai,
dan (kalau ada) konteks yang diberikan ke penjawab, supaya penilaian murni
dari ISI TEKS, bukan bias "condition C pasti lebih baik karena pakai KG".

INSTALL DEPENDENCY (tambahan)
-------------------
    pip install scikit-learn   # cohen_kappa_score, utk --kappa-validation

SETUP .env (opsional -- semua override-able via CLI)
----------------------------------------------------------------------------
    JUDGE_PROVIDER=openai
    JUDGE_MODEL=gpt-4o-mini
    JUDGE_TEMPERATURE=0.1
    JUDGE_MAJORITY_ROUNDS=1
    SECONDARY_JUDGE_PROVIDER=openai
    SECONDARY_JUDGE_MODEL=gpt-4o
    KAPPA_SAMPLE_SIZE=50

CARA PAKAI
----------
    # Judge biasa, 1 putaran per pertanyaan -- output path SELALU otomatis
    python llm_judge_hallucination.py \\
        --input-path ../c_graphrag/results/condition_c_openai_gpt-4o-mini_n30_seed42_fw0-7-0-3.jsonl \\
        --condition C

    # Jalankan LAGI dgn config identik -> "already_complete", 0 panggilan LLM baru
    python llm_judge_hallucination.py --input-path <file sama> --condition C

    # Config BERBEDA (mis. majority_rounds) -> file BARU, tidak menimpa yang lama
    python llm_judge_hallucination.py --input-path <file sama> --condition C --majority-rounds 3

    # Paksa nilai ulang dari nol utk kombinasi source+config yang SAMA
    python llm_judge_hallucination.py --input-path <file sama> --condition C --force

    # + validasi Cohen's Kappa pada subsample kecil (biaya API murah)
    python llm_judge_hallucination.py \\
        --input-path ../c_graphrag/results/condition_c_openai_gpt-4o-mini_n30_seed42_fw0-7-0-3.jsonl \\
        --condition C --kappa-validation \\
        --secondary-judge-provider openai --secondary-judge-model gpt-4o \\
        --kappa-sample-size 5

Resumable secara alami (append-only): output_path dibaca dulu kalau sudah
ada -- question_id yang sudah dinilai di-skip (status "cached"), tidak
dinilai ulang, proses bisa dihentikan (Ctrl+C) dan dilanjutkan kapan saja
dengan command yang SAMA PERSIS tanpa kehilangan progres.
"""

import argparse
import json
import os
import random
import re
import statistics
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm.client_factory import get_llm_client  # noqa: E402

log = print
LOG_DIR = Path("logs")
RESULTS_DIR = Path("results")

VALID_LABELS = ("FAKTUAL", "HALUSINASI_SEBAGIAN", "HALUSINASI_PENUH")

CONDITION_MODE = {
    "A": "no_context",
    "B": "context_grounded",
    "C": "context_grounded",
    "D": "context_grounded",
}


# ---------------------------------------------------------------------
# Output path -- deterministik dari (file sumber, config judge), supaya
# lookup lewat manifest (judge_run_history.jsonl) tetap konsisten dan run
# dgn config berbeda tidak pernah saling menimpa.
# ---------------------------------------------------------------------

def build_judge_output_path(input_path, judge_provider: str, judge_model: str,
                             judge_temperature: float, majority_rounds: int) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    safe_model = re.sub(r"[^A-Za-z0-9]+", "-", judge_model).strip("-")
    temp_str = str(judge_temperature).replace(".", "-")
    filename = f"judged__{stem}__{judge_provider}-{safe_model}_t{temp_str}_mr{majority_rounds}.jsonl"
    return RESULTS_DIR / filename


def append_judge_run_history(condition: str, input_path, output_path, judge_provider: str,
                              judge_model: str, judge_temperature: float, majority_rounds: int,
                              summary: dict, kappa_result: dict | None = None) -> Path:
    """Manifest APPEND-ONLY -- satu baris per eksekusi script ini (termasuk
    saat status "already_complete"), dipakai backend utk join hasil judge
    ke halaman History yang sudah ada by input_path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    history_path = LOG_DIR / "judge_run_history.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "judge_temperature": judge_temperature,
        "majority_rounds": majority_rounds,
        "n_total": summary.get("n_total"),
        "n_evaluated_baru": summary.get("n_evaluated_baru"),
        "n_dari_cache": summary.get("n_dari_cache"),
        "pct_faktual": summary.get("pct_faktual"),
        "pct_sebagian": summary.get("pct_halusinasi_sebagian"),
        "pct_penuh": summary.get("pct_halusinasi_penuh"),
        "mean_faithfulness": summary.get("mean_faithfulness_score"),
        "mean_answer_relevance": summary.get("mean_answer_relevance_score"),
    }
    if kappa_result is not None:
        record["kappa_value"] = kappa_result.get("kappa_value")
        record["kappa_interpretation"] = kappa_result.get("interpretation")
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return history_path


# ---------------------------------------------------------------------
# Prompt construction -- TERPISAH dari llm/prompts.py karena ini prompt
# untuk MENILAI, bukan untuk menjawab pertanyaan. TIDAK DIUBAH dari
# implementasi sebelumnya.
# ---------------------------------------------------------------------

def _format_context_block(retrieved_context: list | None) -> str:
    if not retrieved_context:
        return ""
    lines = []
    for c in retrieved_context:
        qid = c.get("question_id")
        label = f"[SO-{qid}]" if qid is not None else "[SO-?]"
        lines.append(f"{label} {c.get('chunk_text', '')}")
    return "\n\n".join(lines)


def build_judge_prompt(question_title: str, question_body: str, llm_answer: str,
                        ground_truth_answer: str, retrieved_context: list | None,
                        mode: str) -> list[dict]:
    """Bangun 2-turn prompt (system + user) untuk judge. TIDAK PERNAH
    menyertakan condition/provider/model generator di ISI TEKS -- lihat
    catatan integritas evaluasi di docstring modul ini.

    `mode`:
      - "context_grounded": judge memeriksa SETIAP klaim di llm_answer,
        cek didukung retrieved_context (faithfulness) DAN konsisten dengan
        ground_truth_answer (hallucination_label): FAKTUAL kalau didukung
        konteks DAN konsisten ground truth; HALUSINASI_SEBAGIAN kalau
        sebagian klaim tidak didukung/tidak konsisten; HALUSINASI_PENUH
        kalau mayoritas/seluruh klaim tidak didukung/bertentangan.
      - "no_context": tidak ada retrieved_context untuk dicek, jadi
        faithfulness_score WAJIB null; hallucination_label HANYA dari
        kesesuaian llm_answer vs ground_truth_answer + pengetahuan umum
        judge sendiri. answer_relevance_score tetap dihitung di kedua mode.
    """
    system_content = (
        "You are an impartial evaluator (LLM-as-judge) for a software-engineering "
        "Q&A system. You will receive a question, an answer to be evaluated, and "
        "(if available) the context that was given to whoever produced that answer. "
        "Do NOT assume or mention which system, model, or experimental condition "
        "produced this answer -- evaluate purely from the text content given to you. "
        "Respond with ONLY a single JSON object, no markdown fences, no extra text."
    )

    context_block = _format_context_block(retrieved_context) if mode == "context_grounded" else ""

    question_section = f"Question title: {question_title}\n"
    if question_body:
        question_section += f"Question body: {question_body}\n"

    if mode == "context_grounded":
        task_instructions = (
            "Evaluate the answer below in two steps:\n"
            "1. FAITHFULNESS -- check EVERY factual claim in the answer against the "
            "given context. faithfulness_score (0.0-1.0) is the fraction of claims "
            "that ARE supported by the context (1.0 = every claim is grounded in the "
            "context, 0.0 = no claim is supported by it).\n"
            "2. HALLUCINATION LABEL -- considering BOTH how well the answer is "
            "supported by the context AND how consistent it is with the reference "
            "(ground-truth) answer, assign exactly one label:\n"
            "   - \"FAKTUAL\": claims are supported by the context AND consistent "
            "with the reference answer.\n"
            "   - \"HALUSINASI_SEBAGIAN\": some claims are unsupported by the "
            "context or inconsistent with the reference answer, but not most of it.\n"
            "   - \"HALUSINASI_PENUH\": most or all claims are unsupported by the "
            "context or contradict the reference answer.\n\n"
            f"Context given to the answerer:\n{context_block if context_block else '(no context was retrieved for this question)'}\n\n"
        )
    else:
        task_instructions = (
            "No external context was given to whoever produced this answer (this is "
            "a no-retrieval condition). Evaluate the answer using your own general "
            "knowledge:\n"
            "1. FAITHFULNESS is not applicable here (there is no given context to "
            "check groundedness against) -- you MUST return faithfulness_score as "
            "JSON null.\n"
            "2. HALLUCINATION LABEL -- based on your own knowledge and the reference "
            "(ground-truth) answer, judge how factually correct the answer is:\n"
            "   - \"FAKTUAL\": consistent with the reference answer and your own knowledge.\n"
            "   - \"HALUSINASI_SEBAGIAN\": partially incorrect or partially "
            "inconsistent with the reference answer.\n"
            "   - \"HALUSINASI_PENUH\": mostly or entirely incorrect or contradicts "
            "the reference answer / your own knowledge.\n\n"
        )

    user_content = (
        f"{question_section}\n"
        f"{task_instructions}"
        f"Reference (ground-truth) answer:\n{ground_truth_answer}\n\n"
        f"Answer to evaluate:\n{llm_answer}\n\n"
        "Also assign answer_relevance_score (0.0-1.0): how directly and completely "
        "the answer addresses the question asked, regardless of factual correctness.\n\n"
        "Respond with ONLY this JSON object (no markdown fences):\n"
        "{\n"
        '  "hallucination_label": "FAKTUAL" | "HALUSINASI_SEBAGIAN" | "HALUSINASI_PENUH",\n'
        '  "faithfulness_score": <float 0.0-1.0, or null>,\n'
        '  "answer_relevance_score": <float 0.0-1.0>,\n'
        '  "justification": "<at most 2 short sentences>"\n'
        "}"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------
# Judge output parsing
# ---------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


def _parse_judge_output(raw: str, mode: str) -> dict | None:
    """Return the validated dict, or None if parsing/validation failed
    (caller retries once, then gives up)."""
    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    label = data.get("hallucination_label")
    if label not in VALID_LABELS:
        return None

    faithfulness = data.get("faithfulness_score")
    if mode == "no_context":
        faithfulness = None  # enforced regardless of what the judge returned
    elif faithfulness is not None:
        try:
            faithfulness = float(faithfulness)
        except (TypeError, ValueError):
            return None

    relevance = data.get("answer_relevance_score")
    try:
        relevance = float(relevance)
    except (TypeError, ValueError):
        return None

    justification = data.get("justification")
    justification = justification if isinstance(justification, str) else ""

    return {
        "hallucination_label": label,
        "faithfulness_score": faithfulness,
        "answer_relevance_score": relevance,
        "justification": justification,
    }


# ---------------------------------------------------------------------
# Per-record judging -- LOGIKA TIDAK DIUBAH dari implementasi sebelumnya
# (mode selection, retry sekali, majority voting pakai modus, temperature
# judge terpisah dari generator); field output disederhanakan (tanpa
# prefix judge_) karena file ini sekarang TERPISAH, tidak lagi digabung
# dengan field record generator.
# ---------------------------------------------------------------------

def judge_record(llm_client, call_llm_fn, judge_model: str, judge_temperature: float,
                  record: dict, condition: str) -> dict:
    """Panggil judge SEKALI utk satu record. Parse JSON, retry sekali kalau
    gagal parse/validasi. Raise ValueError kalau tetap gagal setelah retry."""
    mode = CONDITION_MODE[condition]
    messages = build_judge_prompt(
        question_title=record.get("title", ""),
        question_body=record.get("body", ""),  # lihat catatan keterbatasan data di docstring modul
        llm_answer=record.get("llm_answer", ""),
        ground_truth_answer=record.get("ground_truth_answer", ""),
        retrieved_context=record.get("retrieved_context") if mode == "context_grounded" else None,
        mode=mode,
    )

    raw = call_llm_fn(llm_client, messages, judge_model, temperature=judge_temperature)
    parsed = _parse_judge_output(raw, mode)
    if parsed is None:
        raw = call_llm_fn(llm_client, messages, judge_model, temperature=judge_temperature)
        parsed = _parse_judge_output(raw, mode)
    if parsed is None:
        raise ValueError(f"Judge output tidak valid (gagal parse/validasi setelah retry): {raw!r}")

    parsed["question_id"] = record.get("question_id")
    return parsed


def judge_record_majority(llm_client, call_llm_fn, judge_model: str, judge_temperature: float,
                           record: dict, condition: str, majority_rounds: int = 1) -> dict:
    """majority_rounds > 1: panggil judge_record() berkali-kali PADA
    TEMPERATURE YANG SAMA (variasi berasal dari randomness LLM bawaan, bukan
    dari menaikkan temperature), ambil label MODUS/mayoritas (BUKAN mean,
    BUKAN best-of -- majority voting yang terbukti efektif di literatur),
    dan median untuk skor numerik. Seluruh hasil per-putaran mentah
    disimpan di all_rounds untuk audit. majority_rounds == 1: panggil
    judge_record() sekali saja, tidak ada overhead tambahan (all_rounds
    TIDAK disertakan)."""
    if majority_rounds <= 1:
        return judge_record(llm_client, call_llm_fn, judge_model, judge_temperature, record, condition)

    rounds = [
        judge_record(llm_client, call_llm_fn, judge_model, judge_temperature, record, condition)
        for _ in range(majority_rounds)
    ]

    labels = [r["hallucination_label"] for r in rounds]
    majority_label = Counter(labels).most_common(1)[0][0]

    faithfulness_vals = [r["faithfulness_score"] for r in rounds if r["faithfulness_score"] is not None]
    faithfulness_median = statistics.median(faithfulness_vals) if faithfulness_vals else None
    relevance_median = statistics.median([r["answer_relevance_score"] for r in rounds])

    return {
        "question_id": record.get("question_id"),
        "hallucination_label": majority_label,
        "faithfulness_score": faithfulness_median,
        "answer_relevance_score": relevance_median,
        "justification": rounds[0]["justification"],
        "all_rounds": rounds,
    }


# ---------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------

def load_already_judged(output_path) -> set:
    """question_id yang SUDAH punya hasil di output_path (kalau ada) --
    dipakai utk resume, TANPA perlu tahu total baris di input_path."""
    done: set = set()
    output_path = Path(output_path)
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["question_id"])
            except Exception:
                continue
    return done


def _input_question_ids(input_path) -> list:
    """Ambil HANYA field question_id per baris input_path (generator,
    tidak accumulate full record) -- dipakai is_judge_complete() dan
    subsample kappa validation."""
    ids = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.append(json.loads(line)["question_id"])
            except Exception:
                continue
    return ids


def is_judge_complete(input_path, output_path) -> bool:
    already = load_already_judged(output_path)
    if not already:
        return False
    input_ids = set(_input_question_ids(input_path))
    return input_ids.issubset(already)


def _summary_from_output(output_path) -> dict[str, Any]:
    """Distribusi label & mean skor dibaca ULANG dari output_path (bukan
    dari state di memori), supaya akurat termasuk hasil dari run
    sebelumnya (resume/already_complete)."""
    labels = []
    faithfulness_vals = []
    relevance_vals = []
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
                labels.append(r.get("hallucination_label"))
                if r.get("faithfulness_score") is not None:
                    faithfulness_vals.append(r["faithfulness_score"])
                if r.get("answer_relevance_score") is not None:
                    relevance_vals.append(r["answer_relevance_score"])
    n = len(labels)
    counts = Counter(labels)
    return {
        "n_total": n,
        "pct_faktual": round(counts.get("FAKTUAL", 0) / n * 100, 1) if n else None,
        "pct_halusinasi_sebagian": round(counts.get("HALUSINASI_SEBAGIAN", 0) / n * 100, 1) if n else None,
        "pct_halusinasi_penuh": round(counts.get("HALUSINASI_PENUH", 0) / n * 100, 1) if n else None,
        "mean_faithfulness_score": round(statistics.mean(faithfulness_vals), 4) if faithfulness_vals else None,
        "mean_answer_relevance_score": round(statistics.mean(relevance_vals), 4) if relevance_vals else None,
    }


# ---------------------------------------------------------------------
# Batch runner -- dipakai CLI main() DAN dashboard backend/engine_service.
# Output path SELALU dihitung otomatis (build_judge_output_path), file
# BARU TERPISAH dari input_path -- TIDAK PERNAH menulis ke input_path.
# ---------------------------------------------------------------------

def run_judge_batch(input_path, judge_provider: str, judge_model: str, judge_temperature: float,
                     majority_rounds: int, condition: str, force: bool = False,
                     on_progress=None, check_cancel=None) -> dict:
    input_path = Path(input_path)
    output_path = build_judge_output_path(input_path, judge_provider, judge_model, judge_temperature, majority_rounds)

    if force and output_path.exists():
        output_path.unlink()

    if not force and is_judge_complete(input_path, output_path):
        summary = _summary_from_output(output_path)
        summary["output_path"] = str(output_path)
        summary["n_evaluated_baru"] = 0
        summary["n_dari_cache"] = summary["n_total"]
        if on_progress:
            on_progress({"question_id": None, "status": "already_complete"})
        log(f"      [already_complete] {output_path} sudah lengkap ({summary['n_total']} pertanyaan) "
            f"-- 0 panggilan LLM baru.")
        return summary

    already_done = load_already_judged(output_path)
    if already_done:
        log(f"      [resume] {len(already_done)} pertanyaan sudah dinilai sebelumnya -> {output_path}")

    llm_client, call_llm_fn = get_llm_client(judge_provider, judge_model, log=log)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_evaluated_baru = 0
    n_dari_cache = 0

    with open(input_path) as f_in, open(output_path, "a") as f_out:
        for line in f_in:
            if check_cancel is not None and check_cancel():
                log(f"      [cancelled] Dihentikan setelah {n_evaluated_baru} pertanyaan baru dievaluasi.")
                break

            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            qid = record.get("question_id")
            n_total += 1

            if qid in already_done:
                n_dari_cache += 1
                if on_progress:
                    on_progress({"question_id": qid, "status": "cached"})
                continue

            try:
                judged = judge_record_majority(
                    llm_client, call_llm_fn, judge_model, judge_temperature, record, condition, majority_rounds,
                )
            except Exception as e:
                log(f"      Id={qid} [FAIL] {e}")
                if on_progress:
                    on_progress({"question_id": qid, "status": "failed", "error": str(e)})
                continue

            judged["evaluated_at"] = datetime.now(timezone.utc).isoformat()
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

            log(f"      Id={qid} label={judged['hallucination_label']} "
                f"faithfulness={judged['faithfulness_score']} "
                f"relevance={judged['answer_relevance_score']:.2f}")

            if on_progress:
                on_progress({"question_id": qid, "status": "evaluated",
                             "hallucination_label": judged["hallucination_label"]})

    summary = _summary_from_output(output_path)
    summary["n_total"] = n_total
    summary["output_path"] = str(output_path)
    summary["n_evaluated_baru"] = n_evaluated_baru
    summary["n_dari_cache"] = n_dari_cache
    return summary


# ---------------------------------------------------------------------
# Cohen's Kappa validation
# ---------------------------------------------------------------------

def compute_cohens_kappa(primary_labels: list[str], secondary_labels: list[str]) -> tuple[float, str]:
    """3-kelas Cohen's Kappa via sklearn. Interpretasi memakai 4 pita yang
    diminta secara eksplisit (< 0.4 lemah, 0.4-0.6 moderat, 0.6-0.8 kuat,
    > 0.8 sangat kuat) -- ini penyederhanaan dari 6 pita asli Landis & Koch
    (1977), dipakai apa adanya sesuai spesifikasi."""
    from sklearn.metrics import cohen_kappa_score

    kappa = float(cohen_kappa_score(primary_labels, secondary_labels, labels=list(VALID_LABELS)))
    if kappa < 0.4:
        interpretation = "lemah (poor agreement)"
    elif kappa < 0.6:
        interpretation = "moderat (moderate agreement)"
    elif kappa < 0.8:
        interpretation = "kuat (substantial agreement)"
    else:
        interpretation = "sangat kuat (almost perfect agreement)"
    return kappa, interpretation


def run_kappa_validation(input_path, judge_provider: str, judge_model: str, judge_temperature: float,
                          secondary_judge_provider: str, secondary_judge_model: str,
                          secondary_judge_temperature: float, majority_rounds: int, condition: str,
                          kappa_sample_size: int, seed: int, force: bool = False) -> dict:
    """1. Jalankan run_judge_batch() utk judge utama (file lengkap).
    2. Tulis subsample acak (seed tetap) question_id dari input_path ke
       file SEMENTARA, jalankan run_judge_batch() KEDUA dgn judge kedua
       HANYA atas subsample itu (biaya API kecil) -- file sementara
       dihapus setelah selesai, TIDAK disimpan permanen.
    3. compute_cohens_kappa() dari kedua output (join by question_id).
    4. Simpan ringkasan ke
       llm/evaluation/results/kappa_validation_{condition}_{timestamp}.json
       (mereferensikan kedua output_path, bukan menduplikasi datanya).
    """
    input_path = Path(input_path)

    primary_summary = run_judge_batch(
        input_path, judge_provider, judge_model, judge_temperature, majority_rounds, condition, force=force,
    )
    primary_output_path = Path(primary_summary["output_path"])

    all_ids = _input_question_ids(input_path)
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

    tmp_path = input_path.parent / f"{input_path.stem}__kappa_subsample_{uuid.uuid4().hex[:8]}.jsonl"
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

        secondary_summary = run_judge_batch(
            tmp_path, secondary_judge_provider, secondary_judge_model, secondary_judge_temperature,
            majority_rounds, condition, force=True,
        )
        secondary_output_path = Path(secondary_summary["output_path"])
    finally:
        tmp_path.unlink(missing_ok=True)

    primary_by_qid = {}
    with open(primary_output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("question_id") in sample_ids:
                primary_by_qid[r["question_id"]] = r

    secondary_by_qid = {}
    with open(secondary_output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            secondary_by_qid[r["question_id"]] = r

    comparison = []
    for qid in sorted(sample_ids, key=lambda x: (x is None, x)):
        p = primary_by_qid.get(qid)
        s = secondary_by_qid.get(qid)
        if p is None or s is None:
            continue
        comparison.append({
            "question_id": qid,
            "primary_label": p["hallucination_label"],
            "secondary_label": s["hallucination_label"],
            "agree": p["hallucination_label"] == s["hallucination_label"],
        })

    primary_labels = [c["primary_label"] for c in comparison]
    secondary_labels = [c["secondary_label"] for c in comparison]
    kappa_value, interpretation = compute_cohens_kappa(primary_labels, secondary_labels)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kappa_out_path = RESULTS_DIR / f"kappa_validation_{condition}_{ts}.json"
    kappa_result = {
        "n_sample": len(comparison),
        "kappa_value": round(kappa_value, 4),
        "interpretation": interpretation,
        "judge_models_identical": judge_models_identical,
        "primary_output_path": str(primary_output_path),
        "secondary_output_path": str(secondary_output_path),
        "comparison": comparison,
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
    parser.add_argument("--majority-rounds", type=int, default=int(os.getenv("JUDGE_MAJORITY_ROUNDS", 1)),
                         help="Disarankan >1 HANYA utk subsample kappa-validation, BUKAN batch penuh -- "
                              "biaya API naik linear per putaran tambahan.")
    parser.add_argument("--force", action="store_true",
                         help="Hapus hasil judge lama (kalau ada) utk kombinasi source+config yang SAMA "
                              "persis, nilai ulang dari nol -- bukan resume.")
    parser.add_argument("--kappa-validation", action="store_true")
    parser.add_argument("--secondary-judge-provider", default=os.getenv("SECONDARY_JUDGE_PROVIDER"),
                         help="Wajib diisi (atau via env SECONDARY_JUDGE_PROVIDER) kalau --kappa-validation "
                              "aktif. Pilihan: openai, anthropic, local, ollama.")
    parser.add_argument("--secondary-judge-model", default=os.getenv("SECONDARY_JUDGE_MODEL"))
    parser.add_argument("--kappa-sample-size", type=int, default=int(os.getenv("KAPPA_SAMPLE_SIZE", 50)))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.kappa_validation and not (args.secondary_judge_provider and args.secondary_judge_model):
        log("[ERROR] --secondary-judge-provider dan --secondary-judge-model wajib diisi kalau "
            "--kappa-validation aktif (atau set SECONDARY_JUDGE_PROVIDER/SECONDARY_JUDGE_MODEL di .env).")
        sys.exit(1)

    input_path = Path(args.input_path)
    if not input_path.exists():
        log(f"[ERROR] --input-path tidak ditemukan: {input_path}")
        sys.exit(1)

    log(f"[config] condition={args.condition} mode={CONDITION_MODE[args.condition]} "
        f"judge={args.judge_provider}/{args.judge_model} temperature={args.judge_temperature} "
        f"majority_rounds={args.majority_rounds} force={args.force}")

    kappa_result = None
    if args.kappa_validation:
        log(f"[kappa] Menjalankan judge utama + validasi Cohen's Kappa (subsample n={args.kappa_sample_size}, "
            f"secondary judge={args.secondary_judge_provider}/{args.secondary_judge_model})...")
        kappa_result = run_kappa_validation(
            input_path, args.judge_provider, args.judge_model, args.judge_temperature,
            args.secondary_judge_provider, args.secondary_judge_model, args.judge_temperature,
            args.majority_rounds, args.condition, args.kappa_sample_size, args.seed, force=args.force,
        )
        summary = kappa_result["primary_summary"]
    else:
        summary = run_judge_batch(
            input_path, args.judge_provider, args.judge_model, args.judge_temperature,
            args.majority_rounds, args.condition, force=args.force,
        )

    output_path = summary["output_path"]

    log("\n" + "=" * 70)
    log(f"RINGKASAN LLM-AS-JUDGE -- Kondisi {args.condition}")
    log("=" * 70)
    log(f"Total pertanyaan       : {summary.get('n_total')}")
    log(f"Baru dievaluasi        : {summary.get('n_evaluated_baru')}")
    log(f"Dari cache (sudah ada) : {summary.get('n_dari_cache')}")
    log(f"% FAKTUAL              : {summary.get('pct_faktual', '-')}%")
    log(f"% HALUSINASI_SEBAGIAN  : {summary.get('pct_halusinasi_sebagian', '-')}%")
    log(f"% HALUSINASI_PENUH     : {summary.get('pct_halusinasi_penuh', '-')}%")
    log(f"Mean faithfulness_score      : {summary.get('mean_faithfulness_score', '-')}"
        + (" (n/a utk Kondisi A -- tidak ada retrieved_context)" if args.condition == "A" else ""))
    log(f"Mean answer_relevance_score  : {summary.get('mean_answer_relevance_score', '-')}")
    log(f"\nHasil judge tersimpan -> {output_path}")

    if kappa_result is not None:
        log(f"\nCohen's Kappa: {kappa_result['kappa_value']} ({kappa_result['interpretation']})")
        log(f"[kappa] Hasil lengkap disimpan -> {kappa_result['kappa_json_path']}")

    history_path = append_judge_run_history(
        args.condition, input_path, output_path, args.judge_provider, args.judge_model,
        args.judge_temperature, args.majority_rounds, summary, kappa_result=kappa_result,
    )
    log(f"[logging] Manifest run judge dicatat -> {history_path}")


if __name__ == "__main__":
    sys.exit(main())
