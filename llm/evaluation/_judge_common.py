"""
_judge_common.py
=====================================
Helper KECIL dipakai BERSAMA oleh dua judge baru --
llm_judge_context_relevance.py dan llm_judge_answer_relevance.py -- supaya
logika resume, path output, manifest, dan HTML-stripping tidak
terduplikasi. `llm_judge_hallucination.py` (judge lama, Faithfulness/
Hallucination Rate) TIDAK diubah dan TIDAK bergantung pada modul ini --
modul ini murni tambahan baru.

Konvensi path SAMA dengan llm_judge_hallucination.py: RESULTS_DIR/LOG_DIR
relatif ke cwd (script dijalankan dari llm/evaluation/, sama seperti judge
lama dan seperti entri run.py yang lain).
"""

import html as html_lib
import json
import random
import re
import sys
import threading
import time
from pathlib import Path

RESULTS_DIR = Path("results")
LOG_DIR = Path("logs")


# ---------------------------------------------------------------------
# Parsing output judge (JSON, mungkin dibungkus ```json ... ```)
# ---------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def strip_fences(text: str) -> str:
    """Sama persis dengan `_strip_fences` di llm_judge_hallucination.py --
    dibuat publik di sini supaya bisa dipakai ulang tanpa impor private
    member dari modul lain."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


# ---------------------------------------------------------------------
# HTML -> plain text (untuk question body & ground_truth_answer yang
# masih HTML mentah dari parquet SORD)
# ---------------------------------------------------------------------

_BLOCK_BREAK_RE = re.compile(r"</(p|div|li|pre|blockquote|h[1-6])\s*>|<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def html_to_text(html: str | None, max_chars: int = 4000) -> str:
    """Strip tag HTML + unescape entities, TANPA membuang isi blok kode
    (<pre>/<code> hanya kehilangan tag pembungkusnya, teksnya tetap utuh --
    beda dari sekadar men-drop seluruh elemen). Tag block-level (</p>,
    </div>, </li>, <br>, dst.) diganti newline dulu SEBELUM tag di-strip
    supaya paragraf/list tidak menyatu jadi satu baris panjang yang sulit
    dibaca judge. Dipotong ke max_chars (bukan token) -- cukup untuk
    keperluan prompt judge, tidak perlu presisi token exact."""
    if not html:
        return ""
    text = str(html)
    text = _BLOCK_BREAK_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = html_lib.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    text = text.strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + " …[truncated]"
    return text


# ---------------------------------------------------------------------
# Question body loader -- SATU query DuckDB batch, gaya SAMA dengan
# load_question_tags() di analyze_retrieval_quality.py (pragma memory-
# limit/threads/preserve_insertion_order utk target MacBook M2 8GB).
# ---------------------------------------------------------------------

def load_question_bodies(questions_parquet: str, question_ids: list[int]) -> dict[int, str]:
    """Return {question_id: Body (HTML mentah, BELUM di-strip)}. Kalau
    suatu question_id tidak ketemu di parquet, key-nya TIDAK ADA di dict
    hasil (bukan string kosong) -- caller mengecek `qid in bodies` utk
    menentukan body_available.

    KETERBATASAN: kolom `match_source` dipakai utk dedup baris duplikat
    Id (SORD men-dedup Contain vs LikeMinusContain match) -- pola query
    ini SAMA PERSIS dengan load_question_tags() di
    analyze_retrieval_quality.py dan _load_oversample_pool() di
    a_baseline_replication.py, sengaja disalin polanya (bukan diimpor)
    supaya modul ini tidak punya dependency ke script kondisi manapun."""
    if not question_ids:
        return {}
    import duckdb

    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    ids_str = ",".join(str(int(i)) for i in set(question_ids))
    query = f"""
        SELECT Id, Body
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{questions_parquet}')
            WHERE Id IN ({ids_str})
        )
        WHERE rn = 1
    """
    df = con.execute(query).df()
    con.close()
    result = {}
    for row in df.itertuples():
        body = row.Body
        result[int(row.Id)] = body if isinstance(body, str) else ""
    return result


# ---------------------------------------------------------------------
# Output path -- deterministik dari (prefix, file sumber, config judge),
# pola SAMA dengan build_judge_output_path() di llm_judge_hallucination.py.
# ---------------------------------------------------------------------

def build_output_path(prefix: str, input_path, provider: str, model: str,
                       temperature: float, extra: str = "") -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    safe_model = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")
    temp_str = str(temperature).replace(".", "-")
    extra_str = f"_{extra}" if extra else ""
    filename = f"{prefix}__{stem}__{provider}-{safe_model}_t{temp_str}{extra_str}.jsonl"
    return RESULTS_DIR / filename


# ---------------------------------------------------------------------
# Resume helpers -- pola SAMA dengan load_already_judged()/
# is_judge_complete() di llm_judge_hallucination.py.
# ---------------------------------------------------------------------

def load_already_done(output_path) -> set:
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


def input_question_ids(input_path) -> list:
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


def is_complete(input_path, output_path) -> bool:
    already = load_already_done(output_path)
    if not already:
        return False
    input_ids = set(input_question_ids(input_path))
    return input_ids.issubset(already)


# ---------------------------------------------------------------------
# Manifest append-only -- SATU file per jenis judge (BUKAN
# judge_run_history.jsonl yang sudah ada -- itu punya backend consumer
# (judge_lookup_service.py) yang formatnya TIDAK BOLEH diganggu).
# ---------------------------------------------------------------------

def append_manifest(manifest_name: str, record: dict) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / manifest_name
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return path


# ---------------------------------------------------------------------
# Self-judging warning -- self-preference bias kalau judge model == model
# generator yang dinilai. TIDAK menghentikan run, hanya peringatan.
# ---------------------------------------------------------------------

def warn_if_self_judging(records_sample: list[dict], judge_model: str) -> None:
    models_seen = {r.get("llm_model") for r in records_sample if r.get("llm_model")}
    if judge_model in models_seen:
        print(
            f"PERINGATAN: judge model ('{judge_model}') sama dengan llm_model pada "
            f"sebagian/seluruh record yang dinilai -- berisiko self-preference bias "
            f"(judge cenderung menilai jawaban dari model yang sama dengan dirinya "
            f"lebih menguntungkan). Pertimbangkan memakai judge model yang berbeda "
            f"dari model generator, atau catat ini sebagai keterbatasan di Bab V.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------
# Cohen's Kappa -- versi generalized dari compute_cohens_kappa() di
# llm_judge_hallucination.py, tapi menerima label_set apa pun (bukan
# selalu 3 kelas hallucination) supaya bisa dipakai utk item-level
# relevance labels (3 kelas) MAUPUN sufficiency labels (3 kelas beda set).
# ---------------------------------------------------------------------

def cohens_kappa(labels_a: list[str], labels_b: list[str], label_set: list[str]) -> tuple[float, str]:
    """4 pita interpretasi SAMA PERSIS dengan compute_cohens_kappa() judge
    lama (penyederhanaan dari 6 pita asli Landis & Koch 1977)."""
    from sklearn.metrics import cohen_kappa_score

    kappa = float(cohen_kappa_score(labels_a, labels_b, labels=list(label_set)))
    if kappa < 0.4:
        interpretation = "lemah (poor agreement)"
    elif kappa < 0.6:
        interpretation = "moderat (moderate agreement)"
    elif kappa < 0.8:
        interpretation = "kuat (substantial agreement)"
    else:
        interpretation = "sangat kuat (almost perfect agreement)"
    return kappa, interpretation


# ---------------------------------------------------------------------
# Shared judge-v1/v2-style runner plumbing -- version-INDEPENDENT logic
# used by llm_judge_hallucination_v1.py and llm_judge_hallucination_v2.py
# (and any future vN sibling). Moved here (rather than kept duplicated
# per version, or living only in v1's module) so a future bugfix to
# retry/resume/failure-logging only has to be made once. Behavior is
# UNCHANGED from what v1 had inline -- proven by v1's own existing test
# suite (test_llm_judge_hallucination_v1.py) passing unmodified after
# v1 was switched to import these instead of defining them itself.
#
# Deliberately NOT shared: judge_one() (prompt-specific -- builds
# messages via each version's own build_messages()/parses via each
# version's own parse_judgment(), and merges different extra fields
# into the output record), and run_batch()'s per-task orchestration
# (differs only in which judge_one it calls, not worth abstracting).
# ---------------------------------------------------------------------

JUDGE_RESUME_FIELDS = (
    "run_id", "config_hash", "question_id", "judge_id", "judge_model",
    "prompt_version", "blinding_version",
)


def judge_resume_key(r: dict) -> tuple:
    """(run_id, config_hash, question_id, judge_id, judge_model,
    prompt_version, blinding_version) -- ALL seven must match for a
    record to count as "already done". Version-independent: prompt_
    version/blinding_version are read from the RECORD's own fields, not
    hardcoded, so a v1 record and a v2 record never collide (different
    prompt_version values) even though they're read by the same
    function. See llm_judge_hallucination_v1.py's original docstring
    for why each field is included."""
    return tuple(r.get(f) for f in JUDGE_RESUME_FIELDS)


def load_judge_already_done(output_path) -> set[tuple]:
    done: set[tuple] = set()
    output_path = Path(output_path)
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(judge_resume_key(json.loads(line)))
            except Exception:
                continue
    return done


def load_judge_failed_keys(failures_path) -> set[tuple]:
    """Keys of every item EVER recorded in <out>_failures.jsonl (across
    all past runs) -- used only to detect when a retry succeeds, so a
    resolution can be logged. The failures file itself is never read
    for resume/skip purposes and is NEVER deleted or rewritten."""
    keys: set[tuple] = set()
    failures_path = Path(failures_path)
    if not failures_path.exists():
        return keys
    with open(failures_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(judge_resume_key(json.loads(line)))
            except Exception:
                continue
    return keys


_JUDGE_WRITE_LOCK = threading.Lock()


def write_judge_record(output_path, record: dict) -> None:
    with _JUDGE_WRITE_LOCK:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def append_judge_resolution_record(resolved_path, record: dict) -> None:
    """A previously-failed (run_id, question_id, judge_id, prompt_
    version) succeeded on this run -- noted in a SEPARATE append-only
    log, never touching the original failure record."""
    resolution = {
        "run_id": record["run_id"], "question_id": record["question_id"],
        "judge_id": record["judge_id"], "prompt_version": record["prompt_version"],
        "resolved": True, "resolved_at_utc": record["timestamp_utc"],
    }
    write_judge_record(resolved_path, resolution)


JUDGE_RETRYABLE_MAX_ATTEMPTS = 6


def call_judge_llm_with_retry(client, config, messages: list[dict], request_temperature: float = 0.0,
                               log=print) -> tuple[dict | None, list[dict]]:
    """Returns (response_payload, attempts). response_payload is None
    only if every attempt failed. `attempts` lists EVERY attempt made
    (not just the last), so nothing about a retried call is dropped,
    even on eventual success. response_payload carries the FULL raw
    response metadata (not just parsed content): response id/model/
    created/system_fingerprint/finish_reason/usage, plus the request
    params actually sent -- enough to audit exactly what was asked for
    and what came back."""
    import openai

    retryable = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,  # covers 5xx
    )

    attempts: list[dict] = []
    for attempt in range(1, JUDGE_RETRYABLE_MAX_ATTEMPTS + 1):
        try:
            start = time.monotonic()
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=request_temperature,
            )
            latency_s = time.monotonic() - start
            attempts.append({"attempt": attempt, "error": None, "status_code": None})
            usage = response.usage
            return {
                "raw": response.choices[0].message.content,
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "latency_s": round(latency_s, 3),
                "response_id": response.id,
                "response_model": response.model,
                "response_created": response.created,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "finish_reason": response.choices[0].finish_reason,
                "usage_full": usage.model_dump() if usage else None,
                "request_params": {"model": config.model, "temperature": request_temperature},
            }, attempts
        except retryable as e:
            status_code = getattr(e, "status_code", None)
            error_text = f"{type(e).__name__}: {e}"
            attempts.append({"attempt": attempt, "error": error_text, "status_code": status_code})
            if attempt == JUDGE_RETRYABLE_MAX_ATTEMPTS:
                break
            sleep_s = min(60, (2 ** (attempt - 1))) + random.uniform(0, 1)
            log(f"      [retry {attempt}/{JUDGE_RETRYABLE_MAX_ATTEMPTS}] {error_text} -- sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
        except Exception as e:
            # Non-retryable (e.g. 400 bad request, auth error) -- fail fast.
            status_code = getattr(e, "status_code", None)
            error_text = f"{type(e).__name__}: {e}"
            attempts.append({"attempt": attempt, "error": error_text, "status_code": status_code})
            break

    return None, attempts


def check_served_model_mismatch(response_model: str | None, config, judge_id: str, log=print) -> bool:
    """True if the provider's response.model differs from the registry's
    expected_served_model for this judge_id -- logs a [WARN] when so (a
    silent provider-side model swap would otherwise go unnoticed)."""
    expected = getattr(config, "expected_served_model", None)
    mismatch = bool(expected) and response_model != expected
    if mismatch:
        log(f"      [WARN] judge '{judge_id}': response_model='{response_model}' "
            f"differs from registry's expected_served_model='{expected}' -- the provider may have "
            f"changed which model/variant it actually serves for this judge_id. Investigate before "
            f"trusting further records from this judge_id.")
    return mismatch
