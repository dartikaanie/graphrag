"""
llm/client_factory.py
=====================================
Factory terpusat untuk membuat client LLM & memanggil model -- dipakai
BERSAMA oleh semua kondisi eksperimen (Kondisi A, B, C). Sebelumnya kode
ini terduplikasi persis di setiap script kondisi (a_baseline_replecation.py,
b_condition_b_rag.py); sekarang cukup di satu tempat:
  - Tambah provider baru cukup 1x di sini, otomatis kepakai semua kondisi.
  - Perbaikan/bug fix cara panggil suatu provider cukup 1x, tidak perlu
    disinkronkan manual ke tiap file kondisi.

Semua fungsi call_llm_* menerima `messages: list[dict]` -- struktur 4-turn
chat dialog dari llm.prompts (build_base_messages / build_rag_messages),
BUKAN string prompt tunggal.

RAW RESPONSE METADATA (Step 5, docs/Agent prompt grounding factor ui.md
-- data retention/audit trail) -- LOW-RISK ADDITIVE APPROACH, bukan
mengubah signature call_llm_* yang sudah ada:
  - Tiap provider sekarang juga punya `call_llm_*_with_meta()`, yang
    mengembalikan dict {"content", "response_id", "response_model",
    "response_created", "system_fingerprint", "finish_reason",
    "usage_full", "request_params"} -- bukan cuma string content.
  - call_llm_*() LAMA (dipertahankan APA ADANYA, signature sama persis)
    sekarang cuma wrapper tipis yang memanggil versi _with_meta lalu
    ambil ["content"] saja -- jadi SATU implementasi per provider (di
    _with_meta), bukan dua logika terpisah yang bisa drift.
  - Provider yang tidak punya suatu field (mis. Ollama tidak punya
    response id/system_fingerprint; model lokal HF tidak punya hampir
    semua field ini sama sekali) mengisi None utk field itu -- TIDAK
    PERNAH crash karena field hilang.
  - get_llm_client(..., with_meta=True) mengembalikan call_fn versi
    _with_meta -- caller LAMA yang tidak di-migrasi (with_meta tetap
    default False) sama sekali tidak terpengaruh.
"""

import os
import sys


def call_llm_openai_with_meta(client, messages: list[dict], model: str, temperature: float | None = None) -> dict:
    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(**kwargs)
    usage = response.usage
    return {
        "content": response.choices[0].message.content,
        "response_id": response.id,
        "response_model": response.model,
        "response_created": response.created,
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "finish_reason": response.choices[0].finish_reason,
        "usage_full": usage.model_dump() if usage else None,
        "request_params": {"model": model, "temperature": temperature},
    }


def call_llm_openai(client, messages: list[dict], model: str, temperature: float | None = None) -> str:
    return call_llm_openai_with_meta(client, messages, model, temperature)["content"]


def call_llm_anthropic_with_meta(client, messages: list[dict], model: str, temperature: float | None = None) -> dict:
    """Anthropic API memisahkan system message dari array messages (beda
    dari OpenAI/Ollama yang menaruh role='system' langsung di dalam
    array) -- system diekstrak lalu dikirim lewat parameter `system`,
    sisanya (user/assistant priming/user) dikirim apa adanya supaya
    urutan 4-turn tetap identik dengan struktur asli.

    Anthropic TIDAK punya response.created/system_fingerprint sama
    sekali (field itu spesifik OpenAI) -- selalu None, bukan ditebak."""
    system_content = next((m["content"] for m in messages if m["role"] == "system"), None)
    chat_messages = [m for m in messages if m["role"] != "system"]
    max_tokens = 1024
    kwargs = {"model": model, "max_tokens": max_tokens, "system": system_content, "messages": chat_messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.messages.create(**kwargs)
    usage = getattr(response, "usage", None)
    return {
        "content": response.content[0].text,
        "response_id": getattr(response, "id", None),
        "response_model": getattr(response, "model", None),
        "response_created": None,
        "system_fingerprint": None,
        "finish_reason": getattr(response, "stop_reason", None),
        "usage_full": usage.model_dump() if usage else None,
        "request_params": {"model": model, "temperature": temperature, "max_tokens": max_tokens},
    }


def call_llm_anthropic(client, messages: list[dict], model: str, temperature: float | None = None) -> str:
    return call_llm_anthropic_with_meta(client, messages, model, temperature)["content"]


def call_llm_local_with_meta(pipe, messages: list[dict], model: str, temperature: float | None = None) -> dict:
    """Untuk model lokal via Hugging Face (mis. LLaMA), pola yang sama
    dipakai paper baseline asli (load Llama-2-7b-chat-hf secara lokal).
    Pakai chat template tokenizer kalau tersedia (representasi 4-turn
    paling akurat); fallback ke penggabungan manual role+content kalau
    model/tokenizer tidak punya chat template.

    HF text-generation pipeline tidak mengembalikan response metadata
    terstruktur apa pun (tidak ada response id/usage/finish_reason) --
    semua field itu None di sini, bukan ditebak/dihitung manual."""
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    output = pipe(prompt_text, max_new_tokens=512, do_sample=False)
    content = output[0]["generated_text"][len(prompt_text):].strip()
    return {
        "content": content,
        "response_id": None,
        "response_model": model,
        "response_created": None,
        "system_fingerprint": None,
        "finish_reason": None,
        "usage_full": None,
        "request_params": {"model": model, "temperature": temperature, "max_new_tokens": 512, "do_sample": False},
    }


def call_llm_local(pipe, messages: list[dict], model: str, temperature: float | None = None) -> str:
    return call_llm_local_with_meta(pipe, messages, model, temperature)["content"]


def call_llm_ollama_with_meta(client, messages: list[dict], model: str, temperature: float | None = None) -> dict:
    """Provider dev/testing: model kecil lokal via Ollama (mis. qwen2.5:1.5b,
    phi3:mini). TIDAK dipakai untuk hasil evaluasi final laporan -- hanya
    untuk iterasi cepat pipeline sebelum run resmi pakai model besar
    (openai/anthropic/local Llama-2-7b). Ollama chat API sudah native
    mendukung array messages dengan role system/user/assistant, jadi
    4-turn dialog terkirim langsung tanpa perlu konversi. Butuh
    `ollama serve` jalan di background & model sudah di-pull.

    num_ctx dinaikkan eksplisit ke 8192 (bukan default Ollama yang untuk
    banyak model cuma 2048) -- penting karena Kondisi B/C menyisipkan
    konteks retrieval (bisa >2000 token utk 5 chunk x 400 token) ke
    prompt; tanpa num_ctx yang cukup, prompt panjang ke-truncate diam-diam
    dan menyebabkan generasi jawaban yang tidak relevan/berulang-ulang
    (terkonfirmasi dari drop similarity mendadak di run n=30 Kondisi B).

    num_predict dibatasi ke 1536 token: tanpa batas ini, model kecil yang
    masuk mode repetition loop akan terus generate sampai mentok num_ctx
    (bisa >10 menit per pertanyaan di CPU M2), bukan berhenti wajar.
    1536 token (~6000 karakter) sudah lebih dari cukup utk gaya jawaban
    "explain how to fix" yang dipakai di seluruh eksperimen ini.

    repeat_penalty dinaikkan sedikit (1.3, default Ollama 1.1) utk
    mengurangi kecenderungan model kecil masuk mode repetisi sejak awal,
    bukan cuma membatasi dampaknya lewat num_predict.

    Ollama tidak punya response id/system_fingerprint (field spesifik
    OpenAI) -- selalu None. usage_full diisi dari prompt_eval_count/
    eval_count Ollama sendiri (nama field beda dari OpenAI, TIDAK
    dipetakan ke prompt_tokens/completion_tokens supaya tidak terlihat
    seolah-olah itu angka OpenAI asli)."""
    options = {"num_ctx": 8192, "num_predict": 1536, "repeat_penalty": 1.3}
    if temperature is not None:
        options["temperature"] = temperature
    response = client.chat(
        model=model,
        messages=messages,
        options=options,
    )
    prompt_eval_count = response.get("prompt_eval_count")
    eval_count = response.get("eval_count")
    usage_full = (
        {"prompt_eval_count": prompt_eval_count, "eval_count": eval_count}
        if (prompt_eval_count is not None or eval_count is not None) else None
    )
    return {
        "content": response["message"]["content"],
        "response_id": None,
        "response_model": response.get("model"),
        "response_created": response.get("created_at"),
        "system_fingerprint": None,
        "finish_reason": response.get("done_reason"),
        "usage_full": usage_full,
        "request_params": {"model": model, "temperature": temperature, "options": options},
    }


def call_llm_ollama(client, messages: list[dict], model: str, temperature: float | None = None) -> str:
    return call_llm_ollama_with_meta(client, messages, model, temperature)["content"]


_CALL_FN_WITH_META = {
    "openai": call_llm_openai_with_meta,
    "anthropic": call_llm_anthropic_with_meta,
    "local": call_llm_local_with_meta,
    "ollama": call_llm_ollama_with_meta,
}
_CALL_FN_PLAIN = {
    "openai": call_llm_openai,
    "anthropic": call_llm_anthropic,
    "local": call_llm_local,
    "ollama": call_llm_ollama,
}


def get_llm_client(provider: str, model: str, log=print, with_meta: bool = False):
    """Factory: siapkan client sesuai provider + fungsi call_llm_* yang
    cocok. Menambah provider baru di masa depan cukup tambah 1 cabang di
    sini + 1 fungsi call_llm_* di atas -- otomatis tersedia untuk semua
    kondisi (A/B/C/D) yang mengimpor modul ini, tidak perlu ubah script
    kondisi manapun.

    `log`: fungsi untuk print status (default: print biasa). Tiap script
    kondisi sebaiknya oper logger.info miliknya sendiri di sini supaya
    status provider tetap tercatat di file log masing-masing kondisi,
    bukan cuma tampil di console.

    `with_meta` (default False, BACKWARD COMPATIBLE): False mengembalikan
    call_fn versi LAMA (kembalikan string content saja, perilaku TIDAK
    BERUBAH utk caller manapun yang belum di-migrasi); True mengembalikan
    versi _with_meta (kembalikan dict content+metadata) -- lihat modul
    docstring.
    """
    call_fn_table = _CALL_FN_WITH_META if with_meta else _CALL_FN_PLAIN

    if provider == "openai":
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            log("[ERROR] OPENAI_API_KEY tidak ditemukan di .env")
            sys.exit(1)
        return OpenAI(), call_fn_table["openai"]

    if provider == "anthropic":
        import anthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            log("[ERROR] ANTHROPIC_API_KEY tidak ditemukan di .env "
                "(catatan: ini API key dari console.anthropic.com, "
                "BEDA dari langganan Claude.ai)")
            sys.exit(1)
        return anthropic.Anthropic(), call_fn_table["anthropic"]

    if provider == "local":
        from transformers import pipeline
        log(f"      Loading model lokal '{model}' (bisa lama untuk pertama kali)...")
        pipe = pipeline("text-generation", model=model, device_map="auto")
        return pipe, call_fn_table["local"]

    if provider == "ollama":
        import ollama
        try:
            ollama.list()
        except Exception as e:
            log(f"[ERROR] Tidak bisa konek ke Ollama di localhost:11434 -- "
                f"pastikan 'ollama serve' sudah jalan. Detail: {e}")
            sys.exit(1)
        log(f"      [dev/testing only] Provider ollama, model '{model}' -- "
            f"hasil ini TIDAK untuk laporan evaluasi final.")
        return ollama, call_fn_table["ollama"]

    raise ValueError(f"Provider '{provider}' tidak dikenal. Pilihan: openai, anthropic, local, ollama")
