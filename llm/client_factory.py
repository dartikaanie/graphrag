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
"""

import os
import sys


def call_llm_openai(client, messages: list[dict], model: str, temperature: float | None = None) -> str:
    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def call_llm_anthropic(client, messages: list[dict], model: str, temperature: float | None = None) -> str:
    """Anthropic API memisahkan system message dari array messages (beda
    dari OpenAI/Ollama yang menaruh role='system' langsung di dalam
    array) -- system diekstrak lalu dikirim lewat parameter `system`,
    sisanya (user/assistant priming/user) dikirim apa adanya supaya
    urutan 4-turn tetap identik dengan struktur asli."""
    system_content = next((m["content"] for m in messages if m["role"] == "system"), None)
    chat_messages = [m for m in messages if m["role"] != "system"]
    kwargs = {"model": model, "max_tokens": 1024, "system": system_content, "messages": chat_messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.messages.create(**kwargs)
    return response.content[0].text


def call_llm_local(pipe, messages: list[dict], model: str, temperature: float | None = None) -> str:
    """Untuk model lokal via Hugging Face (mis. LLaMA), pola yang sama
    dipakai paper baseline asli (load Llama-2-7b-chat-hf secara lokal).
    Pakai chat template tokenizer kalau tersedia (representasi 4-turn
    paling akurat); fallback ke penggabungan manual role+content kalau
    model/tokenizer tidak punya chat template."""
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    output = pipe(prompt_text, max_new_tokens=512, do_sample=False)
    return output[0]["generated_text"][len(prompt_text):].strip()


def call_llm_ollama(client, messages: list[dict], model: str, temperature: float | None = None) -> str:
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
    bukan cuma membatasi dampaknya lewat num_predict."""
    options = {"num_ctx": 8192, "num_predict": 1536, "repeat_penalty": 1.3}
    if temperature is not None:
        options["temperature"] = temperature
    response = client.chat(
        model=model,
        messages=messages,
        options=options,
    )
    return response["message"]["content"]


def get_llm_client(provider: str, model: str, log=print):
    """Factory: siapkan client sesuai provider + fungsi call_llm_* yang
    cocok. Menambah provider baru di masa depan cukup tambah 1 cabang di
    sini + 1 fungsi call_llm_* di atas -- otomatis tersedia untuk semua
    kondisi (A/B/C) yang mengimpor modul ini, tidak perlu ubah script
    kondisi manapun.

    `log`: fungsi untuk print status (default: print biasa). Tiap script
    kondisi sebaiknya oper logger.info miliknya sendiri di sini supaya
    status provider tetap tercatat di file log masing-masing kondisi,
    bukan cuma tampil di console.
    """
    if provider == "openai":
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            log("[ERROR] OPENAI_API_KEY tidak ditemukan di .env")
            sys.exit(1)
        return OpenAI(), call_llm_openai

    if provider == "anthropic":
        import anthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            log("[ERROR] ANTHROPIC_API_KEY tidak ditemukan di .env "
                "(catatan: ini API key dari console.anthropic.com, "
                "BEDA dari langganan Claude.ai)")
            sys.exit(1)
        return anthropic.Anthropic(), call_llm_anthropic

    if provider == "local":
        from transformers import pipeline
        log(f"      Loading model lokal '{model}' (bisa lama untuk pertama kali)...")
        pipe = pipeline("text-generation", model=model, device_map="auto")
        return pipe, call_llm_local

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
        return ollama, call_llm_ollama

    raise ValueError(f"Provider '{provider}' tidak dikenal. Pilihan: openai, anthropic, local, ollama")