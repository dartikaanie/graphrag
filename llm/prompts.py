"""
llm/prompts.py
=====================================
Konstruksi prompt (4-turn chat messages) dipakai BERSAMA oleh semua
kondisi eksperimen. build_base_messages() mereplikasi PERSIS
get_base_message() dari ll_model.py (repo asli leusonmario/chat-stack) --
dipakai apa adanya oleh Kondisi A, dan jadi dasar bagi Kondisi B/C yang
menambahkan konteks retrieval di turn terakhir.

Struktur 4-turn (sama di semua kondisi, supaya validitas perbandingan
antar kondisi terjaga -- satu-satunya yang boleh berbeda antar kondisi
adalah ADA/TIDAKNYA konteks retrieval yang disisipkan, bukan struktur
dasarnya):
    1. system    -> persona umum software engineering
    2. user      -> persona spesifik dibentuk dari tags pertanyaan
    3. assistant -> priming/konfirmasi persona (LLM "mengucapkan" perannya
                    sendiri)
    4. user      -> [opsional: konteks retrieval] + pertanyaan + deskripsi
"""


def _clean_tags(tags: str) -> str:
    return tags.replace("<", "").replace(">", " ").strip() if tags else "software development"


def build_base_messages(title: str, body: str, tags: str) -> list[dict]:
    """Replikasi persis get_base_message() (repo asli chat-stack):
    system (persona umum) -> user (persona spesifik dari tags) ->
    assistant (priming/konfirmasi persona) -> user (pertanyaan + deskripsi).
    Dipakai LANGSUNG oleh Kondisi A (LLM murni, tanpa retrieval)."""
    tag_list = _clean_tags(tags)
    return [
        {"role": "system",
         "content": "You are an expert in software engineering with much experience on programming."},
        {"role": "user",
         "content": f"Please, act as you have solid experience on these topics: {tag_list} ."},
        {"role": "assistant",
         "content": f"Okay, I have a solid background on {tag_list} ."},
        {"role": "user",
         "content": f"Please, explain how to fix the problem below. {title}. "
                    f"Below, you can find more details. {body}."},
    ]


def build_rag_messages(title: str, body: str, tags: str, retrieved: list) -> list[dict]:
    """Struktur 4-turn SAMA PERSIS dengan build_base_messages(), DITAMBAH
    konteks komunitas hasil retrieval yang disisipkan di AWAL turn user
    terakhir, sebelum pertanyaan (retrieval-augmented generation
    konvensional -- tanpa constraint grounding eksplisit, itu bedanya
    dari Kondisi C). Dipakai oleh Kondisi B; Kondisi C bisa memakai fungsi
    yang sama kalau format `retrieved` (list of dict dengan key
    'chunk_text') tetap konsisten."""
    tag_list = _clean_tags(tags)

    if retrieved:
        context_block = "\n\n".join(
            f"[Referensi {i+1}] {r['chunk_text']}" for i, r in enumerate(retrieved)
        )
        context_section = (
            f"Here is some potentially relevant context from the Stack Overflow "
            f"community that may help you answer (use your own judgment; not all "
            f"references may be directly applicable):\n\n{context_block}\n\n"
        )
    else:
        context_section = ""

    return [
        {"role": "system",
         "content": "You are an expert in software engineering with much experience on programming."},
        {"role": "user",
         "content": f"Please, act as you have solid experience on these topics: {tag_list} ."},
        {"role": "assistant",
         "content": f"Okay, I have a solid background on {tag_list} ."},
        {"role": "user",
         "content": f"{context_section}"
                    f"Please, explain how to fix the problem below. {title}. "
                    f"Below, you can find more details. {body}."},
    ]
