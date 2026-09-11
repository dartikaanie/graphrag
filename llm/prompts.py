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


def build_graphrag_messages(title: str, body: str, tags: str, retrieved: list) -> list[dict]:
    """Struktur 4-turn SAMA PERSIS dengan build_base_messages()/build_rag_messages()
    (system persona umum -> user persona dari tags -> assistant priming -> user
    pertanyaan+deskripsi), DITAMBAH konteks hasil hybrid retrieval (vector search +
    graph traversal berbobot trust) di turn user terakhir -- BEDA dari
    build_rag_messages() (Kondisi B) dalam DUA hal:

    1. Tiap item konteks diberi label sumber eksplisit "[SO-{question_id}]"
       (bukan cuma "[Referensi N]" generik) supaya bisa dikutip balik oleh LLM
       secara spesifik per SO thread.
    2. Instruksi eksplisit DUAL-CONSTRAINT ditambahkan:
         (a) RAG Grounding    -- jawaban HARUS berdasarkan konteks yang
             diberikan, TIDAK BOLEH mengarang klaim di luar konteks.
         (b) Citation/KG-Constraint -- SETIAP klaim faktual HARUS diikuti
             label "[SO-<id>]" yang sesuai, supaya provenance bisa
             ditelusuri balik ke node KG sumbernya (F5, NF2 -- 100%
             jawaban dengan >=1 kutipan sumber).

    `retrieved`: list of dict dengan key WAJIB 'question_id' (SO thread id
    utk label sitasi), 'chunk_text' (isi konteks), OPSIONAL 'trust_weight'
    dan 'hop' (ditampilkan sbg metadata transparansi, tidak wajib dipakai
    LLM tapi membantu evaluator manusia menilai kualitas retrieval).
    """
    tag_list = _clean_tags(tags)

    if retrieved:
        context_lines = []
        for r in retrieved:
            meta_bits = []
            if r.get("trust_weight") is not None:
                meta_bits.append(f"trust={r['trust_weight']:.2f}")
            if r.get("hop") is not None:
                meta_bits.append(f"{r['hop']}-hop")
            meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
            context_lines.append(f"[SO-{r['question_id']}]{meta} {r['chunk_text']}")
        context_block = "\n\n".join(context_lines)
        context_section = (
            f"Here is context retrieved from the Stack Overflow community knowledge "
            f"graph that may help you answer, each labeled with its source thread id "
            f"(e.g. [SO-1234]):\n\n{context_block}\n\n"
            f"IMPORTANT -- follow these two constraints strictly:\n"
            f"1. Ground your answer ONLY in the context above. Do not introduce facts, "
            f"APIs, or claims that are not supported by the given context.\n"
            f"2. For EVERY factual claim or piece of advice in your answer, cite the "
            f"source thread it came from using its label (e.g. [SO-1234]) immediately "
            f"after the claim. If a claim draws on multiple sources, cite all of them "
            f"(e.g. [SO-1234][SO-5678]).\n\n"
            f"Example of the exact citation style required (note the format is always "
            f"square brackets, the letters SO, a hyphen, then digits -- with NO other "
            f"variation such as a colon or the word 'thread'):\n"
            f"\"You can fix this by adding a null check before accessing the array "
            f"[SO-1234567]. If the error persists after that, verify your build "
            f"configuration matches the recommended setup [SO-2233445][SO-8899001].\"\n"
            f"(The numbers above are just an example format, not real sources -- "
            f"always use the actual [SO-<id>] labels from the context given to you "
            f"above, never invent a number that is not one of those labels.)\n\n"
        )
    else:
        # Tidak ada konteks ditemukan (retrieval gagal/graf terlalu jarang utk
        # pertanyaan ini) -- TETAP diberi instruksi eksplisit supaya perilaku
        # LLM konsisten (tidak diam-diam berperilaku seperti Kondisi A tanpa
        # disadari) dan supaya has_citation=False di output bisa diinterpretasi
        # sbg limitation retrieval, bukan limitation generation.
        context_section = (
            "No relevant context was found in the Stack Overflow community knowledge "
            "graph for this question. Answer based on your own knowledge, but "
            "explicitly state at the start of your answer that no community context "
            "was retrieved, so this answer is NOT grounded in retrieved sources.\n\n"
        )

    return [
        {"role": "system",
         "content": "You are an expert in software engineering with much experience on "
                    "programming. When context is provided, you MUST cite the source "
                    "thread label (e.g. [SO-1234]) for every claim you make."},
        {"role": "user",
         "content": f"Please, act as you have solid experience on these topics: {tag_list} ."},
        {"role": "assistant",
         "content": f"Okay, I have a solid background on {tag_list} ."},
        {"role": "user",
         "content": f"{context_section}"
                    f"Please, explain how to fix the problem below. {title}. "
                    f"Below, you can find more details. {body}."
                    + ("\n\nReminder: cite [SO-<id>] immediately after every factual "
                       "claim, using the labels from the context above." if retrieved
                       else "")},
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