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

PROMPT_VERSION
------------------------------------------------------------
"v1" = versi asli, ID contoh sitasi di instruksi citation NUMERIK
       ([SO-1234], [SO-5678], dst.) -- ditemukan model (terutama model
       kecil lokal) kadang menyalin literal angka contoh ini sbg sitasi
       saat konteks yang diberikan lemah/tidak relevan (lihat docs/README.md
       §8). Berlaku untuk SEMUA run yang ditulis SEBELUM fix ini masuk --
       run lama TIDAK menyimpan field prompt_version sama sekali (field ini
       belum ada saat run itu dibuat).
"v2" = ID contoh diganti placeholder non-numerik ([SO-<id>], [SO-<id1>]
       [SO-<id2>]) supaya tidak pernah terlihat seperti ID SO asli yang
       bisa disalin. TIDAK ADA perubahan wording lain di instruksi.
       MASIH BERMASALAH: diagnosis NF2 v2 (lihat docs/
       NF2_ROOT_CAUSE_PLACEHOLDER_CITATIONS.md) menemukan model TETAP
       kadang menyalin literal PLACEHOLDER-nya sendiri ("[SO-<id>]") kalau
       konteks yang diberikan lemah/kurang relevan -- hanya memindahkan
       simtom (digit palsu -> placeholder literal), bukan menghilangkannya.
"v3" (PROMPT_VERSION saat ini) = DUA perubahan, HASIL KEPUTUSAN EKSPLISIT
       setelah diagnosis NF2 v2 (lihat docs/
       NF2_ROOT_CAUSE_PLACEHOLDER_CITATIONS.md untuk analisis lengkap dan
       opsi yang DITOLAK -- id konteks asli sbg contoh ditolak karena bisa
       membuat model menempelkan ID valid-looking ke klaim yang sebenarnya
       tidak didukung, menyembunyikan kegagalan dari NF2 bukan
       memperbaikinya):
         1. Kalimat "worked example" yang dikutip penuh ("You can fix
            this by adding a null check... [SO-<id>]...") DIHAPUS dari
            SEMUA builder (B/C/D) -- itulah satu-satunya teks yang benar2
            quotable/dicopy verbatim. Hanya deskripsi format singkat yang
            disisakan, dan deskripsi itu TIDAK memuat token "<id>"/
            "<id1>"/"<id2>" apa pun lagi (diuji di llm/tests/
            test_prompts.py::test_no_v3_prompt_string_contains_placeholder).
         2. Instruksi baru: HANYA kutip label yang BENAR-BENAR ADA di
            konteks yang diberikan DAN benar2 mendukung klaim tersebut;
            kalau tidak ada sumber yang mendukung suatu klaim, tulis klaim
            itu TANPA sitasi -- jangan menebak atau mengeluarkan
            placeholder apa pun.
       Kondisi A (build_base_messages) tidak pernah menyebut sitasi sama
       sekali, jadi TEKSNYA TIDAK DIUBAH -- tapi run Kondisi A SETELAH fix
       ini tetap MEREKAM prompt_version="v3" (via PROMPT_VERSION yang
       sama, diimpor a_baseline_replication.py), supaya satu perbandingan
       A/B/C/D selalu berbagi SATU versi prompt, bukan tiga kondisi di v3
       dan satu di v2 karena teksnya "tidak perlu diubah".
       Direkam sbg field `prompt_version` di run_history.jsonl setiap
       kondisi (CLI maupun dashboard) -- run BARU otomatis "v3"; semua
       run v1/v2 LAMA dan filenya TIDAK DISENTUH (tetap v1 (inferred)/v2
       sesuai kapan dijalankan), demi reproducibility hasil yang sudah
       ada.

       REFINED lagi sebelum generation pertama dgn v3 pernah dijalankan
       (jadi masih "v3", bukan v4 -- belum ada data yg perlu dibedakan):
       Prompt PARITY -- lihat docs/PROMPT_PARITY_V3.md untuk detail
       lengkap dan alasannya:
         - Rule 1 vs rule 3 di atas SEMPAT KONTRADIKTIF (rule 1 melarang
           klaim tak didukung, rule 3 lama MENGIZINKANNYA tanpa sitasi) --
           rule 3 ditulis ulang: kalau konteks tak mendukung suatu klaim
           yg dibutuhkan, klaim itu TIDAK DIBUAT sama sekali (bukan dibuat
           tanpa sitasi), dengan model menyatakan sumber tak mencakup
           bagian itu.
         - build_rag_messages() (Kondisi B) sekarang punya parameter
           `require_grounding`, dgn teks grounded BYTE-IDENTICAL ke C/D
           (lewat helper bersama _grounded_context_section() dkk) --
           sebelumnya B tidak punya constraint grounding sama sekali.
         - SEMUA metadata retrieval (trust=, hop, low_level/high_level)
           dihapus dari TEKS PROMPT di ketiga kondisi -- tetap direkam di
           record hasil tersimpan, tapi tidak lagi dilihat LLM, supaya
           prompt benar2 retrieval-method-agnostic.
         - Instruksi empty-context (grounded) diganti dari "jawab dari
           pengetahuan umum tapi tandai tidak grounded" (yg SEBENARNYA
           bertentangan dgn rule 3 yg strict) jadi "nyatakan tidak ada
           sumber relevan, JANGAN jawab dari pengetahuan umum" -- varian
           non-grounded TIDAK diubah.
         - Satu formatter context (_format_context_block) dan satu set
           template grounded/non-grounded dipakai bersama oleh B/C/D --
           dijamin oleh test_prompt_parity_v3.py (byte-identical, bukan
           cuma "terlihat mirip").
"""

PROMPT_VERSION = "v3"

# ---------------------------------------------------------------------------
# Shared building blocks -- dipakai IDENTIK oleh build_rag_messages()
# (Kondisi B)/build_graphrag_messages() (C)/build_lightrag_messages() (D)
# supaya satu-satunya hal yang bisa berbeda antar prompt mereka adalah
# RETRIEVAL ITU SENDIRI (item apa yang terambil), bukan cara memformat/
# menginstruksikannya. Lihat docs/PROMPT_PARITY_V3.md.
# ---------------------------------------------------------------------------

BASE_SYSTEM_CONTENT = "You are an expert in software engineering with much experience on programming."

GROUNDED_SYSTEM_ADDITION = (
    " When context is provided, you MUST cite the source thread label shown "
    "next to it for every claim it supports. If the context does not support a "
    "claim needed to answer, do not make that claim -- state that the retrieved "
    "sources do not cover that part instead. Never cite a label that isn't "
    "literally present in that context."
)

NON_GROUNDED_SYSTEM_ADDITION = (
    " When context is provided, you MUST cite the source thread label shown "
    "next to it for every claim it supports, and must never cite a label "
    "that isn't literally present in that context."
)

GROUNDED_REMINDER = (
    "\n\nReminder: cite the exact source label shown above immediately after "
    "every factual claim it supports. If the context doesn't support a claim "
    "needed to answer, don't make that claim -- say the retrieved sources "
    "don't cover it instead."
)

NON_GROUNDED_REMINDER = (
    "\n\nReminder: cite the exact source label shown above immediately after "
    "every factual claim it supports; if no label supports a claim, leave "
    "that claim uncited."
)

# Grounded empty-context instruction -- MUST be consistent with rule 3's
# strict "do not make an unsupported claim" above: the old wording ("answer
# based on your own knowledge, but state it's not grounded") directly
# contradicted that rule (it's an unsupported-by-definition answer). This
# one instead asks the model to do what rule 3 already asks for in the
# degenerate case where NO context exists at all: state that, and
# decline to answer from general knowledge. The non-grounded empty-context
# branch is UNCHANGED ("" -- no special instruction), since it never had a
# "don't introduce unsupported claims" rule to be consistent with.
GROUNDED_EMPTY_CONTEXT = (
    "No relevant context was found in the Stack Overflow community for this "
    "question. State clearly that no relevant sources were found for this "
    "question, and do not attempt to answer from your own general "
    "knowledge.\n\n"
)


def _format_context_block(retrieved: list) -> str:
    """Shared context formatter -- label + chunk text ONLY, no
    trust_weight/hop/source_stage annotation. That metadata stays in the
    stored result record for post-hoc analysis (see each condition's own
    retrieved_context field) but is deliberately never shown to the model
    -- showing it risked the model's citation behavior being influenced by
    which retrieval METHOD produced an item (e.g. trusting a high
    trust_weight item more), rather than purely by what the item's text
    actually says. It's also the reason B/C/D's prompts can now be
    byte-identical: B never had this metadata to begin with, so stripping
    it from C/D removes the last retrieval-method-specific text."""
    return "\n\n".join(f"[SO-{r['question_id']}] {r['chunk_text']}" for r in retrieved)


def _grounded_context_section(context_block: str) -> str:
    """Byte-identical grounded (dual-constraint) instruction block,
    shared by B (require_citation=True, require_grounding=True)/
    C/D (require_grounding=True)."""
    return (
        f"Here is context retrieved from the Stack Overflow community that may "
        f"help you answer, each labeled with its own source thread id in square "
        f"brackets (format: the letters SO, a hyphen, then the thread's numeric "
        f"id -- no colon, no space, and not the word 'thread'):\n\n{context_block}\n\n"
        f"IMPORTANT -- follow these constraints strictly:\n"
        f"1. Ground your answer ONLY in the context above. Do not introduce facts, "
        f"APIs, or claims that are not supported by the given context.\n"
        f"2. For EVERY factual claim or piece of advice in your answer, cite the "
        f"source thread it came from using the EXACT label shown next to that "
        f"context item above, immediately after the claim. If a claim draws on "
        f"multiple sources, cite each one, back to back.\n"
        f"3. If the context does not support a claim needed to answer, do not make "
        f"that claim; instead briefly state that the retrieved sources do not "
        f"cover that part. Only cite a label literally present in the context "
        f"above that actually supports the claim. Never invent or output a "
        f"placeholder label.\n\n"
    )


def _non_grounded_context_section(context_block: str) -> str:
    """Byte-identical non-grounded (citation-only) instruction block,
    shared by B (require_citation=True, require_grounding=False)/
    C/D (require_grounding=False)."""
    return (
        f"Here is some potentially relevant context from the Stack Overflow "
        f"community that may help you answer (use your own judgment; not all "
        f"references may be directly applicable), each labeled with its own "
        f"source thread id in square brackets (format: the letters SO, a hyphen, "
        f"then the thread's numeric id):\n\n{context_block}\n\n"
        f"For EVERY factual claim or piece of advice in your answer, cite the "
        f"source thread it came from using the EXACT label shown next to that "
        f"context item above, immediately after the claim. If a claim draws on "
        f"multiple sources, cite each one, back to back. Only cite a label that is "
        f"literally present in the context above AND actually supports the claim -- "
        f"if no provided source supports a claim, write that claim without a "
        f"citation rather than guessing or inventing a label.\n\n"
    )


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


def build_graphrag_messages(title: str, body: str, tags: str, retrieved: list,
                             require_grounding: bool = True) -> list[dict]:
    """Struktur 4-turn SAMA PERSIS dengan build_base_messages()/build_rag_messages()
    (system persona umum -> user persona dari tags -> assistant priming -> user
    pertanyaan+deskripsi), DITAMBAH konteks hasil hybrid retrieval (vector search +
    graph traversal berbobot trust) di turn user terakhir.

    Sejak prompt-parity v3 (docs/PROMPT_PARITY_V3.md), instruksi dan
    formatting konteksnya PERSIS SAMA dengan build_rag_messages() Kondisi B
    (require_citation=True) dan build_lightrag_messages() Kondisi D, lewat
    helper bersama _format_context_block()/_grounded_context_section()/
    _non_grounded_context_section() -- satu-satunya yang membedakan prompt
    ketiga kondisi ini sekarang ADALAH RETRIEVAL ITU SENDIRI (item apa yang
    terambil), bukan cara memformat/menginstruksikannya. trust_weight/hop
    pada item `retrieved` TIDAK LAGI ditampilkan di teks prompt (tetap
    tersimpan di record hasil utk analisis).

    `require_grounding` (default True) -- ablation switch, HANYA mengganti
    blok instruksi teks, TIDAK mengubah bagaimana `retrieved` diformat:
      - True : dual-constraint (grounding + citation), perilaku asli.
      - False: HANYA instruksi citation -- TANPA instruksi grounding "jangan
        mengarang di luar konteks". Dipakai utk mengisolasi kontribusi
        constraint grounding itu sendiri terhadap constraint sitasi.

    `retrieved`: list of dict dengan key WAJIB 'question_id' (SO thread id
    utk label sitasi) dan 'chunk_text' (isi konteks). Key lain (trust_weight,
    hop, dst.) boleh ada (dipakai evaluator/analisis lain) tapi diabaikan
    oleh fungsi ini -- tidak pernah masuk ke teks prompt.
    """
    tag_list = _clean_tags(tags)

    if retrieved:
        context_block = _format_context_block(retrieved)
        context_section = (_grounded_context_section(context_block) if require_grounding
                            else _non_grounded_context_section(context_block))
    else:
        # Tidak ada konteks ditemukan (retrieval gagal/graf terlalu jarang utk
        # pertanyaan ini) -- TETAP diberi instruksi eksplisit supaya perilaku
        # LLM konsisten (tidak diam-diam berperilaku seperti Kondisi A tanpa
        # disadari) dan supaya has_citation=False di output bisa diinterpretasi
        # sbg limitation retrieval, bukan limitation generation.
        context_section = GROUNDED_EMPTY_CONTEXT if require_grounding else ""

    system_content = BASE_SYSTEM_CONTENT
    if require_grounding:
        system_content += GROUNDED_SYSTEM_ADDITION
    elif retrieved:
        system_content += NON_GROUNDED_SYSTEM_ADDITION

    reminder = ""
    if retrieved:
        reminder = GROUNDED_REMINDER if require_grounding else NON_GROUNDED_REMINDER

    return [
        {"role": "system",
         "content": system_content},
        {"role": "user",
         "content": f"Please, act as you have solid experience on these topics: {tag_list} ."},
        {"role": "assistant",
         "content": f"Okay, I have a solid background on {tag_list} ."},
        {"role": "user",
         "content": f"{context_section}"
                    f"Please, explain how to fix the problem below. {title}. "
                    f"Below, you can find more details. {body}."
                    + reminder},
    ]


def build_lightrag_messages(title: str, body: str, tags: str, retrieved: list,
                             require_grounding: bool = True) -> list[dict]:
    """Struktur 4-turn SAMA PERSIS dengan build_rag_messages()/
    build_graphrag_messages(). Dipakai Kondisi D (dual-level retrieval,
    adaptasi LightRAG).

    Sejak prompt-parity v3 (docs/PROMPT_PARITY_V3.md), instruksi dan
    formatting konteksnya PERSIS SAMA dengan Kondisi B/C lewat helper
    bersama (_format_context_block()/_grounded_context_section()/
    _non_grounded_context_section()) -- satu-satunya yang membedakan
    ketiga kondisi ini sekarang adalah RETRIEVAL ITU SENDIRI, bukan cara
    memformat/menginstruksikannya. `source_stage` ("low_level"/
    "high_level") pada item `retrieved` TIDAK LAGI ditampilkan di teks
    prompt (tetap tersimpan di record hasil utk analisis).

    `require_grounding` (default True) HANYA mengganti blok instruksi
    teks, TIDAK mengubah cara `retrieved` diproses/diformat:
      - True: dual-constraint (grounding + citation).
      - False: HANYA instruksi citation -- TANPA instruksi grounding.
    """
    tag_list = _clean_tags(tags)

    if retrieved:
        context_block = _format_context_block(retrieved)
        context_section = (_grounded_context_section(context_block) if require_grounding
                            else _non_grounded_context_section(context_block))
    else:
        context_section = GROUNDED_EMPTY_CONTEXT if require_grounding else ""

    system_content = BASE_SYSTEM_CONTENT
    if require_grounding:
        system_content += GROUNDED_SYSTEM_ADDITION
    elif retrieved:
        system_content += NON_GROUNDED_SYSTEM_ADDITION

    reminder = ""
    if retrieved:
        reminder = GROUNDED_REMINDER if require_grounding else NON_GROUNDED_REMINDER

    return [
        {"role": "system",
         "content": system_content},
        {"role": "user",
         "content": f"Please, act as you have solid experience on these topics: {tag_list} ."},
        {"role": "assistant",
         "content": f"Okay, I have a solid background on {tag_list} ."},
        {"role": "user",
         "content": f"{context_section}"
                    f"Please, explain how to fix the problem below. {title}. "
                    f"Below, you can find more details. {body}."
                    + reminder},
    ]


def build_rag_messages(title: str, body: str, tags: str, retrieved: list,
                        require_citation: bool = False, require_grounding: bool = False) -> list[dict]:
    """Struktur 4-turn SAMA PERSIS dengan build_base_messages(), DITAMBAH
    konteks komunitas hasil retrieval yang disisipkan di AWAL turn user
    terakhir, sebelum pertanyaan. Dipakai oleh Kondisi B.

    `require_citation` (default False, opsional -- lihat
    CONDITION_B_REQUIRE_CITATION di .env / --require-citation CLI flag
    Kondisi B): kalau True, tiap item konteks diberi label eksplisit
    "[SO-{question_id}]" (persis format Kondisi C/D, BUKAN "[Referensi N]"
    generik) dan model diminta mengutip label itu per klaim. Kalau False,
    `require_grounding` diabaikan -- RAG konvensional tanpa instruksi
    sitasi sama sekali ("[Referensi N]" generik), varian ketiga yang sudah
    ada sejak awal, tidak disentuh oleh prompt-parity v3.

    `require_grounding` (default False, HANYA berlaku kalau
    require_citation=True juga True -- lihat --require-grounding CLI flag
    Kondisi B): kalau True ("B-grounded"), pakai instruksi dual-constraint
    (grounding + citation) yang BYTE-IDENTICAL dengan Kondisi C/D
    (require_grounding=True) lewat helper bersama _grounded_context_section()
    dkk -- lihat docs/PROMPT_PARITY_V3.md. Kalau False ("B-plain", default,
    perilaku SEBELUM prompt-parity v3), instruksi citation-only TANPA
    constraint grounding -- mengukur apakah instruksi sitasi saja (tanpa
    constraint retrieval-method apa pun) sudah cukup menghasilkan sitasi
    valid pada RAG datar, sebagai baseline pembanding NF2 terhadap Kondisi
    C/D. run_label ("B-plain"/"B-grounded") membedakan kedua varian ini di
    run_history.jsonl, sama seperti "C-uniform"/"C-trust_weighted" --
    lihat llm/evaluation/_run_metadata.py.
    """
    tag_list = _clean_tags(tags)

    if retrieved and require_citation:
        context_block = _format_context_block(retrieved)
        context_section = (_grounded_context_section(context_block) if require_grounding
                            else _non_grounded_context_section(context_block))
    elif retrieved:
        context_block = "\n\n".join(
            f"[Referensi {i+1}] {r['chunk_text']}" for i, r in enumerate(retrieved)
        )
        context_section = (
            f"Here is some potentially relevant context from the Stack Overflow "
            f"community that may help you answer (use your own judgment; not all "
            f"references may be directly applicable):\n\n{context_block}\n\n"
        )
    else:
        context_section = GROUNDED_EMPTY_CONTEXT if (require_citation and require_grounding) else ""

    system_content = BASE_SYSTEM_CONTENT
    if require_citation and require_grounding:
        system_content += GROUNDED_SYSTEM_ADDITION
    elif require_citation and retrieved:
        system_content += NON_GROUNDED_SYSTEM_ADDITION

    reminder = ""
    if retrieved and require_citation:
        reminder = GROUNDED_REMINDER if require_grounding else NON_GROUNDED_REMINDER

    return [
        {"role": "system",
         "content": system_content},
        {"role": "user",
         "content": f"Please, act as you have solid experience on these topics: {tag_list} ."},
        {"role": "assistant",
         "content": f"Okay, I have a solid background on {tag_list} ."},
        {"role": "user",
         "content": f"{context_section}"
                    f"Please, explain how to fix the problem below. {title}. "
                    f"Below, you can find more details. {body}."
                    + reminder},
    ]
