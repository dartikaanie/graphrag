"""llm/citations.py
=====================================
[SO-<id>] citation extraction/validation -- moved here from c_graphrag.py so
Condition B can reuse the EXACT same validation logic when its optional
`--require-citation` mode is on (see build_rag_messages() in prompts.py and
b_condition_b_rag.py). Having one shared implementation, not two copies, is
what makes a B-vs-C citation-compliance comparison actually apples-to-apples.
"""

import re

CITATION_PATTERN = re.compile(r"\[SO[-: ]?(?:thread\s*)?(\d+)\]", re.IGNORECASE)


def extract_citations(llm_answer: str, retrieved: list) -> tuple[bool, list, bool, list]:
    """Return (has_citation, cited_ids, has_valid_citation, valid_ids).

    has_citation/cited_ids: deteksi longgar -- toleransi variasi format kecil
    ([SO-1234], [SO:1234], [SO 1234], [SO thread 1234]) supaya percobaan
    kutipan model kecil yang formatnya sedikit meleset tetap terdeteksi
    sbg "mencoba mengutip", bukan otomatis dianggap tidak ada kutipan
    sama sekali.

    has_valid_citation/valid_ids: versi ketat -- id yang dikutip di-cross-
    check terhadap question_id di retrieved_context yang BENAR diberikan ke
    model. Ditambahkan setelah ditemukan kasus model menghasilkan
    citation-like token dgn ID yang di-hallucinate (tidak cocok konteks asli
    sama sekali, mis. [SO:4329876] padahal konteks yg diberikan SO-5928724)
    -- format benar TIDAK CUKUP, ID-nya juga harus benar-benar ada di
    konteks yang diberikan.
    """
    ids = sorted(set(int(m) for m in CITATION_PATTERN.findall(llm_answer)))
    context_ids = {r.get("question_id") for r in (retrieved or []) if r.get("question_id") is not None}
    valid_ids = sorted(i for i in ids if i in context_ids)
    return (len(ids) > 0, ids, len(valid_ids) > 0, valid_ids)
