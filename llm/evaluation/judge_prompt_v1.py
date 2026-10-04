"""
judge_prompt_v1.py
------------------
Prompt LLM-as-Judge 3-kelas untuk evaluasi halusinasi (Kondisi A, B, C, D).

Desain:
- Single-answer grading (bukan pairwise): satu jawaban dinilai sendiri,
  sehingga tidak ada position bias.
- Reference-based: pembanding utamanya accepted answer Stack Overflow, sama
  untuk semua kondisi, jadi Kondisi A (tanpa retrieval) tetap comparable.
- Claim decomposition dulu, baru label (reasoning sebelum label).
- Aturan keputusan eksplisit + rekalkulasi label secara deterministik di kode
  (derive_label), untuk mendeteksi inkonsistensi judge.
- Blind: penanda sitasi dihapus sebelum dinilai, supaya judge tidak bisa
  menebak kondisi (A tidak punya sitasi; B/C/D punya).

Prompt yang SAMA persis dipakai untuk semua judge (Llama-3.3-70B dan
gpt-4o-mini). Definisi di SYSTEM_PROMPT juga menjadi pedoman anotasi manual.

CATATAN LOKASI: file ini awalnya ditaruh di docs/judge_prompt_v1.py oleh
pemilik repo, dipindah ke sini (llm/evaluation/) agar sejalan dengan
modul evaluasi lain yang sudah ada di folder ini (llm_judge_hallucination.py,
llm_judge_context_relevance.py, dst.) dan bisa diimpor sebagai
`from judge_prompt_v1 import ...` saat script dijalankan dari
llm/evaluation/, sama seperti konvensi modul lain di folder ini.
SYSTEM_PROMPT/USER_TEMPLATE TIDAK diubah sama sekali dari versi asli --
ini adalah instrumen evaluasi yang dibekukan (judge-v1). Perubahan HANYA
ada di CITATION_PATTERN/SOURCES_BLOCK_PATTERN di bawah, untuk disesuaikan
dengan format sitasi NYATA yang dipakai prompt v2 (llm/prompts.py) --
lihat llm/citations.py -- dan sejak blind-v2, juga mencakup placeholder
literal "[SO-<id>]" dkk yang kadang disalin verbatim oleh model dari
worked example milik prompt v2 sendiri. Lihat BLINDING_VERSION di bawah.
"""

import json
import re

PROMPT_VERSION = "judge-v1"

LABELS = ("FAKTUAL", "HALUSINASI_SEBAGIAN", "HALUSINASI_PENUH", "ABSTAIN")

SYSTEM_PROMPT = """You are an expert software engineer acting as an impartial evaluator of answers to Stack Overflow programming questions. Your task is to decide whether a CANDIDATE ANSWER contains hallucinations, using the REFERENCE ANSWER (the accepted answer on Stack Overflow) and your own well-established technical knowledge.

## Definitions

A CLAIM is a technical statement the candidate presents as fact, e.g.: an API, function, method, parameter, flag, option, config key, or library; its behavior; the expected effect of a code snippet; a version or compatibility statement; or the proposed solution itself.

Verdict for each claim:
- SUPPORTED: consistent with the reference answer, or a correct statement you can confirm with high confidence.
- CONTRADICTED: conflicts with the reference answer or with well-established technical facts.
- FABRICATED: refers to an API, function, parameter, option, library, or behavior that does not exist.
- UNVERIFIABLE: cannot be confirmed or refuted from the reference or your confident knowledge. This is NOT a hallucination.

Severity (only for CONTRADICTED or FABRICATED claims):
- CORE: affects whether the main solution works or whether the question is actually answered.
- MINOR: a peripheral detail; the main solution would still work.

## Decision rules (apply in order)

0. ABSTAIN: the candidate does not attempt an answer (it states that information is insufficient, refuses, or only asks for clarification).
1. HALUSINASI_PENUH: at least one CORE claim is CONTRADICTED or FABRICATED, or the answer addresses a different problem than the one asked.
2. HALUSINASI_SEBAGIAN: no CORE errors, but at least one MINOR claim is CONTRADICTED or FABRICATED.
3. FAKTUAL: none of the above.

## Important rules

- An alternative solution that differs from the reference but is technically correct and solves the question is FAKTUAL. Do not penalize an answer for not matching the reference.
- Incompleteness alone is not a hallucination. Only penalize an omission if it makes the stated solution wrong.
- Ignore length, style, formatting, and tone. A longer answer is not a better answer.
- Judge technical correctness only. Do not try to guess which system produced the answer.
- Be conservative: if you are not confident a claim is wrong, mark it UNVERIFIABLE.
- List at most 8 claims, prioritizing those most important to the solution.

## Output

Respond with JSON only, using exactly these keys in this order:
{
  "claims": [
    {"claim": "<short paraphrase>",
     "verdict": "SUPPORTED | CONTRADICTED | FABRICATED | UNVERIFIABLE",
     "severity": "CORE | MINOR | null",
     "evidence": "<one short sentence>"}
  ],
  "reasoning": "<2-4 sentences applying the decision rules>",
  "label": "FAKTUAL | HALUSINASI_SEBAGIAN | HALUSINASI_PENUH | ABSTAIN"
}"""

USER_TEMPLATE = """### QUESTION
Title: {title}
Tags: {tags}
Body:
{body}

### REFERENCE ANSWER (accepted on Stack Overflow)
{reference}

### CANDIDATE ANSWER
{candidate}"""


# ---------------------------------------------------------------------------
# Preprocessing: blinding
# ---------------------------------------------------------------------------
# blind-v1 (first cut) only matched DIGIT citations: "[SO-<question_id>]",
# e.g. [SO-38118194], sometimes several in a row [SO-<id1>][SO-<id2>].
# blind-v2 (current) ALSO matches the literal placeholder text a model
# sometimes copies verbatim from llm/prompts.py's own worked example --
# "[SO-<id>]", "[SO-<id1>]", "[SO-<id2>]" (see docs/README.md Sec 8,
# "the [SO-1234] 'hallucinated' citation... was the model copying the
# prompt's own worked example") -- that literal placeholder is ITSELF a
# condition-revealing artifact (only ever seen in B/C/D output, never A),
# so leaving it unstripped would let the judge guess the condition from
# blinding being incomplete, defeating the whole point of blind_candidate().
# Found via a real leak in export-human-csv's blinded CSV (see git history/
# conversation) -- caught by testing, not by inspection.
#
# The pattern below requires a separator char ([-: <\]]) RIGHT AFTER "SO"
# so it only ever matches an actual [SO...] token family -- never
# incidental brackets that happen to start with those two letters, e.g.
# "[Solution]"/"[SOLVED]" (no match: next char after "SO" is a letter,
# not one of "-", ":", whitespace, "<", or the closing "]"). Within that,
# the body is deliberately unrestricted ([^\]]{0,20}, capped short --
# real/placeholder citation tokens are always a handful of characters)
# so it covers the full family in one pattern: [SO-123], [SO:123],
# [SO 123], [SO thread 123], [SO-<id>], [SO-<id1>], [SO-<id2>], and any
# future minor variant of the same family -- not just the two forms seen
# so far. Still does NOT touch prose outside brackets (e.g. a disclaimer
# sentence like "this answer is NOT grounded in retrieved sources" is
# never bracketed, so it always survives blinding untouched).
CITATION_PATTERN = re.compile(r"\[SO(?=[-:\s<\]])[^\]]{0,20}\]", re.IGNORECASE)
# Blok "Sources:/References:/Referensi:/Sumber:" di akhir jawaban, jika ada
# (tidak pernah diminta oleh prompt v2, tapi disisakan sbg jaring pengaman
# kalau suatu model menambahkannya sendiri secara spontan).
SOURCES_BLOCK_PATTERN = re.compile(
    r"\n\s*(?:sources|references|referensi|sumber)\s*:.*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)

# Bumped alongside CITATION_PATTERN -- recorded on every judge output
# record (see llm_judge_hallucination_v1.py) so a result produced before
# this fix (blind-v1, digit-only stripping) is distinguishable from one
# produced after (blind-v2, also strips the literal placeholder family).
BLINDING_VERSION = "blind-v2"


def blind_candidate(answer: str) -> str:
    """Hapus penanda sitasi (termasuk placeholder literal [SO-<id>] dkk)
    supaya judge tidak bisa menebak kondisi. HANYA menghapus token sitasi
    yang dibungkus kurung siku -- kalimat disclaimer di luar kurung siku
    (mis. "NOT grounded in retrieved sources") TIDAK PERNAH disentuh."""
    text = SOURCES_BLOCK_PATTERN.sub("", answer)
    text = CITATION_PATTERN.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def build_messages(title, body, tags, reference, candidate,
                   max_chars_each=6000):
    """Bangun messages untuk chat.completions. Teks panjang dipotong."""
    def cut(s):
        s = s or ""
        return s if len(s) <= max_chars_each else s[:max_chars_each] + "\n[...truncated]"

    user = USER_TEMPLATE.format(
        title=title,
        tags=tags,
        body=cut(body),
        reference=cut(reference),
        candidate=cut(blind_candidate(candidate)),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Postprocessing: parse + rekalkulasi label deterministik
# ---------------------------------------------------------------------------
def derive_label(claims):
    """Terapkan aturan keputusan 1-3 dari daftar klaim.
    ABSTAIN tidak bisa diturunkan dari klaim, jadi ditangani terpisah."""
    errors = [c for c in claims
              if str(c.get("verdict", "")).upper() in ("CONTRADICTED", "FABRICATED")]
    if any(str(c.get("severity", "")).upper() == "CORE" for c in errors):
        return "HALUSINASI_PENUH"
    if errors:
        return "HALUSINASI_SEBAGIAN"
    return "FAKTUAL"


def parse_judgment(raw: str) -> dict:
    """Parse output judge. Mengembalikan dict dengan:
    label        : label dari judge
    derived_label: label hasil aturan di kode
    consistent   : apakah keduanya sama (ABSTAIN dianggap konsisten)
    parse_error  : True jika JSON gagal di-parse
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return {"parse_error": True, "raw": raw}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"parse_error": True, "raw": raw}

    label = str(data.get("label", "")).strip().upper()
    claims = data.get("claims", []) or []
    derived = "ABSTAIN" if label == "ABSTAIN" else derive_label(claims)

    return {
        "parse_error": False,
        "label": label if label in LABELS else None,
        "derived_label": derived,
        "consistent": label == derived,
        "claims": claims,
        "reasoning": data.get("reasoning", ""),
        "prompt_version": PROMPT_VERSION,
    }
