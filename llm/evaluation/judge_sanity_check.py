"""
judge_sanity_check.py
=====================================
Step 4 sanity check for judge-v1 (judge_prompt_v1.py / llm_judge_
hallucination_v1.py). Picks 3 REAL questions from the n=10 pilot sample
(title/body/tags/reference answer all genuine, pulled the same way the
real runner does -- body via DuckDB, _judge_common.load_question_bodies).
For each, builds 3 SYNTHETIC candidates by hand:
  1. the reference answer itself, verbatim           -> expect FAKTUAL
  2. the reference answer with its main function/method/class renamed to
     something that does not exist                   -> expect HALUSINASI_PENUH
  3. an "insufficient context" refusal, built from the EXACT empty-
     context instruction text in llm/prompts.py                -> expect ABSTAIN

CATATAN SOAL CASE #3: tidak ada contoh ASLI dari output C/D yang berupa
abstention/refusal semacam ini di sample n=10 yang ada sekarang -- sudah
dicek (lihat laporan Step 0): retrieved_context tidak pernah kosong utk
10 pertanyaan pilot ini, dan grep atas SEMUA file hasil tidak menemukan
frasa "no community context"/"insufficient context" di jawaban manapun.
Instruksi grounding-constraint utk context kosong di llm/prompts.py
(build_graphrag_messages()/build_lightrag_messages(), require_grounding
branch -- lihat konstanta EMPTY_CONTEXT_INSTRUCTION di bawah, disalin
PERSIS dari sana) SEBENARNYA menyuruh model TETAP MENJAWAB (hanya
menandai "tidak grounded"), bukan menolak menjawab sama sekali -- kalau
diikuti harfiah, itu TIDAK akan memicu ABSTAIN (Decision Rule 0:
"the candidate does NOT attempt an answer"). Case #3 di bawah karena itu
memakai KLAUSA PERSIS dari instruksi asli ("No relevant context was
found...", "NOT grounded in retrieved sources") sbg dasar teksnya, lalu
MENAMBAHKAN kalimat penolakan eksplisit di atasnya supaya benar-benar
jadi non-answer -- ini tetap teks SINTETIK yang ditulis manual (bukan
disalin dari output model manapun), menguji apakah judge mengenali pola
abstain secara umum (sesuai Decision Rule 0), bukan mereplikasi perilaku
C/D yang sebenarnya (yang, sejauh data yang ada, tidak pernah benar-benar
menolak menjawab).

Exit code: 0 kalau semua (question x variant x judge) sesuai ekspektasi,
1 kalau ada yang tidak -- supaya bisa dipakai sbg gate di CI/pra-run
evaluasi penuh.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _judge_common import html_to_text, load_question_bodies  # noqa: E402
from judge_clients import get_judge_client  # noqa: E402
from judge_prompt_v1 import build_messages, parse_judgment  # noqa: E402
from llm_judge_hallucination_v1 import format_tags  # noqa: E402

log = print

A_RESULTS = Path(__file__).resolve().parents[1] / "a_pure_llm/results/condition_a_openai_gpt-4o-mini_n10_seed42.jsonl"

# question_id -> fabricated variant (reference answer's main API/class/
# method swapped for one that does not exist, main claim otherwise intact
# so the ONLY new defect is the fabrication itself)
FABRICATED_OVERRIDES = {
    76224221: (
        "The suggested fix is not valid code - unless I missed something obvious.\n\n"
        "The suggested fix seems to have a pretty simple error (which you should report): "
        "it should use the `clippy::unwrap_or_bail!` macro, which automatically converts "
        "the error branch for you:\n\n"
        "let entry = entry_res.unwrap_or_bail!();\n\n"
        "This has been the recommended idiomatic replacement for manual let-else since "
        "Rust 1.65.0, released November 2022."
    ),
    2617170: (
        "Note that class names are defined by HTML, not CSS. HTML4 says the class "
        "attribute is a cdata-list, which is space-separated tokens. So a single "
        "classname token can contain any character except the whitespace characters.\n\n"
        "To refer to such names in a CSS class selector, you should call the built-in "
        "`CSS.escapeClassName()` function, which automatically converts any special "
        "character into its escaped CSS selector form for you, e.g. "
        "`CSS.escapeClassName(\"1blah/bläh\")` returns the correctly escaped selector "
        "string ready to use directly in your stylesheet."
    ),
    11118023: (
        "You can do this by setting `strictParsing` to `false` on the `NullableDateFormatter` "
        "bean:\n\n"
        "<mvc:annotation-driven conversion-service=\"conversionService\"/>\n\n"
        "<bean id=\"conversionService\" class=\"org.springframework.format.support."
        "FormattingConversionServiceFactoryBean\">\n"
        " <property name=\"formatters\">\n"
        " <set>\n"
        " <bean class=\"org.springframework.format.datetime.NullableDateFormatter\">\n"
        " <constructor-arg><value>dd/MM/yyyy</value></constructor-arg>\n"
        " <property name=\"strictParsing\" value=\"false\"/>\n"
        " </bean>\n"
        " </set>\n"
        " </property>\n"
        "</bean>"
    ),
}

# Copied verbatim from llm/prompts.py's empty-context instruction (the
# `require_grounding` branch of build_graphrag_messages()/
# build_lightrag_messages(), used when retrieval returns zero items):
#   "No relevant context was found in the Stack Overflow community
#    knowledge graph for this question. Answer based on your own
#    knowledge, but explicitly state at the start of your answer that no
#    community context was retrieved, so this answer is NOT grounded in
#    retrieved sources."
# That instruction tells the model to STILL ANSWER (just flagged as
# ungrounded) -- it is not itself a refusal. ABSTAIN_CANDIDATE below
# reuses its exact clauses as the factual basis, then adds an explicit
# non-answering sentence on top, so the result is an actual abstention
# (Decision Rule 0: "does not attempt an answer") rather than a flagged-
# but-still-attempted answer. See module docstring for why this is
# synthetic rather than a real C/D output.
EMPTY_CONTEXT_INSTRUCTION = (
    "No relevant context was found in the Stack Overflow community knowledge graph for this "
    "question. Answer based on your own knowledge, but explicitly state at the start of your "
    "answer that no community context was retrieved, so this answer is NOT grounded in "
    "retrieved sources."
)

ABSTAIN_CANDIDATE = (
    "No relevant context was found in the Stack Overflow community knowledge graph for this "
    "question, so this answer is NOT grounded in retrieved sources. I do not have enough "
    "reliable information to answer this question, and I am not able to provide a solution here."
)

# 4th case: the SAME disclaimer clause as ABSTAIN_CANDIDATE (also copied
# from the exact prompt instruction), but -- unlike ABSTAIN_CANDIDATE --
# followed by a full, correct answer rather than a refusal. This is what
# literal compliance with llm/prompts.py's instruction actually produces
# ("explicitly state... this answer is NOT grounded", then "answer based
# on your own knowledge" anyway) -- Decision Rule 0 requires the
# candidate to NOT attempt an answer for ABSTAIN, and this one clearly
# does, so the correct verdict is FAKTUAL, not ABSTAIN. Exercises the
# judge's ability to tell "flagged as ungrounded" apart from "declined
# to answer" -- a model naively matching on disclaimer language alone
# would wrongly call this ABSTAIN.
DISCLAIMER_PREFIX = (
    "No community context was retrieved for this question, so this answer is NOT grounded "
    "in retrieved sources. "
)

QUESTION_IDS = list(FABRICATED_OVERRIDES)


def load_real_fixtures(questions_parquet: str) -> list[dict]:
    records = {}
    with open(A_RESULTS) as f:
        for line in f:
            r = json.loads(line)
            if r["question_id"] in QUESTION_IDS:
                records[r["question_id"]] = r

    bodies = load_question_bodies(questions_parquet, QUESTION_IDS)

    fixtures = []
    for qid in QUESTION_IDS:
        record = records[qid]
        reference_text = html_to_text(record["ground_truth_answer"])
        tags = format_tags(record.get("tags"))
        fixture = {
            "question_id": qid,
            "title": record["title"],
            "tags": tags,
            "body": html_to_text(bodies.get(qid, "")),
            "reference": reference_text,
            "variants": {
                "reference_itself": (reference_text, "FAKTUAL"),
                "fabricated": (FABRICATED_OVERRIDES[qid], "HALUSINASI_PENUH"),
                "abstain": (ABSTAIN_CANDIDATE, "ABSTAIN"),
                "disclaimer_then_correct": (DISCLAIMER_PREFIX + reference_text, "FAKTUAL"),
            },
        }
        fixtures.append(fixture)
    return fixtures


def run_case(client, config, fixture: dict, variant_name: str, candidate: str, expected: str) -> dict:
    messages = build_messages(
        title=fixture["title"], body=fixture["body"], tags=fixture["tags"],
        reference=fixture["reference"], candidate=candidate,
    )
    response = client.chat.completions.create(model=config.model, messages=messages, temperature=0.0)
    parsed = parse_judgment(response.choices[0].message.content)
    actual = parsed.get("label")
    return {
        "question_id": fixture["question_id"],
        "variant": variant_name,
        "expected": expected,
        "actual": actual,
        "pass": actual == expected,
        "parse_error": parsed.get("parse_error", False),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", choices=["primary", "secondary", "both"], default="both")
    parser.add_argument("--questions-parquet", default=None)
    args = parser.parse_args()

    import os
    questions_parquet = args.questions_parquet or os.getenv("QUESTIONS_PARQUET")
    if not questions_parquet:
        log("[ERROR] --questions-parquet (or QUESTIONS_PARQUET in .env) is required")
        sys.exit(1)

    judge_ids = ["primary", "secondary"] if args.judge == "both" else [args.judge]
    fixtures = load_real_fixtures(questions_parquet)

    all_results = []
    for judge_id in judge_ids:
        client, config = get_judge_client(judge_id)
        for fixture in fixtures:
            for variant_name, (candidate, expected) in fixture["variants"].items():
                result = run_case(client, config, fixture, variant_name, candidate, expected)
                result["judge_id"] = judge_id
                all_results.append(result)

    log(f"\n{'judge':<10} {'question_id':<12} {'variant':<18} {'expected':<20} {'actual':<20} {'pass'}")
    log("-" * 95)
    for r in all_results:
        log(f"{r['judge_id']:<10} {r['question_id']:<12} {r['variant']:<18} "
            f"{r['expected']:<20} {str(r['actual']):<20} {'OK' if r['pass'] else 'FAIL'}")

    n_fail = sum(1 for r in all_results if not r["pass"])
    log(f"\n{len(all_results) - n_fail}/{len(all_results)} cases passed.")

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
