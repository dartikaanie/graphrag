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

CATATAN SOAL CASE #3 (update sejak prompt-parity v3, docs/
PROMPT_PARITY_V3.md): tidak ada contoh ASLI dari output B/C/D yang berupa
abstention/refusal semacam ini di sample n=10 yang ada sekarang -- belum
pernah di-generate ulang dengan prompt v3. TAPI instruksi empty-context
(grounded) di llm/prompts.py SEKARANG SUDAH literal meminta non-answer
("state no relevant sources were found... do not attempt to answer from
your own general knowledge" -- lihat GROUNDED_EMPTY_CONTEXT_INSTRUCTION
di bawah, disalin PERSIS dari llm/prompts.py.GROUNDED_EMPTY_CONTEXT),
BEDA dari instruksi v2 lama yang justru menyuruh TETAP MENJAWAB (itu dulu
sempat bertentangan dgn Decision Rule 0 ABSTAIN sama sekali). Case #3 di
bawah SEKARANG adalah KEPATUHAN LITERAL thd instruksi v3 itu sendiri --
masih ditulis manual (bukan disalin dari output model manapun, karena
belum ada generation dgn v3), tapi sudah tidak perlu "menambahkan kalimat
penolakan di atas instruksi" lagi seperti sebelumnya, karena instruksi v3
itu sendiri sudah berupa permintaan utk TIDAK menjawab.

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

# Copied verbatim from llm/prompts.py's GROUNDED_EMPTY_CONTEXT constant
# (used by build_rag_messages()/build_graphrag_messages()/
# build_lightrag_messages() when require_grounding=True and retrieval
# returns zero items -- prompt-parity v3, docs/PROMPT_PARITY_V3.md):
#   "No relevant context was found in the Stack Overflow community for
#    this question. State clearly that no relevant sources were found
#    for this question, and do not attempt to answer from your own
#    general knowledge."
# Unlike the pre-v3 wording, this instruction ITSELF asks for a non-
# answer -- ABSTAIN_CANDIDATE below is now literal compliance with it,
# not an embellishment on top of a still-answering instruction.
GROUNDED_EMPTY_CONTEXT_INSTRUCTION = (
    "No relevant context was found in the Stack Overflow community for this question. State "
    "clearly that no relevant sources were found for this question, and do not attempt to "
    "answer from your own general knowledge."
)

ABSTAIN_CANDIDATE = (
    "No relevant sources were found for this question in the Stack Overflow community. I am "
    "not able to provide a reliable answer without relevant source material."
)

# 4th case: disclaimer-sounding language followed by a full, correct
# answer anyway -- under prompt v3 this is a DEVIATION from
# GROUNDED_EMPTY_CONTEXT_INSTRUCTION (which now explicitly forbids
# answering from general knowledge), not literal compliance with it. Kept
# as a case regardless, because its purpose is independent of what the
# instruction says: it exercises whether the judge can tell "disclaimer
# language present" apart from "the candidate actually declined to
# answer" (Decision Rule 0) -- a judge that pattern-matches on disclaimer
# wording alone would wrongly call this ABSTAIN even though a complete
# answer follows it.
DISCLAIMER_PREFIX = (
    "No relevant sources were found for this question in the Stack Overflow community. "
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
