"""
ctxrel_prompt_v2.py
=====================================
ctxrel-v2 context-relevance judge prompt (docs/DECISION_C_SCORING.md,
Amendment 2). Changes ONLY the label definitions relative to ctxrel-v1
(llm_judge_context_relevance_v1.build_messages, which stays unchanged):
same inputs (question title/tags/body + exactly one context item), same
JSON output shape, same parser (llm_judge_context_relevance_v1.
parse_judgment).

Tuned ONLY on control items (Amendment 1/2), never on stage-1 retrieved
contexts or per-alpha results.
"""

PROMPT_VERSION = "ctxrel-v2"


def build_messages(title: str, body_text: str, tags: str, chunk_text: str) -> list[dict]:
    system_content = (
        "You are an impartial evaluator (LLM-as-judge) assessing RETRIEVAL "
        "RELEVANCE for a software-engineering Q&A retrieval system. You will "
        "receive a question and exactly ONE retrieved text snippet (a Q&A "
        "thread excerpt: a question title and one answer, possibly cut off). "
        "You do NOT know which system retrieved it, its rank/score, or any "
        "other metadata. Judge purely from whether this snippet's CONTENT "
        "would help someone solve or understand THIS question's problem. "
        "Respond with ONLY a single JSON object, no markdown fences, no extra text."
    )
    question_section = f"Question title: {title}\n"
    if tags:
        question_section += f"Tags: {tags}\n"
    if body_text:
        question_section += f"Question body: {body_text}\n"

    user_content = (
        f"{question_section}\n"
        f"Retrieved item:\n{chunk_text}\n\n"
        "Decide whether the retrieved item contains information that helps solve "
        "or explain THIS question's specific problem.\n\n"
        "Assign exactly one label:\n"
        "- \"RELEVANT\": the item contains information that directly helps solve or "
        "explain THIS question's problem: a fix, a workaround, the cause, a recommended "
        "approach or tool, or a key part of the solution. It does NOT need to be "
        "complete, match the exact versions or wording, or follow the approach the "
        "asker already tried. An alternative approach that achieves the asker's "
        "underlying goal counts as RELEVANT. The item may be cut off; judge only "
        "what is shown.\n"
        "- \"PARTIAL\": same technology or topic, but it addresses a different "
        "problem, or it is only tangentially useful (general background the asker "
        "could not act on for THIS problem).\n"
        "- \"IRRELEVANT\": unrelated to this problem.\n\n"
        "Ask yourself: if the asker read this item, would it move them meaningfully "
        "closer to solving or understanding their problem? If yes, the label is "
        "RELEVANT. Sharing a technology, library, or keyword with the question is "
        "NOT enough on its own.\n\n"
        "Respond with ONLY this JSON object:\n"
        "{\n"
        '  "label": "RELEVANT" | "PARTIAL" | "IRRELEVANT",\n'
        '  "reason": "<one sentence>"\n'
        "}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
