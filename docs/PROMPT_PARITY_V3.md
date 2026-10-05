# Prompt parity v3

**Status:** implemented, not yet exercised by any generation run. All changes live in `llm/prompts.py` (plus small wiring in `llm/b_rag/b_condition_b_rag.py`, `backend/app/services/engine_service.py`, `backend/app/models/schemas.py`). This note summarizes *what* changed and *why*, for the thesis methodology/error-analysis write-up.

## 1. Rule 1 vs rule 3 contradiction, fixed

Prompt v3's grounding rules (Condition C/D, `require_grounding=True`) originally read:

1. Ground your answer ONLY in the context above. Do not introduce facts, APIs, or claims that are not supported by the given context.
2. (citation instruction)
3. ...If no provided source supports a claim, **write that claim without any citation** — do not guess...

Rule 1 forbids introducing an unsupported claim at all; rule 3 explicitly permitted writing one anyway, just without a citation. Rule 3 is now:

> 3. If the context does not support a claim needed to answer, **do not make that claim**; instead briefly state that the retrieved sources do not cover that part. Only cite a label literally present in the context above that actually supports the claim. Never invent or output a placeholder label.

The system-message addition and the trailing reminder were updated to match. A typo in the format description ("no colon, no space, no the word 'thread'") was also fixed to "...and not the word 'thread'".

The **non-grounded** branches (`require_grounding=False`) are unchanged — they never had a "no unsupported claims" rule to begin with, so permitting an uncited claim there was never contradictory.

## 2. Condition B now supports grounding, byte-identical to C/D

`build_rag_messages()` (Condition B) gained a `require_grounding` parameter. When `require_citation=True`:

- `require_grounding=True` (**"B-grounded"**) — dual-constraint instruction text that is **byte-identical** to C/D's grounded branch (shared helper functions, not a hand-copied duplicate).
- `require_grounding=False` (**"B-plain"**, the default — B's behavior *before* this change) — citation-only instruction, unchanged from what B always did.

`require_citation=False` (B's third, pre-existing mode — plain RAG, no citation instruction at all) is untouched by this change; `require_grounding` has no effect there.

New CLI flag `--require-grounding`/`--no-require-grounding` (env `CONDITION_B_REQUIRE_GROUNDING`, default off) and dashboard field `require_grounding` (schema default `None`, so an old/unaware client that omits it keeps getting B-plain, while C/D keep defaulting to `True` as before — the per-condition default lives in `engine_service.py`, not in the shared schema default).

**Run-label distinction**: B's `run_history.jsonl` records now set `fusion_mode` to `"grounded"` or `"plain"` (reusing the exact same field name `history_service.py`/`_run_metadata.py` already read generically for every condition — this is how `C-uniform` vs `C-trust_weighted` are told apart, and the same mechanism now distinguishes `B-plain` vs `B-grounded` with zero new code in `_run_metadata.py`).

## 3. All retrieval metadata stripped from prompt text

`trust_weight`/`hop` (Condition C) and `source_stage` (Condition D) were previously rendered into the context block shown to the model — e.g. `[SO-123] (trust=0.90, 1-hop) ...`. Condition B never had anything like this (its context items were always bare `[SO-123] ...`).

All three conditions now use one shared formatter (`_format_context_block()`) that renders **only** the label and chunk text — no metadata of any kind. The metadata itself is **not removed from the data**: it's still written into each result record's `retrieved_context` field for post-hoc analysis (e.g. "did high-trust items get cited more often"). It's only removed from what the *model* sees, so the model's citation/grounding behavior can't be influenced by which retrieval method produced an item.

## 4. Empty-context (grounded) instruction rewritten

The old grounded empty-context instruction (used when retrieval returns zero items) said: *"...Answer based on your own knowledge, but explicitly state... this answer is NOT grounded in retrieved sources."* — i.e., answer anyway, just flag it. This directly contradicted the strict rule 3 above (an answer built entirely from general knowledge is, by definition, a claim the given context doesn't support).

It now reads:

> No relevant context was found in the Stack Overflow community for this question. State clearly that no relevant sources were found for this question, and do not attempt to answer from your own general knowledge.

This is consistent with rule 3 (it's the limiting case — zero context items means zero support for *any* claim). The **non-grounded** empty-context branch is unchanged (still an empty string — no special instruction, since there's no "don't introduce unsupported claims" rule to be consistent with there).

## 5. One shared formatter/template set, enforced by a parity test

`_format_context_block()`, `_grounded_context_section()`, `_non_grounded_context_section()`, `GROUNDED_SYSTEM_ADDITION`, `NON_GROUNDED_SYSTEM_ADDITION`, `GROUNDED_REMINDER`, `NON_GROUNDED_REMINDER`, `GROUNDED_EMPTY_CONTEXT` are module-level constants/functions in `llm/prompts.py`, used identically by `build_rag_messages()` (B), `build_graphrag_messages()` (C), and `build_lightrag_messages()` (D). `llm/tests/test_prompt_parity_v3.py` asserts, given the same retrieved items, that the grounded prompts from B/C/D are **byte-identical**, and separately that the non-grounded prompts are byte-identical — not just "similar," checked by direct list-of-dict equality on the full 4-turn message structure. This test caught a real bug during implementation (B was adding its citation system-message even when there was no retrieved context, where C/D correctly omit it) before it shipped.

Condition A (`build_base_messages()`) is untouched — it never mentions retrieval or citation, so parity doesn't apply to it.

## 6. Judge-v1 sanity check updated to match

`llm/evaluation/judge_sanity_check.py`'s synthetic ABSTAIN case is now literal compliance with the new `GROUNDED_EMPTY_CONTEXT` wording (previously it had to *add* a refusal clause on top of an instruction that itself asked the model to keep answering — no longer necessary, since the instruction itself now asks for a non-answer).

## What this means for retrieval-only comparison

With all of the above, the only thing that can differ between a B(-grounded), C, and D prompt for the same question is **what retrieval actually returned** — which items, in what order, how many. No wording, formatting, or metadata differences remain as confounds. `PROMPT_VERSION` stays `"v3"` (not bumped to v4): nothing has been generated with v3 yet, so there's no existing data this round of edits needs to be distinguished from.
