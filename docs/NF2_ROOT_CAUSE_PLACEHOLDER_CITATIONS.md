# NF2 root-cause report: placeholder citation copying (prompt v2 → v3)

**Status:** diagnosis complete, fix implemented as prompt v3. Generation has
not been re-run under v3 yet — all numbers below describe v2 behavior,
measured against the existing n=10 pilot sample.

## 1. Background

After switching the citation worked-example from numeric IDs (`[SO-1234]`,
prompt v1) to non-numeric placeholders (`[SO-<id>]`, `[SO-<id1>][SO-<id2>]`,
prompt v2), NF2 (percentage of answers with ≥1 *valid* citation) dropped on
the same n=10 pilot sample:

| Run | v1 | v2 |
|---|---|---|
| B | — | 60% |
| C (trust_weighted) | — | 80% |
| C (uniform) | — | 90% |
| D | — | 80% |

(v1 numbers for these specific runs were not separately re-measured here —
see the "6 of 10 changed" determinism-fix note in `docs/README.md` §8 for
the related v1→v2 retrieval-content comparison. The point of this report is
v2's own failure modes, independent of that earlier diagnosis.)

## 2. Where the diagnosis started: what "fails NF2" actually looks like

Every v2 NF2 failure across B/C-trust_weighted/C-uniform/D was read
individually (no LLM calls — direct inspection of each failing answer's
text against its `retrieved_context`) and classified into one of five
categories:

- (a) empty context
- (b) no citation attempt at all
- (c) placeholder copied literally, e.g. `[SO-<id>]`, `[SO-<id1>]`
- (d) a citation in a format the validator doesn't match (no brackets, wrong punctuation, bare `[123]`)
- (e) a well-formed citation whose ID is not in the given context

**Result: 9 total NF2 failures, zero in categories (a), (d), or (e).**
6 are category (c) (placeholder copied literally), 3 are category (b) (all
in Condition B specifically — a separate, B-specific weakening of citation
discipline, not analyzed further here since it isn't a copying artifact).

| Condition | QID | v2 answer snippet | v1 (same question) |
|---|---|---|---|
| B | 76224221 | `...if issues persist ([SO-<id>])....` | PASSED — `[SO-77212336]` |
| C-trust | 20443560 | `...the OpenGL Wiki [SO-<id>].` | PASSED — `[SO-31899973]` |
| C-trust | 411756 | `...is not graphical [SO-<id>].` (×4) | PASSED — `[SO-16915770]`, `[SO-29773039]` |
| C-uniform | 76224221 | `...concise error handling [SO-<id>].` | PASSED — `[SO-53670348]` |
| D | 20443560 | `...as shown in your example [SO-<id>].` (×2) | PASSED — `[SO-16951376]`, `[SO-17738738]` |
| D | 411756 | `...symbol information [SO-<id>].` (×4) | v1 had **digit** copying instead — `[SO-1234]` |

D/QID 411756 is the clearest before/after: prompt v1 copied the old numeric
example `[SO-1234]`; prompt v2 copies the new literal example `[SO-<id>]`.
Same underlying behavior (copying the prompt's worked example), the fix
only moved the failure mode, it didn't remove it.

## 3. Counts per run (v2, n=10 pilot)

| Run | Answers with ≥1 literal placeholder | Total placeholder tokens |
|---|---|---|
| B | 1/10 (QID 76224221) | 1 |
| C-trust_weighted | 2/10 (QID 20443560: 1, QID 411756: 4) | 5 |
| C-uniform | 2/10 (QID 76224221: 4, QID 411756: 1) | 5 |
| D | 2/10 (QID 20443560: 2, QID 411756: 4) | 6 |

**Not random — recurring questions.** QID 411756 ("Win32: Graphical
debugger that supports symbol server?") triggers copying in C-trust_weighted,
C-uniform, *and* D. QID 20443560 triggers it in both C-trust_weighted and D.
Checking D's actual retrieved context for QID 411756: the top 3 items are
about gcc warnings, a C# debugger, and Xcode symbol errors — all only
tangentially related to the actual question. The pattern correlates with
**weak/marginal retrieval**, not a uniform per-answer copying rate: when the
model has nothing it can confidently cite, it falls back to the prompt's own
example pattern instead of citing a weak source or omitting the citation.

One additional nuance found while diagnosing: C-uniform/QID 411756 contains
one `[SO-<id>]` placeholder but still **passes** NF2, because the same
answer also has 3 genuine valid citations elsewhere. NF2 (≥1 valid citation)
doesn't catch partial copying when a real citation exists somewhere else in
the same answer — the *true* defect rate (any placeholder present, anywhere
in the answer) is therefore slightly higher than the NF2 failure count alone
suggests.

## 4. Where `[SO-<id>]` lived in the prompt (v2)

Two forms, repeated once per builder (B/C/D each builds its own messages,
so the same text existed ~3× in `llm/prompts.py`):

1. **Inline mentions** — "cite the source thread it came from using its
   label (e.g. `[SO-<id>]`)". Lower risk: reads as a format description,
   not a complete sentence to copy.
2. **The worked-example sentence** — the actual offender, identical shape
   in every builder's `require_grounding=True` branch:

   ```
   Example of the exact citation style required (note the format is always
   square brackets, the letters SO, a hyphen, then digits -- with NO other
   variation such as a colon or the word 'thread'):
   "You can fix this by adding a null check before accessing the array
   [SO-<id>]. If the error persists after that, verify your build
   configuration matches the recommended setup [SO-<id1>][SO-<id2>]."
   (The numbers above are just an example format, not real sources --
   always use the actual [SO-<id>] labels from the context given to you
   above, never invent a number that is not one of those labels.)
   ```

   A complete, fluent, *quotable* sentence with citation markers inline —
   exactly the pattern an LLM reproduces when it has nothing confident to
   say, regardless of the parenthetical warning right next to it.

## 5. Options considered

**(i) Dynamic real-context-ID example** — build the worked-example sentence
using a real `question_id` from that question's own retrieved context (e.g.
"cite as `[SO-38118194]`"), so there's no placeholder and no fabricated
number. **Rejected.** Using a real context ID in the example risks the model
copying that *valid-looking* ID onto a claim it doesn't actually support —
that would make NF2 look fine (a well-formed, in-context citation) while
*hiding* the underlying failure (an uncited/unsupported claim), which is
worse than the visible placeholder-copying failure mode it would replace.

**(ii) Keep v2, make the citation-detection pattern more tolerant.**
Rejected for a different reason: not applicable here. Zero of the 9 v2
failures are format near-misses — the detector already tolerates `[SO:123]`/
`[SO 123]`/`[SO thread 123]`. Loosening the pattern further can't fix "no
citation" (b) or "placeholder, no real id at all" (c) — there's no digit
substring to match in either case.

**(iii) Revert to v1, accept digit-copying as a documented limitation.**
Rejected: regresses back to the numeric-ID copying risk (D/411756 shows v1
did exactly that) without fixing category (b) at all, so it isn't a net
improvement — just a different flavor of the same root problem.

## 6. Decision: prompt v3 (implemented)

Two changes, applied to every builder (B/C/D); Condition A's text is
untouched (it never mentions citations):

1. **Remove the full worked-example sentence.** Keep only a short, non-
   quotable format description ("each labeled with its own source thread
   id in square brackets — format: the letters SO, a hyphen, then the
   thread's numeric id"). No digits, no `<id>`-style placeholder, no
   complete sentence anywhere in the instruction text.
2. **New instruction:** only cite a label that is literally present in the
   given context *and* actually supports the claim; if no provided source
   supports a claim, write that claim without a citation; never invent or
   output a placeholder label.

`PROMPT_VERSION` bumped to `"v3"` (shared constant, `llm/prompts.py`) — all
four conditions record `prompt_version="v3"` in their `run_history.jsonl`
once re-run, including Condition A (so a four-way comparison always shares
one version, even though A's prompt *text* didn't change). v1/v2 code paths,
result files, and `run_history.jsonl` entries are untouched for
reproducibility — nothing historical was rewritten or deleted.

**Companion NF2-validator fix (`llm/citations.py`):** `extract_citations()`
previously classified a placeholder-only answer as "no citation attempt at
all" (`has_citation=False`), because its digit-only pattern simply never
matched non-digit placeholder content — categories (b) and (c) were
indistinguishable from the metric's point of view. A new broader pattern
(`CITATION_TOKEN_PATTERN`, matches any `[SO...]`-shaped bracket token, digit
or not) makes `has_citation=True` for a placeholder while `has_valid_citation`
correctly stays `False` — "attempted but invalid" now reads as invalid, not
as silently absent. `cited_ids`/`valid_ids` are unaffected (a placeholder has
no real ID to report).

## 7. What this report does not resolve

Condition B's 3 no-citation-attempt failures (category b) look like an
independent, B-specific weakening of citation discipline — not an
example-copying artifact, since B never had a worked-example sentence removed
that it previously relied on in the same way C/D did. This would need its
own investigation if it's worth closing; prompt v3 does not target it.

Prompt v3 has not been exercised with a real generation run yet. The next
step, when the thesis author chooses to re-run, is comparing v3's actual
placeholder-copying rate against this report's v2 baseline on the same n=10
pilot sample (or the full n=384 run) to confirm the fix reduces category (c)
without introducing a new failure mode.
