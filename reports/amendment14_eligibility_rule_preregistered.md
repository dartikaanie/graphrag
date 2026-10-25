# Amendment 14 (DRAFT, not committed, not approved): frozen judge panel without a human-agreement gate

Status: **draft for the author's review.** Only the section "Qwen judge eligibility rule (screening)" was written
before any screening call; the other sections are added later (step 5) and are marked as such.

## Qwen judge eligibility rule (screening)

*Recorded 2026-10-19T23:05Z (UTC), before any screening call. The SHA-256 of this file at screening time is stored
in every screening run's history entry (`judge_panel_run_history.jsonl`, field `screening.rule_sha256`).*

**Purpose.** Qwen2.5-72B-Instruct is replaced in the panel by a cheaper Qwen model. The replacement is chosen by this
rule only. This is **screening, not tuning**: the judge prompt (judge-v2.3), the parser (pv2.3-1), the blinding
(blind-v3), the request parameters (temperature 0, max_tokens 2000, JSON mode, the runner's retry policy of at most
2 retries) and the 2-of-3 aggregation stay frozen. No prompt, parser or threshold is changed after a screening result
is seen.

**Candidates, in this order** (DeepInfra, OpenAI-compatible API; listing checked 2026-10-19, price per 1M tokens):

1. `Qwen/Qwen3-235B-A22B-Instruct-2507`: non-thinking model by design ("updated version of the Qwen3-235B-A22B
   non-thinking mode"); listed with `json` / `structured-output` tags; FP8; $0.09 input / $0.55 output. Request
   parameters: none beyond the frozen ones.
2. Backup, only if 1 fails: `Qwen/Qwen3-32B`: hybrid thinking model; listed with `json`, `reasoning` and
   `non-reasoning` tags; FP8; $0.08 input / $0.28 output; the cheapest non-deprecated Qwen3 text model for this
   workload (Qwen3-14B $0.12 / $0.24, Qwen3-30B-A3B $0.12 / $0.50, Qwen3-Next-80B-A3B-Instruct $0.09 / $1.10).
   Thinking is disabled with `reasoning_effort: "none"` (DeepInfra's reasoning switch) and
   `chat_template_kwargs: {"enable_thinking": false}` (the Qwen3 chat-template switch), both sent in every request.
   Whether thinking was actually off is decided by criterion (3) on every response.

**Data.** The 60 frozen calibration items (`judge_panel.py --calibration-set`, the annotation freeze is verified),
judge-v2.3, one call per item plus the runner's retries. Cost cap `--max-cost-usd 0.10` per candidate.

**Eligibility criteria (all three must hold):**

1. **Not a degenerate rater:** among the items with a usable label whose category (`judge_binary.category_of`) is
   not ABSTAIN, at least 5% are HALLUCINATED (with 58 non-ABSTAIN items: at least 3).
2. **No candidate text presented as a reference quote:** for every claim, each `[R]`-prefixed quoted string in its
   evidence (straight or curly double quotes) is normalised (lower case, whitespace collapsed, surrounding
   punctuation and quote marks stripped). A **violation** is a quote that is **not** a substring of the normalised
   reference answer (`html_to_text` of the ground truth) **and is** a substring of the normalised candidate answer
   (raw or blinded). The number of violations must be **0**. Quotes found in neither text are counted and reported,
   but are not violations.
3. **Clean parsing, no reasoning content:** all 60 items have a usable record after the frozen retry policy
   (`parse_ok`, no failed call); no record has reasoning content (`reasoning_present`: a `reasoning_content` or
   `reasoning` field, or a `<think>` block); no raw response contains `<think>`. First-attempt parse failures are
   reported for information. A record answered by a different model than requested (`served_model_mismatch`)
   invalidates the screening.

**Information only (does not decide eligibility):** agreement with the frozen v1 human labels (3-category and binary
kappa, recall, false-alarm rate), the actual cost, and the latency.

**Decision.** If candidate 1 meets all three criteria, it is the panel's third judge (panel-v2 =
Llama-3.3-70B-Instruct-Turbo + DeepSeek-V3.2 (`reasoning_effort` "none") + the Qwen model; 2-of-3 aggregation
unchanged). Otherwise candidate 2 is screened the same way. If both fail, nothing is decided automatically: the
author decides (no automatic switch to a two-judge panel).

**Cost guard of the screening runs.** The runner's projection uses, for each item, the actual prompt-token count
already recorded for the identical judge messages (same `messages_sha256`) in the stored judge-v2.3 calibration
outputs (largest count over the stored judges), and `max_tokens` (2000) as the output upper bound for a model
without records. It also stops issuing calls at runtime when the spend so far plus the worst case of the next call
would exceed `--max-cost-usd`.
