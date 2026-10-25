# Alpha stage 2: plan and commands

Updated for Amendment 17 (C retrieval v4, answerable-only anchors). The candidates are **derived** from the stage-1
report of record (`llm/c_graphrag/stage2_candidates.py` reads `stage1_anchors_report_20261025T165734Z.json`; top_two = [0.5, 0.25]), not written by
hand. Alpha 0 is an **exploratory reference** (pre-declared): it is run and reported like the candidates but the
selection rule never sees it.

Prerequisites: the author's explicit "go"; Neo4j running (each C run reads the graph's `trustScoreVersion` at start, and
`stage2_report.py` rejects a run whose manifest does not record `v3-logscore`). Stage 2 is a dev-split run, so it does not
need `AMENDMENT14_APPROVED` (`docs/AMENDMENT14_APPROVAL_CHANGE.md`, section 4).

## a. Runs

Six generated runs on the 50 dev questions (positions 385-434): alpha in {0.5, 0.25} (candidates) and
0 (exploratory) x grounding on / off = 300 answers.

## b. Commands (from the repo root)

1. Generate (OpenAI gpt-4o-mini, default generation settings; dev split; leakage `thread`):

```bash
cd llm/c_graphrag
for A in 0.5 0.25 0; do for G in --require-grounding --no-require-grounding; do
  python3 c_graphrag.py --n-sample 50 --seed 42 --oversample-pool 1536 --sample-split dev \
    --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --n-anchor 3 \
    --n-semantic-expansion 3 --c-retrieval-version v4 --alpha $A --fusion-mode trust_weighted $G \
    --expansion-require-answer --leakage-mode thread
done; done
cd ../..
```

2. Register the six outputs as `dev-stage2`:
   `python3 -m llm.run_registry register --status dev-stage2 <the six files>`.
3. Judge all 300 answers with the frozen panel-v2 (judge-v2.3), `--max-cost-usd` as agreed (dry run first):
   `python3 llm/evaluation/judge_panel.py --panel panel-v2 --judge-version judge-v2.3 --run-files <six files> --token-basis ... --out ... --max-cost-usd <cap>`.
4. Report: `python3 llm/c_graphrag/stage2_report.py --run-files <six files> --judge-files <panel file> --judge-version judge-v2.3 --panel panel-v2`
   (primary: hallucinated answers excluding ABSTAIN, pooled over grounding, tie margin 2 answers; secondary: ABSTAIN
   counted as failed, same margin; then the larger alpha). **Never** with `--apply` until the author decides: `--apply`
   writes alpha\* to `official_params.json`.

## c. Cost projection (`results/cost_projection.md`, stored token counts, +10%)

Four runs x 50 answers: generation $0.123, judging $0.761. Six runs: about 1.5x, i.e. generation about $0.18, judging
about $1.14, in total about $1.32.
