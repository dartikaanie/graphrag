# Alpha stage 2: plan and commands (DRAFT; nothing below has been run)

Prerequisites: Amendment 14 approved and committed; the author's explicit "go"; Neo4j running (each C run reads the
graph's `trustScoreVersion` at start, and `stage2_report.py` rejects a run whose manifest does not record
`v3-logscore`).

## a. Which stage-2 runs exist

None. Every archived Condition C dev run with alpha 0.25 or 0 is retrieval-only (stage 1; no generated answer, no
hallucination judging), see `results/_archive/2026-10-19_pre-official/ARCHIVE_INDEX.md`, "Stage-2 inputs". All four
runs are generated: alpha in {0.25, 0} x grounding on / off, 50 dev questions each (positions 385-434), 200 answers.

## b. Commands (from the repo root)

1. Generate (OpenAI gpt-4o-mini; settings pinned as in `llm/official_run_matrix.py`, dev split):

```bash
cd llm/c_graphrag
for A in 0.25 0; do for G in --require-grounding --no-require-grounding; do
  python3 c_graphrag.py --n-sample 50 --seed 42 --oversample-pool 1536 --sample-split dev \
    --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --n-anchor 3 \
    --n-semantic-expansion 3 --c-retrieval-version v3 --alpha $A --fusion-mode trust_weighted $G \
    --expansion-require-answer --leakage-mode thread
done; done
cd ../..
```

2. Register the four outputs as `dev-stage2` (checked against the frozen values, including per-answer end-to-end
   timing of record schema r3; refused otherwise). The DeepInfra budget is already set to $9.00 (2026-10-20):

```bash
python3 -m llm.run_registry register --status dev-stage2 llm/c_graphrag/results/condition_c_openai_gpt-4o-mini_n50_seed42_*_v3-alpha0-25_dev_v3_*.jsonl llm/c_graphrag/results/condition_c_openai_gpt-4o-mini_n50_seed42_*_v3-alpha0-0_dev_v3_*.jsonl
python3 -m llm.run_registry show   # budget $9.00, spent $0.0937 (screening), remaining $8.9063
```

3. Judge all 200 answers with the frozen panel-v2 (judge-v2.3, pv2.3-1, blind-v3, JSON mode, temperature 0). The
   runner refuses unregistered inputs, an unset budget, a projection above the remaining budget or above
   `--max-cost-usd`, and never lets the spend exceed `--max-cost-usd` at runtime; it records the spend and registers
   the judge run as `dev-stage2`:

```bash
cd llm/evaluation
python3 judge_panel.py --panel panel-v2 --judge-version judge-v2.3 \
  --run-files <the 4 files from step 1> \
  --token-basis results/panel_calibration_judge-v2.3.jsonl results/screening_a14_qwen3-235b_judge-v2.3.jsonl \
  --out results/panel_stage2_judge-v2.3.jsonl --max-cost-usd 1.00 --dry-run   # then the same without --dry-run
cd ../..
```

   Any judge record still failed after the retries is in `<out>_failures.jsonl`; re-running the same command retries
   only those (cached records are not re-paid). Stage 2 refuses to report while any record is missing.

4. Stage-2 report with the PANEL label (binary hallucination rate excluding ABSTAIN, pooled over grounding; tie
   margin 2 answers; tie-breakers: secondary criterion, then larger alpha; per-judge results alongside):

```bash
python3 llm/c_graphrag/stage2_report.py --run-files <the 4 files> \
  --judge-files llm/evaluation/results/panel_stage2_judge-v2.3.jsonl \
  --judge-version judge-v2.3 --panel panel-v2            # review the report first
python3 llm/c_graphrag/stage2_report.py ... --apply      # then: writes alpha* / alpha_abl to official_params.json
```

   `--apply` sets `official_params.json` to `status: "selected"` with `alpha_primary` (the winner) and
   `alpha_ablation` (the other candidate) and the current commit; this unlocks Condition C on the test sample
   (`official_params.check_c_test_run`). Then commit the report, `official_params.json` and the Outcome text.

## c. Cost projection (`results/cost_projection.md`, stored token counts, +10%)

| | Generation (OpenAI) | Llama | DeepSeek | Qwen3-235B | Judging (DeepInfra) |
|---|---|---|---|---|---|
| 4 x 50 answers | $0.112 | $0.151 | $0.366 | $0.175 | $0.692 |
| **+10%** | **$0.123** | $0.166 | $0.403 | $0.192 | **$0.761** |
