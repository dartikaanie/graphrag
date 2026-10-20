"""
cost_projection.py
==================
Cost projection for alpha stage 2 and the official runs (Amendment 14 draft,
plan steps 6c / 7b), from ACTUAL per-call token counts in stored outputs:

  generation (OpenAI gpt-4o-mini): mean prompt / completion tokens per answer
    of the archived dev runs of the same condition and grounding (full input
    price, i.e. prompt caching ignored -> conservative);
  judging (DeepInfra, panel-v2): mean prompt / completion tokens per call and
    judge from the judge-v2.3 calibration records (Llama, DeepSeek) and the
    Qwen3-235B screening records, by condition (A / B-plain / B-grounded).

No API call. Every total carries a 10% retry/variance buffer.

    python3 llm/cost_projection.py            # writes results/cost_projection.{json,md}
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "llm" / "evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from llm import archive_paths  # noqa: E402

BUFFER = 0.10
GEN_PRICE = {"in": 0.15, "out": 0.60}  # gpt-4o-mini USD / 1M tokens (judge_clients registry; calibration_generate)
N_TEST, N_DEV = 384, 50

# Stored generation runs used as the token basis (original paths; read through the archive resolver).
GEN_BASIS = {
    ("A", None): "llm/a_pure_llm/results/condition_a_openai_gpt-4o-mini_n50_seed42_dev_v3_bbef0a3acc.jsonl",
    ("B", False): "llm/b_rag/results/condition_b_openai_gpt-4o-mini_n35_seed42_plain_dev_v3_c8fb5cfbe0.jsonl",
    ("B", True): "llm/b_rag/results/condition_b_openai_gpt-4o-mini_n35_seed42_grounded_dev_v3_0e2be4937f.jsonl",
    ("C", True): "llm/c_graphrag/results/condition_c_openai_gpt-4o-mini_n35_seed42_fw0-7-0-3_v3-alpha0-5_dev_v3_0900a9c562.jsonl",
    ("D", True): "llm/d_lightrag/results/condition_d_openai_gpt-4o-mini_n35_seed42_grounded_dev_v3_405b8ba0f3.jsonl",
}
# No plain (grounding off) dev run exists for C or D: their completion tokens are scaled by B's plain/grounded ratio.
SCALED_FROM_B = {("C", False): ("C", True), ("D", False): ("D", True)}

JUDGE_BASIS = {
    "llama": "llm/evaluation/results/panel_calibration_judge-v2.3.jsonl",
    "deepseek": "llm/evaluation/results/panel_calibration_judge-v2.3.jsonl",
    "qwen3-235b": "llm/evaluation/results/screening_a14_qwen3-235b_judge-v2.3.jsonl",
}
JUDGE_LABEL = {("A", None): "A [dev]", False: "B-plain [dev]", True: "B-grounded [dev]"}

STAGE2_RUNS = [("S2-a0.25-on", "C", True), ("S2-a0.25-off", "C", False), ("S2-a0-on", "C", True), ("S2-a0-off", "C", False)]
OFFICIAL_STAGE1 = [("R1", "A", None), ("R3", "B", True), ("R5", "C", True), ("R7", "C", True), ("R9", "D", True)]
OFFICIAL_STAGE2 = [("R2", "B", False), ("R4", "C", False), ("R6", "C", False), ("R8", "D", False)]


def _read(rel: str) -> list[dict]:
    with open(archive_paths.resolve(rel)) as f:
        return [json.loads(line) for line in f if line.strip()]


def generation_tokens() -> dict:
    out = {}
    for key, rel in GEN_BASIS.items():
        usage = [r.get("usage_full") or {} for r in _read(rel)]
        out[key] = {"in": st.mean(u.get("prompt_tokens", 0) for u in usage),
                    "out": st.mean(u.get("completion_tokens", 0) for u in usage), "n": len(usage), "basis": rel}
    ratio = out[("B", False)]["out"] / out[("B", True)]["out"]
    for key, src in SCALED_FROM_B.items():
        out[key] = {"in": out[src]["in"], "out": out[src]["out"] * ratio, "n": out[src]["n"],
                    "basis": f"{out[src]['basis']} (completion x {ratio:.3f}, B plain/grounded ratio)"}
    return out


def judge_tokens() -> dict:
    from judge_panel_config import PANEL_V2_JUDGES
    prices = {j.judge_id: (j.price_per_m_input, j.price_per_m_output) for j in PANEL_V2_JUDGES}
    out = {}
    for jid, rel in JUDGE_BASIS.items():
        recs = [r for r in _read(rel) if r.get("judge_id") == jid]
        for key, label in JUDGE_LABEL.items():
            rs = [r for r in recs if r.get("run_label") == label]
            attempts = st.mean(r.get("n_paid_attempts") or 1 for r in rs)
            out[(jid, key)] = {"in": st.mean(r["prompt_tokens"] for r in rs) * attempts,
                               "out": st.mean(r["completion_tokens"] for r in rs) * attempts,
                               "n": len(rs), "price": prices[jid], "basis": rel}
    return out


def _judge_key(condition, grounding):
    return ("A", None) if condition == "A" else grounding


def project(runs: list[tuple], n: int, gen: dict, jt: dict) -> dict:
    rows, tot = [], {"generation": 0.0, "judging": {j: 0.0 for j in JUDGE_BASIS}}
    for run_id, cond, grounding in runs:
        g = gen[(cond, grounding)]
        gen_cost = n * (g["in"] * GEN_PRICE["in"] + g["out"] * GEN_PRICE["out"]) / 1e6
        judges = {}
        for jid in JUDGE_BASIS:
            t = jt[(jid, _judge_key(cond, grounding))]
            judges[jid] = n * (t["in"] * t["price"][0] + t["out"] * t["price"][1]) / 1e6
            tot["judging"][jid] += judges[jid]
        tot["generation"] += gen_cost
        rows.append({"run": run_id, "condition": cond, "grounding": grounding, "n": n,
                     "generation_usd": round(gen_cost, 4), "judging_usd": {k: round(v, 4) for k, v in judges.items()},
                     "judging_total_usd": round(sum(judges.values()), 4)})
    judging = sum(tot["judging"].values())
    return {"runs": rows, "n_answers": n * len(runs),
            "generation_usd": round(tot["generation"], 4),
            "judging_usd_per_judge": {k: round(v, 4) for k, v in tot["judging"].items()},
            "judging_usd": round(judging, 4),
            "with_buffer": {"generation_usd": round(tot["generation"] * (1 + BUFFER), 4),
                            "judging_usd_per_judge": {k: round(v * (1 + BUFFER), 4) for k, v in tot["judging"].items()},
                            "judging_usd": round(judging * (1 + BUFFER), 4)}}


def build() -> dict:
    gen, jt = generation_tokens(), judge_tokens()
    return {
        "buffer": BUFFER, "generation_price_per_m": GEN_PRICE,
        "basis": {"generation": {f"{c}|{g}": v for (c, g), v in gen.items()},
                  "judging": {f"{j}|{k}": v for (j, k), v in jt.items()}},
        "alpha_stage2": project(STAGE2_RUNS, N_DEV, gen, jt),
        "official_stage1": project(OFFICIAL_STAGE1, N_TEST, gen, jt),
        "official_stage2": project(OFFICIAL_STAGE2, N_TEST, gen, jt),
    }


def markdown(p: dict) -> str:
    lines = ["# Cost projection (stored token counts, +10% buffer)", "",
             "Generation: OpenAI gpt-4o-mini $0.15 / $0.60 per 1M tokens (prompt caching ignored). Judging: DeepInfra "
             "panel-v2 (Llama $0.10/$0.32, DeepSeek $0.26/$0.38, Qwen3-235B $0.09/$0.55). C and D are judged with the "
             "token means of B's records of the same grounding (no C/D judge-v2.3 records exist).", ""]
    for key, title in (("alpha_stage2", "Alpha stage 2 (4 x 50 dev answers)"),
                       ("official_stage1", "Official stage 1: R1, R3, R5, R7, R9 (5 x 384)"),
                       ("official_stage2", "Official stage 2: R2, R4, R6, R8 (4 x 384)")):
        s = p[key]
        lines += [f"## {title}", "", "| Run | n | Generation | Llama | DeepSeek | Qwen3-235B | Judging |", "|---|---|---|---|---|---|---|"]
        for r in s["runs"]:
            j = r["judging_usd"]
            lines.append(f"| {r['run']} ({r['condition']}, grounding {r['grounding']}) | {r['n']} | ${r['generation_usd']:.3f} | "
                         f"${j['llama']:.3f} | ${j['deepseek']:.3f} | ${j['qwen3-235b']:.3f} | ${r['judging_total_usd']:.3f} |")
        b = s["with_buffer"]
        lines += [f"| **total** | {s['n_answers']} | ${s['generation_usd']:.3f} | ${s['judging_usd_per_judge']['llama']:.3f} | "
                  f"${s['judging_usd_per_judge']['deepseek']:.3f} | ${s['judging_usd_per_judge']['qwen3-235b']:.3f} | ${s['judging_usd']:.3f} |",
                  f"| **+10%** | | **${b['generation_usd']:.3f}** | ${b['judging_usd_per_judge']['llama']:.3f} | "
                  f"${b['judging_usd_per_judge']['deepseek']:.3f} | ${b['judging_usd_per_judge']['qwen3-235b']:.3f} | **${b['judging_usd']:.3f}** |", ""]
    return "\n".join(lines)


def main() -> int:
    p = build()
    out = REPO_ROOT / "results" / "cost_projection"
    out.with_suffix(".json").write_text(json.dumps(p, indent=1, default=str) + "\n")
    out.with_suffix(".md").write_text(markdown(p))
    print(markdown(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
