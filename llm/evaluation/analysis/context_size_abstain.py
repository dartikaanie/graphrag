"""Read-only: context size vs panel ABSTAIN, official n=384 runs C and D. Run from repo root.
Writes results/context_size_abstain.md. No generation/judge/API calls."""
import json, random, statistics as st
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "llm" / "evaluation" / "results"
reg = json.load(open(ROOT / "results" / "run_registry.json"))
gen = {r["condition"]: r for r in reg["runs"] if r["status"] == "official" and r["kind"] == "generation"}
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
SHORT = {"FAKTUAL": "FAKTUAL", "HALUSINASI_SEBAGIAN": "SEBAGIAN", "HALUSINASI_PENUH": "PENUH", "ABSTAIN": "ABSTAIN"}
ORDER = ["FAKTUAL", "SEBAGIAN", "PENUH", "ABSTAIN"]
EXPECT = {"C": [242, 4, 66, 72], "D": [241, 9, 43, 91]}
CELL = {"C": "R5", "D": "R9"}

data = {}
for c in "CD":
    lab = {int(r["question_id"]): SHORT.get(r["panel_label"], r["panel_label"]) for r in jl(RES / f"panel_official_{CELL[c]}_judge-v2.3_panel.jsonl")}
    ctx = {}
    for a in jl(ROOT / gen[c]["files"]["output"]):
        rc = a.get("retrieved_context") or []
        ctx[int(a["question_id"])] = (len(rc), sum(len(i.get("chunk_text") or "") for i in rc))
    data[c] = (lab, ctx)
    cnt = Counter(lab.values()); got = [cnt.get(k, 0) for k in ORDER]
    if got != EXPECT[c] or set(lab) != set(ctx):
        raise SystemExit(f"LABEL MISMATCH {c}: got {got} expected {EXPECT[c]}; cnt={dict(cnt)}")

out = ["# Context size vs ABSTAIN, official n=384 (C vs D)\n",
       "Label counts verified: C = 242/4/66/72, D = 241/9/43/91 (FAKTUAL/SEBAGIAN/PENUH/ABSTAIN).\n",
       "Context size = number of items in `retrieved_context`; chars = sum of `chunk_text` length.\n",
       "## 1. Context size\n", "| cond | metric | mean | median | min | max | N ctx=0 |", "|---|---|---|---|---|---|---|"]
for c in "CD":
    for i, m in ((0, "items"), (1, "chars")):
        v = [x[i] for x in data[c][1].values()]
        out.append(f"| {c} | {m} | {st.mean(v):.2f} | {st.median(v):g} | {min(v)} | {max(v)} | {sum(x == 0 for x in v)} |")

def bins(key):
    pooled = [data[c][1][q][key] for c in "CD" for q in data[c][1]]
    nz = [x for x in pooled if x > 0]
    e = np.quantile(nz, [.25, .5, .75]) if nz else []
    return e
def binof(x, e):
    return "ctx=0" if x == 0 else f"Q{1 + int(np.searchsorted(e, x, side='left'))}"
for key, name in ((0, "items"), (1, "chars")):
    e = bins(key)
    out += [f"\n## 2{'ab'[key]}. ABSTAIN by context-size bin ({name}); ctx=0 separate, Q1-Q4 = quartiles of pooled non-zero C+D values (edges {', '.join(f'{x:g}' for x in e)}; bin = value <= edge)\n",
            "| cond | bin | N | ABSTAIN | % | FAKTUAL | SEBAGIAN | PENUH |", "|---|---|---|---|---|---|---|---|"]
    for c in "CD":
        lab, ctx = data[c]
        for b in ["ctx=0", "Q1", "Q2", "Q3", "Q4"]:
            qs = [q for q in lab if binof(ctx[q][key], e) == b]
            k = Counter(lab[q] for q in qs)
            out.append(f"| {c} | {b} | {len(qs)} | {k['ABSTAIN']} | {100*k['ABSTAIN']/len(qs):.1f}% | {k['FAKTUAL']} | {k['SEBAGIAN']} | {k['PENUH']} |" if qs else f"| {c} | {b} | 0 | 0 | - | 0 | 0 | 0 |")

(lc, xc), (ld, xd) = data["C"], data["D"]
qs = sorted(q for q in ld if ld[q] == "ABSTAIN" and lc[q] != "ABSTAIN")
n0 = sum(xd[q][0] == 0 for q in qs); lt = sum(xd[q][0] > 0 and xd[q][0] < xc[q][0] for q in qs); ge = sum(xd[q][0] > 0 and xd[q][0] >= xc[q][0] for q in qs)
out += [f"\n## 3. D=ABSTAIN and C!=ABSTAIN (items): N = {len(qs)}\n",
        f"- D context = 0: {n0}", f"- D context > 0 and < C context: {lt}", f"- D context >= C context (D>0): {ge}",
        f"- (sum check: {n0 + lt + ge})",
        f"- By chars: D chars < C chars: {sum(xd[q][1] < xc[q][1] for q in qs)}; D chars >= C chars: {sum(xd[q][1] >= xc[q][1] for q in qs)}",
        f"- D items < 5 among these: {sum(xd[q][0] < 5 for q in qs)}",
        "\n10 random examples (random.Random(42).sample over sorted question_ids):\n",
        "| question_id | C ctx | D ctx | C label | D label |", "|---|---|---|---|---|"]
for q in random.Random(42).sample(qs, min(10, len(qs))):
    out.append(f"| {q} | {xc[q][0]} | {xd[q][0]} | {lc[q]} | {ld[q]} |")
(ROOT / "results" / "context_size_abstain.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
