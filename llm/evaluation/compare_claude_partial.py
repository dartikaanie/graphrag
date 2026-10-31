"""
compare_claude_partial.py
-------------------------
Compare the Claude sensitivity judge with the three official judges and the official panel, restricted to the items Claude
has judged (partial/subset runs). Read-only; prints a report and writes results/sensitivity_claude/compare_partial.md.

    python llm/evaluation/compare_claude_partial.py [--sens-dir DIR]
"""
import argparse
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import summarize_claude_sensitivity as scs  # noqa: E402
from judge_binary import CATEGORY_ABSTAIN, CATEGORY_HALLUCINATED, category_of  # noqa: E402
from judge_panel_aggregate import aggregate_labels, panel_category  # noqa: E402

H, N, A = CATEGORY_HALLUCINATED, "NOT_HALLUCINATED", CATEGORY_ABSTAIN


def kappa(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    n = len(pairs)
    if not n:
        return None, 0, None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / n / n
    return (None if pe == 1 else round((po - pe) / (1 - pe), 3)), n, round(po, 3)


def wilson(k, n, z=1.96):
    if not n:
        return "n/a"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / d
    return f"{100 * p:.0f}% [{100 * (c - h):.0f}-{100 * (c + h):.0f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens-dir", default=str(HERE.parents[1] / "results" / "sensitivity_claude"))
    args = ap.parse_args()
    data, cells = scs.load_records()
    scs.load_claude(Path(args.sens_dir), data)
    J = scs.ALL_JUDGES
    items = [(c, q) for c in "ABCD" for q in sorted(data[c][scs.CLAUDE])]
    cats = {j: [category_of(data[c][j][q]) if q in data[c][j] else None for c, q in items] for j in J}
    panel = [panel_category(aggregate_labels({j: scs.label_of(data[c][j].get(q)) for j in scs.OFFICIAL_JUDGES})["panel_label"])
             for c, q in items]
    L = [f"# Claude vs official judges on the {len(items)} items Claude judged", "",
         "Per run: " + ", ".join(f"{c}={sum(1 for i in items if i[0] == c)}" for c in "ABCD"), "",
         "## Rates on these items (hallucinated / (hallucinated + not hallucinated); 95% Wilson interval)", "",
         "| Judge | hallucinated | not halluc. | ABSTAIN | rate [CI] |", "|---|---|---|---|---|"]
    for name, cs in [*[(j, cats[j]) for j in J], ("official panel (3)", panel)]:
        c = Counter(cs)
        L.append(f"| {name} | {c[H]} | {c[N]} | {c[A]} | {wilson(c[H], c[H] + c[N])} |")
    L += ["", "## Claude vs each judge (Cohen κ / raw agreement, 3 categories ABSTAIN|NOT|HALLUC, items both judged)", "",
          "| vs | κ | agree | n |", "|---|---|---|---|"]
    for name, cs in [*[(j, cats[j]) for j in scs.OFFICIAL_JUDGES], ("official panel (3)", panel)]:
        k, n, po = kappa(cats[scs.CLAUDE], cs)
        L.append(f"| {name} | {k} | {po} | {n} |")
    L += ["", "## Reference: κ between the official judges themselves (same items)", "", "| pair | κ | agree |", "|---|---|---|"]
    for a, b in combinations(scs.OFFICIAL_JUDGES, 2):
        k, n, po = kappa(cats[a], cats[b])
        L.append(f"| {a} / {b} | {k} | {po} |")
    L += ["", "## Binary (hallucinated yes/no; items where both judges answered)", "", "| vs | κ | agree | n | Claude halluc / other halluc |",
          "|---|---|---|---|---|"]
    for name, cs in [*[(j, cats[j]) for j in scs.OFFICIAL_JUDGES], ("official panel (3)", panel)]:
        pr = [(a, b) for a, b in zip(cats[scs.CLAUDE], cs) if a in (H, N) and b in (H, N)]
        k, n, po = kappa([a for a, _ in pr], [b for _, b in pr])
        L.append(f"| {name} | {k} | {po} | {n} | {sum(a == H for a, _ in pr)} / {sum(b == H for _, b in pr)} |")
    L += ["", "## Claude (rows) vs official panel (columns), 3 categories", "", "| | panel ABSTAIN | panel NOT | panel HALLUC | panel UNRESOLVED |", "|---|---|---|---|---|"]
    for r in (A, N, H, None):
        row = [sum(1 for a, b in zip(cats[scs.CLAUDE], panel) if a == r and b == col) for col in (A, N, H, None)]
        L.append(f"| Claude {r or 'missing'} | " + " | ".join(map(str, row)) + " |")
    L += ["", "## Per run: Claude vs official panel (hallucinated, items both answered)", "", "| Run | n | Claude halluc | panel halluc | same label |", "|---|---|---|---|---|"]
    for c in "ABCD":
        idx = [i for i, it in enumerate(items) if it[0] == c]
        pr = [(cats[scs.CLAUDE][i], panel[i]) for i in idx]
        bb = [(a, b) for a, b in pr if a in (H, N) and b in (H, N)]
        L.append(f"| {c} | {len(bb)} | {sum(a == H for a, _ in bb)} | {sum(b == H for _, b in bb)} | {sum(a == b for a, b in bb)} |")
    out = "\n".join(L) + "\n"
    print(out)
    (Path(args.sens_dir) / "compare_partial.md").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
