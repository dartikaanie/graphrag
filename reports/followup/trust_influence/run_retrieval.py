"""Retrieval-only replay of official C (v4, 384 test Qs) at a given alpha. No LLM calls, no pipeline edits.
Only run-time patches: (1) skip the CLI's 'alpha must be official' guard, (2) skip run_history append,
(3) record hop-1/hop-2 candidate counts per traverse_graph call (sidecar json).
usage: python3 run_retrieval.py <alpha>"""
import sys, json, atexit
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "llm/c_graphrag")]
import os; os.chdir(ROOT / "llm/c_graphrag")
import c_graphrag as cg
alpha = float(sys.argv[1]); out = ROOT / "reports/followup/trust_influence"
cg.official_role_for_args = lambda args, official=None: "c_primary"
cg.append_run_history = lambda rec: None
calls = []; orig = cg.traverse_graph
def tg(driver, database, anchor_ids, *a, **k):
    r = orig(driver, database, anchor_ids, *a, **k)
    calls.append({"anchors": list(anchor_ids), "n_hop1": sum(c["hop"] == 1 for c in r), "n_hop2": sum(c["hop"] == 2 for c in r)})
    return r
cg.traverse_graph = tg
atexit.register(lambda: (out / f"hopcounts_alpha{alpha}.json").write_text(json.dumps(calls)))
sys.argv = ["c_graphrag.py", "--retrieval-only", "--c-retrieval-version", "v4", "--alpha", str(alpha), "--n-sample", "384",
            "--seed", "42", "--top-k", "5", "--max-hops", "2", "--output-dir", str(out / f"raw_alpha{alpha}")]
cg.main()
