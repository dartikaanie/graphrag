import json, sys, random, statistics as st, collections, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; D = ROOT / "reports/followup/trust_influence"
sys.path.insert(0, str(ROOT / "llm/evaluation"))
from judge_panel_aggregate import aggregate_labels
OFF = ROOT / "llm/c_graphrag/results/condition_c_openai_gpt-4o-mini_n384_seed42_fw0-7-0-3_v4-alpha0-5_v3_95d2f39d2c.jsonl"
def load(p): return {r["question_id"]: r for r in map(json.loads, open(p))}
off = load(OFF); a5 = load(glob.glob(str(D / "raw_alpha0.5/condition_c_*[0-9a-f].jsonl"))[0]); a0 = load(glob.glob(str(D / "raw_alpha0.0/condition_c_*[0-9a-f].jsonl"))[0])
assert len(off) == len(a5) == len(a0) == 384 and set(off) == set(a5) == set(a0)
ids = lambda r: [c["answer_id"] for c in r["retrieved_context"]]
# step 1: reproduction
rep_diff = [q for q in off if set(ids(off[q])) != set(ids(a5[q]))]
rank_diff = [q for q in off if ids(off[q]) != ids(a5[q])]
print("step1 differing sets:", len(rep_diff), "differing order:", len(rank_diff))
# hallucination groups
J = collections.defaultdict(dict)
for l in open(ROOT / "llm/evaluation/results/panel_official_R5_judge-v2.3.jsonl"):
    r = json.loads(l); J[int(r["question_id"])][r["judge_id"]] = r["label"] if r.get("parse_ok") and not r.get("call_failed") else None
H = {q for q, labs in J.items() if aggregate_labels(labs, "panel-v2")["panel_label"] in ("HALUSINASI_PENUH", "HALUSINASI_SEBAGIAN")}
assert len(H) == 70 and H <= set(off)
def origin(c): return "expansion" if c["source_stage"] == "semantic_expansion" else f"hop{c['hop']}"
def items(r): return r["retrieved_context"]
qs = sorted(off)
changed = {q: len(set(ids(a5[q])) - set(ids(a0[q]))) for q in qs}   # items replaced (0-5)
def boot(vals, seed=12345):
    rng = random.Random(seed); n = len(vals); m = []
    for _ in range(2000): m.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    m.sort(); return m[49], m[1949]
def block(qset, label):
    out = {"n_questions": len(qset)}
    out["n_changed"] = sum(changed[q] > 0 for q in qset); out["share_changed"] = out["n_changed"] / len(qset)
    out["mean_replaced"] = st.mean(changed[q] for q in qset)
    for name, run in (("a0.0", a0), ("a0.5", a5)):
        it = [c for q in qset for c in items(run[q])]; n = len(it)
        o = collections.Counter(origin(c) for c in it)
        tw = [c["trust_weight"] for c in it]; ne = [c for c in it if origin(c) != "expansion"]
        out[name] = {"n_items": n, **{f"share_{k}": o[k] / n for k in ("hop1", "hop2", "expansion")},
                     "tw_mean": st.mean(tw), "tw_median": st.median(tw), "share_tw_lt05": sum(t < .5 for t in tw) / n,
                     "share_tw_lt05_noexp": sum(c["trust_weight"] < .5 for c in ne) / len(ne), "n_noexp": len(ne),
                     "sim_mean": st.mean(c["sim"] for c in it)}
    return out
res = {"all": block(qs, "all"), "halluc": block(sorted(H), "H"), "other": block(sorted(set(qs) - H), "O")}
v = [1.0 if changed[q] > 0 else 0.0 for q in qs]; res["ci_change_rate"] = boot(v); res["ci_mean_replaced"] = boot([changed[q] for q in qs])
res["dist_replaced"] = dict(sorted(collections.Counter(changed.values()).items()))
# pool + LIMIT 50
hc = {}
for tag in ("0.5", "0.0"):
    calls = json.load(open(D / f"hopcounts_alpha{tag}.json")); hc[tag] = {tuple(c["anchors"]): c for c in calls}
pool = {tag: [r["candidate_pool_size"] for r in run.values()] for tag, run in (("0.5", a5), ("0.0", a0))}
res["pool"] = {t: {"mean": st.mean(p), "median": st.median(p), "min": min(p), "max": max(p)} for t, p in pool.items()}
res["pool_identical_across_alpha"] = all(a5[q]["candidate_pool_size"] == a0[q]["candidate_pool_size"] for q in qs)
bind = {}; mism = 0
for tag, run in (("0.5", a5), ("0.0", a0)):
    n2 = []
    for q, r in run.items():
        c = hc[tag].get(tuple(r["anchor_question_ids"])); n2.append(c["n_hop2"] if c else None)
    miss = sum(x is None for x in n2); n2 = [x for x in n2 if x is not None]
    bind[tag] = {"matched": len(n2), "unmatched": miss, "n_hop2_mean": st.mean(n2), "n_binds": sum(x == 50 for x in n2)}
res["limit50"] = bind
res["candidate_sets_equal"] = all(a5[q]["graph_candidate_answer_ids"] == a0[q]["graph_candidate_answer_ids"] for q in qs)
json.dump(res, open(D / "results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
