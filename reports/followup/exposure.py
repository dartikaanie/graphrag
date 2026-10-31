import json,sys,csv,random,collections,statistics as st
from pathlib import Path
sys.path.insert(0,"llm/evaluation")
from judge_panel_aggregate import aggregate_labels
OFF="OFFICIAL a=0.5 grounded (n=384)"
J=collections.defaultdict(dict)
for l in open("llm/evaluation/results/panel_official_R5_judge-v2.3.jsonl"):
    r=json.loads(l); J[int(r["question_id"])][r["judge_id"]]=r["label"] if r.get("parse_ok") and not r.get("call_failed") else None
grp={}
for q,labs in J.items():
    p=aggregate_labels(labs,"panel-v2")["panel_label"]
    grp[q]="H" if p in("HALUSINASI_PENUH","HALUSINASI_SEBAGIAN") else "ABS" if p=="ABSTAIN" else "N" if p=="FAKTUAL" else "UNRES"
print(collections.Counter(grp.values()))
items=collections.defaultdict(list)
for l in open("reports/context_provenance_C.jsonl"):
    r=json.loads(l)
    if r["config"]==OFF: items[r["question_id"]].append(r)
assert len(items)==384 and all(len(v)==5 for v in items.values())
Ts=[0.3,0.4,0.5,0.6]
def qmetrics(v):
    m={"mean_tw":st.mean(i["trust_weight"] for i in v),"mean_sim":st.mean(i["sim"] for i in v)}
    for T in Ts:
        lo=[i for i in v if i["trust_weight"]<T]; ne=[i for i in lo if i["origin"]!="expansion"]
        m[f"n_low_{T}"]=len(lo); m[f"n_low_noexp_{T}"]=len(ne)
        m[f"share_low_{T}"]=len(lo)/5; m[f"share_low_noexp_{T}"]=len(ne)/5
        m[f"any_low_{T}"]=float(len(lo)>0); m[f"any_low_noexp_{T}"]=float(len(ne)>0)
    return m
M={q:qmetrics(v) for q,v in items.items()}
H=[q for q in M if grp[q]=="H"]; N=[q for q in M if grp[q]=="N"]; A=[q for q in M if grp[q]=="ABS"]
print(len(H),len(N),len(A))
assert len(H)==70,"join mismatch"
keys=list(next(iter(M.values())).keys())
rng=random.Random(42)
def diff(h,n,k): return st.mean(M[q][k] for q in h)-st.mean(M[q][k] for q in n)
rows=[]; summ={}
for k in keys:
    d=diff(H,N,k)
    bs=[]
    for _ in range(2000):
        h=[rng.choice(H) for _ in H]; n=[rng.choice(N) for _ in N]; bs.append(diff(h,n,k))
    bs.sort(); lo,hi=bs[49],bs[1949]
    pool=H+N; cnt=0
    for _ in range(2000):
        rng.shuffle(pool); cnt+= abs(diff(pool[:70],pool[70:],k))>=abs(d)-1e-12
    p=(cnt+1)/2001
    rows.append({"metric":k,"hallucinated_mean":st.mean(M[q][k] for q in H),"not_halluc_mean":st.mean(M[q][k] for q in N),"abstain_mean":st.mean(M[q][k] for q in A),
        "diff_H_minus_N":d,"boot95_lo":lo,"boot95_hi":hi,"perm_p":p,"n_H":len(H),"n_N":len(N),"n_ABS":len(A)})
with open("reports/followup/exposure_table.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(f"{r['metric']:22s} H={r['hallucinated_mean']:.3f} N={r['not_halluc_mean']:.3f} A={r['abstain_mean']:.3f} d={r['diff_H_minus_N']:+.3f} [{r['boot95_lo']:+.3f},{r['boot95_hi']:+.3f}] p={r['perm_p']:.3f}")
