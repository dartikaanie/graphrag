"""Task 2a: build (NOT send) the sufficiency prompts for the 70 hallucinated official-C answers. No API calls."""
import json,sys,re,collections
sys.path.insert(0,"llm/evaluation")
from judge_panel_aggregate import aggregate_labels
from judge_panel_config import PANEL_V2_JUDGES
import tiktoken
enc=tiktoken.get_encoding("cl100k_base")
GEN="llm/c_graphrag/results/condition_c_openai_gpt-4o-mini_n384_seed42_fw0-7-0-3_v4-alpha0-5_v3_95d2f39d2c.jsonl"
G={int(json.loads(l)["question_id"]):json.loads(l) for l in open(GEN)}
J=collections.defaultdict(dict)
for l in open("llm/evaluation/results/panel_official_R5_judge-v2.3.jsonl"):
    r=json.loads(l); J[int(r["question_id"])][r["judge_id"]]=r
SYSTEM=("You audit a retrieval context. You are given a programming question, the retrieved context a system was shown, and "
 "one or more claims that were found to be incorrect. Use ONLY the context text; do not use your own knowledge to decide what the context contains. "
 "Return JSON only.")
TEMPLATE="""QUESTION:
{question}

RETRIEVED CONTEXT (exactly as shown to the system):
{context}

CLAIMS FOUND INCORRECT:
{claims}

Answer two things.
1. sufficiency: does the context contain enough information to answer the question correctly?
   SUFFICIENT = all key information needed is present; PARTIAL = some but not all; INSUFFICIENT = little or none.
2. contradicting_evidence: does the context contain text that contradicts, or shows as wrong, at least one of the incorrect claims? YES or NO.
   If YES, quote the shortest span (max 200 chars) copied verbatim from the context, and give its label (e.g. SO-123).

Return exactly:
{{"sufficiency":"SUFFICIENT|PARTIAL|INSUFFICIENT","contradicting_evidence":"YES|NO","quote":"<verbatim span or empty>","quote_label":"<SO-id or empty>","reason":"<one sentence>"}}"""
items=[]
for q,jr in sorted(J.items()):
    labs={j:(x["label"] if x.get("parse_ok") and not x.get("call_failed") else None) for j,x in jr.items()}
    if aggregate_labels(labs,"panel-v2")["panel_label"] not in("HALUSINASI_PENUH","HALUSINASI_SEBAGIAN"): continue
    msg=G[q]["prompt_messages"][-1]["content"]
    ctx=msg[msg.index("[SO-"):msg.index("\n\nIMPORTANT")].strip()
    qt=msg[msg.index("Please, explain how to fix"):msg.index("\n\nReminder")].strip()
    claims=[]
    for j,x in jr.items():
        if labs[j] in("HALUSINASI_PENUH","HALUSINASI_SEBAGIAN"):
            for c in x.get("claims") or []:
                if c["verdict"] in("CONTRADICTED","FABRICATED") and c["claim"] not in claims: claims.append(c["claim"])
    items.append({"question_id":q,"n_claims":len(claims),"user":TEMPLATE.format(question=qt,context=ctx,claims="\n".join(f"- {c}" for c in claims[:5]))})
assert len(items)==70,len(items)
with open("reports/followup/sufficiency_prompts_DRAFT.jsonl","w") as f:
    for it in items: f.write(json.dumps({**it,"system":SYSTEM})+"\n")
tk=[len(enc.encode(SYSTEM+it["user"])) for it in items]
OUT=260
print("items",len(items),"input tok/call min/mean/max",min(tk),sum(tk)//70,max(tk),"claims mean",sum(i["n_claims"] for i in items)/70)
tot=0
for j in PANEL_V2_JUDGES:
    c=sum(tk)/1e6*j.price_per_m_input+70*OUT/1e6*j.price_per_m_output; tot+=c
    print(j.judge_id,j.model,f"calls=70 in={sum(tk)} out~{70*OUT} cost=${c:.4f}")
print(f"TOTAL calls=210 in={3*sum(tk)} out~{3*70*OUT} cost=${tot:.4f} (x1.5 safety ${tot*1.5:.4f}; retries up to 2 -> worst case ${tot*3:.4f})")
for it in (items[0],items[35]):
    print("\n"+"="*30,"EXAMPLE question_id",it["question_id"]); print("[SYSTEM]",SYSTEM); print("[USER]",it["user"])
