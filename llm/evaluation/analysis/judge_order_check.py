"""Post-hoc, read-only: ordering of conditions A-D by hallucination rate per judge and panel.
Run from the repo root. Writes results/judge_order_check.md. Not part of the frozen analysis plan."""
import json, itertools, random
import numpy as np
from scipy.stats import spearmanr, kendalltau, binomtest
R={'A':1,'B':3,'C':5,'D':9}; JS=['llama','deepseek','qwen3-235b','panel']
D={}  # cond -> qid -> {judge: cat}
def cat(l): return 'ABS' if l=='ABSTAIN' else ('NOT' if l=='FAKTUAL' else 'HAL')
for c,r in R.items():
    D[c]={}
    for l in open(f'llm/evaluation/results/panel_official_R{r}_judge-v2.3_panel.jsonl'):
        d=json.loads(l); assert not d['panel_incomplete']
        e={j:cat(v) for j,v in d['judge_labels'].items()}; e['panel']=cat(d['panel_label']); D[c][d['question_id']]=e
Q=sorted(D['A']); assert all(sorted(D[c])==Q for c in D), 'question sets differ'
out=[]; P=out.append
def rates(j,v):
    res={}
    for c in R:
        x=[D[c][q][j] for q in Q]; h=x.count('HAL'); n=x.count('NOT'); a=x.count('ABS')
        res[c]=(h,n,a)
    return res
rt={}
P("# Judge order check (hallucination rate, conditions A-D)\n")
P("Read-only. Source: llm/evaluation/results/panel_official_R{1,3,5,9}_judge-v2.3_panel.jsonl (R1=A, R3=B, R5=C, R9=D); 384 questions, same list in all four, 0 incomplete, 0 missing labels.\n")
P("## Definitions used\n- Per judge: `judge_labels[judge]` in the panel file. FAKTUAL = not hallucinated; HALUSINASI_PENUH and HALUSINASI_SEBAGIAN = hallucinated; ABSTAIN = ABSTAIN (judge_binary.py:46-53 `category_of`, constants judge_binary.py:21-23).\n- Panel: `panel_label` (2-of-3 majority as stored).\n- (a) excl. ABSTAIN = hallucinated / (hallucinated + not hallucinated); (b) ABSTAIN as failed = (hallucinated + ABSTAIN) / all 384 (official_report.py:84-88, `three_numbers`).\n")
for ver in 'ab':
    P(f"## Version ({ver}): "+("ABSTAIN excluded" if ver=='a' else "ABSTAIN counted as failed")+"\n")
    for j in JS:
        rs=rates(j,ver); P(f"### {j}\n\n| cond | hallucinated | ABSTAIN | denominator | rate |\n|---|---|---|---|---|")
        rt[(j,ver)]={}
        for c,(h,n,a) in rs.items():
            num,den=(h,h+n) if ver=='a' else (h+a,h+n+a)
            rt[(j,ver)][c]=num/den
            P(f"| {c} | {num} | {a} | {den} | {100*num/den:.2f}% |")
        P("")
    P(f"### Rank order (lowest to highest rate), version ({ver})\n\n| judge | order | ties/near-ties (<1 pp) |\n|---|---|---|")
    ords={}
    for j in JS:
        r=rt[(j,ver)]; o=sorted(r,key=lambda c:r[c]); ords[j]=o
        nt=[f"{x}-{y} ({100*abs(r[x]-r[y]):.2f} pp)" for x,y in itertools.combinations('ABCD',2) if abs(r[x]-r[y])<0.01]
        P(f"| {j} | {' < '.join(o)} | {', '.join(nt) or 'none'} |")
    same=len({tuple(v) for v in ords.values()})==1
    P(f"\nOrder identical across the three judges and the panel: **{'YES' if same else 'NO'}**")
    if not same:
        for x,y in itertools.combinations('ABCD',2):
            sg={j:(rt[(j,ver)][x]>rt[(j,ver)][y]) for j in JS}
            if len(set(sg.values()))>1: P(f"- pair {x}/{y} differs: "+", ".join(f"{j}: {x}{'>' if s else '<'}{y}" for j,s in sg.items()))
    P("\n#### Rank correlation between judges (4 condition rates)\n\n| pair | Spearman rho | Kendall tau |\n|---|---|---|")
    for a,b in itertools.combinations(JS[:3],2):
        va=[rt[(a,ver)][c] for c in 'ABCD']; vb=[rt[(b,ver)][c] for c in 'ABCD']
        P(f"| {a} vs {b} | {spearmanr(va,vb)[0]:.3f} | {kendalltau(va,vb)[0]:.3f} |")
    P("")
    P(f"#### Pairwise differences (rate X minus rate Y), version ({ver})\n\nSign per judge; panel: paired on questions both conditions have usable"+(" non-ABSTAIN" if ver=='a' else "")+f" panel labels (n shown), diff with paired bootstrap 95% CI (10000 resamples, seed 0) and exact McNemar p.\n")
    P("| X vs Y | llama | deepseek | qwen3-235b | panel diff (pp) | 95% CI (pp) | n pairs | b (X fail,Y ok) | c (X ok,Y fail) | McNemar p | judges disagree in direction |\n|---|---|---|---|---|---|---|---|---|---|---|")
    rng=np.random.default_rng(0)
    for x,y in itertools.combinations('ABCD',2):
        sg={j:rt[(j,ver)][x]-rt[(j,ver)][y] for j in JS}
        sgn=lambda v:'+' if v>0 else ('-' if v<0 else '0')
        if ver=='a': qs=[q for q in Q if D[x][q]['panel']!='ABS' and D[y][q]['panel']!='ABS']
        else: qs=Q
        f=lambda c,q: 1 if D[c][q]['panel']=='HAL' or (ver=='b' and D[c][q]['panel']=='ABS') else 0
        u=np.array([f(x,q) for q in qs]); w=np.array([f(y,q) for q in qs]); d=u-w
        idx=rng.integers(0,len(qs),(10000,len(qs))); bs=d[idx].mean(1)
        lo,hi=np.percentile(bs,[2.5,97.5]); b=int(((u==1)&(w==0)).sum()); c_=int(((u==0)&(w==1)).sum())
        p=binomtest(b,b+c_,0.5).pvalue if b+c_ else 1.0
        s3=[sgn(sg[j]) for j in JS[:3]]; dis=len({s for s in s3 if s!='0'})>1 or ('0' in s3 and len(set(s3))>1)
        P(f"| {x} vs {y} | {sgn(sg['llama'])} ({100*sg['llama']:+.2f}) | {sgn(sg['deepseek'])} ({100*sg['deepseek']:+.2f}) | {sgn(sg['qwen3-235b'])} ({100*sg['qwen3-235b']:+.2f}) | {100*d.mean():+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {len(qs)} | {b} | {c_} | {p:.4g} | {'YES' if dis else 'no'} |")
    P("\n(+ means X has the higher rate than Y; 0 = exactly equal.)\n")

# ============================ post-hoc: common-question set ============================
P("\n---\n\n# post-hoc: common-question set\n")
P("Common set = questions whose PANEL label is not ABSTAIN in ANY of A-D. One set, one denominator, for all three judges and the panel. Post-hoc: not in the frozen analysis plan.\n")
P("Definitions: hallucinated = HALUSINASI_PENUH or HALUSINASI_SEBAGIAN, not hallucinated = FAKTUAL, ABSTAIN per judge_binary.py:46-53 (`category_of`); panel ABSTAIN = `panel_label == \"ABSTAIN\"` in the panel file, same mapping as judge_panel_aggregate.py:116-123 (`panel_category`). Judges' own labels are used for the judge columns, even where a judge says ABSTAIN on a common-set question (counted below).\n")
absq={c:{q for q in Q if D[c][q]['panel']=='ABS'} for c in R}
dropped=set().union(*absq.values()); common=[q for q in Q if q not in dropped]
P("## 1. Size and overlap\n")
P(f"- All questions: {len(Q)}; common set: **{len(common)}**; dropped: **{len(dropped)}** (union of panel-ABSTAIN questions).")
P("\n| condition | panel-ABSTAIN (questions) |\n|---|---|")
for c in R: P(f"| {c} | {len(absq[c])} |")
P("\n| ABSTAIN overlap | count |\n|---|---|")
for k in (2,3):
    for cs in itertools.combinations('BCD',k):
        P(f"| {' & '.join(cs)} | {len(set.intersection(*[absq[c] for c in cs]))} |")
P(f"| B or C or D (union = dropped) | {len(dropped)} |")
P(f"| only one of B, C, D | {sum(1 for q in dropped if sum(q in absq[c] for c in 'BCD')==1)} |")
P(f"| exactly two | {sum(1 for q in dropped if sum(q in absq[c] for c in 'BCD')==2)} |")
P(f"| all three | {sum(1 for q in dropped if sum(q in absq[c] for c in 'BCD')==3)} |")
n=len(common)
jab={j:sum(D[c][q][j]=='ABS' for c in R for q in common) for j in JS[:3]}
P(f"\nJudge-level ABSTAIN labels inside the common set (judge says ABSTAIN where the panel does not), summed over A-D: "+", ".join(f"{j} {v}" for j,v in jab.items())+". They stay in the denominator as not-hallucinated-or-hallucinated only if the judge gave one of those labels; the denominator below is therefore the common-set size minus that judge's own ABSTAINs in that cell, and is printed per cell.\n")
cr={}
P("## 2. Rates on the common set\n")
for j in JS:
    P(f"### {j}\n\n| cond | hallucinated | judge ABSTAIN in set | denominator | rate |\n|---|---|---|---|---|")
    cr[j]={}
    for c in R:
        x=[D[c][q][j] for q in common]; h=x.count('HAL'); a=x.count('ABS'); den=h+x.count('NOT')
        cr[j][c]=h/den; P(f"| {c} | {h} | {a} | {den} | {100*h/den:.2f}% |")
    P("")
P("## 3. Rank order (lowest to highest), common set\n\n| judge | order | near-ties (<1 pp) |\n|---|---|---|")
ords={}
for j in JS:
    r=cr[j]; o=sorted(r,key=lambda c:r[c]); ords[j]=o
    nt=[f"{x}-{y} ({100*abs(r[x]-r[y]):.2f} pp)" for x,y in itertools.combinations('ABCD',2) if abs(r[x]-r[y])<0.01]
    P(f"| {j} | {' < '.join(o)} | {', '.join(nt) or 'none'} |")
same=len({tuple(v) for v in ords.values()})==1
P(f"\nOrder identical across the three judges and the panel: **{'YES' if same else 'NO'}**")
if not same:
    for x,y in itertools.combinations('ABCD',2):
        sg={j:cr[j][x]>cr[j][y] for j in JS}
        if len(set(sg.values()))>1: P(f"- pair {x}/{y} differs: "+", ".join(f"{j}: {x}{'>' if s else '<'}{y}" for j,s in sg.items()))
P("\n| pair | Spearman rho | Kendall tau |\n|---|---|---|")
for a,b in itertools.combinations(JS[:3],2):
    va=[cr[a][c] for c in 'ABCD']; vb=[cr[b][c] for c in 'ABCD']
    P(f"| {a} vs {b} | {spearmanr(va,vb)[0]:.3f} | {kendalltau(va,vb)[0]:.3f} |")
P("\n## 4. Paired comparisons on the common set\n")
P("Pairs: questions in the common set where both judge labels are hallucinated/not hallucinated (for the panel and normally for judges this is all common questions; n shown). Difference = rate X minus rate Y; paired bootstrap 95% CI (10,000 resamples, seed 0 per row); exact McNemar (b = only X hallucinated, c = only Y hallucinated); Holm over the 6 pairs within each judge/panel.\n")
def holm(ps):
    o=sorted(range(len(ps)),key=lambda i:ps[i]); adj=[0]*len(ps); m=len(ps); run=0
    for k,i in enumerate(o):
        run=max(run,min(1,(m-k)*ps[i])); adj[i]=run
    return adj
pairs=list(itertools.combinations('ABCD',2)); res={}
for j in JS:
    rows=[]
    for x,y in pairs:
        qs=[q for q in common if D[x][q][j]!='ABS' and D[y][q][j]!='ABS']
        u=np.array([D[x][q][j]=='HAL' for q in qs],int); w=np.array([D[y][q][j]=='HAL' for q in qs],int); d=u-w
        rng=np.random.default_rng(0); bs=d[rng.integers(0,len(qs),(10000,len(qs)))].mean(1)
        lo,hi=np.percentile(bs,[2.5,97.5]); b=int(((u==1)&(w==0)).sum()); c_=int(((u==0)&(w==1)).sum())
        rows.append((x,y,len(qs),d.mean(),lo,hi,b,c_,binomtest(b,b+c_,.5).pvalue if b+c_ else 1.0))
    ad=holm([r[-1] for r in rows]); res[j]=rows
    P(f"### {j}\n\n| X vs Y | n | diff (pp) | 95% CI (pp) | b | c | McNemar p | Holm p |\n|---|---|---|---|---|---|---|---|")
    for r,a in zip(rows,ad): P(f"| {r[0]} vs {r[1]} | {r[2]} | {100*r[3]:+.2f} | [{100*r[4]:+.2f}, {100*r[5]:+.2f}] | {r[6]} | {r[7]} | {r[8]:.4g} | {a:.4g} |")
    P("")
P("### Direction across judges (sign of rate difference X minus Y)\n\n| X vs Y | llama | deepseek | qwen3-235b | panel | judges disagree in direction |\n|---|---|---|---|---|---|")
sg=lambda v:'+' if v>1e-12 else ('-' if v<-1e-12 else '0')
for i,(x,y) in enumerate(pairs):
    s=[sg(res[j][i][3]) for j in JS]
    P(f"| {x} vs {y} | {s[0]} | {s[1]} | {s[2]} | {s[3]} | {'YES' if len(set(s[:3]))>1 else 'no'} |")
P("\n## 5. Selection check: dropped vs common questions\n")
P("Condition A rate (A has no panel ABSTAIN, so all dropped questions have an A answer), hallucinated / denominator:\n")
P("| judge | A on common set | A on dropped set |\n|---|---|---|")
for j in JS:
    def ra(qs):
        x=[D['A'][q][j] for q in qs]; h=x.count('HAL'); dn=h+x.count('NOT'); return f"{h}/{dn} = {100*h/dn:.2f}%"
    P(f"| {j} | {ra(common)} | {ra(sorted(dropped))} |")
P("\nPer-condition panel ABSTAIN rate, on the common set (0 by construction) and on the dropped set:\n")
P("| cond | common set | dropped set |\n|---|---|---|")
for c in R:
    P(f"| {c} | 0/{len(common)} = 0.00% | {len(absq[c])}/{len(dropped)} = {100*len(absq[c])/len(dropped):.2f}% |")
P("\n## 6. Relation to the frozen analysis plan\n")
P("- docs/DECISION_C_SCORING.md:810-823 (Amendment 14 (e)) pre-registers only C vs A, C vs B, C vs D (official_report.py:61 `COMPARISONS`), labelled consistent/inconclusive by sign agreement of panel and DeepSeek-only differences (DECISION_C_SCORING.md:815-818). \"The 2 x 2 table (only C hallucinated, only the other, both, neither) is the primary paired report\" (DECISION_C_SCORING.md:822).")
P("- official_report.py:473-507 `paired_counts` builds that 2x2 table per question; primary excludes pairs where either answer is ABSTAIN, secondary counts ABSTAIN as failed. official_report.py:510-531 `paired_rate_difference` gives the rate difference on the same non-ABSTAIN pairs. Neither contains a bootstrap CI, McNemar test or Holm adjustment (no match in official_report.py or DECISION_C_SCORING.md).")
P("- This analysis: duplicates the pairwise-sign idea, **differs** in the set (questions non-ABSTAIN in all of A-D, instead of pairwise non-ABSTAIN), and **extends** with the three non-C pairs (A-B, A-D, B-D), per-judge results, CIs, McNemar and Holm. It is post-hoc and is not a replacement for the pre-registered report.")
open('results/judge_order_check.md','w').write("\n".join(out)+"\n")
print("\n".join(out))
