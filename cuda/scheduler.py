"""Resumable gain-weighted n=1..1000 CUDA search and official-gated merging."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
import heapq
import json
import math
import os
from pathlib import Path
import subprocess
import time
from squares.core import certify, grid, read_submission, score, write_submission
from squares.optimizer import tilted_five
from demo import certified_candidate

FOCUS={5,10,11,17,18,19,26,27,37,50}

def atomic_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2)+"\n")
    os.replace(tmp,path)

def strip_construction(k):
    """Original extension of the credited tilted-five, not record coordinates.

    Keep the 2.707... base in the lower-left. Add full unit strips along
    top and right. This gives k*k+1 squares in side k+1/sqrt(2).
    Decimal base coordinates and 60-digit arithmetic avoid float contacts.
    """
    import mpmath as mp
    if not 2<=k<=31: raise ValueError("k must be 2..31")
    with mp.workdps(65):
        base=tilted_five()
        L=mp.mpf(certify(5,base))
        t=k-2
        q=[dict(r) for r in base]
        # Positive clearance, also measured in the final bounding box.
        pitch=mp.mpf("1.000000001")
        def add(x,y):
            q.append(dict(square_id=str(len(q)),x=mp.nstr(x,60),
                          y=mp.nstr(y,60),angle="0"))
        for row in range(t):
            for col in range(2):
                add((col+mp.mpf(".5"))*pitch,L+(row+mp.mpf(".5"))*pitch)
        for col in range(t):
            for row in range(k):
                add(L+(col+mp.mpf(".5"))*pitch,(row+mp.mpf(".5"))*pitch)
        return q

def goebel(a):
    """Analytic Göbel staircase/45-degree strip, credited 1979 construction.

    Derived from the formula at Ellsworth's successor page, not imported SVGs.
    Coordinates at 65-digit precision; measured tiny clearance is included in L.
    """
    import mpmath as mp
    if not 1<=a<=30: raise ValueError("a must be 1..30")
    with mp.workdps(65):
        h=1/mp.sqrt(2);L=a+1+h;dilation=mp.mpf("1.0000000000000000000000000000000000000001")
        rows=[]
        def add(x,y,angle=0):
            rows.append(dict(square_id=str(len(rows)),x=mp.nstr(x*dilation,60),
                             y=mp.nstr(y*dilation,60),angle=mp.nstr(angle,60)))
        for i in range(a):
            for j in range(a-i):
                x=i+mp.mpf(".5");y=j+mp.mpf(".5")
                add(x,y);add(L-x,L-y)
        add(mp.mpf(".5"),L-mp.mpf(".5"))
        add(L-mp.mpf(".5"),mp.mpf(".5"))
        for j in range(1+int(mp.floor((a-1)*mp.sqrt(2)))):
            c=1+h/2+j*h;add(c,L-c,mp.pi/4)
        return rows

def priority(n, side, attempts):
    upper=max(0.,side/math.sqrt(n)-1)
    return upper*(4 if n in FOCUS else 1)/((1+attempts)*math.sqrt(n))

def run_gpu(executable,n,rows,directory,epochs,steps,chains,seed):
    side=float(certify(n,rows))
    seedfile=directory/f"seed{n}.txt"; output=directory/f"raw{n}.json"
    seedfile.write_text(f"{n} {side*1.00002:.12g}\n"+"".join(
        " ".join(format(float(r[k])*(1.00002 if k!="angle" else 1),".12g")
                 for k in ("x","y","angle"))+"\n" for r in rows))
    subprocess.run([executable,str(seedfile),str(epochs),str(steps),
                    str(chains),str(seed),str(output)],check=True,timeout=180)
    raw=json.loads(output.read_text())
    return certified_candidate(raw),raw

def _evaluate_item(item):
    n,rows=item
    return n,certify(n,rows)

def provenance_counts(state,max_n):
    """Best-source counts partition n; execution counts are independent."""
    counts=dict(analytic_best=0,gpu_best=0,baseline_best=0,other_best=0,
                actual_gpu_cases=0,actual_gpu_runs=0)
    for n in range(1,max_n+1):
        item=state[str(n)]
        attempts=item.get("attempts",0)
        counts["actual_gpu_cases"]+=attempts>0
        counts["actual_gpu_runs"]+=attempts
        src=item.get("source","")
        key=("baseline_best" if src=="exact grid" else
             "analytic_best" if ("Goebel" in src or "tilted-five" in src) else
             "gpu_best" if "CUDA" in src else "other_best")
        counts[key]+=1
    return counts

def final_report(directory,max_n):
    puzzles={}
    for n in range(1,max_n+1):
        p=directory/"best"/f"{n}.json"
        puzzles[n]=json.loads(p.read_text())["rows"] if p.exists() else grid(n)
    # The unchanged score function re-certifies every puzzle and computes
    # the official aggregate, including rejection/missing counters.
    measured=score(puzzles,max_n)
    if measured["accepted"]!=max_n or measured["missing"] or measured["rejected"]:
        raise ValueError("full merged submission failed official certification")
    write_submission(directory/"submission.csv",puzzles)
    baseline=sum((math.isqrt(n-1)+1)/math.sqrt(n)-1 for n in range(1,max_n+1))
    state=json.loads((directory/"state.json").read_text())
    report=dict(baseline_score=baseline,score=measured["score"],
                gain=baseline-measured["score"],accepted=measured["accepted"],
                max_n=max_n,nonperfect_processed=sum(v.get("attempts",0)>0 for v in state.values()),
                nonperfect_total=sum(math.isqrt(n)**2!=n for n in range(1,max_n+1)),
                improved=sum(float(state[str(n)]["L"])<math.isqrt(n-1)+1 for n in range(1,max_n+1)),
                all_nonperfect_processed=all(v.get("attempts",0)>0 for n,v in state.items() if math.isqrt(int(n))**2!=int(n)))
    report.update(provenance_counts(state,max_n))
    report.update(csv_data_rows=sum(len(rows) for rows in puzzles.values()),
                  csv_columns=["id","n","square_id","x","y","angle"],
                  side_lengths={str(r["n"]):r["L"] for r in measured["results"]},
                  score_definition="sum(L/sqrt(n)-1), lower is better")
    atomic_json(directory/"report.json",report)
    print(json.dumps(report),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--directory",type=Path,default=Path("outputs/sweep"))
    p.add_argument("--executable",default="./cuda/packing")
    p.add_argument("--max-n",type=int,default=1000)
    p.add_argument("--jobs",type=int,default=20,help="finite GPU tasks per invocation")
    p.add_argument("--epochs",type=int,default=0,help="0 = auto-tune per puzzle tier")
    p.add_argument("--steps",type=int,default=0,help="0 = auto-tune per puzzle tier")
    p.add_argument("--chains",type=int,default=0,help="0 = auto-scale to full GPU SM wave")
    p.add_argument("--imports",type=Path,nargs="*",default=[])
    p.add_argument("--initialize",action="store_true")
    p.add_argument("--finalize",action="store_true")
    p.add_argument("--coverage",action="store_true",help="one first visit to every nonsquare before repeats")
    p.add_argument("--solver", choices=("packing","algos"), default="packing")
    p.add_argument("--mode", choices=("1","2","3"), default="1")
    a=p.parse_args()
    os.environ["ALGOS_MODE"]=a.mode
    if a.solver=="algos": a.executable="./cuda/algos"
    if not 1<=a.max_n<=1000 or not 0<=a.jobs<=10000: p.error("out of range")
    d=a.directory;d.mkdir(parents=True,exist_ok=True);(d/"best").mkdir(exist_ok=True)
    statepath=d/"state.json"
    state=json.loads(statepath.read_text()) if statepath.exists() else {}
    def consider(n,rows,source):
        L=certify(n,rows)
        if Decimal(L)<Decimal(state[str(n)]["L"]):
            atomic_json(d/"best"/f"{n}.json",dict(n=n,L=L,rows=rows,source=source))
            state[str(n)]["L"]=L;state[str(n)]["source"]=source
            atomic_json(statepath,state)
    for n in range(1,a.max_n+1):
        if str(n) not in state:
            state[str(n)]=dict(L=str(math.isqrt(n-1)+1),attempts=0,source="exact grid")
    atomic_json(statepath,state)
    if a.initialize:
        for parameter in range(1,31):
            template=goebel(parameter);m=len(template)
            if m>a.max_n: break
            # Every hereditary subset above the previous perfect square helps.
            lower=math.isqrt(m-1)**2+1
            for n in range(lower,m+1):
                consider(n,template[:n],"analytic Goebel 1979 construction, no imported coordinates")
        for k in range(2,math.isqrt(a.max_n-1)+1):
            m=k*k+1
            rows=strip_construction(k)
            # Hereditary deletion is safe, but every selected subset is still gated.
            for n in range(k*k+1,min(m,a.max_n)+1):
                consider(n,rows[:n],"derived tilted-five unit-strip extension")
        for path in a.imports:
            for n,rows in read_submission(path).items():
                if n<=a.max_n and math.isqrt(n)**2!=n:
                    consider(n,rows,f"local certified candidate {path}")
    heap=[]
    for n in range(1,a.max_n+1):
        if math.isqrt(n)**2!=n:
            s=state[str(n)]
            # First-visit coverage boost does not remove gain weighting.
            if not a.coverage or s["attempts"]==0:
                heapq.heappush(heap,(-priority(n,float(s["L"]),s["attempts"]),n))
    for job in range(a.jobs):
        if not heap: break
        _,n=heapq.heappop(heap);s=state[str(n)]
        best=d/"best"/f"{n}.json"
        rows=json.loads(best.read_text())["rows"] if best.exists() else grid(n)
        start=time.monotonic()
        try:
            (packing,L,dilation),raw=run_gpu(a.executable,n,rows,d,a.epochs,a.steps,
                                            a.chains,42+s["attempts"])
            consider(n,packing,"CUDA search official interval certified")
            s.update(last_seconds=time.monotonic()-start,last_raw=raw,
                     last_dilation=dilation,last_status="certified")
        except (ValueError,subprocess.SubprocessError) as e:
            s.update(last_status="rejected",last_error=str(e),last_seconds=time.monotonic()-start)
        s["attempts"]+=1
        atomic_json(statepath,state)
        print(json.dumps(dict(n=n,L=s["L"],attempts=s["attempts"],status=s["last_status"])),flush=True)
        if not a.coverage:
            heapq.heappush(heap,(-priority(n,float(s["L"]),s["attempts"]),n))
    if a.finalize: final_report(d,a.max_n)

if __name__=="__main__": main()
