"""One resumable complete GPU pass, with literal and numeric-transport gates."""
from __future__ import annotations
import argparse, hashlib, json, math, os, subprocess, time
from decimal import Decimal
from pathlib import Path
from squares.core import certify, grid, kaggle_roundtrip, read_submission, score, write_submission
from scheduler import atomic_json

def cpp(exe, path, maximum, details):
    summaries=[]
    for rt in (False,True):
        cmd=[str(exe),str(path),"--max-n",str(maximum),"--threads","4"]
        if rt: cmd+=["--roundtrip"]
        result=subprocess.run(cmd,capture_output=True,text=True,check=True)
        obj=json.loads(result.stdout);summaries.append(obj)
    atomic_json(details,summaries)
    return all(x["rejected"]==0 for x in summaries)

def main():
    p=argparse.ArgumentParser();p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--incumbent",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args();d=a.directory;d.mkdir(parents=True,exist_ok=True)
    exe=Path("cuda/packing").resolve();cert=Path("cuda/certifier").resolve()
    start=time.time();statefile=d/"state.json"
    if statefile.exists(): state=json.loads(statefile.read_text())
    else:
        seed=read_submission(a.incumbent)
        assert len(seed)==1000
        assert cpp(cert,a.incumbent,1000,d/"incumbent-cpp.json")
        measured=score(kaggle_roundtrip(seed))
        assert measured["accepted"]==1000
        state=dict(start=time.time(),incumbent_score=measured["score"],cases={},gpu_improved=[])
        (d/"best").mkdir(exist_ok=True)
        for n,rows in seed.items():
            atomic_json(d/"best"/f"{n}.json",dict(rows=rows,L=certify(n,rows)))
        atomic_json(statefile,state)
    targets=sorted((n for n in range(1,1001) if math.isqrt(n)**2!=n),
                   key=lambda n: -float(Decimal(json.loads((d/"best"/f"{n}.json").read_text())["L"]))/math.sqrt(n))
    for n in targets:
        if str(n) in state["cases"]:continue
        state["current_n"]=n;atomic_json(statefile,state)
        bestpath=d/"best"/f"{n}.json";best=json.loads(bestpath.read_text());rows=best["rows"]
        seedfile=d/"seed.txt";rawfile=d/f"raw{n}.json"
        seedfile.write_text(f"{n} {float(best['L'])*1.00002:.17g}\n"+"".join(
            " ".join(format(float(r[k])*(1.00002 if k!="angle" else 1),".17g")
                     for k in ("x","y","angle"))+"\n" for r in rows))
        t=time.time();case=dict(n=n,status="rejected",gpu_improved=False)
        try:
            cmd=[str(exe),str(seedfile),"0","0","0",str(a.seed),str(rawfile)]
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=900,check=True)
            (d/f"gpu{n}.log").write_text(result.stdout+result.stderr)
            raw=json.loads(rawfile.read_text())
            assert raw["n"]==n and len(raw["q"])==n
            for dilation in (1.000002,1.00001,1.0001):
                candidate=[dict(square_id=str(i),x=format(x*dilation,".17g"),
                                y=format(y*dilation,".17g"),angle=format(angle,".17g"))
                           for i,(x,y,angle) in enumerate(raw["q"])]
                if not all(math.isfinite(float(r[k])) for r in candidate for k in ("x","y","angle")):continue
                csv=d/"candidate.csv";write_submission(csv,{n:candidate},validate=False)
                if not cpp(cert,csv,n,d/f"cpp{n}.json"):continue
                # Both unchanged official paths gate replacement, never the C++ verdict alone.
                L=certify(n,candidate);rtL=certify(n,kaggle_roundtrip({n:candidate})[n])
                case.update(status="certified",L=L,roundtrip_L=rtL,dilation=dilation)
                if Decimal(L)<Decimal(best["L"]) and Decimal(rtL)<Decimal(best.get("roundtrip_L",best["L"])):
                    atomic_json(bestpath,dict(rows=candidate,L=L,roundtrip_L=rtL,source="GPU",previous_L=best["L"]))
                    state["gpu_improved"].append(n);case["gpu_improved"]=True
                break
        except (subprocess.SubprocessError,ValueError,AssertionError) as e:
            case["error"]=str(e)
        case["seconds"]=time.time()-t;state["cases"][str(n)]=case
        atomic_json(statefile,state);print(json.dumps(case),flush=True)
    puzzles={n:json.loads((d/"best"/f"{n}.json").read_text())["rows"] for n in range(1,1001)}
    final=d/"submission-kaggle-safe.csv";write_submission(final,puzzles)
    assert cpp(cert,final,1000,d/"final-cpp.json")
    literal=score(puzzles);transport=score(kaggle_roundtrip(puzzles))
    for result in (literal,transport):
        assert result["accepted"]==1000 and result["rejected"]==0 and result["missing"]==0
    assert len(state["cases"])==969
    report=dict(incumbent_score=state["incumbent_score"],score=literal["score"],
                roundtrip_score=transport["score"],gpu_improved=len(state["gpu_improved"]),
                improved_n=state["gpu_improved"],record_imports_this_run=0,
                cases=len(state["cases"]),wall_seconds=time.time()-state["start"],
                accepted=1000,csv=str(final),sha256=hashlib.sha256(final.read_bytes()).hexdigest())
    atomic_json(d/"report.json",report);atomic_json(d/"official-literal.json",literal)
    atomic_json(d/"official-roundtrip.json",transport);print(json.dumps(report),flush=True)
if __name__=="__main__":main()
