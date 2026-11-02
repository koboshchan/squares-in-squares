"""Finite seeded basin hopping of locally generated CUDA candidates."""
import argparse,json,time
from pathlib import Path
import numpy as np
from polish import polish
from squares.core import read_submission,certify,write_submission

def main():
    p=argparse.ArgumentParser();p.add_argument("input",type=Path)
    p.add_argument("--output",type=Path,default=Path("outputs/refine"))
    p.add_argument("--restarts",type=int,default=40)
    p.add_argument("--seed",type=int,default=47)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(a.seed);puzzles=read_submission(a.input)
    report=[]
    for n,base in puzzles.items():
        if n not in (11,17,18,19,26,27,37,50):continue
        rows=base;side=float(certify(n,rows));archive=[rows];start=time.monotonic()
        for k in range(a.restarts):
            q=np.array([[float(r[f]) for f in ("x","y","angle")] for r in archive[k%len(archive)]])
            scale=(.005,.02,.05,.1,.2,.4)[k%6]
            q+=rng.normal(0,scale,q.shape)
            try:
                cand,r=polish(q,maxiter=600)
                if r["L"]<side:
                    rows=cand;side=r["L"];archive.insert(0,cand);archive=archive[:4]
                    write_submission(a.output/f"best{n}.csv",{n:rows})
                    print(json.dumps(dict(n=n,restart=k,L=side,seconds=time.monotonic()-start)),flush=True)
            except ValueError:pass
        puzzles[n]=rows;report.append(dict(n=n,L=side,restarts=a.restarts,seconds=time.monotonic()-start))
    write_submission(a.output/"certified.csv",puzzles)
    (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
if __name__=="__main__":main()
