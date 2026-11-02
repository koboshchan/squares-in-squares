"""Explicit finite coverage batch for executing GPU-only binaries remotely.

Prepare/merge run beside Python dependencies. The generated POSIX script runs
only the specified binary in its working directory, without Python or networking.
"""
import argparse,json,math,shlex,tarfile
from pathlib import Path
from scheduler import atomic_json,priority,final_report
from squares.core import grid,certify
from demo import certified_candidate

def prepare(directory,max_n,epochs,steps,chains):
    state=json.loads((directory/"state.json").read_text())
    seeds=directory/"batch";seeds.mkdir(exist_ok=True)
    ordering=sorted((n for n in range(1,max_n+1) if math.isqrt(n)**2!=n),
                    key=lambda n:-priority(n,float(state[str(n)]["L"]),state[str(n)]["attempts"]))
    jobs=[]
    for n in ordering:
        best=directory/"best"/f"{n}.json"
        rows=json.loads(best.read_text())["rows"] if best.exists() else grid(n)
        side=float(state[str(n)]["L"])*1.00002
        (seeds/f"seed{n}.txt").write_text(f"{n} {side:.12g}\n"+"".join(
            " ".join(format(float(r[k])*(1.00002 if k!="angle" else 1),".12g")
                     for k in ("x","y","angle"))+"\n" for r in rows))
        jobs.append(dict(n=n,epochs=epochs,steps=steps,chains=chains,seed=142))
    atomic_json(seeds/"manifest.json",jobs)
    script=["#!/bin/sh","set -eu",'cd "$(dirname "$0")"',"mkdir -p raw",
            "# Each launch is finite; no overlapping kernels or unbounded loop."]
    for j in jobs:
        n=j["n"]
        script.append(f'if [ ! -s raw/{n}.json ]; then ../packing seed{n}.txt {epochs} {steps} {chains} 142 raw/{n}.json; fi')
    script.append("tar -czf raw.tar.gz raw")
    (seeds/"run.sh").write_text("\n".join(script)+"\n")
    with tarfile.open(directory/"batch.tar.gz","w:gz") as tar:
        for path in sorted(seeds.iterdir()):
            if path.is_file():tar.add(path,arcname="coverage/"+path.name)
    print(json.dumps(dict(jobs=len(jobs),epochs=epochs,steps=steps,chains=chains)),flush=True)

def merge(directory,rawdir):
    statepath=directory/"state.json";state=json.loads(statepath.read_text())
    manifest=json.loads((directory/"batch"/"manifest.json").read_text())
    for job in manifest:
        n=job["n"];path=rawdir/f"{n}.json"
        if not path.exists():continue
        # Durable de-duplication permits interruption without double attempt count.
        fingerprint=__import__("hashlib").sha256(path.read_bytes()).hexdigest()
        s=state[str(n)]
        if s.get("batch_sha256")==fingerprint:continue
        raw=json.loads(path.read_text())
        try:
            rows,L,dilation=certified_candidate(raw)
            exact=certify(n,rows)
            if __import__("decimal").Decimal(exact)<__import__("decimal").Decimal(s["L"]):
                atomic_json(directory/"best"/f"{n}.json",dict(n=n,L=exact,rows=rows,
                            source="CUDA coverage batch, official interval certified"))
                s.update(L=exact,source="CUDA coverage batch")
            s.update(last_status="certified",last_L=L,last_dilation=dilation)
        except ValueError as e:
            s.update(last_status="rejected",last_error=str(e))
        s.update(attempts=s.get("attempts",0)+1,batch_sha256=fingerprint,
                 last_seconds=raw.get("seconds"),last_chain_steps=raw.get("chain_steps"))
        atomic_json(statepath,state)
    print(json.dumps(dict(processed=sum(v.get("attempts",0)>0 for v in state.values()),
                          improved=sum(float(v["L"])<math.isqrt(int(n)-1)+1 for n,v in state.items()))))

def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["prepare","merge"])
    p.add_argument("--directory",type=Path,default=Path("outputs/sweep"))
    p.add_argument("--raw",type=Path)
    p.add_argument("--max-n",type=int,default=1000)
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--steps",type=int,default=16)
    p.add_argument("--chains",type=int,default=64)
    a=p.parse_args()
    if not 1<=a.max_n<=1000 or not 1<=a.epochs<=100000 or not 1<=a.steps<=4096 or not 8<=a.chains<=4096 or a.chains%8:p.error("invalid range")
    if a.mode=="prepare":prepare(a.directory,a.max_n,a.epochs,a.steps,a.chains)
    else:
        if a.raw is None:p.error("--raw required")
        merge(a.directory,a.raw)
if __name__=="__main__":main()
