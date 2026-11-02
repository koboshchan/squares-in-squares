"""Finite launch/ladder/move experiments, certified quality before throughput."""
import argparse,json
from pathlib import Path
from demo import certified_candidate
from polish import polish
from squares.core import certify,write_submission

def prepare(directory):
    """Reproduce the finite 12-launch matrix beside packing{,128,256}."""
    directory.mkdir(parents=True,exist_ok=True)
    script=["#!/bin/sh","set -eu",'cd "$(dirname "$0")"',"mkdir -p tuning"]
    for threads in (64,128,256):
        executable="./packing" if threads==64 else f"./packing{threads}"
        for chains in (256,1024):
            for name,config in [
                ("cold","PACK_TMIN=1e-9 PACK_TMAX=.001 PACK_MOVE=.003 PACK_ROTATION=.35 PACK_SHRINK=.0005"),
                ("warm","PACK_TMIN=1e-7 PACK_TMAX=.03 PACK_MOVE=.02 PACK_ROTATION=.8 PACK_SHRINK=.001")]:
                script.append(f"{config} {executable} seed11.txt 4000 64 {chains} 49 tuning/tune-t{threads}-c{chains}-{name}.json")
    script.append("tar -czf tuning.tar.gz tuning")
    (directory/"tune-run.sh").write_text("\n".join(script)+"\n")

def summarize(directory):
    reports=[];bests={}
    for path in sorted(directory.glob("tune*.json")):
        raw=json.loads(path.read_text())
        r=dict(source=path.name,n=raw["n"],seconds=raw["seconds"],
               chain_steps_per_second=raw["chain_steps"]/raw["seconds"],
               proposals_per_second=raw["proposals"]/raw["seconds"],
               config=path.stem)
        try:
            rows,L,dilation=certified_candidate(raw);r.update(certified_L=L,dilation=dilation)
            polished,p=polish(raw["q"],maxiter=600);r["polished_L"]=p["L"]
            r["full_precision_L"]=certify(raw["n"],polished)
            if raw["n"] not in bests or p["L"]<float(certify(raw["n"],bests[raw["n"]])):
                bests[raw["n"]]=polished
        except ValueError as e:r["rejected"]=str(e)
        reports.append(r)
    reports.sort(key=lambda r:(r.get("polished_L",float("inf")),-r["chain_steps_per_second"]))
    (directory/"report.json").write_text(json.dumps(reports,indent=2)+"\n")
    if bests:write_submission(directory/"certified.csv",bests)
    print(json.dumps(reports,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path)
    p.add_argument("--prepare",action="store_true")
    a=p.parse_args()
    if a.prepare:prepare(a.directory)
    else:summarize(a.directory)
if __name__=="__main__":main()
