"""Finite launch/ladder/move experiments, certified quality before throughput."""
import argparse,json,math,subprocess,time
from pathlib import Path
from demo import certified_candidate
from polish import polish
from squares.core import certify,write_submission

def summarize(directory,n):
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
    a=p.parse_args();summarize(a.directory,11)
if __name__=="__main__":main()
