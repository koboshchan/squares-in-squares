"""Single-owner endless passes. Existing seed-42 process is observed, never killed.

Run from the repo root on Linux/WSL. All promotion happens after four gates.
Ctrl-C stops this supervisor, not an already-running seed-42 process.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from tqdm import tqdm
from cpp_loop import atomic_json, gates, write_csv

def load(path):
    return json.loads(Path(path).read_text())

def live_score(directory):
    lengths=directory/"cpp-lengths";lengths.mkdir(exist_ok=True)
    changed=[];values={};stamps={}
    for p in sorted((directory/"best").glob("*.json")):
        best=load(p);out=lengths/p.name;n=int(p.stem)
        if best.get("certifier")=="cpp-both":
            values[n]=best["L"]
        else:
            cached=load(out) if out.exists() else {}
            stamp=p.stat().st_mtime_ns
            if cached.get("source_mtime")!=stamp:
                changed.append((n,best["rows"]));stamps[n]=stamp
            else:values[n]=cached["L"]
    if changed:
        snap=directory/"live-candidate.csv";write_csv(snap,changed)
        literal,rt=gates(snap,1000)
        for n,_ in changed:
            cached=dict(L=literal["results"][n-1]["L"],
                        roundtrip_L=rt["results"][n-1]["L"],source_mtime=stamps[n])
            atomic_json(lengths/f"{n}.json",cached);values[n]=cached["L"]
    return math.fsum(max(0.,float(L)/math.sqrt(n)-1) for n,L in values.items())

def existing_runner(directory):
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            args=(p/"cmdline").read_bytes().split(b"\0")
            cwd=(p/"cwd").resolve()
            if any(x.endswith((b"cuda/full_run.py",b"cuda/cpp_pass.py")) for x in args):
                i=args.index(b"--directory")
                if (cwd/os.fsdecode(args[i+1])).resolve()==directory.resolve():
                    return int(p.name)
        except (OSError,ValueError,IndexError):
            continue
    return None

def verify_promote(source, directory, pass_number, seed):
    """Do not alter the previous known-good CSV until every gate succeeds."""
    staging=directory/"best-so-far.pending.csv"
    shutil.copyfile(source,staging)
    literal,transport=gates(staging)
    for result in (literal,transport):
        assert result["accepted"]==1000 and result["rejected"]==result["missing"]==0
    # Refresh small per-n C++ length metadata for live scoring, no Python geometry.
    for a,b in zip(literal["results"],transport["results"]):
        target=directory/"cpp-lengths"/f'{a["n"]}.json'
        target.parent.mkdir(exist_ok=True)
        atomic_json(target,dict(L=a["L"],roundtrip_L=b["L"]))
    old=directory/"best-so-far.json"
    if old.exists():
        previous=load(old)
        assert literal["score"]<=previous["score"]+1e-12
        assert transport["score"]<=previous["roundtrip_score"]+1e-12
    report=dict(pass_number=pass_number,seed=seed,score=literal["score"],
                roundtrip_score=transport["score"],accepted=1000,
                sha256=hashlib.sha256(staging.read_bytes()).hexdigest(),time=time.time())
    # CSV atomic replacement is the commit point; on crash reverify and republish.
    os.replace(staging,directory/"best-so-far.csv")
    atomic_json(old,report)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--directory",type=Path,default=Path("outputs/full-4070s"))
    p.add_argument("--incumbent",type=Path,default=Path("inputs/incumbent-safe.csv"))
    a=p.parse_args();d=a.directory.resolve();d.mkdir(parents=True,exist_ok=True)
    lock=(d/"endless.lock").open("a+")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit("Another endless supervisor owns this directory")
    atomic_json(d/"endless-owner.json",dict(pid=os.getpid(),started=time.time()))
    meta=d/"endless.json"
    if not meta.exists():
        atomic_json(meta,dict(pass_number=1,seed=42,phase="running"))
    m=load(meta)
    if not (d/"cpp-lengths").exists():
        print("Verifying initial known-good CSV, seed 42 remains untouched",flush=True)
        verify_promote(d/"best-so-far.csv" if (d/"best-so-far.csv").exists() else a.incumbent,d,0,42)
    while True:
        m=load(meta)
        if m["phase"]=="advance":
            # Idempotent transition: archive pass state before resetting cases.
            archive=d/"passes"/f'{m["pass_number"]:06d}'
            archive.mkdir(parents=True,exist_ok=True)
            if not (archive/"state.json").exists():
                shutil.copyfile(d/"state.json",archive/"state.json")
            state=load(archive/"state.json")
            state.update(cases={},gpu_improved=[],start=time.time(),current_n=None)
            atomic_json(d/"state.json",state)
            m=dict(pass_number=m["pass_number"]+1,seed=m["seed"]+1,phase="running")
            atomic_json(meta,m)
        child=None
        if existing_runner(d) is None:
            log=(d/"worker.log").open("a")
            child=subprocess.Popen([sys.executable,"-u","cuda/cpp_pass.py",
                "--directory",str(d),"--incumbent",str(a.incumbent),"--seed",str(m["seed"])],
                stdout=log,stderr=subprocess.STDOUT)
            log.close()
        bar=tqdm(total=969,desc=f'Pass {m["pass_number"]} seed {m["seed"]}',dynamic_ncols=True)
        try:
            while True:
                state=load(d/"state.json")
                bar.n=len(state["cases"])
                current=state.get("current_n")
                if m["pass_number"]==1 and current is None and existing_runner(d):
                    # Old runner has no current_n field; its unique raw filename is in argv.
                    for proc in Path("/proc").iterdir():
                        try:
                            args=(proc/"cmdline").read_bytes().split(b"\0")
                            if args and args[0].endswith(b"/cuda/packing") and (proc/"cwd").resolve()==Path.cwd():
                                current=Path(os.fsdecode(args[-2])).stem.removeprefix("raw")
                        except OSError: pass
                value=live_score(d)
                bar.set_postfix(n=current,wins=len(state["gpu_improved"]),score=f"{value:.10f}")
                bar.refresh()
                atomic_json(d/"live.json",dict(pass_number=m["pass_number"],seed=m["seed"],
                    done=bar.n,current_n=current,wins=len(state["gpu_improved"]),score=value,time=time.time()))
                alive=child.poll() is None if child else existing_runner(d) is not None
                if not alive: break
                time.sleep(3)
        finally: bar.close()
        if child and child.returncode:
            raise RuntimeError(f'Worker exited {child.returncode}; inspect worker.log, rerun to resume')
        assert len(load(d/"state.json")["cases"])==969,"Pass incomplete; rerun to resume"
        report=verify_promote(d/"submission-kaggle-safe.csv",d,m["pass_number"],m["seed"])
        archive=d/"passes"/f'{m["pass_number"]:06d}'
        archive.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(d/"best-so-far.csv",archive/"best-so-far.csv")
        atomic_json(archive/"score.json",report)
        # Rebuild log from idempotent per-pass records, avoiding duplicate crash entries.
        reports=[load(x) for x in sorted((d/"passes").glob("*/score.json"))]
        atomic_json(d/"score-per-pass.json",reports)
        print(json.dumps(report),flush=True)
        m["phase"]="advance";atomic_json(meta,m)

if __name__=="__main__": main()
