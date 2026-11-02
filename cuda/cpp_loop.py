"""C++-only gates and bounded concurrent packing workers."""
import csv,json,math,os,subprocess,time
from pathlib import Path
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED

def atomic_json(path,obj):
    path=Path(path);tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)

def write_csv(path,puzzles):
    with Path(path).open("w",newline="") as f:
        w=csv.writer(f);w.writerow(("id","n","square_id","x","y","angle"));i=0
        for n,rows in puzzles:
            for r in rows:
                w.writerow((i,n,r["square_id"],r["x"],r["y"],r["angle"]));i+=1

def gates(path,maximum=1000,threads=2):
    results=[]
    for rt in (False,True):
        detail=Path(str(path)+(".rt.json" if rt else ".literal.json"))
        cmd=[str(Path("cuda/certifier").resolve()),str(path),"--max-n",str(maximum),
             "--threads",str(threads),"--details",str(detail)]
        if rt:cmd+=["--roundtrip"]
        subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=180)
        obj=json.loads(detail.read_text())
        if obj["rejected"]:raise ValueError("C++ rejected candidate")
        results.append(obj)
    return results

def gpu_status():
    try:
        line=subprocess.check_output(["nvidia-smi","--query-gpu=memory.free,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).splitlines()[0]
        return tuple(int(x.strip()) for x in line.split(","))
    except (OSError,ValueError,subprocess.SubprocessError):return (0,0,100)

def ram_free():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):return int(line.split()[1])//1024
    return 0

def concurrency():
    free,used,util=gpu_status()
    # Reserve 2GiB VRAM and RAM. Allow <=4 workers, each budgeted 1.5GiB VRAM,
    # 512MiB host RAM. Gate threads per worker bounded at 2.
    return max(1,min(4,(free-2048)//1536,(ram_free()-2048)//512))

def case(d,n,seed):
    begin=time.monotonic();bestpath=d/"best"/f"{n}.json";best=json.loads(bestpath.read_text())
    work=d/"concurrent"/str(n);work.mkdir(parents=True,exist_ok=True)
    seedfile=work/"seed.txt";rawfile=work/"raw.json"
    seedfile.write_text(f"{n} {float(best['L'])*1.00002:.17g}\n"+"".join(
        " ".join(format(float(r[k])*(1.00002 if k!="angle" else 1),".17g") for k in ("x","y","angle"))+"\n" for r in best["rows"]))
    result=dict(n=n,gpu_improved=best.get("seed")==seed,status="rejected")
    try:
        # Certify legacy best with C++, never trust a Python-origin L for comparisons.
        incumbent=work/"incumbent.csv";write_csv(incumbent,[(n,best["rows"])])
        old=gates(incumbent,n)
        oldL=[x["results"][n-1]["L"] for x in old]
        best.update(L=oldL[0],roundtrip_L=oldL[1],certifier="cpp-both")
        atomic_json(bestpath,best)
        with (work/"gpu.log").open("w") as log:
            subprocess.run([str(Path("cuda/packing").resolve()),str(seedfile),"0","0","0",str(seed),str(rawfile)],
                           stdout=log,stderr=subprocess.STDOUT,timeout=900,check=True)
        raw=json.loads(rawfile.read_text());assert raw["n"]==n and len(raw["q"])==n
        for dilation in (1.000002,1.00001,1.0001):
            rows=[dict(square_id=str(i),x=format(x*dilation,".17g"),y=format(y*dilation,".17g"),angle=format(a,".17g"))
                  for i,(x,y,a) in enumerate(raw["q"])]
            if not all(math.isfinite(float(r[k])) for r in rows for k in ("x","y","angle")):continue
            candidate=work/"candidate.csv";write_csv(candidate,[(n,rows)])
            try:new=gates(candidate,n)
            except ValueError:continue
            lengths=[x["results"][n-1]["L"] for x in new]
            result.update(status="certified",L=lengths[0],roundtrip_L=lengths[1])
            if all(Decimal(a)<Decimal(b) for a,b in zip(lengths,oldL)):
                atomic_json(bestpath,dict(rows=rows,L=lengths[0],roundtrip_L=lengths[1],certifier="cpp-both",source="GPU",seed=seed))
                result["gpu_improved"]=True
            break
    except (ValueError,AssertionError,subprocess.SubprocessError) as e:result["error"]=str(e)
    result["seconds"]=time.monotonic()-begin
    return result

def run_pass(d,seed):
    state=json.loads((d/"state.json").read_text())
    targets=sorted((n for n in range(1,1001) if math.isqrt(n)**2!=n and str(n) not in state["cases"]),
                   key=lambda n:-float(json.loads((d/"best"/f"{n}.json").read_text())["L"])/math.sqrt(n))
    pending=iter(targets);running={};limit=concurrency()
    tuning=d/"batch-tuning.json"
    if tuning.exists():limit=min(limit,json.loads(tuning.read_text())["concurrency"])
    atomic_json(d/"concurrency.json",dict(limit=limit))
    with ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            # Never submit an unbounded queue. Recheck VRAM/RAM before each dispatch.
            capacity=min(limit,concurrency())
            while len(running)<capacity:
                n=next(pending,None)
                if n is None:break
                running[pool.submit(case,d,n,seed)]=n
            if not running:break
            state["current_n"]=list(running.values());atomic_json(d/"state.json",state)
            done,_=wait(running,return_when=FIRST_COMPLETED)
            for future in done:
                n=running.pop(future);r=future.result();state["cases"][str(n)]=r
                if r["gpu_improved"]:state["gpu_improved"].append(n)
            atomic_json(d/"state.json",state)
    final=d/"submission-kaggle-safe.csv"
    write_csv(final,((n,json.loads((d/"best"/f"{n}.json").read_text())["rows"]) for n in range(1,1001)))
