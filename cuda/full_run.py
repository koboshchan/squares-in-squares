"""One resumable complete GPU pass, with literal and numeric-transport gates.

Integrates:
1. Ask #1 (`families.py`): Fast deterministic parametric family pre-pass across all 969 non-squares.
2. Ask #2 (`budget.py`): Gap-targeted search budget (rank by expected score gain dL/sqrt(n),
   allocate epochs/steps proportional to headroom, skip low-headroom puzzles on multi-seed passes).
3. Ask #4 (`--batch-size`): Multi-n batched GPU kernel launches when batch_size > 1.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, subprocess, time
from decimal import Decimal
from pathlib import Path
from squares.core import certify, grid, kaggle_roundtrip, read_submission, score, write_submission
from scheduler import atomic_json
from budget import compute_headroom_table, rank_puzzles_by_headroom

def cpp(exe, path, maximum, details):
    summaries=[]
    for rt in (False,True):
        cmd=[str(exe),str(path),"--max-n",str(maximum),"--threads","4"]
        if rt: cmd+=["--roundtrip"]
        result=subprocess.run(cmd,capture_output=True,text=True,check=True)
        obj=json.loads(result.stdout);summaries.append(obj)
    atomic_json(details,summaries)
    return all(x["rejected"]==0 for x in summaries)

def certify_and_maybe_promote(n, raw_q, bestpath, best, cert_exe, d, state, source_tag="GPU"):
    rows = best["rows"]
    for dilation in (1.0, 1.000002, 1.00001, 1.0001):
        candidate=[dict(square_id=str(i),x=format(x*dilation,".17g"),
                        y=format(y*dilation,".17g"),angle=format(angle,".17g"))
                   for i,(x,y,angle) in enumerate(raw_q)]
        if not all(math.isfinite(float(r[k])) for r in candidate for k in ("x","y","angle")):
            continue
        csv=d/"candidate.csv"
        write_submission(csv,{n:candidate},validate=False)
        if not cpp(cert_exe,csv,n,d/f"cpp{n}.json"):
            continue
        try:
            L=certify(n,candidate)
            rtL=certify(n,kaggle_roundtrip({n:candidate})[n])
        except ValueError:
            continue
        prev_rtL = best["roundtrip_L"] if "roundtrip_L" in best else certify(n,kaggle_roundtrip({n:rows})[n])
        improved = Decimal(L)<Decimal(best["L"]) and Decimal(rtL)<Decimal(prev_rtL)
        if improved:
            atomic_json(bestpath,dict(rows=candidate,L=L,roundtrip_L=rtL,source=source_tag,previous_L=best["L"]))
            state["gpu_improved"].append(n)
        return dict(status="certified",L=L,roundtrip_L=rtL,dilation=dilation,gpu_improved=improved)
    return dict(status="rejected",gpu_improved=False)

def run_parametric_family_prepass(d: Path, cert_exe: Path, state: dict, statefile: Path):
    """Run Ask #1 deterministic parametric family sweep on current best/<n>.json."""
    if state.get("families_prepass_done"):
        return
    try:
        from families import enumerate_families
        puzzles = {n: json.loads((d/"best"/f"{n}.json").read_text())["rows"] for n in range(1,1001)}
        cur_csv = d/"families-input.csv"
        write_submission(cur_csv, puzzles, validate=False)
        rep = enumerate_families(
            cur_csv, d/"families_prepass",
            max_n=1000, polish_iters=120,
            threads=min(8, os.cpu_count() or 4),
            update_best_dir=d,
        )
        for n_str in rep.get("improvements", {}):
            n_val = int(n_str)
            if n_val not in state["gpu_improved"]:
                state["gpu_improved"].append(n_val)
        state["families_prepass_done"] = True
        state["family_improved_count"] = rep.get("total_improved", 0)
        atomic_json(statefile, state)
    except Exception as e:
        state["families_prepass_error"] = str(e)
        atomic_json(statefile, state)

def main():
    p=argparse.ArgumentParser();p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--incumbent",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--use-families",action="store_true",default=True)
    p.add_argument("--no-families",action="store_false",dest="use_families")
    p.add_argument("--batch-size",type=int,default=1,help="Multi-n batch size for Ask #4 (1 = single-n)")
    p.add_argument("--solver",choices=["packing","seeded","langevin"],default="packing")
    a=p.parse_args();d=a.directory;d.mkdir(parents=True,exist_ok=True)
    exe=Path(f"cuda/{a.solver}").resolve();cert=Path("cuda/certifier").resolve()
    statefile=d/"state.json"
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

    if a.use_families:
        run_parametric_family_prepass(d, cert, state, statefile)

    # Ask #2: Gap-targeted search budget & ordering by expected score gain dL / sqrt(n)
    current_L = {
        n: float(Decimal(json.loads((d/"best"/f"{n}.json").read_text())["L"]))
        for n in range(1, 1001)
    }
    headroom_table = compute_headroom_table(current_L)
    atomic_json(d/"headroom-plan.json", {str(n): b for n, b in headroom_table.items()})
    is_multi_seed_pass = (a.seed != 42)
    ranked_items = rank_puzzles_by_headroom(current_L, pass_number=(2 if is_multi_seed_pass else 1))

    pending = []
    for b_info in ranked_items:
        n = b_info["n"]
        if str(n) in state["cases"]:
            continue
        if b_info.get("skip_on_multi_seed", False):
            # Ask #2: Run multiple seeds only on n with meaningful remaining headroom
            case = dict(
                n=n, status="skipped_low_headroom", gpu_improved=False,
                expected_score_gain=b_info["expected_score_gain"],
                headroom_L=b_info["headroom_L"], seconds=0.0
            )
            state["cases"][str(n)] = case
            continue
        pending.append(n)
    atomic_json(statefile, state)

    batch_size = max(1, a.batch_size)
    idx = 0
    while idx < len(pending):
        batch_ns = pending[idx : idx + batch_size]
        idx += batch_size
        if len(batch_ns) == 1:
            n = batch_ns[0]
            b_info = headroom_table[n]
            state["current_n"]=n;atomic_json(statefile,state)
            bestpath=d/"best"/f"{n}.json";best=json.loads(bestpath.read_text());rows=best["rows"]
            seedfile=d/"seed.txt";rawfile=d/f"raw{n}.json"
            seedfile.write_text(f"{n} {float(best['L'])*1.00002:.17g}\n"+"".join(
                " ".join(format(float(r[k])*(1.00002 if k!="angle" else 1),".17g")
                         for k in ("x","y","angle"))+"\n" for r in rows))
            t=time.time()
            case=dict(
                n=n,status="rejected",gpu_improved=False,
                expected_score_gain=b_info["expected_score_gain"],
                epochs=b_info["epochs"],steps=b_info["steps"]
            )
            try:
                cmd=[str(exe),str(seedfile),str(b_info["epochs"]),str(b_info["steps"]),"0",str(a.seed),str(rawfile)]
                result=subprocess.run(cmd,capture_output=True,text=True,timeout=900,check=True)
                (d/f"gpu{n}.log").write_text(result.stdout+result.stderr)
                raw=json.loads(rawfile.read_text())
                assert raw["n"]==n and len(raw["q"])==n
                promo = certify_and_maybe_promote(n, raw["q"], bestpath, best, cert, d, state, source_tag="GPU")
                case.update(promo)
            except (subprocess.SubprocessError,ValueError,AssertionError) as e:
                case["error"]=str(e)
            case["seconds"]=time.time()-t;state["cases"][str(n)]=case
            atomic_json(statefile,state);print(json.dumps(case),flush=True)
        else:
            # Ask #4: Batched multi-n kernel launch
            state["current_n"]=batch_ns[0];atomic_json(statefile,state)
            manifest_lines = []
            bests = {}
            max_epochs = 0
            max_steps = 0
            for n in batch_ns:
                b_info = headroom_table[n]
                max_epochs = max(max_epochs, b_info["epochs"])
                max_steps = max(max_steps, b_info["steps"])
                bestpath = d/"best"/f"{n}.json"
                best = json.loads(bestpath.read_text())
                bests[n] = (bestpath, best)
                rows = best["rows"]
                sfile = d/f"seed{n}.txt"
                rfile = d/f"raw{n}.json"
                sfile.write_text(f"{n} {float(best['L'])*1.00002:.17g}\n"+"".join(
                    " ".join(format(float(r[k])*(1.00002 if k!="angle" else 1),".17g")
                             for k in ("x","y","angle"))+"\n" for r in rows))
                manifest_lines.append(f"{sfile} {rfile} 0")
            manifest_path = d/"batch_manifest.txt"
            manifest_path.write_text("\n".join(manifest_lines) + "\n")
            summary_path = d/"batch_summary.json"
            t = time.time()
            batch_err = None
            try:
                cmd = [str(exe), f"--batch:{manifest_path}", str(max_epochs), str(max_steps), "0", str(a.seed), str(summary_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=True)
                (d/f"gpu_batch_{batch_ns[0]}.log").write_text(result.stdout + result.stderr)
            except subprocess.SubprocessError as e:
                batch_err = str(e)
            elapsed = (time.time() - t) / len(batch_ns)
            for n in batch_ns:
                b_info = headroom_table[n]
                bestpath, best = bests[n]
                rfile = d/f"raw{n}.json"
                case = dict(
                    n=n, status="rejected", gpu_improved=False,
                    expected_score_gain=b_info["expected_score_gain"],
                    epochs=max_epochs, steps=max_steps, batch_size=len(batch_ns)
                )
                if batch_err is not None:
                    case["error"] = batch_err
                else:
                    try:
                        raw = json.loads(rfile.read_text())
                        assert raw["n"] == n and len(raw["q"]) == n
                        promo = certify_and_maybe_promote(n, raw["q"], bestpath, best, cert, d, state, source_tag="GPU-batch")
                        case.update(promo)
                    except (ValueError, AssertionError, OSError) as e:
                        case["error"] = str(e)
                case["seconds"] = elapsed
                state["cases"][str(n)] = case
                print(json.dumps(case), flush=True)
            atomic_json(statefile, state)

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
