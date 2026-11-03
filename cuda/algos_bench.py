"""Equal end-to-end search budgets; certification is deferred and separately timed.

Every arm starts from the same incumbent, runs independent short restart chunks,
and gets the same wall deadline, including Python scheduling and CUDA startup.
Only fully written JSON from successful processes counts. No GPU time claim is
inferred from requested work. A timed-out final chunk contributes no chain_steps.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
INCUMBENT_SHA = "49f10654a8edc9b5eead5cb96c60f4aa68e7dcb7c5259a545ebd95d7a39147fc"
DEFAULT_NS = (19, 102, 147, 232, 260, 333, 412, 500, 700, 900)
DEFAULT_INCUMBENT = Path("/home/kit/seed42-verification/outputs/full-4070s/submission-kaggle-safe.csv")


def core():
    # Lazy imports keep subprocess-budget tests usable without metric dependencies.
    from squares import core as c
    return c


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def cpp_gate(certifier, csv, expected, maximum, directory, prefix):
    """Partial candidates must accept EXACTLY one n, with all other n missing."""
    reports = []
    for rt in (False, True):
        details = directory / f"{prefix}-{'roundtrip' if rt else 'literal'}.json"
        cmd = [str(certifier), str(csv), "--max-n", str(maximum),
               "--threads", "4", "--details", str(details)]
        if rt:
            cmd.append("--roundtrip")
        p = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1800)
        report = json.loads(p.stdout)
        if (report["accepted"], report["rejected"], report["missing"]) != (
                expected, 0, maximum - expected):
            raise ValueError(f"certifier exact counts failed: {report}")
        reports.append(report)
    return reports


def verify_incumbent(path, certifier, directory):
    if hashlib.sha256(path.read_bytes()).hexdigest() != INCUMBENT_SHA:
        raise ValueError("incumbent SHA256 differs from the authorized seed42 CSV")
    puzzles = core().read_submission(path)
    if set(puzzles) != set(range(1, 1001)):
        raise ValueError("incumbent must contain exactly n=1..1000")
    reports = cpp_gate(certifier, path, 1000, 1000, directory, "incumbent")
    atomic_json(directory / "incumbent-gates.json", reports)
    return puzzles


def write_seed(path, n, rows, side):
    factor = 1.00002
    path.write_text(f"{n} {float(side) * factor:.17g}\n" + "".join(
        " ".join(format(float(r[k]) * (factor if k != "angle" else 1), ".17g")
                 for k in ("x", "y", "angle")) + "\n" for r in rows))


def command(executable, seedfile, epochs, steps, chains, seed, output, mode):
    # algos accepts the packing six-argument CLI and an optional mode argument.
    args = [str(executable), str(seedfile), str(epochs), str(steps), str(chains),
            str(seed), str(output)]
    if mode is not None:
        args.append(str(mode))
    return args


def run_budget(executable, seedfile, directory, budget, epochs, steps, chains, seed,
               mode=None):
    """Search only. Keep each completed checkpoint; never trust a partial timeout."""
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be positive and finite")
    directory.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    deadline = begin + budget
    attempts = []
    completed = []
    work = 0
    while time.monotonic() < deadline:
        index = len(attempts)
        output = directory / f"raw-{index:05d}.json"
        output.unlink(missing_ok=True)
        cmd = command(executable, seedfile, epochs, steps, chains,
                      seed + index * 1000003, output, mode)
        env = dict(os.environ)
        # Remove inherited mode so packing and explicit mode arms are independent.
        env.pop("ALGOS_MODE", None)
        if mode is not None:
            env["ALGOS_MODE"] = str(mode)
        record = dict(index=index, command=cmd, status="error", chain_steps=0)
        t = time.monotonic()
        remaining = deadline - t
        if remaining <= 0:
            break
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                               timeout=remaining, check=False)
            record["returncode"] = p.returncode
            (directory / f"process-{index:05d}.log").write_text(p.stdout + p.stderr)
            # Conservative deadline: late teardown or disk write cannot earn work.
            if time.monotonic() > deadline:
                record["status"] = "late"
            elif p.returncode == 0:
                raw = json.loads(output.read_text())
                steps_done = raw.get("chain_steps")
                if not isinstance(steps_done, int) or isinstance(steps_done, bool) or steps_done < 0:
                    raise ValueError("missing/nonintegral measured chain_steps")
                record.update(status="completed", chain_steps=steps_done,
                              path=str(output))
                completed.append(str(output))
                work += steps_done
        except subprocess.TimeoutExpired as e:
            record["status"] = "timeout"
            log = (e.stdout or b"") + (e.stderr or b"")
            if isinstance(log, bytes):
                log = log.decode(errors="replace")
            (directory / f"process-{index:05d}.log").write_text(log)
        except (OSError, ValueError, TypeError) as e:
            record["error"] = str(e)
        record["wall_seconds"] = time.monotonic() - t
        attempts.append(record)
        # A binary/config failure is not a reason to hammer the GPU until deadline.
        if record["status"] == "error":
            break
    utilized = time.monotonic() - begin
    return dict(budget_seconds=budget, utilized_wall_seconds=utilized,
                chain_steps=work, chain_steps_per_second=work / utilized,
                attempts=attempts, completed=completed,
                timeouts=sum(a["status"] in ("timeout", "late") for a in attempts),
                process_errors=sum(a["status"] == "error" for a in attempts))


def certify_raw(rawpath, n, certifier, directory):
    """C++ literal + roundtrip, then unchanged official literal + pandas gate."""
    raw = json.loads(Path(rawpath).read_text())
    q = raw.get("q")
    if raw.get("n") != n or not isinstance(q, list) or len(q) != n:
        raise ValueError("candidate n/coordinate count mismatch")
    if not all(isinstance(v, (float, int)) and not isinstance(v, bool)
               and math.isfinite(v) for triple in q
               if isinstance(triple, list) for v in triple):
        raise ValueError("nonfinite or malformed coordinates")
    if not all(isinstance(t, list) and len(t) == 3 for t in q):
        raise ValueError("coordinates must be triples")
    errors = []
    for factor in (1.0, 1.000002, 1.00001, 1.0001):
        rows = [dict(square_id=str(i), x=format(x * factor, ".17g"),
                     y=format(y * factor, ".17g"), angle=format(a, ".17g"))
                for i, (x, y, a) in enumerate(q)]
        csv = directory / (Path(rawpath).stem + f"-{factor}.csv")
        try:
            core().write_submission(csv, {n: rows}, validate=False)
            reports = cpp_gate(certifier, csv, 1, n, directory, csv.stem)
            side = core().certify(n, rows)
            rt_side = core().certify(n, core().kaggle_roundtrip({n: rows})[n])
            return dict(rows=rows, L=side, roundtrip_L=rt_side, dilation=factor,
                        gates=reports, source=str(rawpath))
        except (ValueError, subprocess.SubprocessError) as e:
            errors.append(str(e))
    raise ValueError("candidate failed all literal/roundtrip dilations: " + "; ".join(errors))


def trial(n, rows, side, rt_side, arm, random_seed, directory, args):
    directory.mkdir(parents=True, exist_ok=True)
    seedfile = directory / "seed.txt"
    write_seed(seedfile, n, rows, side)
    mode = None if arm == "packing" else int(arm[-1])
    executable = args.packing if mode is None else args.algos
    result = run_budget(executable, seedfile, directory, args.budget_seconds,
                        args.epochs, args.steps, args.chains, random_seed, mode)
    result.update(n=n, arm=arm, seed=random_seed, incumbent_L=side,
                  incumbent_roundtrip_L=rt_side, L=side, roundtrip_L=rt_side,
                  candidate_accepted=0, candidate_invalid=0, improved=False,
                  best_rows=rows)
    t = time.monotonic()
    for rawpath in result["completed"]:
        try:
            candidate = certify_raw(rawpath, n, args.certifier, directory)
            result["candidate_accepted"] += 1
            # Require strict improvement in BOTH unchanged transport paths.
            if (Decimal(candidate["L"]) < Decimal(result["L"]) and
                    Decimal(candidate["roundtrip_L"]) < Decimal(result["roundtrip_L"])):
                result.update(L=candidate["L"], roundtrip_L=candidate["roundtrip_L"],
                              best_rows=candidate["rows"], best_source=rawpath,
                              improved=True)
        except (OSError, ValueError, TypeError, subprocess.SubprocessError) as e:
            result["candidate_invalid"] += 1
            result.setdefault("candidate_errors", []).append(dict(path=rawpath, error=str(e)))
    result["certification_seconds"] = time.monotonic() - t
    atomic_json(directory / "trial.json", result)
    return result


def summarize(results):
    groups = defaultdict(list)
    for r in results:
        groups[(r["n"], r["arm"])].append(r)
    summaries = []
    for (n, arm), group in sorted(groups.items()):
        row = dict(n=n, arm=arm, trials=len(group),
                   median_L=statistics.median(Decimal(r["L"]) for r in group),
                   median_chain_steps_per_second=statistics.median(
                       r["chain_steps_per_second"] for r in group),
                   wins_below_incumbent=sum(r["improved"] for r in group),
                   invalid=sum(r["candidate_invalid"] for r in group),
                   accepted_candidates=sum(r["candidate_accepted"] for r in group),
                   timeouts=sum(r["timeouts"] for r in group),
                   process_errors=sum(r["process_errors"] for r in group),
                   utilized_wall_seconds=sum(r["utilized_wall_seconds"] for r in group),
                   allocated_wall_seconds=sum(r["budget_seconds"] for r in group),
                   certification_seconds=sum(r["certification_seconds"] for r in group))
        row["median_L"] = str(row["median_L"])
        summaries.append(row)
        print(json.dumps(row), flush=True)
    return summaries


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--incumbent", type=Path, default=DEFAULT_INCUMBENT)
    p.add_argument("--directory", type=Path, default=Path("algos-benchmark"))
    p.add_argument("--packing", type=Path, default=ROOT / "cuda/packing")
    p.add_argument("--algos", type=Path, default=ROOT / "cuda/algos")
    p.add_argument("--certifier", type=Path, default=ROOT / "cuda/certifier")
    p.add_argument("--ns", type=int, nargs="+", default=list(DEFAULT_NS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--arms", nargs="+", choices=["packing", "mode1", "mode2", "mode3"],
                   default=["packing", "mode1", "mode2", "mode3"])
    p.add_argument("--budget-seconds", type=float, default=30)
    p.add_argument("--epochs", type=int, default=1,
                   help="short completed checkpoint chunk, NOT the trial budget")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--chains", type=int, default=0)
    return p


def validate_args(args):
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        raise ValueError("invalid wall budget")
    if not 1 <= args.epochs <= 120 or not 1 <= args.steps <= 4096:
        raise ValueError("epochs must be 1..120 and steps 1..4096")
    if args.chains < 0 or args.chains > 4096 or args.chains % 8:
        raise ValueError("chains must be 0 (autotune) or multiple of 8 up to 4096")
    if not all(1 <= n <= 1000 for n in args.ns):
        raise ValueError("n out of range")
    if len(set(args.ns)) != len(args.ns) or len(set(args.arms)) != len(args.arms):
        raise ValueError("duplicate n/arm would overwrite a trial directory")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**63 for s in args.seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    for key in ("incumbent", "packing", "algos", "certifier", "directory"):
        setattr(args, key, getattr(args, key).resolve())
    for executable in {args.certifier} | ({args.packing} if "packing" in args.arms else set()) | (
            {args.algos} if any(a != "packing" for a in args.arms) else set()):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"missing executable {executable}; run make -C cuda all ARCH=sm_80")


def main():
    args = parser().parse_args()
    validate_args(args)
    args.directory.mkdir(parents=True, exist_ok=True)
    puzzles = verify_incumbent(args.incumbent, args.certifier, args.directory)
    atomic_json(args.directory / "run-config.json",
                dict(incumbent_sha256=INCUMBENT_SHA, settings={
                    k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    binaries={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in {args.packing, args.algos, args.certifier} if p.is_file()},
                    commit=subprocess.check_output(
                        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()))
    results = []
    for n in args.ns:
        side = core().certify(n, puzzles[n])
        rt_side = core().certify(n, core().kaggle_roundtrip({n: puzzles[n]})[n])
        for si, seed in enumerate(args.seeds):
            # Rotate order to reduce first-arm thermal / shared-GPU bias.
            arms = args.arms[si % len(args.arms):] + args.arms[:si % len(args.arms)]
            for arm in arms:
                result = trial(n, puzzles[n], side, rt_side, arm, seed,
                               args.directory / f"n{n}-{arm}-s{seed}", args)
                results.append(result)
                print(json.dumps({k: v for k, v in result.items()
                                  if k not in ("best_rows", "attempts")}), flush=True)
                atomic_json(args.directory / "summary.json", summarize(results))


class BudgetTests(unittest.TestCase):
    def fake(self, directory, body):
        executable = directory / "fake"
        executable.write_text("#!" + sys.executable + "\n" + body)
        executable.chmod(0o755)
        return executable

    def test_completed_then_timeout_retains_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            exe = self.fake(d, """import json,sys,time
from pathlib import Path
out=Path(sys.argv[6])
if out.stem != 'raw-00000': time.sleep(2)
out.write_text(json.dumps(dict(n=1,q=[[0,0,0]],chain_steps=17)))
""")
            r = run_budget(exe, d / "seed", d / "run", .25, 1, 1, 8, 42, 2)
            self.assertEqual(len(r["completed"]), 1)
            self.assertEqual(r["chain_steps"], 17)
            self.assertEqual(r["timeouts"], 1)
            self.assertGreaterEqual(r["utilized_wall_seconds"], .24)
            self.assertEqual(r["attempts"][0]["command"][-1], "2")

    def test_partial_failed_output_never_counts(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            exe = self.fake(d, "import sys\nfrom pathlib import Path\nPath(sys.argv[6]).write_text('{')\nsys.exit(2)\n")
            r = run_budget(exe, d / "seed", d / "run", .25, 1, 1, 8, 42)
            self.assertEqual(r["completed"], [])
            self.assertEqual(r["chain_steps"], 0)
            self.assertEqual(r["process_errors"], 1)

    def test_bad_work_counter_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            exe = self.fake(d, """import sys,json
from pathlib import Path
Path(sys.argv[6]).write_text(json.dumps(dict(chain_steps=-1)))
""")
            r = run_budget(exe, d / "seed", d / "run", .25, 1, 1, 8, 42)
            self.assertEqual(r["process_errors"], 1)
            self.assertEqual(r["chain_steps"], 0)

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            run_budget(Path("missing"), Path("missing"), Path("."), float("nan"), 1, 1, 8, 1)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
    else:
        main()
