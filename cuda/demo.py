"""Seed generation and official interval certification for the standalone CUDA POC."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from squares.core import grid, certify, write_submission
from squares.optimizer import tilted_five

def seed(n):
    return tilted_five() if n == 5 else grid(n)

def prepare(directory, ns):
    directory.mkdir(parents=True, exist_ok=True)
    for n in ns:
        rows = seed(n)
        side = float(certify(n, rows))
        text = f"{n} {side * 1.00002:.12g}\n"
        text += "".join(
            " ".join(format(float(r[k]) * (1.00002 if k != "angle" else 1), ".12g")
                     for k in ("x", "y", "angle")) + "\n" for r in rows)
        (directory / f"seed{n}.txt").write_text(text)

def certified_candidate(raw):
    n = raw["n"]
    q = np.asarray(raw["q"], dtype=np.float64)
    if q.shape != (n, 3) or not np.isfinite(q).all():
        raise ValueError("no finite GPU zero-energy candidate")
    # FP32 zero is not a proof. Bounded center dilation, then official interval
    # certification. This is not an overlap-tolerance bypass.
    for dilation in (1.000002, 1.00001, 1.0001):
        rows = [dict(square_id=str(i), x=format(x*dilation, ".17g"),
                     y=format(y*dilation, ".17g"), angle=format(a, ".17g"))
                for i, (x, y, a) in enumerate(q)]
        try:
            return rows, float(certify(n, rows)), dilation
        except ValueError:
            pass
    raise ValueError("official certifier rejected GPU candidate")

def report(directory, ns):
    report = []
    puzzles = {}
    for n in ns:
        raw = json.loads((directory / f"raw{n}.json").read_text())
        baseline = float(certify(n, seed(n)))
        row = dict(n=n, grid_L=float(certify(n, grid(n))), repo_seed_L=baseline,
                   seconds=raw["seconds"],
                   chain_steps_per_second=raw["chain_steps"]/raw["seconds"],
                   proposals_per_second=raw["proposals"]/raw["seconds"],
                   feasible_checkpoints=raw["feasible_checkpoints"],
                   shrinks=raw["shrinks"], gpu_certified_L=None)
        try:
            packing, side, dilation = certified_candidate(raw)
            row.update(gpu_certified_L=side, dilation=dilation,
                       improves_repo_seed=side < baseline)
            puzzles[n] = packing
        except ValueError as exc:
            row["rejection"] = str(exc)
        report.append(row)
    # Write only certified GPU proposals, without silently substituting seeds.
    if puzzles:
        write_submission(directory / "gpu-certified.csv", puzzles)
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "report", "run"))
    p.add_argument("--directory", type=Path, default=Path("outputs/cuda"))
    p.add_argument("--ns", type=int, nargs="+", default=[5,10,11,17,50])
    p.add_argument("--executable", default="./packing")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--chains", type=int, default=128)
    a = p.parse_args()
    if any(n < 1 or n > 50 for n in a.ns):
        p.error("POC supports n=1..50")
    if a.mode in ("prepare", "run"):
        prepare(a.directory, a.ns)
    if a.mode == "run":
        for n in a.ns:
            subprocess.run([a.executable, str(a.directory/f"seed{n}.txt"),
                            str(a.epochs), str(a.steps), str(a.chains), "42",
                            str(a.directory/f"raw{n}.json")], check=True)
    if a.mode in ("report", "run"):
        report(a.directory, a.ns)

if __name__ == "__main__":
    main()
