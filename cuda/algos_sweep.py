"""Optional resumable equal-wall sweep; only dual-certified improvements replace seed42.

State is tied to the exact commit, binary hashes, incumbent and search settings.
A completed trial can be resumed from its atomic trial.json after interruption.
Final CSV always contains all 1000 incumbent puzzles, with certified replacements.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from algos_bench import (INCUMBENT_SHA, atomic_json, core, cpp_gate, parser,
                         summarize, trial, validate_args, verify_incumbent)


def fingerprint(args, targets):
    commit = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True).strip()
    return dict(commit=commit, incumbent_sha256=INCUMBENT_SHA, targets=targets,
                seeds=args.seeds, arms=args.arms, budget_seconds=args.budget_seconds,
                epochs=args.epochs, steps=args.steps, chains=args.chains,
                binaries={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in {args.packing, args.algos, args.certifier}
                          if p.is_file()},
                driver_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__),
                                         Path(__file__).with_name("algos_bench.py"))})


def export(puzzles, directory, certifier):
    destination = directory / "submission-kaggle-safe.csv"
    # Every retained replacement has already passed both independent gates.
    core().write_submission(destination, puzzles, validate=False)
    cpp_reports = cpp_gate(certifier, destination, 1000, 1000, directory, "final")
    official = []
    for transported in (puzzles, core().kaggle_roundtrip(puzzles)):
        result = core().score(transported)
        if (result["accepted"], result["rejected"], result["missing"]) != (1000, 0, 0):
            raise ValueError("final official gate rejected CSV")
        official.append(result)
    atomic_json(directory / "official-literal.json", official[0])
    atomic_json(directory / "official-roundtrip.json", official[1])
    return dict(csv=str(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                literal_score=official[0]["score"], roundtrip_score=official[1]["score"],
                accepted=1000, rejected=0, missing=0, cpp_gates=cpp_reports)


def main():
    p = parser()
    p.description = __doc__
    p.set_defaults(directory=Path("algos-sweep"),
                   ns=[n for n in range(1, 1001) if math.isqrt(n) ** 2 != n])
    args = p.parse_args()
    validate_args(args)
    args.directory.mkdir(parents=True, exist_ok=True)
    targets = sorted(set(args.ns))
    if any(math.isqrt(n) ** 2 == n for n in targets):
        raise ValueError("sweep targets must be nonsquare")
    config = fingerprint(args, targets)
    statefile = args.directory / "state.json"
    if statefile.exists():
        state = json.loads(statefile.read_text())
        if state["config"] != config:
            raise ValueError("resume configuration/binary/commit changed; use a fresh directory")
    else:
        state = dict(config=config, completed={}, started_at=time.time())
        atomic_json(statefile, state)
    puzzles = verify_incumbent(args.incumbent, args.certifier, args.directory)
    original = dict(puzzles)
    bestdir = args.directory / "best"
    bestdir.mkdir(exist_ok=True)
    # Atomic per-n best files survive a crash between replacement and state save.
    for n in targets:
        bestpath = bestdir / f"{n}.json"
        if bestpath.exists():
            best = json.loads(bestpath.read_text())
            # Re-run both official gates after resume; never trust a corrupt state.
            side = core().certify(n, best["rows"])
            rt_side = core().certify(n, core().kaggle_roundtrip({n: best["rows"]})[n])
            if side != best["L"] or rt_side != best["roundtrip_L"]:
                raise ValueError(f"resume best mismatch n={n}")
            puzzles[n] = best["rows"]
    results = []
    for n in targets:
        side = core().certify(n, original[n])
        rt_side = core().certify(n, core().kaggle_roundtrip({n: original[n]})[n])
        current_side = core().certify(n, puzzles[n])
        current_rt = core().certify(n, core().kaggle_roundtrip({n: puzzles[n]})[n])
        for si, seed in enumerate(args.seeds):
            arms = args.arms[si % len(args.arms):] + args.arms[:si % len(args.arms)]
            for arm in arms:
                key = f"n{n}-{arm}-s{seed}"
                directory = args.directory / key
                cached = directory / "trial.json"
                # All trials start at ORIGINAL incumbent, not preceding arm output.
                if key in state["completed"] or cached.exists():
                    result = json.loads(cached.read_text())
                    if (result["n"], result["arm"], result["seed"]) != (n, arm, seed):
                        raise ValueError("cached trial identity mismatch")
                else:
                    state["current"] = key
                    atomic_json(statefile, state)
                    result = trial(n, original[n], side, rt_side, arm, seed, directory, args)
                if (Decimal(result["L"]) < Decimal(current_side) and
                        Decimal(result["roundtrip_L"]) < Decimal(current_rt)):
                    # Re-certify official geometry even for cached completed trials.
                    rows = result["best_rows"]
                    new_side = core().certify(n, rows)
                    new_rt = core().certify(n, core().kaggle_roundtrip({n: rows})[n])
                    if new_side != result["L"] or new_rt != result["roundtrip_L"]:
                        raise ValueError("cached trial geometry mismatch")
                    puzzles[n] = rows
                    current_side, current_rt = new_side, new_rt
                    atomic_json(bestdir / f"{n}.json",
                                dict(rows=rows, L=new_side, roundtrip_L=new_rt, source=key))
                state["completed"][key] = str(cached)
                state.pop("current", None)
                atomic_json(statefile, state)
                results.append(result)
                print(json.dumps(dict(trial=key, improved=result["improved"],
                                      completed=len(state["completed"]),
                                      expected=len(targets) * len(args.arms) * len(args.seeds))),
                      flush=True)
    report = export(puzzles, args.directory, args.certifier)
    report.update(targets=len(targets), completed_trials=len(state["completed"]),
                  expected_trials=len(targets) * len(args.arms) * len(args.seeds),
                  improved_n=[n for n in targets if (bestdir / f"{n}.json").exists()],
                  end_to_end_wall_seconds=time.time() - state["started_at"],
                  summaries=summarize(results), incumbent_sha256=INCUMBENT_SHA)
    if report["completed_trials"] != report["expected_trials"]:
        raise ValueError("sweep completed trial count mismatch")
    atomic_json(args.directory / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "summaries"}), flush=True)


if __name__ == "__main__":
    main()
