"""Local-only command line tools; no Kaggle credentials or submissions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import certify, grid, merge_best, read_submission, score, write_submission


def bounded_n(value: str) -> int:
    n = int(value)
    if not 1 <= n <= 1000:
        raise argparse.ArgumentTypeError("must be in 1..1000")
    return n


def emit(report: dict, path: str | None = None) -> None:
    text = json.dumps(report, indent=2) + "\n"
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "results"}
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    baseline = commands.add_parser("baseline", help="exact grid for n=1..max-n")
    baseline.add_argument("--max-n", type=bounded_n, default=1000)
    baseline.add_argument("--output", default="submission.csv")

    for name in ("validate", "score"):
        command = commands.add_parser(name)
        command.add_argument("input")
        command.add_argument("--max-n", type=bounded_n, default=1000)
        command.add_argument("--report")
        if name == "validate":
            command.add_argument("--require-complete", action="store_true")

    merge = commands.add_parser("merge", help="certified best side per n")
    merge.add_argument("inputs", nargs="+")
    merge.add_argument("--output", required=True)

    records = commands.add_parser("references", help="load credited local JSON records")
    records.add_argument("input")
    records.add_argument("--output", required=True)
    records.add_argument("--provenance", required=True)

    optimize = commands.add_parser("optimize", help="small-n certified heuristic")
    optimize.add_argument("--n", type=bounded_n, required=True)
    optimize.add_argument("--input", help="optional CSV warm start")
    optimize.add_argument("--attempts", type=int, default=20)
    optimize.add_argument("--maxiter", type=int, default=250)
    optimize.add_argument("--seed", type=int, default=42)
    optimize.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "baseline":
            puzzles = {n: grid(n) for n in range(1, args.max_n + 1)}
            # Half-integer centers, disjoint unit grid cells, angle=0 prove
            # feasibility exactly. No floating-point heuristic is involved.
            write_submission(args.output, puzzles, validate=False)
            from . import _official_metric as metric
            emit(dict(
                output=args.output, puzzles=args.max_n,
                rows=sum(len(rows) for rows in puzzles.values()),
                score=sum(metric.infeasible_score(n)
                          for n in range(1, args.max_n + 1)),
                certification="exact rational grid construction",
            ))
        elif args.command in ("validate", "score"):
            puzzles = read_submission(args.input)
            if args.command == "validate":
                for n, rows in puzzles.items():
                    certify(n, rows)
                if args.require_complete and set(puzzles) != set(range(1, args.max_n + 1)):
                    raise ValueError("submission does not contain exactly 1..max-n")
            # score() deliberately retains official fallback behavior.
            emit(score(puzzles, args.max_n), args.report)
        elif args.command == "merge":
            puzzles = merge_best([read_submission(p) for p in args.inputs])
            write_submission(args.output, puzzles)
            emit(dict(output=args.output, puzzles=len(puzzles)))
        elif args.command == "references":
            from .references import load_records
            puzzles, provenance = load_records(args.input)
            write_submission(args.output, puzzles)
            emit(dict(records=provenance), args.provenance)
        elif args.command == "optimize":
            from .optimizer import optimize_packing
            initial = None
            if args.input:
                initial = read_submission(args.input).get(args.n)
                if initial is None:
                    raise ValueError(f"warm start has no n={args.n}")
            rows, report = optimize_packing(
                args.n, initial, attempts=args.attempts,
                seed=args.seed, maxiter=args.maxiter,
            )
            write_submission(args.output, {args.n: rows})
            emit(report)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
