"""Decimal-preserving CSV IO and official-metric validation.

The bundled metric owns the geometry and tolerance policy. Do not replace its
interval contact checks with a floating-point overlap tolerance.
"""
from __future__ import annotations

import csv
import io
import os
import tempfile
from decimal import Decimal
from pathlib import Path

from . import _official_metric as metric

COLUMNS = ("id", "n", "square_id", "x", "y", "angle")
Packing = list[dict[str, str]]
Puzzles = dict[int, Packing]


def check_n(n: int) -> None:
    if not 1 <= n <= metric.MAX_N:
        raise ValueError("n must be in 1..1000")


def grid(n: int) -> Packing:
    """Exact half-integer grid centers with integer pitch, zero rotation."""
    check_n(n)
    width = metric.math.isqrt(n - 1) + 1
    return [
        dict(square_id=str(i), x=f"{i % width}.5",
             y=f"{i // width}.5", angle="0")
        for i in range(n)
    ]


def read_submission(path: str | Path) -> Puzzles:
    """Read strings, never floats; enforce the submission's row-id envelope."""
    text = Path(path).read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise ValueError("CSV header must be id,n,square_id,x,y,angle")
    for i, row in enumerate(reader):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"row {i + 2} has the wrong number of fields")
        if row["id"] != str(i):
            raise ValueError(f"row {i + 2} must have id {i}")
    return metric._parse_submission_csv(text)


def evaluate(n: int, rows: Packing) -> tuple[int, float, str | None, str]:
    check_n(n)
    return metric._evaluate_puzzle_worker((n, rows))


def certify(n: int, rows: Packing) -> str:
    """Return L as a 50-decimal string; refuse malformed/uncertified results."""
    _, _, side, reason = evaluate(n, rows)
    if side is None or reason != "ok":
        raise ValueError(f"n={n}: {reason}")
    return side


def write_submission(path: str | Path, puzzles: Puzzles, *,
                     validate: bool = True) -> None:
    """Certify before atomically replacing a CSV. Empty/partial files are valid.

    validate=False is reserved for grid(), whose exact rational geometry is
    certified by construction. All imported/searched/merged output uses True.
    """
    for n, rows in puzzles.items():
        check_n(n)
        if validate:
            certify(n, rows)
        else:
            error = metric._validate_puzzle(n, rows)
            if error:
                raise ValueError(error)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            index = 0
            for n, rows in sorted(puzzles.items()):
                for row in sorted(rows, key=lambda r: int(r["square_id"])):
                    writer.writerow(dict(
                        id=index, n=n, square_id=row["square_id"],
                        x=row["x"], y=row["y"], angle=row["angle"],
                    ))
                    index += 1
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def score(puzzles: Puzzles, max_n: int = 1000) -> dict:
    """Official score with grid penalties for absent or rejected puzzles."""
    check_n(max_n)
    total, results = metric._evaluate_puzzles_parallel(
        puzzles, set(range(1, max_n + 1)),
    )
    return dict(
        score=total, max_n=max_n,
        accepted=sum(r[3] == "ok" for r in results),
        missing=sum(r[3] == "missing from submission" for r in results),
        rejected=sum(r[3] not in ("ok", "missing from submission") for r in results),
        results=[dict(n=n, score=value, L=side, reason=reason)
                 for n, value, side, reason in results],
    )


def merge_best(candidates: list[Puzzles]) -> Puzzles:
    """Keep certified smallest L per n, preserving original decimal strings.

    Invalid candidates never replace valid ones; ties keep the first input.
    Missing n stay missing, with the official grid penalty at scoring time.
    """
    best: Puzzles = {}
    sides: dict[int, Decimal] = {}
    for puzzles in candidates:
        for n, rows in sorted(puzzles.items()):
            _, _, side, reason = evaluate(n, rows)
            if reason != "ok" or side is None:
                continue
            length = Decimal(side)
            if n not in sides or length < sides[n]:
                sides[n] = length
                best[n] = [dict(row) for row in rows]
    return best
