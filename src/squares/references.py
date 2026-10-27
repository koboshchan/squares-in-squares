"""Offline adapter stub for credited, manually transcribed packing records.

No page scraping, SVG execution, network requests or symbolic eval. Record
tables with lengths alone cannot be submitted: explicit centers are required.
"""
import json
from pathlib import Path

from .core import Puzzles, certify, check_n

FRIEDMAN_URL = "https://erich-friedman.github.io/packing/index.html"
ELLSWORTH_URL = "https://kingbird.myphotos.cc/packing/squares_in_squares.html"


def load_records(path: str | Path) -> tuple[Puzzles, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError("expected an object with a records array")
    puzzles = {}
    provenance = {}
    for record in data["records"]:
        n = record["n"]
        if type(n) is not int:
            raise ValueError("record n must be an integer")
        check_n(n)
        if n in puzzles:
            raise ValueError(f"duplicate reference n={n}")
        source = record.get("source", "")
        attribution = record.get("attribution", "")
        if not isinstance(source, str) or not source.startswith("https://"):
            raise ValueError("record requires an https source URL")
        if not isinstance(attribution, str) or not attribution.strip():
            raise ValueError("record requires attribution")
        rows = []
        for i, square in enumerate(record["squares"]):
            if any(not isinstance(square.get(k), str) for k in ("x", "y", "angle")):
                raise ValueError("reference coordinates must be decimal strings")
            rows.append(dict(square_id=str(i), **{
                k: square[k] for k in ("x", "y", "angle")
            }))
        side = certify(n, rows)
        puzzles[n] = rows
        provenance[n] = dict(source=source, attribution=attribution, L=side)
    return puzzles, provenance
