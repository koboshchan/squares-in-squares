import json
from pathlib import Path

import pytest

from squares.references import load_records


def test_example():
    root = Path(__file__).resolve().parents[1]
    puzzles, provenance = load_records(root / "examples/records.json")
    assert len(puzzles[1]) == 1
    assert provenance[1]["attribution"]


@pytest.mark.parametrize("mutation", ["number", "missing-credit", "overlap", "duplicate"])
def test_bad_reference(tmp_path, mutation):
    data = dict(records=[dict(
        n=1, source="https://example.com/packing", attribution="Example",
        squares=[dict(x=".5", y=".5", angle="0")],
    )])
    record = data["records"][0]
    if mutation == "number":
        record["squares"][0]["x"] = .5
    elif mutation == "missing-credit":
        del record["attribution"]
    elif mutation == "overlap":
        record["n"] = 2
        record["squares"] *= 2
    else:
        data["records"] *= 2
    path = tmp_path / "records.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_records(path)
