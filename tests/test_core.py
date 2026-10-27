import csv
import math
from decimal import Decimal

import pytest

from squares import _official_metric as metric
from squares.cli import main
from squares.core import (
    certify, evaluate, grid, merge_best, read_submission, score, write_submission,
)


def pair(second_x):
    return [
        dict(square_id="0", x=".5", y=".5", angle="0"),
        dict(square_id="1", x=second_x, y=".5", angle="0"),
    ]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 9, 10, 16, 31])
def test_grid_metric(n):
    side = certify(n, grid(n))
    assert Decimal(side) == math.ceil(math.sqrt(n))


@pytest.mark.parametrize("x,accepted", [
    ("1.5", True),
    ("1.499999999999999999999999999999", False),
    ("1.500000000000000000000000000001", True),
    (".6", False),
])
def test_contact_no_overlap_tolerance(x, accepted):
    result = evaluate(2, pair(x))
    assert (result[3] == "ok") is accepted


@pytest.mark.parametrize("field,value", [
    ("x", "sqrt(2)"), ("y", "nan"), ("x", "1e400"),
    ("x", "10001"), ("angle", "101"), ("x", "1." + "0" * 101),
])
def test_official_input_limits(field, value):
    rows = grid(1)
    rows[0][field] = value
    with pytest.raises(ValueError):
        certify(1, rows)


def test_duplicate_ids_and_count():
    rows = grid(2)
    rows[1]["square_id"] = "0"
    assert evaluate(2, rows)[3] != "ok"
    assert evaluate(3, grid(2))[3] != "ok"


def test_translation_invariance():
    a = grid(4)
    b = [dict(r, x=str(Decimal(r["x"]) - 17),
              y=str(Decimal(r["y"]) + 500)) for r in a]
    assert certify(4, a) == certify(4, b)


def test_csv_roundtrip_preserves_decimal(tmp_path):
    rows = pair("1.500000000000000000000000000001")
    path = tmp_path / "packing.csv"
    write_submission(path, {2: rows})
    loaded = read_submission(path)
    assert loaded[2][1]["x"] == rows[1]["x"]
    assert len(list(csv.DictReader(path.open()))) == 2


def test_write_refuses_invalid_and_preserves_existing(tmp_path):
    path = tmp_path / "packing.csv"
    path.write_text("keep me")
    with pytest.raises(ValueError):
        write_submission(path, {2: pair(".6")})
    assert path.read_text() == "keep me"
    assert list(tmp_path.iterdir()) == [path]


def test_bad_row_ids(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,n,square_id,x,y,angle\n3,1,0,0,0,0\n")
    with pytest.raises(ValueError, match="id 0"):
        read_submission(path)


def test_missing_and_invalid_have_grid_penalty():
    report = score({2: pair(".6")}, max_n=3)
    assert report["score"] == sum(metric.infeasible_score(n) for n in range(1, 4))
    assert report["rejected"] == 1
    assert report["missing"] == 2


def test_score_matches_official_entrypoint(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.chdir(tmp_path)
    rows = pair("1.5")
    sub = pd.DataFrame([dict(id=i, n=2, **r) for i, r in enumerate(rows)])
    sol = pd.DataFrame(dict(id=[0, 1, 2], n=[1, 2, 3]))
    expected = metric.score(sol, sub, "id")
    assert score({2: rows}, max_n=3)["score"] == expected
    assert (tmp_path / "submission_details.csv").is_file()


def test_merge_keeps_best_and_rejects_invalid():
    worse = pair("2.5")
    best = pair("1.5")
    merged = merge_best([{2: worse}, {2: best}, {2: pair(".6")}, {1: grid(1)}])
    assert merged[2] == best
    assert set(merged) == {1, 2}


def test_grid_perfect_squares():
    for n in (1, 4, 9, 16):
        assert evaluate(n, grid(n))[1] == 0


def test_complete_validation(tmp_path):
    path = tmp_path / "partial.csv"
    write_submission(path, {1: grid(1)})
    with pytest.raises(SystemExit) as error:
        main(["validate", str(path), "--max-n", "2", "--require-complete"])
    assert error.value.code == 2


def test_baseline_cli(tmp_path):
    path = tmp_path / "baseline.csv"
    main(["baseline", "--max-n", "10", "--output", str(path)])
    puzzles = read_submission(path)
    assert set(puzzles) == set(range(1, 11))
    assert sum(map(len, puzzles.values())) == 55


@pytest.mark.parametrize("n", [0, -1, 1001])
def test_invalid_n(n):
    with pytest.raises(ValueError):
        grid(n)
