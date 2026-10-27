import numpy as np
import pytest

from squares.core import certify, grid
from squares.optimizer import objective, optimize_packing, tilted_five


def test_tilted_five_certified_improvement():
    rows = tilted_five()
    assert len(rows) == 5
    assert 2.7071 < float(certify(5, rows)) < 2.7072


def test_optimizer_returns_certified_seed():
    rows, report = optimize_packing(5, attempts=0, seed=7)
    assert float(certify(5, rows)) < 3
    assert report["improvements"] == 1


def test_numeric_search_smoke():
    rows, report = optimize_packing(2, attempts=1, maxiter=3, seed=42)
    assert float(certify(2, rows)) <= 2
    assert report["attempts"] == 1


def test_perfect_square_skips_search():
    rows, report = optimize_packing(4, attempts=2)
    assert float(certify(4, rows)) == 2
    assert report["attempts"] == 0


def test_penalty_distinguishes_overlap():
    assert objective(np.array([[.5, .5, 0], [1.5, .5, 0]]), 2, 2) == 0
    assert objective(np.array([[.5, .5, 0], [.5, .5, 0]]), 2, 2) > 0


def test_invalid_warmstart_never_accepted():
    rows = grid(2)
    rows[1]["x"] = rows[0]["x"]
    with pytest.raises(ValueError):
        optimize_packing(2, rows, attempts=0)


def test_small_n_guard():
    with pytest.raises(ValueError):
        optimize_packing(51, attempts=0)
