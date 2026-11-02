"""Unit and regression tests for Ask #1 (parametric families) and Ask #2 (gap-targeted budget)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "cuda"))
from budget import PUBLISHED_TARGETS, compute_headroom_table, geometric_target_L, rank_puzzles_by_headroom
from families import FAMILY_NAMES, enumerate_families
from squares.core import certify, grid, kaggle_roundtrip, read_submission, write_submission


@pytest.fixture(scope="module")
def cpp_certifier():
    root = Path(__file__).parents[1] / "cuda"
    src = root / "certifier.cpp"
    exe = root / "certifier"
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            ["c++", "-O3", "-std=c++17", "-pthread", str(src), "-o", str(exe)],
            check=True,
        )
    return exe


def test_five_parametric_families_beat_grid_and_pass_official_certification(cpp_certifier, tmp_path):
    """Verify that enumerate_families evaluates all 5 families, beats grid, and passes both literal & roundtrip."""
    puzzles = {n: grid(n) for n in range(1, 70)}
    inc_csv = tmp_path / "grid69.csv"
    write_submission(inc_csv, puzzles, validate=False)

    out_dir = tmp_path / "fam_out"
    report = enumerate_families(
        inc_csv,
        out_dir,
        max_n=69,
        polish_iters=80,
        threads=4,
    )
    assert set(report["win_counts_by_family"].keys()) == set(FAMILY_NAMES)
    assert report["total_improved"] >= 20
    assert report["total_score_gain"] > 0.3

    merged_csv = Path(report["certified_csv"])
    assert merged_csv.exists()
    merged_puzzles = read_submission(merged_csv)
    assert len(merged_puzzles) == 69

    # Verify every improved puzzle passes official Python certify in both literal and roundtrip
    for n_str, imp in report["improvements"].items():
        n = int(n_str)
        rows = merged_puzzles[n]
        official_L = certify(n, rows)
        rt_L = certify(n, kaggle_roundtrip({n: rows})[n])
        grid_L = certify(n, puzzles[n])
        assert Decimal(official_L) < Decimal(grid_L)
        assert Decimal(rt_L) < Decimal(grid_L)
        assert imp["family"] in FAMILY_NAMES


def test_gap_targeted_search_budget_ranking_and_allocation():
    """Verify headroom estimation, monotone neighbor bound, and proportional GPU budget allocation."""
    current_L = {n: float(geometric_target_L(n)) for n in range(1, 101)}
    current_L[19] = 5.0  # Significant headroom over ~4.757 target
    current_L[5] = 2.7071067812  # Already tight at optimal

    table = compute_headroom_table(current_L)
    assert len(table) == 100
    assert table[16]["epochs"] == 0  # Exact square has 0 epochs
    assert table[19]["expected_score_gain"] > table[5]["expected_score_gain"]
    assert table[19]["epochs"] > table[5]["epochs"]
    assert table[19]["multi_seed_eligible"] is True
    assert table[5]["multi_seed_eligible"] is False

    ranked_pass1 = rank_puzzles_by_headroom(current_L, pass_number=1)
    ranked_pass2 = rank_puzzles_by_headroom(current_L, pass_number=2)
    assert len(ranked_pass1) == 90
    by_n_p2 = {item["n"]: item for item in ranked_pass2}
    assert by_n_p2[19]["skip_on_multi_seed"] is False
    assert by_n_p2[5]["skip_on_multi_seed"] is True
