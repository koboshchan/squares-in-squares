"""Regression tests for the multithreaded C++ certifier vs official metric."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "cuda"))
from scheduler import goebel, strip_construction
from squares.core import certify, evaluate, grid, kaggle_roundtrip, prepare_kaggle, write_submission


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


def test_cpp_certifier_matches_50_digit_official_metric(cpp_certifier, tmp_path):
    puzzles = {
        1: grid(1),
        2: grid(2),
        4: grid(4),
        5: goebel(1),
        10: goebel(2),
        17: strip_construction(4),
        27: goebel(4),
    }
    csv_path = tmp_path / "sub.csv"
    details_path = tmp_path / "details.json"
    write_submission(csv_path, puzzles, validate=True)

    res = subprocess.run(
        [str(cpp_certifier), str(csv_path), "--max-n", "27", "--details", str(details_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(res.stdout)
    assert summary["accepted"] == len(puzzles)
    assert summary["rejected"] == 0

    details = json.loads(details_path.read_text())["results"]
    by_n = {row["n"]: row for row in details}
    for n, rows in puzzles.items():
        official_L = certify(n, rows)
        assert by_n[n]["reason"] == "ok"
        assert by_n[n]["L"] == official_L


def test_cpp_certifier_catches_roundtrip_and_micro_overlaps(cpp_certifier, tmp_path):
    # goebel(2) has a 1e-40 dilation that vanishes under float64 roundtrip
    raw_puzzles = {10: goebel(2)}
    safe_puzzles = prepare_kaggle(raw_puzzles)

    raw_csv = tmp_path / "raw.csv"
    safe_csv = tmp_path / "safe.csv"
    write_submission(raw_csv, raw_puzzles, validate=True)
    write_submission(safe_csv, safe_puzzles, validate=True)

    # Literal check passes on raw_csv
    lit_out = json.loads(
        subprocess.run(
            [str(cpp_certifier), str(raw_csv), "--max-n", "10"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    assert lit_out["accepted"] == 1 and lit_out["rejected"] == 0

    # Roundtrip check rejects raw_csv (matching kaggle_roundtrip + evaluate)
    rt_raw = kaggle_roundtrip(raw_puzzles)
    assert evaluate(10, rt_raw[10])[3] != "ok"
    rt_out = json.loads(
        subprocess.run(
            [str(cpp_certifier), str(raw_csv), "--max-n", "10", "--roundtrip"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    assert rt_out["accepted"] == 0 and rt_out["rejected"] == 1

    # Roundtrip check accepts prepare_kaggle safe_csv
    rt_safe_out = json.loads(
        subprocess.run(
            [str(cpp_certifier), str(safe_csv), "--max-n", "10", "--roundtrip"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    assert rt_safe_out["accepted"] == 1 and rt_safe_out["rejected"] == 0
