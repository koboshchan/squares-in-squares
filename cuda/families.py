"""Parametric family enumeration, continuous parameter optimization, and official certification across n=1..1000."""
from __future__ import annotations
import argparse
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import subprocess
import time
from scheduler import atomic_json
from squares.core import certify, grid, kaggle_roundtrip, read_submission, score, write_submission

FAMILY_NAMES = (
    "gobel_staircase",
    "grid_tilted_bands",
    "corner_rect_corridor",
    "staggered_interior",
    "record_neighbors",
)


def ensure_families_cli() -> Path:
    root = Path(__file__).resolve().parent
    src = root / "families_cli.cpp"
    hdr = root / "families.hpp"
    exe = root / ("families_cli.exe" if os.name == "nt" else "families_cli")
    need_build = not exe.exists()
    if not need_build:
        exe_mtime = exe.stat().st_mtime
        need_build = src.stat().st_mtime > exe_mtime or hdr.stat().st_mtime > exe_mtime
    if need_build:
        cxx = os.environ.get("CXX", "c++")
        subprocess.run(
            [cxx, "-O3", "-std=c++17", "-pthread", str(src), "-o", str(exe)],
            check=True,
        )
    return exe


def ensure_certifier_cli() -> Path:
    root = Path(__file__).resolve().parent
    src = root / "certifier.cpp"
    exe = root / ("certifier.exe" if os.name == "nt" else "certifier")
    if not exe.exists() or src.stat().st_mtime > exe.stat().st_mtime:
        cxx = os.environ.get("CXX", "c++")
        subprocess.run(
            [cxx, "-O3", "-std=c++17", "-pthread", str(src), "-o", str(exe)],
            check=True,
        )
    return exe


def enumerate_families(
    incumbent_csv: Path,
    output_dir: Path,
    *,
    max_n: int = 1000,
    polish_iters: int = 60,
    threads: int | None = None,
    update_best_dir: Path | None = None,
) -> dict:
    """Enumerate all 5 parametric families across non-square n in 1..max_n and certify winners."""
    if not 1 <= max_n <= 1000:
        raise ValueError("max_n must be in 1..1000")
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cli = ensure_families_cli()
    cert_exe = ensure_certifier_cli()

    raw_json = output_dir / "families-raw.json"
    cmd = [
        str(cli),
        str(incumbent_csv),
        str(raw_json),
        "--max-n",
        str(max_n),
        "--polish-iters",
        str(polish_iters),
    ]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    subprocess.run(cmd, check=True)
    raw = json.loads(raw_json.read_text())

    incumbent = read_submission(incumbent_csv)
    subset_inc = {n: incumbent[n] for n in range(1, max_n + 1) if n in incumbent}
    for n in range(1, max_n + 1):
        if n not in subset_inc:
            subset_inc[n] = grid(n)
    rt_inc = kaggle_roundtrip(subset_inc)

    wins_by_family = {name: [] for name in FAMILY_NAMES}
    standalone_vs_inc = {name: [] for name in FAMILY_NAMES}
    standalone_vs_grid = {name: [] for name in FAMILY_NAMES}
    family_predictions = {}
    certified_improvements = {}

    for item in raw["puzzles"]:
        n = int(item["n"])
        if n > max_n:
            continue
        inc_L_float = float(item["incumbent_L"])
        grid_L_float = float(math.isqrt(n - 1) + 1)
        fam_L = item["family_L"]
        best_pred = inc_L_float
        for name in FAMILY_NAMES:
            val = fam_L.get(name)
            if val is not None:
                val = float(val)
                if val < best_pred:
                    best_pred = val
                if val < inc_L_float - 1e-6:
                    standalone_vs_inc[name].append(n)
                if val < grid_L_float - 1e-6:
                    standalone_vs_grid[name].append(n)
        family_predictions[str(n)] = dict(
            best_family_L=best_pred,
            winning_family=item["winning_family"],
            family_L=fam_L,
        )

        wf = item["winning_family"]
        if wf != "incumbent" and "q" in item:
            cand_rows = [
                dict(
                    square_id=str(i),
                    x=format(float(x), ".17g"),
                    y=format(float(y), ".17g"),
                    angle=format(float(a), ".17g"),
                )
                for i, (x, y, a) in enumerate(item["q"])
            ]
            try:
                lit_L = certify(n, cand_rows)
                rt_L = certify(n, kaggle_roundtrip({n: cand_rows})[n])
                old_lit_L = certify(n, subset_inc[n])
                old_rt_L = certify(n, rt_inc[n])
                if Decimal(lit_L) < Decimal(old_lit_L) and Decimal(rt_L) < Decimal(old_rt_L):
                    wins_by_family[wf].append(n)
                    subset_inc[n] = cand_rows
                    certified_improvements[str(n)] = dict(
                        n=n,
                        family=wf,
                        previous_L=old_lit_L,
                        L=lit_L,
                        roundtrip_L=rt_L,
                        score_gain=float((Decimal(old_lit_L) - Decimal(lit_L)) / Decimal(n).sqrt()),
                    )
                    if update_best_dir is not None:
                        best_file = Path(update_best_dir) / "best" / f"{n}.json"
                        if best_file.exists():
                            cur_best = json.loads(best_file.read_text())
                            cur_L = Decimal(cur_best["L"])
                            cur_rt_L = Decimal(
                                cur_best["roundtrip_L"]
                                if "roundtrip_L" in cur_best
                                else certify(n, kaggle_roundtrip({n: cur_best["rows"]})[n])
                            )
                            if Decimal(lit_L) < cur_L and Decimal(rt_L) < cur_rt_L:
                                atomic_json(
                                    best_file,
                                    dict(
                                        rows=cand_rows,
                                        L=lit_L,
                                        roundtrip_L=rt_L,
                                        source=f"family:{wf}",
                                        previous_L=cur_best["L"],
                                    ),
                                )
                        else:
                            atomic_json(
                                best_file,
                                dict(
                                    rows=cand_rows,
                                    L=lit_L,
                                    roundtrip_L=rt_L,
                                    source=f"family:{wf}",
                                    previous_L=old_lit_L,
                                ),
                            )
            except ValueError:
                pass

    merged_csv = output_dir / "families-certified.csv"
    write_submission(merged_csv, subset_inc, validate=False)
    for rt_flag in (False, True):
        cmd = [str(cert_exe), str(merged_csv), "--max-n", str(max_n), "--threads", "4"]
        if rt_flag:
            cmd.append("--roundtrip")
        res = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
        if res["rejected"] != 0 or res["missing"] != 0 or res["accepted"] != max_n:
            raise ValueError(f"C++ certifier failed on merged family CSV: {res}")

    total_gain = sum(v["score_gain"] for v in certified_improvements.values())
    report = dict(
        max_n=max_n,
        seconds=time.time() - t0,
        cpp_seconds=raw["seconds"],
        total_improved=len(certified_improvements),
        total_score_gain=total_gain,
        win_counts_by_family={k: len(v) for k, v in wins_by_family.items()},
        standalone_wins_vs_incumbent={k: len(v) for k, v in standalone_vs_inc.items()},
        standalone_wins_vs_grid={k: len(v) for k, v in standalone_vs_grid.items()},
        wins_by_family=wins_by_family,
        standalone_n_vs_incumbent=standalone_vs_inc,
        standalone_n_vs_grid=standalone_vs_grid,
        improvements=certified_improvements,
        predictions=family_predictions,
        certified_csv=str(merged_csv),
    )
    atomic_json(output_dir / "family-report.json", report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--incumbent", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/families"))
    p.add_argument("--max-n", type=int, default=1000)
    p.add_argument("--polish-iters", type=int, default=60)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--update-best-dir", type=Path, default=None)
    args = p.parse_args()
    rep = enumerate_families(
        args.incumbent,
        args.output,
        max_n=args.max_n,
        polish_iters=args.polish_iters,
        threads=args.threads,
        update_best_dir=args.update_best_dir,
    )
    summary = {
        k: rep[k]
        for k in (
            "max_n",
            "seconds",
            "total_improved",
            "total_score_gain",
            "win_counts_by_family",
            "standalone_wins_vs_incumbent",
            "standalone_wins_vs_grid",
            "certified_csv",
        )
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
