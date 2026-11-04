"""Discrete 1D shelf-knapsack dynamic programming solver for unit square packing.

Discretizes a container [0, L] x [0, L] into horizontal shelves of variable
heights chosen from candidate angles Theta = [0, pi/4, atan(0.5), asin(1/L)],
computes analytical square capacity along container width L, solves the 1D
bounded knapsack dynamic program to minimize L for target n, and reconstructs
exact coordinates formatted for official metric certification.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
from typing import Callable

# Ensure src/ is on sys.path so squares.core can be imported if available
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from squares.core import certify, evaluate, Packing
except ImportError:
    certify = None  # type: ignore[assignment]
    evaluate = None  # type: ignore[assignment]
    Packing = list[dict[str, str]]  # type: ignore[misc,assignment]


@dataclass
class Shelf:
    """A horizontal shelf configuration spanning width L."""
    angle: float
    shelf_type: str  # "collinear" or "chain"
    height: float
    capacity: int
    placer: Callable[[float, float], list[tuple[float, float, float]]] = field(repr=False)


def candidate_angles(L: float) -> list[float]:
    """Return candidate angles Theta = [0, pi/4, atan(0.5), asin(1/L)]."""
    angles = [0.0, math.pi / 4.0, math.atan(0.5)]
    if L >= 1.0:
        asin_val = math.asin(min(1.0, 1.0 / L))
        if asin_val not in angles:
            angles.append(asin_val)
    return sorted(angles)


def generate_candidate_shelves(L: float, gap: float = 1e-6) -> list[Shelf]:
    """Generate candidate shelf types for container width L.

    Supports both:
    1. Collinear horizontal shelves of height h(theta) = cos(theta) + sin(theta)
       where square centers share y and have horizontal pitch dx >= 1 / cos(theta).
    2. Rotated tilted chains where squares advance along (cos(theta), sin(theta)),
       yielding horizontal capacity floor((L - margin) / cos(theta)).
    """
    shelves: list[Shelf] = []
    angles = candidate_angles(L)

    for theta in angles:
        c = math.cos(theta)
        s = math.sin(theta)
        # Bounding box half-spans of a single square rotated by theta
        r_x = 0.5 * (abs(c) + abs(s))
        r_y = 0.5 * (abs(c) + abs(s))
        h_single = 2.0 * r_y

        # --- 1. Collinear horizontal shelf ---
        # Minimum horizontal separation between centers at same y is 1 / cos(theta) for theta in [0, pi/4]
        dx_collinear = (1.0 / max(c, 1e-9)) * (1.0 + gap)
        if L >= 2.0 * r_x + gap:
            cap_collinear = 1 + int((L - 2.0 * r_x - gap) / dx_collinear)
        else:
            cap_collinear = 0

        if cap_collinear > 0:
            def make_collinear_placer(th: float, rx: float, ry: float, dx: float, cap: int):
                def place(y_base: float, width: float) -> list[tuple[float, float, float]]:
                    y_center = y_base + ry + 0.5 * gap
                    span = (cap - 1) * dx + 2.0 * rx
                    x_start = max(rx + 0.5 * gap, (width - span) * 0.5 + rx)
                    return [(x_start + i * dx, y_center, th) for i in range(cap)]
                return place

            shelves.append(Shelf(
                angle=theta,
                shelf_type="collinear",
                height=h_single + gap,
                capacity=cap_collinear,
                placer=make_collinear_placer(theta, r_x, r_y, dx_collinear, cap_collinear),
            ))

        # --- 2. Rotated tilted chain shelf ---
        # Consecutive squares step along (cos(theta), sin(theta)) touching edge-to-edge
        if theta > 1e-8:
            pitch = 1.0 + gap
            step_x = pitch * c
            step_y = pitch * s
            # Width for k squares: (k - 1) * step_x + 2 * r_x <= L
            # Capacity formula: floor((L - margin) / cos(theta))
            margin_x = 2.0 * r_x + gap
            if L >= margin_x:
                max_chain_cap = 1 + int((L - margin_x) / step_x)
            else:
                max_chain_cap = 0

            # Generate chain options up to max capacity
            for k in range(2, max_chain_cap + 1):
                h_chain = (k - 1) * step_y + 2.0 * r_y + gap
                if h_chain <= L:
                    def make_chain_placer(th: float, rx: float, ry: float, sx: float, sy: float, count: int):
                        def place(y_base: float, width: float) -> list[tuple[float, float, float]]:
                            span_x = (count - 1) * sx + 2.0 * rx
                            x_start = max(rx + 0.5 * gap, (width - span_x) * 0.5 + rx)
                            y_start = y_base + ry + 0.5 * gap
                            return [(x_start + i * sx, y_start + i * sy, th) for i in range(count)]
                        return place

                    shelves.append(Shelf(
                        angle=theta,
                        shelf_type="chain",
                        height=h_chain,
                        capacity=k,
                        placer=make_chain_placer(theta, r_x, r_y, step_x, step_y, k),
                    ))

    return shelves


def solve_shelf_knapsack(L: float, target_n: int, gap: float = 1e-6) -> tuple[bool, list[Shelf]]:
    """Solve 1D unbounded shelf knapsack DP to pack >= target_n squares in height <= L.

    Returns:
        (feasible, chosen_shelves) where sum(s.height) <= L and sum(s.capacity) >= target_n.
    """
    shelves = generate_candidate_shelves(L, gap)
    if not shelves:
        return False, []

    # Filter out zero-capacity shelves
    valid_shelves = [s for s in shelves if s.capacity > 0 and s.height <= L]
    if not valid_shelves:
        return False, []

    # dp[count] = min total height needed to achieve count squares
    # count ranges from 0 to target_n
    inf = float("inf")
    dp = [inf] * (target_n + 1)
    parent: list[tuple[int, Shelf | None]] = [(0, None)] * (target_n + 1)
    dp[0] = 0.0

    # Solve unbounded knapsack
    for shelf in valid_shelves:
        c = shelf.capacity
        h = shelf.height
        for count in range(target_n + 1):
            if dp[count] == inf:
                continue
            nxt = min(target_n, count + c)
            new_h = dp[count] + h
            if new_h < dp[nxt]:
                dp[nxt] = new_h
                parent[nxt] = (count, shelf)

    if dp[target_n] > L:
        return False, []

    # Backtrack to reconstruct chosen shelves
    chosen: list[Shelf] = []
    curr = target_n
    while curr > 0:
        prev, sh = parent[curr]
        if sh is None:
            break
        chosen.append(sh)
        curr = prev

    chosen.reverse()
    return True, chosen


def reconstruct_packing(
    chosen_shelves: list[Shelf],
    L: float,
    target_n: int,
) -> list[tuple[float, float, float]]:
    """Reconstruct exact coordinates (x, y, angle) for placed squares across shelves."""
    placed: list[tuple[float, float, float]] = []
    y_base = 0.0

    for shelf in chosen_shelves:
        shelf_squares = shelf.placer(y_base, L)
        for sq in shelf_squares:
            placed.append(sq)
            if len(placed) >= target_n:
                return placed
        y_base += shelf.height

    return placed[:target_n]


def format_for_certification(squares: list[tuple[float, float, float]]) -> list[dict[str, str]]:
    """Format placed coordinates into decimal string rows for official metric certification."""
    rows: list[dict[str, str]] = []
    for i, (x, y, ang) in enumerate(squares):
        rows.append({
            "square_id": str(i),
            "x": f"{x:.12f}",
            "y": f"{y:.12f}",
            "angle": f"{ang:.12f}",
        })
    return rows


def pack_target(
    n: int,
    tol: float = 1e-4,
    max_iter: int = 40,
    gap: float = 1e-6,
) -> tuple[float, list[dict[str, str]], list[Shelf]]:
    """Find minimal container size L via binary search on shelf DP, returning certified rows."""
    low = math.sqrt(n)
    high = float(math.ceil(math.sqrt(n)))
    best_L = high
    best_shelves: list[Shelf] = []

    # Ensure baseline high (grid ceiling) is feasible
    ok, shelves = solve_shelf_knapsack(high, n, gap)
    if ok:
        best_shelves = shelves
    else:
        # Fallback trivial grid shelf of height 1.0
        grid_dim = math.ceil(math.sqrt(n))
        h_unit = 1.0 + gap
        best_shelves = [
            Shelf(
                angle=0.0,
                shelf_type="collinear",
                height=h_unit,
                capacity=grid_dim,
                placer=lambda y_b, w, c=grid_dim: [(0.5 + i * h_unit, y_b + 0.5, 0.0) for i in range(c)],
            )
            for _ in range(grid_dim)
        ]

    # Binary search for minimal feasible L
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        ok, shelves = solve_shelf_knapsack(mid, n, gap)
        if ok:
            best_L = mid
            best_shelves = shelves
            high = mid
        else:
            low = mid
        if (high - low) < tol:
            break

    coords = reconstruct_packing(best_shelves, best_L, n)
    rows = format_for_certification(coords)
    return best_L, rows, best_shelves


def evaluate_shelf_solver(
    targets: list[int] | None = None,
    certify_metric: bool = True,
    gap: float = 1e-6,
) -> list[dict]:
    """Evaluate shelf-knapsack solver across target n values."""
    if targets is None:
        targets = [17, 26, 37, 50, 65, 101]

    results: list[dict] = []

    for n in targets:
        L_grid = float(math.ceil(math.sqrt(n)))
        score_grid = L_grid / math.sqrt(n) - 1.0

        solver_L, rows, shelves = pack_target(n, gap=gap)
        score_solver = solver_L / math.sqrt(n) - 1.0

        certified_L: str | None = None
        cert_reason = "skipped"

        if certify_metric and certify is not None and evaluate is not None:
            try:
                _, _, c_side, reason = evaluate(n, rows)
                cert_reason = reason
                if c_side is not None and reason == "ok":
                    certified_L = c_side
            except Exception as e:
                cert_reason = str(e)

        results.append({
            "n": n,
            "L_grid": L_grid,
            "score_grid": score_grid,
            "L_solver": solver_L,
            "score_solver": score_solver,
            "certified_L": certified_L,
            "cert_reason": cert_reason,
            "shelves_count": len(shelves),
            "rows": rows,
        })

    return results


def print_evaluation_summary(results: list[dict]) -> None:
    """Print formatted summary table for CLI evaluation."""
    header = f"{'n':>4} | {'Grid L':>7} | {'Solver L':>9} | {'Grid Score':>10} | {'Solver Score':>12} | {'Certified L':>15} | {'Status':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        cert_str = r["certified_L"][:15] if r["certified_L"] else "-"
        print(
            f"{r['n']:>4} | {r['L_grid']:>7.3f} | {r['L_solver']:>9.4f} | "
            f"{r['score_grid']:>10.6f} | {r['score_solver']:>12.6f} | "
            f"{cert_str:>15} | {r['cert_reason']:>8}"
        )
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discrete Shelf-Knapsack Square Packing DP Solver")
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[17, 26, 37, 50, 65, 101],
        help="Target n values to evaluate",
    )
    parser.add_argument(
        "--no-certify",
        action="store_true",
        help="Skip official interval certification check",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=1e-6,
        help="Clearance gap between adjacent squares and shelves",
    )
    args = parser.parse_args()

    results = evaluate_shelf_solver(
        targets=args.targets,
        certify_metric=not args.no_certify,
        gap=args.gap,
    )
    print_evaluation_summary(results)


if __name__ == "__main__":
    main()
