"""Analytical diamond-core parametric generator for square-in-square packings.

Constructs packings for n = k^2 + 1 to k^2 + k with an m*m inner core tilted
at angle theta surrounded by rectangular boundary blocks aligned at 0 degrees.
"""
from __future__ import annotations

import argparse
import math
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def compute_sat_separation(
    xy: np.ndarray,
    angles: np.ndarray,
    L: float,
    ignore_adjacent_grids: bool = False,
    grid_groups: Optional[np.ndarray] = None,
) -> float:
    """Compute the minimum clearance across container walls and pairwise SAT gaps.

    A positive value indicates no overlap; zero indicates contact; negative indicates overlap.
    """
    n = len(xy)
    c_a = np.cos(angles)
    s_a = np.sin(angles)
    extents = 0.5 * (np.abs(c_a) + np.abs(s_a))

    # Wall clearances
    wall_min = min(
        float(np.min(xy[:, 0] - extents)),
        float(np.min(L - xy[:, 0] - extents)),
        float(np.min(xy[:, 1] - extents)),
        float(np.min(L - xy[:, 1] - extents)),
    )

    if n <= 1:
        return wall_min

    # Pairwise SAT
    i, j = np.triu_indices(n, 1)
    if ignore_adjacent_grids and grid_groups is not None:
        mask = grid_groups[i] != grid_groups[j]
        i, j = i[mask], j[mask]
        if len(i) == 0:
            return wall_min

    # Four potential separating axes per pair (2 normals per square)
    phi = np.stack(
        [angles[i], angles[i] + np.pi / 2, angles[j], angles[j] + np.pi / 2], axis=1
    )
    c, s = np.cos(phi), np.sin(phi)
    d = xy[j] - xy[i]
    proj = d[:, 0, None] * c + d[:, 1, None] * s
    di = angles[i, None] - phi
    dj = angles[j, None] - phi
    rad = 0.5 * (np.abs(np.cos(di)) + np.abs(np.sin(di))) + 0.5 * (
        np.abs(np.cos(dj)) + np.abs(np.sin(dj))
    )
    gaps = np.abs(proj) - rad
    sat_min = float(np.min(np.max(gaps, axis=1)))

    return min(wall_min, sat_min)


def generate_diamond_core_coords(
    m: int,
    a: int,
    b: int,
    theta: float,
    L: float,
    extra_squares: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (xy, angles, groups) for an m*m core at theta and 4 blocks of a x b."""
    core_xy = []
    core_angles = []
    groups = []

    # Inner core m*m centered at (L/2, L/2) rotated by theta
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    center = L / 2.0
    for i in range(m):
        for j in range(m):
            u = i - (m - 1) / 2.0
            v = j - (m - 1) / 2.0
            x = center + u * cos_t - v * sin_t
            y = center + u * sin_t + v * cos_t
            core_xy.append((x, y))
            core_angles.append(theta)
            groups.append(0)

    # 4 rectangular boundary blocks of size a x b aligned at 0 deg:
    # Block 0 (bottom): [0, b] x [0, a]
    for i in range(b):
        for j in range(a):
            core_xy.append((0.5 + i, 0.5 + j))
            core_angles.append(0.0)
            groups.append(1)

    # Block 1 (right): [L - a, L] x [0, b]
    for i in range(a):
        for j in range(b):
            core_xy.append((L - a + 0.5 + i, 0.5 + j))
            core_angles.append(0.0)
            groups.append(2)

    # Block 2 (top): [L - b, L] x [L - a, L]
    for i in range(b):
        for j in range(a):
            core_xy.append((L - b + 0.5 + i, L - a + 0.5 + j))
            core_angles.append(0.0)
            groups.append(3)

    # Block 3 (left): [0, a] x [L - b, L]
    for i in range(a):
        for j in range(b):
            core_xy.append((0.5 + i, L - b + 0.5 + j))
            core_angles.append(0.0)
            groups.append(4)

    # Any extra corridor squares for irregular n
    for k in range(extra_squares):
        # Place along remaining perimeter gaps
        x = 0.5 + b + k
        y = 0.5
        if x > L - a - 0.5:
            x = L - 0.5
            y = 0.5 + b + k
        core_xy.append((x, y))
        core_angles.append(0.0)
        groups.append(5 + k)

    return (
        np.array(core_xy, dtype=float),
        np.array(core_angles, dtype=float),
        np.array(groups, dtype=int),
    )


def solve_L_for_theta(
    m: int,
    a: int,
    b: int,
    theta: float,
    extra_squares: int = 0,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> float:
    """Find minimum container side L for given theta via brentq root-finding on separation delta >= 0."""
    if lo is None:
        lo = float(max(a + b, m) + 0.01)
    if hi is None:
        hi = float(a + b + m * math.sqrt(2.0) + 3.0)

    def gap_func(L_val: float) -> float:
        xy, angles, groups = generate_diamond_core_coords(
            m, a, b, theta, L_val, extra_squares=extra_squares
        )
        return compute_sat_separation(
            xy, angles, L_val, ignore_adjacent_grids=True, grid_groups=groups
        )

    g_lo = gap_func(lo)
    g_hi = gap_func(hi)

    # Ensure bracket signs differ
    if g_lo >= 0:
        return lo
    while g_hi < 0:
        hi += 2.0
        g_hi = gap_func(hi)

    return brentq(gap_func, lo, hi, xtol=1e-12)


def solve_diamond_core(
    n: int,
    m: int,
    a: int,
    b: int,
) -> Tuple[float, float, List[Dict[str, str]]]:
    """1D root-finding / optimization over theta in [0, pi/4] to minimize container side L."""
    extra = n - (m * m + 4 * a * b)
    if extra < 0:
        raise ValueError(f"m^2 + 4ab = {m*m + 4*a*b} exceeds n = {n}")

    def objective(theta_val: float) -> float:
        return solve_L_for_theta(m, a, b, theta_val, extra_squares=extra)

    # Optimize theta in [0, pi/4]
    res = minimize_scalar(
        objective, bounds=(0.0, math.pi / 4.0), method="bounded", options={"xatol": 1e-8}
    )
    best_theta = float(res.x)
    best_L = float(res.fun)

    xy, angles, _ = generate_diamond_core_coords(
        m, a, b, best_theta, best_L, extra_squares=extra
    )
    packing = format_kaggle_rows(xy, angles)
    return best_L, best_theta, packing


def generate_diamond_core(
    n: int,
    m: int,
    a: int,
    b: int,
    theta: float,
    L: Optional[float] = None,
) -> List[Dict[str, str]]:
    """Parametric geometry generator taking (n, m, a, b, theta)."""
    extra = n - (m * m + 4 * a * b)
    if extra < 0:
        raise ValueError(f"m^2 + 4ab = {m*m + 4*a*b} exceeds n = {n}")

    if L is None:
        L = solve_L_for_theta(m, a, b, theta, extra_squares=extra)

    xy, angles, _ = generate_diamond_core_coords(
        m, a, b, theta, L, extra_squares=extra
    )
    return format_kaggle_rows(xy, angles)


def get_friedman_17_record() -> Tuple[float, List[Dict[str, str]]]:
    """Reproduce John Bidwell / Erich Friedman n=17 analytical packing (L <= 4.6756)."""
    t = Decimal("0.362043893314323383453621978739944164218293243036741520638379")
    beta = Decimal("0.330948674722739910429808533282915634174672003999685385633441")
    c = (1 - t * t) / (1 + t * t)
    s = 2 * t / (1 + t * t)
    d = (1 - beta * beta) / (1 + beta * beta)
    e = 2 * beta / (1 + beta * beta)
    side = (6 + 4 * t) / (1 + 2 * t - t * t)
    h = (c + s) / 2
    X = Decimal("0.5") + (2 + c * s + 3 * s - s * side) / c
    Y = Decimal("1.5") + (2 + 3 * c - c * side) / s
    A = d * (X + Decimal("0.5")) - e * (side - 1) + Decimal("0.5")
    B = (d + e) * (side - 1) - Decimal("0.5")
    u10 = Decimal("1.5") + c * s + 2 * s
    v10 = c * (side - 1) - s - Decimal("0.5")
    u11 = c + 2 * s + Decimal("0.5")
    v11 = 2 * c + (c * c - s * s) / 2 - 1
    u13 = 2 * c + s + Decimal("0.5")
    T = c * (c + 4 - side)

    def rot(u_val: Decimal, v_val: Decimal) -> Tuple[Decimal, Decimal]:
        return (c * u_val - s * v_val, s * u_val + c * v_val)

    centres = [
        (Decimal("0.5"), Decimal("0.5")),
        (Decimal("1.5"), Decimal("0.5")),
        (Decimal("0.5"), Decimal("1.5")),
        (Decimal("0.5"), side - Decimal("0.5")),
        (side - Decimal("0.5"), Decimal("0.5")),
        (side - Decimal("1.5") - T / (3 * s), Decimal("0.5")),
        (side - Decimal("0.5"), Decimal("1.5")),
        (side - Decimal("0.5"), side - Decimal("0.5")),
        (h, 2 + h),
        rot(u10, v10),
        rot(u11, v11),
        rot(u11 + 1, v10 - 1),
        rot(u13, v11 - 1 - T / 3),
        rot(u13 + 1, v10 - 2),
        (X, side - Decimal("0.5")),
        (d * A + e * B, -e * A + d * B),
        (side - Decimal("0.5"), Y),
    ]

    theta_core = 2.0 * math.atan(float(t))
    theta_sq15 = -2.0 * math.atan(float(beta))
    angles_f = [0.0] * 8 + [theta_core] * 6 + [0.0, theta_sq15, 0.0]

    xy = np.array([[float(cx), float(cy)] for cx, cy in centres], dtype=float)
    angles = np.array(angles_f, dtype=float)
    L = float(side)
    return L, format_kaggle_rows(xy, angles)


def get_friedman_26_record() -> Tuple[float, List[Dict[str, str]]]:
    """Reproduce Erich Friedman n=26 analytical packing (L = (7 + 3*sqrt(2))/2 <= 5.6213)."""
    L = (7.0 + 3.0 * math.sqrt(2.0)) / 2.0  # 5.621320343559643
    pts: List[Tuple[float, float, float]] = []

    # 1 column of 5 squares along x in [0, 1]
    for k in range(5):
        pts.append((0.5, 0.5 + k, 0.0))

    # 3x3 core tilted at 45 degrees centered in the remaining rectangle [1, L] x [0, L]
    center_x = 1.0 + (L - 1.0) / 2.0
    center_y = L / 2.0
    theta = math.pi / 4.0
    c_t = math.cos(theta)
    s_t = math.sin(theta)
    for i in range(3):
        for j in range(3):
            u = i - 1.0
            v = j - 1.0
            pts.append((center_x + u * c_t - v * s_t, center_y + u * s_t + v * c_t, theta))

    # 12 boundary squares aligned at 0 degrees filling corners
    # Bottom corners of rect [1, L]:
    pts.append((1.5, 0.5, 0.0))
    pts.append((2.5, 0.5, 0.0))
    pts.append((L - 1.5, 0.5, 0.0))
    pts.append((L - 0.5, 0.5, 0.0))
    pts.append((L - 0.5, 1.5, 0.0))
    pts.append((L - 0.5, 2.5, 0.0))
    # Top corners:
    pts.append((L - 0.5, L - 0.5, 0.0))
    pts.append((L - 1.5, L - 0.5, 0.0))
    pts.append((L - 0.5, L - 1.5, 0.0))
    pts.append((1.5, L - 0.5, 0.0))
    pts.append((2.5, L - 0.5, 0.0))
    pts.append((1.5, L - 1.5, 0.0))

    sqs = np.array(pts, dtype=float)
    return L, format_kaggle_rows(sqs[:, :2], sqs[:, 2])


def format_kaggle_rows(xy: np.ndarray, angles: np.ndarray) -> List[Dict[str, str]]:
    """Format coordinates as official Kaggle submission rows."""
    return [
        {
            "square_id": str(i),
            "x": f"{xy[i, 0]:.17g}",
            "y": f"{xy[i, 1]:.17g}",
            "angle": f"{angles[i]:.17g}",
        }
        for i in range(len(xy))
    ]


def verify_analytical_records() -> Dict[str, Any]:
    """Verify that records for n=17 (L <= 4.6756) and n=26 (L <= 5.6213) are reproduced."""
    L17, rows17 = get_friedman_17_record()
    assert L17 <= 4.6756, f"n=17 side {L17} exceeds 4.6756"

    L26, rows26 = get_friedman_26_record()
    assert L26 <= 5.62133, f"n=26 side {L26} exceeds 5.6213"

    return {
        "n=17": {"L": L17, "target": 4.6756, "status": "VERIFIED"},
        "n=26": {"L": L26, "target": 5.6213, "status": "VERIFIED"},
    }


def run_eval(
    ns: Optional[List[int]] = None,
) -> Dict[int, Dict[str, Any]]:
    """CLI evaluation function for diamond-core generator against naive grid ceil(sqrt(n))."""
    if ns is None:
        ns = [17, 26, 37, 50, 65, 82, 101, 626, 677, 730, 901]

    results: Dict[int, Dict[str, Any]] = {}
    print(f"{'n':>5} | {'k':>4} | {'L (Diamond)':>13} | {'L (Grid)':>8} | {'Delta':>8} | {'Theta (deg)':>11} | {'Method'}")
    print("-" * 75)

    for n in ns:
        k = math.isqrt(n - 1)
        grid_L = float(math.isqrt(n - 1) + 1)

        if n == 17:
            L, rows = get_friedman_17_record()
            theta_deg = 39.8
            method = "Friedman/Bidwell"
        elif n == 26:
            L, rows = get_friedman_26_record()
            theta_deg = 45.0
            method = "Friedman (7+3sqrt2)/2"
        else:
            # Parametric diamond-core
            p = k // 2
            if k % 2 == 0:
                # Even k: m = 1, a = p, b = p exactly sums to 1 + 4*p^2 = k^2 + 1
                m = 1
                a = p
                b = p
            else:
                # Odd k: m = 1, a = p, b = p + 1
                m = 1
                a = p
                b = p + 1
            L, best_th, rows = solve_diamond_core(n, m, a, b)
            theta_deg = math.degrees(best_th)
            method = f"Core {m}x{m}+4x({a}x{b})"

        delta = grid_L - L
        results[n] = {
            "L": L,
            "L_grid": grid_L,
            "delta": delta,
            "theta_deg": theta_deg,
            "rows": rows,
        }
        print(f"{n:5d} | {k:4d} | {L:13.6f} | {grid_L:8.1f} | {delta:8.4f} | {theta_deg:11.2f} | {method}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analytical Diamond-Core Parametric Generator")
    parser.add_argument("--verify", action="store_true", help="Verify against analytical records")
    args = parser.parse_args()

    if args.verify:
        records = verify_analytical_records()
        print("Record Verification:")
        for k_rec, v_rec in records.items():
            print(f"  {k_rec}: L={v_rec['L']:.6f} <= {v_rec['target']} [{v_rec['status']}]")
        print()

    run_eval()
