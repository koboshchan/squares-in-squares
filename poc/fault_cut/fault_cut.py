"""Topological fault-line cut and cluster translation for unit-square packings.

Given an incumbent packing of size n + d, this module identifies a minimum-weight
topological cut of d squares stretching across the container (e.g. left-to-right
or bottom-to-top) in the Delaunay / Voronoi contact graph. Removing these d squares
partitions the remaining n squares into two disconnected rigid clusters, which are
then translated toward each other along the cut normal by up to Delta = 1.0 (or the
clearance limit before contact) to directly compress the container side length L.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from squares.core import Packing, check_n, evaluate, grid, read_submission
from squares._official_metric import compute_corners_float


def _load_packing(source: Any) -> tuple[int, Packing]:
    """Parse source into (n, Packing). Supports path, JSON, CSV, or dict."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if "records" in data and data["records"]:
                rec = data["records"][0]
                n = int(rec["n"])
                rows = [
                    dict(square_id=str(i), x=str(sq["x"]), y=str(sq["y"]), angle=str(sq["angle"]))
                    for i, sq in enumerate(rec["squares"])
                ]
                return n, rows
            if "squares" in data:
                rows = [
                    dict(square_id=str(i), x=str(sq["x"]), y=str(sq["y"]), angle=str(sq["angle"]))
                    for i, sq in enumerate(data["squares"])
                ]
                return len(rows), rows
            raise ValueError(f"Unrecognized JSON structure in {path}")
        puzzles = read_submission(path)
        if not puzzles:
            raise ValueError(f"No puzzles found in {path}")
        n = max(puzzles.keys())
        return n, puzzles[n]

    if isinstance(source, dict):
        if all(isinstance(k, int) for k in source.keys()):
            n = max(source.keys())
            return n, source[n]
        if "squares" in source:
            rows = [
                dict(square_id=str(i), x=str(sq["x"]), y=str(sq["y"]), angle=str(sq["angle"]))
                for i, sq in enumerate(source["squares"])
            ]
            return len(rows), rows

    if isinstance(source, list):
        return len(source), [dict(r) for r in source]

    raise TypeError(f"Unsupported source packing type: {type(source)}")


def _measure_side(coords: list[tuple[float, float, float]]) -> float:
    """Compute bounding side max(max_x - min_x, max_y - min_y) from (x, y, angle)."""
    if not coords:
        return 0.0
    all_x: list[float] = []
    all_y: list[float] = []
    for x, y, a in coords:
        for cx, cy in compute_corners_float(x, y, a):
            all_x.append(cx)
            all_y.append(cy)
    return max(max(all_x) - min(all_x), max(all_y) - min(all_y))


def _build_contact_graph(coords: list[tuple[float, float, float]]):
    """Construct the Delaunay / Voronoi contact graph of the squares."""
    import networkx as nx
    import numpy as np
    from scipy.spatial import Delaunay

    n_squares = len(coords)
    g = nx.Graph()
    for i in range(n_squares):
        g.add_node(i)

    points = np.array([[c[0], c[1]] for c in coords], dtype=float)

    if n_squares >= 3:
        tri = Delaunay(points)
        for simplex in tri.simplices:
            for u, v in ((simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])):
                if not g.has_edge(u, v):
                    dist = float(np.linalg.norm(points[u] - points[v]))
                    g.add_edge(u, v, weight=dist)
    elif n_squares == 2:
        dist = float(np.linalg.norm(points[0] - points[1]))
        g.add_edge(0, 1, weight=dist)

    # Supplement Delaunay with proximity contacts within unit square reach
    for i in range(n_squares):
        for j in range(i + 1, n_squares):
            dist = float(np.linalg.norm(points[i] - points[j]))
            if dist <= math.sqrt(2) + 0.05 and not g.has_edge(i, j):
                g.add_edge(i, j, weight=dist)

    return g, points


def _check_sat_overlap(
    sq1: tuple[float, float, float],
    sq2: tuple[float, float, float],
    eps: float = 1e-9,
) -> bool:
    """Return True if unit squares sq1 and sq2 overlap via the Separating Axis Theorem."""
    x1, y1, a1 = sq1
    x2, y2, a2 = sq2

    dx = x2 - x1
    dy = y2 - y1

    c1, s1 = math.cos(a1), math.sin(a1)
    c2, s2 = math.cos(a2), math.sin(a2)

    axes = [
        (c1, s1),
        (-s1, c1),
        (c2, s2),
        (-s2, c2),
    ]

    for ax, ay in axes:
        # Half extents projected onto unit axis
        r1 = 0.5 * (abs(ax * c1 + ay * s1) + abs(-ax * s1 + ay * c1))
        r2 = 0.5 * (abs(ax * c2 + ay * s2) + abs(-ax * s2 + ay * c2))
        dist_proj = abs(dx * ax + dy * ay)
        if dist_proj >= r1 + r2 - eps:
            return False  # Separating axis found
    return True


def _find_topological_cut(
    coords: list[tuple[float, float, float]],
    d: int,
) -> tuple[list[int], str]:
    """Find a minimum-weight connected cut of d squares stretching across boundary to boundary.

    Returns (cut_indices, cut_direction) where cut_direction is 'horizontal' or 'vertical'.
    """
    import networkx as nx
    import numpy as np

    total_m = len(coords)
    if d <= 0 or d >= total_m:
        raise ValueError(f"d must be between 1 and {total_m - 1}, got {d}")

    g, points = _build_contact_graph(coords)

    # Determine bounding box
    all_corners = [compute_corners_float(x, y, a) for x, y, a in coords]
    min_x = min(c[0] for corners in all_corners for c in corners)
    max_x = max(c[0] for corners in all_corners for c in corners)
    min_y = min(c[1] for corners in all_corners for c in corners)
    max_y = max(c[1] for corners in all_corners for c in corners)

    width = max_x - min_x
    height = max_y - min_y
    margin = min(1.0, 0.3 * max(width, height))

    # Identify boundary squares
    left_nodes = [i for i, c in enumerate(all_corners) if min(pt[0] for pt in c) <= min_x + margin]
    right_nodes = [i for i, c in enumerate(all_corners) if max(pt[0] for pt in c) >= max_x - margin]
    bottom_nodes = [i for i, c in enumerate(all_corners) if min(pt[1] for pt in c) <= min_y + margin]
    top_nodes = [i for i, c in enumerate(all_corners) if max(pt[1] for pt in c) >= max_y - margin]

    # Helper to evaluate path weight
    def path_weight(p: list[int]) -> float:
        w = 0.0
        for idx in range(len(p) - 1):
            w += g[p[idx]][p[idx + 1]].get("weight", 1.0)
        return w

    best_cut: list[int] | None = None
    best_weight = float("inf")
    best_dir = "horizontal"

    # Search for boundary-to-boundary paths of length d
    # Direction 1: Left-to-Right (Horizontal cut separating Top and Bottom clusters)
    for src in left_nodes:
        for dst in right_nodes:
            if d == 1:
                if src == dst:
                    return [src], "horizontal"
                continue
            if not nx.has_path(g, src, dst):
                continue
            try:
                for p in nx.all_simple_paths(g, src, dst, cutoff=d - 1):
                    if len(p) == d:
                        pw = path_weight(p)
                        if pw < best_weight:
                            best_weight = pw
                            best_cut = list(p)
                            best_dir = "horizontal"
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    # Direction 2: Bottom-to-Top (Vertical cut separating Left and Right clusters)
    for src in bottom_nodes:
        for dst in top_nodes:
            if d == 1:
                if src == dst and best_cut is None:
                    return [src], "vertical"
                continue
            if not nx.has_path(g, src, dst):
                continue
            try:
                for p in nx.all_simple_paths(g, src, dst, cutoff=d - 1):
                    if len(p) == d:
                        pw = path_weight(p)
                        if pw < best_weight:
                            best_weight = pw
                            best_cut = list(p)
                            best_dir = "vertical"
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    if best_cut is not None:
        return best_cut, best_dir

    # Fallback when no exact length-d boundary-to-boundary path exists:
    # 1. Articulation points / cut vertices if d == 1
    if d == 1:
        cut_nodes = list(nx.articulation_points(g))
        if cut_nodes:
            # Pick articulation point closest to container center
            cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
            best_node = min(cut_nodes, key=lambda i: (points[i][0] - cx) ** 2 + (points[i][1] - cy) ** 2)
            # Determine direction by variance of remaining coordinates
            rem = [points[i] for i in range(total_m) if i != best_node]
            var_x = float(np.var([p[0] for p in rem])) if rem else 1.0
            var_y = float(np.var([p[1] for p in rem])) if rem else 1.0
            return [best_node], "horizontal" if var_y >= var_x else "vertical"

    # 2. Minimum node cut separating opposite boundaries
    for src_set, dst_set, direction in (
        (left_nodes, right_nodes, "horizontal"),
        (bottom_nodes, top_nodes, "vertical"),
    ):
        for s in src_set:
            for t in dst_set:
                if s == t or not nx.has_path(g, s, t):
                    continue
                try:
                    node_cut = nx.minimum_node_cut(g, s, t)
                    if len(node_cut) <= d:
                        cut_list = list(node_cut)
                        # Pad with connected neighbors to reach d if necessary
                        while len(cut_list) < d:
                            cand = None
                            for node in cut_list:
                                for neighbor in g.neighbors(node):
                                    if neighbor not in cut_list:
                                        cand = neighbor
                                        break
                                if cand is not None:
                                    break
                            if cand is None:
                                # Pick any remaining node
                                cand = next(i for i in range(total_m) if i not in cut_list)
                            cut_list.append(cand)
                        return cut_list[:d], direction
                except nx.NetworkXError:
                    continue

    # 3. Shortest boundary-to-boundary path sliced/padded to d squares
    all_paths = []
    for s in left_nodes:
        for t in right_nodes:
            if nx.has_path(g, s, t):
                all_paths.append((nx.shortest_path(g, s, t, weight="weight"), "horizontal"))
    for s in bottom_nodes:
        for t in top_nodes:
            if nx.has_path(g, s, t):
                all_paths.append((nx.shortest_path(g, s, t, weight="weight"), "vertical"))

    if all_paths:
        p, direction = min(all_paths, key=lambda item: path_weight(item[0]))
        if len(p) >= d:
            # Subsample or take central d squares along the path
            start_idx = max(0, (len(p) - d) // 2)
            return p[start_idx : start_idx + d], direction
        # Otherwise pad with graph neighbors
        padded = list(p)
        while len(padded) < d:
            nbr = next((n for node in padded for n in g.neighbors(node) if n not in padded), None)
            if nbr is None:
                nbr = next(i for i in range(total_m) if i not in padded)
            padded.append(nbr)
        return padded[:d], direction

    # Default fallback: central squares
    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    sorted_by_dist = sorted(range(total_m), key=lambda i: (points[i][0] - cx) ** 2 + (points[i][1] - cy) ** 2)
    return sorted_by_dist[:d], "horizontal"


def _split_into_clusters(
    coords: list[tuple[float, float, float]],
    cut_indices: list[int],
    cut_direction: str,
) -> tuple[list[int], list[int]]:
    """Partition remaining squares into two disconnected rigid clusters."""
    total_m = len(coords)
    cut_set = set(cut_indices)
    remaining = [i for i in range(total_m) if i not in cut_set]

    if not remaining:
        return [], []

    # Fault-line reference coordinate
    if cut_direction == "horizontal":
        # Cut runs horizontally; partition by y into Top and Bottom clusters
        cut_y = [coords[i][1] for i in cut_indices]
        ref_y = sum(cut_y) / len(cut_y)
        c1 = [i for i in remaining if coords[i][1] >= ref_y]
        c2 = [i for i in remaining if coords[i][1] < ref_y]
    else:
        # Cut runs vertically; partition by x into Right and Left clusters
        cut_x = [coords[i][0] for i in cut_indices]
        ref_x = sum(cut_x) / len(cut_x)
        c1 = [i for i in remaining if coords[i][0] >= ref_x]
        c2 = [i for i in remaining if coords[i][0] < ref_x]

    # Ensure neither cluster is empty
    if not c1 or not c2:
        # Fall back to median split along the partition axis
        axis = 1 if cut_direction == "horizontal" else 0
        rem_sorted = sorted(remaining, key=lambda i: coords[i][axis])
        mid = len(rem_sorted) // 2
        c2 = rem_sorted[:mid]
        c1 = rem_sorted[mid:]

    return c1, c2


def _find_max_translation(
    coords: list[tuple[float, float, float]],
    c1: list[int],
    c2: list[int],
    normal: tuple[float, float],
    max_delta: float = 1.0,
) -> float:
    """Find the maximum collision-free translation distance Delta in [0, max_delta]."""
    nx_norm, ny_norm = normal

    def has_overlap(delta: float) -> bool:
        # Translate c1 by -delta * normal relative to c2
        for i in c1:
            x_i, y_i, a_i = coords[i]
            trans_sq1 = (x_i - delta * nx_norm, y_i - delta * ny_norm, a_i)
            for j in c2:
                if _check_sat_overlap(trans_sq1, coords[j]):
                    return True
        return False

    # Check if max_delta is free of collisions
    if not has_overlap(max_delta):
        return max_delta

    # Binary search for contact threshold
    low = 0.0
    high = max_delta
    for _ in range(30):
        mid = (low + high) / 2.0
        if has_overlap(mid):
            high = mid
        else:
            low = mid

    # Apply safety margin
    safe_delta = max(0.0, low - 1e-6)
    return safe_delta


def run_cut_eval(source_packing: Any, d: int = 1) -> dict[str, Any]:
    """Execute topological cut and cluster translation, certifying candidate packing."""
    m, rows = _load_packing(source_packing)
    if d >= m:
        raise ValueError(f"d ({d}) must be strictly less than packing size ({m})")
    n = m - d
    check_n(n)

    coords = [(float(r["x"]), float(r["y"]), float(r["angle"])) for r in rows]
    initial_l = _measure_side(coords)

    # 1. Topological cut
    cut_indices, cut_direction = _find_topological_cut(coords, d)

    # 2. Rigid cluster split
    c1, c2 = _split_into_clusters(coords, cut_indices, cut_direction)

    # 3. Normal vector of the separating cut
    # Horizontal cut separates y: normal is (0, 1), moving c1 downward towards c2
    # Vertical cut separates x: normal is (1, 0), moving c1 leftward towards c2
    normal = (0.0, 1.0) if cut_direction == "horizontal" else (1.0, 0.0)

    # 4. Translation clearance
    best_delta = _find_max_translation(coords, c1, c2, normal, max_delta=1.0)

    # Search along normal for optimal certified compression
    applied_delta = best_delta
    candidate_rows: Packing = []
    certified = False
    final_l = initial_l
    status_reason = "uncertified"

    # Try decreasing delta values if certification requires margin
    for factor in (1.0, 0.9999, 0.999, 0.99, 0.95, 0.9, 0.5, 0.0):
        current_delta = applied_delta * factor
        nx_norm, ny_norm = normal

        new_coords: list[tuple[float, float, float]] = []
        for i in sorted(c1 + c2):
            x, y, a = coords[i]
            if i in c1:
                new_coords.append((x - current_delta * nx_norm, y - current_delta * ny_norm, a))
            else:
                new_coords.append((x, y, a))

        prop_rows: Packing = [
            dict(
                square_id=str(new_id),
                x=format(pt[0], ".17g"),
                y=format(pt[1], ".17g"),
                angle=format(pt[2], ".17g"),
            )
            for new_id, pt in enumerate(new_coords)
        ]

        _, _, l_str, reason = evaluate(n, prop_rows)
        if reason == "ok" and l_str is not None:
            candidate_rows = prop_rows
            final_l = float(l_str)
            certified = True
            status_reason = reason
            applied_delta = current_delta
            break
        status_reason = reason

    # If non-translated remainder is certified when translated version fails
    if not certified:
        rem_coords = [coords[i] for i in sorted(c1 + c2)]
        fallback_rows = [
            dict(
                square_id=str(new_id),
                x=format(pt[0], ".17g"),
                y=format(pt[1], ".17g"),
                angle=format(pt[2], ".17g"),
            )
            for new_id, pt in enumerate(rem_coords)
        ]
        _, _, l_str, reason = evaluate(n, fallback_rows)
        if reason == "ok" and l_str is not None:
            candidate_rows = fallback_rows
            final_l = float(l_str)
            certified = True
            status_reason = reason
            applied_delta = 0.0

    delta_l = initial_l - final_l

    return {
        "n": n,
        "d": d,
        "m": m,
        "cut_direction": cut_direction,
        "cut_squares": cut_indices,
        "cluster1_size": len(c1),
        "cluster2_size": len(c2),
        "translation_delta": applied_delta,
        "initial_L": initial_l,
        "final_L": final_l,
        "delta_L": delta_l,
        "certified": certified,
        "reason": status_reason,
        "candidate_packing": candidate_rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Topological fault-line cut and cluster translation (n+d -> n)"
    )
    parser.add_argument("source", help="Source packing CSV, JSON, or grid size integer")
    parser.add_argument("-d", type=int, default=1, help="Number of cut squares to delete (default: 1)")
    parser.add_argument("-o", "--output", help="Optional path to write candidate packing CSV")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only output compact JSON summary")

    args = parser.parse_args(argv)

    # Allow integer n+d source to generate grid baseline on the fly
    source_arg: Any = args.source
    if str(args.source).isdigit():
        source_arg = grid(int(args.source))

    result = run_cut_eval(source_arg, d=args.d)

    if args.output and result["candidate_packing"]:
        from squares.core import write_submission
        write_submission(args.output, {result["n"]: result["candidate_packing"]}, validate=result["certified"])

    # Prepare summary for stdout
    summary = {k: v for k, v in result.items() if k != "candidate_packing"}
    print(json.dumps(summary, indent=2, default=lambda x: int(x) if hasattr(x, '__int__') else str(x)))


if __name__ == "__main__":
    main()
