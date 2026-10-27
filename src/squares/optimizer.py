"""Small-n annealed restarts of a rotating-square penalty minimizer.

This is a heuristic, not a proof or a record claim. Floating-point search is
only a proposal mechanism; the unchanged official certifier is the write gate.
"""
from __future__ import annotations

import math

from .core import Packing, certify, check_n, evaluate, grid


def tilted_five() -> Packing:
    """Classic four corners plus a central diamond, with a safety dilation.

    n=5 construction credited to Frits Goebel (1979), via Friedman's survey.
    """
    import mpmath as mp

    with mp.workdps(60):
        side = 2 + 1 / mp.sqrt(2)
        factor = mp.mpf("1.000000001")
        centers = [
            (mp.mpf(".5"), mp.mpf(".5"), 0),
            (side - mp.mpf(".5"), mp.mpf(".5"), 0),
            (mp.mpf(".5"), side - mp.mpf(".5"), 0),
            (side - mp.mpf(".5"), side - mp.mpf(".5"), 0),
            (side / 2, side / 2, mp.pi / 4),
        ]
        rows = [dict(
            square_id=str(i), x=mp.nstr(x * factor, 55),
            y=mp.nstr(y * factor, 55), angle=mp.nstr(a, 55),
        ) for i, (x, y, a) in enumerate(centers)]
    certify(5, rows)
    return rows


def objective(z, n: int, side: float) -> float:
    import numpy as np

    q = np.asarray(z).reshape(n, 3)
    xy, angle = q[:, :2], q[:, 2]
    c, s = np.cos(angle), np.sin(angle)
    u, v = np.stack([c, s], axis=1), np.stack([-s, c], axis=1)
    i, j = np.triu_indices(n, 1)
    delta = xy[j] - xy[i]
    axes = np.stack([u[i], v[i], u[j], v[j]], axis=1)
    ri = .5 * (abs(np.einsum("pkd,pd->pk", axes, u[i]))
               + abs(np.einsum("pkd,pd->pk", axes, v[i])))
    rj = .5 * (abs(np.einsum("pkd,pd->pk", axes, u[j]))
               + abs(np.einsum("pkd,pd->pk", axes, v[j])))
    depth = np.maximum(0, np.min(
        ri + rj - abs(np.einsum("pkd,pd->pk", axes, delta)), axis=1,
    ))
    extent = (abs(c) + abs(s)) / 2
    walls = np.maximum(0, abs(xy - side / 2) + extent[:, None] - side / 2)
    return float(np.sum(depth ** 2) + np.sum(walls ** 2))


def optimize_packing(n: int, initial: Packing | None = None, *,
                     attempts: int = 20, seed: int = 42,
                     maxiter: int = 250) -> tuple[Packing, dict]:
    import numpy as np
    from scipy.optimize import minimize

    check_n(n)
    if n > 50:
        raise ValueError("the quadratic heuristic is limited to n <= 50")
    if attempts < 0 or maxiter < 1:
        raise ValueError("attempts must be nonnegative and maxiter positive")
    rows = [dict(r) for r in (initial if initial is not None else grid(n))]
    side = float(certify(n, rows))
    start_side = side
    improvements = 0
    if n == 5:
        tilted = tilted_five()
        candidate_side = float(certify(n, tilted))
        if candidate_side < side:
            rows, side = tilted, candidate_side
            improvements += 1
    rng = np.random.default_rng(seed)
    completed = 0
    converged = 0
    for attempt in range(attempts):
        if side <= math.sqrt(n) + 1e-12:
            break
        q = np.array([[float(r[k]) for k in ("x", "y", "angle")] for r in rows])
        q[:, :2] -= q[:, :2].min(axis=0) - .5
        temperature = .15 * (1 - attempt / max(attempts, 1)) + .005
        if attempt:
            q += rng.normal(0, temperature, q.shape)
        target = side - (.01 if attempt % 5 == 0 else .002)
        q[:, :2] *= target / side
        result = minimize(
            objective, q.ravel(), args=(n, target), method="L-BFGS-B",
            bounds=[(-1, 33), (-1, 33), (-math.pi, math.pi)] * n,
            options=dict(maxiter=maxiter, ftol=1e-14, gtol=1e-8, maxls=30),
        )
        completed += 1
        converged += bool(result.success)
        # Even an unconverged numerical proposal is safe to consider because
        # no numerical status ever bypasses the exact acceptance gate.
        if not math.isfinite(float(result.fun)) or result.fun > 1e-14:
            continue
        q = result.x.reshape(n, 3).copy()
        q[:, :2] *= 1.00000001
        candidate = [dict(
            square_id=str(i), x=format(x, ".17g"),
            y=format(y, ".17g"), angle=format(a, ".17g"),
        ) for i, (x, y, a) in enumerate(q)]
        _, _, text, reason = evaluate(n, candidate)
        if reason == "ok" and text is not None and float(text) < side:
            rows, side = candidate, float(text)
            improvements += 1
    certify(n, rows)
    return rows, dict(n=n, seed=seed, attempts=completed,
                      converged=converged, initial_L=start_side,
                      final_L=side, improvements=improvements)
