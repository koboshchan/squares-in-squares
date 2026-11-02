"""Gap-targeted search budget and per-n headroom estimation for n=1..1000."""
from __future__ import annotations
import math
from record_report import TARGETS as PUBLISHED_TARGETS

# Analytical fast lower-envelope prediction from Göbel staircases + L-borders
_ANALYTIC_CACHE: dict[int, float] | None = None


def _build_analytic_envelope(max_n: int = 1000) -> dict[int, float]:
    global _ANALYTIC_CACHE
    if _ANALYTIC_CACHE is not None and len(_ANALYTIC_CACHE) >= max_n:
        return _ANALYTIC_CACHE
    s2 = math.sqrt(2.0)
    inv_s2 = 1.0 / s2
    eps = 1e-7
    env = {n: float(math.isqrt(n - 1) + 1) if n > 1 else 1.0 for n in range(1, max_n + 1)}
    for r in range(1, int(math.isqrt(max_n)) + 1):
        env[r * r] = float(r)
    for n, val in PUBLISHED_TARGETS.items():
        if n <= max_n:
            env[n] = min(env[n], float(val))

    for w in range(1, 26):
        for a1 in range(1, 48):
            for a2 in range(1, a1 + 1):
                if abs(a1 - a2) > 4:
                    continue
                L = 0.5 * (a1 + a2) + 1.0 + w * inv_s2 + 60 * eps
                if L > 33.0:
                    continue
                lim = int(math.floor(L - 1.0 - 20 * eps))
                if lim < 0 or a1 > lim + 1 or a2 > lim + 1:
                    continue
                c_a1 = a1 * (a1 + 1) // 2
                c_a2 = a2 * (a2 + 1) // 2
                max_b = max(0, int(math.floor(L - max(a1, a2) - 20 * eps)))
                u_start = (a1 + 1.0) * inv_s2 + 0.5 + 20 * eps
                for b1 in range(0, max_b + 1):
                    c_b1 = b1 * (b1 + 1) // 2
                    for b2 in range(0, b1 + 1):
                        c_b2 = b2 * (b2 + 1) // 2
                        corner_cnt = c_a1 + c_a2 + c_b1 + c_b2
                        if corner_cnt > max_n:
                            continue
                        til_cnt = 0
                        for k in range(w):
                            u = u_start + k * (1.0 + 2 * eps)
                            v_lo = -min(u - 1.0 - 10 * eps, s2 * L - u - 1.0 - 10 * eps)
                            v_hi = min(u - 1.0 - 10 * eps, s2 * L - u - 1.0 - 10 * eps)
                            if b1 > 0:
                                v_hi = min(v_hi, (L - b1 - 1.0) * inv_s2 - 0.5 - 20 * eps)
                            if b2 > 0:
                                v_lo = max(v_lo, -(L - b2 - 1.0) * inv_s2 + 0.5 + 20 * eps)
                            if v_hi >= v_lo:
                                til_cnt += int(math.floor((v_hi - v_lo) / (1.0 + 2 * eps))) + 1
                        cap = corner_cnt + til_cnt
                        for n in range(max(2, cap - 6), min(max_n, cap) + 1):
                            if L >= math.isqrt(n) + 0.45 and L < env[n]:
                                env[n] = L
    # L-bordered core propagation
    for m in range(5, max_n + 1):
        Lm = env[m]
        k0 = int(math.floor(Lm - 1e-6))
        if k0 < 2:
            continue
        for t in range(1, 32 - k0):
            k = k0 + t
            cap = m + k * k - k0 * k0
            L_new = Lm + t + 3e-6
            for n in range(max(2, cap - 4), min(max_n, cap) + 1):
                if L_new >= math.isqrt(n) + 0.45 and L_new < env[n]:
                    env[n] = L_new
    _ANALYTIC_CACHE = env
    return env


def geometric_target_L(n: int) -> float:
    """Smooth empirical lower-envelope estimate of achievable packing side L*(n)."""
    if n in PUBLISHED_TARGETS:
        return float(PUBLISHED_TARGETS[n])
    k = math.isqrt(n)
    if k * k == n:
        return float(k)
    if k == 1:
        return 2.0
    if k == 2:
        return 2.7071067811865475 if n == 5 else 3.0
    if k == 3:
        if n == 10:
            return 3.7071067811865475
        if n == 11:
            return 3.877084
        return 4.0
    rem = n - k * k
    frac = max(0.0, (rem - 1.0) / float(2 * k))
    # Empirical excess curve for k >= 4: k^2+1 starts around k + 0.485, rising toward k + 1.0
    excess = min(1.0, 0.485 + 0.515 * (frac ** 0.65))
    return max(math.sqrt(n), float(k) + excess)


def compute_headroom_table(
    current_L_by_n: dict[int, float],
    family_predictions: dict[int, float] | None = None,
) -> dict[int, dict]:
    """Compute per-n target L, expected score gain headroom, and proportional GPU budget."""
    max_n = max(current_L_by_n.keys(), default=1000)
    analytic_env = _build_analytic_envelope(max_n)

    # Monotone suffix minimum of current_L_by_n: since L*(n) <= L*(m) for all n < m
    mono_min = {}
    running_min = float("inf")
    for n in range(max_n, 0, -1):
        if n in current_L_by_n:
            running_min = min(running_min, float(current_L_by_n[n]))
        mono_min[n] = running_min

    table: dict[int, dict] = {}
    raw_headrooms = []
    for n, cur_L in sorted(current_L_by_n.items()):
        cur_L = float(cur_L)
        k = math.isqrt(n)
        if k * k == n:
            table[n] = dict(
                n=n,
                current_L=cur_L,
                target_L=float(k),
                lower_bound_L=float(k),
                headroom_L=0.0,
                expected_score_gain=0.0,
                epochs=0,
                steps=0,
                multi_seed_eligible=False,
            )
            continue

        lb = math.sqrt(n)
        candidates = [
            cur_L,
            mono_min.get(n, cur_L),
            analytic_env.get(n, cur_L),
            geometric_target_L(n),
        ]
        if n in PUBLISHED_TARGETS:
            candidates.append(float(PUBLISHED_TARGETS[n]))
        if family_predictions and n in family_predictions:
            candidates.append(float(family_predictions[n]))

        target_L = max(lb, min(candidates))
        realistic_gap_L = max(0.0, cur_L - target_L)
        lb_gap_L = max(0.0, cur_L - lb)
        # Expected score gain Delta L / sqrt(n), primarily driven by realistic headroom
        expected_gain = (realistic_gap_L + 0.01 * lb_gap_L) / math.sqrt(n)
        raw_headrooms.append(expected_gain)
        table[n] = dict(
            n=n,
            current_L=cur_L,
            target_L=target_L,
            lower_bound_L=lb,
            headroom_L=realistic_gap_L,
            expected_score_gain=expected_gain,
        )

    mean_gain = (sum(raw_headrooms) / len(raw_headrooms)) if raw_headrooms else 1e-4
    mean_gain = max(mean_gain, 1e-6)
    sorted_gains = sorted([e["expected_score_gain"] for e in table.values() if e.get("headroom_L", 0.0) > 1e-3], reverse=True)
    # Top 35% of non-square puzzles with meaningful headroom qualify for multi-seed passes
    multi_seed_cutoff = sorted_gains[min(len(sorted_gains) - 1, max(0, int(0.35 * len(sorted_gains))))] if sorted_gains else 0.0

    for n, entry in table.items():
        if entry.get("epochs") == 0:
            continue
        gain = entry["expected_score_gain"]
        ratio = gain / mean_gain
        base_epochs = 50 if n <= 64 else (40 if n <= 256 else 30)
        base_steps = 64 if n <= 64 else (48 if n <= 256 else 32)
        scale_e = max(0.25, min(2.75, ratio))
        scale_s = max(0.50, min(1.40, math.sqrt(max(0.1, ratio))))
        epochs = max(8, min(110, int(round(base_epochs * scale_e))))
        steps = max(16, min(96, int(round(base_steps * scale_s))))
        entry["budget_ratio"] = ratio
        entry["epochs"] = epochs
        entry["steps"] = steps
        entry["multi_seed_eligible"] = bool(entry["headroom_L"] > 1e-3 and gain >= max(multi_seed_cutoff, 2e-4))

    return table


def rank_puzzles_by_headroom(
    current_L_by_n: dict[int, float],
    *,
    family_predictions: dict[int, float] | None = None,
    pass_number: int = 1,
) -> list[dict]:
    """Return non-square puzzles ordered by descending expected score gain."""
    table = compute_headroom_table(current_L_by_n, family_predictions=family_predictions)
    items = [v for n, v in table.items() if math.isqrt(n) ** 2 != n]
    for item in items:
        # On multi-seed passes (pass_number > 1), puzzles below the multi-seed threshold get a fast probe or skip
        item["skip_on_multi_seed"] = bool(pass_number > 1 and not item["multi_seed_eligible"])
    return sorted(items, key=lambda x: (-x["expected_score_gain"], x["n"]))
