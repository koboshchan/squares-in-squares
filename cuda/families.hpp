// Self-contained C++17 parametric family enumerator, continuous parameter optimizer,
// and deterministic Cusp-Aware PRP+ CG polisher for n in [1, 1000].
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#if !defined(__CUDACC__) && !defined(__host__) && !defined(__device__)
#define __host__
#define __device__
#endif

namespace sqfam {

struct Sq {
    double x, y, a;
};

struct Grad {
    double x, y, a;
};

struct WallForce {
    double fx;
    double fy;
    double torque;
    double e;
};

enum FamilyId {
    FAM_GOBEL_STAIRCASE = 0,
    FAM_GRID_TILTED_BANDS = 1,
    FAM_CORNER_RECT_CORRIDOR = 2,
    FAM_STAGGERED_INTERIOR = 3,
    FAM_RECORD_NEIGHBORS = 4,
    FAM_COUNT = 5
};

inline const char* family_name(int id) {
    switch (id) {
        case FAM_GOBEL_STAIRCASE: return "gobel_staircase";
        case FAM_GRID_TILTED_BANDS: return "grid_tilted_bands";
        case FAM_CORNER_RECT_CORRIDOR: return "corner_rect_corridor";
        case FAM_STAGGERED_INTERIOR: return "staggered_interior";
        case FAM_RECORD_NEIGHBORS: return "record_neighbors";
        default: return "unknown";
    }
}

constexpr double PI2 = 1.57079632679489661923;

__host__ __device__ inline double wrap_pi2(double a) {
    constexpr double pi2 = 1.57079632679489661923;
    a = fmod(a, pi2);
    if (a < 0.0) a += pi2;
    return a;
}

__host__ __device__ inline double wrap_angle(double a) {
    return wrap_pi2(a);
}

__host__ __device__ inline double sg_smooth_cpu(double x, double eps = 1e-6) {
    if (fabs(x) >= eps) return (x > 0.0) - (x < 0.0);
    double u = x / eps;
    return u * (1.5 - 0.5 * u * u);
}

__host__ __device__ inline double dr_smooth_cpu(double c, double s) {
    return 0.5 * (-sg_smooth_cpu(c) * s + sg_smooth_cpu(s) * c);
}

__host__ __device__ inline double sg_smooth(double x, double eps = 1e-6) {
    return sg_smooth_cpu(x, eps);
}

__host__ __device__ inline double dr_smooth(double c, double s) {
    return dr_smooth_cpu(c, s);
}

inline double pair_energy_cpu(const Sq& p, const Sq& q, double c, double s, double d, double t) {
    double dx = q.x - p.x, dy = q.y - p.y;
    if (dx * dx + dy * dy >= 2.000001) return 0.0;
    double cd = c * d + s * t, sd = s * d - c * t;
    double radius = 0.5 + 0.5 * (fabs(cd) + fabs(sd));
    double ux[4] = {c, -s, d, -t}, uy[4] = {s, c, t, d};
    double gap = -1e100;
    for (int k = 0; k < 4; ++k) {
        double v = dx * ux[k] + dy * uy[k];
        double z = fabs(v) - radius;
        if (z > gap) gap = z;
    }
    return (gap >= 0.0) ? 0.0 : gap * gap;
}

inline double pair_smooth_grad_cpu(
    const Sq& p, const Sq& q, double c, double s, double d, double t,
    Grad& gp, Grad& gq
) {
    double dx = q.x - p.x, dy = q.y - p.y;
    if (dx * dx + dy * dy >= 2.000001) return 0.0;
    double cd = c * d + s * t, sd = s * d - c * t;
    double radius = 0.5 + 0.5 * (fabs(cd) + fabs(sd));
    double deriv = 0.5 * (-sg_smooth_cpu(cd) * sd + sg_smooth_cpu(sd) * cd);
    double ux[4] = {c, -s, d, -t}, uy[4] = {s, c, t, d};
    double gap = -1e100, gx = 0.0, gy = 0.0, gpa = 0.0, gqa = 0.0;
    for (int k = 0; k < 4; ++k) {
        double v = dx * ux[k] + dy * uy[k];
        double z = fabs(v) - radius;
        if (z > gap) {
            gap = z;
            double sv = (v >= 0.0) ? 1.0 : -1.0;
            gx = -sv * ux[k];
            gy = -sv * uy[k];
            double cross = sv * (-dx * uy[k] + dy * ux[k]);
            gpa = -deriv + (k < 2 ? cross : 0.0);
            gqa =  deriv + (k >= 2 ? cross : 0.0);
        }
    }
    if (gap >= 0.0) return 0.0;
    double f = 2.0 * gap;
    gp.x += f * gx; gp.y += f * gy; gp.a += f * gpa;
    gq.x -= f * gx; gq.y -= f * gy; gq.a -= f * gqa;
    return gap * gap;
}

__host__ __device__ inline WallForce compute_wall_4vertex(double x, double y, double a, double s) {
    WallForce wf = {0.0, 0.0, 0.0, 0.0};
    double ca = cos(a);
    double sa = sin(a);
    const double local_offsets[4][2] = {
        {-0.5, -0.5},
        { 0.5, -0.5},
        { 0.5,  0.5},
        {-0.5,  0.5}
    };
    for (int k = 0; k < 4; ++k) {
        double lx = local_offsets[k][0];
        double ly = local_offsets[k][1];
        double rx = lx * ca - ly * sa;
        double ry = lx * sa + ly * ca;
        double vx = x + rx;
        double vy = y + ry;

        double fvx = 0.0;
        double fvy = 0.0;

        if (vx < 0.0) {
            double d = -vx;
            wf.e += d * d;
            fvx += d;
        } else if (vx > s) {
            double d = vx - s;
            wf.e += d * d;
            fvx -= d;
        }

        if (vy < 0.0) {
            double d = -vy;
            wf.e += d * d;
            fvy += d;
        } else if (vy > s) {
            double d = vy - s;
            wf.e += d * d;
            fvy -= d;
        }

        wf.fx += fvx;
        wf.fy += fvy;
        wf.torque += rx * fvy - ry * fvx;
    }
    return wf;
}

__host__ __device__ inline void sat_contact_force_torque(
    double px, double py, double pa,
    double qx, double qy, double qa,
    double* fx, double* fy, double* torque
) {
    if (fx) *fx = 0.0;
    if (fy) *fy = 0.0;
    if (torque) *torque = 0.0;

    double dx = qx - px;
    double dy = qy - py;
    if (dx * dx + dy * dy >= 2.000001) return;

    double c = cos(pa), s = sin(pa);
    double d = cos(qa), t = sin(qa);
    double cd = c * d + s * t;
    double sd = s * d - c * t;
    double radius = 0.5 + 0.5 * (fabs(cd) + fabs(sd));
    double deriv = 0.5 * (-sg_smooth_cpu(cd) * sd + sg_smooth_cpu(sd) * cd);
    double ux[4] = {c, -s, d, -t};
    double uy[4] = {s, c, t, d};

    double gap = -1e100;
    double gx = 0.0, gy = 0.0, gpa = 0.0;

    for (int k = 0; k < 4; ++k) {
        double v = dx * ux[k] + dy * uy[k];
        double z = fabs(v) - radius;
        if (z > gap) {
            gap = z;
            double sv = (v >= 0.0) ? 1.0 : -1.0;
            gx = -sv * ux[k];
            gy = -sv * uy[k];
            double cross = sv * (-dx * uy[k] + dy * ux[k]);
            gpa = -deriv + (k < 2 ? cross : 0.0);
        }
    }

    if (gap >= 0.0) return;

    double pen = -gap;
    if (fx) *fx = pen * gx;
    if (fy) *fy = pen * gy;
    if (torque) *torque = pen * gpa;
}

inline double certified_side_cpu(
    const std::vector<Sq>& q,
    std::vector<Sq>* best_out = nullptr,
    double current_best_L = 1e9,
    double tau = 2.6e-8
) {
    int n = (int)q.size();
    if (n == 0) return 0.0;
    double c[1024], s[1024];
    int head[784], next_idx[1024], cell_id[1024];
    std::fill(head, head + 784, -1);
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(q[i].x) || !std::isfinite(q[i].y) || !std::isfinite(q[i].a)) return 1e9;
        c[i] = std::cos(q[i].a);
        s[i] = std::sin(q[i].a);
        int cx = std::clamp((int)std::floor((q[i].x + 2.0) * (1.0 / 1.5)), 0, 27);
        int cy = std::clamp((int)std::floor((q[i].y + 2.0) * (1.0 / 1.5)), 0, 27);
        int cid = cy * 28 + cx;
        cell_id[i] = cid;
        next_idx[i] = head[cid];
        head[cid] = i;
    }
    double lam = 1.0;
    for (int i = 0; i < n; ++i) {
        double xi = q[i].x, yi = q[i].y, ci = c[i], si = s[i];
        int cx = cell_id[i] % 28, cy = cell_id[i] / 28;
        for (int dy_c = -1; dy_c <= 1; ++dy_c) {
            int ny = cy + dy_c;
            if (ny < 0 || ny >= 28) continue;
            for (int dx_c = -1; dx_c <= 1; ++dx_c) {
                int nx = cx + dx_c;
                if (nx < 0 || nx >= 28) continue;
                for (int j = head[ny * 28 + nx]; j != -1; j = next_idx[j]) {
                    if (j <= i) continue;
                    double dx = xi - q[j].x, dy = yi - q[j].y;
                    if (dx * dx + dy * dy >= 2.05) continue;
                    double cj = c[j], sj = s[j];
                    double r = 0.5 + 0.5 * (fabs(ci * cj + si * sj) + fabs(-si * cj + ci * sj));
                    double v1 = fabs(dx * ci + dy * si);
                    double v2 = fabs(-dx * si + dy * ci);
                    double v3 = fabs(dx * cj + dy * sj);
                    double v4 = fabs(-dx * sj + dy * cj);
                    double vmax = std::max(std::max(v1, v2), std::max(v3, v4));
                    double req = (vmax > 1e-12) ? ((r + tau) / vmax) : 1e9;
                    if (req > lam) lam = req;
                }
            }
        }
    }
    if (lam > 1.015) return 1e9;
    double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
    for (int i = 0; i < n; ++i) {
        double x = q[i].x * lam, y = q[i].y * lam;
        double r = 0.5 * (fabs(c[i]) + fabs(s[i]));
        if (x - r < min_x) min_x = x - r;
        if (x + r > max_x) max_x = x + r;
        if (y - r < min_y) min_y = y - r;
        if (y + r > max_y) max_y = y + r;
    }
    double L_cert = std::max(max_x - min_x, max_y - min_y);
    if (L_cert < current_best_L && best_out != nullptr) {
        best_out->resize(n);
        for (int i = 0; i < n; ++i) {
            (*best_out)[i] = {q[i].x * lam - min_x, q[i].y * lam - min_y, wrap_pi2(q[i].a)};
        }
    }
    return L_cert;
}

inline double polish_and_certify_cpu(
    const std::vector<Sq>& init_q,
    double target_L,
    std::vector<Sq>& best_out,
    double& best_L,
    int max_iters = 120
) {
    int n = (int)init_q.size();
    if (n <= 0) return 1e9;
    std::vector<Sq> q = init_q;
    double c0 = certified_side_cpu(q, &best_out, best_L);
    if (c0 < best_L) best_L = c0;
    if (max_iters <= 0 || n <= 1) return c0;

    std::vector<double> c(n), s(n);
    std::vector<Grad> g(n), gold(n, {0.0, 0.0, 0.0}), d(n, {0.0, 0.0, 0.0});
    std::vector<Sq> saved(n);
    int head[784], next_idx[1024], cell_id[1024];

    auto eval_total = [&](const std::vector<Sq>& pts, double L, bool with_grad) -> double {
        std::fill(head, head + 784, -1);
        for (int i = 0; i < n; ++i) {
            c[i] = std::cos(pts[i].a);
            s[i] = std::sin(pts[i].a);
            if (with_grad) g[i] = {0.0, 0.0, 0.0};
            int cx = std::clamp((int)std::floor((pts[i].x + 2.0) * (1.0 / 1.5)), 0, 27);
            int cy = std::clamp((int)std::floor((pts[i].y + 2.0) * (1.0 / 1.5)), 0, 27);
            int cid = cy * 28 + cx;
            cell_id[i] = cid;
            next_idx[i] = head[cid];
            head[cid] = i;
        }
        double e = 0.0;
        for (int i = 0; i < n; ++i) {
            double r_ext = 0.5 * (fabs(c[i]) + fabs(s[i]));
            double d_r = with_grad ? dr_smooth_cpu(c[i], s[i]) : 0.0;
            if (pts[i].x < r_ext) {
                double u = r_ext - pts[i].x;
                e += u * u;
                if (with_grad) { g[i].x -= 2.0 * u; g[i].a += 2.0 * u * d_r; }
            } else if (pts[i].x > L - r_ext) {
                double u = pts[i].x + r_ext - L;
                e += u * u;
                if (with_grad) { g[i].x += 2.0 * u; g[i].a += 2.0 * u * d_r; }
            }
            if (pts[i].y < r_ext) {
                double u = r_ext - pts[i].y;
                e += u * u;
                if (with_grad) { g[i].y -= 2.0 * u; g[i].a += 2.0 * u * d_r; }
            } else if (pts[i].y > L - r_ext) {
                double u = pts[i].y + r_ext - L;
                e += u * u;
                if (with_grad) { g[i].y += 2.0 * u; g[i].a += 2.0 * u * d_r; }
            }
            int cx = cell_id[i] % 28, cy = cell_id[i] / 28;
            for (int dy_c = -1; dy_c <= 1; ++dy_c) {
                int ny = cy + dy_c;
                if (ny < 0 || ny >= 28) continue;
                for (int dx_c = -1; dx_c <= 1; ++dx_c) {
                    int nx = cx + dx_c;
                    if (nx < 0 || nx >= 28) continue;
                    for (int j = head[ny * 28 + nx]; j != -1; j = next_idx[j]) {
                        if (j <= i) continue;
                        if (with_grad) {
                            e += pair_smooth_grad_cpu(pts[i], pts[j], c[i], s[i], c[j], s[j], g[i], g[j]);
                        } else {
                            e += pair_energy_cpu(pts[i], pts[j], c[i], s[i], c[j], s[j]);
                        }
                    }
                }
            }
        }
        return e;
    };

    double step_cg = 0.04;
    double energy = eval_total(q, target_L, false);
    for (int iter = 0; iter < max_iters; ++iter) {
        if (energy < 1e-22) break;
        energy = eval_total(q, target_L, true);
        double gg = 0.0, dgg = 0.0;
        for (int i = 0; i < n; ++i) {
            gg  += gold[i].x * gold[i].x + gold[i].y * gold[i].y + gold[i].a * gold[i].a;
            dgg += (g[i].x - gold[i].x) * g[i].x + (g[i].y - gold[i].y) * g[i].y + (g[i].a - gold[i].a) * g[i].a;
        }
        double beta = (iter == 0 || gg < 1e-30 || iter % 25 == 0) ? 0.0 : std::max(0.0, dgg / gg);
        double norm2 = 0.0;
        for (int i = 0; i < n; ++i) {
            d[i].x = -g[i].x + beta * d[i].x;
            d[i].y = -g[i].y + beta * d[i].y;
            d[i].a = -g[i].a + beta * d[i].a;
            gold[i] = g[i];
            saved[i] = q[i];
            norm2 += d[i].x * d[i].x + d[i].y * d[i].y + d[i].a * d[i].a;
        }
        if (norm2 < 1e-28) break;
        double alpha = step_cg;
        bool ls_ok = false;
        for (int ls = 0; ls < 5; ++ls) {
            for (int i = 0; i < n; ++i) {
                q[i] = {
                    saved[i].x + alpha * d[i].x,
                    saved[i].y + alpha * d[i].y,
                    wrap_pi2(saved[i].a + alpha * d[i].a)
                };
            }
            double ne = eval_total(q, target_L, false);
            if (ne < energy) {
                energy = ne;
                step_cg = std::min(0.15, alpha * 1.25);
                ls_ok = true;
                break;
            }
            alpha *= 0.45;
        }
        if (!ls_ok) {
            q = saved;
            for (int i = 0; i < n; ++i) d[i] = {0.0, 0.0, 0.0};
            step_cg = std::max(1e-7, step_cg * 0.5);
        }
    }
    double c1 = certified_side_cpu(q, &best_out, best_L);
    if (c1 < best_L) best_L = c1;
    return std::min(c0, c1);
}

// Multi-step contraction + PRP+ CG polish on CPU
inline double contract_and_polish_cpu(
    const std::vector<Sq>& init_q,
    std::vector<Sq>& best_out,
    double& best_L,
    int stages = 4,
    int iters_per_stage = 80
) {
    int n = (int)init_q.size();
    if (n <= 0) return 1e9;
    std::vector<Sq> cur = init_q;
    double cur_L = certified_side_cpu(cur, &best_out, best_L);
    if (cur_L < best_L) best_L = cur_L;
    if (cur_L >= 1e8) {
        // Estimate bounding box for uncertified initial point
        double max_xy = std::sqrt((double)n) + 0.5;
        for (const auto& p : cur) max_xy = std::max({max_xy, p.x + 0.5, p.y + 0.5});
        cur_L = polish_and_certify_cpu(cur, max_xy - 0.02, best_out, best_L, iters_per_stage);
        if (cur_L < best_L) cur = best_out;
    } else {
        cur = best_out;
    }
    double lb = std::sqrt((double)n) + 1e-5;
    double shrink_step = 0.004;
    for (int st = 0; st < stages; ++st) {
        if (best_L >= 1e8) break;
        double target = std::max(lb, best_L - shrink_step);
        double sc = target / best_L;
        std::vector<Sq> trial = best_out;
        for (auto& p : trial) { p.x *= sc; p.y *= sc; }
        double prev_best = best_L;
        polish_and_certify_cpu(trial, target, best_out, best_L, iters_per_stage);
        if (best_L < prev_best - 1e-7) {
            shrink_step *= 1.2;
        } else {
            shrink_step *= 0.4;
        }
    }
    return best_L;
}

// Helper to materialize a 4-corner (staircase or beveled-rectangle) + w-band layout
inline std::vector<Sq> materialize_4corner_bands(
    int n, double L, int w, int a1, int a2, int b1, int b2,
    int cx1, int cy1, int cx2, int cy2, bool per_band_stagger, bool corners_first = true
) {
    constexpr double pi2 = 1.57079632679489661923;
    const double s2 = std::sqrt(2.0), inv_s2 = 1.0 / s2, eps = 1e-7, pi4 = 0.5 * pi2;
    int lim = (int)std::floor(L - 1.0 - 20 * eps);
    std::vector<Sq> corners, tilted;
    for (int i = 0; i < std::min(lim + 1, a1); ++i)
        for (int j = 0; j < std::min(lim + 1, a1 - i); ++j)
            corners.push_back({0.5 + i, 0.5 + j, 0.0});
    for (int i = 0; i < std::min(lim + 1, a2); ++i)
        for (int j = 0; j < std::min(lim + 1, a2 - i); ++j)
            corners.push_back({L - 0.5 - i, L - 0.5 - j, 0.0});
    for (int i = 0; i < std::min(cx1, b1); ++i)
        for (int j = 0; j < std::min(cy1, b1 - i); ++j)
            corners.push_back({0.5 + i, L - 0.5 - j, 0.0});
    for (int i = 0; i < std::min(cx2, b2); ++i)
        for (int j = 0; j < std::min(cy2, b2 - i); ++j)
            corners.push_back({L - 0.5 - i, 0.5 + j, 0.0});

    double u_start = (a1 + 1.0) * inv_s2 + 0.5 + 20 * eps;
    for (int k = 0; k < w; ++k) {
        double u = u_start + k * (1.0 + 2 * eps);
        double v_lo = -std::min(u - 1.0 - 10 * eps, s2 * L - u - 1.0 - 10 * eps);
        double v_hi =  std::min(u - 1.0 - 10 * eps, s2 * L - u - 1.0 - 10 * eps);
        if (b1 > 0) {
            if (!per_band_stagger) {
                v_hi = std::min(v_hi, (L - b1 - 1.0) * inv_s2 - 0.5 - 20 * eps);
            } else {
                for (int i = 0; i < std::min(cx1, b1); ++i) {
                    int j_max = std::min(cy1, b1 - i) - 1;
                    if (j_max < 0) continue;
                    int j_hi = std::min(j_max, (int)std::floor(L + i + 1.0 - s2 * (u - 0.5 - 15 * eps)));
                    int j_lo = std::max(0, (int)std::ceil(L + i - 1.0 - s2 * (u + 0.5 + 15 * eps)));
                    if (j_hi >= j_lo) {
                        int eff_b1 = i + j_hi + 1;
                        v_hi = std::min(v_hi, (L - eff_b1 - 1.0) * inv_s2 - 0.5 - 20 * eps);
                    }
                }
            }
        }
        if (b2 > 0) {
            if (!per_band_stagger) {
                v_lo = std::max(v_lo, -(L - b2 - 1.0) * inv_s2 + 0.5 + 20 * eps);
            } else {
                for (int i = 0; i < std::min(cx2, b2); ++i) {
                    int j_max = std::min(cy2, b2 - i) - 1;
                    if (j_max < 0) continue;
                    int j_hi = std::min(j_max, (int)std::floor(s2 * (u + 0.5 + 15 * eps) + 1.0 - L + i));
                    int j_lo = std::max(0, (int)std::ceil(s2 * (u - 0.5 - 15 * eps) - 1.0 - L + i));
                    if (j_hi >= j_lo) {
                        int eff_b2 = i + j_hi + 1;
                        v_lo = std::max(v_lo, -(L - eff_b2 - 1.0) * inv_s2 + 0.5 + 20 * eps);
                    }
                }
            }
        }
        if (v_hi >= v_lo) {
            int cnt = (int)std::floor((v_hi - v_lo) / (1.0 + 2 * eps)) + 1;
            double span = (cnt - 1) * (1.0 + 2 * eps);
            double v0 = 0.5 * (v_lo + v_hi) - 0.5 * span;
            for (int m = 0; m < cnt; ++m) {
                double v = v0 + m * (1.0 + 2 * eps);
                tilted.push_back({(u - v) * inv_s2, (u + v) * inv_s2, pi4});
            }
        }
    }
    std::vector<Sq> pts;
    pts.reserve(n);
    if (corners_first) {
        for (const auto& p : corners) if ((int)pts.size() < n) pts.push_back(p);
        for (const auto& p : tilted)  if ((int)pts.size() < n) pts.push_back(p);
    } else {
        for (const auto& p : tilted)  if ((int)pts.size() < n) pts.push_back(p);
        for (const auto& p : corners) if ((int)pts.size() < n) pts.push_back(p);
    }
    return pts;
}

// Construct a Grid + 1 or 2 Horizontal/L-shaped Tilted Bands at continuous angle theta
inline std::vector<Sq> build_grid_horiz_l_band(int n, int cx, int cy, double L_box, double theta, int variant) {
    std::vector<Sq> pts;
    pts.reserve(n);
    for (int i = 0; i < cx && (int)pts.size() < n; ++i) {
        for (int j = 0; j < cy && (int)pts.size() < n; ++j) {
            if (variant == 1 && i == cx - 1 && j == cy - 1 && cx >= 2 && cy >= 2) continue;
            pts.push_back({0.5 + i, 0.5 + j, 0.0});
        }
    }
    if ((variant & 2) && (int)pts.size() < n) pts.push_back({L_box - 0.5, L_box - 0.5, 0.0});
    if ((variant & 4) && (int)pts.size() < n) {
        pts.push_back({0.5, L_box - 0.5, 0.0});
        if ((int)pts.size() < n) pts.push_back({L_box - 0.5, 0.5, 0.0});
    }
    int rem = n - (int)pts.size();
    if (rem <= 0) return pts;
    theta = wrap_pi2(theta);
    double r_t = 0.5 * (fabs(cos(theta)) + fabs(sin(theta)));
    double lo = r_t + 1e-4, hi = std::max(lo + 0.1, L_box - r_t - 1e-4);
    int n_right = (cx < (int)std::floor(L_box)) ? ((rem + 1) / 2) : 0;
    int n_top = rem - n_right;
    for (int k = 0; k < n_right; ++k) {
        double y = lo + (hi - lo) * (k + 0.5) / std::max(1, n_right);
        double x = 0.5 * (cx + L_box) + ((k & 1) ? 0.14 : -0.14);
        pts.push_back({std::clamp(x, lo, hi), std::clamp(y, lo, hi), theta});
    }
    for (int k = 0; k < n_top; ++k) {
        double x = lo + (hi - lo) * (k + 0.5) / std::max(1, n_top);
        double y = 0.5 * (cy + L_box) + ((k & 1) ? 0.14 : -0.14);
        pts.push_back({std::clamp(x, lo, hi), std::clamp(y, lo, hi), theta});
    }
    return pts;
}

// Construct a Staggered Interior lattice inside a 4-corner or perimeter frame
inline std::vector<Sq> build_staggered_frame_lattice(int n, double L_box, double theta, double stagger_shift, int frame_depth) {
    std::vector<Sq> pts;
    pts.reserve(n);
    // Corner anchors or partial perimeter frame
    int k_floor = (int)std::floor(L_box);
    int fd = std::min(frame_depth, std::max(1, k_floor / 2));
    for (int i = 0; i < fd && (int)pts.size() < n; ++i) {
        for (int j = 0; j < fd - i && (int)pts.size() < n; ++j) {
            pts.push_back({0.5 + i, 0.5 + j, 0.0});
            if ((int)pts.size() < n) pts.push_back({L_box - 0.5 - i, L_box - 0.5 - j, 0.0});
            if ((int)pts.size() < n) pts.push_back({0.5 + i, L_box - 0.5 - j, 0.0});
            if ((int)pts.size() < n) pts.push_back({L_box - 0.5 - i, 0.5 + j, 0.0});
        }
    }
    int rem = n - (int)pts.size();
    if (rem <= 0) return pts;
    theta = wrap_pi2(theta);
    int cols = std::max(1, (int)std::ceil(std::sqrt((double)rem)));
    int rows = (rem + cols - 1) / cols;
    double r_t = 0.5 * (fabs(cos(theta)) + fabs(sin(theta)));
    double lo = r_t + 0.05, hi = std::max(lo + 0.1, L_box - r_t - 0.05);
    for (int k = 0; k < rem; ++k) {
        int ix = k % cols, iy = k / cols;
        double fx = (ix + 0.5 + ((iy & 1) ? stagger_shift : -stagger_shift)) / cols;
        double fy = (iy + 0.5) / std::max(1, rows);
        double x = std::clamp(lo + (hi - lo) * fx, r_t + 1e-4, L_box - r_t - 1e-4);
        double y = std::clamp(lo + (hi - lo) * fy, r_t + 1e-4, L_box - r_t - 1e-4);
        pts.push_back({x, y, theta});
    }
    return pts;
}

// Fast portfolio builder for single-n GPU launches (compatible with dpack::P and seeded::P)
template <typename PType>
inline void build_portfolio_generic(
    int n, int chains, double seed_L, const std::vector<PType>& seed, std::vector<PType>& h_init
) {
    h_init.resize((size_t)n * chains);
    for (int b = 0; b < chains; ++b) {
        for (int i = 0; i < n; ++i) h_init[(size_t)b * n + i] = seed[i];
    }
    if (n <= 1) return;

    const double s2 = std::sqrt(2.0), inv_s2 = 1.0 / s2, eps = 1e-7;
    struct TopoCand {
        double L;
        int w, a1, a2, b1, b2, cx1, cy1, cx2, cy2;
        bool stagger;
        int inject;
    };
    std::vector<TopoCand> cands;

    for (int w = 1; w <= 25; ++w) {
        for (int a1 = 0; a1 <= 48; ++a1) {
            for (int a2 = 0; a2 <= a1; ++a2) {
                double L = 0.5 * (a1 + a2) + 1.0 + w * inv_s2 + 60 * eps;
                if (L < std::sqrt((double)n) - 0.08 || L > seed_L + 0.20) continue;
                int lim = (int)std::floor(L - 1.0 - 20 * eps);
                if (lim < 0 || a2 > lim + 1) continue;
                bool is_clipped = (a1 > lim + 1);
                int c_a1 = 0;
                for (int i = 0; i < std::min(lim + 1, a1); ++i) c_a1 += std::max(0, std::min(lim + 1, a1 - i));
                int c_a2 = a2 * (a2 + 1) / 2;
                int cx1 = is_clipped ? 0 : std::max(0, (int)std::floor(L - a2 - 20 * eps));
                int cy1 = is_clipped ? 0 : std::max(0, (int)std::floor(L - a1 - 20 * eps));
                int cx2 = cy1, cy2 = cx1;
                int max_b1 = is_clipped ? 0 : std::max(0, std::min(cx1 + cy1 - 1, (int)std::floor(L - 1.0 - 20 * eps)));

                for (int b1 = 0; b1 <= max_b1; ++b1) {
                    int c_b1 = 0;
                    for (int i = 0; i < std::min(cx1, b1); ++i) c_b1 += std::max(0, std::min(cy1, b1 - i));
                    int max_b2 = std::min(b1, std::max(0, (int)std::floor(L - 1.0 - 20 * eps)));
                    for (int b2 = 0; b2 <= max_b2; ++b2) {
                        if (cx1 + cx2 > L - 20 * eps && cy1 + cy2 > L - 20 * eps && b1 + b2 > L - 20 * eps) continue;
                        int c_b2 = 0;
                        for (int i = 0; i < std::min(cx2, b2); ++i) c_b2 += std::max(0, std::min(cy2, b2 - i));
                        int corner_cnt = c_a1 + c_a2 + c_b1 + c_b2;
                        if (corner_cnt > n + 20) continue;
                        double u_start = (a1 + 1.0) * inv_s2 + 0.5 + 20 * eps;
                        int til_cnt = 0;
                        for (int k = 0; k < w; ++k) {
                            double u = u_start + k * (1.0 + 2 * eps);
                            double v_lo = -std::min(u - 1.0 - 10 * eps, s2 * L - u - 1.0 - 10 * eps);
                            double v_hi =  std::min(u - 1.0 - 10 * eps, s2 * L - u - 1.0 - 10 * eps);
                            if (b1 > 0) v_hi = std::min(v_hi, (L - b1 - 1.0) * inv_s2 - 0.5 - 20 * eps);
                            if (b2 > 0) v_lo = std::max(v_lo, -(L - b2 - 1.0) * inv_s2 + 0.5 + 20 * eps);
                            if (v_hi >= v_lo) til_cnt += (int)std::floor((v_hi - v_lo) / (1.0 + 2 * eps)) + 1;
                        }
                        int cap = corner_cnt + til_cnt;
                        if (cap >= n && cap <= n + 24) {
                            cands.push_back({L + 1e-4 * (cap - n), w, a1, a2, b1, b2, cx1, cy1, cx2, cy2, false, 0});
                        } else if (b1 == 0 && b2 == 0 && (cap == n - 1 || cap == n - 2) && L > std::max(a1, a2) + 1.0) {
                            cands.push_back({L + 0.005 * (n - cap), w, a1, a2, 0, 0, cx1, cy1, cx2, cy2, false, n - cap});
                        }
                    }
                }
            }
        }
    }
    std::sort(cands.begin(), cands.end(), [](const TopoCand& a, const TopoCand& b) { return a.L < b.L; });
    if (cands.size() > 64) cands.resize(64);

    int k_floor = (int)std::floor(std::sqrt((double)n));
    for (int b = 0; b < chains; ++b) {
        int ladder = b / 8;
        int group = b % 8;
        if (group <= 1) {
            if (ladder >= 2 && !cands.empty()) {
                const auto& tc = cands[(ladder - 2) % cands.size()];
                auto pts = materialize_4corner_bands(
                    n - tc.inject, tc.L, tc.w, tc.a1, tc.a2, tc.b1, tc.b2,
                    tc.cx1, tc.cy1, tc.cx2, tc.cy2, tc.stagger, true
                );
                if (tc.inject >= 1) pts.push_back({0.5 + eps, tc.L - 0.5 - eps, 0.0});
                if (tc.inject >= 2) pts.push_back({tc.L - 0.5 - eps, 0.5 + eps, 0.0});
                for (int i = 0; i < n && i < (int)pts.size(); ++i)
                    h_init[(size_t)b * n + i] = {pts[i].x, pts[i].y, wrap_pi2(pts[i].a)};
            }
        } else if (group <= 4 && !cands.empty()) {
            const auto& tc = cands[ladder % cands.size()];
            bool corners_first = (group != 4);
            auto pts = materialize_4corner_bands(
                n - tc.inject, tc.L, tc.w, tc.a1, tc.a2, tc.b1, tc.b2,
                tc.cx1, tc.cy1, tc.cx2, tc.cy2, tc.stagger, corners_first
            );
            if (tc.inject >= 1) pts.push_back({0.5 + eps, tc.L - 0.5 - eps, 0.0});
            if (tc.inject >= 2) pts.push_back({tc.L - 0.5 - eps, 0.5 + eps, 0.0});
            for (int i = 0; i < n && i < (int)pts.size(); ++i)
                h_init[(size_t)b * n + i] = {pts[i].x, pts[i].y, wrap_pi2(pts[i].a)};
        } else if (group == 5 || group == 6) {
            int cx = std::max(1, k_floor - 1 - (ladder % 2));
            int cy = std::max(1, k_floor - 1 - ((ladder / 2) % 2));
            double L_box = std::max(std::sqrt((double)n) + 0.05, seed_L - 0.03);
            double th = 0.22 + 0.08 * (ladder % 7);
            auto pts = build_grid_horiz_l_band(n, cx, cy, L_box, th, (group == 6 ? 1 : 0) | ((ladder % 3) ? 2 : 0));
            for (int i = 0; i < n && i < (int)pts.size(); ++i)
                h_init[(size_t)b * n + i] = {pts[i].x, pts[i].y, wrap_pi2(pts[i].a)};
        } else {
            double L_box = std::max(std::sqrt((double)n) + 0.05, seed_L - 0.02);
            double th = 0.30 + 0.07 * (ladder % 6);
            double shift = 0.15 + 0.05 * (ladder % 4);
            auto pts = build_staggered_frame_lattice(n, L_box, th, shift, 1 + (ladder % 2));
            for (int i = 0; i < n && i < (int)pts.size(); ++i)
                h_init[(size_t)b * n + i] = {pts[i].x, pts[i].y, wrap_pi2(pts[i].a)};
        }
    }
}

} // namespace sqfam

