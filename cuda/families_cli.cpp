// Multithreaded C++17 driver for parametric family enumeration and continuous parameter
// optimization across all non-square n in [1, 1000].
#include "families.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace sqfam;

struct FamCand {
    double L = 1e9;
    std::vector<Sq> pts;
};

static bool load_csv(const std::string& path, std::vector<std::vector<Sq>>& puzzles, std::vector<double>& inc_L) {
    std::ifstream in(path);
    if (!in) return false;
    puzzles.assign(1001, {});
    inc_L.assign(1001, 1e9);
    std::string line;
    if (!std::getline(in, line)) return false;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, ',')) cols.push_back(tok);
        if (cols.size() < 6) continue;
        int n = std::atoi(cols[1].c_str());
        if (n < 1 || n > 1000) continue;
        double x = std::strtod(cols[3].c_str(), nullptr);
        double y = std::strtod(cols[4].c_str(), nullptr);
        double a = std::strtod(cols[5].c_str(), nullptr);
        puzzles[n].push_back({x, y, a});
    }
    for (int n = 1; n <= 1000; ++n) {
        if ((int)puzzles[n].size() != n) continue;
        double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
        for (auto& p : puzzles[n]) {
            double r = 0.5 * (std::fabs(std::cos(p.a)) + std::fabs(std::sin(p.a)));
            min_x = std::min(min_x, p.x - r); max_x = std::max(max_x, p.x + r);
            min_y = std::min(min_y, p.y - r); max_y = std::max(max_y, p.y + r);
        }
        for (auto& p : puzzles[n]) { p.x -= min_x; p.y -= min_y; p.a = wrap_pi2(p.a); }
        inc_L[n] = std::max(max_x - min_x, max_y - min_y);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: families_cli <incumbent.csv> <output.json> [--max-n N] [--polish-iters M] [--threads T]\n";
        return 2;
    }
    std::string inc_path = argv[1];
    std::string out_path = argv[2];
    int max_n = 1000;
    int polish_iters = 60;
    int num_threads = std::max(1u, std::thread::hardware_concurrency());
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--max-n" && i + 1 < argc) max_n = std::atoi(argv[++i]);
        else if (arg == "--polish-iters" && i + 1 < argc) polish_iters = std::atoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) num_threads = std::max(1, std::atoi(argv[++i]));
    }

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<Sq>> inc_puzzles;
    std::vector<double> inc_L;
    if (!load_csv(inc_path, inc_puzzles, inc_L)) {
        std::cerr << "Failed to load " << inc_path << "\n";
        return 2;
    }

    // Per-n, per-family best candidate
    std::vector<std::vector<FamCand>> fam_best(max_n + 1, std::vector<FamCand>(FAM_COUNT));

    const double s2 = std::sqrt(2.0), inv_s2 = 1.0 / s2, eps = 1e-7;
    struct AnalyticSpec {
        double L = 1e9;
        int w = 0, a1 = 0, a2 = 0, b1 = 0, b2 = 0, cx1 = 0, cy1 = 0, cx2 = 0, cy2 = 0, cap = 0, inject = 0;
        bool stagger = false;
        bool valid = false;
    };
    std::vector<AnalyticSpec> best_gobel_spec(max_n + 1), best_clipped_spec(max_n + 1),
                              best_corridor_spec(max_n + 1), best_stagger_spec(max_n + 1),
                              best_inject_spec(max_n + 1);

    for (int w = 1; w <= 26; ++w) {
        for (int a1 = 0; a1 <= 62; ++a1) {
            for (int a2 = 0; a2 <= a1; ++a2) {
                double L = 0.5 * (a1 + a2) + 1.0 + w * inv_s2 + 60 * eps;
                if (L > 33.0) continue;
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

                double u_start = (a1 + 1.0) * inv_s2 + 0.5 + 20 * eps;
                for (int b1 = 0; b1 <= max_b1; ++b1) {
                    int c_b1 = 0;
                    for (int i = 0; i < std::min(cx1, b1); ++i) c_b1 += std::max(0, std::min(cy1, b1 - i));
                    int max_b2 = std::min(b1, std::max(0, (int)std::floor(L - 1.0 - 20 * eps)));
                    for (int b2 = 0; b2 <= max_b2; ++b2) {
                        if (cx1 + cx2 > L - 20 * eps && cy1 + cy2 > L - 20 * eps && b1 + b2 > L - 20 * eps) continue;
                        int c_b2 = 0;
                        for (int i = 0; i < std::min(cx2, b2); ++i) c_b2 += std::max(0, std::min(cy2, b2 - i));
                        int corner_cnt = c_a1 + c_a2 + c_b1 + c_b2;
                        if (corner_cnt > max_n) continue;

                        int til_unif = 0, til_stag = 0;
                        for (int k = 0; k < w; ++k) {
                            double u = u_start + k * (1.0 + 2 * eps);
                            double v_lo_u = -std::min(u - 1.0 - 10 * eps, s2 * L - u - 1.0 - 10 * eps);
                            double v_hi_u =  std::min(u - 1.0 - 10 * eps, s2 * L - u - 1.0 - 10 * eps);
                            double v_lo_s = v_lo_u, v_hi_s = v_hi_u;
                            if (b1 > 0) {
                                v_hi_u = std::min(v_hi_u, (L - b1 - 1.0) * inv_s2 - 0.5 - 20 * eps);
                                for (int i = 0; i < std::min(cx1, b1); ++i) {
                                    int j_max = std::min(cy1, b1 - i) - 1;
                                    if (j_max < 0) continue;
                                    int j_hi = std::min(j_max, (int)std::floor(L + i + 1.0 - s2 * (u - 0.5 - 15 * eps)));
                                    int j_lo = std::max(0, (int)std::ceil(L + i - 1.0 - s2 * (u + 0.5 + 15 * eps)));
                                    if (j_hi >= j_lo) {
                                        int eff_b1 = i + j_hi + 1;
                                        v_hi_s = std::min(v_hi_s, (L - eff_b1 - 1.0) * inv_s2 - 0.5 - 20 * eps);
                                    }
                                }
                            }
                            if (b2 > 0) {
                                v_lo_u = std::max(v_lo_u, -(L - b2 - 1.0) * inv_s2 + 0.5 + 20 * eps);
                                for (int i = 0; i < std::min(cx2, b2); ++i) {
                                    int j_max = std::min(cy2, b2 - i) - 1;
                                    if (j_max < 0) continue;
                                    int j_hi = std::min(j_max, (int)std::floor(s2 * (u + 0.5 + 15 * eps) + 1.0 - L + i));
                                    int j_lo = std::max(0, (int)std::ceil(s2 * (u - 0.5 - 15 * eps) - 1.0 - L + i));
                                    if (j_hi >= j_lo) {
                                        int eff_b2 = i + j_hi + 1;
                                        v_lo_s = std::max(v_lo_s, -(L - eff_b2 - 1.0) * inv_s2 + 0.5 + 20 * eps);
                                    }
                                }
                            }
                            if (v_hi_u >= v_lo_u) til_unif += (int)std::floor((v_hi_u - v_lo_u) / (1.0 + 2 * eps)) + 1;
                            if (v_hi_s >= v_lo_s) til_stag += (int)std::floor((v_hi_s - v_lo_s) / (1.0 + 2 * eps)) + 1;
                        }
                        int cap_u = corner_cnt + til_unif;
                        int cap_s = corner_cnt + til_stag;
                        bool is_corridor = (b1 > std::min(cx1, cy1)) || (b2 > std::min(cx2, cy2));
                        bool is_asymmetric_stagger = (!is_clipped && !is_corridor && ((a1 - a2 >= 2) || (b1 - b2 >= 2)));

                        auto& target_vec = is_clipped ? best_clipped_spec :
                                           is_corridor ? best_corridor_spec :
                                           is_asymmetric_stagger ? best_stagger_spec : best_gobel_spec;
                        for (int n = std::max(1, cap_u - 30); n <= std::min(max_n, cap_u); ++n) {
                            if (L < target_vec[n].L) {
                                target_vec[n] = {L, w, a1, a2, b1, b2, cx1, cy1, cx2, cy2, cap_u, 0, false, true};
                            }
                        }
                        if (cap_s > cap_u) {
                            for (int n = std::max(1, cap_s - 30); n <= std::min(max_n, cap_s); ++n) {
                                if (L < best_stagger_spec[n].L) {
                                    best_stagger_spec[n] = {L, w, a1, a2, b1, b2, cx1, cy1, cx2, cy2, cap_s, 0, true, true};
                                }
                            }
                        }
                        if (!is_clipped && b1 == 0 && b2 == 0 && L > std::max(a1, a2) + 1.0) {
                            for (int inj = 1; inj <= 2; ++inj) {
                                int n = cap_u + inj;
                                if (n >= 1 && n <= max_n && L + 0.004 * inj < best_inject_spec[n].L) {
                                    best_inject_spec[n] = {L, w, a1, a2, 0, 0, cx1, cy1, cx2, cy2, cap_u, inj, false, true};
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Materialize analytical specs for Families 0, 1 (clipped), 2 (corridor), 3 (staggered)
    std::vector<double> pool_L = inc_L;
    std::vector<std::vector<Sq>> pool_pts = inc_puzzles;

    for (int n = 1; n <= max_n; ++n) {
        int r = (int)std::round(std::sqrt((double)n));
        if (r * r == n) continue;

        if (best_gobel_spec[n].valid) {
            const auto& sp = best_gobel_spec[n];
            auto pts = materialize_4corner_bands(n, sp.L, sp.w, sp.a1, sp.a2, sp.b1, sp.b2, sp.cx1, sp.cy1, sp.cx2, sp.cy2, false, true);
            certified_side_cpu(pts, &fam_best[n][FAM_GOBEL_STAIRCASE].pts, fam_best[n][FAM_GOBEL_STAIRCASE].L);
            fam_best[n][FAM_GOBEL_STAIRCASE].L = certified_side_cpu(fam_best[n][FAM_GOBEL_STAIRCASE].pts);
            if (fam_best[n][FAM_GOBEL_STAIRCASE].L < pool_L[n]) {
                pool_L[n] = fam_best[n][FAM_GOBEL_STAIRCASE].L;
                pool_pts[n] = fam_best[n][FAM_GOBEL_STAIRCASE].pts;
            }
        }
        if (best_clipped_spec[n].valid) {
            const auto& sp = best_clipped_spec[n];
            auto pts = materialize_4corner_bands(n, sp.L, sp.w, sp.a1, sp.a2, sp.b1, sp.b2, sp.cx1, sp.cy1, sp.cx2, sp.cy2, false, true);
            std::vector<Sq> out;
            double cL = certified_side_cpu(pts, &out, fam_best[n][FAM_GRID_TILTED_BANDS].L);
            if (cL < fam_best[n][FAM_GRID_TILTED_BANDS].L) {
                fam_best[n][FAM_GRID_TILTED_BANDS] = {cL, out};
            }
        }
        if (best_corridor_spec[n].valid) {
            const auto& sp = best_corridor_spec[n];
            auto pts = materialize_4corner_bands(n, sp.L, sp.w, sp.a1, sp.a2, sp.b1, sp.b2, sp.cx1, sp.cy1, sp.cx2, sp.cy2, false, true);
            std::vector<Sq> out;
            double cL = certified_side_cpu(pts, &out, fam_best[n][FAM_CORNER_RECT_CORRIDOR].L);
            if (cL < fam_best[n][FAM_CORNER_RECT_CORRIDOR].L) {
                fam_best[n][FAM_CORNER_RECT_CORRIDOR] = {cL, out};
            }
            if (cL < pool_L[n]) { pool_L[n] = cL; pool_pts[n] = out; }
        }
        if (best_stagger_spec[n].valid) {
            const auto& sp = best_stagger_spec[n];
            auto pts = materialize_4corner_bands(n, sp.L, sp.w, sp.a1, sp.a2, sp.b1, sp.b2, sp.cx1, sp.cy1, sp.cx2, sp.cy2, sp.stagger, true);
            std::vector<Sq> out;
            double cL = certified_side_cpu(pts, &out, fam_best[n][FAM_STAGGERED_INTERIOR].L);
            if (cL < fam_best[n][FAM_STAGGERED_INTERIOR].L) {
                fam_best[n][FAM_STAGGERED_INTERIOR] = {cL, out};
            }
            if (cL < pool_L[n]) { pool_L[n] = cL; pool_pts[n] = out; }
        }
    }

    // Family 1 (grid_tilted_bands): L-bordered core extensions from pool_pts[m]
    struct BorderSpec { double L = 1e9; int m = 0, k0 = 0, t = 0; bool valid = false; };
    std::vector<BorderSpec> best_border(max_n + 1);
    for (int m = 2; m <= max_n; ++m) {
        if ((int)pool_pts[m].size() != m) continue;
        double Lm = pool_L[m];
        int k0 = (int)std::floor(Lm - 1e-6);
        if (k0 < 1) continue;
        for (int t = 1; k0 + t <= 32; ++t) {
            int k = k0 + t;
            int cap = m + k * k - k0 * k0;
            double L_new = Lm + t + 3e-6;
            for (int n = std::max(1, cap - 20); n <= std::min(max_n, cap); ++n) {
                if (L_new < best_border[n].L) {
                    best_border[n] = {L_new, m, k0, t, true};
                }
            }
        }
    }
    for (int n = 1; n <= max_n; ++n) {
        int r = (int)std::round(std::sqrt((double)n));
        if (r * r == n || !best_border[n].valid) continue;
        const auto& bs = best_border[n];
        const auto& core = pool_pts[bs.m];
        double Lm = pool_L[bs.m] + 1e-7;
        int k0 = bs.k0, t = bs.t, k = k0 + t;
        double p = 1.0 + 1e-7;
        std::vector<Sq> pts = core;
        // Right block: t x k grid at x >= Lm
        for (int i = 0; i < t && (int)pts.size() < n; ++i) {
            for (int j = 0; j < k && (int)pts.size() < n; ++j) {
                pts.push_back({Lm + 0.5 + i * p, 0.5 + j * p, 0.0});
            }
        }
        // Top block: k0 x t grid at y >= Lm
        for (int i = 0; i < k0 && (int)pts.size() < n; ++i) {
            for (int j = 0; j < t && (int)pts.size() < n; ++j) {
                pts.push_back({0.5 + i * p, Lm + 0.5 + j * p, 0.0});
            }
        }
        if ((int)pts.size() > n) pts.resize(n);
        std::vector<Sq> out;
        double cL = certified_side_cpu(pts, &out, fam_best[n][FAM_GRID_TILTED_BANDS].L);
        if (cL < fam_best[n][FAM_GRID_TILTED_BANDS].L) {
            fam_best[n][FAM_GRID_TILTED_BANDS] = {cL, out};
        }
        if (cL < pool_L[n]) { pool_L[n] = cL; pool_pts[n] = out; }
    }

    // Parallel continuous parameter & coordinate optimization across all non-square n
    std::atomic<int> next_n{2};
    std::vector<std::thread> workers;
    for (int tid = 0; tid < num_threads; ++tid) {
        workers.emplace_back([&]() {
            while (true) {
                int n = next_n.fetch_add(1);
                if (n > max_n) break;
                int k_floor = (int)std::floor(std::sqrt((double)n));
                if (k_floor * k_floor == n) continue;

                // 0. Gobel staircase: corner-injected + contraction polish
                if (best_inject_spec[n].valid && polish_iters > 0) {
                    const auto& sp = best_inject_spec[n];
                    auto pts = materialize_4corner_bands(
                        n - sp.inject, sp.L, sp.w, sp.a1, sp.a2, 0, 0, sp.cx1, sp.cy1, sp.cx2, sp.cy2, false, true
                    );
                    if (sp.inject >= 1) pts.push_back({0.5 + 1e-6, sp.L - 0.5 - 1e-6, 0.0});
                    if (sp.inject >= 2) pts.push_back({sp.L - 0.5 - 1e-6, 0.5 + 1e-6, 0.0});
                    if ((int)pts.size() == n) {
                        polish_and_certify_cpu(pts, sp.L + 0.002, fam_best[n][FAM_GOBEL_STAIRCASE].pts, fam_best[n][FAM_GOBEL_STAIRCASE].L, polish_iters);
                    }
                }
                if (!fam_best[n][FAM_GOBEL_STAIRCASE].pts.empty() && polish_iters > 0 && best_gobel_spec[n].cap > n) {
                    auto start_pts = fam_best[n][FAM_GOBEL_STAIRCASE].pts;
                    contract_and_polish_cpu(start_pts, fam_best[n][FAM_GOBEL_STAIRCASE].pts, fam_best[n][FAM_GOBEL_STAIRCASE].L, 2, polish_iters / 2);
                }

                // 1. Grid + tilted bands: continuous theta optimization for small/medium n or trimmed border polish
                if (!fam_best[n][FAM_GRID_TILTED_BANDS].pts.empty() && polish_iters > 0 && n <= 250) {
                    auto start_pts = fam_best[n][FAM_GRID_TILTED_BANDS].pts;
                    contract_and_polish_cpu(start_pts, fam_best[n][FAM_GRID_TILTED_BANDS].pts, fam_best[n][FAM_GRID_TILTED_BANDS].L, 2, polish_iters / 2);
                }
                if (n <= 120 && polish_iters > 0) {
                    double L_box = std::min((double)k_floor + 0.95, inc_L[n] - 0.005);
                    for (int th_idx = 0; th_idx < 3; ++th_idx) {
                        double th = 0.25 + 0.20 * th_idx;
                        auto pts = build_grid_horiz_l_band(n, std::max(1, k_floor - 1), k_floor, L_box, th, 0);
                        polish_and_certify_cpu(pts, L_box, fam_best[n][FAM_GRID_TILTED_BANDS].pts, fam_best[n][FAM_GRID_TILTED_BANDS].L, polish_iters);
                    }
                }

                // 2. Corner rectangle + tilted corridor: continuous theta optimization + beveled corridor polish
                if (!fam_best[n][FAM_CORNER_RECT_CORRIDOR].pts.empty() && polish_iters > 0 && best_corridor_spec[n].cap > n && n <= 300) {
                    auto start_pts = fam_best[n][FAM_CORNER_RECT_CORRIDOR].pts;
                    contract_and_polish_cpu(start_pts, fam_best[n][FAM_CORNER_RECT_CORRIDOR].pts, fam_best[n][FAM_CORNER_RECT_CORRIDOR].L, 2, polish_iters / 2);
                }
                if (n <= 120 && polish_iters > 0) {
                    double L_box = std::min((double)k_floor + 0.92, inc_L[n] - 0.005);
                    for (int v = 1; v <= 3; ++v) {
                        double th = 0.30 + 0.15 * v;
                        auto pts = build_grid_horiz_l_band(n, std::max(1, k_floor - 1), std::max(1, k_floor - 1), L_box, th, v);
                        polish_and_certify_cpu(pts, L_box, fam_best[n][FAM_CORNER_RECT_CORRIDOR].pts, fam_best[n][FAM_CORNER_RECT_CORRIDOR].L, polish_iters);
                    }
                }

                // 3. Staggered interior: per-band staggered polish + frame lattice optimization
                if (!fam_best[n][FAM_STAGGERED_INTERIOR].pts.empty() && polish_iters > 0 && n <= 300) {
                    auto start_pts = fam_best[n][FAM_STAGGERED_INTERIOR].pts;
                    contract_and_polish_cpu(start_pts, fam_best[n][FAM_STAGGERED_INTERIOR].pts, fam_best[n][FAM_STAGGERED_INTERIOR].L, 2, polish_iters / 2);
                }
                if (n <= 100 && polish_iters > 0) {
                    double L_box = std::min((double)k_floor + 0.92, inc_L[n] - 0.005);
                    for (int s_idx = 0; s_idx < 2; ++s_idx) {
                        auto pts = build_staggered_frame_lattice(n, L_box, 0.40 + 0.25 * s_idx, 0.22, 1 + s_idx);
                        polish_and_certify_cpu(pts, L_box, fam_best[n][FAM_STAGGERED_INTERIOR].pts, fam_best[n][FAM_STAGGERED_INTERIOR].L, polish_iters);
                    }
                }

                // 4. Record / Incumbent neighbors: Deletion from m in [n+1, n+6] + Contraction CG Polish, and Void Insertion from n-1
                for (int m = n + 1; m <= std::min(max_n, n + 6); ++m) {
                    if ((int)inc_puzzles[m].size() != m) continue;
                    const auto& src = inc_puzzles[m];
                    // Candidate deletion A: remove m-n squares with largest max(x + r, y + r) (boundary extremum)
                    std::vector<std::pair<double, int>> score_ext(m), score_crowd(m);
                    for (int i = 0; i < m; ++i) {
                        double r = 0.5 * (std::fabs(std::cos(src[i].a)) + std::fabs(std::sin(src[i].a)));
                        score_ext[i] = {std::max(src[i].x + r, src[i].y + r), i};
                        double crowd = 0.0;
                        for (int j = 0; j < m; ++j) if (i != j) {
                            double dx = src[i].x - src[j].x, dy = src[i].y - src[j].y;
                            double d2 = dx * dx + dy * dy;
                            if (d2 < 2.2) crowd += 1.0 / (0.1 + d2);
                        }
                        score_crowd[i] = {crowd, i};
                    }
                    std::sort(score_ext.rbegin(), score_ext.rend());
                    std::sort(score_crowd.rbegin(), score_crowd.rend());

                    for (int mode = 0; mode < 2; ++mode) {
                        if (mode == 1 && m > n + 2) continue;
                        const auto& ord = (mode == 0) ? score_ext : score_crowd;
                        std::vector<bool> drop(m, false);
                        for (int k = 0; k < m - n; ++k) drop[ord[k].second] = true;
                        std::vector<Sq> sub;
                        sub.reserve(n);
                        for (int i = 0; i < m; ++i) if (!drop[i]) sub.push_back(src[i]);
                        double cL = certified_side_cpu(sub, &fam_best[n][FAM_RECORD_NEIGHBORS].pts, fam_best[n][FAM_RECORD_NEIGHBORS].L);
                        if (cL < fam_best[n][FAM_RECORD_NEIGHBORS].L) fam_best[n][FAM_RECORD_NEIGHBORS].L = cL;
                        // Run contraction polish if close to beating best known for n
                        if (polish_iters > 0 && m <= n + 2 && cL < inc_L[n] + 0.025 && n <= 400) {
                            contract_and_polish_cpu(sub, fam_best[n][FAM_RECORD_NEIGHBORS].pts, fam_best[n][FAM_RECORD_NEIGHBORS].L, 3, polish_iters);
                        }
                    }
                }
                // Void insertion from m = n - 1 for small/medium n
                if (n >= 2 && n <= 160 && polish_iters > 0 && (int)inc_puzzles[n - 1].size() == n - 1) {
                    const auto& src = inc_puzzles[n - 1];
                    double L_prev = inc_L[n - 1];
                    if (L_prev < inc_L[n] - 0.002) {
                        double best_d2 = -1.0, best_vx = 0.5 * L_prev, best_vy = 0.5 * L_prev;
                        for (int ix = 0; ix < 10; ++ix) {
                            for (int iy = 0; iy < 10; ++iy) {
                                double vx = 0.55 + std::max(0.1, L_prev - 1.1) * (ix + 0.5) / 10.0;
                                double vy = 0.55 + std::max(0.1, L_prev - 1.1) * (iy + 0.5) / 10.0;
                                double min_d2 = 1e9;
                                for (const auto& p : src) {
                                    double dx = vx - p.x, dy = vy - p.y;
                                    min_d2 = std::min(min_d2, dx * dx + dy * dy);
                                }
                                if (min_d2 > best_d2) { best_d2 = min_d2; best_vx = vx; best_vy = vy; }
                            }
                        }
                        std::vector<Sq> ins = src;
                        ins.push_back({best_vx, best_vy, 0.7853981633974483});
                        polish_and_certify_cpu(ins, L_prev + 0.005, fam_best[n][FAM_RECORD_NEIGHBORS].pts, fam_best[n][FAM_RECORD_NEIGHBORS].L, polish_iters);
                    }
                }
            }
        });
    }
    for (auto& th : workers) th.join();

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Write JSON output with per-n winners and per-family statistics
    std::ofstream out(out_path);
    if (!out) return 2;
    out << std::setprecision(17);
    out << "{\"max_n\":" << max_n << ",\"seconds\":" << elapsed << ",\"puzzles\":[";
    bool first_p = true;
    for (int n = 1; n <= max_n; ++n) {
        int r = (int)std::round(std::sqrt((double)n));
        if (r * r == n) continue;
        int best_f = -1;
        double best_L = inc_L[n] - 1e-6;
        for (int f = 0; f < FAM_COUNT; ++f) {
            if (fam_best[n][f].L < best_L - 1e-12 && (int)fam_best[n][f].pts.size() == n) {
                best_L = fam_best[n][f].L;
                best_f = f;
            }
        }
        if (best_f < 0) best_L = inc_L[n];
        if (!first_p) out << ",";
        first_p = false;
        out << "\n{\"n\":" << n << ",\"incumbent_L\":" << inc_L[n] << ",\"best_L\":" << best_L
            << ",\"winning_family\":\"" << (best_f >= 0 ? family_name(best_f) : "incumbent") << "\""
            << ",\"family_L\":{";
        for (int f = 0; f < FAM_COUNT; ++f) {
            if (f) out << ",";
            out << "\"" << family_name(f) << "\":";
            if (fam_best[n][f].L < 1e8) out << fam_best[n][f].L;
            else out << "null";
        }
        out << "}";
        if (best_f >= 0) {
            out << ",\"q\":[";
            const auto& pts = fam_best[n][best_f].pts;
            for (int i = 0; i < n; ++i) {
                if (i) out << ",";
                out << "[" << pts[i].x << "," << pts[i].y << "," << pts[i].a << "]";
            }
            out << "]";
        }
        out << "}";
    }
    out << "\n]}\n";
    return 0;
}
