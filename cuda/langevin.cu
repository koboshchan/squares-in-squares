// Ensemble Langevin Heavy-Ball continuous penalty optimizer for square packing in a square.
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "families.hpp"

#define CK(x) do { \
    cudaError_t ck_error = (x); \
    if (ck_error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(ck_error) << "\n"; \
        exit(2); \
    } \
} while (0)

constexpr int MAXN = 1024;
constexpr float PI_2 = 1.5707963267948966f;
constexpr double PI_2_D = 1.57079632679489661923;

struct CertPoint {
    double x, y, a;
};

__device__ inline float sg_smooth_f(float x, float eps = 1e-4f) {
    if (fabsf(x) >= eps) return (x > 0.0f) ? 1.0f : -1.0f;
    float u = x / eps;
    return u * (1.5f - 0.5f * u * u);
}

__host__ __device__ inline double wrap_angle(double a) {
    a = fmod(a, PI_2_D);
    if (a < 0.0) a += PI_2_D;
    return a;
}

__device__ inline float wrap_angle_f(float a) {
    a = fmodf(a, PI_2);
    if (a < 0.0f) a += PI_2;
    return a;
}

__device__ inline float block_sum_overlap(float val, float* s_tmp, int num_threads) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) {
        s_tmp[wid] = val;
    }
    __syncthreads();

    int num_warps = (num_threads + 31) >> 5;
    if (wid == 0) {
        float warp_val = (lane < num_warps) ? s_tmp[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_val += __shfl_down_sync(0xffffffff, warp_val, offset);
        }
        if (lane == 0) {
            s_tmp[0] = warp_val;
        }
    }
    __syncthreads();
    return s_tmp[0];
}

__device__ inline double block_max(double val, double* tmp) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) tmp[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    if (wid == 0) {
        val = (lane < num_warps) ? tmp[lane] : -1e100;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane == 0) tmp[0] = val;
    }
    __syncthreads();
    return tmp[0];
}

__device__ inline double block_min(double val, double* tmp) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmin(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) tmp[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    if (wid == 0) {
        val = (lane < num_warps) ? tmp[lane] : 1e100;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmin(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane == 0) tmp[0] = val;
    }
    __syncthreads();
    return tmp[0];
}

__device__ double device_certified_side(
    const double* q_x, const double* q_y, const double* q_a,
    int n, const double* c, const double* s, double* tmp,
    CertPoint* best_out, double current_best_L
) {
    int t = threadIdx.x;
    constexpr double tau = 2.6e-8;
    double local_lam = 1.0;
    for (int i = t; i < n; i += blockDim.x) {
        double xi = q_x[i], yi = q_y[i], ci = c[i], si = s[i];
        for (int j = i + 1; j < n; j++) {
            double dx = xi - q_x[j], dy = yi - q_y[j];
            if (dx * dx + dy * dy >= 2.05) continue;
            double cj = c[j], sj = s[j];
            double c12 = fabs(ci * cj + si * sj);
            double s12 = fabs(-si * cj + ci * sj);
            double r = 0.5 + 0.5 * (c12 + s12);
            double v1 = fabs(dx * ci + dy * si);
            double v2 = fabs(-dx * si + dy * ci);
            double v3 = fabs(dx * cj + dy * sj);
            double v4 = fabs(-dx * sj + dy * cj);
            double vmax = fmax(fmax(v1, v2), fmax(v3, v4));
            double req = (vmax > 1e-12) ? ((r + tau) / vmax) : 1e9;
            if (req > local_lam) local_lam = req;
        }
    }
    double lam = block_max(local_lam, tmp);
    if (lam > 1.008) return 1e9;

    double my_min_x = 1e9, my_max_x = -1e9, my_min_y = 1e9, my_max_y = -1e9;
    for (int i = t; i < n; i += blockDim.x) {
        double x = q_x[i] * lam;
        double y = q_y[i] * lam;
        double r = 0.5 * (fabs(c[i]) + fabs(s[i]));
        if (x - r < my_min_x) my_min_x = x - r;
        if (x + r > my_max_x) my_max_x = x + r;
        if (y - r < my_min_y) my_min_y = y - r;
        if (y + r > my_max_y) my_max_y = y + r;
    }
    double min_x = block_min(my_min_x, tmp);
    double max_x = block_max(my_max_x, tmp);
    double min_y = block_min(my_min_y, tmp);
    double max_y = block_max(my_max_y, tmp);
    double L_cert = fmax(max_x - min_x, max_y - min_y);

    if (L_cert < current_best_L && best_out != nullptr) {
        for (int i = t; i < n; i += blockDim.x) {
            best_out[i] = {q_x[i] * lam - min_x, q_y[i] * lam - min_y, wrap_angle(q_a[i])};
        }
        __syncthreads();
    }
    return L_cert;
}

__global__ void init_rng_kernel(curandStatePhilox4_32_10_t* rng, int total_threads, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_threads) {
        curand_init(seed, idx, 0, rng + idx);
    }
}

__global__ void evaluate_forces_and_step(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ theta,
    float* __restrict__ vel_x,
    float* __restrict__ vel_y,
    float* __restrict__ vel_theta,
    float* __restrict__ box_s,
    curandStatePhilox4_32_10_t* rng_states,
    int n,
    int steps,
    float delta_s,
    float base_noise
) {
    int b = blockIdx.x;
    int i = threadIdx.x;

    __shared__ float s_pos_x[MAXN];
    __shared__ float s_pos_y[MAXN];
    __shared__ float s_theta[MAXN];
    __shared__ float s_cos[MAXN];
    __shared__ float s_sin[MAXN];
    __shared__ float s_reduce[32];
    __shared__ float s_box_s;
    __shared__ int s_jam_count;
    __shared__ float s_last_overlap;

    if (threadIdx.x == 0) {
        s_box_s = box_s[b];
        s_jam_count = 0;
        s_last_overlap = 1e9f;
    }

    float my_pos_x = 0.0f;
    float my_pos_y = 0.0f;
    float my_theta = 0.0f;
    float my_vel_x = 0.0f;
    float my_vel_y = 0.0f;
    float my_vel_theta = 0.0f;

    if (i < n) {
        int idx = b * n + i;
        my_pos_x = pos_x[idx];
        my_pos_y = pos_y[idx];
        my_theta = theta[idx];
        my_vel_x = vel_x[idx];
        my_vel_y = vel_y[idx];
        my_vel_theta = vel_theta[idx];
    }

    curandStatePhilox4_32_10_t rng_state;
    if (threadIdx.x < blockDim.x) {
        rng_state = rng_states[b * blockDim.x + threadIdx.x];
    }
    __syncthreads();

    for (int step = 0; step < steps; step++) {
        if (i < n) {
            s_pos_x[i] = my_pos_x;
            s_pos_y[i] = my_pos_y;
            s_theta[i] = my_theta;
            __sincosf(my_theta, &s_sin[i], &s_cos[i]);
        }
        __syncthreads();

        float current_s = s_box_s;
        float fx = 0.0f;
        float fy = 0.0f;
        float torque = 0.0f;
        float my_overlap = 0.0f;

        if (i < n) {
            // 4-vertex container wall forces
            float c = s_cos[i] * 0.5f;
            float s = s_sin[i] * 0.5f;
            const float vx[4] = {c - s, -c - s, -c + s, c + s};
            const float vy[4] = {s + c, -s + c, -s - c, s - c};

            for (int k = 0; k < 4; k++) {
                float xk = my_pos_x + vx[k];
                float yk = my_pos_y + vy[k];

                // Left wall: xk < 0
                if (xk < 0.0f) {
                    float pen = -xk;
                    float f = 2.0f * pen;
                    fx += f;
                    torque += -vy[k] * f;
                    my_overlap += pen * pen;
                } else if (xk > current_s) { // Right wall: xk > current_s
                    float pen = xk - current_s;
                    float f = -2.0f * pen;
                    fx += f;
                    torque += -vy[k] * f;
                    my_overlap += pen * pen;
                }

                // Bottom wall: yk < 0
                if (yk < 0.0f) {
                    float pen = -yk;
                    float f = 2.0f * pen;
                    fy += f;
                    torque += vx[k] * f;
                    my_overlap += pen * pen;
                } else if (yk > current_s) { // Top wall: yk > current_s
                    float pen = yk - current_s;
                    float f = -2.0f * pen;
                    fy += f;
                    torque += vx[k] * f;
                    my_overlap += pen * pen;
                }
            }

            // Pairwise SAT repulsion forces against other squares
            float cp = s_cos[i];
            float sp = s_sin[i];

            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                float dx = s_pos_x[j] - my_pos_x;
                float dy = s_pos_y[j] - my_pos_y;
                if (dx * dx + dy * dy >= 2.05f) continue;

                float cq = s_cos[j];
                float sq = s_sin[j];

                float cd = cp * cq + sp * sq;
                float sd = sp * cq - cp * sq;
                float radius = 0.5f + 0.5f * (fabsf(cd) + fabsf(sd));
                float deriv = 0.5f * (-sg_smooth_f(cd) * sd + sg_smooth_f(sd) * cd);

                const float ux[4] = {cp, -sp, cq, -sq};
                const float uy[4] = {sp, cp, sq, cq};

                float gap = -1e10f;
                float gx = 0.0f, gy = 0.0f, ga = 0.0f;
                for (int k = 0; k < 4; k++) {
                    float v = dx * ux[k] + dy * uy[k];
                    float z = fabsf(v) - radius;
                    if (z > gap) {
                        gap = z;
                        float sv = (v >= 0.0f) ? 1.0f : -1.0f;
                        gx = -sv * ux[k];
                        gy = -sv * uy[k];
                        ga = -deriv + (k < 2 ? sv * (-dx * uy[k] + dy * ux[k]) : 0.0f);
                    }
                }

                if (gap < 0.0f) {
                    fx -= 2.0f * gap * gx;
                    fy -= 2.0f * gap * gy;
                    torque -= 2.0f * gap * ga;
                    my_overlap += 0.5f * gap * gap;
                }
            }
        }

        // Dynamic container squeeze: compute total instance overlap with warp shuffle
        float total_overlap = block_sum_overlap(my_overlap, s_reduce, blockDim.x);

        if (threadIdx.x == 0) {
            if (total_overlap < 1e-5f) {
                s_box_s = fmaxf(sqrtf((float)n) + 1e-6f, s_box_s * (1.0f - delta_s));
                s_jam_count = 0;
            } else {
                if (fabsf(total_overlap - s_last_overlap) < 1e-6f || total_overlap >= s_last_overlap) {
                    s_jam_count++;
                } else if (s_jam_count > 0) {
                    s_jam_count--;
                }
            }
            s_last_overlap = total_overlap;
        }
        __syncthreads();

        // Integrate Langevin Heavy-Ball update
        if (i < n) {
            float noise_scale = base_noise;
            if (s_jam_count >= 16) {
                // Apply thermal noise pulse when jammed
                noise_scale = base_noise * 5.0f;
            }

            float noise_x = curand_normal(&rng_state) * noise_scale;
            float noise_y = curand_normal(&rng_state) * noise_scale;
            float noise_theta = curand_normal(&rng_state) * noise_scale;

            constexpr float friction = 0.85f;
            constexpr float alpha = 0.02f;

            my_vel_x = friction * my_vel_x + alpha * fx + noise_x;
            my_vel_y = friction * my_vel_y + alpha * fy + noise_y;
            my_vel_theta = friction * my_vel_theta + alpha * torque + noise_theta;

            // Velocity guard to preserve numerical stability
            my_vel_x = fmaxf(-0.5f, fminf(0.5f, my_vel_x));
            my_vel_y = fmaxf(-0.5f, fminf(0.5f, my_vel_y));
            my_vel_theta = fmaxf(-0.5f, fminf(0.5f, my_vel_theta));

            my_pos_x += my_vel_x;
            my_pos_y += my_vel_y;
            my_theta = wrap_angle_f(my_theta + my_vel_theta);
        }
        __syncthreads();

        if (threadIdx.x == 0 && s_jam_count >= 32) {
            s_jam_count = 8;
        }
        __syncthreads();
    }

    if (i < n) {
        int idx = b * n + i;
        pos_x[idx] = my_pos_x;
        pos_y[idx] = my_pos_y;
        theta[idx] = my_theta;
        vel_x[idx] = my_vel_x;
        vel_y[idx] = my_vel_y;
        vel_theta[idx] = my_vel_theta;
    }
    if (threadIdx.x == 0) {
        box_s[b] = s_box_s;
    }
    if (threadIdx.x < blockDim.x) {
        rng_states[b * blockDim.x + threadIdx.x] = rng_state;
    }
}

__global__ void certify_chains_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ theta,
    int n,
    double* d_cert_L,
    CertPoint* d_best_q,
    double* d_best_L
) {
    int b = blockIdx.x;
    int t = threadIdx.x;

    __shared__ double s_cert_x[MAXN];
    __shared__ double s_cert_y[MAXN];
    __shared__ double s_cert_a[MAXN];
    __shared__ double s_cert_c[MAXN];
    __shared__ double s_cert_s[MAXN];
    __shared__ double s_cert_tmp[32];

    for (int i = t; i < n; i += blockDim.x) {
        s_cert_x[i] = (double)pos_x[b * n + i];
        s_cert_y[i] = (double)pos_y[b * n + i];
        double a = (double)theta[b * n + i];
        s_cert_a[i] = a;
        sincos(a, &s_cert_s[i], &s_cert_c[i]);
    }
    __syncthreads();

    double L_cert = device_certified_side(
        s_cert_x, s_cert_y, s_cert_a, n, s_cert_c, s_cert_s, s_cert_tmp,
        d_best_q + b * n, d_best_L[b]
    );

    if (t == 0) {
        d_cert_L[b] = L_cert;
        if (L_cert < d_best_L[b]) {
            d_best_L[b] = L_cert;
        }
    }
}

static double env_double(const char* name, double def_val, double lo, double hi) {
    const char* val = getenv(name);
    if (!val) return def_val;
    double d = atof(val);
    if (!std::isfinite(d)) return def_val;
    return std::max(lo, std::min(hi, d));
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: langevin seed.txt epochs steps chains random_seed output.json\n";
        return 2;
    }

    int epochs = atoi(argv[2]);
    int steps = atoi(argv[3]);
    int chains = atoi(argv[4]);
    if (epochs < 0 || epochs > 100000 || steps < 0 || steps > 16384 || chains < 0 || chains > 8192) {
        std::cerr << "Invalid epochs, steps, or chains\n";
        return 2;
    }
    unsigned long long rng_seed = strtoull(argv[5], nullptr, 10);

    int n = 0;
    double seed_L = 0.0;
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "Cannot open seed file: " << argv[1] << "\n";
        return 2;
    }
    in >> n >> seed_L;
    if (!in || n < 1 || n > MAXN || !std::isfinite(seed_L) || seed_L <= 0.0) {
        std::cerr << "Invalid seed file contents\n";
        return 2;
    }

    std::vector<CertPoint> seed(n);
    for (int i = 0; i < n; i++) {
        in >> seed[i].x >> seed[i].y >> seed[i].a;
        if (!in || !std::isfinite(seed[i].x) || !std::isfinite(seed[i].y) || !std::isfinite(seed[i].a)) {
            std::cerr << "Invalid coordinates in seed file\n";
            return 2;
        }
    }

    if (epochs == 0) epochs = (n <= 64) ? 50 : (n <= 256) ? 40 : 30;
    if (steps == 0)  steps  = (n <= 64) ? 64 : (n <= 256) ? 48 : 32;
    if (chains == 0) chains = 64;
    if (chains % 8 != 0) chains = ((chains + 7) / 8) * 8;

    float delta_s = (float)env_double("LANGEVIN_DELTA_S", 0.001, 1e-6, 0.1);
    float base_noise = (float)env_double("LANGEVIN_NOISE", 0.001, 1e-6, 0.1);

    int block_threads = 32;
    while (block_threads < n && block_threads < 1024) {
        block_threads <<= 1;
    }
    block_threads = std::min(1024, std::max(32, block_threads));

    // Structure of Arrays (SoA) layout
    float *d_pos_x = nullptr, *d_pos_y = nullptr, *d_theta = nullptr;
    float *d_vel_x = nullptr, *d_vel_y = nullptr, *d_vel_theta = nullptr;
    float *d_box_s = nullptr;

    size_t total_elements = (size_t)chains * n;
    CK(cudaMalloc(&d_pos_x, total_elements * sizeof(float)));
    CK(cudaMalloc(&d_pos_y, total_elements * sizeof(float)));
    CK(cudaMalloc(&d_theta, total_elements * sizeof(float)));
    CK(cudaMalloc(&d_vel_x, total_elements * sizeof(float)));
    CK(cudaMalloc(&d_vel_y, total_elements * sizeof(float)));
    CK(cudaMalloc(&d_vel_theta, total_elements * sizeof(float)));
    CK(cudaMalloc(&d_box_s, chains * sizeof(float)));

    // Certification and state arrays
    CertPoint* d_best_q = nullptr;
    double *d_cert_L = nullptr, *d_best_L = nullptr;
    curandStatePhilox4_32_10_t* d_rng = nullptr;

    CK(cudaMalloc(&d_best_q, total_elements * sizeof(CertPoint)));
    CK(cudaMalloc(&d_cert_L, chains * sizeof(double)));
    CK(cudaMalloc(&d_best_L, chains * sizeof(double)));
    CK(cudaMalloc(&d_rng, (size_t)chains * block_threads * sizeof(curandStatePhilox4_32_10_t)));

    // Initialize RNG
    int total_rng_threads = chains * block_threads;
    init_rng_kernel<<<(total_rng_threads + 255) / 256, 256>>>(d_rng, total_rng_threads, rng_seed);
    CK(cudaGetLastError());

    // Host portfolio initialization
    std::vector<CertPoint> h_init;
    sqfam::build_portfolio_generic(n, chains, seed_L, seed, h_init);

    std::vector<float> h_pos_x(total_elements);
    std::vector<float> h_pos_y(total_elements);
    std::vector<float> h_theta(total_elements);
    std::vector<float> h_vel_x(total_elements, 0.0f);
    std::vector<float> h_vel_y(total_elements, 0.0f);
    std::vector<float> h_vel_theta(total_elements, 0.0f);
    std::vector<float> h_box_s(chains, (float)seed_L);

    for (int b = 0; b < chains; ++b) {
        for (int i = 0; i < n; ++i) {
            int idx = b * n + i;
            h_pos_x[idx] = (float)h_init[idx].x;
            h_pos_y[idx] = (float)h_init[idx].y;
            h_theta[idx] = (float)h_init[idx].a;
        }
    }

    CK(cudaMemcpy(d_pos_x, h_pos_x.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pos_y, h_pos_y.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_theta, h_theta.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_vel_x, h_vel_x.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_vel_y, h_vel_y.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_vel_theta, h_vel_theta.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_box_s, h_box_s.data(), chains * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<double> h_best_L(chains, 1e9);
    CK(cudaMemcpy(d_best_L, h_best_L.data(), chains * sizeof(double), cudaMemcpyHostToDevice));

    // Initial certification pass
    certify_chains_kernel<<<chains, block_threads>>>(
        d_pos_x, d_pos_y, d_theta, n, d_cert_L, d_best_q, d_best_L
    );
    CK(cudaGetLastError());

    std::vector<double> h_cert_L(chains);
    CK(cudaMemcpy(h_cert_L.data(), d_cert_L, chains * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<CertPoint> best_q = seed;
    double bestL = seed_L;
    int shrinks = 0;

    for (int b = 0; b < chains; b++) {
        if (h_cert_L[b] < bestL) {
            bestL = h_cert_L[b];
            CK(cudaMemcpy(best_q.data(), d_best_q + b * n, n * sizeof(CertPoint), cudaMemcpyDeviceToHost));
        }
    }

    auto start_time = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < epochs; epoch++) {
        evaluate_forces_and_step<<<chains, block_threads>>>(
            d_pos_x, d_pos_y, d_theta,
            d_vel_x, d_vel_y, d_vel_theta,
            d_box_s, d_rng, n, steps,
            delta_s, base_noise
        );
        CK(cudaGetLastError());

        certify_chains_kernel<<<chains, block_threads>>>(
            d_pos_x, d_pos_y, d_theta, n, d_cert_L, d_best_q, d_best_L
        );
        CK(cudaGetLastError());

        CK(cudaMemcpy(h_cert_L.data(), d_cert_L, chains * sizeof(double), cudaMemcpyDeviceToHost));

        int best_chain = -1;
        for (int b = 0; b < chains; b++) {
            if (h_cert_L[b] < bestL - 1e-11) {
                bestL = h_cert_L[b];
                best_chain = b;
            }
        }

        if (best_chain >= 0) {
            shrinks++;
            CK(cudaMemcpy(best_q.data(), d_best_q + best_chain * n, n * sizeof(CertPoint), cudaMemcpyDeviceToHost));

            // Rescale ladder chains toward new certified record
            const double mults[8] = {0.02, 0.1, 0.4, 1.0, 2.5, 5.0, 10.0, 20.0};
            for (int b = 0; b < chains; b++) {
                if (b % 8 == 0) {
                    double target_s = std::max(std::sqrt((double)n) + 1e-6, bestL - delta_s * mults[b % 8]);
                    double scale = target_s / bestL;
                    for (int i = 0; i < n; i++) {
                        int idx = b * n + i;
                        h_pos_x[idx] = (float)(best_q[i].x * scale);
                        h_pos_y[idx] = (float)(best_q[i].y * scale);
                        h_theta[idx] = (float)best_q[i].a;
                        h_vel_x[idx] = 0.0f;
                        h_vel_y[idx] = 0.0f;
                        h_vel_theta[idx] = 0.0f;
                    }
                    h_box_s[b] = (float)target_s;
                }
            }
            CK(cudaMemcpy(d_pos_x, h_pos_x.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_pos_y, h_pos_y.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_theta, h_theta.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_vel_x, h_vel_x.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_vel_y, h_vel_y.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_vel_theta, h_vel_theta.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_box_s, h_box_s.data(), chains * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    CK(cudaDeviceSynchronize());
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    unsigned long long total_proposals = (unsigned long long)chains * epochs * steps;

    std::ofstream out(argv[6]);
    if (!out) {
        std::cerr << "Failed to open output JSON: " << argv[6] << "\n";
        return 2;
    }

    out << std::setprecision(17)
        << "{\"n\":" << n
        << ",\"target_L\":" << bestL
        << ",\"seconds\":" << seconds
        << ",\"proposals\":" << total_proposals
        << ",\"chains\":" << chains
        << ",\"epochs\":" << epochs
        << ",\"steps\":" << steps
        << ",\"shrinks\":" << shrinks
        << ",\"q\":[";
    for (size_t i = 0; i < best_q.size(); i++) {
        if (i > 0) out << ",";
        out << "[" << best_q[i].x << "," << best_q[i].y << "," << best_q[i].a << "]";
    }
    out << "]}\n";

    CK(cudaFree(d_pos_x));
    CK(cudaFree(d_pos_y));
    CK(cudaFree(d_theta));
    CK(cudaFree(d_vel_x));
    CK(cudaFree(d_vel_y));
    CK(cudaFree(d_vel_theta));
    CK(cudaFree(d_box_s));
    CK(cudaFree(d_best_q));
    CK(cudaFree(d_cert_L));
    CK(cudaFree(d_best_L));
    CK(cudaFree(d_rng));

    return out ? 0 : 2;
}
