// MIT. Float search only; Python's official interval certifier owns acceptance.
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<"\n";exit(2);} }while(0)
#ifndef PACK_THREADS
#define PACK_THREADS 64
#endif
constexpr int NMAX=50, NT=PACK_THREADS, CELLS=256, LADDER=8;
static_assert(NT==32||NT==64||NT==128||NT==256,"power-of-two thread count");
struct Settings {float tmin,tmax,move,rotation;};
__constant__ Settings settings;
struct Q {float x,y,a;};
__device__ int cell(Q p) {
 int x=max(0,min(15,(int)floorf(p.x/2))), y=max(0,min(15,(int)floorf(p.y/2)));
 return y*16+x;
}
__device__ float depth(Q p,Q q) {
 float c=cosf(p.a),s=sinf(p.a),d=cosf(q.a),t=sinf(q.a);
 float ux[4]={c,-s,d,-t},uy[4]={s,c,t,d}, z=1e10f;
 for(int k=0;k<4;k++){
 float r=.5f*(fabsf(ux[k]*c+uy[k]*s)+fabsf(-ux[k]*s+uy[k]*c)
                  +fabsf(ux[k]*d+uy[k]*t)+fabsf(-ux[k]*t+uy[k]*d));
 z=fminf(z,r-fabsf((p.x-q.x)*ux[k]+(p.y-q.y)*uy[k]));
 } return fmaxf(0,z);
}
__device__ float wall(Q p,float L) {
 float e=.5f*(fabsf(cosf(p.a))+fabsf(sinf(p.a)));
 float x=fmaxf(0,fabsf(p.x-L/2)+e-L/2), y=fmaxf(0,fabsf(p.y-L/2)+e-L/2);
 return x*x+y*y;
}
__device__ float local(Q p,int i,Q* q,int* head,int* next,float L) {
 float e=wall(p,L); int c=cell(p),x=c%16,y=c/16;
 // Cell width 2 > maximum interaction radius sqrt(2).
 for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
 int xx=x+dx,yy=y+dy;if(xx<0||xx>=16||yy<0||yy>=16)continue;
 for(int j=head[yy*16+xx];j!=-1;j=next[j]) if(j!=i){float d=depth(p,q[j]); e+=d*d;}
 } return e;
}
__global__ void init_rng(curandStatePhilox4_32_10_t* r,int chains,unsigned long long seed){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<chains*NT)curand_init(seed,i,0,r+i);
}
__global__ void walk(Q* all,float* energies,curandStatePhilox4_32_10_t* rng,
                     int n,float L,int steps,float cooling,unsigned long long* counts){
 __shared__ Q q[NMAX], proposed[NMAX];
 __shared__ int head[CELLS],next[NMAX],selected[NMAX],offset;
 __shared__ float sum[NT];
 int t=threadIdx.x,b=blockIdx.x;
 auto r=rng[b*NT+t]; unsigned long long count=0;
 if(t<n)q[t]=all[b*n+t]; __syncthreads();
 float temp=settings.tmin*powf(settings.tmax/settings.tmin,(b%LADDER)/float(LADDER-1))*cooling;
 for(int step=0;step<steps;step++){
  for(int j=t;j<CELLS;j+=NT)head[j]=-1;
  if(t==0)offset=curand(&r)%n; __syncthreads();
  if(t<n)next[t]=atomicExch(head+cell(q[t]),t); __syncthreads();
  // Conservative independent set. Max translational move per coordinate .12,
  // so centers >3.3 apart cannot interact before OR after either move.
  if(t<n){
   bool take=true; int rank=(t+offset)%n;
   for(int j=0;j<n;j++)if(j!=t&&(j+offset)%n<rank){
    float dx=q[t].x-q[j].x,dy=q[t].y-q[j].y;
    if(dx*dx+dy*dy<3.3f*3.3f)take=false;
   }
   selected[t]=take;
   proposed[t]=q[t];
   if(take){
    float scale=(step%8==0?.12f:settings.move)*sqrtf(cooling);
    Q p=q[t];p.x+=scale*(2*curand_uniform(&r)-1);p.y+=scale*(2*curand_uniform(&r)-1);
    p.a+=(step%8==0?settings.rotation:scale)*(2*curand_uniform(&r)-1);p.a=remainderf(p.a,1.57079632679f);
    float delta=local(p,t,q,head,next,L)-local(q[t],t,q,head,next,L);
    if(delta<=0||curand_uniform(&r)<expf(-delta/temp))proposed[t]=p;
    count++;
   }
  }
  __syncthreads();
  if(t<n&&selected[t])q[t]=proposed[t];
  __syncthreads();
 }
 // Independent all-pairs checkpoint, not the neighbor filter.
 float e=0;
 if(t<n){e=wall(q[t],L);for(int j=t+1;j<n;j++){float d=depth(q[t],q[j]);e+=d*d;}all[b*n+t]=q[t];}
 sum[t]=e; __syncthreads();
 for(int stride=NT/2;stride;stride/=2){if(t<stride)sum[t]+=sum[t+stride];__syncthreads();}
 if(t==0)energies[b]=sum[0];
 rng[b*NT+t]=r; atomicAdd(counts,count);
}

// Large-n path: immutable global snapshot, pair-parallel single-square moves.
// No n-sized shared allocation. All reads finish before the sole coordinate write.
__global__ void walk_global(Q* all,float* energies,curandStatePhilox4_32_10_t* rng,
                           int n,float L,int steps,float cooling,unsigned long long* counts){
 __shared__ Q oldp,proposal;
 __shared__ int chosen,accept;
 __shared__ float partial[NT];
 int t=threadIdx.x,b=blockIdx.x; Q* q=all+b*n;
 auto r=rng[b*NT+t];
 float temp=settings.tmin*powf(settings.tmax/settings.tmin,(b%LADDER)/float(LADDER-1))*cooling;
 for(int step=0;step<steps;step++){
  if(t==0){
   chosen=curand(&r)%n;oldp=q[chosen];proposal=oldp;
   float scale=(step%8==0?.12f:settings.move)*sqrtf(cooling);
   proposal.x+=scale*(2*curand_uniform(&r)-1);
   proposal.y+=scale*(2*curand_uniform(&r)-1);
   proposal.a=remainderf(proposal.a+scale*(2*curand_uniform(&r)-1),1.57079632679f);
  } __syncthreads();
  float delta=t==0?wall(proposal,L)-wall(oldp,L):0;
  for(int j=t;j<n;j+=NT)if(j!=chosen){
   float dx=q[j].x-oldp.x,dy=q[j].y-oldp.y;
   if(dx*dx+dy*dy>9)continue;
   float a=depth(proposal,q[j]),c=depth(oldp,q[j]);delta+=a*a-c*c;
  }
  partial[t]=delta;__syncthreads();
  for(int stride=NT/2;stride;stride/=2){if(t<stride)partial[t]+=partial[t+stride];__syncthreads();}
  if(t==0){accept=partial[0]<=0||curand_uniform(&r)<expf(-partial[0]/temp);}
  __syncthreads();
  if(t==0&&accept)q[chosen]=proposal;
  __syncthreads();
 }
 float e=0;
 for(int i=t;i<n;i+=NT){
  e+=wall(q[i],L);
  for(int j=i+1;j<n;j++){
   float dx=q[j].x-q[i].x,dy=q[j].y-q[i].y;
   if(dx*dx+dy*dy<=2.000001f){float a=depth(q[i],q[j]);e+=a*a;}
  }
 }
 partial[t]=e;__syncthreads();
 for(int stride=NT/2;stride;stride/=2){if(t<stride)partial[t]+=partial[t+stride];__syncthreads();}
 if(t==0){energies[b]=partial[0];atomicAdd(counts,(unsigned long long)steps);}
 rng[b*NT+t]=r;
}
__global__ void exchange(Q* q,float* e,curandStatePhilox4_32_10_t* rng,int n,int chains,int parity,float cooling){
 int a=blockIdx.x*2+parity,b=a+1;
 if(b>=chains||a/LADDER!=b/LADDER)return;
 __shared__ int accept;
 if(threadIdx.x==0){
  auto r=rng[a*NT];float ta=settings.tmin*powf(settings.tmax/settings.tmin,(a%LADDER)/float(LADDER-1))*cooling;
  float tb=settings.tmin*powf(settings.tmax/settings.tmin,(b%LADDER)/float(LADDER-1))*cooling;
  accept=logf(curand_uniform(&r))<(1/ta-1/tb)*(e[a]-e[b]);
  rng[a*NT]=r;if(accept){float z=e[a];e[a]=e[b];e[b]=z;}
 } __syncthreads();
 if(accept)for(int i=threadIdx.x;i<n;i+=NT){Q z=q[a*n+i];q[a*n+i]=q[b*n+i];q[b*n+i]=z;}
}
float envf(const char* name,float fallback,float low,float high){
 const char* s=getenv(name);if(!s)return fallback;
 char* end;float v=strtof(s,&end);if(*end||!std::isfinite(v)||v<low||v>high){std::cerr<<"invalid "<<name<<"\n";exit(2);}return v;
}

namespace dpack {
constexpr int MAXN_D = 1000;
struct P { double x, y, a; };
struct G { double x, y, a; };

__host__ __device__ inline double sg(double x) { return (x > 0) - (x < 0); }
__host__ __device__ inline double sg_smooth(double x, double eps = 1e-6) {
    return (fabs(x) > eps) ? ((x >= 0.0) ? 1.0 : -1.0) : 0.0;
}
__host__ __device__ inline double wrap_angle(double a) {
    constexpr double pi2 = 1.57079632679489661923;
    a = fmod(a, pi2);
    if (a < 0.0) a += pi2;
    return a;
}
__host__ __device__ inline double dr_smooth(double c, double s) {
    return .5 * (-sg_smooth(c) * s + sg_smooth(s) * c);
}

__host__ __device__ double pair_cached(P p, P q, double c, double s, double d, double t) {
    double dx = q.x - p.x, dy = q.y - p.y;
    if (dx * dx + dy * dy >= 2.000001) return 0.0;
    double cd = c * d + s * t, sd = s * d - c * t;
    double radius = .5 + .5 * (fabs(cd) + fabs(sd));
    double ux[4] = {c, -s, d, -t}, uy[4] = {s, c, t, d};
    double gap = -1e100;
    for (int k = 0; k < 4; k++) {
        double v = dx * ux[k] + dy * uy[k], z = fabs(v) - radius;
        if (z > gap) gap = z;
    }
    return (gap >= 0.0) ? 0.0 : gap * gap;
}

__host__ __device__ double pair_cached_smooth(P p, P q, double c, double s, double d, double t, G* grad) {
    double dx = q.x - p.x, dy = q.y - p.y;
    if (dx * dx + dy * dy >= 2.000001) return 0.0;
    double cd = c * d + s * t, sd = s * d - c * t;
    double radius = .5 + .5 * (fabs(cd) + fabs(sd));
    double deriv = .5 * (-sg_smooth(cd) * sd + sg_smooth(sd) * cd);
    double ux[4] = {c, -s, d, -t}, uy[4] = {s, c, t, d};
    double gap = -1e100, gx = 0, gy = 0, ga = 0;
    for (int k = 0; k < 4; k++) {
        double v = dx * ux[k] + dy * uy[k], z = fabs(v) - radius;
        if (z > gap) {
            gap = z;
            double sv = (v >= 0.0) ? 1.0 : -1.0;
            gx = -sv * ux[k];
            gy = -sv * uy[k];
            ga = -deriv + (k < 2 ? sv * (-dx * uy[k] + dy * ux[k]) : 0.0);
        }
    }
    if (gap >= 0.0) return 0.0;
    grad->x += 2.0 * gap * gx;
    grad->y += 2.0 * gap * gy;
    grad->a += 2.0 * gap * ga;
    return gap * gap;
}

__host__ __device__ double boundary(P p, double L) {
    double c = cos(p.a), s = sin(p.a);
    double r = .5 * (fabs(c) + fabs(s));
    double v[4] = {r - p.x, p.x + r - L, r - p.y, p.y + r - L}, e = 0;
    for (int k = 0; k < 4; k++) if (v[k] > 0) e += v[k] * v[k];
    return e;
}

__device__ double block_sum(double val, double* tmp) {
    int t = threadIdx.x; tmp[t] = val; __syncthreads();
    for (int s = NT / 2; s > 0; s >>= 1) { if (t < s) tmp[t] += tmp[t + s]; __syncthreads(); }
    return tmp[0];
}
__device__ double block_max(double val, double* tmp) {
    int t = threadIdx.x; tmp[t] = val; __syncthreads();
    for (int s = NT / 2; s > 0; s >>= 1) { if (t < s && tmp[t + s] > tmp[t]) tmp[t] = tmp[t + s]; __syncthreads(); }
    return tmp[0];
}
__device__ double block_min(double val, double* tmp) {
    int t = threadIdx.x; tmp[t] = val; __syncthreads();
    for (int s = NT / 2; s > 0; s >>= 1) { if (t < s && tmp[t + s] < tmp[t]) tmp[t] = tmp[t + s]; __syncthreads(); }
    return tmp[0];
}

__device__ double total_cached(const P* q, int n, double L, const double* c, const double* s, double* tmp) {
    int t = threadIdx.x;
    double e = 0.0;
    for (int i = t; i < n; i += NT) {
        double r = 0.5 * (fabs(c[i]) + fabs(s[i]));
        if (q[i].x < r) { double u = r - q[i].x; e += u * u; }
        else if (q[i].x > L - r) { double u = q[i].x + r - L; e += u * u; }
        if (q[i].y < r) { double u = r - q[i].y; e += u * u; }
        else if (q[i].y > L - r) { double u = q[i].y + r - L; e += u * u; }
        for (int j = i + 1; j < n; j++) e += pair_cached(q[i], q[j], c[i], s[i], c[j], s[j]);
    }
    return block_sum(e, tmp);
}

__device__ double device_certified_side(
    const P* q, int n, const double* c, const double* s, double* tmp,
    P* best_out, double current_best_L
) {
    int t = threadIdx.x;
    constexpr double tau = 2.6e-8;
    double local_lam = 1.0;
    for (int i = t; i < n; i += NT) {
        double xi = q[i].x, yi = q[i].y, ci = c[i], si = s[i];
        for (int j = i + 1; j < n; j++) {
            double dx = xi - q[j].x, dy = yi - q[j].y;
            if (dx * dx + dy * dy >= 2.05) continue;
            double cj = c[j], sj = s[j];
            double r = 0.5 + 0.5 * (fabs(ci * cj + si * sj) + fabs(-si * cj + ci * sj));
            double v1 = fabs(dx * ci + dy * si), v2 = fabs(-dx * si + dy * ci);
            double v3 = fabs(dx * cj + dy * sj), v4 = fabs(-dx * sj + dy * cj);
            double vmax = fmax(fmax(v1, v2), fmax(v3, v4));
            double req = (vmax > 1e-12) ? ((r + tau) / vmax) : 1e9;
            if (req > local_lam) local_lam = req;
        }
    }
    double lam = block_max(local_lam, tmp);
    if (lam > 1.008) return 1e9;

    double my_min_x = 1e9, my_max_x = -1e9, my_min_y = 1e9, my_max_y = -1e9;
    for (int i = t; i < n; i += NT) {
        double x = q[i].x * lam, y = q[i].y * lam;
        double r = 0.5 * (fabs(c[i]) + fabs(s[i]));
        if (x - r < my_min_x) my_min_x = x - r;
        if (x + r > my_max_x) my_max_x = x + r;
        if (y - r < my_min_y) my_min_y = y - r;
        if (y + r > my_max_y) my_max_y = y + r;
    }
    double min_x = block_min(my_min_x, tmp), max_x = block_max(my_max_x, tmp);
    double min_y = block_min(my_min_y, tmp), max_y = block_max(my_max_y, tmp);
    double L_cert = fmax(max_x - min_x, max_y - min_y);
    if (L_cert < current_best_L && best_out != nullptr) {
        for (int i = t; i < n; i += NT) {
            best_out[i] = {q[i].x * lam - min_x, q[i].y * lam - min_y, wrap_angle(q[i].a)};
        }
        __syncthreads();
    }
    return L_cert;
}

__global__ void random_init(curandStatePhilox4_32_10_t* r, int chains, unsigned long long seed) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < chains) curand_init(seed, b, 0, r + b);
}

static void build_portfolio(int n, int chains, double seed_L, const std::vector<P>& seed, std::vector<P>& h_init) {
    h_init.resize(n * chains);
    for (int b = 0; b < chains; ++b) for (int i = 0; i < n; ++i) h_init[b * n + i] = seed[i];
    if (n <= 1) return;
    const double s2 = std::sqrt(2.0), inv_s2 = 1.0 / s2, eps = 1e-7, pi4 = std::acos(-1.0) * 0.25;
    struct TopoCand { double L; std::vector<P> pts; };
    std::vector<TopoCand> cands;
    for (int w = 1; w <= 24; ++w) {
        for (int a1 = 0; a1 <= 32; ++a1) {
            for (int a2 = a1; a2 <= 32; ++a2) {
                double L = 0.5 * (a1 + a2) + 1.0 + w * inv_s2 + (w + 4) * eps;
                if (L < std::sqrt((double)n) - 0.05 || L > seed_L + 0.15) continue;
                int b_lim = (int)std::floor(L - std::max(a1, a2));
                for (int b1 = 0; b1 <= std::max(0, b_lim); ++b1) {
                    for (int b2 = 0; b2 <= b1; ++b2) {
                        int corner_cnt = a1 * (a1 + 1) / 2 + a2 * (a2 + 1) / 2 + b1 * (b1 + 1) / 2 + b2 * (b2 + 1) / 2;
                        if (corner_cnt > n + 3) continue;
                        double u_start = (a1 + 1.0) * inv_s2 + 0.5 + 2 * eps;
                        int til_cnt = 0;
                        for (int k = 0; k < w; ++k) {
                            double u = u_start + k * (1.0 + eps);
                            double v_lo = -std::min(u - 1.0 - eps, s2 * L - u - 1.0 - eps);
                            double v_hi =  std::min(u - 1.0 - eps, s2 * L - u - 1.0 - eps);
                            if (b1 > 0) v_hi = std::min(v_hi, (L - b1 - 1.0) * inv_s2 - 0.5 - 2 * eps);
                            if (b2 > 0) v_lo = std::max(v_lo, -(L - b2 - 1.0) * inv_s2 + 0.5 + 2 * eps);
                            if (v_hi >= v_lo) til_cnt += (int)std::floor((v_hi - v_lo) / (1.0 + eps)) + 1;
                        }
                        int cap = corner_cnt + til_cnt;
                        if (cap < n || cap > n + 4) continue;
                        std::vector<P> corners, tilted;
                        for (int i = 0; i < a1; ++i) for (int j = 0; i + j < a1; ++j) corners.push_back({0.5 + i + eps, 0.5 + j + eps, 0.0});
                        for (int i = 0; i < a2; ++i) for (int j = 0; i + j < a2; ++j) corners.push_back({L - 0.5 - i - eps, L - 0.5 - j - eps, 0.0});
                        for (int i = 0; i < b1; ++i) for (int j = 0; i + j < b1; ++j) corners.push_back({0.5 + i + eps, L - 0.5 - j - eps, 0.0});
                        for (int i = 0; i < b2; ++i) for (int j = 0; i + j < b2; ++j) corners.push_back({L - 0.5 - i - eps, 0.5 + j + eps, 0.0});
                        for (int k = 0; k < w; ++k) {
                            double u = u_start + k * (1.0 + eps);
                            double v_lo = -std::min(u - 1.0 - eps, s2 * L - u - 1.0 - eps);
                            double v_hi =  std::min(u - 1.0 - eps, s2 * L - u - 1.0 - eps);
                            if (b1 > 0) v_hi = std::min(v_hi, (L - b1 - 1.0) * inv_s2 - 0.5 - 2 * eps);
                            if (b2 > 0) v_lo = std::max(v_lo, -(L - b2 - 1.0) * inv_s2 + 0.5 + 2 * eps);
                            if (v_hi >= v_lo) {
                                int cnt = (int)std::floor((v_hi - v_lo) / (1.0 + eps)) + 1;
                                double span = (cnt - 1) * (1.0 + eps), v0 = 0.5 * (v_lo + v_hi) - 0.5 * span;
                                for (int m = 0; m < cnt; ++m) {
                                    double v = v0 + m * (1.0 + eps);
                                    tilted.push_back({(u - v) * inv_s2, (u + v) * inv_s2, pi4});
                                }
                            }
                        }
                        {
                            std::vector<P> pts = corners;
                            for (int i = 0; i < (int)tilted.size() && (int)pts.size() < n; ++i) pts.push_back(tilted[i]);
                            pts.resize(n); cands.push_back({L, pts});
                        }
                        if (cap > n && !corners.empty()) {
                            std::vector<P> pts = tilted;
                            for (int i = 0; i < (int)corners.size() && (int)pts.size() < n; ++i) pts.push_back(corners[i]);
                            pts.resize(n); cands.push_back({L, pts});
                        }
                    }
                }
                {
                    double u_start = (a1 + 1.0) * inv_s2 + 0.5 + 2 * eps;
                    std::vector<P> pts;
                    for (int i = 0; i < a1; ++i) for (int j = 0; i + j < a1; ++j) pts.push_back({0.5 + i + eps, 0.5 + j + eps, 0.0});
                    for (int i = 0; i < a2; ++i) for (int j = 0; i + j < a2; ++j) pts.push_back({L - 0.5 - i - eps, L - 0.5 - j - eps, 0.0});
                    for (int k = 0; k < w; ++k) {
                        double u = u_start + k * (1.0 + eps);
                        double v_lo = -std::min(u - 1.0 - eps, s2 * L - u - 1.0 - eps);
                        double v_hi =  std::min(u - 1.0 - eps, s2 * L - u - 1.0 - eps);
                        if (v_hi >= v_lo) {
                            int cnt = (int)std::floor((v_hi - v_lo) / (1.0 + eps)) + 1;
                            double span = (cnt - 1) * (1.0 + eps), v0 = 0.5 * (v_lo + v_hi) - 0.5 * span;
                            for (int m = 0; m < cnt; ++m) {
                                double v = v0 + m * (1.0 + eps);
                                pts.push_back({(u - v) * inv_s2, (u + v) * inv_s2, pi4});
                            }
                        }
                    }
                    if ((int)pts.size() == n - 1 && L > std::max(a1, a2) + 1.0) {
                        pts.push_back({0.5 + eps, L - 0.5 - eps, 0.0});
                        cands.push_back({L + 0.005, pts});
                    } else if ((int)pts.size() == n - 2 && L > std::max(a1, a2) + 1.0) {
                        pts.push_back({0.5 + eps, L - 0.5 - eps, 0.0});
                        pts.push_back({L - 0.5 - eps, 0.5 + eps, 0.0});
                        cands.push_back({L + 0.01, pts});
                    }
                }
            }
        }
    }
    std::sort(cands.begin(), cands.end(), [](const TopoCand& a, const TopoCand& b) { return a.L < b.L; });
    if (cands.size() > 64) cands.resize(64);
    int k_floor = (int)std::floor(std::sqrt((double)n));
    for (int b = 0; b < chains; ++b) {
        int ladder = b / 8, group = b % 8;
        if (group <= 1) {
            if (ladder >= 2 && !cands.empty()) {
                const auto& tc = cands[(ladder - 2) % cands.size()];
                for (int i = 0; i < n; ++i) h_init[b * n + i] = tc.pts[i];
            }
        } else if (group <= 4 && !cands.empty()) {
            const auto& tc = cands[ladder % cands.size()];
            for (int i = 0; i < n; ++i) h_init[b * n + i] = tc.pts[i];
        } else if (group == 5 || group == 6) {
            int cx = std::max(1, k_floor - 1 - (ladder % 2));
            int cy = std::max(1, k_floor - 1 - ((ladder / 2) % 2));
            bool cut_corner = (group == 6 && cx >= 2 && cy >= 2);
            double L_box = std::max(std::sqrt((double)n) + 0.05, seed_L - 0.03);
            std::vector<P> pts;
            for (int i = 0; i < cx && (int)pts.size() < n; ++i)
                for (int j = 0; j < cy && (int)pts.size() < n; ++j)
                    if (!(cut_corner && i == cx - 1 && j == cy - 1)) pts.push_back({0.5 + i, 0.5 + j, 0.0});
            if ((ladder % 3) != 0 && (int)pts.size() < n) pts.push_back({L_box - 0.5, L_box - 0.5, 0.0});
            if ((ladder % 3) == 1 && (int)pts.size() < n) {
                pts.push_back({0.5, L_box - 0.5, 0.0});
                if ((int)pts.size() < n) pts.push_back({L_box - 0.5, 0.5, 0.0});
            }
            double th = 0.25 + 0.08 * (ladder % 7);
            int rem = n - (int)pts.size(), n_right = (rem + 1) / 2, n_top = rem - n_right;
            for (int k = 0; k < n_right; ++k) {
                double y = 0.7 + (L_box - 1.4) * (k + 0.5) / std::max(1, n_right);
                double x = 0.5 * (cx + L_box) + ((k % 2 == 0) ? -0.18 : 0.18);
                pts.push_back({std::clamp(x, 0.6, L_box - 0.6), std::clamp(y, 0.6, L_box - 0.6), th});
            }
            for (int k = 0; k < n_top; ++k) {
                double x = 0.7 + (L_box - 1.4) * (k + 0.5) / std::max(1, n_top);
                double y = 0.5 * (cy + L_box) + ((k % 2 == 0) ? -0.18 : 0.18);
                pts.push_back({std::clamp(x, 0.6, L_box - 0.6), std::clamp(y, 0.6, L_box - 0.6), th});
            }
            for (int i = 0; i < n; ++i) h_init[b * n + i] = pts[i];
        } else {
            double L_box = std::max(std::sqrt((double)n) + 0.05, seed_L - 0.02);
            std::vector<P> pts;
            pts.push_back({0.5, 0.5, 0.0});
            if (n > 1) pts.push_back({L_box - 0.5, 0.5, 0.0});
            if (n > 2) pts.push_back({0.5, L_box - 0.5, 0.0});
            if (n > 3) pts.push_back({L_box - 0.5, L_box - 0.5, 0.0});
            int rem = n - (int)pts.size(), cols = std::max(1, (int)std::ceil(std::sqrt((double)rem)));
            double th = 0.35 + 0.07 * (ladder % 6);
            for (int k = 0; k < rem; ++k) {
                double x = 0.8 + (L_box - 1.6) * ((k % cols) + 0.5) / cols;
                double y = 0.8 + (L_box - 1.6) * ((k / cols) + 0.5) / cols;
                pts.push_back({x, y, th});
            }
            for (int i = 0; i < n; ++i) h_init[b * n + i] = pts[i];
        }
    }
}

__global__ void structured_init(P* all, curandStatePhilox4_32_10_t* rng, int n, int chains, double L) {
    int b = blockIdx.x, t = threadIdx.x;
    if (b >= chains) return;
    __shared__ double jitter;
    if (t == 0) {
        auto r = rng[b];
        int group = b % 8;
        jitter = (group == 0) ? 0.0 : (group == 1) ? 1e-4 * pow(100.0, curand_uniform_double(&r)) : 0.03;
        rng[b] = r;
    }
    __syncthreads();
    if (jitter == 0.0) return;
    auto r = rng[b];
    skipahead((unsigned long long)(t + 1) * 4096, &r);
    for (int i = t; i < n; i += blockDim.x) {
        P p = all[b * n + i];
        p.x = fmin(L - 0.5, fmax(0.5, p.x + jitter * (2.0 * curand_uniform_double(&r) - 1.0)));
        p.y = fmin(L - 0.5, fmax(0.5, p.y + jitter * (2.0 * curand_uniform_double(&r) - 1.0)));
        p.a = wrap_angle(p.a + jitter * (2.0 * curand_uniform_double(&r) - 1.0));
        all[b * n + i] = p;
    }
}

template <int KPT>
__global__ void search_kernel(
    P* all, P* best_all, double* energies, double* best_Ls,
    curandStatePhilox4_32_10_t* rng, int n, double baseL, int steps, int epoch,
    int polish, double tmin, double tmax, double baseMove, double shrink,
    unsigned long long* stats
) {
    constexpr int SHM_N = (KPT * NT < MAXN_D) ? (KPT * NT) : MAXN_D;
    __shared__ P q[SHM_N];
    __shared__ double cachedc[SHM_N], cacheds[SHM_N];
    __shared__ double tmp[NT];
    __shared__ double energy, targetL, move, rot, cx, cy, theta, scale, adapt, step_cg;
    __shared__ int chosen, mode, accept, windowAccepted, worst_sq;
    __shared__ P old_chosen, prop_chosen;
    __shared__ curandStatePhilox4_32_10_t r;

    int t = threadIdx.x, b = blockIdx.x;
    for (int i = t; i < n; i += NT) {
        q[i] = all[b * n + i];
        cachedc[i] = cos(q[i].a);
        cacheds[i] = sin(q[i].a);
    }
    if (t == 0) {
        r = rng[b];
        adapt = 1.0;
        windowAccepted = 0;
        const double mults[8] = {0.02, 0.1, 0.4, 1.0, 2.5, 5.0, 10.0, 20.0};
        targetL = fmax(sqrt((double)n) + 1e-6, baseL - shrink * mults[b % LADDER]);
    }
    __syncthreads();

    double init_cert = device_certified_side(q, n, cachedc, cacheds, tmp, best_all + b * n, best_Ls[b]);
    if (t == 0 && init_cert < best_Ls[b]) { best_Ls[b] = init_cert; atomicAdd(stats + 2, 1ULL); }
    __syncthreads();

    double e = total_cached(q, n, targetL, cachedc, cacheds, tmp);
    if (t == 0) energy = e;
    __syncthreads();

    double cool = pow(.002, double(epoch % 200) / 199.0);
    double temp = tmin * pow(tmax / tmin, (b % LADDER) / 7.0) * cool;

    P my_p[KPT];
    double my_c[KPT], my_s[KPT];

    for (int step = 0; step < steps; step++) {
        if (t == 0) {
            chosen = curand(&r) % n;
            mode = curand(&r) % 7;
            cx = q[chosen].x; cy = q[chosen].y;
            scale = adapt * baseMove * (.15 + 1.85 * (b % LADDER) / 7.0) * sqrt(cool);
            if (step % 16 == 0) scale *= 3.5;
            move = scale * (2.0 * curand_uniform_double(&r) - 1.0);
            rot  = scale * (2.0 * curand_uniform_double(&r) - 1.0);
            theta = scale * 0.7 * (2.0 * curand_uniform_double(&r) - 1.0);
            if (mode == 0) {
                old_chosen = q[chosen];
                prop_chosen = {
                    fmin(targetL - 0.45, fmax(0.45, old_chosen.x + move)),
                    fmin(targetL - 0.45, fmax(0.45, old_chosen.y + rot)),
                    wrap_angle(old_chosen.a + theta)
                };
            }
        }
        __syncthreads();
        if (mode == 0) {
            double pc = cos(prop_chosen.a), ps = sin(prop_chosen.a);
            double oc = cachedc[chosen], os = cacheds[chosen];
            double delta = (t == 0) ? (boundary(prop_chosen, targetL) - boundary(old_chosen, targetL)) : 0.0;
            for (int i = t; i < n; i += NT) {
                if (i != chosen) {
                    delta += pair_cached(prop_chosen, q[i], pc, ps, cachedc[i], cacheds[i])
                           - pair_cached(old_chosen,  q[i], oc, os, cachedc[i], cacheds[i]);
                }
            }
            double total_delta = block_sum(delta, tmp);
            if (t == 0) {
                double ne = fmax(0.0, energy + total_delta);
                accept = (total_delta <= 0.0) || (log(curand_uniform_double(&r)) < -total_delta / temp);
                atomicAdd(stats + 1, 1ULL);
                if (accept) {
                    q[chosen] = prop_chosen;
                    cachedc[chosen] = pc; cacheds[chosen] = ps;
                    energy = ne; windowAccepted++;
                }
            }
            __syncthreads();
        } else {
            for (int k = 0; k < KPT; ++k) {
                int i = t + k * NT;
                if (i >= n) break;
                P z = q[i];
                double dx = z.x - cx, dy = z.y - cy;
                bool take = (mode == 1) ? (fabs(dy) < 0.55) :
                            (mode == 2) ? (fabs(dx) < 0.55) :
                            (mode == 3) ? (dx * dx + dy * dy < 2.5) :
                            (mode == 4) ? (fabs(remainder(z.a - q[chosen].a, 1.5707963267948966)) < 0.06) :
                            (mode == 5) ? (fabs(dx + dy) < 0.85 || fabs(dx - dy) < 0.85) :
                                          (z.a > 0.04 && z.a < 1.53);
                if (take) {
                    if (mode == 3 || mode == 4) {
                        double c = cos(theta), s = sin(theta);
                        z.x = cx + c * dx - s * dy + move;
                        z.y = cy + s * dx + c * dy + rot;
                        z.a = wrap_angle(z.a + theta);
                    } else if (mode == 6) {
                        double sgn = (i & 1) ? 1.0 : -1.0;
                        z.x += sgn * move; z.y += sgn * rot; z.a = wrap_angle(z.a + theta);
                    } else {
                        z.x += move; z.y += rot; z.a = wrap_angle(z.a + theta);
                    }
                    z.x = fmin(targetL - 0.45, fmax(0.45, z.x));
                    z.y = fmin(targetL - 0.45, fmax(0.45, z.y));
                }
                my_p[k] = z; my_c[k] = cos(z.a); my_s[k] = sin(z.a);
            }
            for (int k = 0; k < KPT; ++k) {
                int i = t + k * NT;
                if (i < n) {
                    P tmp_p = q[i]; q[i] = my_p[k]; my_p[k] = tmp_p;
                    double tc = cachedc[i]; cachedc[i] = my_c[k]; my_c[k] = tc;
                    double ts = cacheds[i]; cacheds[i] = my_s[k]; my_s[k] = ts;
                }
            }
            __syncthreads();
            double ne = total_cached(q, n, targetL, cachedc, cacheds, tmp);
            if (t == 0) {
                accept = (ne <= energy) || (log(curand_uniform_double(&r)) < (energy - ne) / temp);
                atomicAdd(stats + 1, 1ULL);
                if (accept) { energy = ne; windowAccepted++; }
            }
            __syncthreads();
            if (!accept) {
                for (int k = 0; k < KPT; ++k) {
                    int i = t + k * NT;
                    if (i < n) { q[i] = my_p[k]; cachedc[i] = my_c[k]; cacheds[i] = my_s[k]; }
                }
                __syncthreads();
            }
        }
        if (t == 0 && (step + 1) % 16 == 0) {
            adapt = fmax(.1, fmin(4.0, adapt * exp((windowAccepted / 16.0 - .25) * .5)));
            windowAccepted = 0;
        }
        __syncthreads();
    }

    G my_g[KPT], my_gold[KPT], my_d[KPT];
    P my_saved[KPT];
    for (int k = 0; k < KPT; ++k) { my_gold[k] = {0.0, 0.0, 0.0}; my_d[k] = {0.0, 0.0, 0.0}; }
    if (t == 0) step_cg = 0.04;
    __syncthreads();

    for (int iter = 0; iter < polish; iter++) {
        if (energy < 1e-24) break;
        double my_gg = 0.0, my_dgg = 0.0;
        for (int k = 0; k < KPT; ++k) {
            int i = t + k * NT;
            if (i >= n) break;
            G g{0.0, 0.0, 0.0};
            P pi = q[i];
            double ci = cachedc[i], si = cacheds[i];
            double r_ext = 0.5 * (fabs(ci) + fabs(si)), d_r = dr_smooth(ci, si);
            if (pi.x < r_ext) { double u = r_ext - pi.x; g.x -= 2.0 * u; g.a += 2.0 * u * d_r; }
            else if (pi.x > targetL - r_ext) { double u = pi.x + r_ext - targetL; g.x += 2.0 * u; g.a += 2.0 * u * d_r; }
            if (pi.y < r_ext) { double u = r_ext - pi.y; g.y -= 2.0 * u; g.a += 2.0 * u * d_r; }
            else if (pi.y > targetL - r_ext) { double u = pi.y + r_ext - targetL; g.y += 2.0 * u; g.a += 2.0 * u * d_r; }
            for (int j = 0; j < n; j++) if (i != j) pair_cached_smooth(pi, q[j], ci, si, cachedc[j], cacheds[j], &g);
            my_g[k] = g;
            my_gg  += my_gold[k].x * my_gold[k].x + my_gold[k].y * my_gold[k].y + my_gold[k].a * my_gold[k].a;
            my_dgg += (g.x - my_gold[k].x) * g.x + (g.y - my_gold[k].y) * g.y + (g.a - my_gold[k].a) * g.a;
        }
        double gg = block_sum(my_gg, tmp), dgg = block_sum(my_dgg, tmp);
        double beta = (iter == 0 || gg < 1e-30 || iter % 25 == 0) ? 0.0 : fmax(0.0, dgg / gg);
        double my_norm2 = 0.0;
        for (int k = 0; k < KPT; ++k) {
            int i = t + k * NT;
            if (i >= n) break;
            my_d[k].x = -my_g[k].x + beta * my_d[k].x;
            my_d[k].y = -my_g[k].y + beta * my_d[k].y;
            my_d[k].a = -my_g[k].a + beta * my_d[k].a;
            my_gold[k] = my_g[k]; my_saved[k] = q[i];
            my_norm2 += my_d[k].x * my_d[k].x + my_d[k].y * my_d[k].y + my_d[k].a * my_d[k].a;
        }
        if (block_sum(my_norm2, tmp) < 1e-28) break;
        double alpha = step_cg;
        bool ls_ok = false;
        for (int ls = 0; ls < 5; ++ls) {
            for (int k = 0; k < KPT; ++k) {
                int i = t + k * NT;
                if (i >= n) break;
                P z = { my_saved[k].x + alpha * my_d[k].x, my_saved[k].y + alpha * my_d[k].y, wrap_angle(my_saved[k].a + alpha * my_d[k].a) };
                q[i] = z; cachedc[i] = cos(z.a); cacheds[i] = sin(z.a);
            }
            __syncthreads();
            double ne = total_cached(q, n, targetL, cachedc, cacheds, tmp);
            if (ne < energy) {
                if (t == 0) { energy = ne; step_cg = fmin(0.15, alpha * 1.25); }
                ls_ok = true; __syncthreads(); break;
            }
            alpha *= 0.45;
        }
        if (!ls_ok) {
            for (int k = 0; k < KPT; ++k) {
                int i = t + k * NT;
                if (i < n) { q[i] = my_saved[k]; cachedc[i] = cos(my_saved[k].a); cacheds[i] = sin(my_saved[k].a); my_d[k] = {0.0, 0.0, 0.0}; }
            }
            if (t == 0) step_cg = fmax(1e-7, step_cg * 0.5);
            __syncthreads();
        }
    }

    double post_cert = device_certified_side(q, n, cachedc, cacheds, tmp, best_all + b * n, best_Ls[b]);
    if (t == 0) {
        if (post_cert < best_Ls[b]) { best_Ls[b] = post_cert; atomicAdd(stats + 2, 1ULL); }
        else if (energy < 1e-16) { atomicAdd(stats + 2, 1ULL); }
    }
    __syncthreads();

    if (energy > 1e-12 && polish > 0) {
        double my_max_sqE = -1.0; int my_worst = -1; double my_sqE[KPT];
        for (int k = 0; k < KPT; ++k) {
            int i = t + k * NT;
            if (i >= n) { my_sqE[k] = 0.0; continue; }
            double se = boundary(q[i], targetL);
            for (int j = 0; j < n; j++) if (i != j) se += pair_cached(q[i], q[j], cachedc[i], cacheds[i], cachedc[j], cacheds[j]);
            my_sqE[k] = se;
            if (se > my_max_sqE) { my_max_sqE = se; my_worst = i; }
        }
        double max_sqE = block_max(my_max_sqE, tmp);
        if (my_max_sqE == max_sqE && my_worst >= 0) worst_sq = my_worst;
        __syncthreads();
        if ((epoch % 2 == 1) && max_sqE > 1e-5) {
            int w_idx = worst_sq;
            double my_void_d2 = -1.0, vx = 0.5 * targetL, vy = 0.5 * targetL;
            if (t < 64) {
                int ix = t % 8, iy = t / 8;
                vx = 0.55 + (targetL - 1.1) * (ix + 0.5) / 8.0;
                vy = 0.55 + (targetL - 1.1) * (iy + 0.5) / 8.0;
                my_void_d2 = 1e9;
                for (int j = 0; j < n; j++) if (j != w_idx) {
                    double dx = vx - q[j].x, dy = vy - q[j].y;
                    double d2 = dx * dx + dy * dy;
                    if (d2 < my_void_d2) my_void_d2 = d2;
                }
            }
            double best_void_d2 = block_max(my_void_d2, tmp);
            if (t < 64 && my_void_d2 == best_void_d2) { cx = vx; cy = vy; }
            __syncthreads();
            if (t == 0 && w_idx >= 0 && w_idx < n) {
                q[w_idx].x = fmin(targetL - 0.5, fmax(0.5, cx + 0.05 * (2.0 * curand_uniform_double(&r) - 1.0)));
                q[w_idx].y = fmin(targetL - 0.5, fmax(0.5, cy + 0.05 * (2.0 * curand_uniform_double(&r) - 1.0)));
                q[w_idx].a = wrap_angle(curand_uniform_double(&r) * 1.5707963267948966);
            }
            __syncthreads();
        }
        auto r_sub = r;
        skipahead((unsigned long long)(t + 1) * 8192 + (unsigned long long)epoch * 131, &r_sub);
        for (int k = 0; k < KPT; ++k) {
            int i = t + k * NT;
            if (i >= n) break;
            bool hot = (my_sqE[k] > 1e-6);
            double amp_xy = hot ? 0.08 : 0.008, amp_a = hot ? 0.18 : 0.015;
            q[i].x = fmin(targetL - 0.48, fmax(0.48, q[i].x + amp_xy * (2.0 * curand_uniform_double(&r_sub) - 1.0)));
            q[i].y = fmin(targetL - 0.48, fmax(0.48, q[i].y + amp_xy * (2.0 * curand_uniform_double(&r_sub) - 1.0)));
            q[i].a = wrap_angle(q[i].a + amp_a * (2.0 * curand_uniform_double(&r_sub) - 1.0));
        }
        __syncthreads();
    }
    for (int i = t; i < n; i += NT) all[b * n + i] = q[i];
    if (t == 0) { energies[b] = energy; rng[b] = r; atomicAdd(stats, 1ULL); }
}

__global__ void swap_replicas(P* q, double* e, curandStatePhilox4_32_10_t* rng, int n, int chains, int parity, int epoch, double tmin, double tmax) {
    int a = blockIdx.x * 2 + parity, b = a + 1, t = threadIdx.x;
    if (b >= chains || a / LADDER != b / LADDER) return;
    __shared__ int take;
    if (t == 0) {
        auto r = rng[a];
        double cool = pow(.002, double(epoch % 200) / 199.0);
        double ta = tmin * pow(tmax / tmin, (a % 8) / 7.0) * cool;
        double tb = tmin * pow(tmax / tmin, (b % 8) / 7.0) * cool;
        take = log(curand_uniform_double(&r)) < (1.0 / ta - 1.0 / tb) * (e[a] - e[b]);
        rng[a] = r;
        if (take) { double z = e[a]; e[a] = e[b]; e[b] = z; }
    }
    __syncthreads();
    if (take) for (int i = t; i < n; i += NT) { P z = q[a * n + i]; q[a * n + i] = q[b * n + i]; q[b * n + i] = z; }
}
} // namespace dpack

int main(int argc,char**argv){
 if(argc!=7){std::cerr<<"packing seed.txt epochs steps chains random_seed output.json\n";return 2;}
 int epochs=atoi(argv[2]),steps=atoi(argv[3]),chains=atoi(argv[4]),n=0;
 if(epochs<0||epochs>100000||steps<0||steps>4096||chains<0||chains>4096||(chains>0&&chains%8))return 2;
 std::ifstream in(argv[1]);double L=0;if(!(in>>n>>L))return 2;
 if(n<1||n>1000||!std::isfinite(L)||L<=0||L>64)return 2;
 std::vector<dpack::P> seed(n);
 for(auto& p:seed){in>>p.x>>p.y>>p.a;if(!in||!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.a))return 2;}

 double tmin = envf("PACK_TMIN", 1e-11f, 1e-12f, 1.f);
 double tmax = envf("PACK_TMAX", 1e-4f, 1e-12f, 1.f);
 double mv   = envf("PACK_MOVE", .02f, .000001f, .3f);
 float rot_cfg = envf("PACK_ROTATION", .12f, .000001f, 1.57f);
 (void)rot_cfg;
 if(tmax<tmin)return 2;
 double shrinkStep = envf("PACK_SHRINK", .001f, .000001f, .1f);
 int autotune = (int)envf("PACK_AUTOTUNE", 1.f, 0.f, 1.f);
 int max_chains = (int)envf("PACK_MAX_CHAINS", 768.f, 8.f, 4096.f);
 unsigned long long rng_seed = strtoull(argv[5], nullptr, 10);

 cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,0));
 int max_blocks_per_sm = 1;
 if (n <= NT) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, dpack::search_kernel<1>, NT, 0));
 else if (n <= 4 * NT) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, dpack::search_kernel<4>, NT, 0));
 else if (n <= 8 * NT) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, dpack::search_kernel<8>, NT, 0));
 else CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, dpack::search_kernel<16>, NT, 0));
 int wave_chains = ((prop.multiProcessorCount * std::max(1, max_blocks_per_sm)) / LADDER) * LADDER;
 if (wave_chains < LADDER) wave_chains = LADDER;
 if (chains == 0 || (autotune && chains >= 64)) {
  int target = (chains == 0) ? wave_chains : std::max(chains, std::min(wave_chains, max_chains));
  chains = ((std::min(target, max_chains) + LADDER - 1) / LADDER) * LADDER;
 }
 if (epochs == 0) epochs = (n <= 64) ? 50 : (n <= 256) ? 40 : 30;
 if (steps == 0)  steps  = (n <= 64) ? 64 : (n <= 256) ? 48 : 32;
 // Cap excessive legacy epoch counts (e.g. 4000..20000 from FP32 scripts) into high-precision CG epochs
 int eff_epochs = (epochs > 120) ? std::min(epochs, 80) : epochs;

 std::vector<dpack::P> h, best = seed;
 dpack::build_portfolio(n, chains, L, seed, h);
 std::vector<double> h_bestL(chains, L);
 dpack::P *d, *d_best; double *de, *d_bestL; curandStatePhilox4_32_10_t* rng; unsigned long long* stats;
 CK(cudaMalloc(&d, h.size()*sizeof(dpack::P)));
 CK(cudaMalloc(&d_best, h.size()*sizeof(dpack::P)));
 CK(cudaMalloc(&de, chains*sizeof(double)));
 CK(cudaMalloc(&d_bestL, chains*sizeof(double)));
 CK(cudaMalloc(&rng, chains*sizeof(*rng)));
 CK(cudaMalloc(&stats, 4*sizeof(*stats)));
 CK(cudaMemset(stats, 0, 4*sizeof(*stats)));
 CK(cudaMemcpy(d, h.data(), h.size()*sizeof(dpack::P), cudaMemcpyHostToDevice));
 CK(cudaMemcpy(d_best, h.data(), h.size()*sizeof(dpack::P), cudaMemcpyHostToDevice));
 CK(cudaMemcpy(d_bestL, h_bestL.data(), chains*sizeof(double), cudaMemcpyHostToDevice));

 dpack::random_init<<<(chains+255)/256,256>>>(rng, chains, rng_seed);CK(cudaGetLastError());
 dpack::structured_init<<<chains,NT>>>(d, rng, n, chains, L);CK(cudaGetLastError());

 double bestL = L; int shrink = 0;
 auto start = std::chrono::steady_clock::now();
 for(int epoch=0; epoch<eff_epochs; epoch++){
  if (n <= NT) dpack::search_kernel<1><<<chains,NT>>>(d, d_best, de, d_bestL, rng, n, bestL, steps, epoch, 200, tmin, tmax, mv, shrinkStep, stats);
  else if (n <= 4*NT) dpack::search_kernel<4><<<chains,NT>>>(d, d_best, de, d_bestL, rng, n, bestL, steps, epoch, 200, tmin, tmax, mv, shrinkStep, stats);
  else if (n <= 8*NT) dpack::search_kernel<8><<<chains,NT>>>(d, d_best, de, d_bestL, rng, n, bestL, steps, epoch, 200, tmin, tmax, mv, shrinkStep, stats);
  else dpack::search_kernel<16><<<chains,NT>>>(d, d_best, de, d_bestL, rng, n, bestL, steps, epoch, 200, tmin, tmax, mv, shrinkStep, stats);
  CK(cudaGetLastError());
  CK(cudaMemcpy(h_bestL.data(), d_bestL, chains*sizeof(double), cudaMemcpyDeviceToHost));
  int best_chain = (int)(std::min_element(h_bestL.begin(), h_bestL.end()) - h_bestL.begin());
  if (h_bestL[best_chain] < bestL - 1e-11) {
   double old_bestL = bestL;
   bestL = h_bestL[best_chain];
   best.resize(n);
   CK(cudaMemcpy(best.data(), d_best + best_chain * n, n*sizeof(dpack::P), cudaMemcpyDeviceToHost));
   shrink++;
   const double mults[8] = {0.02, 0.1, 0.4, 1.0, 2.5, 5.0, 10.0, 20.0};
   CK(cudaMemcpy(h.data(), d, h.size()*sizeof(dpack::P), cudaMemcpyDeviceToHost));
   for(int b=0; b<chains; b++){
    if(b%LADDER==0 || (b<LADDER && (old_bestL - bestL > 5e-4))){
     double tL = std::max(std::sqrt((double)n)+1e-6, bestL - shrinkStep*mults[b%LADDER]);
     double sc = tL / bestL;
     for(int i=0; i<n; i++) h[b*n+i] = {best[i].x*sc, best[i].y*sc, best[i].a};
    }
   }
   CK(cudaMemcpy(d, h.data(), h.size()*sizeof(dpack::P), cudaMemcpyHostToDevice));
  } else {
   dpack::swap_replicas<<<(chains+1)/2,NT>>>(d, de, rng, n, chains, epoch%2, epoch, tmin, tmax);
   CK(cudaGetLastError());
  }
 }
 CK(cudaDeviceSynchronize());
 double sec = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 unsigned long long h_stats[4];
 CK(cudaMemcpy(h_stats, stats, sizeof(h_stats), cudaMemcpyDeviceToHost));
 int feasible = std::max(1, (int)h_stats[2]);
 std::ofstream out(argv[6]);if(!out)return 2;out.precision(17);
 out<<"{\"n\":"<<n<<",\"target_L\":"<<bestL<<",\"seconds\":"<<sec<<",\"chain_steps\":"<<(long long)chains*eff_epochs*steps
    <<",\"proposals\":"<<h_stats[1]<<",\"feasible_checkpoints\":"<<feasible<<",\"shrinks\":"<<shrink
    <<",\"chains\":"<<chains<<",\"sm_count\":"<<prop.multiProcessorCount<<",\"blocks_per_sm\":"<<max_blocks_per_sm
    <<",\"wave_chains\":"<<wave_chains<<",\"device\":\""<<prop.name<<"\",\"q\":[";
 for(size_t i=0;i<best.size();i++){if(i)out<<",";out<<"["<<best[i].x<<","<<best[i].y<<","<<best[i].a<<"]";}
 out<<"]}\n";
 CK(cudaFree(d));CK(cudaFree(d_best));CK(cudaFree(de));CK(cudaFree(d_bestL));CK(cudaFree(rng));CK(cudaFree(stats));
 return out?0:2;
}
