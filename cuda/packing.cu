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
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include "families.hpp"
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
    sqfam::build_portfolio_generic(n, chains, seed_L, seed, h_init);
}

// ============================================================================
// Ask #3: Mixed-Precision (FP32 Coarse SA + Coarse CG, FP64 Fine Polish + Cert)
// ============================================================================

struct P32 { float x, y, a; };
struct G32 { float x, y, a; };

__device__ __forceinline__ float wrap_angle_f32(float a) {
    float r = fmodf(a, 1.57079637f);
    return (r < 0.0f) ? (r + 1.57079637f) : r;
}

__device__ __forceinline__ float boundary_f32(P32 p, float L) {
    float s, c;
    __sincosf(p.a, &s, &c);
    float r = 0.5f * (fabsf(c) + fabsf(s));
    float e = 0.0f;
    if (p.x < r) { float u = r - p.x; e += u * u; }
    else if (p.x > L - r) { float u = p.x + r - L; e += u * u; }
    if (p.y < r) { float u = r - p.y; e += u * u; }
    else if (p.y > L - r) { float u = p.y + r - L; e += u * u; }
    return e;
}

__device__ __forceinline__ float pair_cached_f32(P32 a, P32 b, float ca, float sa, float cb, float sb) {
    float dx = a.x - b.x, dy = a.y - b.y;
    float d2 = dx * dx + dy * dy;
    if (d2 >= 2.0f) return 0.0f;
    float r = 0.5f + 0.5f * (fabsf(ca * cb + sa * sb) + fabsf(-sa * cb + ca * sb));
    float m = r - fmaxf(fabsf(dx * ca + dy * sa), fabsf(-dx * sa + dy * ca));
    if (m <= 0.0f) return 0.0f;
    float n = r - fmaxf(fabsf(dx * cb + dy * sb), fabsf(-dx * sb + dy * cb));
    if (n <= 0.0f) return 0.0f;
    float p = fminf(m, n);
    return p * p;
}

__device__ __forceinline__ float block_sum_f32(float v, float* s) {
    int t = threadIdx.x;
    s[t] = v;
    __syncthreads();
    for (int k = NT / 2; k > 0; k >>= 1) {
        if (t < k) s[t] += s[t + k];
        __syncthreads();
    }
    return s[0];
}

__device__ __forceinline__ float total_cached_f32(const P32* q, int n, float L, const float* c, const float* s, float* tmp) {
    int t = threadIdx.x;
    float e = 0.0f;
    for (int i = t; i < n; i += NT) {
        float r = 0.5f * (fabsf(c[i]) + fabsf(s[i]));
        if (q[i].x < r) { float u = r - q[i].x; e += u * u; }
        else if (q[i].x > L - r) { float u = q[i].x + r - L; e += u * u; }
        if (q[i].y < r) { float u = r - q[i].y; e += u * u; }
        else if (q[i].y > L - r) { float u = q[i].y + r - L; e += u * u; }
        for (int j = i + 1; j < n; j++) e += pair_cached_f32(q[i], q[j], c[i], s[i], c[j], s[j]);
    }
    return block_sum_f32(e, tmp);
}

__device__ __forceinline__ float sgn_smooth_f32(float u) {
    return u * rsqrtf(u * u + 1e-12f);
}

__device__ __forceinline__ float dr_smooth_f32(float c, float s) {
    return 0.5f * (-s * sgn_smooth_f32(c) + c * sgn_smooth_f32(s));
}

__device__ __forceinline__ void pair_cached_smooth_f32(
    P32 a, P32 b, float ca, float sa, float cb, float sb, G32* ga
) {
    float dx = a.x - b.x, dy = a.y - b.y;
    if (dx * dx + dy * dy >= 2.0f) return;
    float c_ab = ca * cb + sa * sb;
    float s_ab = sa * cb - ca * sb;
    float r = 0.5f + 0.5f * (fabsf(c_ab) + fabsf(s_ab));

    float u1 = dx * ca + dy * sa, u2 = -dx * sa + dy * ca;
    float a1 = fabsf(u1), a2 = fabsf(u2);
    float ma = fmaxf(a1, a2);
    float pen_a = r - ma;
    if (pen_a <= 0.0f) return;

    float v1 = dx * cb + dy * sb, v2 = -dx * sb + dy * cb;
    float b1 = fabsf(v1), b2 = fabsf(v2);
    float mb = fmaxf(b1, b2);
    float pen_b = r - mb;
    if (pen_b <= 0.0f) return;

    float dr_da = 0.5f * (-s_ab * sgn_smooth_f32(c_ab) + c_ab * sgn_smooth_f32(s_ab));
    float s1 = sgn_smooth_f32(u1), s2 = sgn_smooth_f32(u2);
    float d1_dx = s1 * ca,  d1_dy = s1 * sa,  d1_da = s1 * u2;
    float d2_dx = -s2 * sa, d2_dy = s2 * ca,  d2_da = -s2 * u1;

    float diff_a = a1 - a2;
    float wa = 0.5f * (1.0f + diff_a * rsqrtf(diff_a * diff_a + 1e-10f));
    float dma_dx = wa * d1_dx + (1.0f - wa) * d2_dx;
    float dma_dy = wa * d1_dy + (1.0f - wa) * d2_dy;
    float dma_da = wa * d1_da + (1.0f - wa) * d2_da;
    float dpa_dx = -dma_dx, dpa_dy = -dma_dy, dpa_da = dr_da - dma_da;

    float t1 = sgn_smooth_f32(v1), t2 = sgn_smooth_f32(v2);
    float diff_b = b1 - b2;
    float wb = 0.5f * (1.0f + diff_b * rsqrtf(diff_b * diff_b + 1e-10f));
    float dmb_dx = wb * (t1 * cb) + (1.0f - wb) * (-t2 * sb);
    float dmb_dy = wb * (t1 * sb) + (1.0f - wb) * (t2 * cb);
    float dpb_dx = -dmb_dx, dpb_dy = -dmb_dy, dpb_da = dr_da;

    float diff_p = pen_b - pen_a;
    float wp = 0.5f * (1.0f + diff_p * rsqrtf(diff_p * diff_p + 1e-10f));
    float p = fminf(pen_a, pen_b);
    float two_p = 2.0f * p;
    ga->x += two_p * (wp * dpa_dx + (1.0f - wp) * dpb_dx);
    ga->y += two_p * (wp * dpa_dy + (1.0f - wp) * dpb_dy);
    ga->a += two_p * (wp * dpa_da + (1.0f - wp) * dpb_da);
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

template <int KPT, bool MIXED_PREC = true>
__global__ void search_kernel(
    P* all, P* best_all, double* energies, double* best_Ls,
    curandStatePhilox4_32_10_t* rng, int n, double baseL, int steps, int epoch,
    int polish, double tmin, double tmax, double baseMove, double shrink,
    unsigned long long* stats,
    const int* puzzle_chain_offset = nullptr,
    const int* puzzle_n = nullptr,
    const double* puzzle_baseL = nullptr,
    const int* puzzle_data_offset = nullptr,
    int num_puzzles = 0
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
    int local_b = b;
    int data_idx = b * n;
    if (num_puzzles > 0 && puzzle_chain_offset != nullptr) {
        int pid = 0;
        while (pid + 1 < num_puzzles && b >= puzzle_chain_offset[pid + 1]) pid++;
        local_b = b - puzzle_chain_offset[pid];
        n = puzzle_n[pid];
        baseL = puzzle_baseL[pid];
        data_idx = puzzle_data_offset[pid] + local_b * n;
    }

    for (int i = t; i < n; i += NT) {
        q[i] = all[data_idx + i];
        cachedc[i] = cos(q[i].a);
        cacheds[i] = sin(q[i].a);
    }
    if (t == 0) {
        r = rng[b];
        adapt = 1.0;
        windowAccepted = 0;
        const double mults[8] = {0.02, 0.1, 0.4, 1.0, 2.5, 5.0, 10.0, 20.0};
        targetL = fmax(sqrt((double)n) + 1e-6, baseL - shrink * mults[local_b % LADDER]);
    }
    __syncthreads();

    double init_cert = device_certified_side(q, n, cachedc, cacheds, tmp, best_all + data_idx, best_Ls[b]);
    if (t == 0 && init_cert < best_Ls[b]) { best_Ls[b] = init_cert; atomicAdd(stats + 2, 1ULL); }
    __syncthreads();

    double cool = pow(.002, double(epoch % 200) / 199.0);
    double temp = tmin * pow(tmax / tmin, (local_b % LADDER) / 7.0) * cool;

    if (MIXED_PREC) {
        // ====================================================================
        // Ask #3 Phase 1A: Coarse Simulated Annealing in FP32
        // Ask #3 Phase 1B: Coarse Conjugate-Gradient Relaxation in FP32
        // ====================================================================
        P32* q32 = reinterpret_cast<P32*>(q);
        float* c32 = reinterpret_cast<float*>(cachedc);
        float* s32 = reinterpret_cast<float*>(cacheds);
        float* tmp32 = reinterpret_cast<float*>(tmp);
        __shared__ P32 old_c32, prop_c32;
        __shared__ float energy32, step_cg32;

        P32 my_p32[KPT];
        float my_c32[KPT], my_s32[KPT];
        for (int k = 0; k < KPT; ++k) {
            int i = t + k * NT;
            if (i < n) {
                my_p32[k] = {(float)q[i].x, (float)q[i].y, (float)q[i].a};
                __sincosf(my_p32[k].a, &my_s32[k], &my_c32[k]);
            }
        }
        __syncthreads();
        for (int k = 0; k < KPT; ++k) {
            int i = t + k * NT;
            if (i < n) {
                q32[i] = my_p32[k];
                c32[i] = my_c32[k];
                s32[i] = my_s32[k];
            }
        }
        __syncthreads();

        float targetL32 = (float)targetL;
        float temp32 = fmaxf(1e-18f, (float)temp);
        float e32 = total_cached_f32(q32, n, targetL32, c32, s32, tmp32);
        if (t == 0) energy32 = e32;
        __syncthreads();

        for (int step = 0; step < steps; step++) {
            if (t == 0) {
                chosen = curand(&r) % n;
                mode = curand(&r) % 7;
                cx = q32[chosen].x; cy = q32[chosen].y;
                scale = adapt * baseMove * (.15 + 1.85 * (local_b % LADDER) / 7.0) * sqrt(cool);
                if (step % 16 == 0) scale *= 3.5;
                move = scale * (2.0 * curand_uniform(&r) - 1.0);
                rot  = scale * (2.0 * curand_uniform(&r) - 1.0);
                theta = scale * 0.7 * (2.0 * curand_uniform(&r) - 1.0);
                if (mode == 0) {
                    old_c32 = q32[chosen];
                    prop_c32 = {
                        fminf(targetL32 - 0.45f, fmaxf(0.45f, old_c32.x + (float)move)),
                        fminf(targetL32 - 0.45f, fmaxf(0.45f, old_c32.y + (float)rot)),
                        wrap_angle_f32(old_c32.a + (float)theta)
                    };
                }
            }
            __syncthreads();
            if (mode == 0) {
                float pc, ps;
                __sincosf(prop_c32.a, &ps, &pc);
                float oc = c32[chosen], os = s32[chosen];
                float delta = (t == 0) ? (boundary_f32(prop_c32, targetL32) - boundary_f32(old_c32, targetL32)) : 0.0f;
                for (int i = t; i < n; i += NT) {
                    if (i != chosen) {
                        delta += pair_cached_f32(prop_c32, q32[i], pc, ps, c32[i], s32[i])
                               - pair_cached_f32(old_c32,  q32[i], oc, os, c32[i], s32[i]);
                    }
                }
                float total_delta = block_sum_f32(delta, tmp32);
                if (t == 0) {
                    float ne = fmaxf(0.0f, energy32 + total_delta);
                    accept = (total_delta <= 0.0f) || (logf(curand_uniform(&r)) < -total_delta / temp32);
                    atomicAdd(stats + 1, 1ULL);
                    if (accept) {
                        q32[chosen] = prop_c32;
                        c32[chosen] = pc; s32[chosen] = ps;
                        energy32 = ne; windowAccepted++;
                    }
                }
                __syncthreads();
            } else {
                float cx32 = (float)cx, cy32 = (float)cy;
                float mv32 = (float)move, rt32 = (float)rot, th32 = (float)theta;
                float c_th, s_th;
                __sincosf(th32, &s_th, &c_th);
                for (int k = 0; k < KPT; ++k) {
                    int i = t + k * NT;
                    if (i >= n) break;
                    P32 z = q32[i];
                    float dx = z.x - cx32, dy = z.y - cy32;
                    bool take = (mode == 1) ? (fabsf(dy) < 0.55f) :
                                (mode == 2) ? (fabsf(dx) < 0.55f) :
                                (mode == 3) ? (dx * dx + dy * dy < 2.5f) :
                                (mode == 4) ? (fabsf(remainderf(z.a - q32[chosen].a, 1.57079637f)) < 0.06f) :
                                (mode == 5) ? (fabsf(dx + dy) < 0.85f || fabsf(dx - dy) < 0.85f) :
                                              (z.a > 0.04f && z.a < 1.53f);
                    if (take) {
                        if (mode == 3 || mode == 4) {
                            z.x = cx32 + c_th * dx - s_th * dy + mv32;
                            z.y = cy32 + s_th * dx + c_th * dy + rt32;
                            z.a = wrap_angle_f32(z.a + th32);
                        } else if (mode == 6) {
                            float sgn = (i & 1) ? 1.0f : -1.0f;
                            z.x += sgn * mv32; z.y += sgn * rt32; z.a = wrap_angle_f32(z.a + th32);
                        } else {
                            z.x += mv32; z.y += rt32; z.a = wrap_angle_f32(z.a + th32);
                        }
                        z.x = fminf(targetL32 - 0.45f, fmaxf(0.45f, z.x));
                        z.y = fminf(targetL32 - 0.45f, fmaxf(0.45f, z.y));
                    }
                    my_p32[k] = z;
                    __sincosf(z.a, &my_s32[k], &my_c32[k]);
                }
                for (int k = 0; k < KPT; ++k) {
                    int i = t + k * NT;
                    if (i < n) {
                        P32 tmp_p = q32[i]; q32[i] = my_p32[k]; my_p32[k] = tmp_p;
                        float tc = c32[i]; c32[i] = my_c32[k]; my_c32[k] = tc;
                        float ts = s32[i]; s32[i] = my_s32[k]; my_s32[k] = ts;
                    }
                }
                __syncthreads();
                float ne = total_cached_f32(q32, n, targetL32, c32, s32, tmp32);
                if (t == 0) {
                    accept = (ne <= energy32) || (logf(curand_uniform(&r)) < (energy32 - ne) / temp32);
                    atomicAdd(stats + 1, 1ULL);
                    if (accept) { energy32 = ne; windowAccepted++; }
                }
                __syncthreads();
                if (!accept) {
                    for (int k = 0; k < KPT; ++k) {
                        int i = t + k * NT;
                        if (i < n) { q32[i] = my_p32[k]; c32[i] = my_c32[k]; s32[i] = my_s32[k]; }
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

        // Phase 1B: Coarse FP32 Conjugate-Gradient Relaxation
        int coarse_polish = polish / 2;
        G32 my_g32[KPT], my_gold32[KPT], my_d32[KPT];
        P32 my_saved32[KPT];
        for (int k = 0; k < KPT; ++k) { my_gold32[k] = {0.0f, 0.0f, 0.0f}; my_d32[k] = {0.0f, 0.0f, 0.0f}; }
        if (t == 0) step_cg32 = 0.04f;
        __syncthreads();

        for (int iter = 0; iter < coarse_polish; iter++) {
            if (energy32 < 1e-10f) break;
            float my_gg = 0.0f, my_dgg = 0.0f;
            for (int k = 0; k < KPT; ++k) {
                int i = t + k * NT;
                if (i >= n) break;
                G32 g{0.0f, 0.0f, 0.0f};
                P32 pi = q32[i];
                float ci = c32[i], si = s32[i];
                float r_ext = 0.5f * (fabsf(ci) + fabsf(si)), d_r = dr_smooth_f32(ci, si);
                if (pi.x < r_ext) { float u = r_ext - pi.x; g.x -= 2.0f * u; g.a += 2.0f * u * d_r; }
                else if (pi.x > targetL32 - r_ext) { float u = pi.x + r_ext - targetL32; g.x += 2.0f * u; g.a += 2.0f * u * d_r; }
                if (pi.y < r_ext) { float u = r_ext - pi.y; g.y -= 2.0f * u; g.a += 2.0f * u * d_r; }
                else if (pi.y > targetL32 - r_ext) { float u = pi.y + r_ext - targetL32; g.y += 2.0f * u; g.a += 2.0f * u * d_r; }
                for (int j = 0; j < n; j++) if (i != j) pair_cached_smooth_f32(pi, q32[j], ci, si, c32[j], s32[j], &g);
                my_g32[k] = g;
                my_gg  += my_gold32[k].x * my_gold32[k].x + my_gold32[k].y * my_gold32[k].y + my_gold32[k].a * my_gold32[k].a;
                my_dgg += (g.x - my_gold32[k].x) * g.x + (g.y - my_gold32[k].y) * g.y + (g.a - my_gold32[k].a) * g.a;
            }
            float gg = block_sum_f32(my_gg, tmp32), dgg = block_sum_f32(my_dgg, tmp32);
            float beta = (iter == 0 || gg < 1e-20f || iter % 20 == 0) ? 0.0f : fmaxf(0.0f, dgg / gg);
            float my_norm2 = 0.0f;
            for (int k = 0; k < KPT; ++k) {
                int i = t + k * NT;
                if (i >= n) break;
                my_d32[k].x = -my_g32[k].x + beta * my_d32[k].x;
                my_d32[k].y = -my_g32[k].y + beta * my_d32[k].y;
                my_d32[k].a = -my_g32[k].a + beta * my_d32[k].a;
                my_gold32[k] = my_g32[k]; my_saved32[k] = q32[i];
                my_norm2 += my_d32[k].x * my_d32[k].x + my_d32[k].y * my_d32[k].y + my_d32[k].a * my_d32[k].a;
            }
            if (block_sum_f32(my_norm2, tmp32) < 1e-18f) break;
            float alpha = step_cg32;
            bool ls_ok = false;
            for (int ls = 0; ls < 4; ++ls) {
                for (int k = 0; k < KPT; ++k) {
                    int i = t + k * NT;
                    if (i >= n) break;
                    P32 z = {
                        my_saved32[k].x + alpha * my_d32[k].x,
                        my_saved32[k].y + alpha * my_d32[k].y,
                        wrap_angle_f32(my_saved32[k].a + alpha * my_d32[k].a)
                    };
                    q32[i] = z; __sincosf(z.a, &s32[i], &c32[i]);
                }
                __syncthreads();
                float ne = total_cached_f32(q32, n, targetL32, c32, s32, tmp32);
                if (ne < energy32) {
                    if (t == 0) { energy32 = ne; step_cg32 = fminf(0.15f, alpha * 1.25f); }
                    ls_ok = true; __syncthreads(); break;
                }
                alpha *= 0.45f;
            }
            if (!ls_ok) {
                for (int k = 0; k < KPT; ++k) {
                    int i = t + k * NT;
                    if (i < n) {
                        q32[i] = my_saved32[k];
                        __sincosf(my_saved32[k].a, &s32[i], &c32[i]);
                        my_d32[k] = {0.0f, 0.0f, 0.0f};
                    }
                }
                if (t == 0) step_cg32 = fmaxf(1e-5f, step_cg32 * 0.5f);
                __syncthreads();
            }
        }

        // Convert FP32 state back to FP64 for fine polish & exact certification
        for (int k = 0; k < KPT; ++k) {
            int i = t + k * NT;
            if (i < n) my_p32[k] = q32[i];
        }
        __syncthreads();
        for (int k = 0; k < KPT; ++k) {
            int i = t + k * NT;
            if (i < n) {
                q[i] = {(double)my_p32[k].x, (double)my_p32[k].y, wrap_angle((double)my_p32[k].a)};
                cachedc[i] = cos(q[i].a);
                cacheds[i] = sin(q[i].a);
            }
        }
        __syncthreads();
        double e64 = total_cached(q, n, targetL, cachedc, cacheds, tmp);
        if (t == 0) energy = e64;
        __syncthreads();
    } else {
        // Legacy pure FP64 SA
        double e = total_cached(q, n, targetL, cachedc, cacheds, tmp);
        if (t == 0) energy = e;
        __syncthreads();

        P my_p[KPT];
        double my_c[KPT], my_s[KPT];

        for (int step = 0; step < steps; step++) {
            if (t == 0) {
                chosen = curand(&r) % n;
                mode = curand(&r) % 7;
                cx = q[chosen].x; cy = q[chosen].y;
                scale = adapt * baseMove * (.15 + 1.85 * (local_b % LADDER) / 7.0) * sqrt(cool);
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
    }

    // ====================================================================
    // Ask #3 Phase 2: Fine Conjugate-Gradient Polish + Certification in FP64
    // ====================================================================
    int fine_polish = MIXED_PREC ? ((energy < 5e-3) ? (polish - polish / 2) : 20) : polish;
    G my_g[KPT], my_gold[KPT], my_d[KPT];
    P my_saved[KPT];
    for (int k = 0; k < KPT; ++k) { my_gold[k] = {0.0, 0.0, 0.0}; my_d[k] = {0.0, 0.0, 0.0}; }
    if (t == 0) step_cg = 0.04;
    __syncthreads();

    for (int iter = 0; iter < fine_polish; iter++) {
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

    double post_cert = device_certified_side(q, n, cachedc, cacheds, tmp, best_all + data_idx, best_Ls[b]);
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
    for (int i = t; i < n; i += NT) all[data_idx + i] = q[i];
    if (t == 0) { energies[b] = energy; rng[b] = r; atomicAdd(stats, 1ULL); }
}

__global__ void swap_replicas(
    P* q, double* e, curandStatePhilox4_32_10_t* rng, int n, int chains,
    int parity, int epoch, double tmin, double tmax,
    const int* puzzle_chain_offset = nullptr,
    const int* puzzle_n = nullptr,
    const int* puzzle_data_offset = nullptr,
    int num_puzzles = 0
) {
    int a = blockIdx.x * 2 + parity, b = a + 1, t = threadIdx.x;
    if (b >= chains || a / LADDER != b / LADDER) return;
    int local_a = a, local_b = b;
    int data_a = a * n, data_b = b * n;
    if (num_puzzles > 0 && puzzle_chain_offset != nullptr) {
        int pid_a = 0;
        while (pid_a + 1 < num_puzzles && a >= puzzle_chain_offset[pid_a + 1]) pid_a++;
        int pid_b = 0;
        while (pid_b + 1 < num_puzzles && b >= puzzle_chain_offset[pid_b + 1]) pid_b++;
        if (pid_a != pid_b) return;
        local_a = a - puzzle_chain_offset[pid_a];
        local_b = b - puzzle_chain_offset[pid_b];
        if (local_a / LADDER != local_b / LADDER) return;
        n = puzzle_n[pid_a];
        data_a = puzzle_data_offset[pid_a] + local_a * n;
        data_b = puzzle_data_offset[pid_b] + local_b * n;
    }
    __shared__ int take;
    if (t == 0) {
        auto r = rng[a];
        double cool = pow(.002, double(epoch % 200) / 199.0);
        double ta = tmin * pow(tmax / tmin, (local_a % 8) / 7.0) * cool;
        double tb = tmin * pow(tmax / tmin, (local_b % 8) / 7.0) * cool;
        take = log(curand_uniform_double(&r)) < (1.0 / ta - 1.0 / tb) * (e[a] - e[b]);
        rng[a] = r;
        if (take) { double z = e[a]; e[a] = e[b]; e[b] = z; }
    }
    __syncthreads();
    if (take) for (int i = t; i < n; i += NT) { P z = q[data_a + i]; q[data_a + i] = q[data_b + i]; q[data_b + i] = z; }
}

__global__ void structured_init_batch(
    P* all, curandStatePhilox4_32_10_t* rng, int total_chains,
    const int* puzzle_chain_offset, const int* puzzle_n,
    const double* puzzle_baseL, const int* puzzle_data_offset, int num_puzzles
) {
    int b = blockIdx.x, t = threadIdx.x;
    if (b >= total_chains) return;
    int pid = 0;
    while (pid + 1 < num_puzzles && b >= puzzle_chain_offset[pid + 1]) pid++;
    int local_b = b - puzzle_chain_offset[pid];
    int n = puzzle_n[pid];
    double L = puzzle_baseL[pid];
    int data_idx = puzzle_data_offset[pid] + local_b * n;

    __shared__ double jitter;
    if (t == 0) {
        auto r = rng[b];
        int group = local_b % 8;
        jitter = (group == 0) ? 0.0 : (group == 1) ? 1e-4 * pow(100.0, curand_uniform_double(&r)) : 0.03;
        rng[b] = r;
    }
    __syncthreads();
    if (jitter == 0.0) return;
    auto r = rng[b];
    skipahead((unsigned long long)(t + 1) * 4096, &r);
    for (int i = t; i < n; i += blockDim.x) {
        P p = all[data_idx + i];
        p.x = fmin(L - 0.5, fmax(0.5, p.x + jitter * (2.0 * curand_uniform_double(&r) - 1.0)));
        p.y = fmin(L - 0.5, fmax(0.5, p.y + jitter * (2.0 * curand_uniform_double(&r) - 1.0)));
        p.a = wrap_angle(p.a + jitter * (2.0 * curand_uniform_double(&r) - 1.0));
        all[data_idx + i] = p;
    }
}

template <bool MIXED>
static void dispatch_search(
    int max_n, int grid_chains,
    P* d, P* d_best, double* de, double* d_bestL, curandStatePhilox4_32_10_t* rng,
    int n, double baseL, int steps, int epoch, int polish,
    double tmin, double tmax, double mv, double shrinkStep, unsigned long long* stats,
    const int* d_c_off = nullptr, const int* d_pn = nullptr,
    const double* d_pL = nullptr, const int* d_d_off = nullptr, int num_puzzles = 0
) {
    if (max_n <= NT)
        search_kernel<1, MIXED><<<grid_chains, NT>>>(d, d_best, de, d_bestL, rng, n, baseL, steps, epoch, polish, tmin, tmax, mv, shrinkStep, stats, d_c_off, d_pn, d_pL, d_d_off, num_puzzles);
    else if (max_n <= 4 * NT)
        search_kernel<4, MIXED><<<grid_chains, NT>>>(d, d_best, de, d_bestL, rng, n, baseL, steps, epoch, polish, tmin, tmax, mv, shrinkStep, stats, d_c_off, d_pn, d_pL, d_d_off, num_puzzles);
    else if (max_n <= 8 * NT)
        search_kernel<8, MIXED><<<grid_chains, NT>>>(d, d_best, de, d_bestL, rng, n, baseL, steps, epoch, polish, tmin, tmax, mv, shrinkStep, stats, d_c_off, d_pn, d_pL, d_d_off, num_puzzles);
    else
        search_kernel<16, MIXED><<<grid_chains, NT>>>(d, d_best, de, d_bestL, rng, n, baseL, steps, epoch, polish, tmin, tmax, mv, shrinkStep, stats, d_c_off, d_pn, d_pL, d_d_off, num_puzzles);
}

} // namespace dpack

// Ask #4: Multi-n Batched Execution Mode
// Triggered when argv[1] starts with "--batch:" followed by manifest file path.
// Manifest line format: seed_path out_json_path [chains_override]
static int run_batched_main(
    const std::string& manifest_path, int epochs, int steps, int default_chains,
    unsigned long long rng_seed, const std::string& summary_out
) {
    struct BatchItem {
        std::string seed_path;
        std::string out_path;
        int n = 0;
        int chains = 0;
        int chain_offset = 0;
        int data_offset = 0;
        double seed_L = 0.0;
        double best_L = 0.0;
        int shrinks = 0;
        std::vector<dpack::P> seed;
        std::vector<dpack::P> best;
    };

    std::ifstream mf(manifest_path);
    if (!mf) return 2;
    std::vector<BatchItem> items;
    std::string line;
    while (std::getline(mf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        BatchItem item;
        if (!(iss >> item.seed_path >> item.out_path)) continue;
        int c_override = 0;
        if (iss >> c_override && c_override > 0) item.chains = c_override;
        else item.chains = default_chains;
        items.push_back(std::move(item));
    }
    if (items.empty()) return 2;

    double tmin = envf("PACK_TMIN", 1e-11f, 1e-12f, 1.f);
    double tmax = envf("PACK_TMAX", 1e-4f, 1e-12f, 1.f);
    double mv   = envf("PACK_MOVE", .02f, .000001f, .3f);
    if (tmax < tmin) return 2;
    double shrinkStep = envf("PACK_SHRINK", .001f, .000001f, .1f);
    int mixed_prec = (int)envf("PACK_MIXED_PREC", 1.f, 0.f, 1.f);

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int max_n = 0;
    for (auto& it : items) {
        std::ifstream in(it.seed_path);
        if (!(in >> it.n >> it.seed_L)) return 2;
        if (it.n < 1 || it.n > 1000 || !std::isfinite(it.seed_L) || it.seed_L <= 0 || it.seed_L > 64) return 2;
        it.seed.resize(it.n);
        for (auto& p : it.seed) {
            in >> p.x >> p.y >> p.a;
            if (!in || !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.a)) return 2;
        }
        it.best = it.seed;
        it.best_L = it.seed_L;
        if (it.n > max_n) max_n = it.n;
    }

    int max_blocks_per_sm = 1;
    if (max_n <= NT) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, dpack::search_kernel<1, true>, NT, 0));
    else if (max_n <= 4 * NT) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, dpack::search_kernel<4, true>, NT, 0));
    else if (max_n <= 8 * NT) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, dpack::search_kernel<8, true>, NT, 0));
    else CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, dpack::search_kernel<16, true>, NT, 0));
    int wave_chains = ((prop.multiProcessorCount * std::max(1, max_blocks_per_sm)) / LADDER) * LADDER;
    if (wave_chains < LADDER) wave_chains = LADDER;

    int num_puzzles = (int)items.size();
    int per_puzzle_auto = std::max(LADDER, ((wave_chains / num_puzzles) / LADDER) * LADDER);

    int total_chains = 0, total_data = 0;
    std::vector<int> h_c_off(num_puzzles), h_pn(num_puzzles), h_d_off(num_puzzles);
    std::vector<double> h_pL(num_puzzles);
    for (int p = 0; p < num_puzzles; ++p) {
        if (items[p].chains <= 0) items[p].chains = per_puzzle_auto;
        items[p].chains = ((items[p].chains + LADDER - 1) / LADDER) * LADDER;
        items[p].chain_offset = total_chains;
        items[p].data_offset = total_data;
        h_c_off[p] = total_chains;
        h_pn[p] = items[p].n;
        h_d_off[p] = total_data;
        h_pL[p] = items[p].best_L;
        total_chains += items[p].chains;
        total_data += items[p].chains * items[p].n;
    }

    if (epochs == 0) epochs = (max_n <= 64) ? 50 : (max_n <= 256) ? 40 : 30;
    if (steps == 0)  steps  = (max_n <= 64) ? 64 : (max_n <= 256) ? 48 : 32;
    int eff_epochs = (epochs > 120) ? std::min(epochs, 80) : epochs;

    std::vector<dpack::P> h(total_data);
    std::vector<double> h_bestL(total_chains);
    for (int p = 0; p < num_puzzles; ++p) {
        std::vector<dpack::P> sub_h;
        dpack::build_portfolio(items[p].n, items[p].chains, items[p].seed_L, items[p].seed, sub_h);
        std::copy(sub_h.begin(), sub_h.end(), h.begin() + items[p].data_offset);
        for (int c = 0; c < items[p].chains; ++c) h_bestL[items[p].chain_offset + c] = items[p].seed_L;
    }

    dpack::P *d, *d_best; double *de, *d_bestL, *d_pL;
    int *d_c_off, *d_pn, *d_d_off;
    curandStatePhilox4_32_10_t* rng; unsigned long long* stats;
    CK(cudaMalloc(&d, total_data * sizeof(dpack::P)));
    CK(cudaMalloc(&d_best, total_data * sizeof(dpack::P)));
    CK(cudaMalloc(&de, total_chains * sizeof(double)));
    CK(cudaMalloc(&d_bestL, total_chains * sizeof(double)));
    CK(cudaMalloc(&d_pL, num_puzzles * sizeof(double)));
    CK(cudaMalloc(&d_c_off, num_puzzles * sizeof(int)));
    CK(cudaMalloc(&d_pn, num_puzzles * sizeof(int)));
    CK(cudaMalloc(&d_d_off, num_puzzles * sizeof(int)));
    CK(cudaMalloc(&rng, total_chains * sizeof(*rng)));
    CK(cudaMalloc(&stats, 4 * sizeof(*stats)));
    CK(cudaMemset(stats, 0, 4 * sizeof(*stats)));

    CK(cudaMemcpy(d, h.data(), total_data * sizeof(dpack::P), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_best, h.data(), total_data * sizeof(dpack::P), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_bestL, h_bestL.data(), total_chains * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pL, h_pL.data(), num_puzzles * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_c_off, h_c_off.data(), num_puzzles * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pn, h_pn.data(), num_puzzles * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_d_off, h_d_off.data(), num_puzzles * sizeof(int), cudaMemcpyHostToDevice));

    dpack::random_init<<<(total_chains + 255) / 256, 256>>>(rng, total_chains, rng_seed); CK(cudaGetLastError());
    dpack::structured_init_batch<<<total_chains, NT>>>(d, rng, total_chains, d_c_off, d_pn, d_pL, d_d_off, num_puzzles); CK(cudaGetLastError());

    auto start = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < eff_epochs; epoch++) {
        if (mixed_prec)
            dpack::dispatch_search<true>(max_n, total_chains, d, d_best, de, d_bestL, rng, max_n, 0.0, steps, epoch, 200, tmin, tmax, mv, shrinkStep, stats, d_c_off, d_pn, d_pL, d_d_off, num_puzzles);
        else
            dpack::dispatch_search<false>(max_n, total_chains, d, d_best, de, d_bestL, rng, max_n, 0.0, steps, epoch, 200, tmin, tmax, mv, shrinkStep, stats, d_c_off, d_pn, d_pL, d_d_off, num_puzzles);
        CK(cudaGetLastError());
        CK(cudaMemcpy(h_bestL.data(), d_bestL, total_chains * sizeof(double), cudaMemcpyDeviceToHost));

        bool any_improved = false;
        for (int p = 0; p < num_puzzles; ++p) {
            auto beg = h_bestL.begin() + items[p].chain_offset;
            auto end = beg + items[p].chains;
            int best_c = (int)(std::min_element(beg, end) - beg);
            double cand_L = beg[best_c];
            if (cand_L < items[p].best_L - 1e-11) {
                any_improved = true;
                double old_bestL = items[p].best_L;
                items[p].best_L = cand_L;
                h_pL[p] = cand_L;
                items[p].shrinks++;
                int n = items[p].n;
                CK(cudaMemcpy(items[p].best.data(), d_best + items[p].data_offset + best_c * n, n * sizeof(dpack::P), cudaMemcpyDeviceToHost));
                const double mults[8] = {0.02, 0.1, 0.4, 1.0, 2.5, 5.0, 10.0, 20.0};
                std::vector<dpack::P> sub_h(items[p].chains * n);
                CK(cudaMemcpy(sub_h.data(), d + items[p].data_offset, sub_h.size() * sizeof(dpack::P), cudaMemcpyDeviceToHost));
                for (int b = 0; b < items[p].chains; b++) {
                    if (b % LADDER == 0 || (b < LADDER && (old_bestL - cand_L > 5e-4))) {
                        double tL = std::max(std::sqrt((double)n) + 1e-6, cand_L - shrinkStep * mults[b % LADDER]);
                        double sc = tL / cand_L;
                        for (int i = 0; i < n; i++) sub_h[b * n + i] = {items[p].best[i].x * sc, items[p].best[i].y * sc, items[p].best[i].a};
                    }
                }
                CK(cudaMemcpy(d + items[p].data_offset, sub_h.data(), sub_h.size() * sizeof(dpack::P), cudaMemcpyHostToDevice));
            }
        }
        if (any_improved) {
            CK(cudaMemcpy(d_pL, h_pL.data(), num_puzzles * sizeof(double), cudaMemcpyHostToDevice));
        }
        dpack::swap_replicas<<<(total_chains + 1) / 2, NT>>>(d, de, rng, max_n, total_chains, epoch % 2, epoch, tmin, tmax, d_c_off, d_pn, d_d_off, num_puzzles);
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    unsigned long long h_stats[4];
    CK(cudaMemcpy(h_stats, stats, sizeof(h_stats), cudaMemcpyDeviceToHost));
    int feasible = std::max(1, (int)h_stats[2]);

    for (int p = 0; p < num_puzzles; ++p) {
        std::ofstream out(items[p].out_path);
        if (!out) return 2;
        out.precision(17);
        out << "{\"n\":" << items[p].n << ",\"target_L\":" << items[p].best_L
            << ",\"seconds\":" << sec / num_puzzles
            << ",\"batch_seconds\":" << sec << ",\"batch_size\":" << num_puzzles
            << ",\"chain_steps\":" << (long long)items[p].chains * eff_epochs * steps
            << ",\"proposals\":" << h_stats[1] / num_puzzles
            << ",\"feasible_checkpoints\":" << feasible
            << ",\"shrinks\":" << items[p].shrinks
            << ",\"chains\":" << items[p].chains << ",\"total_batch_chains\":" << total_chains
            << ",\"mixed_prec\":" << mixed_prec
            << ",\"sm_count\":" << prop.multiProcessorCount << ",\"blocks_per_sm\":" << max_blocks_per_sm
            << ",\"wave_chains\":" << wave_chains << ",\"device\":\"" << prop.name << "\",\"q\":[";
        for (size_t i = 0; i < items[p].best.size(); i++) {
            if (i) out << ",";
            out << "[" << items[p].best[i].x << "," << items[p].best[i].y << "," << items[p].best[i].a << "]";
        }
        out << "]}\n";
    }
    if (!summary_out.empty()) {
        std::ofstream sout(summary_out);
        if (sout) {
            sout.precision(17);
            sout << "{\"batch_size\":" << num_puzzles << ",\"total_chains\":" << total_chains
                 << ",\"seconds\":" << sec << ",\"mixed_prec\":" << mixed_prec << "}\n";
        }
    }

    CK(cudaFree(d)); CK(cudaFree(d_best)); CK(cudaFree(de)); CK(cudaFree(d_bestL));
    CK(cudaFree(d_pL)); CK(cudaFree(d_c_off)); CK(cudaFree(d_pn)); CK(cudaFree(d_d_off));
    CK(cudaFree(rng)); CK(cudaFree(stats));
    return 0;
}

int main(int argc,char**argv){
 if(argc!=7){std::cerr<<"packing seed.txt|--batch:manifest.txt epochs steps chains random_seed output.json\n";return 2;}
 int epochs=atoi(argv[2]),steps=atoi(argv[3]),chains=atoi(argv[4]),n=0;
 if(epochs<0||epochs>100000||steps<0||steps>4096||chains<0||chains>4096||(chains>0&&chains%8))return 2;
 unsigned long long rng_seed = strtoull(argv[5], nullptr, 10);
 std::string arg1(argv[1]);
 if (arg1.rfind("--batch:", 0) == 0) {
  return run_batched_main(arg1.substr(8), epochs, steps, chains, rng_seed, argv[6]);
 }
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
 int mixed_prec = (int)envf("PACK_MIXED_PREC", 1.f, 0.f, 1.f);

 cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,0));
 int max_blocks_per_sm = 1;
 if (n <= NT) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, dpack::search_kernel<1, true>, NT, 0));
 else if (n <= 4 * NT) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, dpack::search_kernel<4, true>, NT, 0));
 else if (n <= 8 * NT) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, dpack::search_kernel<8, true>, NT, 0));
 else CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, dpack::search_kernel<16, true>, NT, 0));
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
  if (mixed_prec)
   dpack::dispatch_search<true>(n, chains, d, d_best, de, d_bestL, rng, n, bestL, steps, epoch, 200, tmin, tmax, mv, shrinkStep, stats);
  else
   dpack::dispatch_search<false>(n, chains, d, d_best, de, d_bestL, rng, n, bestL, steps, epoch, 200, tmin, tmax, mv, shrinkStep, stats);
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
    <<",\"chains\":"<<chains<<",\"mixed_prec\":"<<mixed_prec<<",\"sm_count\":"<<prop.multiProcessorCount<<",\"blocks_per_sm\":"<<max_blocks_per_sm
    <<",\"wave_chains\":"<<wave_chains<<",\"device\":\""<<prop.name<<"\",\"q\":[";
 for(size_t i=0;i<best.size();i++){if(i)out<<",";out<<"["<<best[i].x<<","<<best[i].y<<","<<best[i].a<<"]";}
 out<<"]}\n";
 CK(cudaFree(d));CK(cudaFree(d_best));CK(cudaFree(de));CK(cudaFree(d_bestL));CK(cudaFree(rng));CK(cudaFree(stats));
 return out?0:2;
}
