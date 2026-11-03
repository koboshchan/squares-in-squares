// MIT. FP64 experimental search; only the external certifier may accept output.
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<"\n";exit(2);} }while(0)
namespace algos {
constexpr int NT=128;
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


// Deterministic block reductions. No atomics or concurrent coordinate writes.
__device__ double reduce(double v,double* tmp,int kind=0){
 int t=threadIdx.x; tmp[t]=v; __syncthreads();
 for(int s=NT/2;s;s/=2){if(t<s)tmp[t]=kind==1?fmax(tmp[t],tmp[t+s]):kind==2?fmin(tmp[t],tmp[t+s]):tmp[t]+tmp[t+s];__syncthreads();}
 double out=tmp[0]; __syncthreads(); return out;
}
__device__ double energy(const P* q,int n,double L,double* tmp){
 double e=0;for(int i=threadIdx.x;i<n;i+=NT){e+=boundary(q[i],L);double c=cos(q[i].a),s=sin(q[i].a);
 for(int j=i+1;j<n;j++)e+=pair_cached(q[i],q[j],c,s,cos(q[j].a),sin(q[j].a));}
 return reduce(e,tmp);
}
__device__ G gradient(const P* q,int i,int n,double L){
 P p=q[i];double c=cos(p.a),s=sin(p.a),r=.5*(fabs(c)+fabs(s)),dr=dr_smooth(c,s);G g={0,0,0};
 double v[4]={r-p.x,p.x+r-L,r-p.y,p.y+r-L};
 for(int k=0;k<4;k++)if(v[k]>0){double z=2*v[k];g.a+=z*dr;if(k==0)g.x-=z;if(k==1)g.x+=z;if(k==2)g.y-=z;if(k==3)g.y+=z;}
 for(int j=0;j<n;j++)if(j!=i)pair_cached_smooth(p,q[j],c,s,cos(q[j].a),sin(q[j].a),&g);
 return g;
}
// Uniform center dilation makes every SAT pair strictly separated. This is
// only a floating search filter, NOT a certificate. Normalize rotated bbox.
__device__ double checkpoint(P* q,P* best,int n,double bestL,double* tmp){
 double lam=1;
 for(int i=threadIdx.x;i<n;i+=NT){double c=cos(q[i].a),s=sin(q[i].a);
 for(int j=i+1;j<n;j++){
  double dx=q[i].x-q[j].x,dy=q[i].y-q[j].y;
  if(dx*dx+dy*dy>2.00001)continue;
  double d=cos(q[j].a),t=sin(q[j].a),r=.5+.5*(fabs(c*d+s*t)+fabs(s*d-c*t));
  double vmax=fmax(fmax(fabs(dx*c+dy*s),fabs(-dx*s+dy*c)),fmax(fabs(dx*d+dy*t),fabs(-dx*t+dy*d)));
  lam=fmax(lam,vmax>1e-14?(r+3e-10)/vmax:1e100);
 }}
 lam=reduce(lam,tmp,1);if(lam>1.03)return bestL;
 double lx=1e100,ly=1e100,hx=-1e100,hy=-1e100;
 for(int i=threadIdx.x;i<n;i+=NT){double r=.5*(fabs(cos(q[i].a))+fabs(sin(q[i].a)));double x=q[i].x*lam,y=q[i].y*lam;
 lx=fmin(lx,x-r);ly=fmin(ly,y-r);hx=fmax(hx,x+r);hy=fmax(hy,y+r);}
 lx=reduce(lx,tmp,2);ly=reduce(ly,tmp,2);hx=reduce(hx,tmp,1);hy=reduce(hy,tmp,1);
 double L=fmax(hx-lx,hy-ly);
 if(L<bestL){for(int i=threadIdx.x;i<n;i+=NT)best[i]={q[i].x*lam-lx,q[i].y*lam-ly,wrap_angle(q[i].a)};__syncthreads();return L;}
 return bestL;
}
__global__ void init_rng(curandStatePhilox4_32_10_t* rng,unsigned long long seed){curand_init(seed,blockIdx.x,0,rng+blockIdx.x);}
// One chain/block. q, trial and gradient storage live in bounded global arrays,
// avoiding n-dependent shared memory / launch resource failures at n=1000.
__global__ void search(P* all,P* trial,G* gradients,P* best,double* sides,double* targets,int* hops,
 curandStatePhilox4_32_10_t* rng,unsigned long long* counts,int n,int steps,int mode,int epoch){
 int b=blockIdx.x,t=threadIdx.x;P* q=all+b*n;P* prop=trial+b*n;P* keep=best+b*n;G* grad=gradients+b*n;
 __shared__ double tmp[NT],L,alpha;__shared__ int accepted,kick;
 auto r=rng[b];double bestL=sides[b];unsigned long long iterations=0;
 if(t==0){L=targets[b];kick=hops[b];}__syncthreads();
 // Mode 2 adaptive basin perturbations. Others start with small thermal moves.
 if(mode==2){
  if(t==0){for(int k=0;k<kick;k++){int i=curand(&r)%n;
   q[i].x+=.10*(2*curand_uniform_double(&r)-1);q[i].y+=.10*(2*curand_uniform_double(&r)-1);q[i].a=wrap_angle(q[i].a+.20*(2*curand_uniform_double(&r)-1));}}
  __syncthreads();
 }
 double e=energy(q,n,L,tmp);
 for(int step=0;step<steps;step++){
  iterations++;
  // Serial single-square MC move is race-free and preserves detailed energy.
  if(mode==1 && step%8==0){
   if(t==0){int i=curand(&r)%n;P old=q[i],p=old;double amp=.005+.03/(1+epoch);
    p.x+=amp*(2*curand_uniform_double(&r)-1);p.y+=amp*(2*curand_uniform_double(&r)-1);p.a=wrap_angle(p.a+amp*(2*curand_uniform_double(&r)-1));
    double a=boundary(old,L),z=boundary(p,L);
    for(int j=0;j<n;j++)if(j!=i){a+=pair_cached(old,q[j],cos(old.a),sin(old.a),cos(q[j].a),sin(q[j].a));z+=pair_cached(p,q[j],cos(p.a),sin(p.a),cos(q[j].a),sin(q[j].a));}
    double temp=1e-8/(1+epoch);if(z<=a||curand_uniform_double(&r)<exp((a-z)/temp))q[i]=p;
   }__syncthreads();e=energy(q,n,L,tmp);
  }
  double gmax=0;
  for(int i=t;i<n;i+=NT){grad[i]=gradient(q,i,n,L);gmax=fmax(gmax,fmax(fabs(grad[i].x),fmax(fabs(grad[i].y),fabs(grad[i].a))));}
  gmax=reduce(gmax,tmp,1);if(gmax<1e-14)break;
  if(t==0){alpha=fmin(.25,.05/gmax);accepted=0;}__syncthreads();
  for(int bt=0;bt<16;bt++){
   for(int i=t;i<n;i+=NT)prop[i]={q[i].x-alpha*grad[i].x,q[i].y-alpha*grad[i].y,wrap_angle(q[i].a-alpha*grad[i].a)};
   __syncthreads();double next=energy(prop,n,L,tmp);
   if(next<e){for(int i=t;i<n;i+=NT)q[i]=prop[i];__syncthreads();e=next;if(t==0)accepted=1;__syncthreads();break;}
   if(t==0)alpha*=.5;__syncthreads();
  }
  if(!accepted)break;
 }
 double prev=bestL;bestL=checkpoint(q,keep,n,bestL,tmp);
 // LS shrinks after progress; unsuccessful compression backs off geometrically.
 // Basin hopping only retains lower feasible minima; adaptive kick grows by epoch.
 if(t==0){double delta=(mode==1?.001:.002)*(1+(b%8)) / (1+epoch*.05);
  if(bestL<prev){L=fmax(sqrt(double(n)),bestL-delta);}
  else if(mode==1){L=.5*(L+bestL);}else{L=fmax(sqrt(double(n)),bestL-delta);}
  counts[b]+=iterations;
  hops[b]=bestL<prev?max(1,kick-1):min(min(n,16),kick+1);
  sides[b]=bestL;targets[b]=L;rng[b]=r;}
 __syncthreads();
 if(mode==2 || bestL<prev){for(int i=t;i<n;i+=NT){q[i]=keep[i];q[i].x*=L/bestL;q[i].y*=L/bestL;}__syncthreads();}
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


// CPU family portfolio: grid, rotated lattice, grid core with tilted strips,
// and diagonal-band constructions with corner grids. Overlap is polished, never
// assumed valid. Chain zero always preserves the incumbent trajectory.
static std::vector<P> family(int n,int b,double L,const std::vector<P>& seed){
 if(b==0)return seed;
 std::vector<P> q;int w=std::max(1,int(std::ceil(std::sqrt(double(n))))-(b%3));
 double a=(b%8)*std::acos(-1.)/32, pitch=1.000000001;
 if(b%3==0){for(int i=0;i<n;i++)q.push_back({.5+(i%w)*pitch,.5+(i/w)*pitch,0});}
 else if(b%3==1){int core=std::max(1,w-1-(b/8)%3);for(int x=0;x<core && int(q.size())<n;x++)for(int y=0;y<core && int(q.size())<n;y++)q.push_back({x+.5,y+.5,0});
 int rem=n-int(q.size());for(int i=0;i<rem;i++){bool top=i%2;double v=.7+(L-1.4)*(i/2+.5)/std::max(1,(rem+1)/2);
 q.push_back(top?P{v,(core+L)*.5,a}:P{(core+L)*.5,v,a});}}
 else {double c=std::cos(a),s=std::sin(a);for(int i=0;i<n;i++){double u=(i%w)*pitch,v=(i/w)*pitch;q.push_back({u*c-v*s,u*s+v*c,a});}}
 double lx=1e100,ly=1e100;for(auto p:q){double r=.5*(fabs(cos(p.a))+fabs(sin(p.a)));lx=std::min(lx,p.x-r);ly=std::min(ly,p.y-r);}
 for(auto& p:q){p.x-=lx;p.y-=ly;}return q;
}
} // namespace
#ifndef ALGOS_NO_MAIN
int main(int argc,char**argv){
 if(argc!=7&&argc!=8){std::cerr<<"algos seedfile epochs steps chains seed output [mode1|2|3]\n";return 2;}
 int epochs=atoi(argv[2]),steps=atoi(argv[3]),chains=atoi(argv[4]);const char* env=getenv("ALGOS_MODE");int mode=argc==8?atoi(argv[7]):env?atoi(env):1;
 if(epochs<0||epochs>100000||steps<0||steps>4096||chains<0||chains>4096||mode<1||mode>3)return 2;
 if(!epochs)epochs=40;if(!steps)steps=64;
 int n;double L;std::ifstream in(argv[1]);if(!(in>>n>>L)||n<1||n>1000||!std::isfinite(L)||L<=0||L>64)return 2;
 using namespace algos;std::vector<P> seed(n);for(auto& p:seed)if(!(in>>p.x>>p.y>>p.a)||!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.a))return 2;
 cudaDeviceProp device;CK(cudaGetDeviceProperties(&device,0));if(!chains)chains=std::min(256,std::max(8,device.multiProcessorCount));
 std::vector<P> h(n*chains),best=seed;for(int b=0;b<chains;b++){auto q=mode==3?family(n,b,L,seed):seed;std::copy(q.begin(),q.end(),h.begin()+b*n);}
 if(mode==3){std::vector<P> enumerated;build_portfolio(n,chains,L,seed,enumerated);
 for(int b=0;b<chains;b++)if(b%3==2)std::copy(enumerated.begin()+b*n,enumerated.begin()+(b+1)*n,h.begin()+b*n);}
 std::vector<P> keeps(n*chains);for(int b=0;b<chains;b++)std::copy(seed.begin(),seed.end(),keeps.begin()+b*n);
 std::vector<double> sides(chains,L),targets(chains);for(int b=0;b<chains;b++)targets[b]=std::max(std::sqrt(double(n)),L-.001*(1+b%8));
 P *d,*trial,*keep;G* grad;double *ds,*dt;int* hops;unsigned long long* counts;curandStatePhilox4_32_10_t* rng;
 CK(cudaMalloc(&d,h.size()*sizeof(P)));CK(cudaMalloc(&trial,h.size()*sizeof(P)));CK(cudaMalloc(&keep,h.size()*sizeof(P)));CK(cudaMalloc(&grad,h.size()*sizeof(G)));
 CK(cudaMalloc(&counts,chains*sizeof(*counts)));CK(cudaMemset(counts,0,chains*sizeof(*counts)));
 CK(cudaMalloc(&hops,chains*sizeof(int)));std::vector<int> initial_hops(chains,1);CK(cudaMemcpy(hops,initial_hops.data(),chains*sizeof(int),cudaMemcpyHostToDevice));
 CK(cudaMalloc(&ds,chains*sizeof(double)));CK(cudaMalloc(&dt,chains*sizeof(double)));CK(cudaMalloc(&rng,chains*sizeof(*rng)));
 CK(cudaMemcpy(d,h.data(),h.size()*sizeof(P),cudaMemcpyHostToDevice));CK(cudaMemcpy(keep,keeps.data(),h.size()*sizeof(P),cudaMemcpyHostToDevice));
 CK(cudaMemcpy(ds,sides.data(),chains*sizeof(double),cudaMemcpyHostToDevice));CK(cudaMemcpy(dt,targets.data(),chains*sizeof(double),cudaMemcpyHostToDevice));
 init_rng<<<chains,1>>>(rng,strtoull(argv[5],nullptr,10));CK(cudaGetLastError());
 auto start=std::chrono::steady_clock::now();double bestL=L;
 for(int e=0;e<epochs;e++){
  search<<<chains,NT>>>(d,trial,grad,keep,ds,dt,hops,rng,counts,n,steps,mode,e);CK(cudaGetLastError());
  CK(cudaMemcpy(sides.data(),ds,chains*sizeof(double),cudaMemcpyDeviceToHost));int b=int(std::min_element(sides.begin(),sides.end())-sides.begin());
  if(sides[b]<bestL){bestL=sides[b];CK(cudaMemcpy(best.data(),keep+b*n,n*sizeof(P),cudaMemcpyDeviceToHost));}
 }
 CK(cudaDeviceSynchronize());double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::vector<unsigned long long> measured(chains);CK(cudaMemcpy(measured.data(),counts,chains*sizeof(*counts),cudaMemcpyDeviceToHost));unsigned long long total_steps=0;for(auto x:measured)total_steps+=x;
 std::ofstream out(argv[6]);out<<std::setprecision(17)<<"{\"n\":"<<n<<",\"mode\":"<<mode<<",\"target_L\":"<<bestL<<",\"seconds\":"<<seconds
 <<",\"chains\":"<<chains<<",\"chain_steps\":"<<total_steps<<",\"proposals\":0,\"feasible_checkpoints\":0,\"shrinks\":0,\"q\":[";
 for(int i=0;i<n;i++){if(i)out<<",";out<<"["<<best[i].x<<","<<best[i].y<<","<<best[i].a<<"]";}out<<"]}\n";
 CK(cudaFree(d));CK(cudaFree(trial));CK(cudaFree(keep));CK(cudaFree(grad));CK(cudaFree(counts));CK(cudaFree(hops));CK(cudaFree(ds));CK(cudaFree(dt));CK(cudaFree(rng));return out?0:2;
}

#endif
