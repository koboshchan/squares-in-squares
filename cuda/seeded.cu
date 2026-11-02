// MIT. Double-precision seeded GPU search. Acceptance belongs to official metric.
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
#include <vector>
#define CK(x) do{auto ck_error=(x);if(ck_error!=cudaSuccess){std::cerr<<cudaGetErrorString(ck_error)<<"\n";exit(2);}}while(0)
#ifndef SEED_THREADS
#define SEED_THREADS 64
#endif
constexpr int NT=SEED_THREADS, MAXN=128, LADDER=8;
static_assert(NT==32||NT==64||NT==128||NT==256,"threads");
struct P {double x,y,a;};
struct G {double x,y,a;};
__host__ __device__ double sg(double x){return (x>0)-(x<0);}
__host__ __device__ double rad(double x){return .5*(fabs(cos(x))+fabs(sin(x)));}
__host__ __device__ double dr(double x){return .5*(-sg(cos(x))*sin(x)+sg(sin(x))*cos(x));}
// Squared SAT penetration with the derivative of the active separating axis.
__host__ __device__ double pair_cached(P p,P q,double c,double s,double d,double t,G* grad=nullptr){
 double cd=c*d+s*t,sd=s*d-c*t,dx=q.x-p.x,dy=q.y-p.y;
 double radius=.5+.5*(fabs(cd)+fabs(sd));
 double deriv=.5*(-sg(cd)*sd+sg(sd)*cd);
 double ux[4]={c,-s,d,-t},uy[4]={s,c,t,d};
 double gap=-1e100,gx=0,gy=0,ga=0;
 for(int k=0;k<4;k++){
  double v=dx*ux[k]+dy*uy[k],z=fabs(v)-radius;
  if(z>gap){gap=z;gx=-sg(v)*ux[k];gy=-sg(v)*uy[k];
   ga=-deriv+(k<2?sg(v)*(-dx*uy[k]+dy*ux[k]):0);}
 }
 if(gap>=0)return 0;
 if(grad){grad->x+=2*gap*gx;grad->y+=2*gap*gy;grad->a+=2*gap*ga;}
 return gap*gap;
}
__host__ __device__ double pair(P p,P q,G* grad=nullptr){
 return pair_cached(p,q,cos(p.a),sin(p.a),cos(q.a),sin(q.a),grad);
}
__host__ __device__ double boundary(P p,double L,G* g=nullptr){
 double r=rad(p.a),v[4]={r-p.x,p.x+r-L,r-p.y,p.y+r-L},e=0;
 for(int k=0;k<4;k++)if(v[k]>0){e+=v[k]*v[k];if(g){
  double f=2*v[k];g->x+=f*(k==0?-1:k==1?1:0);
  g->y+=f*(k==2?-1:k==3?1:0);g->a+=f*dr(p.a);}}
 return e;
}
__device__ double total(P* q,int n,double L,double* tmp){
 int t=threadIdx.x;double e=0;
 __shared__ double c[MAXN],s[MAXN];
 for(int i=t;i<n;i+=NT){c[i]=cos(q[i].a);s[i]=sin(q[i].a);}__syncthreads();
 for(int i=t;i<n;i+=NT){e+=boundary(q[i],L);for(int j=i+1;j<n;j++)e+=pair_cached(q[i],q[j],c[i],s[i],c[j],s[j]);}
 tmp[t]=e;__syncthreads();
 for(int s=NT/2;s;s/=2){if(t<s)tmp[t]+=tmp[t+s];__syncthreads();}
 return tmp[0];
}
__global__ void random_init(curandStatePhilox4_32_10_t* r,int chains,unsigned long long seed){
 int b=blockIdx.x*blockDim.x+threadIdx.x;if(b<chains)curand_init(seed,b,0,r+b);
}
// One chain per block, immutable proposal snapshot and block-wide energy test.
// Collective moves deliberately cross the independent-single-square local basins.
__global__ void search(P* all,double* energies,curandStatePhilox4_32_10_t* rng,
 int n,double L,int steps,int epoch,int collective,int polish,double tmax,double baseMove,
 unsigned long long* stats){
 __shared__ P q[MAXN],p[MAXN];__shared__ G grads[MAXN];
 __shared__ double cachedc[MAXN],cacheds[MAXN];
 __shared__ double tmp[NT],energy,move,rot,cx,cy,theta,scale,alpha,adapt;
 __shared__ int chosen,mode,accept,windowAccepted;__shared__ curandStatePhilox4_32_10_t r;
 int t=threadIdx.x,b=blockIdx.x;
 for(int i=t;i<n;i+=NT)q[i]=all[b*n+i];
 if(t==0){r=rng[b];adapt=1;windowAccepted=0;}__syncthreads();
 double e=total(q,n,L,tmp);if(t==0)energy=e;__syncthreads();
 // Cool within each reheat, preserving all replicas between epochs.
 double cool=pow(.002,double(epoch%200)/199);
 double temp=1e-11*pow(tmax/1e-11,(b%LADDER)/7.0)*cool;
 for(int step=0;step<steps;step++){
  if(t==0){
   chosen=curand(&r)%n;mode=collective?(curand(&r)%6):0;
   cx=q[chosen].x;cy=q[chosen].y;
   // Adapt scale to the ladder and reheat phase, with rare large kicks.
   scale=adapt*baseMove*(.15+1.85*(b%LADDER)/7.0)*sqrt(cool);
   if(step%16==0)scale*=4;
   move=scale*(2*curand_uniform_double(&r)-1);
   rot=scale*(2*curand_uniform_double(&r)-1);
   theta=scale*.7*(2*curand_uniform_double(&r)-1);
  }__syncthreads();
  for(int i=t;i<n;i+=NT){
   P z=q[i];double dx=z.x-cx,dy=z.y-cy;
   bool take=mode==0?i==chosen:mode==1?fabs(dy)<.55:mode==2?fabs(dx)<.55:
             mode==3?dx*dx+dy*dy<2.5:mode==4?fabs(remainder(z.a-q[chosen].a,1.5707963267948966))<.05:dy>0;
   if(take){
    if(mode==3||mode==4){double c=cos(theta),s=sin(theta);
     z.x=cx+c*dx-s*dy+move;z.y=cy+s*dx+c*dy+rot;z.a+=theta;}
    else {z.x+=move;z.y+=rot;z.a+=theta;}
   }p[i]=z;
  }__syncthreads();double ne=total(p,n,L,tmp);
  if(t==0){accept=ne<=energy||log(curand_uniform_double(&r))<(energy-ne)/temp;
   atomicAdd(stats+1,1ULL);if(accept){energy=ne;windowAccepted++;atomicAdd(stats+2,1ULL);}}
  __syncthreads();if(accept)for(int i=t;i<n;i+=NT)q[i]=p[i];__syncthreads();
  if(t==0&&(step+1)%16==0){adapt=fmax(.1,fmin(4.,adapt*exp((windowAccepted/16.0-.25)*.5)));windowAccepted=0;}
  __syncthreads();
 }
 // Monotone on-device gradient descent with parallel analytic gradients and
 // backtracking. No host optimizer changes the result or gets GPU credit.
 if(t==0)alpha=.08;__syncthreads();
 for(int iter=0;iter<polish;iter++){
  for(int i=t;i<n;i+=NT){cachedc[i]=cos(q[i].a);cacheds[i]=sin(q[i].a);}__syncthreads();
  for(int i=t;i<n;i+=NT){G g{0,0,0};boundary(q[i],L,&g);
   for(int j=0;j<n;j++)if(i!=j)pair_cached(q[i],q[j],cachedc[i],cacheds[i],cachedc[j],cacheds[j],&g);grads[i]=g;}
  __syncthreads();
  for(int i=t;i<n;i+=NT){G g=grads[i];P z=q[i];
   p[i]={z.x-alpha*g.x,z.y-alpha*g.y,z.a-alpha*.4*g.a};}
  __syncthreads();double ne=total(p,n,L,tmp);
  if(t==0){accept=ne<energy;if(accept){energy=ne;alpha=fmin(.15,alpha*1.03);}else alpha*=.5;}
  __syncthreads();if(accept)for(int i=t;i<n;i+=NT)q[i]=p[i];__syncthreads();
 }
 for(int i=t;i<n;i+=NT)all[b*n+i]=q[i];
 if(t==0){energies[b]=energy;rng[b]=r;atomicAdd(stats,1ULL);}
}
__global__ void swap_replicas(P* q,double* e,curandStatePhilox4_32_10_t* rng,
 int n,int chains,int parity,int epoch,double tmax,unsigned long long* stats){
 int a=blockIdx.x*2+parity,b=a+1,t=threadIdx.x;
 if(b>=chains||a/LADDER!=b/LADDER)return;
 __shared__ int take;
 if(t==0){auto r=rng[a];double cool=pow(.002,double(epoch%200)/199);
  double ta=1e-11*pow(tmax/1e-11,(a%8)/7.0)*cool,tb=1e-11*pow(tmax/1e-11,(b%8)/7.0)*cool;
  take=log(curand_uniform_double(&r))<(1/ta-1/tb)*(e[a]-e[b]);rng[a]=r;
  atomicAdd(stats+3,1ULL);if(take){double z=e[a];e[a]=e[b];e[b]=z;atomicAdd(stats+4,1ULL);}}
 __syncthreads();if(take)for(int i=t;i<n;i+=NT){P z=q[a*n+i];q[a*n+i]=q[b*n+i];q[b*n+i]=z;}
}
double env(const char* k,double v,double lo,double hi){
 const char* s=getenv(k);if(!s)return v;char* end;v=strtod(s,&end);
 if(*end||!std::isfinite(v)||v<lo||v>hi){std::cerr<<"invalid "<<k<<"\n";exit(2);}return v;
}
int main(int argc,char**argv){
 if(argc!=7){std::cerr<<"seeded seed.txt epochs steps chains seed output.json\n";return 2;}
 int epochs=atoi(argv[2]),steps=atoi(argv[3]),chains=atoi(argv[4]),n=0;
 if(epochs<1||epochs>100000||steps<0||steps>4096||chains<8||chains>4096||chains%8)return 2;
 double L=0;std::ifstream in(argv[1]);in>>n>>L;
 if(!in||n<1||n>MAXN||!std::isfinite(L)||L<=0||L>64)return 2;
 std::vector<P> seed(n);for(auto& p:seed){in>>p.x>>p.y>>p.a;if(!in||!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.a))return 2;}
 double shrink=env("SEED_SHRINK",.001,1e-7,.1),tmax=env("SEED_TMAX",.0001,1e-11,.1),mv=env("SEED_MOVE",.02,1e-6,.3);
 int collective=env("SEED_COLLECTIVE",1,0,1),polish=env("SEED_POLISH",120,0,4096);
 std::vector<P> h(n*chains),best;std::vector<double> e(chains);
 for(int b=0;b<chains;b++)for(int i=0;i<n;i++)h[b*n+i]=seed[i];
 P* d;double* de;curandStatePhilox4_32_10_t* rng;unsigned long long* stats;
 CK(cudaMalloc(&d,h.size()*sizeof(P)));CK(cudaMalloc(&de,chains*sizeof(double)));
 CK(cudaMalloc(&rng,chains*sizeof(*rng)));CK(cudaMalloc(&stats,5*sizeof(*stats)));CK(cudaMemset(stats,0,5*sizeof(*stats)));
 CK(cudaMemcpy(d,h.data(),h.size()*sizeof(P),cudaMemcpyHostToDevice));
 random_init<<<(chains+255)/256,256>>>(rng,chains,strtoull(argv[5],nullptr,10));CK(cudaGetLastError());
 double bestL=L;int shrinks=0;auto start=std::chrono::steady_clock::now();
 for(int epoch=0;epoch<epochs;epoch++){
  search<<<chains,NT>>>(d,de,rng,n,L,steps,epoch,collective,polish,tmax,mv,stats);CK(cudaGetLastError());
  CK(cudaMemcpy(e.data(),de,chains*sizeof(double),cudaMemcpyDeviceToHost));
  int winner=std::min_element(e.begin(),e.end())-e.begin();
  if(e[winner]<1e-19){
   best.resize(n);CK(cudaMemcpy(best.data(),d+winner*n,n*sizeof(P),cudaMemcpyDeviceToHost));bestL=L;
   double next=std::max(sqrt(double(n)),L-shrink);
   if(next<L){CK(cudaMemcpy(h.data(),d,h.size()*sizeof(P),cudaMemcpyDeviceToHost));
    for(int b=0;b<chains;b++)for(int i=0;i<n;i++){P z=b%8==0?best[i]:h[b*n+i];z.x*=next/L;z.y*=next/L;h[b*n+i]=z;}
    L=next;shrinks++;CK(cudaMemcpy(d,h.data(),h.size()*sizeof(P),cudaMemcpyHostToDevice));}
  }else {swap_replicas<<<(chains+1)/2,NT>>>(d,de,rng,n,chains,epoch%2,epoch,tmax,stats);CK(cudaGetLastError());}
 }
 CK(cudaDeviceSynchronize());double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 unsigned long long counts[5];CK(cudaMemcpy(counts,stats,sizeof(counts),cudaMemcpyDeviceToHost));
 std::ofstream out(argv[6]);out<<std::setprecision(17)<<"{\"n\":"<<n<<",\"target_L\":"<<bestL<<",\"seconds\":"<<seconds<<",\"shrinks\":"<<shrinks
 <<",\"epochs\":"<<epochs<<",\"steps\":"<<steps<<",\"chains\":"<<chains<<",\"collective\":"<<collective<<",\"polish\":"<<polish
 <<",\"proposals\":"<<counts[1]<<",\"accepted\":"<<counts[2]<<",\"exchange_attempts\":"<<counts[3]<<",\"exchange_accepts\":"<<counts[4]<<",\"q\":[";
 for(size_t i=0;i<best.size();i++){if(i)out<<",";out<<"["<<best[i].x<<","<<best[i].y<<","<<best[i].a<<"]";}out<<"]}\n";
 CK(cudaFree(d));CK(cudaFree(de));CK(cudaFree(rng));CK(cudaFree(stats));return out?0:2;
}
