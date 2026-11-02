// MIT. Float search only; Python's official interval certifier owns acceptance.
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
int main(int argc,char**argv){
 if(argc!=7){std::cerr<<"packing seed.txt epochs steps chains random_seed output.json\n";return 2;}
 int epochs=atoi(argv[2]),steps=atoi(argv[3]),chains=atoi(argv[4]),n;
 if(epochs<1||epochs>100000||steps<1||steps>4096||chains<8||chains>4096||chains%8)return 2;
 std::ifstream in(argv[1]);float L=0;n=0;if(!(in>>n>>L))return 2;if(n<1||n>1000||!std::isfinite(L)||L<=0||L>64)return 2;
 std::vector<Q> seed(n);for(auto& p:seed){in>>p.x>>p.y>>p.a;if(!in||!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.a))return 2;}
 std::vector<Q> h(n*chains),best;std::vector<float> energy(chains);float bestL=L;
 for(int b=0;b<chains;b++)for(int i=0;i<n;i++)h[b*n+i]=seed[i];
 Q* d;float* de;curandStatePhilox4_32_10_t* rng;unsigned long long* counts;
 CK(cudaMalloc(&d,h.size()*sizeof(Q)));CK(cudaMalloc(&de,chains*sizeof(float)));
 CK(cudaMalloc(&rng,chains*NT*sizeof(*rng)));CK(cudaMalloc(&counts,sizeof(*counts)));
 CK(cudaMemset(counts,0,sizeof(*counts)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(Q),cudaMemcpyHostToDevice));
 init_rng<<<(chains*NT+255)/256,256>>>(rng,chains,strtoull(argv[5],nullptr,10));CK(cudaGetLastError());CK(cudaDeviceSynchronize());
 Settings config{envf("PACK_TMIN",1e-7f,1e-12f,1.f),envf("PACK_TMAX",.01f,1e-12f,1.f),
                 envf("PACK_MOVE",.015f,.000001f,.12f),envf("PACK_ROTATION",.12f,.000001f,1.57f)};
 if(config.tmax<config.tmin)return 2;
 CK(cudaMemcpyToSymbol(settings,&config,sizeof(config)));
 float shrinkStep=envf("PACK_SHRINK",.002f,.000001f,.1f);
 auto start=std::chrono::steady_clock::now();int feasible=0,shrink=0;
 for(int epoch=0;epoch<epochs;epoch++){
  float cool=powf(.05f,float(epoch%100)/100);
  if(n<=NMAX&&n<=NT)walk<<<chains,NT>>>(d,de,rng,n,L,steps,cool,counts);
  else walk_global<<<chains,NT>>>(d,de,rng,n,L,steps,cool,counts);
  CK(cudaGetLastError());
  CK(cudaMemcpy(energy.data(),de,chains*sizeof(float),cudaMemcpyDeviceToHost));
  int winner=-1;for(int b=0;b<chains;b++)if(energy[b]==0){winner=b;break;}
  if(winner>=0){
   best.resize(n);CK(cudaMemcpy(best.data(),d+winner*n,n*sizeof(Q),cudaMemcpyDeviceToHost));bestL=L;feasible++;
   // Common fixed L across ladder is necessary for valid replica exchange.
   float newL=fmaxf(sqrtf(float(n)),L-shrinkStep);if(newL<L){
    CK(cudaMemcpy(h.data(),d,h.size()*sizeof(Q),cudaMemcpyDeviceToHost));
    for(int b=0;b<chains;b++)for(int i=0;i<n;i++){
     Q p=b%LADDER==0?best[i]:h[b*n+i];p.x*=newL/L;p.y*=newL/L;h[b*n+i]=p;
    }
    L=newL;shrink++;CK(cudaMemcpy(d,h.data(),h.size()*sizeof(Q),cudaMemcpyHostToDevice));
   }
  } else {exchange<<<(chains+1)/2,NT>>>(d,de,rng,n,chains,epoch%2,cool);CK(cudaGetLastError());}
 }
 CK(cudaDeviceSynchronize());double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 unsigned long long moves;CK(cudaMemcpy(&moves,counts,sizeof(moves),cudaMemcpyDeviceToHost));
 cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,0));
 std::ofstream out(argv[6]);if(!out)return 2;out.precision(10);
 out<<"{\"n\":"<<n<<",\"target_L\":"<<bestL<<",\"seconds\":"<<sec<<",\"chain_steps\":"<<(long long)chains*epochs*steps
    <<",\"proposals\":"<<moves<<",\"feasible_checkpoints\":"<<feasible<<",\"shrinks\":"<<shrink<<",\"device\":\""<<prop.name<<"\",\"q\":[";
 for(size_t i=0;i<best.size();i++){if(i)out<<",";out<<"["<<best[i].x<<","<<best[i].y<<","<<best[i].a<<"]";}
 out<<"]}\n"; CK(cudaFree(d));CK(cudaFree(de));CK(cudaFree(rng));CK(cudaFree(counts));return out?0:2;
}
