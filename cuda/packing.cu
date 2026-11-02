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
constexpr int NMAX=50, NT=64, CELLS=256, LADDER=8;
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
 float temp=1e-7f*powf(100000.f,(b%LADDER)/float(LADDER-1))*cooling;
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
    float scale=(step%8==0?.12f:.015f)*sqrtf(cooling);
    Q p=q[t];p.x+=scale*(2*curand_uniform(&r)-1);p.y+=scale*(2*curand_uniform(&r)-1);
    p.a+=scale*(2*curand_uniform(&r)-1);p.a=remainderf(p.a,1.57079632679f);
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
__global__ void exchange(Q* q,float* e,curandStatePhilox4_32_10_t* rng,int n,int chains,int parity,float cooling){
 int a=blockIdx.x*2+parity,b=a+1;
 if(b>=chains||a/LADDER!=b/LADDER)return;
 __shared__ int accept;
 if(threadIdx.x==0){
  auto r=rng[a*NT];float ta=1e-7f*powf(100000.f,(a%LADDER)/float(LADDER-1))*cooling;
  float tb=1e-7f*powf(100000.f,(b%LADDER)/float(LADDER-1))*cooling;
  accept=logf(curand_uniform(&r))<(1/ta-1/tb)*(e[a]-e[b]);
  rng[a*NT]=r;if(accept){float z=e[a];e[a]=e[b];e[b]=z;}
 } __syncthreads();
 if(accept&&threadIdx.x<n){int i=threadIdx.x;Q z=q[a*n+i];q[a*n+i]=q[b*n+i];q[b*n+i]=z;}
}
int main(int argc,char**argv){
 if(argc!=7){std::cerr<<"packing seed.txt epochs steps chains random_seed output.json\n";return 2;}
 int epochs=atoi(argv[2]),steps=atoi(argv[3]),chains=atoi(argv[4]),n;
 if(epochs<1||epochs>100000||steps<1||steps>4096||chains<8||chains>4096||chains%8)return 2;
 std::ifstream in(argv[1]);float L=0;n=0;if(!(in>>n>>L))return 2;if(n<1||n>NMAX||!std::isfinite(L)||L<=0||L>16)return 2;
 std::vector<Q> seed(n);for(auto& p:seed){in>>p.x>>p.y>>p.a;if(!in||!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.a))return 2;}
 std::vector<Q> h(n*chains),best;std::vector<float> energy(chains);float bestL=L;
 for(int b=0;b<chains;b++)for(int i=0;i<n;i++)h[b*n+i]=seed[i];
 Q* d;float* de;curandStatePhilox4_32_10_t* rng;unsigned long long* counts;
 CK(cudaMalloc(&d,h.size()*sizeof(Q)));CK(cudaMalloc(&de,chains*sizeof(float)));
 CK(cudaMalloc(&rng,chains*NT*sizeof(*rng)));CK(cudaMalloc(&counts,sizeof(*counts)));
 CK(cudaMemset(counts,0,sizeof(*counts)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(Q),cudaMemcpyHostToDevice));
 init_rng<<<(chains*NT+255)/256,256>>>(rng,chains,strtoull(argv[5],nullptr,10));CK(cudaGetLastError());CK(cudaDeviceSynchronize());
 auto start=std::chrono::steady_clock::now();int feasible=0,shrink=0;
 for(int epoch=0;epoch<epochs;epoch++){
  float cool=powf(.05f,float(epoch%100)/100);
  walk<<<chains,NT>>>(d,de,rng,n,L,steps,cool,counts);CK(cudaGetLastError());
  CK(cudaMemcpy(energy.data(),de,chains*sizeof(float),cudaMemcpyDeviceToHost));
  int winner=-1;for(int b=0;b<chains;b++)if(energy[b]==0){winner=b;break;}
  if(winner>=0){
   best.resize(n);CK(cudaMemcpy(best.data(),d+winner*n,n*sizeof(Q),cudaMemcpyDeviceToHost));bestL=L;feasible++;
   // Common fixed L across ladder is necessary for valid replica exchange.
   float newL=fmaxf(sqrtf(float(n)),L-.002f);if(newL<L){
    for(int b=0;b<chains;b++)for(int i=0;i<n;i++){Q p=best[i];p.x*=newL/L;p.y*=newL/L;h[b*n+i]=p;}
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
