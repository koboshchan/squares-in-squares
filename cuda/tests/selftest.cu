// Independent float64 SAT oracle plus grid/all-pairs CUDA cross-check.
#define main packing_program_main
#include "../packing.cu"
#undef main
#include <random>
#include <algorithm>
double oracle(Q p,Q q){
 double c=cos(double(p.a)),s=sin(double(p.a)),d=cos(double(q.a)),t=sin(double(q.a));
 double axes[4][2]={{c,s},{-s,c},{d,t},{-t,d}};
 double v=1e9;
 for(auto& a:axes){
  double r=.5*(fabs(a[0]*c+a[1]*s)+fabs(-a[0]*s+a[1]*c)+fabs(a[0]*d+a[1]*t)+fabs(-a[0]*t+a[1]*d));
  v=std::min(v,r-fabs((double(p.x)-q.x)*a[0]+(double(p.y)-q.y)*a[1]));
 }
 return std::max(0.,v);
}
__global__ void verify(Q* input,float* pairs,float* griderr){
 __shared__ Q q[NMAX];__shared__ int head[CELLS],next[NMAX];
 int t=threadIdx.x,b=blockIdx.x;
 for(int j=t;j<CELLS;j+=NT)head[j]=-1;
 if(t<NMAX)q[t]=input[b*NMAX+t];__syncthreads();
 if(t<NMAX)next[t]=atomicExch(head+cell(q[t]),t);__syncthreads();
 if(t<NMAX){
  float full=wall(q[t],12);
  for(int j=0;j<NMAX;j++){
   float d=depth(q[t],q[j]);pairs[(b*NMAX+t)*NMAX+j]=d;
   if(j!=t)full+=d*d;
  }
  griderr[b*NMAX+t]=fabsf(local(q[t],t,q,head,next,12)-full);
 }
}
int main(){
 constexpr int blocks=100;
 std::mt19937 rng(923);std::uniform_real_distribution<float> coord(-.2f,12.2f),angle(-3.14f,3.14f);
 std::vector<Q> q(blocks*NMAX);
 for(auto& p:q)p={coord(rng),coord(rng),angle(rng)};
 // Boundary/touch/just-separated fixtures at cell boundaries.
 q[0]={2,2,0};q[1]={3,2,0};q[2]={3.000001f,2,0};q[3]={2.999999f,2,0};
 Q* d;float *pairs,*err;
 CK(cudaMalloc(&d,q.size()*sizeof(Q)));CK(cudaMalloc(&pairs,blocks*NMAX*NMAX*sizeof(float)));CK(cudaMalloc(&err,q.size()*sizeof(float)));
 CK(cudaMemcpy(d,q.data(),q.size()*sizeof(Q),cudaMemcpyHostToDevice));
 verify<<<blocks,NT>>>(d,pairs,err);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
 std::vector<float> hp(blocks*NMAX*NMAX),he(q.size());
 CK(cudaMemcpy(hp.data(),pairs,hp.size()*sizeof(float),cudaMemcpyDeviceToHost));
 CK(cudaMemcpy(he.data(),err,he.size()*sizeof(float),cudaMemcpyDeviceToHost));
 double maxPair=0,maxGrid=0;
 for(int b=0;b<blocks;b++)for(int i=0;i<NMAX;i++){
  maxGrid=std::max(maxGrid,double(he[b*NMAX+i]));
  for(int j=0;j<NMAX;j++)maxPair=std::max(maxPair,fabs(hp[(b*NMAX+i)*NMAX+j]-oracle(q[b*NMAX+i],q[b*NMAX+j])));
 }
 std::cout<<"250000 SAT pairs max abs float64 error "<<maxPair<<"; 5000 grid/all-pairs energy max error "<<maxGrid<<"\n";
 CK(cudaFree(d));CK(cudaFree(pairs));CK(cudaFree(err));
 return maxPair>2e-6||maxGrid>1e-4?1:0;
}
