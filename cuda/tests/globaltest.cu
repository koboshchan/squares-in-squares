// Global n=1000 snapshot, energy and strided exchange regression.
#define main packing_program_main
#include "../packing.cu"
#undef main
#include <algorithm>
int main(){
 constexpr int n=1000,chains=8;float L=32.0001f;
 std::vector<Q> h(n*chains);
 for(int b=0;b<chains;b++)for(int i=0;i<n;i++)h[b*n+i]={(i%32+.5f)*1.000001f,(i/32+.5f)*1.000001f,0};
 Q* d;float* device_energy;curandStatePhilox4_32_10_t* r;unsigned long long* count;
 CK(cudaMalloc(&d,h.size()*sizeof(Q)));CK(cudaMalloc(&device_energy,chains*sizeof(float)));
 CK(cudaMalloc(&r,chains*NT*sizeof(*r)));CK(cudaMalloc(&count,sizeof(*count)));
 CK(cudaMemcpy(d,h.data(),h.size()*sizeof(Q),cudaMemcpyHostToDevice));CK(cudaMemset(count,0,sizeof(*count)));
 Settings config{1e-7f,.01f,.015f,.12f};CK(cudaMemcpyToSymbol(settings,&config,sizeof(config)));
 init_rng<<<(chains*NT+255)/256,256>>>(r,chains,142);
 walk_global<<<chains,NT>>>(d,device_energy,r,n,L,64,.3f,count);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
 std::vector<float> energies(chains);CK(cudaMemcpy(energies.data(),device_energy,chains*sizeof(float),cudaMemcpyDeviceToHost));
 CK(cudaMemcpy(h.data(),d,h.size()*sizeof(Q),cudaMemcpyDeviceToHost));
 double maxerr=0;
 for(int b=0;b<chains;b++){
  double sum=0;
  for(int i=0;i<n;i++){
   Q p=h[b*n+i];float radius=.5f*(fabsf(cosf(p.a))+fabsf(sinf(p.a)));
   float dx=fmaxf(0,fmaxf(radius-p.x,p.x+radius-L));
   float dy=fmaxf(0,fmaxf(radius-p.y,p.y+radius-L));sum+=dx*dx+dy*dy;
   for(int j=i+1;j<n;j++){
    // Host independently reproduces SAT, with double projection arithmetic.
    Q q=h[b*n+j];double c=cos(double(p.a)),s=sin(double(p.a)),u=cos(double(q.a)),v=sin(double(q.a));
    double axes[4][2]={{c,s},{-s,c},{u,v},{-v,u}},overlap=1e9;
    for(auto& a:axes){
     double rad=.5*(fabs(a[0]*c+a[1]*s)+fabs(-a[0]*s+a[1]*c)+fabs(a[0]*u+a[1]*v)+fabs(-a[0]*v+a[1]*u));
     overlap=std::min(overlap,rad-fabs((double(p.x)-q.x)*a[0]+(double(p.y)-q.y)*a[1]));
    }
    overlap=std::max(0.,overlap);sum+=overlap*overlap;
   }
  }
  maxerr=std::max(maxerr,fabs(sum-energies[b]));
 }
 // Force exchange acceptance. All n coordinates, including the tail, must swap.
 std::vector<Q> before=h;std::fill(energies.begin(),energies.end(),0);
 CK(cudaMemcpy(device_energy,energies.data(),energies.size()*sizeof(float),cudaMemcpyHostToDevice));
 exchange<<<chains/2,NT>>>(d,device_energy,r,n,chains,0,1);CK(cudaDeviceSynchronize());
 CK(cudaMemcpy(h.data(),d,h.size()*sizeof(Q),cudaMemcpyDeviceToHost));
 for(int b=0;b<chains;b++)for(int i=0;i<n;i++){
  Q a=h[b*n+i],z=before[(b^1)*n+i];if(a.x!=z.x||a.y!=z.y||a.a!=z.a)return 3;
 }
 std::cout<<"n1000 global energy max float64 error "<<maxerr<<"; all strided exchange coordinates exact\n";
 CK(cudaFree(d));CK(cudaFree(device_energy));CK(cudaFree(r));CK(cudaFree(count));
 return maxerr>1e-5?1:0;
}
