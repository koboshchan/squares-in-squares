#define main seeded_program_main
#include "seeded.cu"
#undef main
#include <random>
double oracle(P p,P q){
 double c=cos(p.a),s=sin(p.a),d=cos(q.a),t=sin(q.a);
 double ax[4][2]={{c,s},{-s,c},{d,t},{-t,d}},gap=-1e100;
 for(auto& a:ax){double r=.5*(fabs(a[0]*c+a[1]*s)+fabs(-a[0]*s+a[1]*c)+fabs(a[0]*d+a[1]*t)+fabs(-a[0]*t+a[1]*d));
 gap=std::max(gap,fabs((p.x-q.x)*a[0]+(p.y-q.y)*a[1])-r);}
 return pow(std::max(0.,-gap),2);
}
__global__ void check(P* q,double* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n)out[i]=pair(q[2*i],q[2*i+1]);}
int main(){
 std::mt19937 r(832);std::uniform_real_distribution<double> u(-1,1);
 const int n=10000;std::vector<P> q(2*n);double err=0,jerr=0;
 for(int i=0;i<n;i++){
  P p{u(r),u(r),u(r)},z{u(r),u(r),u(r)};q[2*i]=p;q[2*i+1]=z;
  G g{0,0,0};double e=pair(p,z,&g);err=std::max(err,fabs(e-oracle(p,z)));
  for(int k=0;k<3;k++){P a=p,b=p;double h=1e-7;
   if(k==0){a.x+=h;b.x-=h;}if(k==1){a.y+=h;b.y-=h;}if(k==2){a.a+=h;b.a-=h;}
   double fd=(pair(a,z)-pair(b,z))/(2*h),v=k==0?g.x:k==1?g.y:g.a;
   jerr=std::max(jerr,fabs(fd-v));}
 }
 P* d;double* out;CK(cudaMalloc(&d,q.size()*sizeof(P)));CK(cudaMalloc(&out,n*sizeof(double)));
 CK(cudaMemcpy(d,q.data(),q.size()*sizeof(P),cudaMemcpyHostToDevice));check<<<(n+255)/256,256>>>(d,out,n);
 std::vector<double> h(n);CK(cudaMemcpy(h.data(),out,n*sizeof(double),cudaMemcpyDeviceToHost));
 for(int i=0;i<n;i++)err=std::max(err,fabs(h[i]-oracle(q[2*i],q[2*i+1])));
 std::cout<<"SAT host/device oracle error "<<err<<" gradient finite-difference error "<<jerr<<"\n";
 CK(cudaFree(d));CK(cudaFree(out));return err>1e-12||jerr>1e-7;
}
