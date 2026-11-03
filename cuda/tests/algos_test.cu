#define ALGOS_NO_MAIN
#include "../algos.cu"
#include <cassert>
int main(){
 using namespace algos;
 P p={.7,.8,.21},q={1.31,1.03,.63};G g={0,0,0};
 pair_cached_smooth(p,q,cos(p.a),sin(p.a),cos(q.a),sin(q.a),&g);
 double worst=0;
 for(int k=0;k<3;k++){
  P a=p,b=p;double h=1e-6;
  if(k==0){a.x+=h;b.x-=h;}if(k==1){a.y+=h;b.y-=h;}if(k==2){a.a+=h;b.a-=h;}
  double fd=(pair_cached(a,q,cos(a.a),sin(a.a),cos(q.a),sin(q.a))-pair_cached(b,q,cos(b.a),sin(b.a),cos(q.a),sin(q.a)))/(2*h);
  double analytic=k==0?g.x:k==1?g.y:g.a;worst=std::max(worst,fabs(fd-analytic));
 }
 if(worst>1e-7){std::cerr<<"gradient error "<<worst<<"\n";return 1;}
 std::cout<<"SAT analytic gradient max finite-difference error "<<worst<<"\n";
}
