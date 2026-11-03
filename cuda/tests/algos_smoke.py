"""Bounded GPU output/determinism tests, independently C++ gated both ways."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--executable', default='cuda/algos')
    p.add_argument('--certifier', default='cuda/certifier')
    a=p.parse_args()
    exe=str(Path(a.executable).resolve());cert=str(Path(a.certifier).resolve())
    count=0
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp)
        for n in (1,5,19,70,1000):
            width=__import__('math').isqrt(n-1)+1
            points=[((i%width+.5)*1.00002,(i//width+.5)*1.00002,0) for i in range(n)]
            seed=d/'seed.txt'
            seed.write_text(f'{n} {width*1.00002:.17g}\n'+''.join(' '.join(map(str,q))+'\n' for q in points))
            for mode in (1,2,3):
                prev=None
                for repeat in range(2):
                    raw=d/'raw.json'
                    subprocess.run([exe,str(seed),'2','4','2','42',str(raw),str(mode)],check=True,timeout=90)
                    obj=json.loads(raw.read_text());assert obj['n']==n and len(obj['q'])==n
                    if prev is not None:assert obj['q']==prev, (n,mode,'nondeterministic')
                    prev=obj['q']
                csv=d/'candidate.csv'
                csv.write_text('id,n,square_id,x,y,angle\n'+''.join(f'{i},{n},{i},{x:.17g},{y:.17g},{angle:.17g}\n' for i,(x,y,angle) in enumerate(prev)))
                for rt in ([],['--roundtrip']):
                    report=json.loads(subprocess.check_output([cert,str(csv),'--max-n',str(n),*rt],text=True))
                    assert report['accepted']==1 and report['missing']==n-1 and report['rejected']==0,report
                    count+=1
                print(f'n={n} mode={mode} deterministic; literal+roundtrip accepted',flush=True)
    print(f'{count} C++ gates passed, 30 deterministic solver invocations')
if __name__=='__main__':main()
