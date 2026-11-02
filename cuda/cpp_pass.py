import argparse
from pathlib import Path
from cpp_loop import run_pass
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed",type=int,required=True);p.add_argument("--incumbent")
    a=p.parse_args();run_pass(a.directory,a.seed)
