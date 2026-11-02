"""Compare certified decimal sides with cited published decimal targets.

A published decimal is rounded/truncated evidence, not an exact algebraic value.
"""
from decimal import Decimal,localcontext
import argparse,json
from pathlib import Path
from scheduler import atomic_json
from squares.core import certify,grid

SOURCE="https://kingbird.myphotos.cc/packing/squares_in_squares.html"
TARGETS={5:"2.70710678118654",10:"3.70710678118654",
11:"3.87708359002281",17:"4.67553009360455",18:"4.82287565553229",
19:"4.88561808316412",26:"5.62132034355964",27:"5.70710678118654",
37:"6.59861960924436",50:"7.57142857142857"}

def comparison(actual,target,tolerance="1e-14"):
    with localcontext() as ctx:
        ctx.prec=80
        delta=Decimal(actual)-Decimal(target)
        return dict(certified_L=actual,published_L=target,
                    signed_difference=str(delta),absolute_difference=str(abs(delta)),
                    tolerance=tolerance,within_tolerance=abs(delta)<=Decimal(tolerance),
                    exactly_equal_to_published_decimal=delta==0)

def report(directory):
    output={}
    for n,target in TARGETS.items():
        path=directory/"best"/f"{n}.json"
        item=json.loads(path.read_text()) if path.exists() else dict(rows=grid(n),source="exact grid")
        output[str(n)]=dict(comparison(certify(n,item["rows"]),target),source=item["source"])
    result=dict(source=SOURCE,comparison="absolute side-length tolerance, not exact record equality",
                safety="All measured extents include safety clearance; analytic seeds use center dilation 1+1e-40.",
                cases=output)
    atomic_json(directory/"record-comparison.json",result)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("directory",type=Path)
    print(json.dumps(report(parser.parse_args().directory),indent=2))
