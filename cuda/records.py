"""Import published geometry, preserving sources and an unchanged official gate.

Requires the author's separately installed parse_svg_packing.py tool.
No third-party SVG or parser code is vendored here.
"""
from __future__ import annotations
import contextlib,hashlib,io,json,math,re,subprocess,sys
from decimal import Decimal,localcontext
from pathlib import Path
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import numpy as np
from squares.core import certify
from polish import certify_q
from scheduler import atomic_json
SOURCE="https://kingbird.myphotos.cc/packing/squares_in_squares.html"
TOOLS="https://github.com/Davidebyzero/packing_squares_in_squares__tools"
LICENSE="No express reuse license found; factual geometry for authorized research; originals not redistributed."

def catalogue(html):
    soup=BeautifulSoup(html,"html.parser");out=[]
    for box in soup.select("div.box"):
        links=box.find_all("a",href=re.compile(r"^square-.*\.svg$"))
        if not links:continue
        lead=box.find("font",size="+3")
        if not lead:continue
        label=lead.get_text(" ",strip=True).split()[0].rstrip(",")
        nums=re.findall(r"\d+",lead.get_text(" ",strip=True))
        text=box.get_text(" ",strip=True)
        targets=re.findall(r"\\Nn\{([\d.]+)\}",text)
        for link in links:
            m=re.match(r"square-(\d+)",link["href"])
            if not m:continue
            n=int(m[1])
            if n>1000:continue
            # Preserve the source cell, even when n=2 represents 2 and 3.
            out.append(dict(n=n,url=urljoin(SOURCE,link["href"]),
                            published_L=min(targets,key=Decimal) if targets else None,
                            catalogue_text=text,license_status=LICENSE))
    return list({x["url"]:x for x in out}.values())

def fetch(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".part")
    subprocess.run(["curl","--fail","--silent","--show-error","--location",
                    "--max-time","45","--retry","2","--retry-delay","2",
                    "-A","Mozilla/5.0 (authorized square packing research)",url,
                    "-o",str(temp)],check=True,timeout=160)
    if temp.stat().st_size>8_000_000:raise ValueError("source too large")
    temp.replace(path)

def extract(path,tool_directory,expected_n):
    sys.path.insert(0,str(tool_directory))
    from parse_svg_packing import parse_svg
    with contextlib.redirect_stdout(io.StringIO()):
        sq,entities,precision,centered=parse_svg(str(path),65)
    if len(sq)!=expected_n:raise ValueError(f"expected {expected_n} got {len(sq)}")
    # Angles are already radians in SVG x/y coordinates, not the CLI's reflected output.
    q=np.array([[float(s["x"]),float(s["y"]),float(s["theta"])] for s in sq])
    if not np.isfinite(q).all():raise ValueError("non-finite SVG geometry")
    # Independently check float64 SAT before interval acceptance.
    from polish import constraints
    extent=.5*(abs(np.cos(q[:,2]))+abs(np.sin(q[:,2])))
    q[:,:2]-=(q[:,:2]-extent[:,None]).min(axis=0)
    L=float((q[:,:2]+extent[:,None]).max())
    margin=float(constraints(np.r_[q.ravel(),L],expected_n).min())
    if margin < -1e-7:raise ValueError(f"float64 SAT failed {margin}")
    # Preserve full precision first; no coordinate optimization.
    with localcontext() as ctx:
        ctx.prec=85
        for expansion in ["1e-24","1e-20","1e-16"]:
            scale=1+Decimal(expansion)
            rows=[dict(square_id=str(i),x=str(s["x"]*scale),
                       y=str(s["y"]*scale),angle=str(s["theta"])) for i,s in enumerate(sq)]
            try:
                side=certify(expected_n,rows)
                return dict(n=expected_n,L=side,rows=rows,expansion=expansion,
                            float64_min_constraint=margin,source_precision=precision)
            except ValueError:pass
    rows,_,expansion=certify_q(q)
    return dict(n=expected_n,L=certify(expected_n,rows),rows=rows,
                expansion=expansion,float64_min_constraint=margin,source_precision=precision)

def gaps(state,entries):
    targets={}
    for e in entries:
        if e.get("published_L"):
            n=e["n"];v=e["published_L"]
            if n not in targets or Decimal(v)<Decimal(targets[n]):targets[n]=v
    result=[]
    for n in range(1,1001):
        current=state[str(n)]["L"];target=targets.get(n)
        # Catalogue explicitly guarantees trivial packing for omitted n <=324 only.
        if target is None and n<=324:target=str(math.isqrt(n-1)+1)
        delta=None if target is None else max(0.,(float(current)-float(target))/math.sqrt(n))
        result.append(dict(n=n,incumbent_L=current,published_L=target,score_gap=delta,
                           source=SOURCE,target_status="published" if n in targets else
                           "catalogue omitted <=324 trivial" if n<=324 else "unknown, not invented"))
    return result
