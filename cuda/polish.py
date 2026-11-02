"""Float64 constrained local polishing. Only the official interval gate accepts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from squares.core import certify, write_submission

def constraints(z, n, jacobian=False):
    q = z[:-1].reshape(n, 3)
    xy, a, L = q[:, :2], q[:, 2], z[-1]
    i, j = np.triu_indices(n, 1)
    phi = np.stack([a[i], a[i]+np.pi/2, a[j], a[j]+np.pi/2], axis=1)
    c, s = np.cos(phi), np.sin(phi)
    d = xy[j]-xy[i]
    proj = d[:, 0, None]*c+d[:, 1, None]*s
    di, dj = a[i, None]-phi, a[j, None]-phi
    def radius(t):
        return .5*(abs(np.cos(t))+abs(np.sin(t)))
    gaps = abs(proj)-radius(di)-radius(dj)
    k = np.argmax(gaps, axis=1)
    p = np.arange(len(i))
    extent = radius(a)
    wall = np.stack([xy[:,0]-extent, L-xy[:,0]-extent,
                     xy[:,1]-extent, L-xy[:,1]-extent], axis=1)
    values = np.r_[gaps[p,k], wall.ravel()]
    if not jacobian:
        return values
    jac = np.zeros((len(values), len(z)))
    cs, sn = c[p,k], s[p,k]
    sign = np.sign(proj[p,k])
    jac[p,3*i] = -sign*cs
    jac[p,3*i+1] = -sign*sn
    jac[p,3*j] = sign*cs
    jac[p,3*j+1] = sign*sn
    def derivative(t):
        return .5*(-np.sign(np.cos(t))*np.sin(t)+np.sign(np.sin(t))*np.cos(t))
    ri, rj = derivative(di[p,k]), derivative(dj[p,k])
    phase = sign*(-d[:,0]*sn+d[:,1]*cs)+ri+rj
    jac[p,3*i+2] = -ri+phase*(k<2)
    jac[p,3*j+2] = -rj+phase*(k>=2)
    r = len(i)+4*np.arange(n)
    for o, col, val in [(0,0,1),(1,0,-1),(2,1,1),(3,1,-1)]:
        jac[r+o,3*np.arange(n)+col] = val
        jac[r+o,3*np.arange(n)+2] = -derivative(a)
    jac[r+1,-1] = 1
    jac[r+3,-1] = 1
    return jac

def certify_q(q):
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError("invalid coordinates")
    # Center dilation clears residual floating error; L includes every bit of it.
    for dilation in (1+1e-10, 1+1e-9, 1+1e-8, 1+1e-7, 1+1e-6):
        rows = [dict(square_id=str(i), x=format(x*dilation,'.17g'),
                     y=format(y*dilation,'.17g'), angle=format(a,'.17g'))
                for i,(x,y,a) in enumerate(q)]
        try:
            return rows, float(certify(len(q), rows)), dilation
        except ValueError:
            pass
    raise ValueError("official interval certification failed")

def polish(q, maxiter=400):
    q = np.asarray(q, dtype=float).copy()
    n = len(q)
    if not 1 <= n <= 50 or q.shape != (n,3) or not np.isfinite(q).all():
        raise ValueError("dense polish supports finite n=1..50")
    extent = .5*(abs(np.cos(q[:,2]))+abs(np.sin(q[:,2])))
    lo = (q[:,:2]-extent[:,None]).min(axis=0)
    q[:,:2] -= lo
    L = float((q[:,:2]+extent[:,None]).max())
    z = np.r_[q.ravel(), L+1e-6]
    best_z = z.copy()
    best_L = float(z[-1])
    def track_constraints(x):
        nonlocal best_z, best_L
        c = constraints(x, n)
        if np.isfinite(x).all() and c.min() >= -1e-7 and float(x[-1]) <= best_L:
            best_z = x.copy()
            best_L = float(x[-1])
        return c
    grad = np.zeros_like(z); grad[-1] = 1
    result = minimize(lambda x:x[-1], z, jac=lambda x:grad,
                      method="SLSQP", constraints=[dict(type="ineq",
                      fun=track_constraints,
                      jac=lambda x:constraints(x,n,True))],
                      bounds=[(-2,34),(-2,34),(-100,100)]*n+[(np.sqrt(n),34)],
                      options=dict(maxiter=maxiter, ftol=1e-12))
    candidates = [result.x, best_z, z]
    best_out = None
    for cand_x in candidates:
        try:
            rows, side, dilation = certify_q(cand_x[:-1].reshape(n,3))
            if best_out is None or side < best_out[1]:
                best_out = (rows, side, dilation, cand_x)
        except ValueError:
            pass
    if best_out is None:
        raise ValueError("official interval certification failed")
    rows, side, dilation, chosen_x = best_out
    return rows, dict(n=n, L=side, dilation=dilation, iterations=int(result.nit),
                      converged=bool(result.success), message=str(result.message),
                      min_constraint=float(constraints(chosen_x,n).min()))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--output",type=Path,default=Path("outputs/polish"))
    p.add_argument("--maxiter",type=int,default=400)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    report=[]; packings={}
    for path in a.inputs:
        raw=json.loads(path.read_text()); start=time.monotonic()
        try:
            rows,r=polish(raw["q"],a.maxiter)
            r.update(source=str(path),seconds=time.monotonic()-start)
            if r["n"] not in packings or r["L"]<float(certify(r["n"],packings[r["n"]])):
                packings[r["n"]]=rows
            report.append(r)
        except ValueError as e:
            report.append(dict(source=str(path),rejected=str(e)))
        print(json.dumps(report[-1]),flush=True)
    if packings: write_submission(a.output/"certified.csv",packings)
    (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
if __name__=="__main__":
    main()
