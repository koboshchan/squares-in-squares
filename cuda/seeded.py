"""Seeded GPU experiment bookkeeping, without CPU optimization or provenance loss."""
from __future__ import annotations
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from squares.core import certify
from polish import certify_q
from scheduler import atomic_json

INCUMBENT_SHA256="e202208231d31ac82efd36b04f7f59014ac505709729e72619fe2136bd8f8cac"

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def rank_targets(state, targets, max_n=128):
    """Potential score decrease, not time-normalized invented performance."""
    rows=[]
    for n, target in targets.items():
        if 1<=n<=max_n and str(n) in state:
            side=float(state[str(n)]["L"])
            gap=(side*side-float(target)**2)/n
            if gap>1e-12: rows.append(dict(n=n,incumbent_L=state[str(n)]["L"],
                                         target_L=str(target),score_gap=gap))
    return sorted(rows,key=lambda row:(-row["score_gap"],row["n"]))

def write_seed(path, rows, provenance, dilation=1.00000002):
    n=len(rows);side=certify(n,rows)
    if not 1<=n<=128 or not math.isfinite(dilation) or not 1<=dilation<=1.001:
        raise ValueError("invalid seeded search input")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(f"{n} {float(side)*dilation:.17g}\n"+"".join(
        " ".join(format(float(row[k])*(dilation if k!="angle" else 1),".17g")
                 for k in ("x","y","angle"))+"\n" for row in rows))
    meta=dict(n=n,L=side,source=provenance,seed_sha256=digest(path),
              center_dilation=dilation)
    atomic_json(path.with_suffix(".json"),meta)
    return meta

def certify_raw(raw, incumbent_L, seed_metadata):
    """Direct GPU output only. Dilation clears FP error; no SLSQP/minimizer."""
    n=raw["n"];q=np.asarray(raw["q"],dtype=np.float64)
    if n!=seed_metadata["n"] or q.shape!=(n,3) or not np.isfinite(q).all():
        raise ValueError("invalid/mismatched GPU output")
    rows,_,dilation=certify_q(q)
    side=certify(n,rows)
    current=Decimal(incumbent_L)
    new=Decimal(side);seed_L=Decimal(seed_metadata["L"])
    # Imported records are never claimed as earned gains against analytic seeds.
    imported="record" in seed_metadata["source"].lower()
    gpu_earned=(new<current and new<seed_L and not imported)
    return dict(n=n,L=side,rows=rows,source="seeded GPU SA + on-device gradient",
                seed=seed_metadata,raw_sha256=hashlib.sha256(
                    json.dumps(raw,sort_keys=True).encode()).hexdigest(),
                strict_incumbent_improvement=new<current,
                improves_undilated_seed=new<seed_L,
                gpu_earned=gpu_earned,imported_record=imported,
                score_delta=float((current*current-new*new)/Decimal(n)),
                dilation=dilation,cpu_optimizer=False,
                seconds=raw["seconds"],collective=raw["collective"],
                device_polish_iterations=raw["polish"])
