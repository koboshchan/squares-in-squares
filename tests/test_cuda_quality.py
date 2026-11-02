"""Quality and scalable scheduler host regression tests."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/"cuda"))
from polish import constraints, polish
from scheduler import strip_construction, priority, atomic_json
from squares.core import certify, grid
from scipy.optimize._numdiff import approx_derivative

def test_analytic_jacobian():
    z=np.r_[np.random.default_rng(5).uniform(.1,3,(5,3)).ravel(),4]
    expected=approx_derivative(lambda x:constraints(x,5),z,method="3-point")
    assert np.max(abs(expected-constraints(z,5,True)))<1e-7

@pytest.mark.parametrize("k",[2,3,4,7])
def test_derived_strips(k):
    rows=strip_construction(k)
    assert len(rows)==k*k+1
    assert k+.70710678<float(certify(len(rows),rows))<k+.7071069

def test_polish_rejects_invalid():
    with pytest.raises(ValueError):polish([[float("nan"),0,0]])
    with pytest.raises(ValueError):polish(np.zeros((51,3)))

def test_polish_preserves_official_gate():
    rows=strip_construction(2)
    q=[[float(r[k]) for k in ("x","y","angle")] for r in rows]
    out,report=polish(q)
    assert float(certify(5,out))==report["L"]
    assert report["L"]<2.707107

def test_priority_and_atomic_state(tmp_path):
    assert priority(10,4,0)>priority(10,4,1)>0
    assert priority(100,10,0)==0
    p=tmp_path/"state.json";atomic_json(p,{"attempts":1})
    assert p.read_text().endswith("\n")
    assert not p.with_suffix(".tmp").exists()

@pytest.mark.parametrize("a,count",[(1,5),(2,10),(4,27),(5,38)])
def test_goebel_analytic_record_family(a,count):
    from scheduler import goebel
    rows=goebel(a)
    assert len(rows)==count
    target=a+1+1/2**.5
    assert abs(float(certify(count,rows))-target)<1e-14
