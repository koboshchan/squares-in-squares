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

def test_coverage_merge_resume_and_baseline(tmp_path):
    import json
    import importlib
    batch=importlib.import_module("batch")
    d=tmp_path; (d/"best").mkdir()
    state={str(n):dict(L=str(__import__("math").isqrt(n-1)+1),attempts=0,source="exact grid")
           for n in range(1,6)}
    atomic_json(d/"state.json",state)
    batch.prepare(d,5,8,16,8)
    jobs=json.loads((d/"batch"/"manifest.json").read_text())
    assert sorted(j["n"] for j in jobs)==[2,3,5]
    raw=d/"raw";raw.mkdir()
    atomic_json(raw/"2.json",dict(n=2,q=[[.5,.5,0],[1.5,.5,0]],seconds=1,chain_steps=128))
    batch.merge(d,raw);batch.merge(d,raw)
    state=json.loads((d/"state.json").read_text())
    assert state["2"]["attempts"]==1
    assert state["1"]["attempts"]==0
    assert not (d/"best"/"2.json").exists()  # dilation cannot beat exact grid
    atomic_json(raw/"3.json",dict(n=2,q=[[.5,.5,0],[1.5,.5,0]]))
    with pytest.raises(ValueError,match="n mismatch"):batch.merge(d,raw)


def test_provenance_is_partition_execution_independent():
    from scheduler import provenance_counts
    state={"1":dict(source="exact grid",attempts=0),
           "2":dict(source="analytic Goebel 1979 construction",attempts=1),
           "3":dict(source="CUDA coverage batch",attempts=2),
           "4":dict(source="local certified candidate",attempts=0)}
    counts=provenance_counts(state,4)
    assert counts==dict(analytic_best=1,gpu_best=1,baseline_best=1,other_best=1,
                        actual_gpu_cases=2,actual_gpu_runs=3)

def test_record_comparison_uses_decimal_not_rounded_float():
    from record_report import comparison
    result=comparison("2.70710678118654752440084436210484903928500664836659",
                      "2.70710678118654")
    assert result["within_tolerance"]
    assert not result["exactly_equal_to_published_decimal"]
    assert result["signed_difference"].startswith("7.524400")
    assert not comparison("3.8867460288941977","3.87708359002281")["within_tolerance"]

def test_autotune_matrix_is_finite_and_reproducible(tmp_path):
    from autotune import prepare
    prepare(tmp_path)
    script=(tmp_path/"tune-run.sh").read_text()
    launches=[line for line in script.splitlines() if line.startswith("PACK_TMIN=")]
    assert len(launches)==12
    assert all("seed11.txt 4000 64" in line for line in launches)
    assert len({line.split()[-1] for line in launches})==12
    assert "while " not in script
