import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/"cuda"))
from seeded import write_seed,certify_raw,rank_targets
from squares.core import grid,certify
from scheduler import goebel

def test_seed_preserves_source_and_baseline(tmp_path):
    rows=goebel(2)
    meta=write_seed(tmp_path/"seed.txt",rows,"analytic Goebel")
    assert meta["L"]==certify(10,rows)
    assert len(meta["seed_sha256"])==64
    assert meta==json.loads((tmp_path/"seed.json").read_text())
    lines=(tmp_path/"seed.txt").read_text().splitlines()
    assert len(lines)==11 and lines[0].startswith("10 ")

def raw_grid():
    return dict(n=2,q=[[.5,.5,0],[1.5,.5,0]],seconds=1,
                collective=1,polish=120)

def test_gpu_provenance_never_credits_imported_records():
    meta=dict(n=2,L="2",source="imported record")
    r=certify_raw(raw_grid(),"3",meta)
    assert r["strict_incumbent_improvement"] and not r["gpu_earned"]
    assert r["imported_record"] and not r["cpu_optimizer"]

def test_only_strict_direct_gpu_gain_counts():
    r=certify_raw(raw_grid(),"3",dict(n=2,L="3",source="analytic"))
    assert r["gpu_earned"] and r["score_delta"]>0
    r=certify_raw(raw_grid(),"2",dict(n=2,L="2",source="analytic"))
    assert not r["gpu_earned"] and not r["strict_incumbent_improvement"]

@pytest.mark.parametrize("q", [[],[[float("nan"),0,0]],[[.5,.5,0],[.5,.5,0]]])
def test_seeded_rejects_invalid(q):
    raw=raw_grid();raw["q"]=q
    with pytest.raises(ValueError):certify_raw(raw,"3",dict(n=2,L="3",source="analytic"))

def test_priority_is_score_gain_and_omits_closed_gaps():
    state={"11":dict(L="4"),"19":dict(L="5"),"5":dict(L="2.7")}
    r=rank_targets(state,{11:"3.87",19:"4.88",5:"2.71"})
    assert [x["n"] for x in r]==[11,19]
