import sys,json,importlib
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"cuda"))
import endless,cpp_loop

def test_cpp_only_imports():
    for name in ("endless.py","cpp_loop.py","cpp_pass.py"):
        text=(Path(__file__).resolve().parents[1]/"cuda"/name).read_text()
        assert "squares.core" not in text
        assert "from full_run" not in text

def test_live_score(tmp_path):
    (tmp_path/"best").mkdir()
    for n,L in [(1,"1"),(2,"2"),(4,"2")]:
        (tmp_path/"best"/f"{n}.json").write_text(json.dumps(dict(L=L,certifier="cpp-both")))
    assert endless.live_score(tmp_path)==pytest.approx(2/2**.5-1)

def test_bad_gate_preserves_csv(tmp_path,monkeypatch):
    src=tmp_path/"source";src.write_text("new")
    (tmp_path/"best-so-far.csv").write_text("old")
    def fail(*a):raise ValueError("reject")
    monkeypatch.setattr(endless,"gates",fail)
    with pytest.raises(ValueError):endless.verify_promote(src,tmp_path,1,42)
    assert (tmp_path/"best-so-far.csv").read_text()=="old"

def test_promotion_cpp_only(tmp_path,monkeypatch):
    src=tmp_path/"source";src.write_text("new")
    obj=dict(accepted=1000,rejected=0,missing=0,score=24.,results=[])
    monkeypatch.setattr(endless,"gates",lambda *a:(obj,obj))
    result=endless.verify_promote(src,tmp_path,1,42)
    assert result["score"]==24
    assert (tmp_path/"best-so-far.csv").read_text()=="new"

def test_limits(monkeypatch):
    monkeypatch.setattr(cpp_loop,"ram_free",lambda:16000)
    monkeypatch.setattr(cpp_loop,"gpu_status",lambda:(10000,2000,99))
    assert cpp_loop.concurrency()==4
    monkeypatch.setattr(cpp_loop,"gpu_status",lambda:(1000,11000,99))
    assert cpp_loop.concurrency()==1

def test_csv_stream(tmp_path):
    p=tmp_path/"x.csv"
    cpp_loop.write_csv(p,[(1,[dict(square_id="0",x=".5",y=".5",angle="0")])])
    assert p.read_text().splitlines()[1]=="0,1,0,.5,.5,0"
