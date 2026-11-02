import importlib.util
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"cuda"))
import endless

def test_live_score_streams_best(tmp_path):
    (tmp_path/"best").mkdir()
    for n,L in [(1,"1"),(2,"2"),(4,"2")]:
        (tmp_path/"best"/f"{n}.json").write_text(json.dumps(dict(L=L)))
    assert endless.live_score(tmp_path)==pytest.approx(2/2**.5-1)

def test_promotion_keeps_valid_csv_when_gate_fails(tmp_path,monkeypatch):
    source=tmp_path/"input.csv";source.write_text("new")
    saved=tmp_path/"best-so-far.csv";saved.write_text("old")
    monkeypatch.setattr(endless,"cpp",lambda *a:False)
    with pytest.raises(AssertionError): endless.verify_promote(source,tmp_path,1,42)
    assert saved.read_text()=="old"

def test_promotion_four_gates(tmp_path,monkeypatch):
    source=tmp_path/"input.csv";source.write_text("new")
    calls=[]
    monkeypatch.setattr(endless,"cpp",lambda *a:calls.append("cpp-both") or True)
    monkeypatch.setattr(endless,"read_submission",lambda *a:dict.fromkeys(range(1,1001),[]))
    monkeypatch.setattr(endless,"kaggle_roundtrip",lambda x:calls.append("roundtrip") or x)
    monkeypatch.setattr(endless,"score",lambda x:calls.append("official") or dict(accepted=1000,rejected=0,missing=0,score=24.))
    report=endless.verify_promote(source,tmp_path,1,42)
    assert calls==["cpp-both","official","roundtrip","official"]
    assert (tmp_path/"best-so-far.csv").read_text()=="new"
    assert report["seed"]==42

def test_no_regression(tmp_path,monkeypatch):
    source=tmp_path/"input.csv";source.write_text("new")
    (tmp_path/"best-so-far.csv").write_text("old")
    (tmp_path/"best-so-far.json").write_text(json.dumps(dict(score=23.,roundtrip_score=23.)))
    monkeypatch.setattr(endless,"cpp",lambda *a:True)
    monkeypatch.setattr(endless,"read_submission",lambda *a:dict.fromkeys(range(1,1001),[]))
    monkeypatch.setattr(endless,"kaggle_roundtrip",lambda x:x)
    monkeypatch.setattr(endless,"score",lambda x:dict(accepted=1000,rejected=0,missing=0,score=24.))
    with pytest.raises(AssertionError): endless.verify_promote(source,tmp_path,1,42)
    assert (tmp_path/"best-so-far.csv").read_text()=="old"
