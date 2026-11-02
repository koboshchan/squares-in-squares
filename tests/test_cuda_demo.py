"""Host certification gate tests; no GPU needed."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("cuda_demo", Path(__file__).parents[1] / "cuda/demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

def test_prepare(tmp_path):
    demo.prepare(tmp_path, [5, 10])
    for n in [5, 10]:
        lines = (tmp_path / f"seed{n}.txt").read_text().splitlines()
        assert len(lines) == n + 1
        assert int(lines[0].split()[0]) == n

def test_official_gate_accepts_separated_candidate():
    rows, side, dilation = demo.certified_candidate(
        dict(n=2, q=[[.5,.5,0],[1.5,.5,0]]))
    assert len(rows) == 2 and 2 < side < 2.001
    assert dilation == 1.000002

@pytest.mark.parametrize("q", [[], [[.5,.5,0],[.5,.5,0]], [[float("nan"),0,0],[1,0,0]]])
def test_official_gate_refuses_invalid_candidates(q):
    with pytest.raises(ValueError):
        demo.certified_candidate(dict(n=2, q=q))

def test_saved_gpu_candidates_pass_unchanged_official_certifier():
    import json
    root = Path(__file__).parents[1] / "cuda/results"
    for n in (5,10,11,17,50):
        raw = json.loads((root / f"raw{n}.json").read_text())
        assert raw["feasible_checkpoints"] > 0
        rows, side, _ = demo.certified_candidate(raw)
        assert float(demo.certify(n, rows)) == side
