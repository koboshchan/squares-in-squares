import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"cuda"))
from records import catalogue,gaps
from seeded import rank_targets
import math
def test_catalogue_source_and_limit():
    s='<div class="box"><font size="+3">11<a href="square-11.svg">svg</a></font> $s=\\Nn{3.87708359002281}$</div>'
    e=catalogue(s);assert len(e)==1 and e[0]['n']==11
    assert e[0]['published_L']=='3.87708359002281'
    assert e[0]['url'].endswith('/square-11.svg')
def test_real_metric_is_linear_side_not_area():
    r=rank_targets({'11':{'L':'4'}},{11:'3.8'})
    assert abs(r[0]['score_gap']-.2/math.sqrt(11))<1e-15
def test_missing_large_record_not_assumed_grid():
    s={str(n):{'L':'40'} for n in range(1,1001)}
    g=gaps(s,[])
    assert g[324-1]['published_L']=='18'
    assert g[325-1]['published_L'] is None
