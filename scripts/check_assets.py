"""Check that the static showcase and its numeric manifest are complete."""
from pathlib import Path
import json
import re
import math

root=Path(__file__).resolve().parents[1]
assets=root/'docs/assets'
manifest=json.loads((assets/'metrics.json').read_text())
assert set(manifest['experiments']) == set('123456')
for path in ['01-processing.png','02-filtering.png','02-retrieval.png','03-edges.png',
             '04-homography.png','05-stereo.png','house-rotation.gif','06-pca.png','06-eigenfaces.png',
             'bird-source.jpg','bird-mask.png','edges-source.jpg','alignment-target.jpg','alignment-warped.jpg','face-original.png']:
    assert (assets/path).stat().st_size > 0, path
for k in [1,2,4,8,16,32]: assert (assets/f'face-{k}.png').is_file()
for t in [.05,.1,.15,.2]: assert (assets/f'edges-{t:.2f}.png').is_file()
def finite(value):
    if isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,list):
        for v in value:finite(v)
    elif isinstance(value,float):assert math.isfinite(value)
finite(manifest)
pca=manifest['experiments']['6']
assert all(a>=b for a,b in zip(pca['reconstruction_mse'],pca['reconstruction_mse'][1:]))
assert manifest['experiments']['4']['inliers'] >= 4
assert manifest['experiments']['5']['mean_triangulation_reprojection_px'] < 1
for path in re.findall(r'(?:src|href)="([^"#]+)"',(root/'docs/index.html').read_text()):
    if not path.startswith(('http','data:')):assert (root/'docs'/path).exists(),path
print(f'Six experiments, {len(list(assets.glob("*.png"))) + len(list(assets.glob("*.jpg"))) + len(list(assets.glob("*.gif")))} result images, finite metrics and local HTML links verified.')
