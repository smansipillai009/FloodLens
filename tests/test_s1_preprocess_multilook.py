import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.services.sentinel1 import preprocess_s1_pair

PX_DEG = 0.00009                     # ~10 m at 28 N
N = 270


def _tif(path, seed):
    rng = np.random.default_rng(seed)
    vv = rng.gamma(4, 0.05, (N, N)).astype("float32"); vh = (vv * 0.2).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=N, width=N, count=2, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(85.0, 28.0, PX_DEG, PX_DEG), nodata=0.0) as d:
        d.write(vv, 1); d.write(vh, 2)
    return path


def _aoi():
    x0, y0, x1, y1 = 85.0, 28.0 - N * PX_DEG, 85.0 + N * PX_DEG, 28.0
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def _shape(p):
    with rasterio.open(p) as s:
        return s.height, s.width, s.res[0]


def test_default_multilooks_to_30_m_and_override_keeps_10_m(tmp_path):
    b, a = _tif(tmp_path / "b.tif", 1), _tif(tmp_path / "a.tif", 2)
    d3, d1 = tmp_path / "k3", tmp_path / "k1"
    d3.mkdir(); d1.mkdir()
    o3 = preprocess_s1_pair(b, a, d3, _aoi(), resolution_m=10)                        # config default: multilook_factor 3
    o1 = preprocess_s1_pair(b, a, d1, _aoi(), resolution_m=10, multilook_factor=1)    # AI mode
    h3, w3, r3 = _shape(o3["s1_before_vv"]); h1, w1, r1 = _shape(o1["s1_before_vv"])
    assert abs(r3 / r1 - 3.0) < 0.05 and abs(h1 / h3 - 3.0) < 0.2 and abs(w1 / w3 - 3.0) < 0.2


def test_train_loop_does_not_keep_the_autograd_graph_in_the_loss_sum():
    src = open("app/ml/train.py", encoding="utf-8").read()
    assert "float(l.detach())" in src and "float(l)" not in src.replace("float(l.detach())", "")
