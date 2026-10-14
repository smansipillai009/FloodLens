import importlib.util
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("preview_run", ROOT / "scripts" / "preview_run.py")
pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)


def _w(path, arr, dtype="float32", nodata=0.0):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                       crs="EPSG:32645", transform=from_origin(0, 800, 10, 10), nodata=nodata) as d:
        d.write(arr.astype(dtype), 1)


def test_preview_is_written_and_missing_inputs_are_reported(tmp_path, capsys):
    assert pr.main(["--run", str(tmp_path)]) == 1 and "Missing" in capsys.readouterr().out
    rng = np.random.default_rng(0)
    lin = rng.gamma(4, .05, (80, 80))
    _w(tmp_path / "s1_before_vv.tif", lin); _w(tmp_path / "s1_after_vv.tif", lin * 0.5)
    _w(tmp_path / "sar_change_vv.tif", rng.normal(0, 1, (80, 80)), nodata=np.nan)
    _w(tmp_path / "sar_change_vh.tif", rng.normal(0, 1, (80, 80)), nodata=np.nan)
    cls = np.zeros((80, 80), "uint8"); cls[10:20, 10:20] = 1; cls[40:45, 40:45] = 2
    _w(tmp_path / "flood_class.tif", cls, "uint8", 0)
    assert pr.main(["--run", str(tmp_path), "--out", str(tmp_path / "p.png")]) == 0
    assert (tmp_path / "p.png").stat().st_size > 5000
