import importlib.util
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("compare_backscatter", ROOT / "scripts" / "compare_backscatter.py")
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)


def _write_patches(d: Path, offset_db: float, n=6):
    rng = np.random.default_rng(1)
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        sar = np.stack([rng.normal(-10, 2, (32, 32)), rng.normal(-17, 2, (32, 32)),
                        rng.normal(-10, 2, (32, 32)), rng.normal(-17, 2, (32, 32))]).astype("float32") + offset_db
        np.savez(d / f"p{i}.npz", sar=sar, y=np.zeros((32, 32), "uint8"), valid=np.ones((32, 32), "uint8"))


def _write_run(d: Path, offset_db: float):
    rng = np.random.default_rng(2)
    d.mkdir(parents=True, exist_ok=True)
    for name, mu in zip(cb.LIVE_FILES, (-10, -17, -10, -17)):
        db = rng.normal(mu, 2, (64, 64)) + offset_db
        with rasterio.open(d / name, "w", driver="GTiff", height=64, width=64, count=1, dtype="float32",
                           crs="EPSG:32645", transform=from_origin(0, 640, 10, 10), nodata=0.0) as s:
            s.write((10 ** (db / 10)).astype("float32"), 1)


def test_no_offset_is_ok(tmp_path):
    _write_patches(tmp_path / "p", 0.0); _write_run(tmp_path / "r", 0.0)
    rep = cb.compare(cb.load_training(tmp_path / "p"), cb.load_live(tmp_path / "r"))
    assert rep["verdict"] == "OK" and abs(rep["mean_offset_db"]) < 0.5


def test_gamma_sigma_sized_offset_is_recognised(tmp_path):
    _write_patches(tmp_path / "p", 0.0); _write_run(tmp_path / "r", cb.gamma_sigma_expected_db())
    rep = cb.compare(cb.load_training(tmp_path / "p"), cb.load_live(tmp_path / "r"))
    assert rep["verdict"] == "CONSISTENT_WITH_GAMMA0_VS_SIGMA0"
    assert all(0.7 < r["offset_db"] < 1.6 for r in rep["channels"].values())  # incl. VH (no fixed land cut-off bias)


def test_large_offset_is_flagged_unexplained(tmp_path):
    _write_patches(tmp_path / "p", 0.0); _write_run(tmp_path / "r", 4.0)
    rep = cb.compare(cb.load_training(tmp_path / "p"), cb.load_live(tmp_path / "r"))
    assert rep["verdict"] == "OFFSET_UNEXPLAINED"


def test_cli_exit_codes(tmp_path, capsys):
    _write_patches(tmp_path / "p", 0.0); _write_run(tmp_path / "r", 0.0)
    assert cb.main(["--patches", str(tmp_path / "p"), "--run", str(tmp_path / "r")]) == 0
    _write_run(tmp_path / "r2", 4.0)
    assert cb.main(["--patches", str(tmp_path / "p"), "--run", str(tmp_path / "r2")]) == 1
    assert "verdict" in capsys.readouterr().out
