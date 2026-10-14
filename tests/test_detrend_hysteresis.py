import numpy as np
import pytest

from app.services.flood_detection import hysteresis_grow, remove_regional_trend


def _scene(seed=0, n=200):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    trend = 2.5 * np.sin(xx / 60.0) + 1.5 * (yy / n - 0.5)             # smooth km-scale banding (soil moisture / gain)
    truth = np.zeros((n, n), bool)
    for i in range(30, 170):                                            # a 4-px wide diagonal corridor
        truth[i, max(0, i - 3):i + 1] = True
    change = trend + rng.normal(0, 0.9, (n, n)) + np.where(truth, -6.0, 0.0)
    return change.astype("float32"), truth, np.ones((n, n), bool)


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_detrend_removes_smooth_banding_but_keeps_the_corridor():
    change, truth, valid = _scene()
    res = remove_regional_trend(change, valid, sigma_px=40)
    # the white-noise floor is 0.9 dB: detrending can approach it but never go below it
    assert res[~truth].std() < 0.8 * change[~truth].std() and res[~truth].std() < 1.05
    assert res[truth].mean() < -4.0                                      # the corridor survives (clipping keeps it from biasing the trend)


def test_hysteresis_keeps_only_components_with_a_seed():
    weak = np.zeros((20, 20), bool); weak[2:6, 2:15] = True; weak[12:15, 12:15] = True
    seed = np.zeros_like(weak); seed[3, 3] = True
    out = hysteresis_grow(seed, weak)
    assert out[2:6, 2:15].all() and not out[12:15, 12:15].any()


def test_detrend_plus_hysteresis_beats_a_plain_threshold_on_the_corridor():
    change, truth, valid = _scene(1)
    plain = change < -3.0
    det = remove_regional_trend(change, valid, 40)
    hyst = hysteresis_grow(det < -3.0, det < -1.5)
    assert _iou(hyst, truth) > _iou(plain, truth) + 0.05
    assert (hyst & ~truth).sum() / (~truth).sum() < 0.05                 # false-alarm rate stays small


def test_sigma_zero_disables_detrending():
    change, _, valid = _scene()
    assert remove_regional_trend(change, valid, 0) is change


def test_funnel_reports_seeds_and_hysteresis(tmp_path):
    import json
    import rasterio
    from rasterio.transform import from_origin
    from app.services.flood_detection import detect_flood
    n = 120
    rng = np.random.default_rng(3)
    base = rng.gamma(25, 0.02, (n, n)).astype("float32")
    post = base.copy(); post[40:80, 40:80] *= 0.01
    def w(name, a):
        pth = tmp_path / name
        with rasterio.open(pth, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32", crs="EPSG:32645",
                           transform=from_origin(0, n * 30, 30, 30), nodata=0.0) as d:
            d.write(a, 1)
        return pth
    detect_flood(w("a.tif", base), w("b.tif", base), w("c.tif", post), w("d.tif", base), tmp_path)
    f = json.loads((tmp_path / "detection_funnel.json").read_text(encoding="utf-8"))
    assert f["seeds_px"]["water"] > 0 and f["after_hysteresis_px"]["water"] >= f["seeds_px"]["water"]
    assert "log_ratio_vv_db_after_detrend" in f["stats_db"]
