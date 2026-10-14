import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("tune_baseline", ROOT / "scripts" / "tune_baseline_kurosiwo.py")
tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)


def _patch(path: Path, seed: int, drop_db=-9.0, post_level=-19.0):
    rng = np.random.default_rng(seed)
    n = 60
    pre = rng.normal(-8, 0.8, (n, n))                           # land, mild noise
    post = pre + rng.normal(0, 0.8, (n, n))
    y = np.zeros((n, n), "uint8"); y[15:40, 15:40] = 1
    post[y == 1] = post_level + rng.normal(0, 0.8, (y == 1).sum())   # flood: dark and much lower than pre
    sar = np.stack([pre, pre - 7, post, post - 7]).astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, sar=sar, y=y, valid=np.ones((n, n), "uint8"))


def test_grid_search_recovers_a_sensible_rule_and_reports_all_splits(tmp_path):
    for s, seeds in (("train", range(6)), ("val", range(10, 12)), ("test", range(20, 23))):
        for i in seeds:
            _patch(tmp_path / s / "EVT" / f"p{i}.npz", i)
    out = tmp_path / "res.json"
    assert tb.main(["--data", str(tmp_path), "--multilook", "3", "--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert -6.0 <= rep["tuned"]["water_change_db_vv"] <= -2.0 and rep["tuned"]["water_post_max_db_vv"] <= -12.0
    assert rep["scores"]["tuned"]["test"]["iou"] > 0.85
    assert rep["scores"]["tuned"]["train"]["iou"] >= rep["scores"]["current config"]["train"]["iou"] - 1e-9   # tuning cannot be worse on train


def test_missing_training_data_is_reported(tmp_path, capsys):
    assert tb.main(["--data", str(tmp_path)]) == 2
    assert "prepare_kurosiwo" in capsys.readouterr().out


def test_multilook_changes_the_pixel_count_but_keeps_labels_consistent(tmp_path):
    _patch(tmp_path / "p.npz", 0)
    z = np.load(tmp_path / "p.npz")
    pre1, post1, y1 = tb.prepare_patch(z, 1)
    pre3, post3, y3 = tb.prepare_patch(z, 3)
    assert pre1.size == 3600 and pre3.size == 400
    assert abs(y1.mean() - y3.mean()) < 0.06


def test_wrong_units_are_diagnosed_and_no_thresholds_are_suggested(tmp_path, capsys):
    """Run-time regression: all-zero IoU table with no explanation (data converted with the wrong --units)."""
    rng = np.random.default_rng(0)
    for s, seeds in (("train", range(3)), ("val", range(3, 4)), ("test", range(4, 6))):
        for i in seeds:
            n = 60
            lin_as_db = rng.normal(0.0, 0.05, (4, n, n)).astype("float32")      # linear backscatter mistaken for dB: everything ~0 dB
            y = np.zeros((n, n), "uint8"); y[20:40, 20:40] = 1
            (tmp_path / s / "E").mkdir(parents=True, exist_ok=True)
            np.savez(tmp_path / s / "E" / f"p{i}.npz", sar=lin_as_db, y=y, valid=np.ones((n, n), "uint8"))
    rc = tb.main(["--data", str(tmp_path), "--multilook", "1", "--out", str(tmp_path / "o.json")])
    out = capsys.readouterr().out
    assert rc == 3 and "PROBLEM" in out and "units" in out and "Suggested lines" not in out


def test_empty_split_does_not_produce_nan_warnings(tmp_path, capsys):
    _patch(tmp_path / "train" / "E" / "a.npz", 1)
    z = np.load(tmp_path / "train" / "E" / "a.npz")
    np.savez(tmp_path / "val" / "E" / "b.npz", sar=z["sar"], y=z["y"], valid=np.zeros_like(z["valid"])) if (tmp_path / "val" / "E").mkdir(parents=True, exist_ok=True) is None else None
    tb.main(["--data", str(tmp_path), "--multilook", "1", "--out", str(tmp_path / "o.json")])
    out = capsys.readouterr().out
    assert "0 valid pixels" in out and "nan" not in out.lower()


def test_all_empty_prints_the_reason_block(tmp_path, capsys):
    for s_ in ("train", "val", "test"):
        _patch(tmp_path / s_ / "E" / "a.npz", 1)
        z = np.load(tmp_path / s_ / "E" / "a.npz")
        np.savez(tmp_path / s_ / "E" / "a.npz", sar=z["sar"], y=z["y"], valid=np.zeros_like(z["valid"]))     # nothing valid anywhere
    assert tb.main(["--data", str(tmp_path), "--multilook", "3", "--out", str(tmp_path / "o.json")]) == 3
    out = capsys.readouterr().out
    assert "WHY EMPTY?" in out and "valid mean: 0.000" in out
