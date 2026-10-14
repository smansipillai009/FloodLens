"""Synthetic-shard tests for scripts/prepare_kurosiwo.py (no real Kuro Siwo data needed)."""
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("prepare_kurosiwo", ROOT / "scripts" / "prepare_kurosiwo.py")
pk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pk)

H = W = 32


def _npy(a):
    b = io.BytesIO(); np.save(b, a.astype(np.float32)); return b.getvalue()


def make_sample(db=True, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(-12, 1.5, (H, W)).astype(np.float32)           # dB
    post = base.copy(); post[8:16, 8:16] -= 10                         # new flood: dark in post
    conv = (lambda x: x) if db else (lambda x: 10 ** (x / 10))
    mask = np.zeros((H, W), np.float32); mask[8:16, 8:16] = 2; mask[0:4, 0:4] = 1
    valid = np.ones((H, W), np.float32); valid[-3:, :] = 0
    return {"sec1_vv": conv(base), "sec1_vh": conv(base - 6), "sec2_vv": conv(base + .2), "sec2_vh": conv(base - 6),
            "flood_vv": conv(post), "flood_vh": conv(post - 6), "dem": np.zeros((H, W), np.float32),
            "mask": mask, "valid_mask": valid}


def write_shard(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        for key, arrs, info in samples:
            for name, a in arrs.items():
                data = _npy(a); ti = tarfile.TarInfo(f"{key}.{name}.npy"); ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
            data = json.dumps(info).encode(); ti = tarfile.TarInfo(f"{key}.info.json"); ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def build_src(tmp_path, db=True, test_event="EVT_TEST", train_events=("EVT_A", "EVT_B", "EVT_C")):
    src = tmp_path / "ks"
    write_shard(src / "train_GRD" / "shard-00000.tar",
                [(f"s{i}", make_sample(db, i), {"event_id": e}) for i, e in enumerate(train_events)])
    write_shard(src / "test_GRD" / "shard-00000.tar", [("t0", make_sample(db, 9), {"event_id": test_event})])
    return src


def test_unit_detection_both_ways():
    assert pk.detect_units(list(make_sample(True).values())[:4]) == "db"
    assert pk.detect_units(list(make_sample(False).values())[:4]) == "linear"


def test_patch_labels_and_ignore_rules():
    p = pk.build_patch(make_sample(True), "sec1", "db", "none")
    assert p["sar"].shape == (4, H, W) and p["sar"].dtype == np.float32
    assert p["y"][10, 10] == 1 and p["y"][20, 20] == 0
    assert p["valid"][1, 1] == 0            # permanent water ignored
    assert p["valid"][-1, 5] == 0           # invalid pixel ignored
    assert p["valid"][20, 20] == 1
    z = pk.build_patch(make_sample(True), "sec1", "db", "none", permanent="zero")
    assert z["valid"][1, 1] == 1 and z["y"][1, 1] == 0


def test_db_and_linear_inputs_give_same_db_output():
    a = pk.build_patch(make_sample(True), "sec1", "db", "none")["sar"]
    b = pk.build_patch(make_sample(False), "sec1", "linear", "none")["sar"]
    assert np.allclose(a, b, atol=1e-3)


def test_lee_speckle_runs_and_keeps_shape():
    p = pk.build_patch(make_sample(False), "sec1", "linear", "lee")
    assert p["sar"].shape == (4, H, W) and np.isfinite(p["sar"]).all()


def test_end_to_end_conversion_and_dataset_reads_it(tmp_path):
    src, dst = build_src(tmp_path), tmp_path / "out"
    rc = pk.main(["--src", str(src), "--dst", str(dst), "--units", "db", "--speckle", "none",
                  "--val-events", "EVT_B"])
    assert rc == 0
    meta = json.loads((dst / "prepare_meta.json").read_text(encoding="utf-8"))
    assert meta["events"]["val"] == ["EVT_B"] and meta["events"]["test"] == ["EVT_TEST"]
    assert set(meta["events"]["train"]) == {"EVT_A", "EVT_C"} and meta["leaky_events"] == []
    from app.ml.dataset import PatchDataset
    x, y, v = PatchDataset(dst, "train")[0]
    assert x.shape == (6, H, W) and y.shape == (1, H, W) and v.shape == (1, H, W)


def test_leaky_event_is_a_hard_error(tmp_path):
    src = build_src(tmp_path, test_event="EVT_A")
    assert pk.main(["--src", str(src), "--dst", str(tmp_path / "o"), "--units", "db", "--speckle", "none"]) == 5


def test_missing_event_id_is_reported(tmp_path):
    src = tmp_path / "ks"
    write_shard(src / "train_GRD" / "s.tar", [("a", make_sample(), {"foo": 1})])
    write_shard(src / "test_GRD" / "s.tar", [("b", make_sample(), {"foo": 2})])
    assert pk.main(["--src", str(src), "--dst", str(tmp_path / "o"), "--units", "db"]) == 3
    assert pk.main(["--src", str(src), "--dst", str(tmp_path / "o2"), "--units", "db", "--speckle", "none",
                    "--event-key", "foo"]) == 0


def test_both_pre_images_double_the_patches(tmp_path):
    src = build_src(tmp_path)
    assert pk.main(["--src", str(src), "--dst", str(tmp_path / "o"), "--units", "db", "--speckle", "none",
                    "--pre", "both", "--val-events", "EVT_B"]) == 0
    meta = json.loads((tmp_path / "o" / "prepare_meta.json").read_text(encoding="utf-8"))
    assert meta["counts"]["train"] == 4 and meta["counts"]["val"] == 2


def test_inspect_runs(tmp_path, capsys):
    assert pk.main(["--src", str(build_src(tmp_path)), "--inspect", "3"]) == 0
    out = capsys.readouterr().out
    assert "Unit votes" in out and "Proposed: --units db" in out


def test_nothing_written_is_an_error(tmp_path):
    src = tmp_path / "ks"
    arrs = {k: v for k, v in make_sample(True).items() if not k.startswith("sec2")}
    write_shard(src / "train_GRD" / "shard-0.tar", [("s0", arrs, {"event_id": "A"})])
    write_shard(src / "test_GRD" / "shard-0.tar", [("t0", arrs, {"event_id": "B"})])
    rc = pk.main(["--src", str(src), "--dst", str(tmp_path / "o"), "--units", "db", "--speckle", "none", "--pre", "both"])
    assert rc == 6


def test_max_per_split_is_enforced(tmp_path):
    src = build_src(tmp_path)
    rc = pk.main(["--src", str(src), "--dst", str(tmp_path / "o"), "--units", "db", "--speckle", "none",
                  "--val-events", "EVT_B", "--max-per-split", "1"])
    assert rc == 0 and len(list((tmp_path / "o" / "train").rglob("*.npz"))) == 1


def test_nested_download_folder_is_found(tmp_path, capsys):
    build_src(tmp_path)                                   # creates tmp_path/ks/train_GRD ...
    assert pk.resolve_root(tmp_path) == tmp_path / "ks"   # user pointed one level too high
    assert pk.main(["--src", str(tmp_path), "--inspect", "2"]) == 0


def test_missing_folder_prints_helpful_message(tmp_path, capsys):
    assert pk.main(["--src", str(tmp_path / "nope"), "--inspect", "2"]) == 2
    assert "does not exist" in capsys.readouterr().out
    (tmp_path / "empty").mkdir()
    assert pk.main(["--src", str(tmp_path / "empty"), "--inspect", "2"]) == 2
    assert "Expected" in capsys.readouterr().out


def test_float16_storage_loads_as_float32_and_barely_changes_values(tmp_path):
    src = build_src(tmp_path)
    for dt in ("float16", "float32"):
        assert pk.main(["--src", str(src), "--dst", str(tmp_path / dt), "--units", "db", "--speckle", "none",
                        "--val-events", "EVT_B", "--store-dtype", dt]) == 0
    f16 = sorted((tmp_path / "float16" / "train").rglob("*.npz"))[0]
    f32 = sorted((tmp_path / "float32" / "train").rglob("*.npz"))[0]
    assert np.load(f16)["sar"].dtype == np.float16 and np.load(f32)["sar"].dtype == np.float32
    assert np.abs(np.load(f16)["sar"].astype("float32") - np.load(f32)["sar"]).max() < 0.05
    from app.ml.dataset import PatchDataset
    x, _, _ = PatchDataset(tmp_path / "float16", "train")[0]
    assert x.dtype == np.float32
