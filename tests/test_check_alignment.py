import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("check_alignment", ROOT / "scripts" / "check_kurosiwo_alignment.py")
ca = importlib.util.module_from_spec(spec); spec.loader.exec_module(ca)


def _make(tmp_path, transform=lambda a: a, darker=-9.0, n_patches=12):
    rng = np.random.default_rng(0)
    for i in range(n_patches):
        n = 64
        pre = rng.normal(-8, 1.0, (n, n)); post = pre + rng.normal(0, 0.8, (n, n))
        y = np.zeros((n, n), "uint8"); y[5:30, 30:60] = 1                       # NOT square-symmetric, so a transpose is detectable
        post[y == 1] += darker
        stored_y = transform(y)
        d = tmp_path / "train" / "E"; d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"p{i}.npz", sar=np.stack([pre, pre - 7, post, post - 7]).astype("float32"), y=np.ascontiguousarray(stored_y), valid=np.ones((n, n), "uint8"))


def test_aligned_labels_with_dark_flood_are_ok(tmp_path, capsys):
    _make(tmp_path)
    assert ca.main(["--data", str(tmp_path), "--patches", "12"]) == 0
    assert "VERDICT: OK" in capsys.readouterr().out


def test_transposed_labels_are_detected(tmp_path, capsys):
    _make(tmp_path, transform=lambda a: a.T)
    assert ca.main(["--data", str(tmp_path), "--patches", "12"]) == 4
    out = capsys.readouterr().out
    assert "ORIENTATION MISMATCH" in out and "'transpose'" in out


def test_weak_contrast_is_reported_separately(tmp_path, capsys):
    _make(tmp_path, darker=-0.5)
    assert ca.main(["--data", str(tmp_path), "--patches", "12"]) == 1
    assert "WEAK CONTRAST" in capsys.readouterr().out
