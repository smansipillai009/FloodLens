import ast
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent


def _py_files():
    for d in ("app", "scripts", "validation_only"):
        yield from (ROOT / d).rglob("*.py")
    yield ROOT / "run.py"


def test_every_text_file_io_names_its_encoding():
    """Regression: summary.json (OSM names with non-cp1252 characters) crashed on Windows and was left empty."""
    bad = []
    for f in _py_files():
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("write_text", "read_text"):
                if not any(k.arg == "encoding" for k in node.keywords):
                    bad.append(f"{f.name}:{node.lineno} {node.func.attr}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
                if "b" not in str(mode) and not any(k.arg == "encoding" for k in node.keywords):
                    bad.append(f"{f.name}:{node.lineno} open")
    assert not bad, bad


def test_no_gdal_encoding_option_leaked_into_rasterio_open():
    for f in _py_files():
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "open"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "rasterio"):
                assert not any(k.arg == "encoding" for k in node.keywords), f"{f.name}:{node.lineno}"


def test_non_cp1252_characters_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"name": "Sa\u1e45gam"}, ensure_ascii=False), encoding="utf-8")
    assert json.loads(p.read_text(encoding="utf-8"))["name"] == "Sa\u1e45gam"


def test_multilook_block_mean_and_nodata_rule():
    from app.services.sentinel1 import multilook
    a = np.arange(36, dtype="float32").reshape(6, 6) + 1
    out = multilook(a, 3)
    assert out.shape == (2, 2) and np.isclose(out[0, 0], a[:3, :3].mean())
    b = a.copy(); b[:3, :3] = 0; b[0, 3] = 0                 # block 1 fully nodata, block 2 has 8/9 valid
    o2 = multilook(b, 3)
    assert o2[0, 0] == 0 and np.isclose(o2[0, 1], b[:3, 3:6][b[:3, 3:6] > 0].mean())
    assert multilook(a, 1) is a


def test_multilook_cuts_speckle_noise_of_the_log_ratio():
    from app.services.sentinel1 import multilook
    rng = np.random.default_rng(0)
    def img(): return rng.gamma(4, 1 / 4, (300, 300)).astype("float32")     # speckle with 4 looks, true mean 1
    def lr(a, b): return 10 * np.log10(a) - 10 * np.log10(b)
    s1 = lr(img(), img()).std()
    s3 = lr(multilook(img(), 3), multilook(img(), 3)).std()
    assert s3 < 0.5 * s1


def test_proxy_hand_is_not_a_filter_by_default(tmp_path):
    from app.config import get_config
    assert get_config()["dem"].get("hand_proxy_filter") is False and get_config()["sentinel1"]["multilook_factor"] == 3
