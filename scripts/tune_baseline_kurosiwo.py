"""
Tune and MEASURE the threshold baseline (VV drop + dark post image) on labelled Kuro Siwo patches. No EMSR927, no Nepal data involved.

  python scripts/tune_baseline_kurosiwo.py --data data/processed/kurosiwo [--multilook 3] [--max-patches 3000]

What it does
  1. Loads the patches written by prepare_kurosiwo.py (sar = pre VV, pre VH, post VV, post VH in dB; y = new flood; valid).
  2. Mimics the LIVE pipeline: linear power -> block-mean multilook (config sentinel1.multilook_factor) -> dB; labels/valid are block-reduced the same way.
  3. Grid-searches (water_change_db_vv, water_post_max_db_vv) to maximise IoU on the TRAIN events only.
  4. Reports IoU / F1 / precision / recall of the tuned rule and of the current config values on TRAIN, VAL and TEST (held-out events).
Output: models/checkpoints/baseline_tuning.json (+ suggested YAML lines). It never edits the config; you copy the lines yourself.

Limits: Kuro Siwo labels are flood WATER (not debris) and its events are mostly outside the Himalaya, so these numbers support the baseline-vs-U-Net comparison;
they do not prove performance on Himalayan debris floods.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHANGE_GRID = tuple(np.arange(-1.0, -6.01, -0.5))        # dB: post - pre must be BELOW this
POST_GRID = tuple(np.arange(-10.0, -22.01, -1.0))        # dB: post VV must be at or BELOW this
BLOCK_VALID_FRACTION = 0.75
TIE_TOLERANCE = 0.002
MAX_PIXELS_PER_SPLIT = 12_000_000


def _block_reduce_mean(a: np.ndarray, k: int) -> np.ndarray:
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def prepare_patch(z, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (pre_vv_db, post_vv_db, y) flattened over the VALID blocks, mimicking the live multilook."""
    from app.services.sentinel1 import multilook
    sar = z["sar"].astype("float32")
    y, valid = z["y"].astype("float32"), z["valid"].astype("float32")
    pre_lin, post_lin = np.power(10.0, sar[0] / 10.0), np.power(10.0, sar[2] / 10.0)
    if k > 1:
        pre_lin, post_lin = multilook(pre_lin, k), multilook(post_lin, k)
        y, valid = _block_reduce_mean(y, k), _block_reduce_mean(valid, k)
    ok = (valid >= BLOCK_VALID_FRACTION) & (pre_lin > 0) & (post_lin > 0)
    pre_db, post_db = 10 * np.log10(pre_lin[ok]), 10 * np.log10(post_lin[ok])
    return pre_db, post_db, (y[ok] >= 0.5)


def load_split(root: Path, split: str, k: int, max_patches: int):
    files = sorted((root / split).rglob("*.npz"))
    if not files:
        return None
    idx = np.linspace(0, len(files) - 1, min(len(files), max_patches)).astype(int)
    chg, post, lab = [], [], []
    for i in idx:
        pre_db, post_db, y = prepare_patch(np.load(files[i]), k)
        chg.append(post_db - pre_db); post.append(post_db); lab.append(y)
    chg, post, lab = np.concatenate(chg), np.concatenate(post), np.concatenate(lab)
    if chg.size > MAX_PIXELS_PER_SPLIT:
        sel = np.random.default_rng(0).choice(chg.size, MAX_PIXELS_PER_SPLIT, replace=False)
        chg, post, lab = chg[sel], post[sel], lab[sel]
    return {"chg": chg.astype("float32"), "post": post.astype("float32"), "y": lab, "n_patches": len(idx)}


def score(d: dict, c: float, p: float) -> dict:
    if d["y"].size == 0:
        return {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "flood_pixel_fraction": 0.0, "pixels": 0}
    pred = (d["chg"] < c) & (d["post"] <= p)
    y = d["y"]
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    iou = tp / max(tp + fp + fn, 1)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"iou": round(iou, 4), "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "flood_pixel_fraction": round(float(y.mean()), 4), "pixels": int(y.size)}


def explain_empty(root: Path, k: int, n: int = 3) -> None:
    """Show, for the first few training patches, which condition removes every pixel."""
    from app.services.sentinel1 import multilook
    files = sorted((root / "train").rglob("*.npz"))[:n]
    print("\nWHY EMPTY? first training patches, step by step:")
    for f in files:
        z = np.load(f)
        print(f"  {f.name}: arrays {{{', '.join(f'{key}: {z[key].shape} {z[key].dtype}' for key in z.files)}}}")
        sar = z["sar"].astype("float32")
        print(f"    sar finite: {100 * np.isfinite(sar).mean():.1f}%  range [{np.nanmin(sar):.1f}, {np.nanmax(sar):.1f}] dB   valid mean: {z['valid'].astype('float32').mean():.3f}   flood px: {int(z['y'].sum())}")
        pre_lin = np.power(10.0, sar[0] / 10.0)
        print(f"    pre linear: min {np.nanmin(pre_lin):.3g} max {np.nanmax(pre_lin):.3g}  >0: {int((pre_lin > 0).sum())} px")
        if k > 1:
            ml = multilook(pre_lin, k)
            print(f"    after multilook x{k}: shape {ml.shape}, >0: {int((ml > 0).sum())} blocks")
            v = _block_reduce_mean(z["valid"].astype("float32"), k)
            print(f"    valid blocks with fraction >= {BLOCK_VALID_FRACTION}: {int((v >= BLOCK_VALID_FRACTION).sum())} of {v.size}")


def _pct(x: np.ndarray) -> list:
    return [round(float(v), 2) for v in np.percentile(x, [1, 50, 99])] if x.size else []


def diagnose(splits: dict) -> list[str]:
    """Print what the data looks like and return a list of problems that would make the tuning meaningless."""
    issues = []
    print(f"\n{'split':6s} {'patches':>7s} {'pixels':>10s} {'flood %':>8s}  post VV dB [p01,p50,p99]   change dB [p01,p50,p99]")
    for name, d in splits.items():
        if d is None:
            print(f"{name:6s}  (no files)"); issues.append(f"{name}: no patches found"); continue
        n = int(d["y"].size)
        frac = float(d["y"].mean()) if n else 0.0
        print(f"{name:6s} {d['n_patches']:7d} {n:10d} {100 * frac:8.2f}  {_pct(d['post'])!s:26s} {_pct(d['chg'])}")
        if n == 0:
            issues.append(f"{name}: 0 valid pixels (every pixel is invalid or ignored)")
        elif frac < 0.001:
            issues.append(f"{name}: only {100 * frac:.3f}% flood pixels (labels empty or 'valid' removed them)")
        if n:
            med = float(np.median(d["post"]))
            if not (-25.0 <= med <= 0.0):
                issues.append(f"{name}: median post VV = {med:.1f} dB is not a plausible land backscatter (-25..0 dB)")
            elif abs(float(np.median(d["chg"]))) > 3.0:
                issues.append(f"{name}: median change = {float(np.median(d['chg'])):.1f} dB (pre and post should agree on land)")
    return issues


def grid_search(train: dict) -> tuple[tuple[float, float], list]:
    rows = []
    for c in CHANGE_GRID:
        for p in POST_GRID:
            s = score(train, float(c), float(p))
            rows.append({"water_change_db_vv": round(float(c), 2), "water_post_max_db_vv": round(float(p), 2), **s})
    top = max(r["iou"] for r in rows)
    near_best = [r for r in rows if r["iou"] >= top - TIE_TOLERANCE]            # a plateau of equally good rules is common
    best = min(near_best, key=lambda r: (r["water_change_db_vv"], r["water_post_max_db_vv"]))   # ...pick the STRICTEST (fewest false alarms)
    return (best["water_change_db_vv"], best["water_post_max_db_vv"]), rows


def main(argv=None) -> int:
    from app.config import get_config
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--multilook", type=int, default=int(cfg["sentinel1"].get("multilook_factor", 1)))
    ap.add_argument("--max-patches", type=int, default=3000)
    ap.add_argument("--out", default=cfg["ml"]["checkpoint"].rsplit("/", 1)[0] + "/baseline_tuning.json")
    a = ap.parse_args(argv)
    root = Path(a.data)
    splits = {s: load_split(root, s, a.multilook, a.max_patches) for s in ("train", "val", "test")}
    if splits["train"] is None:
        print(f"No training patches under {root / 'train'}. Run prepare_kurosiwo.py first."); return 2
    print(f"patches: " + ", ".join(f"{s}={d['n_patches']}" for s, d in splits.items() if d) + f" | multilook x{a.multilook}")
    issues = diagnose(splits)
    if all(d is not None and d["y"].size == 0 for d in splits.values() if d is not None):
        explain_empty(root, a.multilook)
        print("\nEvery split has 0 usable pixels, so there is nothing to tune. Paste the 'WHY EMPTY?' block above.")
        return 3
    for msg in issues:
        print("PROBLEM:", msg)
    if issues:
        print("\nPossible causes: wrong --units when running prepare_kurosiwo.py (see data/processed/kurosiwo/prepare_meta.json: 'units'), "
              "or the raw files do not mean what the dataset card says. Run: python scripts/prepare_kurosiwo.py --src <raw> --inspect 20 and compare "
              "the value ranges with the units you used. Re-convert after fixing; do not trust any threshold below.")
    (c_best, p_best), rows = grid_search(splits["train"])
    if max(r["iou"] for r in rows) == 0.0:
        print("\nNo threshold pair separates flood from non-flood on the training events (best IoU = 0). Nothing to suggest.")
        return 3
    th = cfg["flood_detection"]
    cur = (th["water_change_db_vv"], th["water_post_max_db_vv"])
    report = {"multilook": a.multilook, "grid_rows": len(rows), "tuned": {"water_change_db_vv": c_best, "water_post_max_db_vv": p_best},
              "current_config": {"water_change_db_vv": cur[0], "water_post_max_db_vv": cur[1]}, "scores": {}}
    print(f"\n{'rule':22s} {'split':6s} {'IoU':>6s} {'F1':>6s} {'prec':>6s} {'recall':>6s}")
    for name, (c, p) in (("tuned", (c_best, p_best)), ("current config", cur)):
        for s, d in splits.items():
            if d is None:
                continue
            sc = score(d, c, p); report["scores"].setdefault(name, {})[s] = sc
            print(f"{name + f' ({c},{p})':22s} {s:6s} {sc['iou']:6.3f} {sc['f1']:6.3f} {sc['precision']:6.3f} {sc['recall']:6.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {a.out}\nSuggested lines for configs/config.yaml (section flood_detection):\n  water_change_db_vv: {c_best}\n  water_post_max_db_vv: {p_best}")
    print("Treat TEST (held-out events) as the honest number; if TEST is far below TRAIN the thresholds do not transfer between events.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
