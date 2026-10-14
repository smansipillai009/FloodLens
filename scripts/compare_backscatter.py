"""
Compare the backscatter distribution of the TRAINING patches with a LIVE FloodLens run (Day-1 sanity check).

  python scripts/compare_backscatter.py --patches data/processed/kurosiwo/train --run outputs/runs/<run_id> [--png out.png]

Why: the live fetch uses a configurable back-coefficient (config: sentinel1.back_coeff, default GAMMA0_TERRAIN) while
Kuro Siwo may be sigma0. A constant dB offset between training and inference shifts the absolute-dB model inputs and the
absolute water threshold. This script measures that offset from the UPPER quantiles (p50, p75, p95) of each channel: water, shadow
and flooded pixels only populate the low end of the distribution, so the upper quantiles are a fair per-channel comparison even
though the scenes differ. (A fixed "land" cut-off was tried and rejected: it biases VH, whose median sits near the cut.)

It is a rough check, not a calibration: terrain, vegetation and season differ between a Kuro Siwo event and a Himalayan AOI.
Use several runs/AOIs before changing anything, and re-train if you change the convention.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

CHANNELS = ("pre_VV", "pre_VH", "post_VV", "post_VH")
LIVE_FILES = ("s1_before_vv.tif", "s1_before_vh.tif", "s1_after_vv.tif", "s1_after_vh.tif")
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
OFFSET_QUANTILES = (50, 75, 95)   # upper quantiles used for the offset; the dark tail (water/shadow/flood) is ignored
OK_OFFSET_DB = 0.5              # |offset| below this: no action
GAMMA_VS_SIGMA_INCIDENCE_DEG = 39.0   # mid-swath IW; sigma0 -> gamma0 adds -10*log10(cos(theta)) dB on flat ground
MAX_PIXELS_PER_CHANNEL = 2_000_000


def gamma_sigma_expected_db(incidence_deg: float = GAMMA_VS_SIGMA_INCIDENCE_DEG) -> float:
    return -10.0 * math.log10(math.cos(math.radians(incidence_deg)))


def _subsample(x: np.ndarray, limit: int, seed: int = 0) -> np.ndarray:
    if x.size <= limit:
        return x
    return np.random.default_rng(seed).choice(x, size=limit, replace=False)


def load_training(patch_dir: Path, max_patches: int = 300) -> dict[str, np.ndarray]:
    """dB values of VALID pixels per channel from PatchDataset .npz files."""
    files = sorted(Path(patch_dir).rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz patches under {patch_dir}")
    idx = np.linspace(0, len(files) - 1, min(len(files), max_patches)).astype(int)
    acc = {c: [] for c in CHANNELS}
    for i in idx:
        z = np.load(files[i])
        valid = z["valid"] > 0
        for k, c in enumerate(CHANNELS):
            acc[c].append(z["sar"][k][valid])
    return {c: _subsample(np.concatenate(v), MAX_PIXELS_PER_CHANNEL) for c, v in acc.items()}


def load_live(run_dir: Path) -> dict[str, np.ndarray]:
    """dB values of valid pixels per channel from a run directory (linear rasters, nodata = 0)."""
    import rasterio
    out = {}
    for c, f in zip(CHANNELS, LIVE_FILES):
        p = Path(run_dir) / f
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run the pipeline with S1 data first.")
        with rasterio.open(p) as s:
            a = s.read(1).astype(np.float64)
        a = a[np.isfinite(a) & (a > 0)]
        out[c] = _subsample(10.0 * np.log10(a), MAX_PIXELS_PER_CHANNEL)
    return out


def summarise(vals: dict[str, np.ndarray]) -> dict:
    res = {}
    for c, v in vals.items():
        res[c] = {"n": int(v.size),
                  "percentiles": {f"p{p:02d}": round(float(np.percentile(v, p)), 2) for p in PERCENTILES} if v.size else {}}
    return res


def compare(train: dict[str, np.ndarray], live: dict[str, np.ndarray]) -> dict:
    t, l = summarise(train), summarise(live)
    rows = {}
    for c in CHANNELS:
        if not t[c]["percentiles"] or not l[c]["percentiles"]:
            rows[c] = {"offset_db": None}
            continue
        diffs = [l[c]["percentiles"][f"p{q:02d}"] - t[c]["percentiles"][f"p{q:02d}"] for q in OFFSET_QUANTILES]
        rows[c] = {"train_p50": t[c]["percentiles"]["p50"], "live_p50": l[c]["percentiles"]["p50"],
                   "offset_db": round(float(np.median(diffs)), 2),
                   "offset_spread_db": round(float(max(diffs) - min(diffs)), 2)}
    offs = [r["offset_db"] for r in rows.values() if r["offset_db"] is not None]
    mean_off = float(np.mean(offs)) if offs else None
    expected = gamma_sigma_expected_db()
    if mean_off is None:
        verdict = "NO_DATA"
    elif abs(mean_off) < OK_OFFSET_DB:
        verdict = "OK"
    elif abs(mean_off - expected) < OK_OFFSET_DB * 2:
        verdict = "CONSISTENT_WITH_GAMMA0_VS_SIGMA0"
    else:
        verdict = "OFFSET_UNEXPLAINED"
    return {"quantiles_used": list(OFFSET_QUANTILES), "channels": rows,
            "mean_offset_db": None if mean_off is None else round(mean_off, 2),
            "expected_gamma_minus_sigma_flat_db": round(expected, 2), "verdict": verdict,
            "train_summary": t, "live_summary": l}


ADVICE = {
    "OK": "No material offset. Still check that the sign of the VV-VH separation looks similar.",
    "CONSISTENT_WITH_GAMMA0_VS_SIGMA0":
        "Live is brighter by about the gamma0-vs-sigma0 offset. If Kuro Siwo is sigma0, either set sentinel1.back_coeff to "
        "SIGMA0_ELLIPSOID (matches training, but slopes are not flattened) or keep GAMMA0_TERRAIN and retrain/renormalise with "
        "a corrected offset. Re-tune the absolute water threshold if you change convention.",
    "OFFSET_UNEXPLAINED":
        "Offset is not explained by gamma0 vs sigma0. Check units (dB vs linear), speckle filtering, nodata handling and that "
        "the training pre/post channel order matches the live one before trusting any metric.",
    "NO_DATA": "Not enough valid pixels in one of the inputs.",
}


def print_report(rep: dict) -> None:
    print(f"Offset = median of (live - train) at quantiles {rep['quantiles_used']}.  Expected gamma0-sigma0 "
          f"(flat, 39 deg): {rep['expected_gamma_minus_sigma_flat_db']} dB")
    print(f"{'channel':8s} {'train p50':>10s} {'live p50':>10s} {'offset':>8s} {'spread':>7s}")
    for c, r in rep["channels"].items():
        if r.get("offset_db") is None:
            print(f"{c:8s}  (no data)"); continue
        print(f"{c:8s} {r['train_p50']:10.2f} {r['live_p50']:10.2f} {r['offset_db']:+8.2f} {r['offset_spread_db']:7.2f}")
    print(f"\nMean offset (live - train): {rep['mean_offset_db']} dB   verdict: {rep['verdict']}")
    print(ADVICE[rep["verdict"]])


def save_png(train: dict, live: dict, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("matplotlib not installed: PNG skipped.")
        return False
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    for ax, c in zip(axes, CHANNELS):
        bins = np.linspace(-40, 5, 91)
        ax.hist(train[c], bins=bins, density=True, alpha=.55, label="training")
        ax.hist(live[c], bins=bins, density=True, alpha=.55, label="live")
        ax.set_title(c); ax.set_xlabel("dB")
    axes[0].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return True


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", required=True, help="e.g. data/processed/kurosiwo/train")
    ap.add_argument("--run", required=True, help="outputs/runs/<run_id> containing s1_{before,after}_{vv,vh}.tif")
    ap.add_argument("--max-patches", type=int, default=300)
    ap.add_argument("--png", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    train = load_training(Path(a.patches), a.max_patches)
    live = load_live(Path(a.run))
    rep = compare(train, live)
    print_report(rep)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if a.png:
        save_png(train, live, Path(a.png))
    return 0 if rep["verdict"] in ("OK", "CONSISTENT_WITH_GAMMA0_VS_SIGMA0") else 1


if __name__ == "__main__":
    sys.exit(main())
