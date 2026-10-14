"""
Make one PNG that shows a run, so an empty or odd result can be understood at a glance (and reused as a report figure).

  python scripts/preview_run.py --run outputs/runs/<run_id> [--out preview.png]

Panels: pre VV (dB) | post VV (dB) | VV log-ratio (blue = darker after, red = brighter after) | VH log-ratio | detected classes (water / debris).
Reads only files the pipeline already wrote (s1_*.tif, sar_change_*.tif, flood_class.tif); needs matplotlib (pip install matplotlib).
Large rasters are decimated so the picture opens quickly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MAX_SIDE_PX = 1200
DB_RANGE = (-25.0, 5.0)
CHANGE_RANGE_DB = 6.0


def _read(path: Path):
    import rasterio
    with rasterio.open(path) as s:
        step = max(1, int(np.ceil(max(s.width, s.height) / MAX_SIDE_PX)))
        a = s.read(1, out_shape=(max(1, s.height // step), max(1, s.width // step))).astype("float64")
        nodata = s.nodata
    if nodata is not None and not np.isnan(nodata):
        a[a == nodata] = np.nan
    return a, step


def _db(lin):
    out = np.full_like(lin, np.nan)
    ok = np.isfinite(lin) & (lin > 0)
    out[ok] = 10 * np.log10(lin[ok])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("matplotlib is not installed: run  pip install matplotlib"); return 2
    run = Path(a.run)
    need = ["s1_before_vv.tif", "s1_after_vv.tif", "sar_change_vv.tif", "sar_change_vh.tif"]
    missing = [n for n in need if not (run / n).exists()]
    if missing:
        print(f"Missing in {run}: {missing}. The run has no Sentinel-1 results (check summary.json warnings)."); return 1
    pre, step = _read(run / "s1_before_vv.tif")
    post, _ = _read(run / "s1_after_vv.tif")
    cvv, _ = _read(run / "sar_change_vv.tif")
    cvh, _ = _read(run / "sar_change_vh.tif")
    cls = _read(run / "flood_class.tif")[0] if (run / "flood_class.tif").exists() else None

    fig, ax = plt.subplots(1, 5, figsize=(22, 4.6))
    for i, (img, title) in enumerate(((_db(pre), "pre VV (dB)"), (_db(post), "post VV (dB)"))):
        im = ax[i].imshow(img, cmap="gray", vmin=DB_RANGE[0], vmax=DB_RANGE[1]); ax[i].set_title(title)
    for i, (img, title) in ((2, (cvv, "VV change (dB): blue = darker after")), (3, (cvh, "VH change (dB): red = brighter after"))):
        ax[i].imshow(img, cmap="seismic", vmin=-CHANGE_RANGE_DB, vmax=CHANGE_RANGE_DB); ax[i].set_title(title)
    if cls is not None:
        ax[4].imshow(np.where(np.isfinite(pre), 0.5, np.nan), cmap="gray", vmin=0, vmax=1)
        ax[4].imshow(np.ma.masked_where(cls == 0, cls), cmap=ListedColormap(["#1f77b4", "#d62728"]), vmin=1, vmax=2, interpolation="nearest")
        ax[4].set_title(f"detected: water {int((cls == 1).sum())} px (blue), debris {int((cls == 2).sum())} px (red)")
    else:
        ax[4].set_title("no flood_class.tif"); ax[4].axis("off")
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.suptitle(f"{run.name}  (preview decimated x{step}; a real flood corridor appears as a connected blue/red band along the valley floor)")
    fig.tight_layout()
    out = Path(a.out) if a.out else run / "preview.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print("Wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
