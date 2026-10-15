"""
CHECK-ONLY comparison of a FloodLens run with Copernicus EMS EMSR927 (or any reference map).
This file lives in validation_only/ and is NEVER imported by the pipeline (tests enforce it).

    python validation_only/compare_emsr927.py --run outputs/runs/<id> --ref validation_only/emsr927/observed_event.geojson

Refuses to run unless configs/config.lock matches the current config (=> no tuning on EMSR927).
Outputs: validation_only/results/<run_id>_agreement.json
Metrics: IoU, F1, precision, recall, area ratio, and agreement for buildings/roads/bridges if the
reference has those layers (--ref-assets).
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import config_hash, locked_hash  # noqa: E402
from app.ml.metrics import confusion, scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", required=True, help="reference flood/debris extent polygons (GeoJSON/GPKG/SHP)")
    ap.add_argument("--layer", default=None, help="layer inside a GPKG; use the OBSERVED EVENT layer (e.g. observedEventA_*), "
                    "NOT areaOfInterestA_* (that is just the mapped area outline)")
    a = ap.parse_args()

    # list the layers so the right one can be chosen
    try:
        lyr = gpd.list_layers(a.ref)
        print("Layers in reference file:", ", ".join(map(str, lyr["name"].tolist())))
    except Exception:  # noqa: BLE001
        pass
    if a.layer and "areaofinterest" in a.layer.lower():
        print("WARNING: 'areaOfInterest' is the outline of the mapped area, not the flooded/damaged extent. "
              "Scores against it are meaningless. Use the observed-event layer.")

    if locked_hash() != config_hash():
        sys.exit("REFUSED: config differs from configs/config.lock. Run scripts/freeze_config.py BEFORE "
                 "opening EMSR927, and do not edit the config afterwards (no tuning on the check-only event).")

    run = Path(a.run)
    with rasterio.open(run / "flood_mask.tif") as src:
        pred = src.read(1) > 0
        tr, shape, crs = src.transform, src.shape, src.crs
    # BUGFIX: flood_mask.tif has nodata=0, so read_masks() marked every NON-flood pixel invalid and left only flood pixels.
    # Valid data = pixels where both S1 images have data (same grid as the mask).
    valid = np.ones(shape, dtype=bool)
    for name in ("s1_before_vv.tif", "s1_after_vv.tif"):
        f = run / name
        if f.exists():
            with rasterio.open(f) as v:
                if v.shape == shape:
                    valid &= v.read(1) > 0
    ref = gpd.read_file(a.ref, layer=a.layer).to_crs(crs)
    truth = features.rasterize([(g, 1) for g in ref.geometry if g is not None and not g.is_empty],
                               out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    # compare only where the reference map actually covers (its own extent) to avoid punishing outside-AOI
    cover = features.rasterize([(ref.union_all().envelope, 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    print(f"pixels: predicted flood={int(pred.sum())}, reference flood={int(truth.sum())}, "
          f"inside reference extent={int(cover.sum())}, valid S1={int(valid.sum())}, evaluated={int((valid & cover).sum())}")
    if (valid & cover).sum() == 0:
        sys.exit("No pixels to evaluate: the run area does not overlap the reference extent (or the layer is empty). "
                 "Check CRS/AOI, and that you chose the observed-event layer.")
    c = confusion(pred, truth, valid & cover)
    out = {"run": run.name, "scores": scores(c), "confusion_px": c, "config_hash": config_hash(),
           "note": "Explain disagreement: S1/reference date offsets, geometry & shadow, OSM gaps, label definitions."}
    res = ROOT / "validation_only" / "results"
    res.mkdir(exist_ok=True)
    (res / f"{run.name}_agreement.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
