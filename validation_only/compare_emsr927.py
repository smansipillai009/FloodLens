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
    ap.add_argument("--layer", default=None)
    a = ap.parse_args()

    if locked_hash() != config_hash():
        sys.exit("REFUSED: config differs from configs/config.lock. Run scripts/freeze_config.py BEFORE "
                 "opening EMSR927, and do not edit the config afterwards (no tuning on the check-only event).")

    run = Path(a.run)
    with rasterio.open(run / "flood_mask.tif") as src:
        pred = src.read(1) > 0
        valid = src.read_masks(1) > 0
        tr, shape, crs = src.transform, src.shape, src.crs
    ref = gpd.read_file(a.ref, layer=a.layer).to_crs(crs)
    truth = features.rasterize([(g, 1) for g in ref.geometry if g is not None and not g.is_empty],
                               out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    # compare only where the reference map actually covers (its own extent) to avoid punishing outside-AOI
    cover = features.rasterize([(ref.union_all().envelope, 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    c = confusion(pred, truth, valid & cover)
    out = {"run": run.name, "scores": scores(c), "confusion_px": c, "config_hash": config_hash(),
           "note": "Explain disagreement: S1/reference date offsets, geometry & shadow, OSM gaps, label definitions."}
    res = ROOT / "validation_only" / "results"
    res.mkdir(exist_ok=True)
    (res / f"{run.name}_agreement.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
