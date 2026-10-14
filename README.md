# FloodLens Nepal  (v0.2)
Track B — Multimodal AI Hackathon 2026. **Educational prototype, not operational emergency guidance.**

Takes an AOI + flood date and answers: where did the flood/debris hit (Sentinel-1 two-sided change + terrain masks
+ clear-pixel Sentinel-2 confirmation, optional U-Net), what is potentially affected (pre-event OSM buildings/roads/bridges),
who is potentially cut off (motorable-road graph G0 vs G1, buffer sensitivity), and writes a template report from `summary.json`.

<img width="1895" height="952" alt="Screenshot 2026-10-14 220216" src="https://github.com/user-attachments/assets/63873100-c449-4c2e-be39-c2ebb96010fb" />
<img width="1872" height="857" alt="Screenshot 2026-10-14 220132" src="https://github.com/user-attachments/assets/5fba6874-df5d-477e-900f-ffd25c9dbf45" />


## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # + torch & segmentation-models-pytorch for the AI part, pysheds for true HAND
cp .env.example .env                       # fill CDSE_CLIENT_ID / CDSE_CLIENT_SECRET
pytest -q                                  # compliance + unit tests must pass before every demo

Files larger than 25MB have a google drive link atteched for reference
\data\reference\dem.zip
\data\processed\kurosiwo.zip
\models\checkpoints\unet_resnet18.pt
\kurosiwo_patches.zip

```
## Run
```bash
python run.py search  --aoi configs/trishuli_demo_aoi.geojson --date 2026-08-26   # scene pair + same-track check
python run.py osm     --aoi configs/trishuli_demo_aoi.geojson --date 2026-08-26   # pre-event OSM counts
python run.py analyse --aoi configs/trishuli_demo_aoi.geojson --date 2026-08-26   # full pipeline (fetches S1/S2 from CDSE)
python run.py analyse --aoi ... --date ... --before-tif pre1.tif pre2.tif --after-tif post.tif --s2-tif s2.tif   # local inputs
streamlit run app/frontend/Home.py
```
Outputs: `outputs/runs/<run_id>/` (manifest.json, summary.json, rasters, GeoJSON, situation_report_en.md).

## AI component
```bash
python scripts/prepare_kurosiwo.py --src <Kuro-Siwo-Webdataset> --inspect 20      # Day 1: check units, labels, event key
python scripts/prepare_kurosiwo.py --src <Kuro-Siwo-Webdataset> --dst data/processed/kurosiwo --units <db|linear>
python scripts/compare_backscatter.py --patches data/processed/kurosiwo/train --run outputs/runs/<run_id> --png bs.png   # gamma0/sigma0 offset check
python -m app.ml.train --data data/processed/kurosiwo --epochs 30
python run.py analyse ... --model-mode ai  # U-Net water + baseline debris; falls back to baseline (with warning) if no checkpoint
```
## Validation (check-only)
```bash
python scripts/freeze_config.py && git add configs/config.lock && git commit -m "freeze config"
python validation_only/compare_emsr927.py --run outputs/runs/<id> --ref validation_only/<reference file>
```
## Data rules
Inputs: Sentinel-1/2, Copernicus DEM, OSM **snapshot before the event** (ohsome), Kuro Siwo / Sen1Floods11.
EMSR927/UNOSAT/post-event OSM live only in `validation_only/` and are never imported (tests enforce it).

## Attributions
Contains modified Copernicus Sentinel data 2026.
Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
© OpenStreetMap contributors. Kuro Siwo: Bountos et al., 2024. Sen1Floods11 (if used): Bonafilia et al., CVPR Workshops 2020.
EMSR927 (comparison only): European Union, Copernicus Emergency Management Service data.
