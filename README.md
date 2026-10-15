# FloodLens Nepal

**Track B, Multimodal AI Hackathon 2026 (IIT Mandi): Mapping Flood Damage from Space.**
Team: LaLaLand    Team Members: Mansi Pillai

<img width="1872" height="857" alt="Screenshot 2026-10-14 220132" src="https://github.com/user-attachments/assets/475a5f62-b0af-4001-a1c1-cf4cec187a9b" />

<img width="1895" height="952" alt="Screenshot 2026-10-14 220216" src="https://github.com/user-attachments/assets/62bff717-87c5-4c72-b5ab-aaf8bbe6b1ff" />



> **Educational prototype, not operational emergency guidance.** All results are satellite-derived estimates that require field verification. No victim imagery is used anywhere.

FloodLens takes an **area and a flood date** and answers three questions from free satellite and map data:

1. **Where did the flood hit?** Two-sided Sentinel-1 radar change detection (water and debris), terrain masks from the Copernicus DEM, an optional U-Net for flood water, and Sentinel-2 confirmation on clear pixels.
2. **What was potentially damaged?** Pre-event OpenStreetMap buildings, roads and bridges overlaid on the flood/debris mask.
3. **Who is potentially cut off?** A motorable-road graph before the event (G0) versus the graph with hazard-affected sections removed (G1), with buffer sensitivity.

Every number is written to `summary.json`; the situation report is generated from that file only.

## Case-study snapshot (run 4849e43e)

Langtang-area AOI, 189 km², event 26 Aug 2026. Sentinel-1 pair 16 Aug to 28 Aug 2026, relative orbit 85 ascending (same track).

| Result | Value |
|---|---|
| Likely flood/debris change area | 0.58 km² (water 0.36, debris 0.22), medium confidence, radar only |
| U-Net, held-out Kuro Siwo events | IoU 0.617, F1 0.763, precision 0.670, recall 0.887 |
| EMSR927 comparison (check-only) | [[FILL: IoU / F1 / precision / recall]] |
| Roads, hospitals, cut-off settlements | **Not assessed** for this run: the public OSM server returned HTTP 504 for roads and health facilities |

See the report for the full evaluation and limitations.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install torch segmentation-models-pytorch   # needed for --model-mode ai
pip install pysheds                              # optional: true HAND filter (needs numpy < 2.4 on some setups)
copy .env.example .env                           # then fill CDSE_CLIENT_ID and CDSE_CLIENT_SECRET
python -m pytest -q                              # compliance + unit tests must pass



Files larger than 25MB have a google drive link atteched to download
\data\reference\dem.zip
\data\processed\kurosiwo.zip
\models\checkpoints\unet_resnet18.pt
\kurosiwo_patches.zip
\outputs\runs\4849e43e
```

Create a free Copernicus Data Space account and an OAuth client at <https://shapps.dataspace.copernicus.eu/dashboard/> (User settings, OAuth clients). **Never commit `.env`.**

## Run

```bash
python run.py search  --aoi configs/trishuli_demo_aoi.geojson --date 2026-08-26   # scene pair + same-track check
python run.py osm     --aoi configs/trishuli_demo_aoi.geojson --date 2026-08-26   # pre-event OSM layer counts
python run.py analyse --aoi configs/trishuli_demo_aoi.geojson --date 2026-08-26 --model-mode ai
streamlit run app/frontend/Home.py                                                 # dashboard
```

- `--model-mode ai` uses the U-Net for water and the baseline for debris. Without a checkpoint it falls back to the baseline with a warning.
- Outputs go to `outputs/runs/<run_id>/`: `manifest.json`, `summary.json`, rasters, GeoJSON layers, `situation_report_en.md`, `detection_funnel.json`.
- If the public OSM servers are busy, fill the local cache first (safe to repeat; it only fetches missing tiles):
  ```bash
  python scripts/warm_osm_cache.py --aoi configs/trishuli_demo_aoi.geojson --date 2026-08-26
  ```
- Preview a run: `python scripts/preview_run.py --run outputs/runs/<run_id> --out preview.png`

## AI component (U-Net, ResNet-18 encoder)

Trained on Kuro Siwo with an event-level train/validation/test split; six input channels (pre VV/VH, post VV/VH, two differences, dB); trained from scratch (no ImageNet weights, to stay within the permitted datasets).

```bash
python scripts/prepare_kurosiwo.py --src <Kuro-Siwo-Webdataset> --inspect 20     # check units, labels, event key first
python scripts/prepare_kurosiwo.py --src <Kuro-Siwo-Webdataset> --dst data/processed/kurosiwo --units <db|linear>
python -m app.ml.train --data data/processed/kurosiwo --epochs 15 --max-train-patches 4000     # we trained on a Colab T4 GPU
python scripts/tune_baseline_kurosiwo.py ...                                      # baseline row for the metrics table
```

The trained checkpoint, normalisation statistics and metrics are in `models/checkpoints/` (`unet_resnet18.pt`, `norm_stats.json`, `metrics.json`). Held-out-event metrics are **not** Himalayan results; the Himalayan check is the EMSR927 comparison below.

## Validation (check-only)

EMSR927 and any published damage map are used **only to check** results, never as inputs.

```bash
python scripts/freeze_config.py            # BEFORE opening EMSR927; do not edit configs afterwards
python validation_only/compare_emsr927.py --run outputs/runs/<run_id> --ref validation_only/emsr927/<vector file>
```

The comparison script refuses to run if the configuration differs from `configs/config.lock`.

## Data rules and how they are enforced

- **Inputs:** Sentinel-1, Sentinel-2, Copernicus DEM GLO-30, OSM as it was before the event, Kuro Siwo.
- **Same-track Sentinel-1 only:** the two scenes must share relative orbit and direction; otherwise the run stops with the exact reason.
- **OSM snapshot strictly before the event:** every request carries an explicit date. The ohsome API is preferred; when its geometry endpoint is blocked (HTTP 403) the Overpass historical (attic) date feature is used, with queue-aware retries, tiling and an on-disk cache. Current-data services are never used.
- `validation_only/` is never imported by pipeline code. Automated tests enforce all of the above (`python -m pytest -q`).

## Known limitations

No early warning (satellites revisit every few days); radar false positives on steep terrain, shadow and wet soil; narrow channels under-detected; the U-Net is trained on flood water outside the Himalaya and debris comes from the radar baseline only; optical data is cloud-limited; a mask cannot confirm structural damage; OSM can be incomplete and a mapped road path does not mean a road is passable; live runs depend on public servers and can be slow. See the report, section 7.

## Attributions

Contains modified Copernicus Sentinel data 2026.

Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

© OpenStreetMap contributors.

Training data: Bountos, N. I. et al. (2024). *Kuro Siwo: 33 billion m² under the water! A global multi-temporal satellite dataset for rapid flood mapping.* NeurIPS Datasets and Benchmarks.

European Union, Copernicus Emergency Management Service data (EMSR927, used for comparison only, never as an input).
