# Review of v0.1 and what changed in v0.2

## A. Bugs that would have crashed or silently produced wrong results
| # | File | Problem | Fix |
|---|------|---------|-----|
| 1 | orchestrator | `flood_result` undefined when no S1 TIFFs given -> NameError; the run never worked without local TIFFs | rewritten; PARTIAL status with reason |
| 2 | orchestrator | `Transformer.fromcrs` (typo, should be `from_crs`) -> AOI area always 0 (swallowed by try/except) | fixed |
| 3 | flood_detection + terrain | terrain mask is 30 m on its own grid, SAR is 10 m on another -> shape mismatch crash in `apply_terrain_mask` | `resample_to_reference` onto the S1 grid |
| 4 | flood_detection | `flood_polygons.geojson` written in UTM metres with no CRS; downstream read it as lon/lat -> every overlay wrong | polygons now written in EPSG:4326 |
| 5 | sentinel1 | AOI (lat/lon) used to clip a raster already in UTM -> empty/failed crop | single warp onto an analysis grid, AOI rasterised in UTM |
| 6 | terrain | DEM URL missing the `_00` suffixes (`..._N27_00_E085_00_DEM`) -> 404 treated as "ocean" -> DEM silently unavailable | corrected |
| 7 | scene_selector | `cloud or 100.0` turned a clear 0 % scene into 100 % | fixed |
| 8 | docker-compose | starts `app.api.main:app` and `app/frontend/Home.py`, neither existed | removed / replaced |
| 9 | config.yaml | not read anywhere (thresholds hard-coded; VH sign contradicted config) | `app/config.py`; test that keys are wired |
| 10 | AOI preset | 3300 km2 vs own 1000 km2 limit | ~525 km2 preset + auto-shrink |

## B. Rule / honesty violations
* **Mismatched-track fallback**: `_find_best_s1_pair` silently used a different orbit when no match existed (plan: never). Now no pair -> mapping disabled, candidates shown.
* **Fake alignment RMSE** (`pixel*0.5`) was printed in the report as a quality metric. Now measured by phase correlation or reported as "not measured".
* `osm_snapshot_verified_pre_event = True` was hard-coded. Now computed. Snapshot date is clamped to the latest ohsome extract (problem says 27 Jul 2026 while the event is 26 Aug).
* "terrain correction" was only a reprojection and DN->power squaring; not valid for GRD in mountains. Now: CDSE Process API (orthorectified + RTC with Copernicus DEM) or your own SNAP/pyroSAR/openEO output.
* "HAND" was a 51-px minimum filter (local relief). Now pysheds D8 HAND if installed, otherwise explicitly labelled proxy + warning.
* Layover/shadow ignored aspect and look direction (and the 20 deg slope cut made it redundant). Now real geometry.
* Cloud-reconstruction flag still in the request schema (plan forbids). Replaced by clear-pixel S2 confirmation.
* Pharmacies counted as "hospitals"; towns weren't destinations (problem says nearest town **or** hospital).

## C. Method weaknesses fixed
* Water rule now needs drop **and** dark post-event VV and not permanent water (was change-only).
* Separate water / debris classes, areas and polygons (plan 4.4); counts are consistent with the final mask (were pre-filter).
* Multi-scene median pre-event reference; Lee filter with proper nodata handling and ENL.
* Slope limit 20 -> 30 deg (20 deg removes narrow Himalayan gorge floors, exactly where this flood went); separate HAND limits for water/debris.
* Cut-off analysis: multi-source Dijkstra, status evaluated at every buffer, flips => uncertain, range reported; buffer sensitivity now also for buildings and bridges; `track` roads included.
* UTM zone chosen per AOI (EPSG:32645 is wrong west of 84 E).
* ohsome via `bpolys` (real polygon) and failures reported instead of "0 features".

## D. New
`app/config.py`, `cdse_fetch.py`, `s2_confirmation.py`, `app/ml/*` (U-Net train/infer/metrics), `app/frontend/Home.py`,
`scripts/freeze_config.py`, `validation_only/compare_emsr927.py`, 6 new test files/sections.

## E. NOT verified (no network and no rasterio/geopandas/torch in the sandbox where this was written) — do these on Day 1
1. Run `pytest -q`. Pure-numpy/networkx/jinja logic was checked by hand in isolation (Lee filter, slope/aspect, layover, optical classes, graph classification, report render); geopandas/rasterio paths were only syntax-checked.
2. CDSE STAC collection ids/property names (`sentinel-1-grd`, `sat:relative_orbit`, ...) and the Process API request body in `cdse_fetch.py` — check against the docs; adjust config.
3. ohsome `/metadata` field path and the `bridge=*` / `type:way` filter syntax.
4. `prepare_kurosiwo.py` is a skeleton: write `load_sample` after inspecting the dataset (units dB vs linear, pre/post steps, labels).
5. Thresholds in config.yaml are starting values; tune on non-EMSR927 data only, then freeze.
6. Settlement-based cut-off needs OSM hospitals/towns inside the AOI; if none are mapped everything is "uncertain" by design — consider widening the AOI to include Kathmandu/Rasuwa district facilities.

## F. Plan-level notes
* The two plan files conflict (architecture file: FastAPI + cloud reconstruction + optional ensemble; revised plan: lean + no reconstruction). v0.2 follows the revised plan.
* Kuro Siwo labels are flood water only; debris relies on the baseline. Say this in Q&A.
* Judges choose area/date live: the Process-API route (seconds) is what makes that feasible; SNAP locally will not meet a live slot.
* Report mentions the config hash/freeze so the "no tuning on EMSR927" claim is checkable.

---
# v0.3 (second review pass: flood_detection, sentinel1, accessibility, orchestrator, osm_snapshot, cdse_fetch)

## Fixed
* Tests: scene-selection test data had an accidental closer same-track pair (a1/bx, orbit 99); report test now relies on
  `REQUIRED_ATTRIBUTIONS`, which is the default of `AnalysisSummary.attributions` (single source in `app/api/schemas.py`).
* **Edge effect in cut-off analysis**: roads, towns and health facilities are now fetched for the AOI + `osm.access_context_margin_km`
  (10 km). Before, a village whose real escape route left the AOI could be reported cut off. Blocking is still evaluated only inside the AOI.
* **Flood confidence** can only be HIGH with a *measured* S2 agreement >= 60 %; a SAR-only run is capped at MEDIUM (no accuracy evidence).
* Warnings added: post-event scene acquired on the event date; local S1 GeoTIFFs without a verified same-track pair; assumed orbit direction.
* CDSE Process API: `back_coeff` is now configurable and defaults to `GAMMA0_TERRAIN` (radiometric terrain flattening). `SIGMA0_ELLIPSOID`
  does not flatten slopes, contradicting the module docstring. VERIFY the parameter on Day 1.

## Still open (not changed)
1. `prepare_kurosiwo.py` is a skeleton -> the AI component cannot run until `load_sample` is written.
2. CDSE STAC/Process API and ohsome calls are untested against the live services.
3. `alignment_rmse_m` is a global phase-correlation shift, not an RMSE (name kept for schema stability).
4. Debris (VH rise >= 3 dB, no absolute-level test) is the weakest rule: wet/dry soil and vegetation change can pass it. Roads blocked
   by debris-only pixels are counted like water-blocked roads; consider reporting them separately and tuning on non-EMSR927 events.
5. Settlements are snapped to the nearest road *node*, not edge; on sparsely digitised roads this can add up to the tolerance in error.
6. Local-GeoTIFF runs trust the user for same-track compatibility (now warned, not enforced).
7. `to_metric_gdf` / `to4326` per feature are slow for hundreds of settlements; fine for the demo size.

---
# v0.3.1 (verification pass on v0.3)
* Added the missing `python run.py fetch-test` command (v0.3 notes referred to it but it did not exist). It checks: STAC collection ids,
  S1 item property names, ohsome metadata, CDSE OAuth token, Copernicus DEM tile URL. Run it first on Day 1.
* Verified in code (not by execution): margin fetch for roads/towns/health, `back_coeff` config key, `REQUIRED_ATTRIBUTIONS` default.
* Minor: the 10 km margin uses 1 deg = 111 km in both axes, so it is ~8.8 km east-west at 28 N. Harmless; raise the margin if edge cases matter.
* Test count: the files define 57 `test_` functions/methods; "79 passed" depends on parametrisation. Re-run `pytest -q` yourself.

---
# v0.3.2
* **Bug in the new `fetch-test`**: the DEM check built `<base>/<tile>/<tile>.tif` from a name that already contained the folder and file,
  giving a doubled URL that would FAIL even with network access. It now uses `COP_DEM_S3_BASE + _dem_tile_name(...)`, exactly like `terrain.py`
  (regression test added).
* Test count clarified: 57 `def test_` + 22 parametrised cases from `test_compliance.py` = 79 collected, 79 passing (executed, not just read).
* Offline behaviour of `fetch-test` checked: every check prints FAIL with the reason (here 403 from the sandbox proxy) and the exit code is non-zero.
  The live PASS path still needs your network and CDSE credentials.

---
# v0.4: Kuro Siwo GRD adapter
* `scripts/prepare_kurosiwo.py` is now a real converter (was a skeleton): reads `train_GRD/` + `test_GRD/` WebDataset tar shards with the
  standard library, builds `sar[4,H,W]` dB (pre VV/VH, post VV/VH), `y` = new flood (mask == 2) and `valid`, and writes the layout
  `app/ml/dataset.py` expects. Layout and label codes follow the HF dataset card text you supplied; **I could not download the dataset,
  so only synthetic shards were tested.**
* `--inspect N` prints value percentiles, label values, info.json keys, event ids and a unit proposal. Run it first.
* Safeguards: split by event (official `test_GRD` -> `test`; an event in both sides is a hard error), permanent water ignored,
  units never silently guessed (the first-sample proposal is printed and fixed for the run; pass `--units` explicitly),
  `prepare_meta.json` records units / speckle / pre-image choice / events for the report.
* Parity: by default the SAME Lee filter as inference is applied before dB. If Kuro Siwo is already despeckled, compare
  `--speckle none` vs `lee` on a few patches and keep the better match with the live pipeline.
* `--pre sec1|sec2|both`: inference uses a median of 1-3 same-track pre scenes; `both` doubles data and teaches the model both references.
* Unverified until you run it on real data: the info.json event key (use `--event-key`), the dB/linear unit, whether `sec1` is the
  older or newer pre-image, NaN/0 nodata coding, and whether `test_GRD` events are disjoint from `train_GRD` (the script checks this).
* Known gap (unchanged): training pre-image is one scene, inference may use a 3-scene median; DEM is not a model input.

---
# v0.4.1 (verification pass on v0.4 adapter)
* Re-ran the converter by hand on synthetic shards (inspect, convert, leakage detection): works as described.
* Fixed: `--pre both` (or sec2) on data without a second pre-event image skipped every sample and still exited 0 with an empty dataset. Now exits 6 with a clear message; empty splits print a warning.
* Fixed: `--max-per-split` was a no-op (a `continue` at the end of the loop). Now enforced.
* OPEN, important: Day-1 check that the training backscatter convention matches inference. The live fetch asks for GAMMA0_TERRAIN; if Kuro Siwo
  GRD is sigma0 (as is typical), absolute-dB channels are offset by roughly 1-3 dB, more on slopes. The diff channels cancel part of it. Options:
  set `sentinel1.back_coeff` to the convention Kuro Siwo documents, or compare per-channel histograms (train vs a live AOI) before trusting metrics.
* OPEN: Kuro Siwo test events are not Himalayan; held-out metrics do not replace the unseen-Himalayan evaluation the problem asks for.

---
# v0.4.2: backscatter-convention check
* New `scripts/compare_backscatter.py`: compares per-channel dB distributions of the training patches against a live run's
  `s1_{before,after}_{vv,vh}.tif` and estimates the constant offset (median of live - train at p50/p75/p95). Verdicts:
  OK / CONSISTENT_WITH_GAMMA0_VS_SIGMA0 (offset near -10*log10(cos 39 deg) = about 1.1 dB) / OFFSET_UNEXPLAINED. Exit code 1 means act before trusting metrics.
* Method note: a first version used a fixed "land" cut-off (> -18 dB); its own test showed it biases VH (median near the cut) and
  was replaced by upper quantiles, which the dark tail (water, shadow, flood) cannot distort.
* Limits: scenes differ (terrain, season, vegetation), so this is a rough screen. Use two or three AOIs; slopes can add several dB on
  top of the flat-ground gamma/sigma difference. It detects an offset; it does not say which convention is "right".

---
# v0.4.8: first live run (small AOI, 2026-08-26) failed in two places
* **CDSE Process API 400**: the real cause is only visible in the JSON body, which was swallowed by `raise_for_status()`. `_post` now raises
  `RuntimeError("Process API HTTP 400: <server message>")`. `orbitDirection` was dropped from `dataFilter` (redundant: the +-90 s window pins one
  acquisition). Root cause NOT yet confirmed: re-run and read the message.
* **ohsome 403 (HTML, Apache) on POST /elements/geometry** while GET /metadata works: a front-end rule, not a data problem. 403 is no longer
  retried; the request is repeated with a bounding box and the features are clipped to the exact AOI locally (still an explicit pre-event `time`).
  A descriptive User-Agent is sent. If a plain `curl` POST to ohsome also returns 403 from your network, the block is network-level.

---
# v0.4.9: the 400 was width=0/height=0 (degenerate AOI)
* Process API said `output.width/height = 0`: the AOI file had (almost) zero extent (a marker/line or a sliver). `load_aoi` now picks the first
  POLYGON in the file (geojson.io files often start with a marker), `_size_for` refuses sub-pixel AOIs, and `aoi.min_area_km2` (1 km2) is
  validated with a message that says what to draw. A degenerate AOI may also explain the ohsome 403s (zero-area bbox/bpolys): unconfirmed.
* `python run.py diagnose-ohsome` runs a ladder of 8 small requests and shows which request shape gets a 403 on your network
  (a plain curl POST to /elements/count works, so the network is not blocked).

---
# v0.4.10: first successful live S1 run: flood area 0.00 km2, why?
* The Sentinel-1 download and processing now work (73 s, small AOI). Detection found nothing; the cause is not visible from the summary alone.
* `detect_flood` now writes `detection_funnel.json` into the run folder (pixel counts after each rule: VV drop, post-dark, not permanent water, HAND, terrain mask,
  min patch size; plus pre/post dB and log-ratio percentiles and the thresholds used) and logs it when nothing is found.
  Reading it tells you whether the AOI simply has no change (e.g. does not cover the flooded corridor), the thresholds are too strict, or a mask removes everything.

---
# v0.4.11: ohsome blocks /elements/geometry; AOI had no flood signal
* `diagnose-ohsome` (your run): /elements/count works (HTTP 200), EVERY /elements/geometry request is 403 regardless of bbox/polygon, properties or User-Agent.
  So the public ohsome API refuses geometry extraction from your network (cause unknown: policy or per-IP).
* New `app/services/osm_overpass.py`: pre-event snapshot through the Overpass ATTIC date (`[date:"<snapshot>T00:00:00Z"]`), used automatically after ohsome 403.
  Compliance guard: no query is ever built without the date header, the snapshot must be strictly before the event, and the module is the only one allowed to name an Overpass
  endpoint (tests). Relations (multipolygon buildings) are skipped. Attic coverage and rate limits are UNVERIFIED live: check the OSM counts on your first run.
* The first run's `detection_funnel.json` shows log-ratio percentiles symmetric around 0 (VV p05/p95 = -1.9/+1.7 dB), i.e. speckle-level noise only: the AOI contained no
  detectable flood signal. HAND then removed 99.6% of the (noise) water candidates: that is the filter doing its job, not a bug.
* New `scripts/preview_run.py`: one PNG per run (pre/post VV, VV/VH change, detected classes).

---
# v0.4.12: run 75683f5b, real change found but HAND proxy rejected it
* Funnel (21.9 km2 AOI): water 2527 -> 519 (post dark) -> 493 (not permanent) -> **0 after HAND**; debris 3347 -> 3110 -> **14 after HAND**. HAND here is only a ~510 m local-relief proxy
  (pysheds not installed), so the true-HAND limits (15 m / 40 m) were applied to the wrong quantity. The proxy now has its own limits (`hand_proxy_max_m_water/debris` = 60 / 150 m, UNVALIDATED),
  and the funnel records the HAND percentiles at the candidates (`hand_at_candidates_m`) so the limits can be tuned on evidence. With pysheds installed the original limits apply.
* OSM via Overpass: `buildings` and `settlements` failed (reason not recorded). Added: 3 s pause between layers, 10 s backoff steps, and the failure reason per layer in the summary warning.
* No hospital/clinic/town found within AOI + 10 km: `access_context_margin_km` raised to 20.

---
# v0.4.13: UnicodeEncodeError, noise, proxy HAND
* **Crash**: `summary.json` could not be written on Windows (cp1252) because OSM names contain characters such as U+1E45; the file was left EMPTY. Every text read/write now names `encoding="utf-8"`
  (58 call sites), the console is switched to UTF-8, and a test scans the code for any text I/O without an encoding. (A first automated edit also pushed `encoding=` into `rasterio.open` calls;
  that was reverted and a test now forbids it.)
* **Noise** (run ce8056d9, 10 m): VV/VH log-ratio p05/p95 = -2.25/+2.38 and -2.51/+2.77 dB, i.e. the +-3 dB thresholds sit inside the speckle tail (44,516 'water drop' pixels, 72,860 'debris rise' pixels, mostly noise).
  New `sentinel1.multilook_factor: 3` block-averages the Lee-filtered LINEAR images to 30 m (synthetic 4-look speckle: log-ratio std reduced by more than half). Set 1 to switch it off.
* **Proxy HAND is no longer a filter** (`dem.hand_proxy_filter: false`): at the candidates the relief proxy has p50 = 640 m (water) / 498 m (debris), so no limit separates signal from terrain. It is still reported in the funnel.
  Slope and radar-shadow masks stay. Install pysheds to get real HAND (limits 15 / 40 m then apply again).
* Interpretation caution: the strongest coherent change in the preview sits in very high-relief terrain. Seasonal wet snow / glacier change or the event's source area both produce dark patches there; C-band SAR alone cannot tell them apart.

---
# v0.4.14: baseline improvements driven by run e1a1ac76 (30 m grid, 332k px)
* The preview shows a long, narrow NW-SE band of strong VV/VH change ending in a dark blob (a candidate corridor), but the detector returned it in fragments (429 water px, 1351 debris px)
  and the change maps carry smooth km-scale red/blue banding. Multilook x3 hardly changed the change-map percentiles (VV p05/p95 -2.25/+2.38 -> -1.99/+2.31): the dominant "noise" is NOT speckle
  but real between-date variability (soil moisture / gain / rain), so averaging cannot remove it.
* `flood_detection.detrend_sigma_m` (1200 m): a clipped, Gaussian-weighted regional mean of the change is subtracted before thresholding (synthetic test: residual std close to the white-noise floor, corridor kept).
* `flood_detection.hysteresis` (on): the existing thresholds are SEEDS; weaker 'grow' thresholds (-1.5 dB water, -12 dB post, +1.5 dB debris) add only pixels connected to a seed
  (synthetic corridor IoU 0.84 -> 0.91, false-alarm rate < 5%). `detection_funnel.json` now has `seeds_px`, `after_hysteresis_px` and the before/after-detrend change percentiles.
* NEW `scripts/tune_baseline_kurosiwo.py`: grid search of the water thresholds on labelled Kuro Siwo TRAIN events, scored on VAL and TEST events, mimicking the live multilook. Ties pick the strictest rule.
  It tunes the simple rule only (not detrend/hysteresis) and only for flood WATER outside the Himalaya.
* None of this is validated on real Nepal data yet: rerun and read the funnel/preview before trusting it.

---
# v0.4.15: run ccb787c2 (detrend + hysteresis): more detections, mostly noise
* Funnel: debris seeds 6,571 px (2.2% of valid pixels = a noise-tail rate: robust sigma ~ 1.3 dB, so the 3 dB seed is only 2.3 sigma) were tripled to 19,164 px by growing at 1.5 dB (1.2 sigma).
  Final: water 1,143 px, debris 5,261 px (was 429 / 1,351); the preview shows scattered specks over the hillsides besides the NW-SE band. Detrending changed the percentiles little
  (VV p05/p95 -2.04/+2.13 vs -1.99/+2.31): the banding was not the main problem (kept, harmless).
* `flood_detection.adaptive_thresholds` (on): seeds = 3.0 x robust sigma, grow = 1.75 x robust sigma of the detrended change (1.4826 x MAD), never weaker than the fixed values; effective values
  and the measured sigma are written to `detection_funnel.json` (`noise`, `thresholds`, `thresholds_are_adaptive`).
  `min_area_m2` 2,500 -> 20,000 (2,500 m2 is only 3 px at 30 m).
* Synthetic check (noise sigma 1.9 dB, 15-px-wide dark corridor): seeds -5.8 dB, all 1,200 corridor pixels found, no false detections. NOT yet validated on real data.

---
# v0.4.16: tuning script printed an all-zero table with no explanation
* First real run of `tune_baseline_kurosiwo.py` (3000 train / 29 val / 3000 test patches): IoU 0.000 for every rule on every split, plus NaN warnings from an empty split.
  An all-zero table means the labels and the backscatter values do not line up (typical: data converted with the wrong `--units`, or the label/valid arrays removed every flood pixel).
* The script now prints per split: pixels, flood %, post-VV and change percentiles, flags implausible values (median post VV outside -25..0 dB, large median change, <0.1% flood, empty splits),
  survives empty splits, and refuses to suggest thresholds when no rule scores above 0 (exit code 3).

---
# v0.4.17: Kuro Siwo labels vs images
* `prepare_meta.json` (units linear, Lee speckle, 5,970 train / 29 val / 11,996 test patches, 16 train events, 1 val event, 10 test events, no leaky events) and one train patch
  (pre VV median -9.84 dB, post -8.34 dB, 372 flood px of 50,176) look physically plausible, so the all-zero IoU table from `tune_baseline_kurosiwo.py` is not a units problem.
* New `scripts/check_kurosiwo_alignment.py`: compares how dark the labelled flood pixels are vs land, for the identity orientation and for transposed/flipped/rotated labels.
  Verdicts: OK / WEAK CONTRAST (labels hard for SAR: use the U-Net) / ORIENTATION MISMATCH (prepare_kurosiwo.py must transform mask and valid_mask).

---
# v0.4.18: alignment OK, but the tuner still sees 0 pixels
* `check_kurosiwo_alignment.py` on 200 train patches: VERDICT OK. Labelled flood pixels sit at -15.7 dB vs -8.9 dB for land (6.8 dB darker; 61% of flood pixels below -15 dB vs 0.4% of land).
  Oddities not yet explained: the flipud row equals the identity row exactly, and transpose/rot90/rot270 report 'no data' (shape mismatch => patches are probably not square).
* `tune_baseline_kurosiwo.py` still reports 0 usable pixels on all splits although the same files work in the checker. I could not reproduce this with synthetic float16 patches of 224x224 or 196x256, with or without
  multilook. The script now prints a 'WHY EMPTY?' block (array shapes/dtypes, finite %, valid mean, pixels surviving each step) when every split is empty. Paste that block.
* Training (`app.ml.train`) reads the same patches through `PatchDataset` and does not use the tuner: the labels-vs-images check passing is the gate for starting it.

---
# v0.4.19: U-Net trained (Colab T4, 15 epochs); live-pipeline grid parity
* Training result (`models/checkpoints/metrics.json`): val IoU 0.729 / F1 0.844 (checkpoint chosen on val, so optimistic); HELD-OUT test events IoU 0.617 / F1 0.763,
  precision 0.670, recall 0.887, predicted area +32% vs label. Val IoU wandered 0.64-0.73 between epochs (small val set), loss still falling slowly (0.19 -> 0.18).
  Kuro Siwo test events are not Himalayan: the Himalayan check is EMSR927 (check-only).
* PARITY BUG FOUND: the U-Net is trained on 10 m Lee-filtered patches, but v0.4.13+ multilooks the live S1 grid to 30 m before inference. AI mode now processes S1 at 10 m
  (`preprocess_s1_pair(multilook_factor=1)`; baseline-only runs keep config multilook 3) and adds a warning. Test added (first unit test of `preprocess_s1_pair`).
* `train.py`: loss is accumulated with `float(l.detach())` (autograd warning). Affine `*` -> `@` (deprecation).
