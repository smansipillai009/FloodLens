"""
FloodLens Nepal — server-side S1 / S2 retrieval via the CDSE Sentinel Hub Process API.

Why: the original code expected ready-made GeoTIFFs and had no way to get them.
Raw S1 GRD .SAFE zips are NOT GeoTIFFs and need SNAP/pyroSAR for terrain correction
(slow, heavy). The Process API does orthorectification + radiometric terrain
flattening server-side with the Copernicus DEM, returns an AOI-clipped GeoTIFF in
seconds and is an official CDSE route (still "raw Sentinel-1 data" -> allowed).

!! NOT TESTED against the live service in the sandbox that produced this file (no
!! network).  Day-1 task: run `python run.py fetch-test` and fix any parameter
!! names against https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/
Credentials: OAuth client created in the CDSE dashboard -> env CDSE_CLIENT_ID /
CDSE_CLIENT_SECRET (never committed).
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

import requests
from shapely.geometry import mapping, shape

from app.api.schemas import S1SceneInfo, S2SceneInfo
from app.config import get_config, utm_crs_for_geom

logger = logging.getLogger(__name__)
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


def get_token() -> str:
    cid, secret = os.getenv("CDSE_CLIENT_ID"), os.getenv("CDSE_CLIENT_SECRET")
    if not cid or not secret:
        raise RuntimeError("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET (see .env.example).")
    r = requests.post(TOKEN_URL, data={"grant_type": "client_credentials",
                                       "client_id": cid, "client_secret": secret}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def _size_for(aoi: dict, res_m: float, max_px: int) -> tuple[int, int, float, Optional[str]]:
    """Pixel size for the request; coarsens resolution automatically if the AOI is too big."""
    minx, miny, maxx, maxy = shape(aoi).bounds
    w_m = (maxx - minx) * 111320 * math.cos(math.radians((miny + maxy) / 2))
    h_m = (maxy - miny) * 110570
    warn = None
    if max(w_m, h_m) / res_m > max_px:
        res_m = math.ceil(max(w_m, h_m) / max_px)
        warn = f"AOI too large for {max_px}px at 10 m; auto-coarsened to {res_m:.0f} m (speed/limit)."
    w, h = int(w_m / res_m), int(h_m / res_m)
    if w < 1 or h < 1:
        raise ValueError(f"AOI is only {w_m:.0f} m x {h_m:.0f} m, which is smaller than one {res_m:.0f} m pixel. "
                         "Is it a point/line instead of a polygon? Draw a polygon of 50-200 km2.")
    return w, h, res_m, warn


S1_EVALSCRIPT = """//VERSION=3
function setup(){return{input:[{bands:["VV","VH","dataMask"]}],
  output:{bands:3,sampleType:"FLOAT32"}}}
function evaluatePixel(s){return [s.VV, s.VH, s.dataMask]}"""

# SCL classes: 3 cloud shadow, 8/9 cloud med/high, 10 thin cirrus, 1 saturated, 0 nodata
S2_EVALSCRIPT = """//VERSION=3
function setup(){return{input:[{bands:["B03","B04","B08","B11","SCL","dataMask"]}],
  output:{bands:6,sampleType:"FLOAT32"}}}
function evaluatePixel(s){return [s.B03,s.B04,s.B08,s.B11,s.SCL,s.dataMask]}"""


def _post(body: dict, files_out: Path, token: str) -> Path:
    r = requests.post(PROCESS_URL, headers={"Authorization": f"Bearer {token}",
                                            "Accept": "image/tiff"}, json=body, timeout=300)
    if not r.ok:
        # Sentinel Hub explains a 400 in the JSON body; a bare "400 Bad Request" hides the real cause
        raise RuntimeError(f"Process API HTTP {r.status_code}: {r.text[:1500]}")
    files_out.write_bytes(r.content)
    return files_out


def fetch_s1_scene(scene: S1SceneInfo, aoi: dict, out_path: Path, token: Optional[str] = None) -> tuple[Path, list[str]]:
    """Orthorectified (COP-DEM) linear backscatter; back_coeff GAMMA0_TERRAIN = radiometric terrain flattening
    (SIGMA0_ELLIPSOID would NOT flatten slopes, which matters in the Himalaya),
    bands [VV, VH, dataMask], clipped to the AOI, restricted to ONE acquisition by a tight
    time range around the scene time (this is how the same-track pair is honoured)."""
    cfg = get_config()
    s1 = cfg["sentinel1"]
    token = token or get_token()
    w, h, res, warn = _size_for(aoi, s1["resolution_m"], s1["max_pixels_per_side"])
    t0, t1 = scene.datetime - timedelta(seconds=90), scene.datetime + timedelta(seconds=90)
    body = {
        "input": {
            "bounds": {"geometry": mapping(shape(aoi)),
                       "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {"timeRange": {"from": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                             "to": t1.strftime("%Y-%m-%dT%H:%M:%SZ")},
                               "acquisitionMode": cfg["scene_selection"]["instrument_mode"],
                               "polarization": "DV"},
                "processing": {"orthorectify": True, "demInstance": "COPERNICUS_30",
                               "backCoeff": s1.get("back_coeff", "GAMMA0_TERRAIN"), "radiometricTerrainOversampling": 2,
                               "speckleFilter": {"type": "NONE"}},
            }],
        },
        "output": {"width": w, "height": h,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": S1_EVALSCRIPT,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _post(body, out_path, token)
    logger.info("S1 %s -> %s (%dx%d px @ %.0f m)", scene.scene_id, out_path.name, w, h, res)
    return out_path, ([warn] if warn else [])


def fetch_s2_scene(scene: S2SceneInfo, aoi: dict, out_path: Path, token: Optional[str] = None) -> Path:
    """L2A bands B03,B04,B08,B11 (reflectance) + SCL + dataMask at 20 m."""
    token = token or get_token()
    w, h, _, _ = _size_for(aoi, 20, 2500)
    t0, t1 = scene.datetime - timedelta(minutes=2), scene.datetime + timedelta(minutes=2)
    body = {
        "input": {"bounds": {"geometry": mapping(shape(aoi)),
                             "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                  "data": [{"type": "sentinel-2-l2a",
                            "dataFilter": {"timeRange": {"from": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                         "to": t1.strftime("%Y-%m-%dT%H:%M:%SZ")}}}]},
        "output": {"width": w, "height": h,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": S2_EVALSCRIPT,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return _post(body, out_path, token)
