import json, requests
from shapely.geometry import shape
from app.services.osm_overpass import HEADERS

gj = json.load(open(r"F:\Hackathons\Multimodal AI IIT Mandi\Testing\FloodLens_v0_4_17\FloodLens\configs\small_aoi.geojson"))
if gj["type"] == "FeatureCollection": gj = gj["features"][0]
geom = gj["geometry"] if gj["type"] == "Feature" else gj
minx, miny, maxx, maxy = shape(geom).bounds
bb = f"{miny:.6f},{minx:.6f},{maxy:.6f},{maxx:.6f}"
H = '[out:json][timeout:180][date:"2026-08-25T00:00:00Z"]'
tests = {
    "any highway": f'way["highway"]({bb});',
    "motorable":   f'way["highway"~"^(trunk|primary|secondary|tertiary|unclassified|residential|track)$"]({bb});',
    "paths only":  f'way["highway"~"^(path|footway)$"]({bb});',
}
for name, body in tests.items():
    r = requests.post("https://overpass-api.de/api/interpreter",
                      data={"data": f"{H};({body});out count;"}, headers=HEADERS, timeout=200)
    try:
        print(name, r.status_code, r.json()["elements"][0]["tags"])
    except Exception:
        print(name, r.status_code, r.text[:200])