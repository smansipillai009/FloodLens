"""
FloodLens Nepal — lean Streamlit dashboard (3 views).  Run:  streamlit run app/frontend/Home.py
The pipeline is called directly (no job server).  Long runs show a progress bar; the CLI
(`python run.py analyse ...`) is the demo fallback.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.api.schemas import ModelMode, RunRequest, AnalysisSummary  # noqa: E402
from app.config import get_config  # noqa: E402

RUNS = ROOT / "outputs" / "runs"
st.set_page_config(page_title="FloodLens Nepal", layout="wide")
st.warning("**Educational prototype — not operational emergency guidance.** All results are satellite-derived "
           "estimates and require field verification.", icon="⚠️")
st.title("FloodLens Nepal")

tab_run, tab_map, tab_access = st.tabs(["1 · Run", "2 · Map", "3 · Access & report"])


def list_runs() -> list[Path]:
    return sorted([p for p in RUNS.glob("*") if (p / "summary.json").exists()], key=lambda p: p.stat().st_mtime, reverse=True)


# ───────────────────────── 1. RUN ─────────────────────────
with tab_run:
    c1, c2 = st.columns(2)
    with c1:
        preset = ROOT / "configs" / "trishuli_demo_aoi.geojson"
        up = st.file_uploader("AOI GeoJSON (or use the Trishuli preset)", type=["geojson", "json"])
        aoi_fc = json.load(up) if up else json.loads(preset.read_text(encoding="utf-8"))
        aoi = aoi_fc["features"][0]["geometry"] if aoi_fc.get("type") == "FeatureCollection" else aoi_fc
        ev = st.date_input("Event date", date(2026, 8, 26))
        mode = st.selectbox("Detection mode", ["baseline", "ai"], help="'ai' = U-Net water + baseline debris; needs a trained checkpoint.")
        optical = st.checkbox("Use clear-pixel Sentinel-2 confirmation", True)
    with c2:
        st.caption("Optional local inputs (terrain-corrected sigma0 GeoTIFF, bands VV,VH). Leave empty to fetch from CDSE.")
        b_tif = st.text_input("Pre-event S1 GeoTIFF(s), comma-separated (first = nearest)")
        a_tif = st.text_input("Post-event S1 GeoTIFF")
        s2_tif = st.text_input("Post-event S2 GeoTIFF (B03,B04,B08,B11,SCL,mask)")
        if st.button("Preview scene selection"):
            from app.services.scene_selector import select_scenes
            sel = select_scenes(aoi, ev)
            st.session_state["sel"] = sel
        sel = st.session_state.get("sel")
        if sel:
            st.markdown(("✅ **Same track**" if sel.same_track else "❌ **No same-track pair**"))
            if sel.s1_before and sel.s1_after:
                st.write(f"Before {sel.s1_before.datetime.date()} → after {sel.s1_after.datetime.date()} "
                         f"· orbit {sel.s1_before.relative_orbit} {sel.s1_before.orbit_direction.value}")
            for w in sel.warnings:
                st.warning(w)

    if st.button("▶ Start analysis", type="primary"):
        from app.services.orchestrator import run_pipeline
        bar, msg = st.progress(0), st.empty()
        req = RunRequest(aoi_geojson=aoi, event_date=ev, event_name="FloodLens run", model_mode=ModelMode(mode),
                         use_optical_confirmation=optical)
        befores = [Path(x.strip()) for x in b_tif.split(",") if x.strip()]
        try:
            summ = run_pipeline(req, s1_before_tif=befores[0] if befores else None, s1_before_extra_tifs=befores[1:],
                                s1_after_tif=Path(a_tif) if a_tif else None, s2_after_tif=Path(s2_tif) if s2_tif else None,
                                progress=lambda s, p: (bar.progress(p), msg.write(s)))
            st.success(f"Run {summ.run_id}: {summ.status.value} in {summ.runtime_s:.0f} s")
            for w in summ.quality.warnings:
                st.warning(w)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Run failed: {exc}. See the log; try the CLI or a smaller AOI.")

runs = list_runs()
sel_run = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name) if runs else None


def load_summary(p: Path) -> AnalysisSummary:
    return AnalysisSummary.model_validate_json((p / "summary.json").read_text(encoding="utf-8"))


# ───────────────────────── 2. MAP ─────────────────────────
with tab_map:
    if not sel_run:
        st.info("No finished run yet.")
    else:
        import folium
        from streamlit_folium import st_folium
        s = load_summary(sel_run)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Flood/debris area (km²)", f"{s.flood_metrics.flood_debris_area_km2:.2f}")
        m2.metric("Potentially affected buildings", s.infrastructure.potentially_affected_buildings)
        m3.metric("Road km potentially affected", f"{s.infrastructure.potentially_affected_road_km:.1f}")
        m4.metric("Potentially cut off", f"{s.settlements_potentially_cut_off} ({s.cut_off_min_over_buffers}–{s.cut_off_max_over_buffers})")
        st.caption(f"Confidence: **{s.quality.flood_confidence.value}** · method: {s.flood_detection_method} · "
                   f"OSM snapshot {s.osm_snapshot_date} · same-track: {s.quality.sar_pair_matched}")
        fm = folium.Map(tiles="OpenStreetMap", zoom_start=11)
        bounds = []

        def add(fname, label, style=None, popup_fields=None, show=True):
            f = sel_run / fname
            if f.exists() and f.stat().st_size > 60:
                data = json.loads(f.read_text(encoding="utf-8"))
                if not data.get("features"):
                    return
                layer = folium.GeoJson(data, name=label, show=show, style_function=style,
                                       popup=folium.GeoJsonPopup(fields=popup_fields) if popup_fields else None)
                layer.add_to(fm)
                bounds.append(layer.get_bounds())

        add("flood_polygons.geojson", "DERIVED · flood/debris candidate",
            lambda f: {"color": "#1d4ed8" if f["properties"].get("class") == "water" else "#a16207",
                       "fillOpacity": 0.45, "weight": 1})
        add("impacted_buildings.geojson", "DERIVED · potentially affected buildings", lambda f: {"color": "#dc2626", "weight": 1})
        add("impacted_roads.geojson", "DERIVED · potentially affected roads", lambda f: {"color": "#ea580c", "weight": 3})
        add("impacted_bridges.geojson", "DERIVED · potentially affected bridges", lambda f: {"color": "#7c3aed", "weight": 5})
        add("route_evidence.geojson", "DERIVED · blocked edges / routes",
            lambda f: {"color": "#000" if f["properties"].get("kind") == "blocked_edge" else "#6b7280", "weight": 3, "dashArray": "4"})
        f = sel_run / "settlement_access.geojson"
        if f.exists():
            colors = {"potentially_cut_off": "red", "uncertain": "gray", "connected": "green"}
            for ft in json.loads(f.read_text(encoding="utf-8"))["features"]:
                lon, lat = ft["geometry"]["coordinates"]
                p = ft["properties"]
                folium.CircleMarker([lat, lon], radius=5, color=colors[p["status"]], fill=True,
                                    tooltip=f"{p['name']}: {p['status'].replace('_', ' ')} ({p['confidence']})",
                                    popup=json.dumps(p.get("status_by_buffer"))).add_to(fm)
        if bounds:
            fm.fit_bounds([[min(b[0][0] for b in bounds), min(b[0][1] for b in bounds)],
                           [max(b[1][0] for b in bounds), max(b[1][1] for b in bounds)]])
        folium.LayerControl().add_to(fm)
        st_folium(fm, height=600, use_container_width=True)
        st.caption("Legend — OBSERVED: OSM base map (pre-event snapshot). DERIVED: blue/brown = likely water/debris change; "
                   "red/orange/purple = potentially affected buildings/roads/bridges; settlement dots red = potentially cut off, "
                   "gray = uncertain, green = connected. Click features for evidence. No reconstructed imagery is shown.")
        try:
            import numpy as np, rasterio
            from PIL import Image
            cols = st.columns(2)
            for col, (fn, title) in zip(cols, (("s1_before_vv.tif", "S1 VV — before"), ("s1_after_vv.tif", "S1 VV — after"))):
                if (sel_run / fn).exists():
                    with rasterio.open(sel_run / fn) as src:
                        a = src.read(1)
                        step = max(1, a.shape[1] // 900)
                        a = a[::step, ::step]
                    db = np.where(a > 0, 10 * np.log10(np.maximum(a, 1e-10)), -25)
                    img = np.clip((db + 25) / 25 * 255, 0, 255).astype("uint8")
                    col.image(Image.fromarray(img), caption=title, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"(SAR preview unavailable: {exc})")

# ───────────────────────── 3. ACCESS & REPORT ─────────────────────────
with tab_access:
    if not sel_run:
        st.info("No finished run yet.")
    else:
        s = load_summary(sel_run)
        import pandas as pd
        if s.settlement_results:
            df = pd.DataFrame([{
                "settlement": r.settlement_name, "status": r.status.value, "confidence": r.confidence.value,
                "km before": r.pre_event_distance_km, "km after": r.hazard_affected_distance_km,
                "by buffer (m)": r.status_by_buffer, "blocked ways": ", ".join(r.evidence_blocked_edge_ids), "reason": r.reason,
            } for r in s.settlement_results])
            order = {"potentially_cut_off": 0, "uncertain": 1, "connected": 2}
            st.dataframe(df.sort_values("status", key=lambda c: c.map(order)), use_container_width=True)
        if s.buffer_sensitivity:
            st.dataframe(pd.DataFrame([b.model_dump() for b in s.buffer_sensitivity]), use_container_width=True)
        rep = sel_run / "situation_report_en.md"
        if rep.exists():
            st.markdown(rep.read_text(encoding="utf-8"))
            st.download_button("Download report (.md)", rep.read_text(encoding="utf-8"), file_name=rep.name)
        st.download_button("Download summary.json", (sel_run / "summary.json").read_text(encoding="utf-8"), file_name="summary.json")
        with st.expander("Limits and attributions"):
            for x in s.limitations + s.attributions:
                st.write("• " + x)
