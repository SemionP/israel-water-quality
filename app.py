"""
app.py
=============================================================================
AquaWatch Global — Water Quality Monitor Dashboard
Updated version:
  - Atmospheric context updates based on map center movement (>50 km)
  - Atmospheric overlay card on the map (Folium Control Box)
=============================================================================
"""

import math
import json
import tempfile
import os
from typing import Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
import ee
from branca.element import MacroElement
from jinja2 import Template

# Basic configuration
st.set_page_config(page_title="AquaWatch Global — Sentinel WQI", layout="wide")

# ==============================================================================
# SEO & Analytics
# ==============================================================================
seo_html = """
<meta name="description" content="A scientific real-time water quality monitoring system (Mediterranean Sea, Sea of Galilee, Dead Sea, Red Sea) using remote sensing, Sentinel satellite data, and Google Earth Engine." />
<meta name="keywords" content="water quality, remote sensing, satellite, Sentinel-2, GEE, Sea of Galilee, Dead Sea, turbidity, chlorophyll, algae, Water Quality Israel, Remote Sensing, Google Earth Engine" />
<meta property="og:title" content="AquaWatch Global — Satellite Water Quality Monitor" />
<meta property="og:description" content="Advanced scientific monitoring of turbidity, chlorophyll, and algal blooms in water bodies using remote sensing." />
<meta property="og:type" content="website" />
<meta property="og:url" content="https://aquawatch-global.streamlit.app/" />
"""
st.markdown('<meta name="google-site-verification" content="INSERT_GOOGLE_VERIFICATION_CODE_HERE" />', unsafe_allow_html=True)

# Analytics components
components.html('<script async src="https://cloud.umami.is/script.js" data-website-id="07a48db1-5aa7-4d88-aaac-9cfb6fc2600d"></script>', height=0)

if "ga_loaded" not in st.session_state:
    st.session_state.ga_loaded = True
    components.html(
        """
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-K37THY2160"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
          gtag('config', 'G-K37THY2160');
        </script>
        """,
        height=0,
    )

# =============================================================================
# 1. Atmospheric Context Module (Open-Meteo)
# =============================================================================
_WB_CENTRES = {
    "🏖️ Mediterranean Coast": (32.40, 34.85),
    "🌊 Sea of Galilee":      (32.82, 35.59),
    "🧂 Dead Sea":            (31.50, 35.47),
    "🐠 Red Sea":             (29.55, 34.95),
}

@st.cache_data(ttl=3600)
def get_atmospheric_context(wb_key: str) -> dict:
    empty = _empty_atm()
    lat, lon = _WB_CENTRES.get(wb_key, (32.0, 35.0))
    try:
        import requests as _requests
        resp = _requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":       lat,
                "longitude":      lon,
                "current":        ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "precipitation",
                    "weather_code",
                ]),
                "wind_speed_unit": "ms",
                "forecast_days":   1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        cur = resp.json().get("current", {})

        wind_speed = cur.get("wind_speed_10m")
        wind_dir   = cur.get("wind_direction_10m")
        temp_c     = cur.get("temperature_2m")
        humidity   = cur.get("relative_humidity_2m")
        precip_mm  = cur.get("precipitation")
        wcode      = cur.get("weather_code", 0)

        return {
            "wind_speed":    round(wind_speed, 1) if wind_speed is not None else None,
            "wind_dir_deg":  round(wind_dir,   1) if wind_dir   is not None else None,
            "temp_c":        round(temp_c,     1) if temp_c     is not None else None,
            "humidity":      round(humidity,   0) if humidity   is not None else None,
            "precip_mm":     round(precip_mm,  2) if precip_mm  is not None else None,
            "weather_code":  wcode,
            "analysis_time": cur.get("time", "—"),
            "centre_lat":    lat,
            "centre_lon":    lon,
            "_error":        None,
            "_source":       "Open-Meteo (GFS/ERA5)",
        }
    except Exception as exc:
        empty["_error"] = f"Error loading weather data: {exc}"
        return empty

def _empty_atm() -> dict:
    return {
        "wind_speed": None, "wind_dir_deg": None, "temp_c": None,
        "humidity": None, "precip_mm": None, "weather_code": None,
        "analysis_time": None, "centre_lat": None, "centre_lon": None,
        "_error": None, "_source": "Open-Meteo (GFS/ERA5)",
    }

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two geographic points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

@st.cache_data(ttl=3600)
def get_atmospheric_context_by_coords(lat: float, lon: float) -> dict:
    """Load atmospheric data by arbitrary coordinates."""
    empty = _empty_atm()
    try:
        import requests as _requests
        resp = _requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":        round(lat, 4),
                "longitude":       round(lon, 4),
                "current":         ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "precipitation",
                    "weather_code",
                ]),
                "wind_speed_unit": "ms",
                "forecast_days":   1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        cur = resp.json().get("current", {})

        wind_speed = cur.get("wind_speed_10m")
        wind_dir   = cur.get("wind_direction_10m")
        temp_c     = cur.get("temperature_2m")
        humidity   = cur.get("relative_humidity_2m")
        precip_mm  = cur.get("precipitation")
        wcode      = cur.get("weather_code", 0)

        return {
            "wind_speed":    round(wind_speed, 1) if wind_speed is not None else None,
            "wind_dir_deg":  round(wind_dir,   1) if wind_dir   is not None else None,
            "temp_c":        round(temp_c,     1) if temp_c     is not None else None,
            "humidity":      round(humidity,   0) if humidity   is not None else None,
            "precip_mm":     round(precip_mm,  2) if precip_mm  is not None else None,
            "weather_code":  wcode,
            "analysis_time": cur.get("time", "—"),
            "centre_lat":    lat,
            "centre_lon":    lon,
            "_error":        None,
            "_source":       "Open-Meteo (GFS/ERA5)",
        }
    except Exception as exc:
        empty["_error"] = f"Error loading weather data: {exc}"
        return empty

def blend_atmospheric_penalty(df, atm: dict, wb_key: str):
    def _penalty(row) -> float:
        if row.get("wqi") is None:
            return 0.0
        p   = 0.0
        ws  = atm.get("wind_speed")
        pr  = atm.get("precip_mm")
        rh  = atm.get("humidity")

        if ws is not None:
            if ws > 15:    p += 10.0
            elif ws > 10:  p += 7.0
            elif ws > 7:   p += 4.0
            elif ws > 4:   p += 1.5

        if pr is not None:
            if pr > 5:     p += 8.0
            elif pr > 2:   p += 5.0
            elif pr > 0.5: p += 2.0

        closed = {"🌊 Sea of Galilee", "🧂 Dead Sea", "🐠 Red Sea"}
        if wb_key in closed and rh is not None:
            if rh > 85:    p += 7.0
            elif rh > 70:  p += 3.0

        return min(p, 25.0)

    df = df.copy()
    df["atm_penalty"] = df.apply(_penalty, axis=1)
    df["composite_with_atm"] = df.apply(
        lambda r: (
            round(max(0.0, r["wqi"] - r["atm_penalty"]), 1)
            if r["wqi"] is not None else None
        ),
        axis=1,
    )
    return df

def render_earth2_sidebar(atm: dict, wb_key: str) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌍 Atmospheric Context\n<small style='color:#888;'>Open-Meteo · GFS/ERA5 · Hourly Update</small>", unsafe_allow_html=True)

    if atm.get("_error"):
        st.sidebar.warning(atm["_error"])
        return

    ws  = atm.get("wind_speed")
    wd  = atm.get("wind_dir_deg")
    tc  = atm.get("temp_c")
    pr  = atm.get("precip_mm")
    rh  = atm.get("humidity")

    if ws is not None:
        arrows = ["↑","↗","→","↘","↓","↙","←","↖"]
        arrow = arrows[int((wd + 22.5) / 45) % 8] if wd is not None else ""
        bf = 12
        for b, t in enumerate([0.3,1.6,3.4,5.5,8.0,10.8,13.9,17.2,20.8,24.5,28.5,32.7]):
            if ws < t: bf = b; break
        st.sidebar.metric(label=f"💨 Wind {arrow}", value=f"{ws:.1f} m/s", delta=f"Beaufort {bf}", delta_color="inverse" if bf >= 5 else "normal")

    if tc is not None: st.sidebar.metric("🌡️ Temperature", f"{tc:.1f} °C")
    if pr is not None: st.sidebar.metric("🌧️ Rain" if pr > 0.5 else "☀️ Dry", f"{pr:.1f} mm/h")
    if rh is not None: st.sidebar.metric("💧 Humidity", f"{int(rh)}%")

    score = 0
    reasons = []
    if ws and ws > 7: score += 1; reasons.append("Strong wind")
    if pr and pr > 0.5: score += 2; reasons.append("Precipitation")
    
    if score == 0:
        st.sidebar.markdown('<div style="background:#f8f9fa;border-radius:10px;padding:10px;border-left:4px solid #27AE60;"><b>✅ Low Atmospheric Risk</b></div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f'<div style="background:#f8f9fa;border-radius:10px;padding:10px;border-left:4px solid #F1C40F;"><b>🟡 Moderate Risk</b><br><span style="font-size:12px;">{", ".join(reasons)}</span></div>', unsafe_allow_html=True)

# =============================================================================
# 2. Google Earth Engine (GEE) Authentication
# =============================================================================
@st.cache_resource
def init_gee():
    creds_dict = dict(st.secrets["gee_credentials"])
    creds_json = json.dumps(creds_dict)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(creds_json)
        tmp_path = f.name
    service_account = creds_dict["client_email"]
    credentials = ee.ServiceAccountCredentials(service_account, tmp_path)
    ee.Initialize(credentials)
    os.unlink(tmp_path)

init_gee()

# =============================================================================
# 3. Geometries and Built-in Legend
# =============================================================================
HAIFA_CENTER = [32.4, 34.85]
HAIFA_BBOX   = ee.Geometry.Rectangle([34.20, 31.20, 35.20, 33.20])

ISRAEL_TERRITORIAL = ee.Geometry.Polygon([[
    [34.95, 33.10], [34.55, 33.10], [34.15, 32.50], [34.10, 32.00],
    [34.15, 31.50], [34.50, 31.25], [34.75, 31.25], [34.95, 31.30],
    [35.02, 31.60], [35.00, 32.10], [35.05, 32.60], [35.10, 33.10],
    [34.95, 33.10]
]])

KINNERET_BBOX = ee.Geometry.Rectangle([35.48, 32.70, 35.68, 32.95])
DEAD_SEA_BBOX = ee.Geometry.Rectangle([35.35, 31.05, 35.58, 31.80])
RED_SEA_BBOX  = ee.Geometry.Rectangle([34.85, 29.40, 35.02, 29.60])

BEACHES = [
    {"name": "Rosh HaNikra", "lat": 33.0765, "lon": 35.0983}, {"name": "Nahariya", "lat": 33.0048, "lon": 35.0832},
    {"name": "Acre", "lat": 32.9280, "lon": 35.0680}, {"name": "Haifa North", "lat": 32.8380, "lon": 34.9820},
    {"name": "Atlit", "lat": 32.6892, "lon": 34.9368}, {"name": "Caesarea", "lat": 32.4948, "lon": 34.8912},
    {"name": "Netanya", "lat": 32.3318, "lon": 34.8512}, {"name": "Herzliya", "lat": 32.1648, "lon": 34.7962},
    {"name": "Tel Aviv Center", "lat": 32.0798, "lon": 34.7618}, {"name": "Ashdod", "lat": 31.7848, "lon": 34.6248},
    {"name": "Ashkelon", "lat": 31.6548, "lon": 34.5448}, {"name": "Zikim", "lat": 31.6098, "lon": 34.5198}
]

WATER_BODIES = {
    "🏖️ Mediterranean Coast": {"center": HAIFA_CENTER, "zoom": 8, "bbox": HAIFA_BBOX, "clip_geom": ISRAEL_TERRITORIAL, "points": BEACHES},
    "🌊 Sea of Galilee": {"center": [32.82, 35.59], "zoom": 12, "bbox": KINNERET_BBOX, "clip_geom": KINNERET_BBOX, "points": [{"name": "Tiberias", "lat": 32.794, "lon": 35.534}, {"name": "North Sea of Galilee", "lat": 32.920, "lon": 35.595}]},
    "🧂 Dead Sea": {"center": [31.50, 35.47], "zoom": 11, "bbox": DEAD_SEA_BBOX, "clip_geom": DEAD_SEA_BBOX, "points": [{"name": "Ein Gedi", "lat": 31.462, "lon": 35.388}, {"name": "Ein Bokek", "lat": 31.198, "lon": 35.352}]},
    "🐠 Red Sea": {"center": [29.55, 34.95], "zoom": 13, "bbox": RED_SEA_BBOX, "clip_geom": RED_SEA_BBOX, "points": [{"name": "Gulf of Eilat", "lat": 29.530, "lon": 34.951}]}
}

class OnMapAtmosphereControl(MacroElement):
    """Atmospheric context card in the bottom-left corner of the map."""
    def __init__(self, atm: dict):
        super(OnMapAtmosphereControl, self).__init__()

        ws  = atm.get("wind_speed")
        wd  = atm.get("wind_dir_deg")
        tc  = atm.get("temp_c")
        pr  = atm.get("precip_mm")
        rh  = atm.get("humidity")

        # Wind direction arrow — SVG rotation defines the wind direction
        arrow_rotation = int(wd) if wd is not None else 0
        ws_str  = f"{ws:.1f} m/s" if ws is not None else "—"
        tc_str  = f"{tc:.1f}°C"  if tc is not None else "—"
        pr_str  = f"{pr:.1f} mm" if pr is not None else "—"
        rh_str  = f"{int(rh)}%"  if rh is not None else "—"
        rain_icon = "🌧️" if (pr is not None and pr > 0.5) else "☀️"

        # Beaufort
        bf = 12
        if ws is not None:
            for b, t in enumerate([0.3,1.6,3.4,5.5,8.0,10.8,13.9,17.2,20.8,24.5,28.5,32.7]):
                if ws < t:
                    bf = b
                    break
        bf_color = "#27AE60" if bf < 4 else "#F39C12" if bf < 7 else "#E74C3C"

        html_content = f"""
            <div style="
                background: rgba(255,255,255,0.93);
                border: 1.5px solid #aaa;
                border-radius: 10px;
                padding: 10px 13px;
                font-family: Arial, sans-serif;
                font-size: 13px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.18);
                min-width: 155px;
            ">
                <div style="font-weight:bold; margin-bottom:7px; text-align:center; font-size:12px; color:#555;">🌍 Atmospheric Context</div>

                <!-- Wind direction arrow SVG + intensity -->
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:5px;">
                    <svg width="28" height="28" viewBox="0 0 28 28">
                        <g transform="rotate({arrow_rotation}, 14, 14)">
                            <polygon points="14,2 18,22 14,18 10,22" fill="#2980B9" opacity="0.85"/>
                        </g>
                    </svg>
                    <div>
                        <div style="font-size:13px; font-weight:bold;">{ws_str}</div>
                        <div style="font-size:11px; color:{bf_color};">Beaufort {bf}</div>
                    </div>
                </div>

                <hr style="margin:5px 0; border-color:#eee;">
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span>🌡️</span><span style="font-weight:bold;">{tc_str}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span>{rain_icon}</span><span style="font-weight:bold;">{pr_str}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span>💧</span><span style="font-weight:bold;">{rh_str}</span>
                </div>
            </div>
        """

        # Wrap in Jinja template so Folium registers the control
        self._template = Template("""
            {% macro script(this, kwargs) %}
            var atmControl = L.control({position: 'bottomleft'});
            atmControl.onAdd = function(map) {
                var div = L.DomUtil.create('div', 'atm-control');
                div.innerHTML = `""" + html_content.replace("`", "'") + """`;
                L.DomEvent.disableClickPropagation(div);
                return div;
            };
            atmControl.addTo({{ this._parent.get_name() }});
            {% endmacro %}
        """)


class OnMapWaterLegend(MacroElement):
    def __init__(self):
        super(OnMapWaterLegend, self).__init__()
        self._template = Template("""
            {% macro script(this, kwargs) %}
            var legend = L.control({position: 'topright'});
            legend.onAdd = function (map) {
                var div = L.DomUtil.create('div', 'info legend');
                div.style.background = 'rgba(255, 255, 255, 0.9)';
                div.style.padding = '12px';
                div.style.border = '2px solid #999';
                div.style.borderRadius = '8px';
                div.style.fontFamily = 'Arial, sans-serif';
                div.style.fontSize = '13px';
                div.style.boxShadow = '0 0 15px rgba(0,0,0,0.2)';
                
                div.innerHTML = `
                    <div style="font-weight: bold; margin-bottom: 8px; text-align: center;">Water Quality Index</div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="height: 150px; width: 16px; background: linear-gradient(to bottom, #00FF00, #FFFF00, #FF0000); border: 1px solid #666; border-radius: 3px; flex-shrink: 0;"></div>
                        <div style="display: flex; flex-direction: column; justify-content: space-between; height: 150px; font-size: 11px; font-weight: bold;">
                            <span style="color: green;">Clean</span>
                            <span style="color: orange;">Moderate</span>
                            <span style="color: red;">Polluted</span>
                        </div>
                    </div>
                `;
                return div;
            };
            legend.addTo({{this._parent.get_name()}});
            {% endmacro %}
        """)

# =============================================================================
# 4. Satellite Data Processing and Loading
# =============================================================================
@st.cache_data(ttl=14400)
def get_available_s3_dates(wb_key: str, days_back: int = 30):
    wb = WATER_BODIES[wb_key]
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)
    
    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
            .filterBounds(wb["bbox"])
            .filterDate(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    
    dates_list = coll.aggregate_array("system:time_start").getInfo()
    return sorted(list(set([datetime.utcfromtimestamp(d / 1000).strftime("%Y-%m-%d") for d in dates_list])), reverse=True)

def process_s3_wqi_data(wb_key, target_date_str):
    wb = WATER_BODIES[wb_key]
    t_date = ee.Date(target_date_str)
    start_window = t_date.advance(-1, 'day')
    end_window = t_date.advance(1, 'day')
    
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    water_mask = gsw.select("occurrence").gte(25)
    
    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
            .filterBounds(wb["bbox"])
            .filterDate(start_window, end_window))
            
    if coll.size().getInfo() == 0: 
        return None, None, "No Sentinel-3 scan found for this date."
        
    img = coll.median().clip(wb["clip_geom"]).updateMask(water_mask)
    
    s3_ndwi = img.normalizedDifference(['Oa06_radiance', 'Oa17_radiance']).rename('S3_NDWI')
    b10, b11, b12 = img.select('Oa10_radiance'), img.select('Oa11_radiance'), img.select('Oa12_radiance')
    mci = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75 - 681.25) / (753.75 - 681.25)))).rename('MCI')
    turbidity = img.select('Oa08_radiance').rename('S3_Turb')
    
    ndwi_norm = s3_ndwi.unitScale(-0.2, 0.5).clamp(0, 1)
    mci_norm  = ee.Image(1).subtract(mci.unitScale(-2, 12)).clamp(0, 1)
    turb_norm = ee.Image(1).subtract(turbidity.unitScale(10, 80)).clamp(0, 1)
    
    raw_composite_wqi = ndwi_norm.add(mci_norm).add(turb_norm).divide(3).multiply(100).rename('WQI')
    boxcar = ee.Kernel.square(radius=1, units='pixels')
    composite_wqi = raw_composite_wqi.reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=boxcar).rename('WQI').updateMask(water_mask)
    
    def get_point_wqi(pt_info):
        pt_geom = ee.Geometry.Point([pt_info["lon"], pt_info["lat"]])
        try:
            val = composite_wqi.reduceRegion(reducer=ee.Reducer.mean(), geometry=pt_geom.buffer(450), scale=300, bestEffort=True).getInfo()
            wqi_val = val.get('WQI')
            return {**pt_info, "wqi": round(wqi_val, 1) if wqi_val is not None else None}
        except:
            return {**pt_info, "wqi": None}

    with ThreadPoolExecutor(max_workers=4) as executor:
        sampled_points = list(executor.map(get_point_wqi, wb["points"]))
        
    return composite_wqi, pd.DataFrame(sampled_points), None


# =============================================================================
# 5 (extended). Global Mode — Dynamic Coastal Points by Map Center and Zoom
# =============================================================================

def generate_coastal_points_in_bbox(lat_min, lat_max, lon_min, lon_max, spacing_deg=0.45):
    """
    Generates a grid of points within a bbox to check proximity to coastline.
    spacing_deg ~ 0.45° ≈ 50 km.
    Returns a list of dicts with lat/lon.
    """
    points = []
    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            points.append({"name": f"{lat:.2f},{lon:.2f}", "lat": round(lat, 4), "lon": round(lon, 4)})
            lon += spacing_deg
        lat += spacing_deg
    return points


@st.cache_data(ttl=7200)
def filter_coastal_points_gee(points: list, bbox_rect: tuple) -> list:
    """
    Filters points that are close to the coastline using GSW (Global Surface Water).
    bbox_rect = (lon_min, lat_min, lon_max, lat_max)
    Returns only points where occurrence >= 1 (water) within a 500m buffer.
    """
    if not points:
        return []
    lon_min, lat_min, lon_max, lat_max = bbox_rect
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    coastal_pts = []
    # Check in batches of 20 to reduce GEE calls
    for i in range(0, len(points), 20):
        batch = points[i:i+20]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([p["lon"], p["lat"]]).buffer(2000), {"idx": j})
            for j, p in enumerate(batch)
        ])
        try:
            reduced = gsw.reduceRegions(collection=fc, reducer=ee.Reducer.max(), scale=300)
            feats = reduced.getInfo().get("features", [])
            for feat in feats:
                idx = feat["properties"].get("idx")
                val = feat["properties"].get("max", 0) or 0
                # occurrence >= 5 = close enough to a water body
                if val >= 5 and idx is not None:
                    coastal_pts.append(batch[idx])
        except Exception:
            pass
    return coastal_pts


@st.cache_data(ttl=14400)
def get_global_wqi_layer(target_date_str: str, bbox_rect: tuple):
    """
    Computes a global WQI layer for a given bbox, without clipping to Israel.
    bbox_rect = (lon_min, lat_min, lon_max, lat_max)
    """
    lon_min, lat_min, lon_max, lat_max = bbox_rect
    bbox = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    water_mask = gsw.select("occurrence").gte(25)

    t_date = ee.Date(target_date_str)
    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
            .filterBounds(bbox)
            .filterDate(t_date.advance(-1, 'day'), t_date.advance(1, 'day')))

    if coll.size().getInfo() == 0:
        return None, "No satellite data found for this area on the selected date."

    img = coll.median().clip(bbox).updateMask(water_mask)

    s3_ndwi = img.normalizedDifference(['Oa06_radiance', 'Oa17_radiance']).rename('S3_NDWI')
    b10 = img.select('Oa10_radiance')
    b11 = img.select('Oa11_radiance')
    b12 = img.select('Oa12_radiance')
    mci = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75 - 681.25) / (753.75 - 681.25)))).rename('MCI')
    turbidity = img.select('Oa08_radiance').rename('S3_Turb')

    ndwi_norm = s3_ndwi.unitScale(-0.2, 0.5).clamp(0, 1)
    mci_norm  = ee.Image(1).subtract(mci.unitScale(-2, 12)).clamp(0, 1)
    turb_norm = ee.Image(1).subtract(turbidity.unitScale(10, 80)).clamp(0, 1)

    raw_wqi = ndwi_norm.add(mci_norm).add(turb_norm).divide(3).multiply(100).rename('WQI')
    boxcar  = ee.Kernel.square(radius=1, units='pixels')
    wqi     = raw_wqi.reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=boxcar).rename('WQI').updateMask(water_mask)
    return wqi, None


def get_bbox_from_map(map_data: dict, zoom: int):
    """
    Computes a rough bbox based on the map center and zoom — for global computation.
    """
    if not map_data or not map_data.get("center"):
        return None
    lat = map_data["center"]["lat"]
    lon = map_data["center"]["lng"]
    # degrees per pixel at equator / zoom:  360 / (256 * 2^zoom)
    # viewport ~800x550 px → half-size in degrees:
    deg_per_px = 360.0 / (256 * (2 ** zoom))
    half_w = deg_per_px * 400   # 800/2 px
    half_h = deg_per_px * 275   # 550/2 px
    lon_min = max(-180, lon - half_w)
    lon_max = min(180,  lon + half_w)
    lat_min = max(-85,  lat - half_h)
    lat_max = min(85,   lat + half_h)
    return (lon_min, lat_min, lon_max, lat_max)

# =============================================================================
# 5. User Interface and Clean Map Display
# =============================================================================
st.title("🛰️ AquaWatch Global — Sentinel-3 Water Quality Index Monitor")
st.markdown("Displays the composite Water Quality Index (WQI) clipped to territorial waters, including an atmospheric context module and a clean coastal map.")

st.sidebar.header("🔧 Settings & Dates")
MODE_GLOBAL = "🌍 Global"
_wb_options = [MODE_GLOBAL] + list(WATER_BODIES.keys())
wb_selection = st.sidebar.selectbox("Select water body to monitor:", _wb_options)
is_global = (wb_selection == MODE_GLOBAL)

# =============================================================================
# Session state initialization for last center and atmospheric data management
# =============================================================================
wb_center_default = [20.0, 0.0] if is_global else WATER_BODIES[wb_selection]["center"]

if "atm_center" not in st.session_state or st.session_state.get("last_wb") != wb_selection:
    st.session_state.atm_center = (wb_center_default[0], wb_center_default[1])
    if not is_global:
        st.session_state.atm_data = get_atmospheric_context(wb_selection)
    else:
        st.session_state.atm_data = _empty_atm()
    st.session_state.last_wb = wb_selection
    st.session_state.global_zoom = 3
    st.session_state.global_bbox = None
    st.session_state.global_coastal_pts = []
    st.session_state.global_wqi_layer = None

atm_data = st.session_state.atm_data

# Sidebar — display current atmospheric data

# Add current location message to sidebar

# ---------------------------------------------------------------
# Date selection — shared between both modes
# ---------------------------------------------------------------
if not is_global:
    with st.spinner("Locating available Sentinel-3 pass dates..."):
        available_dates = get_available_s3_dates(wb_selection)
else:
    available_dates = [(datetime.utcnow() - timedelta(days=d)).strftime('%Y-%m-%d') for d in range(1, 8)]

if available_dates:
    formatted_options = [f"🟢 {d}" for d in available_dates]
    date_selection_raw = st.sidebar.selectbox("Select satellite pass date:", formatted_options)
    selected_date_str = date_selection_raw.replace("🟢 ", "")
else:
    selected_date_str = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')

# ---------------------------------------------------------------
# Global branch
# ---------------------------------------------------------------
if is_global:
    col_map, col_info = st.columns([4.0, 1.0])

    with col_map:
        st.subheader("🌍 Global WQI Monitoring Map")

        zoom_init = st.session_state.get("global_zoom", 3)
        center_init = list(st.session_state.atm_center)

        m_global = folium.Map(location=center_init, zoom_start=zoom_init)

        # If a cached global WQI layer exists — add it
        cached_layer = st.session_state.get("global_wqi_layer")
        if cached_layer is not None:
            vis_params = {'min': 40, 'max': 85, 'palette': ['#FF0000', '#FFFF00', '#00FF00']}
            try:
                map_id_dict = ee.Image(cached_layer).getMapId(vis_params)
                folium.TileLayer(
                    tiles=map_id_dict['tile_fetcher'].url_format,
                    attr='Google Earth Engine Sentinel-3',
                    name="WQI Index",
                    overlay=True,
                    control=False,
                    opacity=0.85
                ).add_to(m_global)
            except Exception:
                pass

        # Coastal points — only at zoom >= 7
        current_zoom = st.session_state.get("global_zoom", 3)
        coastal_pts  = st.session_state.get("global_coastal_pts", [])

        if current_zoom >= 7 and coastal_pts:
            for pt in coastal_pts:
                wqi_val = pt.get("wqi")
                if wqi_val is not None:
                    score = wqi_val
                    color_marker = "green" if score > 65 else "orange" if score > 45 else "red"
                else:
                    color_marker = "gray"
                folium.CircleMarker(
                    location=[pt["lat"], pt["lon"]],
                    radius=5,
                    color="black",
                    weight=1,
                    fill_color=color_marker,
                    fill_opacity=0.85,
                    fill=True,
                ).add_to(m_global)

        m_global.add_child(OnMapWaterLegend())

        map_data_global = st_folium(
            m_global,
            width=800,
            height=550,
            key="global_map_v1",
            returned_objects=["center", "zoom"],
        )

        # Update zoom and bbox based on map movement
        if map_data_global:
            new_zoom = map_data_global.get("zoom") or zoom_init
            new_center = map_data_global.get("center")

            if new_center:
                new_lat = new_center["lat"]
                new_lon = new_center["lng"]
                prev_lat, prev_lon = st.session_state.atm_center
                dist_km = haversine_km(prev_lat, prev_lon, new_lat, new_lon)
                zoom_changed = (new_zoom != st.session_state.get("global_zoom", 3))
                moved_far    = (dist_km > 200)

                if zoom_changed or moved_far:
                    st.session_state.global_zoom   = new_zoom
                    st.session_state.atm_center    = (new_lat, new_lon)
                    new_bbox = get_bbox_from_map(map_data_global, new_zoom)
                    st.session_state.global_bbox   = new_bbox

                    # Compute WQI layer for the new area
                    if new_bbox:
                        with st.spinner("Computing WQI for area..."):
                            g_layer, g_err = get_global_wqi_layer(selected_date_str, new_bbox)
                            st.session_state.global_wqi_layer = g_layer if not g_err else None

                        # Coastal points only at high zoom
                        if new_zoom >= 7:
                            lon_min, lat_min, lon_max, lat_max = new_bbox
                            candidate_pts = generate_coastal_points_in_bbox(lat_min, lat_max, lon_min, lon_max)
                            with st.spinner("Locating coastal points at high zoom..."):
                                coast_pts = filter_coastal_points_gee(candidate_pts, new_bbox)

                            # WQI for each coastal point
                            if g_layer and coast_pts:
                                def _sample_pt(pt):
                                    try:
                                        geom = ee.Geometry.Point([pt["lon"], pt["lat"]])
                                        val  = g_layer.reduceRegion(
                                            reducer=ee.Reducer.mean(),
                                            geometry=geom.buffer(2000),
                                            scale=300,
                                            bestEffort=True
                                        ).getInfo()
                                        wqi_v = val.get("WQI")
                                        return {**pt, "wqi": round(wqi_v, 1) if wqi_v is not None else None}
                                    except Exception:
                                        return {**pt, "wqi": None}
                                with ThreadPoolExecutor(max_workers=6) as ex:
                                    coast_pts = list(ex.map(_sample_pt, coast_pts))

                            st.session_state.global_coastal_pts = coast_pts
                        else:
                            st.session_state.global_coastal_pts = []

                    st.rerun()

    with col_info:
        st.subheader("🏖️ Coastal Water Cleanliness Index")
        current_zoom = st.session_state.get("global_zoom", 3)
        if current_zoom < 7:
            st.info("🔍 Zoom in on a coastal area to see measurement points (~50 km)")
        else:
            pts = st.session_state.get("global_coastal_pts", [])
            if pts:
                def _status_g(score):
                    try:
                        v = float(score)
                    except (ValueError, TypeError):
                        return "❓ No Data"
                    if v >= 70: return "🟢 Clean"
                    if v >= 55: return "🟡 Moderate"
                    return "🔴 Polluted"
                df_g = pd.DataFrame(pts)[["lat", "lon", "wqi"]].copy()
                df_g["Status"] = df_g["wqi"].apply(_status_g)
                df_g = df_g.rename(columns={"lat": "Latitude", "lon": "Longitude"})
                df_g = df_g[["Latitude", "Longitude", "Status"]]
                st.dataframe(df_g, use_container_width=True, hide_index=True)
            else:
                st.write("No coastal points found in the current area.")

# ---------------------------------------------------------------
# Israel / defined water bodies branch
# ---------------------------------------------------------------
else:
    with st.spinner("Computing composite values and generating coastal data..."):
        wqi_layer, df_beaches, error_msg = process_s3_wqi_data(wb_selection, selected_date_str)

    if error_msg:
        st.error(error_msg)
    elif wqi_layer:
        df_beaches = blend_atmospheric_penalty(df_beaches, atm_data, wb_selection)

        col_map, col_info = st.columns([4.0, 1.0])

        with col_map:
            st.subheader(f"📍 Composite WQI Map: {wb_selection}")
            m = folium.Map(location=wb_center_default, zoom_start=WATER_BODIES[wb_selection]["zoom"])

            vis_params = {'min': 40, 'max': 85, 'palette': ['#FF0000', '#FFFF00', '#00FF00']}
            map_id_dict = ee.Image(wqi_layer).getMapId(vis_params)
            folium.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine Sentinel-3',
                name="WQI Index",
                overlay=True,
                control=False,
                opacity=0.85
            ).add_to(m)

            for _, r in df_beaches.iterrows():
                score_for_color = r['composite_with_atm'] if pd.notna(r['composite_with_atm']) else r['wqi']
                color_marker = "green" if (score_for_color and score_for_color > 65) else "orange" if score_for_color else "red"
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=6,
                    color="black",
                    weight=1,
                    fill_color=color_marker,
                    fill_opacity=0.9,
                    fill=True
                ).add_to(m)

            m.add_child(OnMapAtmosphereControl(atm_data))
            m.add_child(OnMapWaterLegend())

            map_data = st_folium(
                m,
                width=800,
                height=550,
                key="s3_map_v7",
                returned_objects=["center"],
            )

            if map_data and map_data.get("center"):
                new_lat = map_data["center"]["lat"]
                new_lon = map_data["center"]["lng"]
                prev_lat, prev_lon = st.session_state.atm_center

                dist_km = haversine_km(prev_lat, prev_lon, new_lat, new_lon)

                if dist_km > 50:
                    st.session_state.atm_data   = get_atmospheric_context_by_coords(new_lat, new_lon)
                    st.session_state.atm_center = (new_lat, new_lon)
                    st.rerun()

        with col_info:
            st.subheader("🏖️ Coastal Water Cleanliness Index")

            if df_beaches is not None and not df_beaches.empty:
                df_display = df_beaches[["name", "wqi", "composite_with_atm"]].copy()
                df_display.columns = ["Station Name", "Raw Satellite WQI", "_score"]
                df_display["Raw Satellite WQI"] = df_display["Raw Satellite WQI"].fillna("No Data")

                def _status(score):
                    try:
                        v = float(score)
                    except (ValueError, TypeError):
                        return "❓ No Data"
                    if v >= 70:  return "🟢 Clean"
                    if v >= 55:  return "🟡 Moderate"
                    return "🔴 Polluted"

                df_display["Status & Cleanliness"] = df_display["_score"].apply(_status)
                df_display = df_display[["Station Name", "Status & Cleanliness"]]
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                st.write("No defined stations found for this area.")
