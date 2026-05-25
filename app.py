"""
app.py
=============================================================================
Israel & Global Water Quality Monitor Dashboard
כולל החלפת שכבות ראסטר לוויניות (מקומי וגלובלי), התאמה אטמוספרית,
ועיבוד מקבילי מהיר ב-Google Earth Engine.
=============================================================================
"""

import math
import json
import os
import tempfile
import requests
import pandas as pd
import streamlit as st
import folium
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from streamlit_folium import st_folium
import ee

# ==============================================================================
# SEO Optimization & Analytics Injection
# ==============================================================================
st.markdown(
    '<meta name="google-site-verification" content="להדביק_כאן_את_הקוד_מגוגל" />', 
    unsafe_allow_html=True
)

# Umami analytics
components.html(
    '<script async src="https://cloud.umami.is/script.js" data-website-id="07a48db1-5aa7-4d88-aaac-9cfb6fc2600d"></script>',
    height=0
)

# Google Analytics
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
# Google Earth Engine (GEE) Authentication
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
# 1. Atmospheric Integration Layer (Open-Meteo Integration)
# =============================================================================
_WB_CENTRES = {
    "🏖️ חוף הים התיכון": (32.40, 34.85),
    "🌊 כנרת":            (32.82, 35.59),
    "🧂 ים המלח":         (31.50, 35.47),
    "🐠 ים סוף":          (29.55, 34.95),
}

@st.cache_data(ttl=3600)
def get_atmospheric_context(wb_key: str) -> dict:
    empty = _empty_atm()
    lat, lon = _WB_CENTRES.get(wb_key, (32.0, 35.0))
    try:
        resp = requests.get(
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
        empty["_error"] = f"שגיאה: {exc}"
        return empty

def _empty_atm() -> dict:
    return {
        "wind_speed": None, "wind_dir_deg": None, "temp_c": None,
        "humidity": None, "precip_mm": None, "weather_code": None,
        "analysis_time": None, "centre_lat": None, "centre_lon": None,
        "_error": None, "_source": "Open-Meteo (GFS/ERA5)",
    }

def blend_atmospheric_penalty(df, atm: dict, wb_key: str):
    def _penalty(row) -> float:
        if row.get("composite") is None:
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

        closed = {"🌊 כנרת", "🧂 ים המלח", "🐠 ים סוף"}
        if wb_key in closed and rh is not None:
            if rh > 85:    p += 7.0
            elif rh > 70:  p += 3.0

        return min(p, 25.0)

    df = df.copy()
    df["atm_penalty"] = df.apply(_penalty, axis=1)
    df["composite_with_atm"] = df.apply(
        lambda r: (
            round(max(0.0, r["composite"] - r["atm_penalty"]), 1)
            if r["composite"] is not None else None
        ),
        axis=1,
    )
    return df

def render_earth2_sidebar(atm: dict, wb_key: str) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "### 🌍 הקשר אטמוספרי\n"
        "<small style='color:#888;'>Open-Meteo · GFS/ERA5</small>",
        unsafe_allow_html=True,
    )
    if atm.get("_error"):
        st.sidebar.warning(atm["_error"])
        return

    ws, wd, tc, pr, rh, ts = atm.get("wind_speed"), atm.get("wind_dir_deg"), atm.get("temp_c"), atm.get("precip_mm"), atm.get("humidity"), atm.get("analysis_time", "—")

    if ws is not None:
        arrow = ["↑","↗","→","↘","↓","↙","←","↖"][int((wd + 22.5) / 45) % 8] if wd is not None else ""
        bf = 12
        for b, t in enumerate([0.3,1.6,3.4,5.5,8.0,10.8,13.9,17.2,20.8,24.5,28.5,32.7]):
            if ws < t: bf = b; break
        st.sidebar.metric(label=f"💨 רוח {arrow}", value=f"{ws:.1f} m/s", delta=f"Beaufort {bf}", delta_color="inverse" if bf >= 5 else "normal")
    if tc is not None: st.sidebar.metric("🌡️ טמפרטורה", f"{tc:.1f} °C")
    if pr is not None: st.sidebar.metric("🌧️ גשם" if pr > 0.5 else "☀️ יבש", f"{pr:.1f} mm/h")
    if rh is not None: st.sidebar.metric("💧 לחות", f"{int(rh)}%")

    score, reasons = 0, []
    if ws and ws > 7: score += 1; reasons.append("רוח חזקה")
    if pr and pr > 0.5: score += 1; reasons.append("גשם")
    if score == 0:
        rl, rc, rt = "✅ סיכון אטמוספרי נמוך", "#27AE60", "תנאים אטמוספריים תומכים באיכות מים טובה."
    else:
        rl, rc, rt = "🟡 סיכון בינוני", "#F1C40F", " · ".join(reasons) + " — ייתכן עיוות קל בנתוני הלוויין."
        
    st.sidebar.markdown(f"""<div style="background:#f8f9fa;border-radius:10px;padding:10px 14px;border-right:4px solid {rc};direction:rtl;font-family:Arial;margin-top:6px;"><b style="color:{rc};">{rl}</b><br><span style="font-size:12px;color:#555;">{rt}</span></div>""", unsafe_allow_html=True)


# =============================================================================
# 2. Global Visitor Geolocation Helper (IP-Based Lookup)
# =============================================================================
def get_visitor_geolocation():
    default_lat, default_lon = 32.40, 34.85
    try:
        response = requests.get("https://ipapi.co/json/", timeout=3)
        if response.status_code == 200:
            data = response.json()
            country_code = data.get("country_code", "IL")
            if country_code != "IL":
                lat = data.get("latitude", default_lat)
                lon = data.get("longitude", default_lon)
                city = data.get("city", "Unknown Proximity")
                country = data.get("country_name", "International")
                return lat, lon, f"{city}, {country}"
    except Exception:
        pass
    return default_lat, default_lon, "ישראל (נקודת ברירת מחדל חוף הים התיכון)"


# =============================================================================
# 3. Global Remote Sensing Engine (Fast-Overhead 100x100km Raster Matrix)
# =============================================================================
def generate_global_raster_thumb(lat, lon, feature_type="Composite Score"):
    point = ee.Geometry.Point([lon, lat])
    bbox = point.buffer(50000).bounds()  # 100x100km Bound Box
    
    now = ee.Date(datetime.utcnow().strftime('%Y-%m-%d'))
    start_date = now.advance(-30, 'day')
    
    s2_collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                     .filterBounds(bbox)
                     .filterDate(start_date, now)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25)))
    
    if s2_collection.size().getInfo() == 0:
        return None, "לא נמצאו סריקות לוויין נקיות מעננים עבור אזור זה ב-30 הימים האחרונים."
        
    s2_image = s2_collection.median().clip(bbox)
    
    ndwi = s2_image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    chl = s2_image.select('B5').divide(s2_image.select('B4')).rename('Chl_proxy')
    turbidity = s2_image.select('B4').rename('Turbidity')
    water_mask = ndwi.gt(0.0)
    
    if "Chlorophyll" in feature_type:
        target_raster = chl.updateMask(water_mask)
        vis_params = {'min': 1.0, 'max': 2.0, 'palette': ['#0000FF', '#FFFF00', '#00FF00']}
    elif "Turbidity" in feature_type:
        target_raster = turbidity.updateMask(water_mask)
        vis_params = {'min': 100, 'max': 1500, 'palette': ['#0000FF', '#00FFFF', '#8B4513']}
    elif "True Color" in feature_type:
        target_raster = s2_image.select(['B4', 'B3', 'B2'])
        vis_params = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}
    else:  # Composite Score
        ndwi_norm = ndwi.unitScale(-0.1, 0.6).clamp(0, 1)
        chl_norm = ee.Image(1).subtract(chl.unitScale(0.9, 1.9)).clamp(0, 1)
        turb_norm = ee.Image(1).subtract(turbidity.unitScale(100, 2000)).clamp(0, 1)
        composite = ndwi_norm.add(chl_norm).add(turb_norm).divide(3).multiply(100)
        target_raster = composite.updateMask(water_mask)
        vis_params = {'min': 35, 'max': 85, 'palette': ['#FF0000', '#FFFF00', '#00FF00']}
        
    try:
        thumb_url = target_raster.getThumbnailURL({
            'params': vis_params,
            'dimensions': 750,
            'format': 'png'
        })
        return thumb_url, None
    except Exception as e:
        return None, f"שגיאה בעיבוד הציור הלווייני: {str(e)}"


# =============================================================================
# 4. Core Local Monitoring Parameters (Israel)
# =============================================================================
HAIFA_CENTER = [32.4, 34.85]
HAIFA_BBOX   = ee.Geometry.Rectangle([34.50, 31.55, 35.15, 33.10])

ISRAEL_TERRITORIAL = ee.Geometry.Polygon([[
    [34.95, 33.10], [34.60, 33.10], [34.20, 32.60], [34.15, 32.00],
    [34.20, 31.55], [34.55, 31.30], [34.75, 31.25], [34.95, 31.30],
    [35.00, 31.55], [35.00, 32.00], [35.10, 32.60], [35.10, 33.10],
    [34.95, 33.10]
]])

BEACHES = [
    {"name": "ראש הנקרה",   "lat": 33.0765, "lon": 35.0983},
    {"name": "נהריה",        "lat": 33.0048, "lon": 35.0832},
    {"name": "עכו",          "lat": 32.9280, "lon": 35.0680},
    {"name": "חיפה צפון",    "lat": 32.8380, "lon": 34.9820},
    {"name": "חיפה מרכז",    "lat": 32.8148, "lon": 34.9648},
    {"name": "עתלית",        "lat": 32.6892, "lon": 34.9368},
    {"name": "קיסריה",       "lat": 32.4948, "lon": 34.8912},
    {"name": "נתניה",        "lat": 32.3318, "lon": 34.8512},
    {"name": "הרצליה",       "lat": 32.1648, "lon": 34.7962},
    {"name": "תל אביב מרכז", "lat": 32.0798, "lon": 34.7618},
    {"name": "בת ים",        "lat": 32.0148, "lon": 34.7448},
    {"name": "אשדוד דרום",   "lat": 31.7848, "lon": 34.6248},
    {"name": "אשקלון",       "lat": 31.6548, "lon": 34.5448},
    {"name": "זיקים",        "lat": 31.6098, "lon": 34.5198},
]

KINNERET_CENTER = [32.82, 35.59]
KINNERET_BBOX   = ee.Geometry.Rectangle([35.50, 32.70, 35.68, 32.95])
KINNERET_POINTS = [{"name": "טבריה", "lat": 32.794, "lon": 35.534}, {"name": "צפון הכנרת", "lat": 32.920, "lon": 35.595}]

DEAD_SEA_CENTER = [31.50, 35.47]
DEAD_SEA_BBOX   = ee.Geometry.Rectangle([35.35, 31.20, 35.60, 31.80])
DEAD_SEA_POINTS = [{"name": "עין גדי", "lat": 31.462, "lon": 35.388}, {"name": "עין בוקק", "lat": 31.198, "lon": 35.352}]

RED_SEA_CENTER = [29.55, 34.95]
RED_SEA_BBOX   = ee.Geometry.Rectangle([34.80, 29.35, 35.10, 29.75])
RED_SEA_POINTS = [{"name": "אילת צפון", "lat": 29.558, "lon": 34.952}, {"name": "מפרץ עקבה", "lat": 29.430, "lon": 34.930}]

WATER_BODIES = {
    "🏖️ חוף הים התיכון": {"center": HAIFA_CENTER, "zoom": 8, "bbox": HAIFA_BBOX, "clip_geom": ISRAEL_TERRITORIAL, "points": BEACHES, "sensor": "S2", "cloud_pct": 20, "days_back": 90},
    "🌊 כנרת": {"center": KINNERET_CENTER, "zoom": 12, "bbox": KINNERET_BBOX, "clip_geom": KINNERET_BBOX, "points": KINNERET_POINTS, "sensor": "S2", "cloud_pct": 10, "days_back": 60},
    "🧂 ים המלח": {"center": DEAD_SEA_CENTER, "zoom": 11, "bbox": DEAD_SEA_BBOX, "clip_geom": DEAD_SEA_BBOX, "points": DEAD_SEA_POINTS, "sensor": "S2", "cloud_pct": 20, "days_back": 90},
    "🐠 ים סוף": {"center": RED_SEA_CENTER, "zoom": 12, "bbox": RED_SEA_BBOX, "clip_geom": RED_SEA_BBOX, "points": RED_SEA_POINTS, "sensor": "S2", "cloud_pct": 10, "days_back": 120}
}

@st.cache_data(ttl=604800)
def snap_points_to_coastline(points_key: str, points_list: list, search_radius: int = 3000) -> list:
    jsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    permanent_water = jsw.select("occurrence").gte(50)
    def snap_one(point):
        try:
            pt = ee.Geometry.Point([point["lon"], point["lat"]])
            water_pts = permanent_water.updateMask(permanent_water).sample(region=pt.buffer(search_radius), scale=30, geometries=True, numPixels=100)
            if water_pts.size().getInfo() == 0: return point
            nearest = water_pts.map(lambda f: f.set("dist", f.geometry().distance(pt, 1))).sort("dist").first().geometry().centroid(1).coordinates().getInfo()
            return {**point, "lat": nearest[1], "lon": nearest[0]}
        except: return point
    with ThreadPoolExecutor(max_workers=8) as executor:
        ordered = list(executor.map(snap_one, points_list))
    return ordered

@st.cache_data(ttl=3600)
def load_data(wb_key: str, start_date: str, end_date: str):
    wb = WATER_BODIES[wb_key]
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(wb["bbox"])
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", wb["cloud_pct"])))
    first_info = collection.sort("system:time_start", False).limit(1).getInfo()
    if not first_info["features"]: return None, None, "אין סריקות", 0, []
    
    image_date = datetime.utcfromtimestamp(first_info["features"][0]["properties"]["system:time_start"] / 1000).strftime("%Y-%m-%d")
    
    def compute_indices_s2(image):
        ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
        chl_proxy = image.select("B5").divide(image.select("B4")).rename("Chl_proxy")
        turbidity = image.select("B4").rename("Turbidity")
        return image.addBands([ndwi, chl_proxy, turbidity])

    processed = collection.map(compute_indices_s2).median().clip(wb["clip_geom"])
    snapped_points = snap_points_to_coastline(wb_key, wb["points"])
    
    def get_point_values(point):
        pt = ee.Geometry.Point([point["lon"], point["lat"]])
        try:
            vals = processed.select(["NDWI", "Chl_proxy", "Turbidity"]).reduceRegion(reducer=ee.Reducer.mean(), geometry=pt.buffer(1000), scale=10, bestEffort=True).getInfo()
            return {**point, "ndwi": vals.get("NDWI"), "chl_proxy": vals.get("Chl_proxy"), "turbidity": vals.get("Turbidity"), "no_data": False}
        except:
            return {**point, "ndwi": None, "chl_proxy": None, "turbidity": None, "no_data": True}

    with ThreadPoolExecutor(max_workers=8) as executor:
        data = list(executor.map(get_point_values, snapped_points))
        
    df = pd.DataFrame(data)
    df["composite"] = df.apply(lambda r: round(((r["ndwi"] or 0)*40 + (2.0 - (r["chl_proxy"] or 1.5))*35), 1) if not r["no_data"] else None, axis=1)
    return processed, df, image_date, len(first_info["features"]), []


# =============================================================================
# 5. Interface Layout & Router Generation
# =============================================================================
st.set_page_config(page_title="Israel & Global Water Quality Monitor", layout="wide")

st.title("🛰️ מערכת לווינית לניטור מדדי איכות מים")
st.markdown("פיתוח מחקרי מתקדם המבוסס על שילוב מנועי תצפיות ומודלים הידרו-אופטיים בזמן אמת.")

tab_local, tab_global = st.tabs(["🇮🇱 ניטור אזורי (ישראל)", "🌐 ניטור גלובלי (Global)"])

with tab_local:
    # -------------------------------------------------------------------------
    # Local Application Layout (Israel View Space)
    # -------------------------------------------------------------------------
    st.sidebar.header("🔧 הגדרות מערכת")
    wb_selection = st.sidebar.selectbox("בחר גוף מים לניטור:", list(WATER_BODIES.keys()))
    
    # תיבת בחירת שכבות הלוויין שהוחזרה למשתמש
    local_product = st.sidebar.selectbox(
        "בחר שכבת מפה להצגה בישראל:",
        ["Composite Score (מדד משולב)", "Chlorophyll (כלורופיל)", "Turbidity (עכירות)", "True Color RGB (צבע אמיתי)"],
        key="local_product_selector"
    )
    
    selected_date = st.sidebar.date_input("תאריך מטרה לניתוח:", datetime.utcnow() - timedelta(days=2))
    date_str = selected_date.strftime('%Y-%m-%d')
    start_date_str = (selected_date - timedelta(days=WATER_BODIES[wb_selection]["days_back"])).strftime('%Y-%m-%d')
    
    atm_data = get_atmospheric_context(wb_selection)
    render_earth2_sidebar(atm_data, wb_selection)
    
    with st.spinner("מריץ חישובים ומשיך שכבות מ-Earth Engine..."):
        processed_layer, df_points, img_date, img_count, _ = load_data(wb_selection, start_date_str, date_str)
        
    if df_points is not None and not df_points.empty:
        df_points = blend_atmospheric_penalty(df_points, atm_data, wb_selection)
        
        col_m, col_t = st.columns([2, 1])
        with col_m:
            st.subheader(f"מפת איכות מים: {wb_selection} — {local_product}")
            st.caption(f"תאריך קליטה עדכני: {img_date} | סך הכל תמונות שנמצאו במערך: {img_count}")
            
            # יצירת מפת Folium
            m = folium.Map(location=WATER_BODIES[wb_selection]["center"], zoom_start=WATER_BODIES[wb_selection]["zoom"])
            
            # הגדרת הפרמטרים הוויזואליים של הראסטר המקומי בהתאם לבחירה
            ndwi = processed_layer.select('NDWI')
            water_mask = ndwi.gt(0.0)
            
            if "Chlorophyll" in local_product:
                layer_to_show = processed_layer.select('Chl_proxy').updateMask(water_mask)
                vis = {'min': 1.0, 'max': 2.0, 'palette': ['#0000FF', '#FFFF00', '#00FF00']}
            elif "Turbidity" in local_product:
                layer_to_show = processed_layer.select('Turbidity').updateMask(water_mask)
                vis = {'min': 100, 'max': 1500, 'palette': ['#0000FF', '#00FFFF', '#8B4513']}
            elif "True Color" in local_product:
                layer_to_show = processed_layer.select(['B4', 'B3', 'B2'])
                vis = {'min': 0, 'max': 3000}
            else:  # Composite Score
                ndwi_norm = ndwi.unitScale(-0.1, 0.6).clamp(0, 1)
                chl_norm = ee.Image(1).subtract(processed_layer.select('Chl_proxy').unitScale(0.9, 1.9)).clamp(0, 1)
                turb_norm = ee.Image(1).subtract(processed_layer.select('Turbidity').unitScale(100, 2000)).clamp(0, 1)
                layer_to_show = ndwi_norm.add(chl_norm).add(turb_norm).divide(3).multiply(100).updateMask(water_mask)
                vis = {'min': 35, 'max': 85, 'palette': ['#FF0000', '#FFFF00', '#00FF00']}
            
            # הזרקת שכבת הראסטר מתוך Google Earth Engine ישירות לתוך Folium
            map_id_dict = ee.Image(layer_to_show).getMapId(vis)
            folium.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=local_product,
                overlay=True,
                control=True,
                opacity=0.75
            ).add_to(m)
            
            # הוספת נקודות דיגום קבועות על גבי המפה
            for _, r in df_points.iterrows():
                if not r["no_data"]:
                    folium.CircleMarker(
                        location=[r["lat"], r["lon"]],
                        radius=6,
                        popup=f"{r['name']}: WQI {r['composite_with_atm']}",
                        color="black",
                        fill_color="green" if (r["composite_with_atm"] or 0) > 55 else "red",
                        fill_opacity=0.9,
                        fill=True
                    ).add_to(m)
            
            st_folium(m, width=750, height=500, key="local_map_viewport")
            
        with col_t:
            st.subheader("📊 נתוני תחנות מדידה")
            st.dataframe(df_points[["name", "composite", "atm_penalty", "composite_with_atm"]].rename(columns={
                "name": "תחנה", "composite": "ציון לווין גולמי", "atm_penalty": "הפחתת מזג אוויר", "composite_with_atm": "ציון משולב סופי"
            }))
    else:
        st.warning("לא נמצאו נתוני לוויין זמינים לטווח התאריכים שנבחר גוף מים זה.")

with tab_global:
    # -------------------------------------------------------------------------
    # Global Application Layout (Automated IP Client Localization Routing)
    # -------------------------------------------------------------------------
    st.subheader("🌐 ניטור מדדים גלובלי — מבוסס מיקום אוטומטי")
    st.markdown("שונית זו מזהה את מיקום השרת/המשתמש בחו\"ל ומציגה מפת ערכים ברזולוציה גבוהה התחומה בדיוק בטווח של **100×100 ק\"מ**.")
    
    if 'global_lat' not in st.session_state:
        with st.spinner("מזהה כתובת פרוקסי ומיקום גיאוגרפי במערכת..."):
            d_lat, d_lon, loc_name = get_visitor_geolocation()
            st.session_state.global_lat = d_lat
            st.session_state.global_lon = d_lon
            st.session_state.global_loc_name = loc_name
            
    st.success(f"📍 **נמצא מיקום רשת קרוב:** {st.session_state.global_loc_name} ({st.session_state.global_lat:.4f}, {st.session_state.global_lon:.4f})")
    
    c_lat, c_lon = st.columns(2)
    target_lat = c_lat.number_input("שינוי קו רוחב מטרה (Latitude)", value=st.session_state.global_lat, format="%.4f")
    target_lon = c_lon.number_input("שינוי קו אורך מטרה (Longitude)", value=st.session_state.global_lon, format="%.4f")
    
    global_product = st.selectbox(
        "בחר שכבת אינדקס לוויינית להצגה גלובלית:",
        ["Composite Score (מדד איכות מים משולב)", "Chlorophyll (כלורופיל)", "Turbidity (עכירות)", "True Color RGB (צבע אמיתי)"],
        key="product_layer_dropdown"
    )
    
    with st.spinner("מפיק שכבת ראסטר רנדר מהיר מ-Google Earth Engine..."):
        raster_link, err = generate_global_raster_thumb(target_lat, target_lon, global_product)
        
        if err:
            st.error(err)
        elif raster_link:
            st.markdown("---")
            st.image(raster_link, caption=f"תמונת ראסטר חציונית 30 יום עבור: {global_product} (גבולות חסומים קשיח: 100x100 ק\"מ)", use_container_width=True)
            
            if "Composite" in global_product:
                st.markdown("<div style='text-align: center; direction: rtl;'>🔴 <b>איכות נמוכה (אנומליה)</b> ─── 🟡 <b>בינוני</b> ─── 🟢 <b>איכות מים מעולה</b></div>", unsafe_allow_html=True)
            elif "Chlorophyll" in global_product:
                st.markdown("<div style='text-align: center; direction: rtl;'>🔵 <b>מים צלולים</b> ─── 🟡 <b>עקבות אצות קלות</b> ─── 🟢 <b>ריכוז פריחה גבוה (Algae Bloom)</b></div>", unsafe_allow_html=True)
            elif "Turbidity" in global_product:
                st.markdown("<div style='text-align: center; direction: rtl;'>🔵 <b>מים צלולים ונקיים</b> ─── 🌐 <b>סחף חלקיקים קל</b> ─── 🟤 <b>עכירות גבוהה / נגר חופשי חזק</b></div>", unsafe_allow_html=True)
