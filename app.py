st.sidebar.info("🔍 earth2_integration loaded")
"""
earth2_integration.py
=====================
Atmospheric context layer for the Israel Water Quality Monitor.
Uses Open-Meteo API (free, no API key, no extra dependencies).
Same interface as the original earth2studio version.
"""

from __future__ import annotations
import math
from typing import Optional
import streamlit as st

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

# ── water-body centre coordinates ────────────────────────────────────────────
_WB_CENTRES = {
    "🏖️ חוף הים התיכון": (32.40, 34.85),
    "🌊 כנרת":            (32.82, 35.59),
    "🧂 ים המלח":         (31.50, 35.47),
    "🐠 ים סוף":          (29.55, 34.95),
}


# =============================================================================
# 1.  Fetch atmospheric data via Open-Meteo
# =============================================================================

@st.cache_data(ttl=3600)
def get_atmospheric_context(wb_key: str) -> dict:
    """
    Fetch current atmospheric conditions for *wb_key* from Open-Meteo.
    Returns a flat dict with scalar values. Degrades gracefully on error.
    """
    empty = _empty_atm()

    if not _REQUESTS_OK:
        empty["_error"] = "requests לא זמין"
        return empty

    lat, lon = _WB_CENTRES.get(wb_key, (32.0, 35.0))

    try:
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
        empty["_error"] = f"שגיאה: {exc}"
        return empty


def _empty_atm() -> dict:
    return {
        "wind_speed":    None,
        "wind_dir_deg":  None,
        "temp_c":        None,
        "humidity":      None,
        "precip_mm":     None,
        "weather_code":  None,
        "analysis_time": None,
        "centre_lat":    None,
        "centre_lon":    None,
        "_error":        None,
        "_source":       "Open-Meteo (GFS/ERA5)",
    }


# =============================================================================
# 2.  Blend atmospheric penalty into the composite score
# =============================================================================

def blend_atmospheric_penalty(df, atm: dict, wb_key: str):
    """
    Adds composite_with_atm column = composite − atmospheric penalty (0–25 pts).
    """
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


# =============================================================================
# 3.  Streamlit sidebar panel
# =============================================================================

def render_earth2_sidebar(atm: dict, wb_key: str) -> None:
    """Render atmospheric context card in the Streamlit sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "### 🌍 הקשר אטמוספרי\n"
        "<small style='color:#888;'>Open-Meteo · GFS/ERA5 · עדכון שעתי</small>",
        unsafe_allow_html=True,
    )

    if atm.get("_error"):
        st.sidebar.warning(atm["_error"])
        return

    ws  = atm.get("wind_speed")
    wd  = atm.get("wind_dir_deg")
    tc  = atm.get("temp_c")
    pr  = atm.get("precip_mm")
    rh  = atm.get("humidity")
    ts  = atm.get("analysis_time", "—")

    # Wind
    if ws is not None:
        arrow = _compass_arrow(wd) if wd is not None else ""
        bf    = _beaufort(ws)
        st.sidebar.metric(
            label=f"💨 רוח {arrow}",
            value=f"{ws:.1f} m/s",
            delta=f"Beaufort {bf}",
            delta_color="inverse" if bf >= 5 else "normal",
        )

    # Temperature
    if tc is not None:
        st.sidebar.metric("🌡️ טמפרטורה", f"{tc:.1f} °C")

    # Precipitation
    if pr is not None:
        label = "🌧️ גשם" if pr > 0.5 else "☀️ יבש"
        st.sidebar.metric(label, f"{pr:.1f} mm/h")

    # Humidity
    if rh is not None:
        st.sidebar.metric("💧 לחות", f"{int(rh)}%")

    # Risk badge
    risk_level, risk_color, risk_text = _atmospheric_risk(ws, pr, rh, wb_key)
    st.sidebar.markdown(
        f"""<div style="background:#f8f9fa;border-radius:10px;padding:10px 14px;
                        border-right:4px solid {risk_color};direction:rtl;
                        font-family:Arial;margin-top:6px;">
            <b style="color:{risk_color};">{risk_level}</b><br>
            <span style="font-size:12px;color:#555;">{risk_text}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    st.sidebar.caption(f"🕐 {ts} · {atm.get('_source','')}")


# =============================================================================
# 4.  Helpers
# =============================================================================

def _beaufort(ws: float) -> int:
    for b, t in enumerate([0.3,1.6,3.4,5.5,8.0,10.8,13.9,17.2,20.8,24.5,28.5,32.7]):
        if ws < t:
            return b
    return 12


def _compass_arrow(deg: float) -> str:
    arrows = ["↑","↗","→","↘","↓","↙","←","↖"]
    return arrows[int((deg + 22.5) / 45) % 8]


def _atmospheric_risk(ws, pr, rh, wb_key) -> tuple[str, str, str]:
    score   = 0
    reasons = []

    if ws is not None:
        if ws > 15:   score += 3; reasons.append("רוח חזקה מאוד")
        elif ws > 10: score += 2; reasons.append("רוח חזקה")
        elif ws > 7:  score += 1; reasons.append("רוח בינונית")

    if pr is not None and pr > 0.5:
        if pr > 5:    score += 3; reasons.append("גשם כבד")
        elif pr > 2:  score += 2; reasons.append("גשם בינוני")
        else:         score += 1; reasons.append("גשם קל")

    if wb_key in {"🌊 כנרת","🧂 ים המלח","🐠 ים סוף"} and rh and rh > 80:
        score += 1; reasons.append("לחות גבוהה")

    if score == 0:
        return ("✅ סיכון אטמוספרי נמוך", "#27AE60",
                "תנאים אטמוספריים תומכים באיכות מים טובה.")
    elif score <= 2:
        return ("🟡 סיכון בינוני", "#F1C40F", " · ".join(reasons))
    elif score <= 4:
        return ("🟠 סיכון גבוה", "#E67E22",
                " · ".join(reasons) + " — ציון הלוויין עלול להיות מוטה.")
    else:
        return ("🔴 סיכון קריטי", "#E74C3C",
                " · ".join(reasons) + " — מומלץ לא לסמוך על הלוויין בלבד.")


import ee
import json
import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import pandas as pd
import streamlit.components.v1 as components
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# אימות Google Search Console (להחליף את ה-content בקוד שקיבלת מגוגל)
# ==============================================================================
seo_html = """
<meta name="description" content="מערכת מדעית לניטור איכות המים בישראל בזמן אמת (ים תיכון, כנרת, ים המלח וים סוף) באמצעות חישה מרחוק, נתוני הלוויין Sentinel-2 ו-Google Earth Engine." />
<meta name="keywords" content="איכות מים, חישה מרחוק, לוויין, Sentinel-2, GEE, כנרת, ים המלח, עכירות, כלורופיל, אצות, Water Quality Israel, Remote Sensing, Google Earth Engine" />

<meta property="og:title" content="ניטור איכות מים לוויני — ישראל (Sentinel-2 & GEE)" />
<meta property="og:description" content="ניטור ומעקב מדעי מתקדם של עכירות, כלורופיל ופריחת אצות בגופי המים בישראל באמצעות חישה מרחוק." />
<meta property="og:type" content="website" />
<meta property="og:url" content="https://israel-water-quality.streamlit.app/" />
"""
st.markdown(
    '<meta name="google-site-verification" content="להדביק_כאן_את_הקוד_מגוגל" />', 
    unsafe_allow_html=True
)

# Umami analytics (נשאר כקומפוננטה כרגיל)
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

# ==============================
# אימות GEE
# ==============================
@st.cache_resource
def init_gee():
    creds_dict = dict(st.secrets["gee_credentials"])
    creds_json = json.dumps(creds_dict)
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(creds_json)
        tmp_path = f.name
    service_account = creds_dict["client_email"]
    credentials = ee.ServiceAccountCredentials(service_account, tmp_path)
    ee.Initialize(credentials)
    os.unlink(tmp_path)

init_gee()

# ==============================
# הגדרות גופי מים
# ==============================

# ── חוף הים התיכון (מקורי) ────────────────────────────────────────────────────
HAIFA_CENTER = [32.4, 34.85]
HAIFA_BBOX   = ee.Geometry.Rectangle([34.50, 31.55, 35.15, 33.10])

ISRAEL_TERRITORIAL = ee.Geometry.Polygon([[
    [34.95, 33.10], [34.60, 33.10],
    [34.20, 32.60], [34.15, 32.00],
    [34.20, 31.55], [34.55, 31.30],
    [34.75, 31.25], [34.95, 31.30],
    [35.00, 31.55], [35.00, 32.00],
    [35.10, 32.60], [35.10, 33.10],
    [34.95, 33.10]
]])

BEACHES = [
    {"name": "ראש הנקרה",   "lat": 33.0765, "lon": 35.0983},  # חוף ראש הנקרה — ממש על קו החוף
    {"name": "נהריה",        "lat": 33.0048, "lon": 35.0832},  # חוף גלים, נהריה
    {"name": "עכו",          "lat": 32.9280, "lon": 35.0680},  # חוף עכו הצפוני
    {"name": "קריית ים",     "lat": 32.8618, "lon": 35.0648},  # חוף קריית ים
    {"name": "חיפה צפון",    "lat": 32.8380, "lon": 34.9820},  # חוף בת גלים
    {"name": "חיפה מרכז",    "lat": 32.8148, "lon": 34.9648},  # חוף כרמל
    {"name": "חיפה דרום",    "lat": 32.7780, "lon": 34.9530},  # חוף זלמן
    {"name": "עתלית",        "lat": 32.6892, "lon": 34.9368},  # חוף עתלית
    {"name": "זיכרון יעקב",  "lat": 32.5712, "lon": 34.9148},  # חוף דור
    {"name": "קיסריה",       "lat": 32.4948, "lon": 34.8912},  # חוף קיסריה
    {"name": "נתניה",        "lat": 32.3318, "lon": 34.8512},  # חוף שמשון, נתניה
    {"name": "הרצליה",       "lat": 32.1648, "lon": 34.7962},  # חוף ארנה, הרצליה פיתוח
    {"name": "תל אביב צפון", "lat": 32.1012, "lon": 34.7648},  # חוף הצוק
    {"name": "תל אביב מרכז", "lat": 32.0798, "lon": 34.7618},  # חוף גורדון / פרישמן
    {"name": "תל אביב דרום", "lat": 32.0548, "lon": 34.7568},  # חוף בוגרשוב / חילטון
    {"name": "בת ים",        "lat": 32.0148, "lon": 34.7448},  # חוף בת ים
    {"name": "ראשון לציון",  "lat": 31.9618, "lon": 34.7268},  # חוף ניצנים צפון
    {"name": "אשדוד צפון",   "lat": 31.8398, "lon": 34.6448},  # חוף לידו, אשדוד
    {"name": "אשדוד דרום",   "lat": 31.7848, "lon": 34.6248},  # חוף דולפינריום, אשדוד
    {"name": "אשקלון",       "lat": 31.6548, "lon": 34.5448},  # חוף אפולוניה, אשקלון
    {"name": "זיקים",        "lat": 31.6098, "lon": 34.5198},  # חוף זיקים
]

# ── כנרת ──────────────────────────────────────────────────────────────────────
KINNERET_CENTER = [32.82, 35.59]
KINNERET_BBOX   = ee.Geometry.Rectangle([35.50, 32.70, 35.68, 32.95])
KINNERET_POINTS = [
    {"name": "טבריה",          "lat": 32.794, "lon": 35.534},
    {"name": "צפון הכנרת",     "lat": 32.920, "lon": 35.595},
    {"name": "מזרח הכנרת",     "lat": 32.830, "lon": 35.635},
    {"name": "דרום הכנרת",     "lat": 32.713, "lon": 35.575},
    {"name": "נהר הירדן (כניסה)", "lat": 32.906, "lon": 35.630},
]

# ── ים המלח ───────────────────────────────────────────────────────────────────
DEAD_SEA_CENTER = [31.50, 35.47]
DEAD_SEA_BBOX   = ee.Geometry.Rectangle([35.35, 31.20, 35.60, 31.80])
DEAD_SEA_POINTS = [
    {"name": "עין גדי",        "lat": 31.462, "lon": 35.388},
    {"name": "עין בוקק",       "lat": 31.198, "lon": 35.352},
    {"name": "צפון ים המלח",   "lat": 31.760, "lon": 35.455},
    {"name": "מרכז ים המלח",   "lat": 31.520, "lon": 35.450},
]

# ── ים סוף (מפרץ עקבה) ────────────────────────────────────────────────────────
RED_SEA_CENTER = [29.55, 34.95]
RED_SEA_BBOX   = ee.Geometry.Rectangle([34.80, 29.35, 35.10, 29.75])
RED_SEA_POINTS = [
    {"name": "אילת צפון",      "lat": 29.558, "lon": 34.952},
    {"name": "אילת דרום",      "lat": 29.499, "lon": 34.920},
    {"name": "מפרץ עקבה",      "lat": 29.430, "lon": 34.930},
]

# ── מיפוי כולל ────────────────────────────────────────────────────────────────
WATER_BODIES = {
    "🏖️ חוף הים התיכון": {
        "center":    HAIFA_CENTER,
        "zoom":      8,
        "bbox":      HAIFA_BBOX,
        "clip_geom": ISRAEL_TERRITORIAL,
        "points":    BEACHES,
        "sensor":    "S2",      # Sentinel-2 (10m)
        "cloud_pct": 20,
        "days_back": 90,
        "days_back_s1": 30,     # S1 חוזר כל 6 ימים — טווח קצר מספיק
        "note":      None,
        "note_s1":   "🛢️ S1 SAR — זיהוי שמן/זיהום לפי dark spots ב-VV backscatter. עובד בכל מזג אוויר ובלילה.",
        "indices":   ["NDWI", "Chl_proxy", "Turbidity", "FAI"],
        "indices_s1":["VV", "VH", "VV_VH_ratio", "Roughness"],
    },
    "🌊 כנרת": {
        "center":    KINNERET_CENTER,
        "zoom":      12,
        "bbox":      KINNERET_BBOX,
        "clip_geom": KINNERET_BBOX,
        "points":    KINNERET_POINTS,
        "sensor":    "S2",
        "cloud_pct": 10,        # כנרת קטנה — בעיות ענן קריטיות יותר
        "days_back": 60,
        "days_back_s1": 30,
        "note":      "⚠️ ב-10m רזולוציה הכנרת מכוסה היטב. שים לב לפריחות אצות בקיץ (NDCI גבוה).",
        "note_s1":   "🛢️ S1 SAR — זיהוי שמן/זיהום ע\"ג מי הכנרת. VV backscatter נמוך = חשד לזיהום.",
        "indices":   ["NDWI", "Chl_proxy", "Turbidity", "FAI"],
        "indices_s1":["VV", "VH", "VV_VH_ratio", "Roughness"],
    },
    "🧂 ים המלח": {
        "center":    DEAD_SEA_CENTER,
        "zoom":      11,
        "bbox":      DEAD_SEA_BBOX,
        "clip_geom": DEAD_SEA_BBOX,
        "points":    DEAD_SEA_POINTS,
        "sensor":    "S2",
        "cloud_pct": 20,
        "days_back": 90,
        "days_back_s1": 30,
        "note":      "⚠️ מלוחות קיצונית — מדדי כלורופיל ועכירות אינם קלינריים. FAI מזהה Dunaliella salina (אצה ורודה). השתמש בציונים כ'אנומליה' בלבד.",
        "note_s1":   "🛢️ S1 SAR — מלוחות גבוהה ומשטחי אדיפה משפיעים על backscatter. VV/VH ratio מזהה אנומליות פני שטח.",
        "indices":   ["NDWI", "FAI", "Turbidity"],   # Chl_proxy פחות רלוונטי
        "indices_s1":["VV", "VH", "VV_VH_ratio", "Roughness"],
    },
    "🐠 ים סוף": {
        "center":    RED_SEA_CENTER,
        "zoom":      12,
        "bbox":      RED_SEA_BBOX,
        "clip_geom": RED_SEA_BBOX,
        "points":    RED_SEA_POINTS,
        "sensor":    "S2",
        "cloud_pct": 10,        # נדיר שיש עננים, אבל נשמר להגדרה עקבית
        "days_back": 120,       # אוליגוטרופי — צריך יותר תמונות
        "days_back_s1": 30,
        "note":      "💡 מים אוליגוטרופיים (כלורופיל נמוך מאוד ~0.05 mg/m³). הרחב את טווח התאריכים אם אין תמונות ברורות. שים לב לסחף אבק מהסיני.",
        "note_s1":   "🛢️ S1 SAR — מים אוליגוטרופיים שקטים מראים VV נמוך טבעי. dark spots ב-VV = חשד לשמן מספינות.",
        "indices":   ["NDWI", "Chl_proxy", "Turbidity"],
        "indices_s1":["VV", "VH", "VV_VH_ratio", "Roughness"],
    },
}

SCORE_COLORS = {0: "#AAAAAA", 1: "#27AE60", 2: "#F1C40F", 3: "#E67E22", 4: "#E74C3C", 5: "#8E44AD"}

# ==============================
# Snap לקו מים רשמי (JRC Global Surface Water) — cached בנפרד מהתמונה
# ==============================
@st.cache_data(ttl=604800)   # שבוע — קו המים לא משתנה
def snap_points_to_coastline(points_key: str, points_list: list, search_radius: int = 3000) -> list:
    """
    מצמיד כל נקודה לנקודת המים הקרובה ביותר.
    JRC Global Surface Water (Google/EU) — רזולוציה 30m, מקור רשמי.
    occurrence >= 50 = מים קבועים.
    """
    jsw             = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    permanent_water = jsw.select("occurrence").gte(50)

    def snap_one(point):
        try:
            pt     = ee.Geometry.Point([point["lon"], point["lat"]])
            buffer = pt.buffer(search_radius)

            # reduced numPixels from 500 → 100: sufficient for snapping, much faster
            water_pts = permanent_water.updateMask(permanent_water).sample(
                region=buffer,
                scale=30,
                geometries=True,
                numPixels=100
            )

            count = water_pts.size().getInfo()
            if count == 0:
                return point

            def add_distance(feat):
                d = feat.geometry().distance(pt, 1)
                return feat.set("dist", d)

            nearest = (water_pts
                       .map(add_distance)
                       .sort("dist")
                       .first()
                       .geometry()
                       .centroid(1)
                       .coordinates()
                       .getInfo())

            if nearest and len(nearest) == 2:
                return {**point, "lat": nearest[1], "lon": nearest[0]}
            return point
        except Exception:
            return point

    # Run all snapping calls in parallel — one GEE round-trip per point
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(snap_one, p): i for i, p in enumerate(points_list)}
        ordered = [None] * len(points_list)
        for future in as_completed(futures):
            idx = futures[future]
            ordered[idx] = future.result()

    return ordered


# ==============================
# טעינת נתונים — גנרי לכל גוף מים + סנסור
# ==============================
@st.cache_data(ttl=3600)
def load_data(wb_key: str, start_date: str, end_date: str, sensor: str = "S2"):
    wb = WATER_BODIES[wb_key]

    if sensor == "S1":
        # Sentinel-1 GRD SAR — 10m, זיהוי שמן/זיהום לפי VV backscatter
        collection = (ee.ImageCollection("COPERNICUS/S1_GRD")
                      .filterBounds(wb["bbox"])
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.eq("instrumentMode", "IW"))
                      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                      .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")))

        first_info = collection.sort("system:time_start", False).limit(1).getInfo()
        if not first_info["features"]:
            # נסה גם ascending אם descending ריק
            collection = (ee.ImageCollection("COPERNICUS/S1_GRD")
                          .filterBounds(wb["bbox"])
                          .filterDate(start_date, end_date)
                          .filter(ee.Filter.eq("instrumentMode", "IW"))
                          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
            first_info = collection.sort("system:time_start", False).limit(1).getInfo()
            if not first_info["features"]:
                return None, None, None, 0, []

        count = len(first_info["features"])
        image_date = datetime.utcfromtimestamp(
            first_info["features"][0]["properties"]["system:time_start"] / 1000
        ).strftime("%Y-%m-%d")

        def compute_indices_s1(image):
            vv = image.select("VV")  # dB
            vh = image.select("VH")  # dB
            # VV/VH ratio (linear) — oil spill מוריד VV יותר מ-VH
            vv_lin    = ee.Image(10).pow(vv.divide(10))
            vh_lin    = ee.Image(10).pow(vh.divide(10))
            ratio     = vv_lin.divide(vh_lin.add(1e-10)).rename("VV_VH_ratio")
            # Roughness proxy — VV normalized
            roughness = vv.add(25).divide(25).clamp(-1, 1).rename("Roughness")
            return image.addBands([vv.rename("VV"), vh.rename("VH"), ratio, roughness])

        processed = collection.map(compute_indices_s1).median()

    elif sensor == "S3":
        # Sentinel-3 OLCI — 300m, מדדים ימיים ייעודיים
        # S3 ב-GEE זמין עד סוף 2025 — נקבע end_date מקסימלי בהתאם
        s3_end   = end_date
        s3_start = start_date

        collection = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                      .filterBounds(wb["bbox"])
                      .filterDate(s3_start, s3_end))

        # Single getInfo() to check existence and grab date simultaneously
        first_info = collection.sort("system:time_start", False).limit(1).getInfo()
        if not first_info["features"]:
            return None, None, None, 0, []

        count = len(first_info["features"])  # at least 1; full count not critical here
        image_date = datetime.utcfromtimestamp(
            first_info["features"][0]["properties"]["system:time_start"] / 1000
        ).strftime("%Y-%m-%d")

        def compute_indices_s3(image):
            # S3 OLCI — radiance גולמי (לא reflectance), צריך ratio ולא NDWI קלאסי
            green   = image.select("Oa06_radiance")   # 560nm — ירוק
            red     = image.select("Oa08_radiance")   # 665nm — אדום
            rededge = image.select("Oa11_radiance")   # 709nm — red edge
            nir     = image.select("Oa17_radiance")   # 865nm — NIR

            # NDWI — עובד גם על radiance יחסי
            ndwi      = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
            # Chl_proxy — NDCI ימי (red edge vs red)
            chl_proxy = rededge.subtract(red).divide(rededge.add(red)).rename("Chl_proxy")
            # Turbidity — ערוץ אדום גולמי
            turbidity = red.rename("Turbidity")
            # FAI — Floating Algae Index
            fai = (nir.subtract(red)
                      .subtract(
                          rededge.subtract(red)
                          .multiply((865.0 - 665.0) / (709.0 - 665.0))
                      )
                      .rename("FAI"))
            return image.addBands([ndwi, chl_proxy, turbidity, fai])

        processed = collection.map(compute_indices_s3).median()

    else:
        # Sentinel-2 MSI — 10m
        collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                      .filterBounds(wb["bbox"])
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", wb["cloud_pct"])))

        # Single getInfo() to check existence and grab date simultaneously
        first_info = collection.sort("system:time_start", False).limit(1).getInfo()
        if not first_info["features"]:
            return None, None, None, 0, []

        count = len(first_info["features"])  # at least 1; full count not critical here
        image_date = datetime.utcfromtimestamp(
            first_info["features"][0]["properties"]["system:time_start"] / 1000
        ).strftime("%Y-%m-%d")

        def compute_indices_s2(image):
            ndwi      = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
            chl_proxy = image.select("B5").divide(image.select("B4")).rename("Chl_proxy")
            turbidity = image.select("B4").rename("Turbidity")
            fai = (image.select("B8")
                   .subtract(image.select("B4"))
                   .subtract(image.select("B11").subtract(image.select("B4"))
                             .multiply((832-665)/(1610-665)))
                   .rename("FAI"))
            return image.addBands([ndwi, chl_proxy, turbidity, fai])

        processed = collection.map(compute_indices_s2).median()

    # ── Snap לקו חוף רשמי (cached שבוע, לא תלוי בתמונה) ──────────────────
    snapped_points = snap_points_to_coastline(wb_key, wb["points"])

    # רזולוציה לפי סנסור
    scale = 300 if sensor == "S3" else 10  # S1 ו-S2 שניהם 10m

    def get_point_values(point):
        snapped_point = point
        pt         = ee.Geometry.Point([point["lon"], point["lat"]])
        buffer_1km = pt.buffer(1000 if sensor in ("S1", "S2") else 5000)
        try:
            if sensor == "S1":
                # S1 SAR — אין NDWI; משתמשים ב-VV כמסכת מים (מים → VV נמוך מאוד)
                vv_img     = processed.select("VV")
                water_mask = vv_img.lt(-10)   # threshold: VV < -10 dB = מים/גלישה שמן
                s1_bands   = ["VV", "VH", "VV_VH_ratio", "Roughness"]
                selected   = processed.select(s1_bands).updateMask(water_mask)

                vals_raw = selected.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=buffer_1km, scale=10, bestEffort=True
                ).getInfo()

                vv_val    = vals_raw.get("VV")
                vh_val    = vals_raw.get("VH")
                ratio_val = vals_raw.get("VV_VH_ratio")
                rough_val = vals_raw.get("Roughness")

                if all(v is None for v in [vv_val, vh_val, ratio_val, rough_val]):
                    return {**snapped_point,
                            "vv": None, "vh": None, "vv_vh_ratio": None, "roughness": None,
                            "ndwi": None, "chl_proxy": None, "turbidity": None, "fai": None,
                            "no_data": True}
                return {**snapped_point,
                        "vv":         round(vv_val,    2) if vv_val    is not None else None,
                        "vh":         round(vh_val,    2) if vh_val    is not None else None,
                        "vv_vh_ratio":round(ratio_val, 3) if ratio_val is not None else None,
                        "roughness":  round(rough_val, 3) if rough_val is not None else None,
                        "ndwi": None, "chl_proxy": None, "turbidity": None, "fai": None,
                        "no_data": False}
            else:
                # ── S2 / S3 — הלוגיקה המקורית ────────────────────────────────
                ndwi_img   = processed.select("NDWI")
                ndwi_thresh = -0.1 if sensor == "S3" else 0.0
                water_mask  = ndwi_img.gt(ndwi_thresh)

                distance_img = ee.Image(0).paint(
                    featureCollection=ee.FeatureCollection([ee.Feature(pt)]),
                    color=1
                ).fastDistanceTransform().sqrt().multiply(scale)

                max_dist      = 1000 if sensor == "S2" else 5000
                weight        = ee.Image(max_dist).subtract(distance_img).divide(max_dist).max(0)
                weight_masked = weight.updateMask(water_mask)
                selected      = processed.select(["NDWI","Chl_proxy","Turbidity","FAI"]).updateMask(water_mask)

                weighted_sum = selected.multiply(weight_masked).reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=buffer_1km, scale=scale, bestEffort=True).getInfo()
                weight_sum = weight_masked.reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=buffer_1km, scale=scale, bestEffort=True).getInfo()

                def wm(band):
                    ws = weighted_sum.get(band)
                    wt = weight_sum.get("constant")
                    if ws is None or wt is None or wt == 0: return None
                    return ws / wt

                vals = {k: wm(k) for k in ["NDWI","Chl_proxy","Turbidity","FAI"]}
                if all(v is None for v in vals.values()):
                    return {**snapped_point,
                            "vv": None, "vh": None, "vv_vh_ratio": None, "roughness": None,
                            **{k: None for k in vals}, "no_data": True}

                return {**snapped_point,
                        "vv": None, "vh": None, "vv_vh_ratio": None, "roughness": None,
                        "ndwi":      round(vals["NDWI"],      3) if vals["NDWI"]      is not None else None,
                        "chl_proxy": round(vals["Chl_proxy"], 3) if vals["Chl_proxy"] is not None else None,
                        "turbidity": round(vals["Turbidity"], 1) if vals["Turbidity"] is not None else None,
                        "fai":       round(vals["FAI"],       4) if vals["FAI"]       is not None else None,
                        "no_data": False}
        except:
            return {**snapped_point,
                    "vv": None, "vh": None, "vv_vh_ratio": None, "roughness": None,
                    "ndwi": None, "chl_proxy": None, "turbidity": None, "fai": None,
                    "no_data": True}

    # Parallel GEE calls — all points fetched concurrently instead of sequentially
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_point_values, p): i for i, p in enumerate(snapped_points)}
        data = [None] * len(snapped_points)
        for future in as_completed(futures):
            idx = futures[future]
            data[idx] = future.result()
    df   = pd.DataFrame(data)

    def water_quality_score(row):
        if row["no_data"]: return 0
        if sensor == "S1":
            # S1: VV נמוך מאוד = מים חלקים = חשד לשמן
            score = 1  # התחלה: "מצוין" — נקי עד שיש ראיה אחרת
            vv = row.get("vv")
            ratio = row.get("vv_vh_ratio")
            if vv is not None:
                if vv < -20:   score += 3  # dark spot חזק — שמן/זיהום
                elif vv < -15: score += 1  # dark spot קל — חשד
            if ratio is not None:
                if ratio < 1.5: score += 1  # ratio נמוך = surface damping
            return max(1, min(5, score))
        # S2 / S3 — לוגיקה מקורית
        score = 3
        if row["ndwi"] is not None:
            if row["ndwi"] > 0.3:    score -= 1
            elif row["ndwi"] < 0.1:  score += 1
        if row["chl_proxy"] is not None:
            if row["chl_proxy"] > 1.5:   score += 1
            elif row["chl_proxy"] < 1.1: score -= 1
        if row["turbidity"] is not None:
            if row["turbidity"] > 500:   score += 1
            elif row["turbidity"] < 200: score -= 1
        return max(1, min(5, score))

    def quality_label(score):
        if sensor == "S1":
            return {0:"⬜ אין מידע", 1:"🟢 נקי", 2:"🟡 חשד קל",
                    3:"🟠 חשד זיהום", 4:"🔴 זיהום סביר", 5:"⛔ זיהום חמור"}.get(score, "❓")
        return {0:"⬜ אין מידע", 1:"🟢 מצוין", 2:"🟡 טוב",
                3:"🟠 בינוני",  4:"🔴 ירוד",  5:"⛔ גרוע"}.get(score, "❓")

    def composite_score(row):
        if row["no_data"]: return None
        if sensor == "S1":
            # ציון S1: 100 = נקי לחלוטין, 0 = זיהום חמור
            score, weights = 0, 0
            vv = row.get("vv")
            ratio = row.get("vv_vh_ratio")
            rough = row.get("roughness")
            if vv is not None:
                # VV טיפוסי למים נקיים: -12 עד -8 dB; שמן: מתחת ל-20
                vv_score = min(100, max(0, (vv + 25) / 15 * 100))
                score += vv_score * 0.5; weights += 0.5
            if ratio is not None:
                # ratio נורמלי: 2-5; נמוך מ-1.5 = damping
                ratio_score = min(100, max(0, (ratio - 1) / 4 * 100))
                score += ratio_score * 0.35; weights += 0.35
            if rough is not None:
                rough_score = min(100, max(0, (rough + 1) / 2 * 100))
                score += rough_score * 0.15; weights += 0.15
            return round(score / weights, 1) if weights > 0 else None
        # S2 / S3
        score, weights = 0, 0
        if row["ndwi"] is not None:
            score += min(100, max(0, (row["ndwi"]+0.3)/1.1*100)) * 0.4;  weights += 0.4
        if row["chl_proxy"] is not None:
            score += min(100, max(0, (2.5-row["chl_proxy"])/1.5*100)) * 0.35; weights += 0.35
        if row["turbidity"] is not None:
            score += min(100, max(0, (1000-row["turbidity"])/1000*100)) * 0.25; weights += 0.25
        return round(score/weights, 1) if weights > 0 else None

    df["quality_score"] = df.apply(water_quality_score, axis=1)
    df["quality_label"] = df["quality_score"].map(quality_label)
    df["composite"]     = df.apply(composite_score, axis=1)

    # ── פוליגוני פיקסלי מים לכל נקודה (לתצוגה ויזואלית) ──────────────────
    def get_water_polygon(point):
        """מחזיר GeoJSON של convex hull של פיקסלי המים ב-buffer סביב הנקודה"""
        try:
            pt  = ee.Geometry.Point([point["lon"], point["lat"]])
            buf = pt.buffer(1000 if sensor in ("S1", "S2") else 5000)
            if sensor == "S1":
                # S1: מים לפי VV < -10 dB
                water_mask = processed.select("VV").lt(-10)
                poly_scale = 20
            else:
                ndwi_img    = processed.select("NDWI")
                ndwi_thresh = -0.1 if sensor == "S3" else 0.0
                water_mask  = ndwi_img.gt(ndwi_thresh)
                poly_scale  = 300 if sensor == "S3" else 20

            vectors = water_mask.updateMask(water_mask).reduceToVectors(
                geometry=buf,
                scale=poly_scale,
                geometryType="polygon",
                bestEffort=True,
                maxPixels=1e6
            )
            hull = vectors.geometry().convexHull(10)
            return hull.getInfo()
        except:
            return None

    # Parallel fetch for water polygons
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_water_polygon, p): i for i, p in enumerate(snapped_points)}
        water_polygons = [None] * len(snapped_points)
        for future in as_completed(futures):
            idx = futures[future]
            water_polygons[idx] = future.result()

    return df, image_date, processed, count, water_polygons


# ==============================
# שאילתה לנקודה произвольная במפה (click-to-query)
# ==============================
def query_clicked_point(lat: float, lon: float, wb_key: str,
                        start_date: str, end_date: str, sensor: str) -> dict:
    """
    מקבל קואורדינטות שנלחצו על המפה ומחזיר ערכי מדדים מ-GEE.
    בונה את ה-processed image מחדש (לא ניתן לסריאליזציה ב-cache).
    """
    wb = WATER_BODIES[wb_key]

    if sensor == "S1":
        def compute_indices_s1_click(image):
            vv = image.select("VV")
            vh = image.select("VH")
            vv_lin = ee.Image(10).pow(vv.divide(10))
            vh_lin = ee.Image(10).pow(vh.divide(10))
            ratio  = vv_lin.divide(vh_lin.add(1e-10)).rename("VV_VH_ratio")
            rough  = vv.add(25).divide(25).clamp(-1, 1).rename("Roughness")
            return image.addBands([vv.rename("VV"), vh.rename("VH"), ratio, rough])

        collection = (ee.ImageCollection("COPERNICUS/S1_GRD")
                      .filterBounds(wb["bbox"])
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.eq("instrumentMode", "IW"))
                      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
        processed = collection.map(compute_indices_s1_click).median()
        scale = 10

    elif sensor == "S3":
        s3_end   = end_date
        s3_start = start_date

        def compute_indices_s3(image):
            green   = image.select("Oa06_radiance")
            red     = image.select("Oa08_radiance")
            rededge = image.select("Oa11_radiance")
            nir     = image.select("Oa17_radiance")
            ndwi      = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
            chl_proxy = rededge.subtract(red).divide(rededge.add(red)).rename("Chl_proxy")
            turbidity = red.rename("Turbidity")
            fai = (nir.subtract(red)
                      .subtract(rededge.subtract(red).multiply((865.0-665.0)/(709.0-665.0)))
                      .rename("FAI"))
            return image.addBands([ndwi, chl_proxy, turbidity, fai])

        collection = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                      .filterBounds(wb["bbox"])
                      .filterDate(s3_start, s3_end))
        processed = collection.map(compute_indices_s3).median()
        scale = 300
    else:
        def compute_indices_s2(image):
            ndwi      = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
            chl_proxy = image.select("B5").divide(image.select("B4")).rename("Chl_proxy")
            turbidity = image.select("B4").rename("Turbidity")
            fai = (image.select("B8")
                   .subtract(image.select("B4"))
                   .subtract(image.select("B11").subtract(image.select("B4"))
                             .multiply((832-665)/(1610-665)))
                   .rename("FAI"))
            return image.addBands([ndwi, chl_proxy, turbidity, fai])

        collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                      .filterBounds(wb["bbox"])
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", wb["cloud_pct"])))
        processed = collection.map(compute_indices_s2).median()
        scale = 10

    pt     = ee.Geometry.Point([lon, lat])
    buffer = pt.buffer(500 if sensor in ("S1", "S2") else 2000)

    try:
        if sensor == "S1":
            vals = processed.select(["VV", "VH", "VV_VH_ratio", "Roughness"])                 .reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer, scale=10, bestEffort=True).getInfo()

            if not vals or all(v is None for v in vals.values()):
                return {"error": "אין נתוני S1 בנקודה זו"}

            vv_v    = vals.get("VV")
            vh_v    = vals.get("VH")
            ratio_v = vals.get("VV_VH_ratio")
            rough_v = vals.get("Roughness")

            # ציון זיהום S1
            score, weights = 0, 0
            if vv_v is not None:
                vv_score = min(100, max(0, (vv_v + 25) / 15 * 100))
                score += vv_score * 0.5; weights += 0.5
            if ratio_v is not None:
                ratio_score = min(100, max(0, (ratio_v - 1) / 4 * 100))
                score += ratio_score * 0.35; weights += 0.35
            if rough_v is not None:
                rough_score = min(100, max(0, (rough_v + 1) / 2 * 100))
                score += rough_score * 0.15; weights += 0.15
            composite = round(score / weights, 1) if weights > 0 else None

            quality_map_s1 = [
                (lambda s: s is None,  "⬜ אין מידע",    "#AAAAAA"),
                (lambda s: s >= 80,    "🟢 נקי",          "#27AE60"),
                (lambda s: s >= 60,    "🟡 חשד קל",       "#F1C40F"),
                (lambda s: s >= 40,    "🟠 חשד זיהום",   "#E67E22"),
                (lambda s: s >= 20,    "🔴 זיהום סביר",  "#E74C3C"),
                (lambda s: True,       "⛔ זיהום חמור",  "#8E44AD"),
            ]
            quality_label, quality_color = "❓", "#888"
            for condition, label, color in quality_map_s1:
                if condition(composite):
                    quality_label, quality_color = label, color
                    break

            return {
                "vv":            round(vv_v,    2) if vv_v    is not None else None,
                "vh":            round(vh_v,    2) if vh_v    is not None else None,
                "vv_vh_ratio":   round(ratio_v, 3) if ratio_v is not None else None,
                "roughness":     round(rough_v, 3) if rough_v is not None else None,
                "ndwi": None, "chl_proxy": None, "turbidity": None, "fai": None,
                "composite":     composite,
                "quality_label": quality_label,
                "quality_color": quality_color,
                "lat":           round(lat, 5),
                "lon":           round(lon, 5),
            }

        # ── S2 / S3 ──────────────────────────────────────────────────────
        vals = processed.select(["NDWI", "Chl_proxy", "Turbidity", "FAI"])             .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=scale,
                bestEffort=True
            ).getInfo()

        if not vals or all(v is None for v in vals.values()):
            return {"error": "אין נתוני לוויין בנקודה זו (אזור יבשה או ענן)"}

        ndwi      = vals.get("NDWI")
        chl_proxy = vals.get("Chl_proxy")
        turbidity = vals.get("Turbidity")
        fai       = vals.get("FAI")

        # חישוב ציון מורכב
        score, weights = 0, 0
        if ndwi is not None:
            score += min(100, max(0, (ndwi + 0.3) / 1.1 * 100)) * 0.4;  weights += 0.4
        if chl_proxy is not None:
            score += min(100, max(0, (2.5 - chl_proxy) / 1.5 * 100)) * 0.35; weights += 0.35
        if turbidity is not None:
            score += min(100, max(0, (1000 - turbidity) / 1000 * 100)) * 0.25; weights += 0.25
        composite = round(score / weights, 1) if weights > 0 else None

        quality_map = [
            (lambda s: s is None,  "⬜ אין מידע",  "#AAAAAA"),
            (lambda s: s >= 80,    "🟢 מצוין",      "#27AE60"),
            (lambda s: s >= 60,    "🟡 טוב",         "#F1C40F"),
            (lambda s: s >= 40,    "🟠 בינוני",     "#E67E22"),
            (lambda s: s >= 20,    "🔴 ירוד",        "#E74C3C"),
            (lambda s: True,       "⛔ גרוע",        "#8E44AD"),
        ]
        quality_label, quality_color = "❓", "#888"
        for condition, label, color in quality_map:
            if condition(composite):
                quality_label, quality_color = label, color
                break

        return {
            "ndwi":          round(ndwi,      3) if ndwi      is not None else None,
            "chl_proxy":     round(chl_proxy, 3) if chl_proxy is not None else None,
            "turbidity":     round(turbidity, 1) if turbidity is not None else None,
            "fai":           round(fai,       4) if fai       is not None else None,
            "vv": None, "vh": None, "vv_vh_ratio": None, "roughness": None,
            "composite":     composite,
            "quality_label": quality_label,
            "quality_color": quality_color,
            "lat":           round(lat, 5),
            "lon":           round(lon, 5),
        }
    except Exception as e:
        return {"error": f"שגיאה בקריאת GEE: {str(e)}"}


# ==============================
# מפת חום ברמת פיקסל — תואם S1, S2 ו-S3
# ==============================
def get_heatmap_url(processed, clip_geom, sensor="S2"):
    # JRC water mask — מסנן יבשה בצורה אמינה
    jsw        = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    water_mask = jsw.select("occurrence").gte(30)

    if sensor == "S1":
        # S1 SAR — מפת זיהום: VV נמוך = dark spot = חשד לשמן
        vv = processed.select("VV").updateMask(water_mask)
        # VV טיפוסי למים: -15 עד -5 dB; שמן: מתחת ל-20
        # ציון גבוה = נקי (VV גבוה), ציון נמוך = dark spot (שמן)
        vv_score = vv.add(25).divide(15).multiply(100).clamp(0, 100)

        # ratio נמוך = surface damping
        ratio = processed.select("VV_VH_ratio").updateMask(water_mask)
        ratio_score = ratio.subtract(1).divide(4).multiply(100).clamp(0, 100)

        composite = (vv_score.multiply(0.65).add(ratio_score.multiply(0.35)))
        composite  = composite.updateMask(water_mask).clip(clip_geom)

        # צבעים: אדום = dark spot/זיהום, ירוק = נקי
        vis_params = {
            "min": 0, "max": 100,
            "palette": ["#8B0000","#E74C3C","#E67E22","#F1C40F","#27AE60","#1A5E20"]
        }
        try:
            return composite.getMapId(vis_params)["tile_fetcher"].url_format
        except:
            return None

    ndwi      = processed.select("NDWI").updateMask(water_mask)
    chl_proxy = processed.select("Chl_proxy").updateMask(water_mask)
    turbidity = processed.select("Turbidity").updateMask(water_mask)

    if sensor == "S3":
        # נרמול דינמי לפי percentile — מציג את ההבדלים האמיתיים בתמונה
        stats = processed.select(["NDWI","Chl_proxy","Turbidity"]).updateMask(water_mask).reduceRegion(
            reducer=ee.Reducer.percentile([5, 95]),
            geometry=clip_geom,
            scale=300,
            bestEffort=True
        ).getInfo()

        ndwi_min  = stats.get("NDWI_p5",  -0.3)
        ndwi_max  = stats.get("NDWI_p95",  0.3)
        chl_min   = stats.get("Chl_proxy_p5",  -0.2)
        chl_max   = stats.get("Chl_proxy_p95",  0.2)
        turb_min  = stats.get("Turbidity_p5",   0)
        turb_max  = stats.get("Turbidity_p95",  150)

        ndwi_range  = max(ndwi_max  - ndwi_min,  0.01)
        chl_range   = max(chl_max   - chl_min,   0.01)
        turb_range  = max(turb_max  - turb_min,  1.0)

        ndwi_score  = ndwi.subtract(ndwi_min).divide(ndwi_range).multiply(100).clamp(0, 100)
        chl_score   = ee.Image(chl_max).subtract(chl_proxy).divide(chl_range).multiply(100).clamp(0, 100)
        turb_score  = ee.Image(turb_max).subtract(turbidity).divide(turb_range).multiply(100).clamp(0, 100)
    else:
        ndwi_score  = ndwi.add(0.3).divide(1.1).multiply(100).clamp(0, 100)
        chl_score   = ee.Image(2.5).subtract(chl_proxy).divide(1.5).multiply(100).clamp(0, 100)
        turb_score  = ee.Image(1000).subtract(turbidity).divide(1000).multiply(100).clamp(0, 100)

    composite = (ndwi_score.multiply(0.4)
                 .add(chl_score.multiply(0.35))
                 .add(turb_score.multiply(0.25)))

    composite = composite.updateMask(water_mask).clip(clip_geom)

    vis_params = {
        "min": 0, "max": 100,
        "palette": ["#8B0000","#E74C3C","#E67E22","#F1C40F","#27AE60","#1A5E20"]
    }
    try:
        return composite.getMapId(vis_params)["tile_fetcher"].url_format
    except:
        return None

# ==============================
# בניית מפה
# ==============================
def build_map(df, image_date, processed, wb_key, water_polygons=None, sensor="S2"):
    wb = WATER_BODIES[wb_key]

    m = folium.Map(
        location=wb["center"],
        zoom_start=wb["zoom"],
        tiles="CartoDB positron",
        control_scale=True
    )

    s1_mode = (sensor == "S1")
    sat_layer_name = "🛰️ S1 VV backscatter" if s1_mode else "🛰️ לווין RGB"
    satellite_group  = folium.FeatureGroup(name=sat_layer_name,               show=s1_mode)
    heatmap_group    = folium.FeatureGroup(name="🌡️ מפת חום (ציון משוכלל)",  show=(not s1_mode))
    oil_group        = folium.FeatureGroup(name="🛢️ מפת זיהום SAR",           show=s1_mode) if s1_mode else None
    sampling_group   = folium.FeatureGroup(name="🔵 אזורי דיגום",              show=False)
    points_group     = folium.FeatureGroup(name="📍 נקודות דיגום",              show=True)

    # שכבת לווין
    try:
        if sensor == "S1":
            # S1 SAR — VV grayscale (backscatter)
            rgb_url = processed.select("VV").getMapId(
                {"min": -25, "max": 0, "palette": ["black","white"]}
            )["tile_fetcher"].url_format
        elif sensor == "S3":
            rgb_url = processed.select(["Oa08_radiance","Oa06_radiance","Oa04_radiance"]).getMapId(
                {"min": 0, "max": 200, "gamma": 1.5}
            )["tile_fetcher"].url_format
        else:
            rgb_url = processed.getMapId({"bands":["B4","B3","B2"],"min":0,"max":3000})["tile_fetcher"].url_format
        folium.TileLayer(tiles=rgb_url, name=sat_layer_name, attr="GEE/Copernicus", overlay=True, opacity=0.8).add_to(satellite_group)
    except:
        pass

    # מפת חום
    heatmap_url = get_heatmap_url(processed, wb["clip_geom"], sensor)
    if heatmap_url:
        folium.TileLayer(
            tiles=heatmap_url, name="מפת חום", attr="GEE/Copernicus",
            overlay=True, opacity=0.75
        ).add_to(heatmap_group)

    # אזורי דיגום — פוליגון מדויק של פיקסלי המים שנלקחו בחישוב
    if water_polygons:
        for i, (_, row) in enumerate(df.iterrows(), 1):
            geojson = water_polygons[i-1] if i-1 < len(water_polygons) else None
            if geojson is None:
                continue
            folium.GeoJson(
                geojson,
                name=f"zone_{i}",
                style_function=lambda x: {
                    "fillColor":   "#3498DB",
                    "color":       "#1A6FBF",
                    "weight":      1.5,
                    "fillOpacity": 0.25,
                },
                tooltip=f"אזור דיגום {i} — {row['name']} (פיקסלי מים בלבד, scale 20m)"
            ).add_to(sampling_group)

    # נקודות דיגום
    for i, (_, row) in enumerate(df.iterrows(), 1):
        color = SCORE_COLORS.get(row["quality_score"], "#888")
        comp  = f"{int(round(row['composite']))}/100" if (row["composite"] is not None and row["composite"] == row["composite"]) else "N/A"

        if row["no_data"]:
            popup_html = (
                f"<div style='font-family:Arial;direction:rtl;'>"
                f"<b>נקודה {i} — {row['name']}</b><br>⬜ אין מידע זמין<br>"
                f"<small>📅 {image_date}</small></div>"
            )
        elif sensor == "S1":
            # S1 SAR — מדדי backscatter
            vv_str    = f"{row['vv']:.2f} dB"    if row.get("vv")          is not None else "N/A"
            vh_str    = f"{row['vh']:.2f} dB"    if row.get("vh")          is not None else "N/A"
            ratio_str = f"{row['vv_vh_ratio']:.3f}" if row.get("vv_vh_ratio") is not None else "N/A"
            rough_str = f"{row['roughness']:.3f}"   if row.get("roughness")   is not None else "N/A"
            popup_html = f"""<div style='font-family:Arial;direction:rtl;min-width:200px;'>
                <h3 style='color:{color};margin:0 0 6px;'>{i}. {row['name']}</h3>
                <b>{row['quality_label']}</b><br>
                <b style='font-size:15px;'>🛢️ {comp}</b><br><br>
                <table style='font-size:13px;border-collapse:collapse;width:100%;'>
                <tr style='background:#f5f5f5'><td style='padding:4px'><b>VV</b></td><td style='padding:4px'>{vv_str}</td><td style='padding:4px;color:#999;font-size:11px'>נמוך=dark spot</td></tr>
                <tr><td style='padding:4px'><b>VH</b></td><td style='padding:4px'>{vh_str}</td><td style='padding:4px;color:#999;font-size:11px'>cross-pol</td></tr>
                <tr style='background:#f5f5f5'><td style='padding:4px'><b>VV/VH</b></td><td style='padding:4px'>{ratio_str}</td><td style='padding:4px;color:#999;font-size:11px'>&lt;1.5=damping</td></tr>
                <tr><td style='padding:4px'><b>Roughness</b></td><td style='padding:4px'>{rough_str}</td><td style='padding:4px;color:#999;font-size:11px'>חספוס</td></tr>
                </table>
                <br><small style='color:#aaa'>📅 {image_date} · SAR</small></div>"""
        else:
            # S2 / S3 — מדדים אופטיים
            fai_row = ""
            if row.get("fai") is not None:
                fai_row = f"<tr style='background:#f5f5f5'><td style='padding:4px'><b>FAI</b></td><td style='padding:4px'>{row['fai']:.4f}</td><td style='padding:4px;color:#999;font-size:11px'>גבוה=אצות צפות</td></tr>"

            popup_html = f"""<div style='font-family:Arial;direction:rtl;min-width:200px;'>
                <h3 style='color:{color};margin:0 0 6px;'>{i}. {row['name']}</h3>
                <b>{row['quality_label']}</b><br>
                <b style='font-size:15px;'>⭐ {comp}</b><br><br>
                <table style='font-size:13px;border-collapse:collapse;width:100%;'>
                <tr style='background:#f5f5f5'><td style='padding:4px'><b>NDWI</b></td><td style='padding:4px'>{row['ndwi']:.3f}</td><td style='padding:4px;color:#999;font-size:11px'>גבוה=נקי</td></tr>
                {'<tr><td style="padding:4px"><b>כלורופיל</b></td><td style="padding:4px">' + f"{row['chl_proxy']:.3f}" + '</td><td style="padding:4px;color:#999;font-size:11px">גבוה=אצות</td></tr>' if row.get('chl_proxy') is not None else ''}
                <tr style='background:#f5f5f5'><td style='padding:4px'><b>עכירות</b></td><td style='padding:4px'>{f"{row['turbidity']:.0f}" if row.get('turbidity') is not None else "N/A"}</td><td style='padding:4px;color:#999;font-size:11px'>גבוה=עכור</td></tr>
                {fai_row}
                </table>
                <br><small style='color:#aaa'>📅 {image_date}</small></div>"""

        folium.CircleMarker(
            location=[row["lat"], row["lon"]], radius=14,
            color="white", weight=2, fill=True, fill_color=color, fill_opacity=0.92,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{i}. {row['name']} — {row['quality_label']}"
        ).add_to(points_group)

        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.DivIcon(
                html=f"<div style='font-size:12px;font-weight:bold;color:white;text-align:center;line-height:28px;width:28px;margin-left:-14px;margin-top:-14px;'>{i}</div>",
                icon_size=(28,28), icon_anchor=(14,14))
        ).add_to(points_group)

        if sensor == "S1":
            verbal_labels = {1: "נקי", 2: "חשד קל", 3: "חשד זיהום", 4: "זיהום סביר", 5: "זיהום חמור"}
        else:
            verbal_labels = {1: "מצוין", 2: "טוב", 3: "בינוני", 4: "ירוד", 5: "גרוע"}
        verbal = verbal_labels.get(row["quality_score"], "אין מידע") if not row["no_data"] else "אין מידע"
        verbal_color = color if not row["no_data"] else "#AAAAAA"
        label_html = (
            f"<div style='white-space:nowrap;text-align:right;direction:rtl;line-height:1.3;'>"
            f"<div style='font-size:10px;color:#333;font-weight:normal;"
            f"text-shadow:0 0 3px white,0 0 3px white,0 0 3px white;'>{row['name']}</div>"
            f"<div style='font-size:15px;font-weight:bold;color:{verbal_color};"
            f"text-shadow:0 0 4px white,0 0 4px white,0 0 4px white;'>{verbal}</div>"
            f"</div>"
        )
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.DivIcon(
                html=label_html,
                icon_size=(120, 38),
                icon_anchor=(-18, 19)
            )
        ).add_to(points_group)

    # כותרת תאריך
    date_html = f"""<div style="position:fixed;top:15px;left:50%;transform:translateX(-50%);z-index:1000;
        background:rgba(0,0,0,0.7);color:white;padding:8px 18px;border-radius:20px;
        font-family:Arial;font-size:14px;font-weight:bold;direction:rtl;">
        🛰️ מבוסס על צילום לווין מתאריך: {image_date}</div>"""
    m.get_root().html.add_child(folium.Element(date_html))

    # אזהרה ספציפית לגוף המים
    if wb["note"]:
        note_html = f"""<div style="position:fixed;top:60px;left:50%;transform:translateX(-50%);z-index:1000;
            background:rgba(255,243,205,0.95);color:#856404;padding:7px 16px;border-radius:12px;
            font-family:Arial;font-size:12px;max-width:420px;text-align:right;direction:rtl;
            border:1px solid #ffc107;">
            {wb['note']}</div>"""
        m.get_root().html.add_child(folium.Element(note_html))

    # מקרא (זהה למקורי)
    legend_html = """<div style="position:fixed;bottom:30px;right:10px;z-index:1000;
        background:white;padding:14px 16px;border-radius:12px;
        box-shadow:0 2px 10px rgba(0,0,0,0.25);font-family:Arial;direction:rtl;font-size:13px;width:210px;">
        <b style='font-size:14px;'>🌊 איכות מים</b><br><br>
        <div style="width:160px;height:90px;margin:0 auto 8px;">
        <svg width="160" height="90" viewBox="0 0 160 90">
            <defs><linearGradient id="qual" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#8E44AD"/>
                <stop offset="25%" stop-color="#E74C3C"/>
                <stop offset="50%" stop-color="#E67E22"/>
                <stop offset="75%" stop-color="#F1C40F"/>
                <stop offset="100%" stop-color="#27AE60"/>
            </linearGradient>
            <clipPath id="half"><rect x="0" y="0" width="160" height="80"/></clipPath></defs>
            <circle cx="80" cy="80" r="72" fill="url(#qual)" clip-path="url(#half)"/>
            <circle cx="80" cy="80" r="48" fill="white" clip-path="url(#half)"/>
            <text x="4" y="74" font-size="10" fill="#8E44AD" font-weight="bold">גרוע</text>
            <text x="156" y="74" font-size="10" fill="#27AE60" font-weight="bold" text-anchor="end">מצוין</text>
            <text x="4" y="88" font-size="10" fill="#666">0</text>
            <text x="156" y="88" font-size="10" fill="#666" text-anchor="end">100</text>
        </svg></div>
        <div style="font-size:12px;line-height:2;">
        <span style='color:#27AE60;font-size:16px;'>●</span> מצוין (80-100)<br>
        <span style='color:#F1C40F;font-size:16px;'>●</span> טוב (60-79)<br>
        <span style='color:#E67E22;font-size:16px;'>●</span> בינוני (40-59)<br>
        <span style='color:#E74C3C;font-size:16px;'>●</span> ירוד (20-39)<br>
        <span style='color:#8E44AD;font-size:16px;'>●</span> גרוע (0-19)<br>
        <span style='color:#AAAAAA;font-size:16px;'>●</span> אין מידע</div>
        <hr style='margin:8px 0;border:none;border-top:1px solid #eee;'>
        <b style='font-size:12px;'>🌡️ מפת חום:</b><br>
        <div style="display:flex;align-items:center;margin-top:4px;">
            <div style="width:120px;height:10px;background:linear-gradient(to right,#8B0000,#E74C3C,#F1C40F,#27AE60);border-radius:3px;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#666;margin-top:2px;">
            <span>נקי</span><span>מזוהם</span>
        </div>
        <small style='color:#999;font-size:11px;'>לחץ על עיגול לפרטים</small>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    groups = [satellite_group, heatmap_group, sampling_group, points_group]
    if oil_group is not None:
        groups.insert(1, oil_group)
        # הוסף מפת זיהום SAR
        oil_url = get_heatmap_url(processed, wb["clip_geom"], sensor)
        if oil_url:
            folium.TileLayer(
                tiles=oil_url, name="מפת זיהום SAR", attr="GEE/Copernicus",
                overlay=True, opacity=0.75
            ).add_to(oil_group)
    for group in groups:
        group.add_to(m)
    folium.LayerControl().add_to(m)
    return m

# ==============================
# ממשק Streamlit
# ==============================
st.set_page_config(page_title="ניטור איכות מים — ישראל", page_icon="🌊", layout="wide")
st.title("ניטור איכות מי ים בחופים בישראל")

# ── בחירת גוף מים ────────────────────────────────────────────────────────────
wb_key = st.radio(
    "בחר גוף מים",
    options=list(WATER_BODIES.keys()),
    horizontal=True,
    key="wb_selector"
)
wb = WATER_BODIES[wb_key]

# ── בחירת סנסור ──────────────────────────────────────────────────────────────
sensor = st.radio(
    "סנסור לוויין",
    options=[
        "S2 — Sentinel-2 (10m, אופטי)",
        "S3 — Sentinel-3 (300m, ימי)",
        "S1 — Sentinel-1 SAR (10m, זיהום שמן)",
    ],
    horizontal=True,
    key="sensor_selector"
)
if sensor.startswith("S3"):
    sensor_key = "S3"
elif sensor.startswith("S1"):
    sensor_key = "S1"
else:
    sensor_key = "S2"

# הצג הסבר קצר על S1 אם נבחר
if sensor_key == "S1":
    st.info(
        "🛢️ **Sentinel-1 SAR** — רדאר (לא מושפע מענן/לילה). "
        "מזהה שמן/זיהום לפי *dark spots* ב-VV backscatter. "
        "ציון גבוה = מים נקיים, ציון נמוך = חשד לזיהום."
    )

# ── בחירת טווח תאריכים ───────────────────────────────────────────────────────
with st.expander("📅 בחר טווח תאריכים", expanded=False):
    today        = datetime.now().date()
    min_date     = today - timedelta(days=365 * 3)
    max_date     = today
    # S1 חוזר כל 6 ימים — טווח קצר מספיק; S2/S3 כרגיל
    default_days = wb.get("days_back_s1", 30) if sensor_key == "S1" else wb["days_back"]
    default_end  = today
    default_start = today - timedelta(days=default_days)

    col_cal1, col_cal2 = st.columns(2)
    with col_cal1:
        sel_start = st.date_input(
            "תאריך התחלה",
            value=default_start,
            min_value=min_date,
            max_value=max_date,
            key=f"date_start_{wb_key}",
        )
    with col_cal2:
        sel_end = st.date_input(
            "תאריך סיום",
            value=default_end,
            min_value=min_date,
            max_value=max_date,
            key=f"date_end_{wb_key}",
        )

    # סליידר — מזיז את חלון הזמן תוך שמירה על הרוחב
    st.markdown("**🎚️ הזזת חלון הזמן**")
    total_days  = (max_date - min_date).days
    window_days = max(1, (sel_end - sel_start).days)
    slider_default = (sel_end - min_date).days

    slider_val = st.slider(
        "הזז את חלון הזמן לאחור / קדימה",
        min_value=window_days,
        max_value=total_days,
        value=slider_default,
        step=1,
        key=f"date_slider_{wb_key}",
        help="הסליידר שומר על רוחב הטווח שנבחר ומזיז אותו לאורך הציר הזמני",
    )

    slider_end   = min_date + timedelta(days=slider_val)
    slider_start = slider_end - timedelta(days=window_days)

    # סליידר גובר אם שונה מהקלנדר
    if slider_end != sel_end:
        sel_start = slider_start
        sel_end   = slider_end

    if sel_start >= sel_end:
        st.error("⚠️ תאריך ההתחלה חייב להיות לפני תאריך הסיום")
        st.stop()

    st.info(f"📆 טווח נבחר: **{sel_start.strftime('%d/%m/%Y')}** עד **{sel_end.strftime('%d/%m/%Y')}** ({window_days} ימים)")

start_date = sel_start.strftime("%Y-%m-%d")
end_date   = sel_end.strftime("%Y-%m-%d")

# ── טעינת נתונים ─────────────────────────────────────────────────────────────
sensor_label = "Sentinel-3 OLCI" if sensor_key == "S3" else "Sentinel-1 SAR" if sensor_key == "S1" else "Sentinel-2"
with st.spinner(f"🛰️ טוען נתוני {sensor_label} עבור {wb_key}..."):
    df, image_date, processed, scene_count, water_polygons = load_data(wb_key, start_date, end_date, sensor_key)

if df is None:
    st.error(
        f"לא נמצאו תמונות {sensor_label} בטווח {sel_start.strftime('%d/%m/%Y')} – {sel_end.strftime('%d/%m/%Y')} עבור {wb_key}. "
        f"נסה להרחיב את טווח התאריכים."
    )
    st.stop()

# ── מפה + טבלה ───────────────────────────────────────────────────────────────
with st.spinner("🌡️ מחשב מפת חום..."):
    m = build_map(df, image_date, processed, wb_key, water_polygons, sensor_key)


# ── מפה (full width) ─────────────────────────────────────────────────────────
map_data = st_folium(m, width="100%", height=680)

# ── sidebar: פאנל ניתוח חכם ─────────────────────────────────────────────────
def _score_color(s):
    if s is None:    return "#AAAAAA", "אין מידע"
    if s >= 80:      return "#27AE60", "מצוין 🟢"
    if s >= 60:      return "#F1C40F", "טוב 🟡"
    if s >= 40:      return "#E67E22", "בינוני 🟠"
    if s >= 20:      return "#E74C3C", "ירוד 🔴"
    return           "#8E44AD",        "גרוע ⛔"

def _anomaly_reasons(r):
    reasons = []
    if r.get("fai")       is not None and r["fai"]       > 0.02:  reasons.append("אצות צפות")
    if r.get("chl_proxy") is not None and r["chl_proxy"] > 1.5:   reasons.append("כלורופיל גבוה")
    if r.get("turbidity") is not None and r["turbidity"] > 500:   reasons.append("עכירות גבוהה")
    if r.get("ndwi")      is not None and r["ndwi"]      < 0.05:  reasons.append("מגע יבשה/סחף")
    return reasons if reasons else ["ערכים נמוכים"]

def _card(bg, border_color, content_html):
    return (
        f"<div style='background:{bg};border-radius:10px;padding:12px 14px;"
        f"margin-bottom:10px;direction:rtl;font-family:Arial;"
        f"border-right:4px solid {border_color};'>{content_html}</div>"
    )

def _legend_card(zoom_level, data_row=None):
    """
    מקרא תמציתי: שקיפות / אצות / סכנה בריאותית.
    data_row — שורת pandas עם ndwi, chl_proxy, fai, turbidity (או None לתצוגה ארצית/אזורית כללית).
    מחזיר HTML string.
    """

    # ── פרשנות שקיפות (NDWI / Turbidity) ─────────────────────────────────────
    if data_row is not None:
        ndwi      = data_row.get("ndwi")
        turbidity = data_row.get("turbidity")
        chl       = data_row.get("chl_proxy")
        fai       = data_row.get("fai")
    else:
        ndwi = turbidity = chl = fai = None

    # שקיפות
    if ndwi is not None and turbidity is not None:
        if ndwi > 0.25 and turbidity < 200:
            transp_color, transp_icon, transp_text = "#27AE60", "💧", "מים שקופים — נראות טובה"
        elif ndwi > 0.1 or turbidity < 500:
            transp_color, transp_icon, transp_text = "#F1C40F", "🌊", "עכירות בינונית — ראות חלקית"
        else:
            transp_color, transp_icon, transp_text = "#E74C3C", "🟤", "מים עכורים — ראות ירודה"
    else:
        transp_color, transp_icon, transp_text = "#AAAAAA", "❓", "אין מידע על שקיפות"

    # אצות
    if chl is not None and fai is not None:
        if fai > 0.02 or chl > 1.8:
            algae_color, algae_icon, algae_text = "#E74C3C", "🦠", "ריכוז אצות גבוה — פריחה אפשרית"
        elif fai > 0.005 or chl > 1.3:
            algae_color, algae_icon, algae_text = "#E67E22", "🌿", "אצות מתונות — תנאים רגילים"
        else:
            algae_color, algae_icon, algae_text = "#27AE60", "✅", "כמות אצות נמוכה — תקין"
    elif chl is not None:
        if chl > 1.8:
            algae_color, algae_icon, algae_text = "#E74C3C", "🦠", "כלורופיל גבוה — אצות סבירות"
        elif chl > 1.3:
            algae_color, algae_icon, algae_text = "#E67E22", "🌿", "כלורופיל מתון"
        else:
            algae_color, algae_icon, algae_text = "#27AE60", "✅", "כלורופיל תקין"
    else:
        algae_color, algae_icon, algae_text = "#AAAAAA", "❓", "אין מידע על אצות"

    # סכנה בריאותית — מבוסס על שילוב FAI + Chl
    if fai is not None and chl is not None:
        if fai > 0.02 and chl > 1.8:
            health_color, health_icon, health_text = "#8E44AD", "⛔", "סכנה — אל תיכנס למים"
        elif fai > 0.02 or chl > 1.8:
            health_color, health_icon, health_text = "#E74C3C", "⚠️", "זהירות — אצות מזיקות אפשריות"
        elif chl > 1.3:
            health_color, health_icon, health_text = "#E67E22", "🟠", "נוח לרחצה — מעקב מומלץ"
        else:
            health_color, health_icon, health_text = "#27AE60", "🏊", "בטוח לרחצה"
    else:
        health_color, health_icon, health_text = "#AAAAAA", "❓", "לא ניתן להעריך"

    zoom_label = {
        "national": "ממוצע ארצי",
        "regional": "ממוצע אזורי",
        "local":    "נקודה זו",
    }.get(zoom_level, "")

    rows = (
        f"<tr>"
        f"<td style='padding:5px 3px;font-size:17px;'>{transp_icon}</td>"
        f"<td style='padding:5px 3px;font-weight:bold;font-size:12px;color:#444;'>שקיפות</td>"
        f"<td style='padding:5px 3px;font-size:12px;color:{transp_color};font-weight:bold;'>{transp_text}</td>"
        f"</tr>"
        f"<tr style='background:#f7f7f7;'>"
        f"<td style='padding:5px 3px;font-size:17px;'>{algae_icon}</td>"
        f"<td style='padding:5px 3px;font-weight:bold;font-size:12px;color:#444;'>אצות</td>"
        f"<td style='padding:5px 3px;font-size:12px;color:{algae_color};font-weight:bold;'>{algae_text}</td>"
        f"</tr>"
        f"<tr>"
        f"<td style='padding:5px 3px;font-size:17px;'>{health_icon}</td>"
        f"<td style='padding:5px 3px;font-weight:bold;font-size:12px;color:#444;'>רחצה</td>"
        f"<td style='padding:5px 3px;font-size:12px;color:{health_color};font-weight:bold;'>{health_text}</td>"
        f"</tr>"
    )

    return (
        f"<div style='background:#fafafa;border-radius:10px;padding:10px 12px;"
        f"margin-top:6px;margin-bottom:10px;direction:rtl;font-family:Arial;"
        f"border:1px solid #e0e0e0;'>"
        f"<div style='font-size:11px;color:#999;margin-bottom:6px;'>📋 מקרא מצב — {zoom_label}</div>"
        f"<table style='width:100%;border-collapse:collapse;'>{rows}</table>"
        f"</div>"
    )

valid_df   = df[df["no_data"] == False].copy()
has_data   = len(valid_df) > 0
anomalies  = valid_df[valid_df["composite"] < 50] if has_data else pd.DataFrame()
avg_score  = valid_df["composite"].mean() if has_data else None
avg_color, avg_label = _score_color(avg_score)

# ── נשלוף זום מ-map_data (אחרי st_folium) ────────────────────────────────────
# ברירת מחדל = זום ההתחלתי של גוף המים הנוכחי
default_zoom = WATER_BODIES[wb_key]["zoom"]
current_zoom = default_zoom
if map_data and map_data.get("zoom"):
    current_zoom = map_data["zoom"]

if current_zoom <= 8:
    zoom_level = "national"    # כל ישראל
elif current_zoom <= 11:
    zoom_level = "regional"    # אזור
else:
    zoom_level = "local"       # נקודה בודדת

with st.sidebar:
    st.markdown(
        f"<div style='direction:rtl;font-family:Arial;'>"
        f"<b style='font-size:16px;'>🌊 ניתוח מצב המים</b><br>"
        f"<span style='font-size:12px;color:#888;'>{wb_key} · {image_date}</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # רמה 1 — כל ישראל (זום ≤ 8)
    # ════════════════════════════════════════════════════════
    if zoom_level == "national":
        st.markdown("**🗺️ תצוגה ארצית**", help="זום ≤ 8")

        if has_data:
            st.markdown(
                _card(
                    "#f8f9fa", avg_color,
                    f"<div style='font-size:12px;color:#666;'>ממוצע כלל החופים</div>"
                    f"<div style='font-size:24px;font-weight:bold;color:{avg_color};'>{avg_label}</div>"
                    f"<div style='font-size:15px;'>{int(round(avg_score))}/100</div>"
                ),
                unsafe_allow_html=True
            )

            # פס ציון ויזואלי
            pct = int(round(avg_score))
            st.markdown(
                f"<div style='background:#eee;border-radius:6px;height:10px;margin-bottom:10px;'>"
                f"<div style='background:{avg_color};width:{pct}%;height:10px;border-radius:6px;'></div>"
                f"</div>",
                unsafe_allow_html=True
            )

            # סטטיסטיקות מהירות
            n_excellent = len(valid_df[valid_df["composite"] >= 80])
            n_good      = len(valid_df[(valid_df["composite"] >= 60) & (valid_df["composite"] < 80)])
            n_bad       = len(valid_df[valid_df["composite"] < 40])

            cols = st.columns(3)
            cols[0].metric("🟢 מצוין", n_excellent)
            cols[1].metric("🟡 טוב",   n_good)
            cols[2].metric("🔴 ירוד+",  n_bad)

            if len(anomalies) > 0:
                st.markdown("**⚠️ אזורים חריגים**")
                for _, r in anomalies.nsmallest(5, "composite").iterrows():
                    reasons = _anomaly_reasons(r)
                    sc = int(round(r["composite"])) if r["composite"] is not None else "—"
                    rc, _ = _score_color(r["composite"])
                    st.markdown(
                        _card(
                            "#fff8f8", rc,
                            f"<b>{r['name']}</b> "
                            f"<span style='color:{rc};font-weight:bold;'>({sc})</span><br>"
                            f"<span style='font-size:12px;color:#666;'>{'  ·  '.join(reasons)}</span>"
                        ),
                        unsafe_allow_html=True
                    )
            else:
                st.success("✅ אין חריגים — כל החופים מעל 50")

            # ── AI ────────────────────────────────────────────────────────
            st.markdown("---")
            ai_key = f"ai_national_{wb_key}_{start_date}_{end_date}"
            if st.button("🤖 ניתוח AI ארצי", key="ai_national"):
                anomaly_str = ", ".join([
                    f"{r['name']} ({int(round(r['composite']))}): {'|'.join(_anomaly_reasons(r))}"
                    for _, r in anomalies.iterrows()
                ]) if len(anomalies) > 0 else "אין"
                summary = (
                    f"נתוני איכות מים — {wb_key}, {image_date}\n"
                    f"ממוצע: {avg_score:.1f}/100\n"
                    f"חריגים (ציון<50): {anomaly_str}"
                )
                with st.spinner("מנתח..."):
                    import requests as _req
                    try:
                        resp = _req.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={"Content-Type": "application/json"},
                            json={
                                "model": "claude-sonnet-4-20250514",
                                "max_tokens": 400,
                                "system": "אתה מומחה לאיכות מים ימיים בישראל. ענה בעברית, 3-4 משפטים בלבד.",
                                "messages": [{"role": "user", "content":
                                    f"{summary}\n\nסכם: מה מצב חופי ישראל היום? האם יש דפוס גיאוגרפי? מה הסיבה הסבירה לחריגים?"}]
                            }, timeout=30
                        )
                        st.session_state[ai_key] = resp.json()["content"][0]["text"]
                    except Exception as e:
                        st.session_state[ai_key] = f"שגיאה: {e}"
            if st.session_state.get(ai_key):
                st.markdown(
                    _card("#f0f7ff", "#3498DB",
                          f"<span style='font-size:13px;line-height:1.6;'>{st.session_state[ai_key]}</span>"),
                    unsafe_allow_html=True
                )

            # ── מקרא מצב — ממוצע ארצי ────────────────────────────────────────
            st.markdown("---")
            # בנה שורה ממוצעת לפרשנות המקרא
            avg_row = {
                "ndwi":      valid_df["ndwi"].mean()      if "ndwi"      in valid_df else None,
                "chl_proxy": valid_df["chl_proxy"].mean() if "chl_proxy" in valid_df else None,
                "turbidity": valid_df["turbidity"].mean() if "turbidity" in valid_df else None,
                "fai":       valid_df["fai"].mean()       if "fai"       in valid_df else None,
            }
            st.markdown(_legend_card("national", avg_row), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # רמה 2 — אזורי (זום 9-11)
    # ════════════════════════════════════════════════════════
    elif zoom_level == "regional":
        st.markdown("**🔍 תצוגה אזורית**", help="זום 9-11")

        if has_data:
            # מרכז המפה הנוכחי → נקודות בטווח
            map_center = map_data.get("center") if map_data else None
            if map_center:
                clat_c, clon_c = map_center["lat"], map_center["lng"]
                # מרחק פשוט בדרגות (~50km בזום אזורי)
                radius_deg = 0.5
                nearby = valid_df[
                    (valid_df["lat"].between(clat_c - radius_deg, clat_c + radius_deg)) &
                    (valid_df["lon"].between(clon_c - radius_deg, clon_c + radius_deg))
                ]
                if len(nearby) == 0:
                    nearby = valid_df  # fallback
            else:
                nearby = valid_df

            nearby_avg = nearby["composite"].mean()
            nc, nl = _score_color(nearby_avg)
            st.markdown(
                _card("#f8f9fa", nc,
                    f"<div style='font-size:12px;color:#666;'>ממוצע באזור הנוכחי</div>"
                    f"<div style='font-size:22px;font-weight:bold;color:{nc};'>{nl}</div>"
                    f"<div style='font-size:14px;'>{int(round(nearby_avg))}/100 · {len(nearby)} נקודות</div>"
                ),
                unsafe_allow_html=True
            )

            st.markdown("**📍 נקודות באזור**")
            for _, r in nearby.sort_values("composite").iterrows():
                sc = int(round(r["composite"])) if r["composite"] is not None else "—"
                rc, rl = _score_color(r["composite"])
                reasons = _anomaly_reasons(r) if r["composite"] < 50 else []
                reason_str = f"<br><span style='font-size:11px;color:#888;'>{'  ·  '.join(reasons)}</span>" if reasons else ""
                st.markdown(
                    _card(
                        "#f8f9fa", rc,
                        f"<b>{r['name']}</b> — <span style='color:{rc};font-weight:bold;'>{rl.split()[0]} ({sc})</span>"
                        f"{reason_str}"
                    ),
                    unsafe_allow_html=True
                )

            # ── AI ────────────────────────────────────────────────────────
            st.markdown("---")
            ai_key = f"ai_regional_{wb_key}_{start_date}_{end_date}_{int(clat_c*10) if map_center else 0}"
            if st.button("🤖 ניתוח AI אזורי", key="ai_regional"):
                summary = "\n".join([
                    f"- {r['name']}: {int(round(r['composite']))}, FAI={r.get('fai','N/A')}, "
                    f"כלורופיל={r.get('chl_proxy','N/A')}, עכירות={r.get('turbidity','N/A')}"
                    for _, r in nearby.iterrows()
                ])
                with st.spinner("מנתח..."):
                    import requests as _req
                    try:
                        resp = _req.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={"Content-Type": "application/json"},
                            json={
                                "model": "claude-sonnet-4-20250514",
                                "max_tokens": 400,
                                "system": "אתה מומחה לאיכות מים ימיים בישראל. ענה בעברית, 3-4 משפטים.",
                                "messages": [{"role": "user", "content":
                                    f"נתוני האזור ({image_date}):\n{summary}\n\n"
                                    f"מה מצב האזור? מה הסיבה לחריגים? האם יש מקורות זיהום סבירים?"}]
                            }, timeout=30
                        )
                        st.session_state[ai_key] = resp.json()["content"][0]["text"]
                    except Exception as e:
                        st.session_state[ai_key] = f"שגיאה: {e}"
            if st.session_state.get(ai_key):
                st.markdown(
                    _card("#f0f7ff", "#3498DB",
                          f"<span style='font-size:13px;line-height:1.6;'>{st.session_state[ai_key]}</span>"),
                    unsafe_allow_html=True
                )

            # ── מקרא מצב — ממוצע אזורי ───────────────────────────────────────
            st.markdown("---")
            nearby_row = {
                "ndwi":      nearby["ndwi"].mean()      if "ndwi"      in nearby else None,
                "chl_proxy": nearby["chl_proxy"].mean() if "chl_proxy" in nearby else None,
                "turbidity": nearby["turbidity"].mean() if "turbidity" in nearby else None,
                "fai":       nearby["fai"].mean()       if "fai"       in nearby else None,
            }
            st.markdown(_legend_card("regional", nearby_row), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # רמה 3 — נקודה בודדת (זום ≥ 12)
    # ════════════════════════════════════════════════════════
    else:
        st.markdown("**📌 תצוגת נקודה**", help="זום ≥ 12")

        # מצא נקודה הכי קרובה למרכז המפה
        map_center = map_data.get("center") if map_data else None
        if map_center and has_data:
            clat_c, clon_c = map_center["lat"], map_center["lng"]
            valid_df["_dist"] = ((valid_df["lat"] - clat_c)**2 + (valid_df["lon"] - clon_c)**2)**0.5
            nearest = valid_df.loc[valid_df["_dist"].idxmin()]
            rc, rl = _score_color(nearest["composite"])
            sc = int(round(nearest["composite"])) if nearest["composite"] is not None else "—"

            st.markdown(
                _card("#f8f9fa", rc,
                    f"<div style='font-size:13px;color:#666;'>נקודה קרובה למרכז המסך</div>"
                    f"<div style='font-size:20px;font-weight:bold;'>{nearest['name']}</div>"
                    f"<div style='font-size:22px;font-weight:bold;color:{rc};'>{rl} · {sc}/100</div>"
                ),
                unsafe_allow_html=True
            )

            # מדדים מפורטים
            if sensor_key == "S1":
                metrics = [
                    ("VV (dB)",      nearest.get("vv"),          "גבוה=נקי, נמוך=dark spot", "{:.2f}"),
                    ("VH (dB)",      nearest.get("vh"),          "cross-pol reference",        "{:.2f}"),
                    ("VV/VH ratio",  nearest.get("vv_vh_ratio"), "נמוך<1.5 = surface damping", "{:.3f}"),
                    ("Roughness",    nearest.get("roughness"),    "חספוס פני הים",              "{:.3f}"),
                ]
            else:
                metrics = [
                    ("NDWI",       nearest.get("ndwi"),      "גבוה = נקי",      "{:.3f}"),
                    ("כלורופיל",   nearest.get("chl_proxy"), "גבוה = אצות",     "{:.3f}"),
                    ("עכירות",     nearest.get("turbidity"),  "גבוה = עכור",     "{:.0f}"),
                    ("FAI",        nearest.get("fai"),        "גבוה = אצות צפות","{:.4f}"),
                ]
            rows = ""
            for name, val, hint, fmt in metrics:
                val_str = fmt.format(val) if val is not None else "—"
                rows += (
                    f"<tr>"
                    f"<td style='padding:5px 4px;font-weight:bold;'>{name}</td>"
                    f"<td style='padding:5px 4px;'>{val_str}</td>"
                    f"<td style='padding:5px 4px;color:#999;font-size:11px;'>{hint}</td>"
                    f"</tr>"
                )
            st.markdown(
                f"<table style='width:100%;font-size:13px;border-collapse:collapse;"
                f"direction:rtl;font-family:Arial;'>{rows}</table>",
                unsafe_allow_html=True
            )

            reasons = _anomaly_reasons(nearest) if nearest["composite"] < 50 else []
            if reasons:
                st.markdown(
                    _card("#fff8f8", "#E74C3C",
                          f"<b>⚠️ סיבות אפשריות:</b><br>"
                          f"<span style='font-size:13px;'>{'<br>'.join(['• ' + r for r in reasons])}</span>"),
                    unsafe_allow_html=True
                )

            # ── מקרא מצב — נקודה ─────────────────────────────────────────────
            st.markdown(_legend_card("local", nearest), unsafe_allow_html=True)

            # ── AI ────────────────────────────────────────────────────────
            st.markdown("---")
            ai_key = f"ai_local_{wb_key}_{nearest['name']}_{start_date}"
            if st.button("🤖 ניתוח AI לנקודה", key="ai_local"):
                with st.spinner("מנתח..."):
                    import requests as _req
                    try:
                        resp = _req.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={"Content-Type": "application/json"},
                            json={
                                "model": "claude-sonnet-4-20250514",
                                "max_tokens": 400,
                                "system": "אתה מומחה לאיכות מים ימיים בישראל. ענה בעברית, 3-4 משפטים.",
                                "messages": [{"role": "user", "content":
                                    f"נקודה: {nearest['name']} ({image_date})\n"
                                    f"ציון: {sc}/100\n"
                                    f"NDWI={nearest.get('ndwi','N/A')}, כלורופיל={nearest.get('chl_proxy','N/A')}, "
                                    f"עכירות={nearest.get('turbidity','N/A')}, FAI={nearest.get('fai','N/A')}\n\n"
                                    f"מה מצב הים בנקודה זו? מה הסיבה הסבירה לערכים? האם בטוח לרחצה?"}]
                            }, timeout=30
                        )
                        st.session_state[ai_key] = resp.json()["content"][0]["text"]
                    except Exception as e:
                        st.session_state[ai_key] = f"שגיאה: {e}"
            if st.session_state.get(ai_key):
                st.markdown(
                    _card("#f0f7ff", "#3498DB",
                          f"<span style='font-size:13px;line-height:1.6;'>{st.session_state[ai_key]}</span>"),
                    unsafe_allow_html=True
                )
        else:
            st.info("הזז את המפה לאזור שברצונך לבדוק")

    st.markdown("---")
    st.caption(f"🔍 זום נוכחי: {current_zoom} · {zoom_level}")


# ── לחיצה חופשית על המפה → שאילתת GEE ──────────────────────────────────────
clicked = map_data.get("last_clicked") if map_data else None

if clicked:
    clat = clicked["lat"]
    clon = clicked["lng"]

    # בדוק אם זו לחיצה חדשה (אחרת לא מחשב שוב)
    prev = st.session_state.get("last_click_coords")
    if prev != (clat, clon):
        st.session_state["last_click_coords"] = (clat, clon)
        st.session_state["click_result"] = None   # מנקה תוצאה ישנה

    st.divider()
    st.markdown(f"### 📍 שאילתה חופשית — `{clat:.5f}, {clon:.5f}`")

    # כפתור חישוב — משתמש לוחץ כדי לא לחשב בכל רענון
    if st.button("🛰️ חשב ערכי לוויין בנקודה זו", type="primary"):
        with st.spinner("שולח שאילתה ל-Google Earth Engine..."):
            result = query_clicked_point(
                clat, clon,
                wb_key, start_date, end_date, sensor_key
            )
        st.session_state["click_result"] = result

    # תצוגת תוצאה
    result = st.session_state.get("click_result")
    if result:
        if "error" in result:
            st.warning(result["error"])
        else:
            color = result["quality_color"]
            comp  = f"{int(round(result['composite']))}/100" if result["composite"] is not None else "N/A"

            # כרטיס תוצאה
            st.markdown(
                f"""<div style="background:#f8f9fa;border-radius:12px;padding:16px 20px;
                    border-right:5px solid {color};direction:rtl;font-family:Arial;">
                <b style="font-size:17px;">{result['quality_label']}</b>
                <span style="font-size:22px;font-weight:bold;margin-right:12px;">⭐ {comp}</span>
                <hr style="margin:10px 0;border:none;border-top:1px solid #ddd;">
                <table style="width:100%;font-size:14px;border-collapse:collapse;">
                <tr><td style="padding:5px 0;"><b>NDWI</b></td>
                    <td style="padding:5px 0;">{result['ndwi'] if result['ndwi'] is not None else 'N/A'}</td>
                    <td style="color:#888;font-size:12px;">גבוה = נקי יותר</td></tr>
                <tr style="background:#f0f0f0;"><td style="padding:5px 4px;"><b>כלורופיל</b></td>
                    <td style="padding:5px 4px;">{result['chl_proxy'] if result['chl_proxy'] is not None else 'N/A'}</td>
                    <td style="color:#888;font-size:12px;">גבוה = אצות</td></tr>
                <tr><td style="padding:5px 0;"><b>עכירות</b></td>
                    <td style="padding:5px 0;">{result['turbidity'] if result['turbidity'] is not None else 'N/A'}</td>
                    <td style="color:#888;font-size:12px;">גבוה = עכור</td></tr>
                <tr style="background:#f0f0f0;"><td style="padding:5px 4px;"><b>FAI</b></td>
                    <td style="padding:5px 4px;">{result['fai'] if result['fai'] is not None else 'N/A'}</td>
                    <td style="color:#888;font-size:12px;">גבוה = אצות צפות</td></tr>
                </table>
                </div>""",
                unsafe_allow_html=True
            )

