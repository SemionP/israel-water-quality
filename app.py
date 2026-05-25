"""
app.py
=============================================================================
Israel Water Quality Monitor Dashboard - Sentinel-3 Exclusive Edition
גרסה ממוקדת: חישוב ערך משוכלל (WQI) על בסיס לוויין Sentinel-3 OLCI בלבד.
חיתוך קשיח למים טריטוריאליים והצגת תאריכי מעבר זמינים של S3.
=============================================================================
"""

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

# קונפיגורציה
st.set_page_config(page_title="Israel Water Quality Monitor - Sentinel-3 WQI", layout="wide")

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
# גיאומטריות וחיתוך קשיח למים טריטוריאליים בלבד
# =============================================================================
HAIFA_CENTER = [32.4, 34.85]
HAIFA_BBOX   = ee.Geometry.Rectangle([34.20, 31.20, 35.20, 33.20])

# פוליגון המים הטריטוריאליים הרשמי של מדינת ישראל בים התיכון
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
    {"name": "ראש הנקרה", "lat": 33.0765, "lon": 35.0983}, {"name": "נהריה", "lat": 33.0048, "lon": 35.0832},
    {"name": "עכו", "lat": 32.9280, "lon": 35.0680}, {"name": "חיפה צפון", "lat": 32.8380, "lon": 34.9820},
    {"name": "עתלית", "lat": 32.6892, "lon": 34.9368}, {"name": "קיסריה", "lat": 32.4948, "lon": 34.8912},
    {"name": "נתניה", "lat": 32.3318, "lon": 34.8512}, {"name": "הרצליה", "lat": 32.1648, "lon": 34.7962},
    {"name": "תל אביב מרכז", "lat": 32.0798, "lon": 34.7618}, {"name": "אשדוד", "lat": 31.7848, "lon": 34.6248},
    {"name": "אשקלון", "lat": 31.6548, "lon": 34.5448}, {"name": "זיקים", "lat": 31.6098, "lon": 34.5198}
]

WATER_BODIES = {
    "🏖️ חוף הים התיכון": {"center": HAIFA_CENTER, "zoom": 8, "bbox": HAIFA_BBOX, "clip_geom": ISRAEL_TERRITORIAL, "points": BEACHES},
    "🌊 כנרת": {"center": [32.82, 35.59], "zoom": 12, "bbox": KINNERET_BBOX, "clip_geom": KINNERET_BBOX, "points": [{"name": "טבריה", "lat": 32.794, "lon": 35.534}, {"name": "צפון הכנרת", "lat": 32.920, "lon": 35.595}]},
    "🧂 ים המלח": {"center": [31.50, 35.47], "zoom": 11, "bbox": DEAD_SEA_BBOX, "clip_geom": DEAD_SEA_BBOX, "points": [{"name": "עין גדי", "lat": 31.462, "lon": 35.388}, {"name": "עין בוקק", "lat": 31.198, "lon": 35.352}]},
    "🐠 ים סוף": {"center": [29.55, 34.95], "zoom": 13, "bbox": RED_SEA_BBOX, "clip_geom": RED_SEA_BBOX, "points": [{"name": "מפרץ אילת", "lat": 29.530, "lon": 34.951}]}
}

# =============================================================================
# מנוע תאריכים זמינים מתוך ארכיון Sentinel-3
# =============================================================================
@st.cache_data(ttl=14400)
def get_available_s3_dates(wb_key: str, days_back: int = 30):
    wb = WATER_BODIES[wb_key]
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)
    
    # שאילתה ישירות לקטלוג Sentinel-3 OLCI
    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
            .filterBounds(wb["bbox"])
            .filterDate(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    
    dates_list = coll.aggregate_array("system:time_start").getInfo()
    unique_dates = sorted(list(set([datetime.utcfromtimestamp(d / 1000).strftime("%Y-%m-%d") for d in dates_list])), reverse=True)
    return unique_dates

# =============================================================================
# מנוע חישוב ערך משוכלל (WQI) המותאם ל-Sentinel-3 OLCI
# =============================================================================
def process_s3_wqi_layer(wb_key, target_date_str):
    wb = WATER_BODIES[wb_key]
    t_date = ee.Date(target_date_str)
    start_window = t_date.advance(-1, 'day')
    end_window = t_date.advance(1, 'day')
    
    # מסכת מים קבועה (חיונית מאוד לרזולוציה של 300 מטר למניעת זליגת חוף יבשתי)
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    water_mask = gsw.select("occurrence").gte(25)
    
    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
            .filterBounds(wb["bbox"])
            .filterDate(start_window, end_window))
            
    if coll.size().getInfo() == 0: 
        return None, "לא נמצאה סריקת Sentinel-3 עבור תאריך זה."
        
    img = coll.median().clip(wb["clip_geom"]).updateMask(water_mask)
    
    # 1. מדד מים מבוסס S3 (Oa06 = Green 560nm, Oa17 = NIR 865nm)
    s3_ndwi = img.normalizedDifference(['Oa06_radiance', 'Oa17_radiance']).rename('S3_NDWI')
    
    # 2. מדד כלורופיל ימי MCI (Maximum Chlorophyll Index) מבוסס ערוצי אדום/קצה-אדום של S3
    b10 = img.select('Oa10_radiance') # 681.25 nm
    b11 = img.select('Oa11_radiance') # 708.75 nm
    b12 = img.select('Oa12_radiance') # 753.75 nm
    mci = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75 - 681.25) / (753.75 - 681.25)))).rename('MCI')
    
    # 3. מדד עכירות מבוסס ערוץ אדום ראשי (Oa08 = Red 665nm)
    turbidity = img.select('Oa08_radiance').rename('S3_Turb')
    
    # נרמול המדדים לסקאלה אחידה של 0-1 בהתאם לתכונות הספקטרליות של OLCI
    ndwi_norm = s3_ndwi.unitScale(-0.2, 0.5).clamp(0, 1)
    mci_norm  = ee.Image(1).subtract(mci.unitScale(-2, 12)).clamp(0, 1)
    turb_norm = ee.Image(1).subtract(turbidity.unitScale(10, 80)).clamp(0, 1)
    
    # שקלול הציון המשוכלל הסופי (Water Quality Index)
    composite_wqi = ndwi_norm.add(mci_norm).add(turb_norm).divide(3).multiply(100)
    
    vis_params = {'min': 40, 'max': 85, 'palette': ['#FF0000', '#FFFF00', '#00FF00']}
    return composite_wqi, vis_params, None

# =============================================================================
# מקרא צבעים קבוע למדד המשוכלל של Sentinel-3
# =============================================================================
def render_wqi_legend():
    title = "מפתח צבעים: מדד איכות מים משולב מבוסס Sentinel-3 OLCI (WQI)"
    colors = [
        ("#FF0000", "איכות מים נמוכה / פריחת אצות חריפה או עכירות חוף גבוהה"), 
        ("#FFFF00", "איכות מים בינונית / ערכים אוקיינוגרפיים מעורבים"), 
        ("#00FF00", "איכות מים מעולה / מים נקיים וצלולים לחלוטין")
    ]
    legend_items = "".join([f'<div style="display: flex; align-items: center; margin-left: 20px;"><div style="width: 24px; height: 14px; background: {c[0]}; border: 1px solid #999; margin-left: 6px; border-radius: 2px;"></div><span style="font-size: 13px; color: #333;">{c[1]}</span></div>' for c in colors])
    st.markdown(f'<div style="background: #F8F9FA; border: 1px solid #E0E0E0; border-radius: 8px; padding: 12px; margin-top: 10px; direction: rtl;"><div style="font-weight: bold; font-size: 14px; margin-bottom: 8px; color: #222;">{title}</div><div style="display: flex; flex-wrap: wrap;">{legend_items}</div></div>', unsafe_allow_html=True)

# =============================================================================
# ממשק המשתמש (UI Layout)
# =============================================================================
st.title("🛰️ מערכת לווינית ייעודית: Sentinel-3 Water Quality Monitor")
st.markdown("ניטור וחישוב הציון המשוכלל (WQI) על בסיס הסנסור האוקיינוגרפי OLCI של לוויין Sentinel-3, חתוך קשיח למים טריטוריאליים בלבד.")

st.sidebar.header("🔧 הגדרות ותאריכים (Sentinel-3)")
wb_selection = st.sidebar.selectbox("בחר גוף מים לניטור:", list(WATER_BODIES.keys()))

# שליפת תאריכי מעבר אמיתיים של Sentinel-3 מהארכיון
with st.spinner("מאתר תאריכי מעבר זמינים של Sentinel-3..."):
    available_dates = get_available_s3_dates(wb_selection)

if available_dates:
    formatted_options = [f"🟢 {d}" for d in available_dates]
    date_selection_raw = st.sidebar.selectbox("בחר תאריך מעבר של הלוויין:", formatted_options)
    selected_date_str = date_selection_raw.replace("🟢 ", "")
else:
    st.sidebar.warning("לא נמצאו סריקות בארכיון ל-30 הימים האחרונים, מציג תאריך ברירת מחדל.")
    selected_date_str = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')

# הרצת העיבוד של Sentinel-3
with st.spinner("מחלץ ומעבד ערוצים אוקיינוגרפיים ב-Earth Engine..."):
    wqi_layer, vis_params, error_msg = process_s3_wqi_layer(wb_selection, selected_date_str)

if error_msg:
    st.error(error_msg)
elif wqi_layer:
    col_map, col_info = st.columns([2.5, 1])
    
    with col_map:
        st.subheader(f"מפת הציון המשוכלל (WQI): {wb_selection}")
        st.caption(f"סנסור: Sentinel-3 OLCI | תאריך ניתוח: {selected_date_str} | **גבולות מים טריטוריאליים**")
        
        # בניית מפת Folium
        m = folium.Map(location=WATER_BODIES[wb_selection]["center"], zoom_start=WATER_BODIES[wb_selection]["zoom"])
        
        # הזרקת שכבת ה-WQI החתוכה מ-GEE למפה
        map_id_dict = ee.Image(wqi_layer).getMapId(vis_params)
        folium.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Google Earth Engine Sentinel-3 OLCI',
            name="Sentinel-3 WQI",
            overlay=True,
            control=False,
            opacity=0.85
        ).add_to(m)
        
        # הוספת נקודות תחנות הדיגום הקבועות
        for pt in WATER_BODIES[wb_selection]["points"]:
            folium.Marker(
                location=[pt["lat"], pt["lon"]],
                popup=pt["name"],
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
            
        st_folium(m, width=850, height=550, key="s3_wqi_map")
        
        # הצגת המקרא
        render_wqi_legend()
        
    with col_info:
        st.subheader("📊 מאפייני הסנסור הימי")
        st.info("לוויין Sentinel-3 מצויד במצלמת OLCI הכוללת 21 ערוצים ספקטרליים צרים המותאמים במיוחד לזיהוי שינויי צבע מים, עכירות אוקיינוגרפית ופריחות אצות.")
        st.markdown(f"""
        - **סנסור בסיס:** Sentinel-3 OLCI (Radiances)
        - **רזולוציה מרחבית:** 300 מטר (רחב)
        - **מדד כלורופיל מובנה:** MCI (Maximum Chlorophyll Index)
        - **סינון יבשתי קשיח:** פעיל (מוגבל למים טריטוריאליים רשמיים ומסכת JRC לניקוי קצוות חוף)
        """)
