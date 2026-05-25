"""
app.py
=============================================================================
Israel Water Quality Monitor Dashboard - Sentinel-3 Pure Map Edition
גרסה מעודכנת: הסרה מוחלטת של Tooltips ו-Popups מהנקודות במפה למניעת כיתובים תקועים.
=============================================================================
"""

import json
import os
import tempfile
import requests
import pandas as pd
import streamlit as st
import folium
from branca.element import MacroElement
from jinja2 import Template
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
# רכיב חכם להזרקת מקרא צבעים רציף ישירות על גבי מפת הפוליום (On-Map Legend)
# =============================================================================
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
                div.style.direction = 'rtl';
                div.style.boxShadow = '0 0 15px rgba(0,0,0,0.2)';
                
                div.innerHTML = `
                    <div style="font-weight: bold; margin-bottom: 6px; text-align: center;">מדד איכות מים (Sentinel-3)</div>
                    <div style="display: flex; align-items: center; justify-content: space-between; font-size: 11px; font-weight: bold; margin-bottom: 3px;">
                        <span style="color: green;">מים נקיים</span>
                        <span style="color: red;">מים מזוהמים/עכורים</span>
                    </div>
                    <div style="height: 15px; width: 180px; background: linear-gradient(to left, #00FF00, #FFFF00, #FF0000); border: 1px solid #666; border-radius: 3px;"></div>
                `;
                return div;
            };
            legend.addTo({{this._parent.get_name()}});
            {% endmacro %}
        """)

# =============================================================================
# מנוע תאריכים זמינים מתוך ארכיון Sentinel-3
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
    unique_dates = sorted(list(set([datetime.utcfromtimestamp(d / 1000).strftime("%Y-%m-%d") for d in dates_list])), reverse=True)
    return unique_dates

# =============================================================================
# מנוע חישוב ערך משוכלל (WQI) עם מסנן החלקה להורדת פסים (Destriping Filter)
# =============================================================================
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
        return None, None, "לא נמצאה סריקת Sentinel-3 עבור תאריך זה."
        
    img = coll.median().clip(wb["clip_geom"]).updateMask(water_mask)
    
    s3_ndwi = img.normalizedDifference(['Oa06_radiance', 'Oa17_radiance']).rename('S3_NDWI')
    
    b10 = img.select('Oa10_radiance')
    b11 = img.select('Oa11_radiance')
    b12 = img.select('Oa12_radiance')
    mci = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75 - 681.25) / (753.75 - 681.25)))).rename('MCI')
    
    turbidity = img.select('Oa08_radiance').rename('S3_Turb')
    
    ndwi_norm = s3_ndwi.unitScale(-0.2, 0.5).clamp(0, 1)
    mci_norm  = ee.Image(1).subtract(mci.unitScale(-2, 12)).clamp(0, 1)
    turb_norm = ee.Image(1).subtract(turbidity.unitScale(10, 80)).clamp(0, 1)
    
    raw_composite_wqi = ndwi_norm.add(mci_norm).add(turb_norm).divide(3).multiply(100).rename('WQI')
    
    boxcar = ee.Kernel.square(radius=1, units='pixels')
    composite_wqi = raw_composite_wqi.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=boxcar
    ).rename('WQI').updateMask(water_mask)
    
    def get_point_wqi(pt_info):
        pt_geom = ee.Geometry.Point([pt_info["lon"], pt_info["lat"]])
        try:
            val = composite_wqi.reduceRegion(
                reducer=ee.Reducer.mean(), 
                geometry=pt_geom.buffer(450), 
                scale=300, 
                bestEffort=True
            ).getInfo()
            wqi_val = val.get('WQI')
            return {**pt_info, "wqi": round(wqi_val, 1) if wqi_val is not None else None}
        except:
            return {**pt_info, "wqi": None}

    with ThreadPoolExecutor(max_workers=4) as executor:
        sampled_points = list(executor.map(get_point_wqi, wb["points"]))
        
    df_res = pd.DataFrame(sampled_points)
    return composite_wqi, df_res, None

# =============================================================================
# ממשק המשתמש (UI Layout)
# =============================================================================
st.title("🛰️ מערכת Sentinel-3: ניטור ערך משוכלל של איכות המים")
st.markdown("תצוגה בלעדית של הציון המשוכלל (WQI), חתוך קשיח למים טריטוריאליים, כולל סקלה רציפה מובנית על המפה ורשימת החופים.")

st.sidebar.header("🔧 הגדרות ותאריכים")
wb_selection = st.sidebar.selectbox("בחר גוף מים לניטור:", list(WATER_BODIES.keys()))

with st.spinner("מאתר תאריכי מעבר זמינים של Sentinel-3..."):
    available_dates = get_available_s3_dates(wb_selection)

if available_dates:
    formatted_options = [f"🟢 {d}" for d in available_dates]
    date_selection_raw = st.sidebar.selectbox("בחר תאריך מעבר של הלוויין:", formatted_options)
    selected_date_str = date_selection_raw.replace("🟢 ", "")
else:
    st.sidebar.warning("לא נמצאו סריקות בארכיון, מציג תאריך ברירת מחדל.")
    selected_date_str = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')

# הרצת מנוע העיבוד
with st.spinner("מחשב ערכים משוכללים ומפיק נתוני חופים..."):
    wqi_layer, df_beaches, error_msg = process_s3_wqi_data(wb_selection, selected_date_str)

if error_msg:
    st.error(error_msg)
elif wqi_layer:
    col_map, col_info = st.columns([2.2, 1.1])
    
    with col_map:
        st.subheader(f"📍 מפת מדד משוכלל (WQI): {wb_selection}")
        
        m = folium.Map(location=WATER_BODIES[wb_selection]["center"], zoom_start=WATER_BODIES[wb_selection]["zoom"])
        
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
        
        # הוספת נקודות החופים למפה - ללא popup וללא tooltip בכלל! מפה נקייה לחלוטין.
        for _, r in df_beaches.iterrows():
            color_marker = "green" if (r['wqi'] and r['wqi'] > 65) else "orange" if r['wqi'] else "red"
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=6,
                color="black",
                fill_color=color_marker,
                fill_opacity=0.9,
                fill=True
            ).add_to(m)
            
        m.add_child(OnMapWaterLegend())
        st_folium(m, width=800, height=550, key="s3_pure_map_no_labels")
        
    with col_info:
        st.subheader("🏖️ סטטוס ורמת ניקיון החופים")
        st.markdown("ערכי המדד המשוכלל (WQI) שנדגמו סביב תחנות הניטור המבוקשות:")
        
        if df_beaches is not None and not df_beaches.empty:
            df_display = df_beaches[["name", "wqi"]].copy()
            df_display.columns = ["שם החוף / תחנה", "מדד איכות מים משוכלל"]
            df_display["מדד איכות מים משוכלל"] = df_display["מדד איכות מים משוכלל"].fillna("אין נתונים")
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.write("לא נמצאו תחנות מוגדרות לאזור זה.")
            
        st.info("💡 **כיצד לקרוא את המדד:** ככל שהציון המשוכלל בטבלה ועל המפה קרוב יותר לירוק (ערכים גבוהים), המים מוגדרים נקיים וצלולים יותר. ערכים נמוכים (צבע אדום) מעידים על עכירות חלקיקים או הצטברות חומר אורגני.")
