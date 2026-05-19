import ee
import json
import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import pandas as pd
import streamlit.components.v1 as components

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
    {"name": "ראש הנקרה",   "lat": 33.074, "lon": 35.100},
    {"name": "נהריה",        "lat": 33.005, "lon": 35.088},
    {"name": "עכו",          "lat": 32.927, "lon": 35.065},
    {"name": "קריית ים",     "lat": 32.865, "lon": 35.058},
    {"name": "חיפה צפון",    "lat": 32.846, "lon": 34.972},
    {"name": "חיפה מרכז",    "lat": 32.819, "lon": 34.960},
    {"name": "חיפה דרום",    "lat": 32.783, "lon": 34.950},
    {"name": "עתלית",        "lat": 32.693, "lon": 34.938},
    {"name": "זיכרון יעקב",  "lat": 32.571, "lon": 34.918},
    {"name": "קיסריה",       "lat": 32.497, "lon": 34.893},
    {"name": "נתניה",        "lat": 32.334, "lon": 34.855},
    {"name": "הרצליה",       "lat": 32.163, "lon": 34.796},
    {"name": "תל אביב צפון", "lat": 32.108, "lon": 34.768},
    {"name": "תל אביב מרכז", "lat": 32.080, "lon": 34.762},
    {"name": "תל אביב דרום", "lat": 32.051, "lon": 34.757},
    {"name": "בת ים",        "lat": 32.017, "lon": 34.749},
    {"name": "ראשון לציון",  "lat": 31.973, "lon": 34.737},
    {"name": "אשדוד צפון",   "lat": 31.844, "lon": 34.658},
    {"name": "אשדוד דרום",   "lat": 31.789, "lon": 34.637},
    {"name": "אשקלון",       "lat": 31.658, "lon": 34.553},
    {"name": "זיקים",        "lat": 31.606, "lon": 34.519},
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
        "note":      None,
        "indices":   ["NDWI", "Chl_proxy", "Turbidity", "FAI"],
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
        "note":      "⚠️ ב-10m רזולוציה הכנרת מכוסה היטב. שים לב לפריחות אצות בקיץ (NDCI גבוה).",
        "indices":   ["NDWI", "Chl_proxy", "Turbidity", "FAI"],
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
        "note":      "⚠️ מלוחות קיצונית — מדדי כלורופיל ועכירות אינם קלינריים. FAI מזהה Dunaliella salina (אצה ורודה). השתמש בציונים כ'אנומליה' בלבד.",
        "indices":   ["NDWI", "FAI", "Turbidity"],   # Chl_proxy פחות רלוונטי
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
        "note":      "💡 מים אוליגוטרופיים (כלורופיל נמוך מאוד ~0.05 mg/m³). הרחב את טווח התאריכים אם אין תמונות ברורות. שים לב לסחף אבק מהסיני.",
        "indices":   ["NDWI", "Chl_proxy", "Turbidity"],
    },
}

SCORE_COLORS = {0: "#AAAAAA", 1: "#27AE60", 2: "#F1C40F", 3: "#E67E22", 4: "#E74C3C", 5: "#8E44AD"}

# ==============================
# טעינת נתונים — גנרי לכל גוף מים
# ==============================
@st.cache_data(ttl=3600)
def load_data(wb_key: str, start_date: str, end_date: str):
    wb = WATER_BODIES[wb_key]

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(wb["bbox"])
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", wb["cloud_pct"])))

    count = s2.size().getInfo()
    if count == 0:
        return None, None, None, 0

    image_date = (ee.Date(
        s2.sort("system:time_start", False).first().get("system:time_start")
    ).format("YYYY-MM-dd").getInfo())

    def compute_indices(image):
        ndwi      = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
        chl_proxy = image.select("B5").divide(image.select("B4")).rename("Chl_proxy")
        turbidity = image.select("B4").rename("Turbidity")
        fai = (image.select("B8")
               .subtract(image.select("B4"))
               .subtract(image.select("B11").subtract(image.select("B4"))
                         .multiply((832-665)/(1610-665)))
               .rename("FAI"))
        return image.addBands([ndwi, chl_proxy, turbidity, fai])

    processed = s2.map(compute_indices).median()

    def get_point_values(point):
        pt         = ee.Geometry.Point([point["lon"], point["lat"]])
        buffer_1km = pt.buffer(1000)
        try:
            ndwi_img  = processed.normalizedDifference(["B3", "B8"])
            water_mask = ndwi_img.gt(0)

            distance_img = ee.Image(0).paint(
                featureCollection=ee.FeatureCollection([ee.Feature(pt)]),
                color=1
            ).fastDistanceTransform().sqrt().multiply(10)

            weight        = ee.Image(1000).subtract(distance_img).divide(1000).max(0)
            weight_masked = weight.updateMask(water_mask)
            selected      = processed.select(["NDWI","Chl_proxy","Turbidity","FAI"]).updateMask(water_mask)

            weighted_sum = selected.multiply(weight_masked).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=buffer_1km, scale=10, bestEffort=True).getInfo()
            weight_sum = weight_masked.reduceRegion(
                reducer=ee.Reducer.sum(), geometry=buffer_1km, scale=10, bestEffort=True).getInfo()

            def wm(band):
                ws = weighted_sum.get(band)
                wt = weight_sum.get("constant")
                if ws is None or wt is None or wt == 0: return None
                return ws / wt

            vals = {k: wm(k) for k in ["NDWI","Chl_proxy","Turbidity","FAI"]}
            if all(v is None for v in vals.values()):
                return {**point, **{k: None for k in vals}, "no_data": True}

            return {**point,
                    "ndwi":      round(vals["NDWI"],      3) if vals["NDWI"]      is not None else None,
                    "chl_proxy": round(vals["Chl_proxy"], 3) if vals["Chl_proxy"] is not None else None,
                    "turbidity": round(vals["Turbidity"], 1) if vals["Turbidity"] is not None else None,
                    "fai":       round(vals["FAI"],       4) if vals["FAI"]       is not None else None,
                    "no_data": False}
        except:
            return {**point, "ndwi": None, "chl_proxy": None, "turbidity": None, "fai": None, "no_data": True}

    data = [get_point_values(p) for p in wb["points"]]
    df   = pd.DataFrame(data)

    def water_quality_score(row):
        if row["no_data"]: return 0
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
        return {0:"⬜ אין מידע", 1:"🟢 מצוין", 2:"🟡 טוב",
                3:"🟠 בינוני",  4:"🔴 ירוד",  5:"⛔ גרוע"}.get(score, "❓")

    def composite_score(row):
        if row["no_data"]: return None
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

    return df, image_date, processed, count

# ==============================
# מפת חום ברמת פיקסל
# ==============================
def get_heatmap_url(processed, clip_geom):
    ndwi      = processed.normalizedDifference(["B3","B8"])
    chl_proxy = processed.select("B5").divide(processed.select("B4"))
    turbidity = processed.select("B4")

    ndwi_score = ndwi.add(0.3).divide(1.1).multiply(100).clamp(0, 100)
    chl_score  = ee.Image(2.5).subtract(chl_proxy).divide(1.5).multiply(100).clamp(0, 100)
    turb_score = ee.Image(1000).subtract(turbidity).divide(1000).multiply(100).clamp(0, 100)

    composite = (ndwi_score.multiply(0.4)
                 .add(chl_score.multiply(0.35))
                 .add(turb_score.multiply(0.25)))

    water_mask = ndwi.gt(0)
    composite  = composite.updateMask(water_mask).clip(clip_geom)

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
def build_map(df, image_date, processed, wb_key):
    wb = WATER_BODIES[wb_key]

    m = folium.Map(
        location=wb["center"],
        zoom_start=wb["zoom"],
        tiles="CartoDB positron",
        control_scale=True
    )

    satellite_group = folium.FeatureGroup(name="🛰️ לווין RGB",           show=False)
    heatmap_group   = folium.FeatureGroup(name="🌡️ מפת חום (ציון משוכלל)", show=True)
    points_group    = folium.FeatureGroup(name="📍 נקודות דיגום",           show=True)

    # שכבת לווין RGB
    try:
        rgb_url = processed.getMapId({"bands":["B4","B3","B2"],"min":0,"max":3000})["tile_fetcher"].url_format
        folium.TileLayer(tiles=rgb_url, name="RGB", attr="GEE", overlay=True, opacity=0.8).add_to(satellite_group)
    except:
        pass

    # מפת חום
    heatmap_url = get_heatmap_url(processed, wb["clip_geom"])
    if heatmap_url:
        folium.TileLayer(
            tiles=heatmap_url, name="מפת חום", attr="GEE/Copernicus",
            overlay=True, opacity=0.75
        ).add_to(heatmap_group)

    # נקודות דיגום
    for i, (_, row) in enumerate(df.iterrows(), 1):
        color = SCORE_COLORS.get(row["quality_score"], "#888")
        comp  = f"{int(round(row['composite']))}/100" if row["composite"] is not None else "N/A"

        if row["no_data"]:
            popup_html = (
                f"<div style='font-family:Arial;direction:rtl;'>"
                f"<b>נקודה {i} — {row['name']}</b><br>⬜ אין מידע זמין<br>"
                f"<small>📅 {image_date}</small></div>"
            )
        else:
            # שורת FAI רק אם יש נתון
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

        if not row["no_data"]:
            folium.Marker(
                location=[row["lat"], row["lon"]-0.02],
                icon=folium.DivIcon(
                    html=f"<div style='font-size:12px;font-weight:bold;background:rgba(255,255,255,0.92);padding:3px 7px;border-radius:5px;border-right:3px solid {color};white-space:nowrap;text-align:right;'>{comp}</div>",
                    icon_size=(90,24), icon_anchor=(90,12))
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

    for group in [satellite_group, heatmap_group, points_group]:
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

# ── טעינת נתונים ─────────────────────────────────────────────────────────────
end_date   = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=wb["days_back"])).strftime("%Y-%m-%d")

with st.spinner(f"🛰️ טוען נתוני לווין עבור {wb_key}..."):
    df, image_date, processed, scene_count = load_data(wb_key, start_date, end_date)

if df is None:
    st.error(
        f"לא נמצאו תמונות Sentinel-2 ב-{wb['days_back']} הימים האחרונים עבור {wb_key}. "
        f"נסה להגדיל את מספר הימים."
    )
    st.stop()

st.info(f"📅 מבוסס על צילום לווין מתאריך: **{image_date}** ({scene_count} סצנות נטענו)")

# ── מדדים כלליים ─────────────────────────────────────────────────────────────
valid = df[df["composite"].notna()]
col1, col2, col3, col4 = st.columns(4)
if len(valid) > 0:

# ── מפה + טבלה ───────────────────────────────────────────────────────────────
with st.spinner("🌡️ מחשב מפת חום..."):
    m = build_map(df, image_date, processed, wb_key)

map_col, table_col = st.columns([2, 1])

with map_col:
    st_folium(m, width="100%", height=680)

with table_col:
    st.markdown(f"#### 📊 נתוני {wb_key}")
    display_df = df[["name", "composite", "quality_label"]].copy()
    display_df.columns = ["נקודה", "ציון", "איכות"]
    display_df["ציון"] = display_df["ציון"].apply(
        lambda x: f"{int(round(x))}/100" if x is not None else "—"
    )
    st.dataframe(display_df, use_container_width=True, height=640)

