"""
app.py — MEDI Platform (Clean Version)
Israel Coast + Global S3 WQI only
"""

import math, json, tempfile, os
from google import genai as genai_client
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
import ee
from branca.element import MacroElement
from jinja2 import Template

# =============================================================================
# MEDI Risk Engine
# =============================================================================
_T = {
    "wqi":             {"low": 70, "mid": 50, "high": 35},
    "turbidity":       {"low": 0.3, "mid": 0.55, "high": 0.75},
    "chlorophyll":     {"low": 0.25, "mid": 0.5, "high": 0.7},
    "sst_anomaly":     {"low": 1.5,  "mid": 3.0, "high": 5.0},
}

PROFILES = {
    "Port Operations":       {"signals": ["wqi","turbidity"], "weights": [0.5,0.5], "description": "Vessel traffic, discharge risk, and water intake quality."},
    "Beach Safety":          {"signals": ["wqi","chlorophyll"], "weights": [0.6,0.4], "description": "Bathing water quality and algae/bloom risk."},
    "Aquaculture":           {"signals": ["wqi","chlorophyll","sst_anomaly"], "weights": [0.35,0.40,0.25], "description": "Bloom conditions, oxygen stress, feed disruption."},
    "ESG Compliance":        {"signals": ["wqi","turbidity","chlorophyll"], "weights": [0.4,0.3,0.3], "description": "Broad environmental footprint monitoring."},
    "Maritime Surveillance": {"signals": ["wqi","turbidity"], "weights": [0.4,0.6], "description": "Discharge events and water anomalies."},
}

@dataclass
class SignalReading:
    name: str; value: float
    raw_value: Optional[float] = None; unit: str = ""
    age_days: float = 0.0; confidence: float = 1.0

@dataclass
class MEDIResult:
    risk_score: float; risk_level: str; risk_color: str; trend: str
    trend_delta: Optional[float] = None; confidence: float = 0.0
    drivers: list = field(default_factory=list)
    profile: str = ""; explanation: str = ""; recommendation: str = ""; zone: str = ""

def _normalize_wqi(v): return max(0.0, min(1.0, 1.0 - v/100.0))

def _signal_risk(value, t):
    lo, mid, hi = t["low"], t["mid"], t["high"]
    if value <= lo:   return value/lo*0.33
    elif value <= mid: return 0.33+(value-lo)/(mid-lo)*0.34
    elif value <= hi:  return 0.67+(value-mid)/(hi-mid)*0.20
    else:
        excess=(value-hi)/(1.0-hi+1e-6)
        return 0.87+0.13*(1-math.exp(-3*excess))

def _confidence_from_signals(signals):
    if not signals: return 0.0
    return round(sum(s.confidence*math.exp(-0.2*s.age_days) for s in signals)/len(signals), 2)

def _detect_drivers(signal_risks, threshold=0.45):
    labels = {"wqi":"water quality degradation","turbidity":"turbidity anomaly",
              "chlorophyll":"algae/bloom signal","sst_anomaly":"SST anomaly"}
    drivers = [(labels.get(k,k),v) for k,v in signal_risks.items() if v>=threshold]
    drivers.sort(key=lambda x: x[1], reverse=True)
    return [d[0] for d in drivers]

def compute_medi(signals, profile_name, previous_score=None, zone=""):
    profile = PROFILES.get(profile_name, PROFILES["Beach Safety"])
    ps, pw  = profile["signals"], profile["weights"]
    signal_risks = {}; active = []
    for sn, w in zip(ps, pw):
        r = signals.get(sn)
        if r is None: continue
        val = _normalize_wqi(r.value) if sn=="wqi" else r.value
        val = max(0.0, min(1.0, val))
        signal_risks[sn] = _signal_risk(val, _T.get(sn, {"low":0.3,"mid":0.55,"high":0.75}))
        active.append(r)
    if not signal_risks:
        return MEDIResult(0,"UNKNOWN","#888888","STABLE",confidence=0.0,profile=profile_name,zone=zone)
    tw = sum(w for sn,w in zip(ps,pw) if sn in signal_risks)
    ws = sum(signal_risks[sn]*w for sn,w in zip(ps,pw) if sn in signal_risks)
    base = ws/tw if tw>0 else 0.0
    mx = max(signal_risks.values())
    if mx>0.85: base=base*0.6+mx*0.4
    score = round(base*100, 1)
    if score<25:   level,color="LOW","#1ecb7b"
    elif score<45: level,color="MODERATE","#7ecb1e"
    elif score<62: level,color="ELEVATED","#f0a500"
    elif score<78: level,color="HIGH","#e07b00"
    else:          level,color="CRITICAL","#e03c3c"
    if previous_score is None: trend,delta="STABLE",None
    else:
        delta=round(score-previous_score,1)
        trend="RISING" if delta>4 else "FALLING" if delta<-4 else "STABLE"
    return MEDIResult(score,level,color,trend,delta,
                      _confidence_from_signals(active),
                      _detect_drivers(signal_risks),profile_name,"","",zone)

def generate_medi_explanation(result: MEDIResult, api_key: str) -> MEDIResult:
    if not result.drivers:
        result.drivers = ["no significant anomalies detected"]
    profile_desc = PROFILES.get(result.profile,{}).get("description","")
    drivers_str  = ", ".join(result.drivers)
    trend_str    = f"{result.trend} ({result.trend_delta:+.1f} pts)" if result.trend_delta else result.trend
    prompt = f"""You are an expert maritime environmental analyst for the MEDI Platform.
Context:
- Monitoring zone: {result.zone or "coastal area"}
- Risk profile: {result.profile} — {profile_desc}
- Current risk score: {result.risk_score}/100 ({result.risk_level})
- Trend: {trend_str}
- Active risk drivers: {drivers_str}
- Confidence: {result.confidence:.0%}

Write exactly TWO short outputs.
1. EXPLANATION (1 sentence, max 25 words): Why is risk at this level? Be specific, mention drivers.
2. RECOMMENDATION (1 sentence, max 20 words): One concrete action. Start with a verb.

Format EXACTLY:
EXPLANATION: <text>
RECOMMENDATION: <text>"""
    try:
        client   = genai_client.Client(api_key=api_key)
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text     = response.text.strip()
        expl=""; rec=""
        for line in text.splitlines():
            if line.startswith("EXPLANATION:"): expl=line.replace("EXPLANATION:","").strip()
            elif line.startswith("RECOMMENDATION:"): rec=line.replace("RECOMMENDATION:","").strip()
        result.explanation    = expl or text
        result.recommendation = rec  or "Monitor situation closely."
    except Exception:
        result.explanation    = f"Risk assessment based on: {drivers_str}."
        result.recommendation = "Continue standard monitoring protocols."
    return result

# =============================================================================
# Page config & Styling
# =============================================================================
st.set_page_config(page_title="MEDI Platform", page_icon="🌊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');
:root{--ocean-deep:#020d18;--ocean-mid:#041e33;--ocean-surface:#062d4a;
--teal-bright:#00c8c8;--teal-dim:#007f8a;--green-safe:#1ecb7b;
--text-primary:#d6eaf8;--text-dim:#7fb3d3;--grid-line:rgba(0,200,200,0.08);}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--ocean-deep)!important;color:var(--text-primary)!important;}
[data-testid="stAppViewContainer"]{background-image:linear-gradient(var(--grid-line) 1px,transparent 1px),linear-gradient(90deg,var(--grid-line) 1px,transparent 1px),radial-gradient(ellipse 80% 60% at 50% -10%,rgba(0,100,140,0.35) 0%,transparent 70%);background-size:60px 60px,60px 60px,100% 100%;}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#031624 0%,#020d18 100%)!important;border-right:1px solid rgba(0,200,200,0.15)!important;}
[data-testid="stSidebar"] *{color:var(--text-primary)!important;}
h1,h2,h3{font-family:'Rajdhani',sans-serif!important;letter-spacing:0.06em;color:var(--teal-bright)!important;}
p,div,span,label{font-family:'Exo 2',sans-serif!important;color:var(--text-primary)!important;}
.medi-header{display:flex;align-items:center;gap:18px;padding:18px 28px;background:linear-gradient(90deg,rgba(0,200,200,0.08) 0%,transparent 100%);border-left:3px solid var(--teal-bright);border-bottom:1px solid rgba(0,200,200,0.15);margin-bottom:24px;}
.logo-text{font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;color:var(--teal-bright);letter-spacing:0.1em;line-height:1;}
.logo-sub{font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:var(--teal-dim);letter-spacing:0.18em;text-transform:uppercase;margin-top:3px;}
.status-badge{margin-left:auto;font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:var(--green-safe);border:1px solid var(--green-safe);padding:4px 10px;border-radius:2px;animation:pulse-badge 2.5s ease-in-out infinite;}
@keyframes pulse-badge{0%,100%{opacity:1;}50%{opacity:0.5;}}
[data-testid="stSelectbox"]>div>div,[data-testid="stRadio"] label{background:var(--ocean-surface)!important;border:1px solid rgba(0,200,200,0.2)!important;color:var(--text-primary)!important;border-radius:3px!important;}
[data-testid="stMetric"]{background:rgba(0,200,200,0.05)!important;border:1px solid rgba(0,200,200,0.15)!important;border-radius:4px!important;padding:10px 14px!important;}
[data-testid="stMetricLabel"]{color:var(--teal-dim)!important;font-size:0.75rem!important;}
[data-testid="stMetricValue"]{color:var(--teal-bright)!important;font-family:'Share Tech Mono',monospace!important;}
[data-testid="stDataFrame"]{border:1px solid rgba(0,200,200,0.2)!important;border-radius:4px!important;}
[data-testid="stDataFrame"] th{background:var(--ocean-surface)!important;color:var(--teal-bright)!important;font-family:'Share Tech Mono',monospace!important;font-size:0.75rem!important;}
[data-testid="stDataFrame"] td{color:var(--text-primary)!important;}
[data-testid="stIFrame"],iframe{border:1px solid rgba(0,200,200,0.2)!important;border-radius:4px!important;}
[data-testid="stSidebar"] hr{border-color:rgba(0,200,200,0.15)!important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--ocean-deep);}
::-webkit-scrollbar-thumb{background:var(--teal-dim);border-radius:3px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="medi-header">
  <div>
    <div class="logo-text">⬡ MEDI PLATFORM</div>
    <div class="logo-sub">Maritime Environmental Decision Intelligence</div>
  </div>
  <div class="status-badge">● LIVE MONITORING</div>
</div>
""", unsafe_allow_html=True)

# Analytics
components.html('<script async src="https://cloud.umami.is/script.js" data-website-id="07a48db1-5aa7-4d88-aaac-9cfb6fc2600d"></script>', height=0)
if "ga_loaded" not in st.session_state:
    st.session_state.ga_loaded = True
    components.html("""<script async src="https://www.googletagmanager.com/gtag/js?id=G-K37THY2160"></script>
<script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','G-K37THY2160');</script>""", height=0)

# =============================================================================
# GEE Auth
# =============================================================================
@st.cache_resource
def init_gee():
    creds=dict(st.secrets["gee_credentials"])
    with tempfile.NamedTemporaryFile(mode="w",suffix=".json",delete=False) as f:
        f.write(json.dumps(creds)); tmp=f.name
    ee.Initialize(ee.ServiceAccountCredentials(creds["client_email"],tmp))
    os.unlink(tmp)
init_gee()

# =============================================================================
# Geometries & Beaches
# =============================================================================
HAIFA_BBOX       = ee.Geometry.Rectangle([34.20,31.20,35.20,33.20])
ISRAEL_CLIP      = ee.Geometry.Polygon([[[34.95,33.10],[34.55,33.10],[34.15,32.50],[34.10,32.00],
    [34.15,31.50],[34.50,31.25],[34.75,31.25],[34.95,31.30],[35.02,31.60],[35.00,32.10],[35.05,32.60],[35.10,33.10],[34.95,33.10]]])

BEACHES = [
    {"name":"Rosh HaNikra","lat":33.0765,"lon":35.0983},
    {"name":"Nahariya","lat":33.0048,"lon":35.0832},
    {"name":"Acre","lat":32.9280,"lon":35.0680},
    {"name":"Haifa North","lat":32.8380,"lon":34.9820},
    {"name":"Atlit","lat":32.6892,"lon":34.9368},
    {"name":"Caesarea","lat":32.4948,"lon":34.8912},
    {"name":"Netanya","lat":32.3318,"lon":34.8512},
    {"name":"Herzliya","lat":32.1648,"lon":34.7962},
    {"name":"Tel Aviv Center","lat":32.0798,"lon":34.7618},
    {"name":"Ashdod","lat":31.7848,"lon":34.6248},
    {"name":"Ashkelon","lat":31.6548,"lon":34.5448},
    {"name":"Zikim","lat":31.6098,"lon":34.5198},
]

# =============================================================================
# Map Components
# =============================================================================
class OnMapWaterLegend(MacroElement):
    def __init__(self):
        super().__init__()
        self._template=Template("""{% macro script(this, kwargs) %}
var lg=L.control({position:'topright'});
lg.onAdd=function(map){var d=L.DomUtil.create('div','info legend');
d.style.cssText='background:rgba(2,13,24,0.92);padding:12px;border:1px solid rgba(0,200,200,0.3);border-radius:6px;font-family:Arial,sans-serif;font-size:12px;color:#d6eaf8;';
d.innerHTML='<div style="font-weight:bold;margin-bottom:8px;text-align:center;color:#00c8c8;">Water Quality Index</div><div style="display:flex;align-items:center;gap:8px;"><div style="height:120px;width:14px;background:linear-gradient(to bottom,#00FF00,#FFFF00,#FF0000);border-radius:3px;flex-shrink:0;"></div><div style="display:flex;flex-direction:column;justify-content:space-between;height:120px;font-size:11px;"><span style="color:#1ecb7b;font-weight:bold;">Clean</span><span style="color:#f0a500;font-weight:bold;">Moderate</span><span style="color:#e03c3c;font-weight:bold;">Polluted</span></div></div>';return d;};
lg.addTo({{this._parent.get_name()}});{% endmacro %}""")

class OnMapAtmosphereControl(MacroElement):
    def __init__(self, atm):
        super().__init__()
        ws=atm.get("wind_speed"); wd=atm.get("wind_dir_deg"); tc=atm.get("temp_c"); pr=atm.get("precip_mm"); rh=atm.get("humidity")
        ar=int(wd) if wd else 0
        ws_s=f"{ws:.1f} m/s" if ws else "—"; tc_s=f"{tc:.1f}°C" if tc else "—"
        pr_s=f"{pr:.1f} mm" if pr else "—"; rh_s=f"{int(rh)}%" if rh else "—"
        ri="🌧️" if pr and pr>0.5 else "☀️"
        bf=12
        if ws:
            for b,t in enumerate([0.3,1.6,3.4,5.5,8.0,10.8,13.9,17.2,20.8,24.5,28.5,32.7]):
                if ws<t: bf=b; break
        bc="#27AE60" if bf<4 else "#F39C12" if bf<7 else "#E74C3C"
        html=f'<div style="background:rgba(2,13,24,0.92);border:1px solid rgba(0,200,200,0.3);border-radius:8px;padding:10px 13px;font-family:Arial,sans-serif;font-size:12px;color:#d6eaf8;min-width:150px;"><div style="font-weight:bold;margin-bottom:7px;text-align:center;font-size:11px;color:#00c8c8;">🌍 Atmospheric Context</div><div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;"><svg width="24" height="24" viewBox="0 0 28 28"><g transform="rotate({ar},14,14)"><polygon points="14,2 18,22 14,18 10,22" fill="#2980B9" opacity="0.85"/></g></svg><div><div style="font-size:12px;font-weight:bold;">{ws_s}</div><div style="font-size:10px;color:{bc};">Beaufort {bf}</div></div></div><hr style="margin:5px 0;border-color:rgba(0,200,200,0.2);"><div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span>🌡️</span><span style="font-weight:bold;">{tc_s}</span></div><div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span>{ri}</span><span style="font-weight:bold;">{pr_s}</span></div><div style="display:flex;justify-content:space-between;"><span>💧</span><span style="font-weight:bold;">{rh_s}</span></div></div>'
        self._template=Template("""{% macro script(this, kwargs) %}
var ac=L.control({position:'bottomleft'});
ac.onAdd=function(map){var d=L.DomUtil.create('div','ac');d.innerHTML=`"""+html.replace("`","'")+"""`;L.DomEvent.disableClickPropagation(d);return d;};
ac.addTo({{this._parent.get_name()}});{% endmacro %}""")

# =============================================================================
# Data Functions
# =============================================================================
def _empty_atm():
    return {"wind_speed":None,"wind_dir_deg":None,"temp_c":None,"humidity":None,
            "precip_mm":None,"_error":None}

@st.cache_data(ttl=3600)
def get_atm(lat, lon):
    try:
        import requests as _req
        r=_req.get("https://api.open-meteo.com/v1/forecast",params={
            "latitude":round(lat,4),"longitude":round(lon,4),
            "current":"temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,weather_code",
            "wind_speed_unit":"ms","forecast_days":1},timeout=10)
        r.raise_for_status(); cur=r.json().get("current",{})
        ws,wd,tc=cur.get("wind_speed_10m"),cur.get("wind_direction_10m"),cur.get("temperature_2m")
        rh,pr=cur.get("relative_humidity_2m"),cur.get("precipitation")
        return {"wind_speed":round(ws,1) if ws else None,"wind_dir_deg":round(wd,1) if wd else None,
                "temp_c":round(tc,1) if tc else None,"humidity":round(rh,0) if rh else None,
                "precip_mm":round(pr,2) if pr else None,"_error":None}
    except Exception as e:
        return {**_empty_atm(),"_error":str(e)}

@st.cache_data(ttl=3600)
def get_sst(lat, lon):
    try:
        import requests as _req
        r=_req.get("https://marine-api.open-meteo.com/v1/marine",params={
            "latitude":lat,"longitude":lon,"current":"sea_surface_temperature","forecast_days":1},timeout=10)
        return r.json().get("current",{}).get("sea_surface_temperature")
    except: return None

@st.cache_data(ttl=14400)
def get_available_s3_dates(days_back=30):
    end=datetime.utcnow(); start=end-timedelta(days=days_back)
    coll=(ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(HAIFA_BBOX)
          .filterDate(start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d')))
    dl=coll.aggregate_array("system:time_start").getInfo()
    return sorted(list(set([datetime.utcfromtimestamp(d/1000).strftime("%Y-%m-%d") for d in dl])),reverse=True)

@st.cache_data(ttl=10800)
def get_modis_sst_anomaly(target_date_str):
    """
    MODIS MOD11A1 — Sea Surface Temperature anomaly.
    anomaly = today SST - 30-day mean SST
    Returns: ee.Image with band 'SST_anomaly' (degrees C) + scalar mean anomaly
    """
    wm  = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(25)
    t   = ee.Date(target_date_str)

    # Today SST (LST_Day_1km in Kelvin × 0.02 → Celsius)
    today_coll = (ee.ImageCollection("MODIS/061/MOD11A1")
                  .filterBounds(HAIFA_BBOX)
                  .filterDate(t.advance(-2,"day"), t.advance(1,"day"))
                  .select("LST_Day_1km"))
    if today_coll.size().getInfo() == 0:
        return None, None

    sst_today = today_coll.mean().multiply(0.02).subtract(273.15).updateMask(wm)

    # 30-day baseline
    baseline_coll = (ee.ImageCollection("MODIS/061/MOD11A1")
                     .filterBounds(HAIFA_BBOX)
                     .filterDate(t.advance(-31,"day"), t.advance(-1,"day"))
                     .select("LST_Day_1km"))
    sst_baseline  = baseline_coll.mean().multiply(0.02).subtract(273.15).updateMask(wm)

    anomaly_img = sst_today.subtract(sst_baseline).rename("SST_anomaly").clip(HAIFA_BBOX)

    # Scalar mean anomaly for MEDI engine
    try:
        val = anomaly_img.reduceRegion(
            reducer   = ee.Reducer.mean(),
            geometry  = HAIFA_BBOX,
            scale     = 1000,
            bestEffort= True,
        ).getInfo()
        mean_anomaly = val.get("SST_anomaly")
        mean_anomaly = round(float(mean_anomaly), 2) if mean_anomaly is not None else None
    except Exception:
        mean_anomaly = None

    return anomaly_img, mean_anomaly


@st.cache_data(ttl=7200)
def process_israel_wqi(target_date_str):
    wm=ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(25)
    t=ee.Date(target_date_str)
    coll=(ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(HAIFA_BBOX)
          .filterDate(t.advance(-1,'day'),t.advance(1,'day')))
    if coll.size().getInfo()==0: return None,None,"No Sentinel-3 data for this date."
    img=coll.median().clip(ISRAEL_CLIP).updateMask(wm)
    ndwi=img.normalizedDifference(['Oa06_radiance','Oa17_radiance'])
    b10,b11,b12=img.select('Oa10_radiance'),img.select('Oa11_radiance'),img.select('Oa12_radiance')
    mci=b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
    turb=img.select('Oa08_radiance')
    raw=ndwi.unitScale(-0.2,0.5).clamp(0,1).add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1)).add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1)).divide(3).multiply(100).rename('WQI')
    wqi=raw.reduceNeighborhood(reducer=ee.Reducer.mean(),kernel=ee.Kernel.square(radius=1,units='pixels')).rename('WQI').updateMask(wm)
    def _pt(pt):
        try:
            v=wqi.reduceRegion(reducer=ee.Reducer.mean(),geometry=ee.Geometry.Point([pt["lon"],pt["lat"]]).buffer(450),scale=300,bestEffort=True).getInfo()
            wv=v.get('WQI'); return {**pt,"wqi":round(wv,1) if wv else None}
        except: return {**pt,"wqi":None}
    with ThreadPoolExecutor(max_workers=4) as ex: pts=list(ex.map(_pt,BEACHES))
    return wqi, pd.DataFrame(pts), None

@st.cache_data(ttl=14400)
def get_global_wqi_layer(target_date_str, bbox_rect):
    lon_min,lat_min,lon_max,lat_max=bbox_rect
    bbox=ee.Geometry.Rectangle([lon_min,lat_min,lon_max,lat_max])
    wm=ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(25)
    t=ee.Date(target_date_str)
    coll=(ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(bbox)
          .filterDate(t.advance(-1,'day'),t.advance(1,'day')))
    if coll.size().getInfo()==0: return None,"No data for this area/date."
    img=coll.median().clip(bbox).updateMask(wm)
    ndwi=img.normalizedDifference(['Oa06_radiance','Oa17_radiance'])
    b10,b11,b12=img.select('Oa10_radiance'),img.select('Oa11_radiance'),img.select('Oa12_radiance')
    mci=b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
    turb=img.select('Oa08_radiance')
    raw=ndwi.unitScale(-0.2,0.5).clamp(0,1).add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1)).add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1)).divide(3).multiply(100).rename('WQI')
    return raw.reduceNeighborhood(reducer=ee.Reducer.mean(),kernel=ee.Kernel.square(radius=1,units='pixels')).rename('WQI').updateMask(wm), None

def get_bbox_from_map(map_data, zoom):
    if not map_data or not map_data.get("center"): return None
    lat=map_data["center"]["lat"]; lon=map_data["center"]["lng"]
    dp=360.0/(256*(2**zoom)); hw=dp*400; hh=dp*275
    return (max(-180,lon-hw),max(-85,lat-hh),min(180,lon+hw),min(85,lat+hh))

def haversine_km(lat1,lon1,lat2,lon2):
    R=6371.0; p1,p2=math.radians(lat1),math.radians(lat2)
    a=math.sin(math.radians(lat2-lat1)/2)**2+math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

# =============================================================================
# UI
# =============================================================================
st.sidebar.markdown("### 🔧 Mission Parameters")
MODE_ISRAEL = "🏖️ Israel Coast"
MODE_GLOBAL = "🌍 Global"
mode = st.sidebar.selectbox("Select monitoring zone:", [MODE_ISRAEL, MODE_GLOBAL])

# Risk profile shown in MEDI tab only — initialized here for session state
medi_profile = "Beach Safety"  # default

# ── Israel Coast ──────────────────────────────────────────────────────────────
if mode == MODE_ISRAEL:
    # Date selector
    with st.spinner("Loading available dates..."):
        dates = get_available_s3_dates()
    if dates:
        sel_date = st.sidebar.selectbox("Select acquisition date:", [f"🟢 {d}" for d in dates]).replace("🟢 ","")
    else:
        sel_date = (datetime.utcnow()-timedelta(days=1)).strftime('%Y-%m-%d')

    # ── Tab selector ──────────────────────────────────────────────────────────
    tab_wqi, tab_medi = st.tabs(["🌊 Water Quality Index", "⬡ MEDI Risk Assessment"])

    with st.spinner("Computing WQI from Sentinel-3..."):
        wqi_layer, df, err = process_israel_wqi(sel_date)

    if err:
        st.error(err)
    elif wqi_layer is not None:
        atm = get_atm(32.4, 34.85)

    # Shared map builder
    def _build_map():
        m = folium.Map(location=[32.4, 34.85], zoom_start=8)
        vis = {'min':40,'max':85,'palette':['#FF0000','#FFFF00','#00FF00']}
        mid = ee.Image(wqi_layer).getMapId(vis)
        folium.TileLayer(tiles=mid['tile_fetcher'].url_format,attr='GEE S3',
                         name="WQI",overlay=True,control=False,opacity=0.85).add_to(m)
        for _,r in df.iterrows():
            sc = r.get('wqi')
            cm = "#1ecb7b" if sc and sc>65 else "#f0a500" if sc and sc>45 else "#e03c3c"
            wqi_str = f"{sc:.1f}" if sc else "N/A"
            folium.Marker(
                location=[r["lat"],r["lon"]],
                tooltip=f"🏖️ {r['name']} | WQI: {wqi_str}",
                popup=folium.Popup(f"<b>{r['name']}</b><br>WQI: <span style='color:{cm};font-weight:bold;'>{wqi_str}</span>", max_width=180),
                icon=folium.DivIcon(
                    html=f'''<div style="background:{cm};border:2px solid white;border-radius:50%;width:13px;height:13px;box-shadow:0 0 6px {cm}99;"></div>
<div style="position:absolute;top:15px;left:-28px;white-space:nowrap;font-family:Arial;font-size:10px;font-weight:600;color:white;text-shadow:0 1px 3px rgba(0,0,0,0.9);">{r["name"]}</div>''',
                    icon_size=(13,13),icon_anchor=(6,6))
            ).add_to(m)
        m.add_child(OnMapWaterLegend())
        m.add_child(OnMapAtmosphereControl(atm))
        return m

    # ── Tab 1: Water Quality Index ─────────────────────────────────────────────
    with tab_wqi:
        if err:
            st.error(err)
        elif wqi_layer is not None:
            col_map, col_info = st.columns([3.5, 1.5])
            with col_map:
                st_folium(_build_map(), width=820, height=550, key="israel_map_wqi", returned_objects=[])
            with col_info:
                st.markdown("#### 🏖️ Station Status")
                if df is not None and not df.empty:
                    def _st(s):
                        try: v=float(s)
                        except: return "❓ N/A"
                        return "🟢 Clean" if v>=70 else "🟡 Moderate" if v>=55 else "🔴 Polluted"
                    df_d = df[["name","wqi"]].copy()
                    df_d["Status"] = df_d["wqi"].apply(_st)
                    df_d = df_d.rename(columns={"name":"Station","wqi":"WQI"})
                    st.dataframe(df_d[["Station","WQI","Status"]], use_container_width=True, hide_index=True)

    # ── Tab 2: MEDI Risk Assessment ───────────────────────────────────────────
    with tab_medi:
        if err:
            st.error(err)
        elif wqi_layer is not None:
            # Profile selector inside the tab
            col_prof, _ = st.columns([2, 3])
            with col_prof:
                medi_profile = st.selectbox("🎯 Risk Profile:", list(PROFILES.keys()), key="medi_profile_select")
                st.caption(PROFILES[medi_profile]["description"])

            col_map2, col_medi = st.columns([3.0, 2.0])
            with col_map2:
                st_folium(_build_map(), width=580, height=520, key="israel_map_medi", returned_objects=[])

            with col_medi:
              try:
                with st.spinner("Computing MEDI score..."):
                    valid = df["wqi"].dropna()
                    avg_wqi = float(valid.mean()) if not valid.empty else 60.0
                    ws = atm.get("wind_speed") or 0.0
                    pr = atm.get("precip_mm") or 0.0
                    turb_proxy = min(1.0,(ws/20.0)*0.5+(pr/10.0)*0.5)
                    chl_proxy  = max(0.0,min(1.0,1.0-(avg_wqi/100.0)))
                    _, sst_anom = get_modis_sst_anomaly(sel_date)
                    sst_signal = max(0.0, min(1.0, (sst_anom or 0.0) / 5.0))
                    sst_conf   = 0.9 if sst_anom is not None else 0.0
                    signals = {
                        "wqi":        SignalReading("wqi",avg_wqi,raw_value=avg_wqi,unit="score",age_days=1.0,confidence=0.85),
                        "turbidity":  SignalReading("turbidity",turb_proxy,age_days=0.1,confidence=0.7),
                        "chlorophyll":SignalReading("chlorophyll",chl_proxy,age_days=1.0,confidence=0.75),
                        "sst_anomaly":SignalReading("sst_anomaly",sst_signal,raw_value=sst_anom,unit="degC",age_days=0.5,confidence=sst_conf),
                    }
                    prev = st.session_state.get("medi_prev_score")
                    medi = compute_medi(signals, medi_profile, previous_score=prev, zone="Israel Mediterranean Coast")
                    st.session_state["medi_prev_score"] = medi.risk_score
                    api_key = st.secrets.get("gemini_api_key","")
                    if api_key:
                        medi = generate_medi_explanation(medi, api_key)
                    ti = "📈" if medi.trend=="RISING" else "📉" if medi.trend=="FALLING" else "➡️"
                    ds = f" ({medi.trend_delta:+.1f})" if medi.trend_delta is not None else ""
                    dr = " · ".join(medi.drivers) if medi.drivers else "No significant anomalies"
                    sst_str = f"{sst_anom:+.1f}°C" if sst_anom is not None else "N/A"
                    st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(2,13,24,0.97),rgba(6,45,74,0.92));
border:1px solid {medi.risk_color};border-radius:8px;padding:24px 28px;
box-shadow:0 0 28px {medi.risk_color}44;margin-top:8px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
    <div>
      <span style="font-family:'Rajdhani',sans-serif;font-size:1rem;color:#7fb3d3;letter-spacing:0.1em;">MEDI RISK SCORE</span><br>
      <span style="font-family:'Rajdhani',sans-serif;font-size:4rem;font-weight:700;color:{medi.risk_color};line-height:1;">{medi.risk_score:.0f}</span>
      <span style="font-size:1.2rem;color:{medi.risk_color};margin-left:10px;font-weight:700;">{medi.risk_level}</span>
    </div>
    <div style="text-align:right;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;color:#7fb3d3;">TREND</div>
      <div style="font-size:1.3rem;color:#d6eaf8;">{ti} {medi.trend}{ds}</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#7fb3d3;margin-top:6px;">CONFIDENCE: {medi.confidence:.0%}</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#7fb3d3;margin-top:3px;">SST ANOMALY: {sst_str}</div>
    </div>
  </div>
  <div style="border-top:1px solid rgba(0,200,200,0.15);padding-top:12px;margin-bottom:10px;">
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#7fb3d3;letter-spacing:0.1em;">RISK DRIVERS</span><br>
    <span style="color:#d6eaf8;font-size:0.95rem;">{dr}</span>
  </div>
  <div style="border-top:1px solid rgba(0,200,200,0.15);padding-top:12px;margin-bottom:10px;">
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#7fb3d3;letter-spacing:0.1em;">ASSESSMENT</span><br>
    <span style="color:#d6eaf8;font-size:0.95rem;font-style:italic;">{medi.explanation}</span>
  </div>
  <div style="background:rgba(0,200,200,0.08);border-left:3px solid #00c8c8;border-radius:4px;padding:12px 16px;">
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#00c8c8;letter-spacing:0.1em;">⚡ RECOMMENDED ACTION</span><br>
    <span style="color:#d6eaf8;font-size:1rem;font-weight:600;">{medi.recommendation}</span>
  </div>
  <div style="margin-top:12px;text-align:right;">
    <span style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#3a6b8a;">PROFILE: {medi.profile.upper()} · {sel_date}</span>
  </div>
</div>
""", unsafe_allow_html=True)
              except Exception as e:
                  st.warning(f"MEDI computation unavailable: {e}")


# ── Global ────────────────────────────────────────────────────────────────────
else:
    available_dates = [(datetime.utcnow()-timedelta(days=d)).strftime('%Y-%m-%d') for d in range(1,8)]
    sel_date = st.sidebar.selectbox("Select acquisition date:",[f"🟢 {d}" for d in available_dates]).replace("🟢 ","")

    st.markdown("### 🌍 Global WQI — Sentinel-3")

    if "g_center" not in st.session_state:
        st.session_state.g_center=(20.0,0.0); st.session_state.g_zoom=3
        st.session_state.g_bbox=None; st.session_state.g_layer=None

    zi=st.session_state.g_zoom; ci=list(st.session_state.g_center)
    mg=folium.Map(location=ci,zoom_start=zi)

    gl=st.session_state.g_layer
    if gl is not None:
        try:
            mid=ee.Image(gl).getMapId({'min':40,'max':85,'palette':['#FF0000','#FFFF00','#00FF00']})
            folium.TileLayer(tiles=mid['tile_fetcher'].url_format,attr='GEE S3',
                             name="WQI",overlay=True,control=False,opacity=0.85).add_to(mg)
        except: pass

    mg.add_child(OnMapWaterLegend())
    mdg=st_folium(mg,width=900,height=580,key="global_map",returned_objects=["center","zoom"])

    if mdg:
        nz=mdg.get("zoom") or zi; nc=mdg.get("center")
        if nc:
            nl,no=nc["lat"],nc["lng"]; pl,po=st.session_state.g_center
            dk=haversine_km(pl,po,nl,no)
            pz=st.session_state.g_zoom; tc=(pz<5)!=(nz<5); mf=(dk>300)
            st.session_state.g_zoom=nz; st.session_state.g_center=(nl,no)
            if tc or mf:
                nb=get_bbox_from_map(mdg,nz); st.session_state.g_bbox=nb
                if nb:
                    with st.spinner("Loading WQI layer..."):
                        layer,e=get_global_wqi_layer(sel_date,nb)
                        st.session_state.g_layer=layer if not e else None
                st.rerun()
