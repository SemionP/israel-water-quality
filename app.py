"""
app.py - MEDI Platform (Clean Version)
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
from folium.plugins import Draw
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
- Risk profile: {result.profile} - {profile_desc}
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
st.set_page_config(page_title="MEDI Platform - Maritime Environmental Decision Intelligence", page_icon="🌊", layout="wide")

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
.medi-header{display:flex;align-items:center;gap:18px;padding:6px 16px;background:linear-gradient(90deg,rgba(0,200,200,0.06) 0%,transparent 100%);border-left:2px solid var(--teal-bright);border-bottom:1px solid rgba(0,200,200,0.1);margin-bottom:4px;}
.logo-text{font-family:'Rajdhani',sans-serif;font-size:1.2rem;font-weight:700;color:var(--teal-bright);letter-spacing:0.1em;line-height:1;}
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
# Persistent Zone Storage
# =============================================================================
ZONES_KEY = "medi-zones-v1"

def load_zones() -> dict:
    try:
        import json as _j
        return _j.loads(open("/tmp/medi_zones.json").read())
    except:
        return {}

def save_zones(zones: dict):
    import json as _j
    try:
        with open("/tmp/medi_zones.json","w") as f:
            f.write(_j.dumps(zones))
    except:
        pass

def load_points() -> dict:
    """Load user-defined monitoring points."""
    try:
        import json as _j
        return _j.loads(open("/tmp/medi_points.json").read())
    except:
        return {}

def save_points(points: dict):
    import json as _j
    try:
        with open("/tmp/medi_points.json","w") as f:
            f.write(_j.dumps(points))
    except:
        pass


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
# Port Zones - for MEDI Port Analysis
# =============================================================================
PORTS = {
    "🚢 Haifa Port": {
        "lat": 32.8230, "lon": 35.0020,
        "bbox": ee.Geometry.Rectangle([34.94, 32.78, 35.06, 32.87]),
        "radius_km": 5,
        "description": "Major Mediterranean cargo & passenger port",
        "atm_coords": (32.82, 35.00),
    },
    "⚓ Ashdod Port": {
        "lat": 31.8167, "lon": 34.6500,
        "bbox": ee.Geometry.Rectangle([34.60, 31.77, 34.70, 31.86]),
        "radius_km": 4,
        "description": "Israel's largest cargo port",
        "atm_coords": (31.82, 34.65),
    },
    "🐠 Eilat Port": {
        "lat": 29.5510, "lon": 34.9480,
        "bbox": ee.Geometry.Rectangle([34.91, 29.51, 34.99, 29.59]),
        "radius_km": 3,
        "description": "Red Sea port - coral reef proximity",
        "atm_coords": (29.55, 34.95),
    },
}


# =============================================================================
# Maritime Zone Polygons - Offshore areas per city (sea only)
# Each polygon: ~3-5km offshore, covers city coastal stretch
# =============================================================================
MARITIME_ZONES = {
    "Nahariya":  ee.Geometry.Polygon([[
        [34.88, 33.00], [34.95, 33.00], [34.95, 33.05], [34.88, 33.05]
    ]]),
    "Acre":      ee.Geometry.Polygon([[
        [34.90, 32.90], [34.97, 32.90], [34.97, 32.95], [34.90, 32.95]
    ]]),
    "Krayot":    ee.Geometry.Polygon([[
        [34.92, 32.83], [34.98, 32.83], [34.98, 32.89], [34.92, 32.89]
    ]]),
    "Haifa":     ee.Geometry.Polygon([[
        [34.88, 32.78], [34.97, 32.78], [34.97, 32.84], [34.88, 32.84]
    ]]),
    "Atlit":     ee.Geometry.Polygon([[
        [34.88, 32.67], [34.95, 32.67], [34.95, 32.72], [34.88, 32.72]
    ]]),
    "Caesarea":  ee.Geometry.Polygon([[
        [34.85, 32.47], [34.93, 32.47], [34.93, 32.53], [34.85, 32.53]
    ]]),
    "Hadera":    ee.Geometry.Polygon([[
        [34.84, 32.42], [34.92, 32.42], [34.92, 32.47], [34.84, 32.47]
    ]]),
    "Netanya":   ee.Geometry.Polygon([[
        [34.82, 32.28], [34.90, 32.28], [34.90, 32.35], [34.82, 32.35]
    ]]),
    "Herzliya":  ee.Geometry.Polygon([[
        [34.77, 32.14], [34.85, 32.14], [34.85, 32.20], [34.77, 32.20]
    ]]),
    "Tel Aviv":  ee.Geometry.Polygon([[
        [34.73, 32.04], [34.81, 32.04], [34.81, 32.12], [34.73, 32.12]
    ]]),
    "Palmahim":  ee.Geometry.Polygon([[
        [34.68, 31.90], [34.76, 31.90], [34.76, 31.96], [34.68, 31.96]
    ]]),
    "Ashdod":    ee.Geometry.Polygon([[
        [34.60, 31.77], [34.68, 31.77], [34.68, 31.84], [34.60, 31.84]
    ]]),
    "Ashkelon":  ee.Geometry.Polygon([[
        [34.52, 31.63], [34.60, 31.63], [34.60, 31.69], [34.52, 31.69]
    ]]),
}

# Representative point for each city (for map marker)
CITY_POINTS = {
    "Nahariya": {"lat": 33.020, "lon": 34.915},
    "Acre":     {"lat": 32.924, "lon": 34.935},
    "Krayot":   {"lat": 32.860, "lon": 34.950},
    "Haifa":    {"lat": 32.810, "lon": 34.925},
    "Atlit":    {"lat": 32.690, "lon": 34.915},
    "Caesarea": {"lat": 32.500, "lon": 34.890},
    "Hadera":   {"lat": 32.445, "lon": 34.880},
    "Netanya":  {"lat": 32.315, "lon": 34.860},
    "Herzliya": {"lat": 32.170, "lon": 34.810},
    "Tel Aviv": {"lat": 32.080, "lon": 34.770},
    "Palmahim": {"lat": 31.930, "lon": 34.720},
    "Ashdod":   {"lat": 31.805, "lon": 34.640},
    "Ashkelon": {"lat": 31.660, "lon": 34.560},
}

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
d.innerHTML='<div style="font-weight:bold;margin-bottom:8px;text-align:center;color:#00c8c8;">Water Quality Index</div><div style="display:flex;align-items:center;gap:8px;"><div style="height:120px;width:14px;background:linear-gradient(to bottom,#4575b4,#74add1,#fdae61,#d73027);border-radius:3px;flex-shrink:0;"></div><div style="display:flex;flex-direction:column;justify-content:space-between;height:120px;font-size:11px;"><span style="color:#1ecb7b;font-weight:bold;">Clean</span><span style="color:#f0a500;font-weight:bold;">Moderate</span><span style="color:#e03c3c;font-weight:bold;">Polluted</span></div></div>';return d;};
lg.addTo({{this._parent.get_name()}});{% endmacro %}""")

class OnMapAtmosphereControl(MacroElement):
    def __init__(self, atm):
        super().__init__()
        ws=atm.get("wind_speed"); wd=atm.get("wind_dir_deg"); tc=atm.get("temp_c"); pr=atm.get("precip_mm"); rh=atm.get("humidity")
        ar=int(wd) if wd else 0
        ws_s=f"{ws:.1f} m/s" if ws else "-"; tc_s=f"{tc:.1f}°C" if tc else "-"
        pr_s=f"{pr:.1f} mm" if pr else "-"; rh_s=f"{int(rh)}%" if rh else "-"
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
def get_available_s3_dates(days_back=60):
    end   = datetime.utcnow()
    start = end - timedelta(days=days_back)
    # Use wider bbox to catch all S3 passes over Israel
    wide_bbox = ee.Geometry.Rectangle([34.0, 29.0, 36.0, 33.5])
    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
            .filterBounds(wide_bbox)
            .filterDate(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    dl = coll.aggregate_array("system:time_start").getInfo()
    dates = sorted(list(set([
        datetime.utcfromtimestamp(d/1000).strftime("%Y-%m-%d") for d in dl
    ])), reverse=True)
    return dates

@st.cache_data(ttl=10800)
def get_modis_sst_anomaly(target_date_str):
    """
    MODIS MOD11A1 - Sea Surface Temperature anomaly.
    anomaly = today SST - 30-day mean SST
    Returns: ee.Image with band 'SST_anomaly' (degrees C) + scalar mean anomaly
    """
    wm  = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
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
def process_modis_wqi(target_date_str):
    """
    MODIS MOD09GA - daily 250-500m WQI for Israel coast.
    Used as fallback when S3 not available, or as supplement.
    Returns: (wqi_layer, df_beaches, error, age_hours, source_label)
    """
    wm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    t  = ee.Date(target_date_str)
    # Merge Terra (MOD) + Aqua (MYD) for better daily coverage
    now_m = datetime.utcnow()
    end_m = ee.Date(now_m.strftime("%Y-%m-%d")).advance(1,"day")
    start_m = ee.Date((now_m - timedelta(days=3)).strftime("%Y-%m-%d"))
    terra = (ee.ImageCollection("MODIS/061/MOD09GA")
             .filterBounds(HAIFA_BBOX)
             .filterDate(start_m, end_m))
    aqua  = (ee.ImageCollection("MODIS/061/MYD09GA")
             .filterBounds(HAIFA_BBOX)
             .filterDate(start_m, end_m))
    coll  = terra.merge(aqua).sort("system:time_start", False)

    if coll.size().getInfo() == 0:
        return None, None, "No MODIS data for this date.", None, "MODIS Terra+Aqua"

    img_first   = coll.first()
    img_time_ms = img_first.get("system:time_start").getInfo()
    img_dt      = datetime.utcfromtimestamp(img_time_ms / 1000)
    age_hours   = (datetime.utcnow() - img_dt).total_seconds() / 3600

    # Cloud mask: bits 0-1 of state_1km == 0 (clear)
    qa    = img_first.select("state_1km")
    clear = qa.bitwiseAnd(0b11).eq(0)
    img   = img_first.updateMask(clear).updateMask(wm)

    b1 = img.select("sur_refl_b01")  # 645nm red
    b2 = img.select("sur_refl_b02")  # 859nm NIR
    b4 = img.select("sur_refl_b04")  # 545nm green

    ndwi_n = b4.subtract(b2).divide(b4.add(b2)).unitScale(-0.3, 0.3).clamp(0, 1)
    chl_n  = b4.divide(b1.add(1e-6)).unitScale(0.8, 2.5).clamp(0, 1)
    turb_n = ee.Image(1).subtract(b1.unitScale(0, 1500)).clamp(0, 1)

    raw = ndwi_n.add(chl_n).add(turb_n).divide(3).multiply(100).rename("WQI")
    wqi = raw.clip(ISRAEL_CLIP).updateMask(wm)

    def _pt(pt):
        try:
            v  = wqi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee.Geometry.Point([pt["lon"], pt["lat"]]).buffer(500),
                scale=500, bestEffort=True).getInfo()
            wv = v.get("WQI")
            return {**pt, "wqi": round(wv, 1) if wv else None}
        except:
            return {**pt, "wqi": None}

    with ThreadPoolExecutor(max_workers=4) as ex:
        pts = list(ex.map(_pt, BEACHES))

    return wqi, pd.DataFrame(pts), None, round(age_hours, 1), "MODIS"



@st.cache_data(ttl=21600)
def process_israel_s2(target_date_str):
    """Sentinel-2 MSI SR - 10m WQI for Israel coast. Always uses latest available."""
    wm   = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    now  = datetime.utcnow()
    end  = ee.Date(now.strftime("%Y-%m-%d")).advance(1,"day")
    start= ee.Date((now - timedelta(days=10)).strftime("%Y-%m-%d"))
    coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(HAIFA_BBOX)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .sort("system:time_start", False))
    if coll.size().getInfo() == 0:
        return None, None, "No Sentinel-2 data.", None, "Sentinel-2"
    img_first   = coll.first()
    img_time_ms = img_first.get("system:time_start").getInfo()
    img_dt      = datetime.utcfromtimestamp(img_time_ms/1000)
    age_hours   = (datetime.utcnow()-img_dt).total_seconds()/3600
    water = img_first.select("SCL").eq(6)
    img   = img_first.updateMask(water).updateMask(wm)
    b3,b4,b5,b8,b8a = (img.select("B3").divide(10000), img.select("B4").divide(10000),
                        img.select("B5").divide(10000), img.select("B8").divide(10000),
                        img.select("B8A").divide(10000))
    ndwi_n = b3.subtract(b8).divide(b3.add(b8)).unitScale(-0.3,0.5).clamp(0,1)
    chl_n  = b5.divide(b4.add(1e-6)).unitScale(1.0,3.5).clamp(0,1)
    turb_n = ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0,0.15)).clamp(0,1)
    wqi    = ndwi_n.add(chl_n).add(turb_n).divide(3).multiply(100).clip(ISRAEL_CLIP).updateMask(wm).rename("WQI")
    def _pt(pt):
        try:
            v  = wqi.reduceRegion(reducer=ee.Reducer.mean(),
                geometry=ee.Geometry.Point([pt["lon"],pt["lat"]]).buffer(300),
                scale=10,bestEffort=True).getInfo()
            wv = v.get("WQI")
            return {**pt,"wqi":round(wv,1) if wv else None}
        except: return {**pt,"wqi":None}
    with ThreadPoolExecutor(max_workers=4) as ex:
        pts = list(ex.map(_pt,BEACHES))
    return wqi, pd.DataFrame(pts), None, round(age_hours,1), "Sentinel-2"









@st.cache_data(ttl=21600)
def get_available_dates_combined(days_back=7):
    """Returns list of dicts: {date, source} - S3 dates + daily MODIS fallback."""
    end      = datetime.utcnow()
    start    = end - timedelta(days=days_back)
    wide     = ee.Geometry.Rectangle([34.0, 29.0, 36.0, 33.5])
    date_fmt = "%Y-%m-%d"

    # S3 dates only (one GEE call)
    s3_coll  = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                .filterBounds(wide)
                .filterDate(start.strftime(date_fmt), end.strftime(date_fmt)))
    s3_ts    = s3_coll.aggregate_array("system:time_start").getInfo()
    s3_dates = set(datetime.utcfromtimestamp(d/1000).strftime(date_fmt) for d in s3_ts)

    # MODIS: assume available every day (no extra GEE call)
    all_dates = [(end - timedelta(days=i)).strftime(date_fmt) for i in range(days_back)]

    result = []
    for d in all_dates:
        if d in s3_dates:
            result.append({"date": d, "source": "S3",    "label": f"🛰️ {d} · S3"})
        else:
            result.append({"date": d, "source": "MODIS", "label": f"📡 {d} · MODIS"})
    return result


@st.cache_data(ttl=7200)
def process_israel_wqi(target_date_str):
    wm=ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    t=ee.Date(target_date_str)
    coll=(ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(HAIFA_BBOX)
          .filterDate(t.advance(-2,'day'),t.advance(1,'day')))
    if coll.size().getInfo()==0: return None,None,"No Sentinel-3 data for this date.",None
    # Get actual image acquisition time
    img_first = coll.sort("system:time_start", False).first()
    img_time_ms = img_first.get("system:time_start").getInfo()
    img_dt = datetime.utcfromtimestamp(img_time_ms / 1000)
    age_hours = (datetime.utcnow() - img_dt).total_seconds() / 3600

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
    return wqi, pd.DataFrame(pts), None, round(age_hours, 1)

@st.cache_data(ttl=14400)
def compute_beach_history_7d():
    """
    Compute WQI for each beach for each available date in last 14 days.
    Returns dict: {beach_name: [{date, wqi}, ...]}
    All dates computed in parallel per-beach via ThreadPoolExecutor.
    """
    end   = datetime.utcnow()
    start = end - timedelta(days=15)
    wide  = ee.Geometry.Rectangle([34.0, 29.0, 36.0, 33.5])
    wm_gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    # Ocean-only: exclude inland water (use SRTM elevation > 0 as land proxy)
    # Keep only pixels where distance to ocean shoreline is small
    # Simple: use GSW "transition" band - permanent sea water
    gsw_full = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    # type 1,2 = ocean/sea in some datasets; use permanent water near coast
    # Practical: exclude Kinneret, Dead Sea using bbox
    inland_mask = ee.Image(1).clip(ee.Geometry.Rectangle([35.3,32.6,35.7,33.0])).unmask(0)  # Kinneret
    inland_mask2= ee.Image(1).clip(ee.Geometry.Rectangle([35.3,31.0,35.6,31.9])).unmask(0)  # Dead Sea
    wm = wm_gsw.And(inland_mask.Not()).And(inland_mask2.Not())

    # Get S3 dates
    s3_coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
               .filterBounds(wide)
               .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
               .sort("system:time_start", False))
    s3_ts_list = s3_coll.aggregate_array("system:time_start").getInfo()
    s3_dates = set()
    for ts in s3_ts_list:
        s3_dates.add(datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d"))

    # S2 dates
    s2_coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(wide)
               .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
               .sort("system:time_start", False))
    s2_ts_list = s2_coll.aggregate_array("system:time_start").getInfo()
    s2_dates = set()
    for ts in s2_ts_list:
        s2_dates.add(datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d"))

    # All dates = S3 + S2 + every day in range (MODIS fallback)
    days_back = 15
    all_day_dates = [(end - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_back)]
    seen_dates = set()
    date_ts = []
    for d in all_day_dates:
        if d not in seen_dates:
            seen_dates.add(d)
            src = "S3" if d in s3_dates else "S2" if d in s2_dates else "MODIS"
            date_ts.append((d, src))

    if not date_ts:
        return {}

    def _wqi_for_date(args):
        """Compute WQI image for one date - S3 preferred, MODIS fallback."""
        date_str, source = args
        try:
            t = ee.Date(date_str)
            if source == "S2":
                try:
                    s2c = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                           .filterBounds(HAIFA_BBOX)
                           .filterDate(t.advance(-5,"day"),t.advance(1,"day"))
                           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",30))
                           .sort("system:time_start",False))
                    if s2c.size().getInfo() == 0: return date_str, None
                    im2 = s2c.first().updateMask(ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30))
                    b3,b4,b5,b8,b8a=(im2.select("B3").divide(10000),im2.select("B4").divide(10000),
                                     im2.select("B5").divide(10000),im2.select("B8").divide(10000),im2.select("B8A").divide(10000))
                    ndwi_n=b3.subtract(b8).divide(b3.add(b8)).unitScale(-0.3,0.5).clamp(0,1)
                    chl_n=b5.divide(b4.add(1e-6)).unitScale(1.0,3.5).clamp(0,1)
                    turb_n=ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0,0.15)).clamp(0,1)
                    wqi=ndwi_n.add(chl_n).add(turb_n).divide(3).multiply(100).rename("WQI").clip(ISRAEL_CLIP)
                    return date_str, wqi
                except: return date_str, None
            if source == "S3":
                coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                        .filterBounds(HAIFA_BBOX)
                        .filterDate(t.advance(-1,"day"), t.advance(1,"day")))
                if coll.size().getInfo() == 0:
                    source = "MODIS"
                else:
                    img  = coll.median().clip(ISRAEL_CLIP).updateMask(wm)
                    ndwi = img.normalizedDifference(["Oa06_radiance","Oa17_radiance"])
                    b10,b11,b12 = img.select("Oa10_radiance"),img.select("Oa11_radiance"),img.select("Oa12_radiance")
                    mci  = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
                    turb = img.select("Oa08_radiance")
                    raw  = (ndwi.unitScale(-0.2,0.5).clamp(0,1)
                            .add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1))
                            .add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1))
                            .divide(3).multiply(100).rename("WQI"))
                    wqi  = raw.reduceNeighborhood(
                        reducer=ee.Reducer.mean(),
                        kernel=ee.Kernel.square(radius=1,units="pixels")
                    ).rename("WQI").updateMask(wm)
                    return date_str, wqi
            if source == "MODIS":
                terra_h = ee.ImageCollection("MODIS/061/MOD09GA").filterBounds(HAIFA_BBOX).filterDate(t.advance(-1,"day"),t.advance(1,"day"))
                aqua_h  = ee.ImageCollection("MODIS/061/MYD09GA").filterBounds(HAIFA_BBOX).filterDate(t.advance(-1,"day"),t.advance(1,"day"))
                qa      = terra_h.merge(aqua_h).sort("system:time_start",False)
                if qa.size().getInfo() == 0:
                    return date_str, None
                img_m = qa.first()
                clear = img_m.select("state_1km").bitwiseAnd(0b11).eq(0)
                img_m = img_m.updateMask(clear).updateMask(wm)
                b1,b2,b4 = img_m.select("sur_refl_b01"),img_m.select("sur_refl_b02"),img_m.select("sur_refl_b04")
                ndwi_n = b4.subtract(b2).divide(b4.add(b2)).unitScale(-0.3,0.3).clamp(0,1)
                chl_n  = b4.divide(b1.add(1e-6)).unitScale(0.8,2.5).clamp(0,1)
                turb_n = ee.Image(1).subtract(b1.unitScale(0,1500)).clamp(0,1)
                wqi    = ndwi_n.add(chl_n).add(turb_n).divide(3).multiply(100).rename("WQI").clip(ISRAEL_CLIP).updateMask(wm)
                return date_str, wqi
        except:
            return date_str, None

    def _sample_beach_on_wqi(args):
        beach, date_str, wqi = args
        if wqi is None:
            return beach["name"], date_str, None
        try:
            v  = wqi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee.Geometry.Point([beach["lon"], beach["lat"]]).buffer(450),
                scale=300, bestEffort=True).getInfo()
            wv = v.get("WQI")
            return beach["name"], date_str, round(wv, 1) if wv else None
        except:
            return beach["name"], date_str, None

    # Compute WQI images for all dates (up to 4 in parallel)
    with ThreadPoolExecutor(max_workers=4) as ex:
        wqi_images = dict(ex.map(_wqi_for_date, date_ts))

    # Sample all beaches × all dates
    tasks = [(b, d, wqi_images.get(d)) for b in BEACHES for d, _ in date_ts]
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(_sample_beach_on_wqi, tasks))

    # Organize into dict
    history = {b["name"]: [] for b in BEACHES}
    for beach_name, date_str, wqi_val in results:
        history[beach_name].append({"date": date_str, "wqi": wqi_val})

    # Sort by date
    for name in history:
        history[name] = sorted(history[name], key=lambda x: x["date"])

    return history


@st.cache_data(ttl=7200)
def process_port_medi(port_key, target_date_str):
    """Compute WQI + SST anomaly for a specific port zone."""
    port = PORTS[port_key]
    bbox = port["bbox"]
    wm   = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    t    = ee.Date(target_date_str)

    # S3 WQI
    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
            .filterBounds(bbox)
            .filterDate(t.advance(-2,"day"), t.advance(1,"day")))
    if coll.size().getInfo() == 0:
        wqi_val, age_h = None, 48.0
    else:
        img_first  = coll.sort("system:time_start", False).first()
        img_time   = img_first.get("system:time_start").getInfo()
        age_h      = (datetime.utcnow() - datetime.utcfromtimestamp(img_time/1000)).total_seconds() / 3600
        img        = coll.median().clip(bbox).updateMask(wm)
        ndwi       = img.normalizedDifference(["Oa06_radiance","Oa17_radiance"])
        b10,b11,b12= img.select("Oa10_radiance"),img.select("Oa11_radiance"),img.select("Oa12_radiance")
        mci        = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
        turb       = img.select("Oa08_radiance")
        raw        = ndwi.unitScale(-0.2,0.5).clamp(0,1).add(
                        ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1)).add(
                        ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1)
                     ).divide(3).multiply(100).rename("WQI")
        try:
            val     = raw.reduceRegion(reducer=ee.Reducer.mean(), geometry=bbox, scale=300, bestEffort=True).getInfo()
            wqi_val = round(val.get("WQI"), 1) if val.get("WQI") else None
        except:
            wqi_val = None

    # MODIS SST anomaly
    try:
        today_sst  = (ee.ImageCollection("MODIS/061/MOD11A1")
                      .filterBounds(bbox)
                      .filterDate(t.advance(-2,"day"), t.advance(1,"day"))
                      .select("LST_Day_1km").mean().multiply(0.02).subtract(273.15).updateMask(wm))
        base_sst   = (ee.ImageCollection("MODIS/061/MOD11A1")
                      .filterBounds(bbox)
                      .filterDate(t.advance(-31,"day"), t.advance(-1,"day"))
                      .select("LST_Day_1km").mean().multiply(0.02).subtract(273.15).updateMask(wm))
        anom_val   = today_sst.subtract(base_sst).reduceRegion(
                        reducer=ee.Reducer.mean(), geometry=bbox, scale=1000, bestEffort=True).getInfo()
        sst_anom   = round(float(anom_val.get("LST_Day_1km")), 2) if anom_val.get("LST_Day_1km") else None
    except:
        sst_anom = None

    return wqi_val, sst_anom, round(age_h, 1)


@st.cache_data(ttl=14400)
def get_global_wqi_layer(target_date_str, bbox_rect):
    lon_min,lat_min,lon_max,lat_max=bbox_rect
    bbox=ee.Geometry.Rectangle([lon_min,lat_min,lon_max,lat_max])
    wm=ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
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
MODE_ISRAEL = "🏖️ Israel Coast"
MODE_GLOBAL = "🌍 Global"
mode = MODE_ISRAEL  # Default to Israel Coast

# Risk profile shown in MEDI tab only - initialized here for session state
medi_profile = "Beach Safety"  # default

# ── Israel Coast ──────────────────────────────────────────────────────────────



@st.cache_data(ttl=7200)
def compute_point_wqi(lat: float, lon: float, target_date_str: str, source: str = "S3") -> float | None:
    """
    WQI at nearest water pixel to (lat, lon).
    Uses a small buffer (500m) and takes only pixels where GSW >= 30%.
    Returns scalar WQI or None.
    """
    wm    = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    pt    = ee.Geometry.Point([lon, lat])
    buf   = pt.buffer(500)
    t     = ee.Date(target_date_str)

    try:
        if source == "S2":
            coll  = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                     .filterBounds(buf)
                     .filterDate(t.advance(-5,"day"), t.advance(1,"day"))
                     .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                     .sort("system:time_start", False))
            if coll.size().getInfo() == 0: return None
            img   = coll.first().updateMask(wm)
            b3,b4,b5,b8,b8a = (img.select("B3").divide(10000), img.select("B4").divide(10000),
                                img.select("B5").divide(10000), img.select("B8").divide(10000),
                                img.select("B8A").divide(10000))
            wqi   = (b3.subtract(b8).divide(b3.add(b8)).unitScale(-0.3,0.5).clamp(0,1)
                     .add(b5.divide(b4.add(1e-6)).unitScale(1.0,3.5).clamp(0,1))
                     .add(ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0,0.15)).clamp(0,1))
                     .divide(3).multiply(100).rename("WQI").updateMask(wm))
        else:
            coll  = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                     .filterBounds(buf)
                     .filterDate(t.advance(-2,"day"), t.advance(1,"day")))
            if coll.size().getInfo() == 0: return None
            img   = coll.median().updateMask(wm)
            ndwi  = img.normalizedDifference(["Oa06_radiance","Oa17_radiance"])
            b10,b11,b12 = img.select("Oa10_radiance"),img.select("Oa11_radiance"),img.select("Oa12_radiance")
            mci   = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
            turb  = img.select("Oa08_radiance")
            wqi   = (ndwi.unitScale(-0.2,0.5).clamp(0,1)
                     .add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1))
                     .add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1))
                     .divide(3).multiply(100).rename("WQI").updateMask(wm))

        val = wqi.reduceRegion(
            reducer   = ee.Reducer.mean(),
            geometry  = buf,
            scale     = 300,
            bestEffort= True
        ).getInfo()
        wv = val.get("WQI")
        return round(float(wv), 1) if wv is not None else None
    except:
        return None


@st.cache_data(ttl=7200)
def compute_city_wqi(target_date_str, source="S3"):
    """
    Compute WQI for each city's maritime zone polygon.
    Returns dict: {city_name: wqi_value}
    """
    wm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    t  = ee.Date(target_date_str)

    def _get_wqi_image():
        if source == "S2":
            coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(HAIFA_BBOX)
                    .filterDate(t.advance(-5,"day"),t.advance(1,"day"))
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",30))
                    .sort("system:time_start",False))
            if coll.size().getInfo() == 0: return None
            img   = coll.first().updateMask(wm)
            b3,b4,b5,b8,b8a = (img.select("B3").divide(10000),img.select("B4").divide(10000),
                                img.select("B5").divide(10000),img.select("B8").divide(10000),
                                img.select("B8A").divide(10000))
            return (b3.subtract(b8).divide(b3.add(b8)).unitScale(-0.3,0.5).clamp(0,1)
                    .add(b5.divide(b4.add(1e-6)).unitScale(1.0,3.5).clamp(0,1))
                    .add(ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0,0.15)).clamp(0,1))
                    .divide(3).multiply(100).rename("WQI").updateMask(wm))
        else:
            coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                    .filterBounds(HAIFA_BBOX)
                    .filterDate(t.advance(-2,"day"),t.advance(1,"day")))
            if coll.size().getInfo() == 0: return None
            img  = coll.median().clip(ISRAEL_CLIP).updateMask(wm)
            ndwi = img.normalizedDifference(["Oa06_radiance","Oa17_radiance"])
            b10,b11,b12 = img.select("Oa10_radiance"),img.select("Oa11_radiance"),img.select("Oa12_radiance")
            mci  = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
            turb = img.select("Oa08_radiance")
            raw  = (ndwi.unitScale(-0.2,0.5).clamp(0,1)
                    .add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1))
                    .add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1))
                    .divide(3).multiply(100).rename("WQI"))
            return raw.reduceNeighborhood(
                reducer=ee.Reducer.mean(),
                kernel=ee.Kernel.square(radius=1,units="pixels")
            ).rename("WQI").updateMask(wm)

    wqi_img = _get_wqi_image()
    if wqi_img is None:
        return {city: None for city in MARITIME_ZONES}

    results = {}
    for city, polygon in MARITIME_ZONES.items():
        try:
            val = wqi_img.reduceRegion(
                reducer  = ee.Reducer.mean(),
                geometry = polygon,
                scale    = 300,
                bestEffort=True
            ).getInfo()
            wv = val.get("WQI")
            results[city] = round(float(wv), 1) if wv else None
        except:
            results[city] = None

    return results


@st.cache_data(ttl=86400)
def compute_beach_history_range(days_back: int):
    """Compute WQI history for N days. S3+MODIS for <=7d, S3 only for longer."""
    end   = datetime.utcnow()
    start = end - timedelta(days=days_back+1)
    wide  = ee.Geometry.Rectangle([34.0,29.0,36.0,33.5])
    wm    = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)

    s3_coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
               .filterBounds(wide)
               .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
               .sort("system:time_start",False))
    s3_ts  = s3_coll.aggregate_array("system:time_start").getInfo()
    s3_set = set(datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d") for ts in s3_ts)

    if days_back <= 7:
        all_days = [(end-timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_back+1)]
        seen=set(); date_ts=[]
        for d in all_days:
            if d not in seen:
                seen.add(d)
                date_ts.append((d,"S3" if d in s3_set else "MODIS"))
    else:
        seen=set(); date_ts=[]
        for ts in s3_ts:
            d=datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d")
            if d not in seen:
                seen.add(d); date_ts.append((d,"S3"))

    if not date_ts: return {}

    def _wqi_for_date(args):
        date_str,source=args
        try:
            t=ee.Date(date_str)
            if source=="S3":
                coll=(ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(HAIFA_BBOX)
                      .filterDate(t.advance(-1,"day"),t.advance(1,"day")))
                if coll.size().getInfo()==0: return date_str,None
                img=coll.median().clip(ISRAEL_CLIP).updateMask(wm)
                ndwi=img.normalizedDifference(["Oa06_radiance","Oa17_radiance"])
                b10,b11,b12=img.select("Oa10_radiance"),img.select("Oa11_radiance"),img.select("Oa12_radiance")
                mci=b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
                turb=img.select("Oa08_radiance")
                raw=(ndwi.unitScale(-0.2,0.5).clamp(0,1)
                     .add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1))
                     .add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1))
                     .divide(3).multiply(100).rename("WQI"))
                wqi=raw.reduceNeighborhood(reducer=ee.Reducer.mean(),
                    kernel=ee.Kernel.square(radius=1,units="pixels")).rename("WQI").updateMask(wm)
                return date_str,wqi
            else:
                t2=ee.ImageCollection("MODIS/061/MOD09GA").filterBounds(HAIFA_BBOX).filterDate(t.advance(-1,"day"),t.advance(1,"day"))
                a2=ee.ImageCollection("MODIS/061/MYD09GA").filterBounds(HAIFA_BBOX).filterDate(t.advance(-1,"day"),t.advance(1,"day"))
                qa=t2.merge(a2).sort("system:time_start",False)
                if qa.size().getInfo()==0: return date_str,None
                im=qa.first(); cl=im.select("state_1km").bitwiseAnd(0b11).eq(0)
                im=im.updateMask(cl).updateMask(wm)
                b1,b2,b4=im.select("sur_refl_b01"),im.select("sur_refl_b02"),im.select("sur_refl_b04")
                wqi=(b4.subtract(b2).divide(b4.add(b2)).unitScale(-0.3,0.3).clamp(0,1)
                     .add(b4.divide(b1.add(1e-6)).unitScale(0.8,2.5).clamp(0,1))
                     .add(ee.Image(1).subtract(b1.unitScale(0,1500)).clamp(0,1))
                     .divide(3).multiply(100).rename("WQI").clip(ISRAEL_CLIP).updateMask(wm))
                return date_str,wqi
        except: return date_str,None

    def _sample(args):
        beach,date_str,wqi=args
        if wqi is None: return beach["name"],date_str,None
        try:
            v=wqi.reduceRegion(reducer=ee.Reducer.mean(),
              geometry=ee.Geometry.Point([beach["lon"],beach["lat"]]).buffer(450),
              scale=300,bestEffort=True).getInfo()
            wv=v.get("WQI")
            return beach["name"],date_str,round(wv,1) if wv else None
        except: return beach["name"],date_str,None

    with ThreadPoolExecutor(max_workers=4) as ex:
        wqi_images=dict(ex.map(_wqi_for_date,date_ts))
    tasks=[(b,d,wqi_images.get(d)) for b in BEACHES for d,_ in date_ts]
    with ThreadPoolExecutor(max_workers=6) as ex:
        results=list(ex.map(_sample,tasks))

    history={b["name"]:[] for b in BEACHES}
    for bn,ds,wv in results:
        history[bn].append({"date":ds,"wqi":wv})
    for n in history:
        history[n]=sorted(history[n],key=lambda x:x["date"])
    return history







if mode == MODE_ISRAEL:
    # Date selector
    # Auto-select latest available date
    with st.spinner("Finding latest satellite data..."):
        date_options = get_available_dates_combined()
    if date_options:
        sel_entry = date_options[0]  # always latest
        sel_date  = sel_entry["date"]
        sel_src   = sel_entry["source"]
    else:
        sel_date = (datetime.utcnow()-timedelta(days=1)).strftime('%Y-%m-%d')
        sel_src  = "S3"

    # Single MEDI Platform view
    # MEDI Platform - single view

    with st.spinner("Computing WQI from S3 · S2 · MODIS..."):
        # Pull all three satellites
        s3_layer,  s3_df,  s3_err,  s3_age            = process_israel_wqi(sel_date)
        s2_layer,  s2_df,  s2_err,  s2_age,  _        = process_israel_s2(sel_date)
        mod_layer, mod_df, mod_err, mod_age, mod_src   = process_modis_wqi(sel_date)

        # Build available images list (sorted by age, freshest first)
        all_candidates = []
        if not s3_err  and s3_layer  is not None and s3_age  is not None:
            all_candidates.append((s3_age,  s3_layer,  s3_df,  s3_age,  "S3",    "Sentinel-3"))
        if not s2_err  and s2_layer  is not None and s2_age  is not None:
            all_candidates.append((s2_age,  s2_layer,  s2_df,  s2_age,  "S2",    "Sentinel-2"))
        if not mod_err and mod_layer is not None and mod_age is not None:
            all_candidates.append((mod_age, mod_layer, mod_df, mod_age, "MOD",   mod_src))
        all_candidates.sort(key=lambda x: x[0])

        # Navigator state
        if "img_idx" not in st.session_state or st.session_state.get("img_total") != len(all_candidates):
            st.session_state.img_idx   = 0
            st.session_state.img_total = len(all_candidates)

        idx = st.session_state.img_idx
        if idx >= len(all_candidates): idx = 0

        if all_candidates:
            _, wqi_layer, df, img_age_hours, _, data_source = all_candidates[idx]
        img_age_hours = img_age_hours if img_age_hours else 99
        err = None

        # If best source has no beach df, try fallbacks
        if df is None or df.empty:
            for _, layer, df_fb, age_fb, src_fb in scores.values():
                if df_fb is not None and not df_fb.empty:
                    df = df_fb
                    break
        if df is None or df.empty:
            for fallback_days in [3, 5, 7, 10]:
                fb_date = (datetime.utcnow() - timedelta(days=fallback_days)).strftime('%Y-%m-%d')
                fb_layer, fb_df, fb_err, fb_age = process_israel_wqi(fb_date)
                if not fb_err and fb_df is not None and not fb_df.empty:
                    df = fb_df
                    if wqi_layer is None:
                        wqi_layer = fb_layer
                        img_age_hours = fb_age
                        data_source = "Sentinel-3"
                    break

    # History range will be shown above map
    if "history_range" not in st.session_state:
        st.session_state.history_range = "7 ימים"

    if err:
        st.error(err)
    elif wqi_layer is not None:
        atm = get_atm(32.4, 34.85)

    # Compute WQI for all monitoring points
    src_label = "S2" if data_source=="Sentinel-2" else "S3"
    if st.session_state.monitor_points:
        with st.spinner("Computing WQI for monitoring points..."):
            for pt_name, pt_data in st.session_state.monitor_points.items():
                wv = compute_point_wqi(
                    pt_data["lat"], pt_data["lon"],
                    sel_date, src_label
                )
                st.session_state.monitor_points[pt_name]["wqi"] = wv
            save_points(st.session_state.monitor_points)
        city_wqi = {n: d.get("wqi") for n,d in st.session_state.monitor_points.items()}
    else:
        city_wqi = {}

    # Legacy city WQI for default stats (fallback if no points defined)
    if not city_wqi:
        with st.spinner("Computing city maritime WQI..."):
            city_wqi = compute_city_wqi(
                sel_date,
                source=src_label
            )

    # Compute user-defined zone WQI
    user_zone_wqi = {}
    if st.session_state.get("user_zones"):
        wm_uz = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
        t_uz  = ee.Date(sel_date)
        try:
            s3c = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                   .filterBounds(HAIFA_BBOX)
                   .filterDate(t_uz.advance(-2,"day"),t_uz.advance(1,"day")))
            if s3c.size().getInfo() > 0:
                img_uz = s3c.median().clip(ISRAEL_CLIP).updateMask(wm_uz)
                ndwi_uz = img_uz.normalizedDifference(["Oa06_radiance","Oa17_radiance"])
                b10u,b11u,b12u = img_uz.select("Oa10_radiance"),img_uz.select("Oa11_radiance"),img_uz.select("Oa12_radiance")
                mci_uz = b11u.subtract(b10u.add(b12u.subtract(b10u).multiply((708.75-681.25)/(753.75-681.25))))
                turb_uz= img_uz.select("Oa08_radiance")
                wqi_uz = (ndwi_uz.unitScale(-0.2,0.5).clamp(0,1)
                          .add(ee.Image(1).subtract(mci_uz.unitScale(-2,12)).clamp(0,1))
                          .add(ee.Image(1).subtract(turb_uz.unitScale(10,80)).clamp(0,1))
                          .divide(3).multiply(100).rename("WQI").updateMask(wm_uz))
                for zname, zdata in st.session_state.user_zones.items():
                    try:
                        poly = ee.Geometry.Polygon([zdata["coords"]])
                        val  = wqi_uz.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=poly, scale=300, bestEffort=True
                        ).getInfo()
                        wv = val.get("WQI")
                        user_zone_wqi[zname] = round(float(wv),1) if wv else None
                    except:
                        user_zone_wqi[zname] = None
        except:
            pass

    # Load persistent monitoring points
    if "user_zones" not in st.session_state:
        st.session_state.user_zones = load_zones()
    if "monitor_points" not in st.session_state:
        st.session_state.monitor_points = load_points()
    if "pending_point" not in st.session_state:
        st.session_state.pending_point = None

    # Shared map builder
    def _build_map(selected_beach=None):
        m = folium.Map(location=[32.4, 34.85], zoom_start=8)
        # Add draw plugin
        Draw(
            export=False,
            draw_options={
                "polygon":   {"allowIntersection": False},
                "rectangle": True,
                "circle":    False,
                "polyline":  False,
                "marker":    False,
                "circlemarker": False,
            },
            edit_options={"edit": False}
        ).add_to(m)
        vis = {'min':30,'max':90,'palette':['#d73027','#f46d43','#fdae61','#fee090','#e0f3f8','#abd9e9','#74add1','#4575b4']}
        try:
            mid = ee.Image(wqi_layer).getMapId(vis)
            folium.TileLayer(tiles=mid['tile_fetcher'].url_format,
                             attr=f'GEE {data_source}',
                             name="WQI",overlay=True,control=False,opacity=0.85).add_to(m)
        except Exception:
            pass  # map shows base tiles only if GEE layer fails
        for _,r in df.iterrows():
            sc = r.get('wqi')
            cm = "#1ecb7b" if sc and sc>65 else "#f0a500" if sc and sc>45 else "#e03c3c"
            wqi_str = f"{sc:.1f}" if sc else "N/A"
            is_selected = selected_beach == r["name"]
            size = "20px" if is_selected else "14px"
            ring = f"box-shadow:0 0 0 3px white, 0 0 0 5px {cm};" if is_selected else f"box-shadow:0 0 8px {cm}99;"
            # Sampling zone circle (450m buffer)
            folium.Circle(
                location=[r["lat"], r["lon"]],
                radius=450,
                color=cm,
                fill=False,
                weight=1.5,
                opacity=0.6,
                tooltip=f"Sampling zone: 450m · WQI: {wqi_str}",
            ).add_to(m)
            folium.Marker(
                location=[r["lat"],r["lon"]],
                tooltip=f"🏖️ {r['name']} | WQI: {wqi_str} - Click for MEDI",
                popup=folium.Popup(f"<b>🏖️ {r['name']}</b><br>WQI: <span style='color:{cm};font-weight:bold;'>{wqi_str}</span><br><small>Sampling zone: 450m radius</small>", max_width=200),
                icon=folium.DivIcon(
                    html=f'''<div style="background:{cm};border:2px solid white;border-radius:50%;width:{size};height:{size};{ring};cursor:pointer;"></div>
<div style="position:absolute;top:18px;left:-32px;white-space:nowrap;font-family:Arial;font-size:11px;font-weight:700;color:white;text-shadow:0 1px 4px rgba(0,0,0,0.95);">{r["name"]}</div>''',
                    icon_size=(20,20),icon_anchor=(10,10))
            ).add_to(m)
        # Draw user monitoring points
        for pt_name, pt_data in st.session_state.monitor_points.items():
            pt_lat = pt_data["lat"]
            pt_lon = pt_data["lon"]
            pt_wqi = pt_data.get("wqi")
            cm_pt  = "#4575b4" if pt_wqi and pt_wqi>=70 else "#fdae61" if pt_wqi and pt_wqi>=50 else "#d73027" if pt_wqi else "#00c8c8"
            wqi_str= f"{pt_wqi:.1f}" if pt_wqi else "..."
            is_sel = (selected_beach == pt_name) if selected_beach else False
            size   = "16px" if is_sel else "11px"
            ring   = "box-shadow:0 0 0 3px white,0 0 0 5px "+cm_pt+";" if is_sel else ""
            folium.Marker(
                location=[pt_lat, pt_lon],
                tooltip=f"📍 {pt_name} | WQI: {wqi_str}",
                popup=folium.Popup(f"<b>{pt_name}</b><br>WQI: <b style='color:{cm_pt}'>{wqi_str}</b>", max_width=150),
                icon=folium.DivIcon(
                    html=f'''<div style="background:{cm_pt};border:2px solid white;border-radius:50%;width:{size};height:{size};{ring};cursor:pointer;"></div>
<div style="position:absolute;top:16px;left:-24px;white-space:nowrap;font-family:Arial;font-size:10px;font-weight:700;color:white;text-shadow:0 1px 3px rgba(0,0,0,0.9);">{pt_name} {wqi_str}</div>''',
                    icon_size=(16,16), icon_anchor=(8,8))
            ).add_to(m)

        # Draw user-defined zones
        for zone_name, zone_data in st.session_state.user_zones.items():
            coords = zone_data.get("coords", [])
            if coords:
                folium.Polygon(
                    locations=[[p[1],p[0]] for p in coords],
                    color="#00c8c8",
                    fill=True,
                    fill_color="#00c8c8",
                    fill_opacity=0.12,
                    weight=2,
                    dash_array="6 4",
                    tooltip=f"📍 {zone_name}",
                ).add_to(m)
                # Label at centroid
                lats = [p[1] for p in coords]
                lons = [p[0] for p in coords]
                folium.Marker(
                    location=[sum(lats)/len(lats), sum(lons)/len(lons)],
                    icon=folium.DivIcon(
                        html=f'''<div style="background:rgba(0,200,200,0.85);color:#020d18;font-family:Arial;font-size:10px;font-weight:700;padding:2px 6px;border-radius:3px;white-space:nowrap;">{zone_name}</div>''',
                        icon_size=(100,20), icon_anchor=(50,10))
                ).add_to(m)

        # No default city markers — user defines monitoring points

        m.add_child(OnMapWaterLegend())
        return m

    # ── MEDI Platform ─────────────────────────────────────────────────────────────
    if True:
        if err:
            st.error(err)
        elif wqi_layer is not None:
            acq_dt  = datetime.utcnow() - timedelta(hours=img_age_hours)
            acq_str = acq_dt.strftime("%Y-%m-%d %H:%M UTC")
            # Compact inline navigator
            src_colors = {"S3":"#00c8c8","S2":"#1ecb7b","MOD":"#f0a500"}
            if all_candidates:
                cur = all_candidates[st.session_state.img_idx]
                cur_dt = (datetime.utcnow()-timedelta(hours=cur[0])).strftime("%b %d %H:%M UTC")
                dots_html = ""
                for i,(age,_,_,_,short,_) in enumerate(all_candidates):
                    col_s = src_colors.get(short,"#888")
                    sz = "10px" if i == st.session_state.img_idx else "7px"
                    bd = "2px solid white" if i == st.session_state.img_idx else "none"
                    dots_html += f'<span style="display:inline-block;width:{sz};height:{sz};border-radius:50%;background:{col_s};border:{bd};margin:0 2px;vertical-align:middle;"></span>'

                nav_l, nav_center, nav_r = st.columns([1, 10, 1])
                with nav_l:
                    if st.button("◀", key="nav_prev", use_container_width=True):
                        n = len(all_candidates)
                        st.session_state.img_idx = (st.session_state.img_idx+1)%n
                        st.rerun()
                with nav_center:
                    st.markdown(
                        f'<div style="text-align:center;font-size:11px;color:#7fb3d3;padding:5px 0;">' +
                        dots_html +
                        f' <b style="color:#d6eaf8;">{cur[5]}</b> · {cur_dt} · {cur[0]:.0f}h ago</div>',
                        unsafe_allow_html=True
                    )
                with nav_r:
                    if st.button("▶", key="nav_next", use_container_width=True):
                        n = len(all_candidates)
                        st.session_state.img_idx = (st.session_state.img_idx-1)%n
                        st.rerun()

            # Always use 7-day history
            history_days  = 7
            history_label = "7 ימים"
            with st.spinner("Loading history..."):
                beach_history = compute_beach_history_range(history_days)
            col_map, col_info = st.columns([1, 1], gap="small")
            with col_map:
                map_data_wqi = st_folium(
                    _build_map(),
                    use_container_width=True, height=740,
                    key="israel_map_wqi",
                    returned_objects=["bounds","last_active_drawing","last_clicked"]
                )
            with col_info:
                # Detect map click → new monitoring point
                last_clicked = map_data_wqi.get("last_clicked") if map_data_wqi else None
                last_drawing  = map_data_wqi.get("last_active_drawing") if map_data_wqi else None
                if last_clicked and last_clicked.get("lat"):
                    clat = round(last_clicked["lat"], 5)
                    clon = round(last_clicked["lng"], 5)
                    # Only set if different from last pending
                    prev = st.session_state.pending_point
                    if prev is None or prev.get("lat") != clat or prev.get("lon") != clon:
                        st.session_state.pending_point = {"lat": clat, "lon": clon}
                        st.rerun()
                if last_drawing:
                    geom = last_drawing.get("geometry",{})
                    if geom.get("type") in ["Polygon","Rectangle"]:
                        st.session_state["pending_polygon"] = geom["coordinates"][0]

                # Detect which beaches are visible in current map bounds
                # Use city names from maritime zones (not beach points)
                bounds = map_data_wqi.get("bounds") if map_data_wqi else None
                if bounds and bounds.get("_southWest") and bounds.get("_northEast"):
                    sw = bounds["_southWest"]
                    ne = bounds["_northEast"]
                    lat_min = sw.get("lat", 29.0)
                    lat_max = ne.get("lat", 34.0)
                    lon_min = sw.get("lng", 34.0)
                    lon_max = ne.get("lng", 36.0)
                    filtered = [
                        city for city, pt in CITY_POINTS.items()
                        if lat_min <= pt["lat"] <= lat_max and lon_min <= pt["lon"] <= lon_max
                    ]
                    all_pts = list(st.session_state.monitor_points.keys()) or list(CITY_POINTS.keys())
                    visible_beaches = filtered if len(filtered) >= 2 else all_pts
                else:
                    visible_beaches = list(st.session_state.monitor_points.keys()) or list(CITY_POINTS.keys())

                # Build comparison chart for visible beaches
                if visible_beaches:
                    import json as _json

                    # Add current df as latest data point if history missing
                    if df is not None and not df.empty:
                        for _, row in df.iterrows():
                            if row["name"] in beach_history and row["wqi"]:
                                # Add if not already present
                                existing_dates = {e["date"] for e in beach_history[row["name"]]}
                                if sel_date not in existing_dates:
                                    beach_history[row["name"]].append({"date": sel_date, "wqi": row["wqi"]})
                            elif row["name"] not in beach_history and row["wqi"]:
                                beach_history[row["name"]] = [{"date": sel_date, "wqi": row["wqi"]}]

                    # Merge city_wqi into beach_history for chart
                    for city_name, cwqi in (city_wqi or {}).items():
                        if cwqi is not None:
                            if city_name not in beach_history:
                                beach_history[city_name] = []
                            existing = {e["date"] for e in beach_history[city_name]}
                            if sel_date not in existing:
                                beach_history[city_name].append({"date": sel_date, "wqi": cwqi})

                    # Merge user zones into beach_history for chart
                    for zname, zwqi in user_zone_wqi.items():
                        if zwqi is not None:
                            if zname not in beach_history:
                                beach_history[zname] = []
                            existing = {e["date"] for e in beach_history[zname]}
                            if sel_date not in existing:
                                beach_history[zname].append({"date": sel_date, "wqi": zwqi})
                            if zname not in visible_beaches:
                                visible_beaches.append(zname)

                    all_dates = sorted(set(
                        e["date"] for name in visible_beaches
                        for e in beach_history.get(name, [])
                    ))

                    def _get_current(name):
                        # City maritime zone WQI takes priority
                        if city_wqi and name in city_wqi and city_wqi[name] is not None:
                            return float(city_wqi[name])
                        # User zone
                        if user_zone_wqi and name in user_zone_wqi and user_zone_wqi[name] is not None:
                            return float(user_zone_wqi[name])
                        # History fallback
                        hist_vals = [e["wqi"] for e in beach_history.get(name,[]) if e["wqi"] and str(e["wqi"]) != "nan"]
                        return hist_vals[-1] if hist_vals else None

                    PALETTE = ["#1D9E75","#378ADD","#7F77DD","#BA7517","#D4537E","#E24B4A","#639922","#D85A30"]
                    beach_colors = {name: PALETTE[i % len(PALETTE)] for i,name in enumerate(visible_beaches)}
                    current_vals = {n: _get_current(n) for n in visible_beaches}
                    valid_vals   = {n:v for n,v in current_vals.items() if v}
                    best  = max(valid_vals, key=valid_vals.get) if valid_vals else None
                    worst = min(valid_vals, key=valid_vals.get) if valid_vals else None

                    datasets = []
                    for name in visible_beaches:
                        hist_map = {e["date"]:e["wqi"] for e in beach_history.get(name,[])}
                        data = [hist_map.get(d) for d in all_dates]
                        datasets.append({
                            "label": name,
                            "data": data,
                            "borderColor": beach_colors[name],
                            "borderDash": [5,3] if (valid_vals.get(name,100) or 100) < 30 else [],
                        })

                    legend_items = []
                    for name in visible_beaches:
                        v   = current_vals.get(name)
                        col = "#1ecb7b" if v and v>=70 else "#f0a500" if v and v>=55 else "#e03c3c" if v else "#888"
                        legend_items.append({
                            "name": name,
                            "color": beach_colors[name],
                            "wqi": round(v,1) if v else "---",
                            "wqiColor": col,
                        })

                    chart_json  = _json.dumps(datasets)
                    labels_json = _json.dumps([d[5:] for d in all_dates])
                    legend_json = _json.dumps(legend_items)
                    best_name   = best or "---"
                    best_val    = round(valid_vals[best],1) if best else "---"
                    worst_name  = worst or "---"
                    worst_val   = round(valid_vals[worst],1) if worst else "---"
                    n_beaches   = len(visible_beaches)

                    # Coast statistics
                    cst_valid   = {k:v for k,v in (city_wqi or {}).items() if v is not None}
                    cst_avg     = f"{sum(cst_valid.values())/len(cst_valid):.1f}" if cst_valid else "N/A"
                    cst_best    = max(cst_valid, key=cst_valid.get) if cst_valid else "N/A"
                    cst_best_v  = f"{cst_valid[cst_best]:.1f}" if cst_valid else ""
                    cst_worst   = min(cst_valid, key=cst_valid.get) if cst_valid else "N/A"
                    cst_worst_v = f"{cst_valid[cst_worst]:.1f}" if cst_valid else ""
                    cst_nclean  = sum(1 for v in cst_valid.values() if v>=70)
                    cst_nmod    = sum(1 for v in cst_valid.values() if 50<=v<70)
                    cst_npoll   = sum(1 for v in cst_valid.values() if v<50)

                    chart_html = f"""
<!DOCTYPE html><html><body style="margin:0;padding:0;background:#020d18;overflow:hidden;">
<div style="padding:0.25rem 0 0.5rem;height:100vh;display:flex;flex-direction:column;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
    <p style="font-size:12px;color:#7fb3d3;margin:0;">איכות פני המים · {history_label} · Sentinel-3</p>
    <p style="font-size:11px;color:#7fb3d3;margin:0;">{n_beaches} חופים</p>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-bottom:7px;">
    <div style="background:rgba(0,200,200,0.06);border:1px solid rgba(0,200,200,0.15);border-radius:5px;padding:5px;text-align:center;">
      <p style="font-size:9px;color:#7fb3d3;margin:0;">ממוצע חוף ישראל</p>
      <p style="font-size:20px;font-weight:700;margin:1px 0;color:#d6eaf8;">{cst_avg}</p>
      <p style="font-size:9px;color:#7fb3d3;margin:0;">WQI</p>
    </div>
    <div style="background:rgba(69,117,180,0.08);border:1px solid rgba(69,117,180,0.2);border-radius:5px;padding:5px;text-align:center;">
      <p style="font-size:9px;color:#7fb3d3;margin:0;">הכי נקי</p>
      <p style="font-size:11px;font-weight:600;margin:1px 0;color:#4575b4;">{cst_best}</p>
      <p style="font-size:15px;font-weight:700;margin:0;color:#4575b4;">{cst_best_v}</p>
    </div>
    <div style="background:rgba(215,48,39,0.08);border:1px solid rgba(215,48,39,0.2);border-radius:5px;padding:5px;text-align:center;">
      <p style="font-size:9px;color:#7fb3d3;margin:0;">הכי מזוהם</p>
      <p style="font-size:11px;font-weight:600;margin:1px 0;color:#d73027;">{cst_worst}</p>
      <p style="font-size:15px;font-weight:700;margin:0;color:#d73027;">{cst_worst_v}</p>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-bottom:8px;">
    <div style="background:rgba(69,117,180,0.12);border-radius:4px;padding:3px;text-align:center;">
      <span style="font-size:15px;font-weight:700;color:#4575b4;">{cst_nclean}</span>
      <span style="font-size:9px;color:#7fb3d3;"> נקיים</span>
    </div>
    <div style="background:rgba(253,174,97,0.12);border-radius:4px;padding:3px;text-align:center;">
      <span style="font-size:15px;font-weight:700;color:#fdae61;">{cst_nmod}</span>
      <span style="font-size:9px;color:#7fb3d3;"> בינוניים</span>
    </div>
    <div style="background:rgba(215,48,39,0.12);border-radius:4px;padding:3px;text-align:center;">
      <span style="font-size:15px;font-weight:700;color:#d73027;">{cst_npoll}</span>
      <span style="font-size:9px;color:#7fb3d3;"> מזוהמים</span>
    </div>
  </div>
  <div style="display:flex;gap:10px;align-items:flex-start;">
    <div style="position:relative;flex:1;min-height:600px;height:calc(100vh - 100px);">
      <canvas id="beachTrend" role="img" aria-label="Water quality trends for {n_beaches} beaches"></canvas>
    </div>
    <div id="beachLegend" style="display:flex;flex-direction:column;justify-content:space-around;min-height:600px;height:calc(100vh - 100px);min-width:110px;"></div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
(function(){{
  var ds={chart_json};
  var lb={labels_json};
  var lg={legend_json};
  var gc='rgba(255,255,255,0.08)';
  var tc='#aaaaaa';
  ds=ds.map(d=>({{...d,backgroundColor:'transparent',tension:0.35,pointRadius:3,
    pointBackgroundColor:d.borderColor,borderWidth:2,spanGaps:true}}));
  new Chart(document.getElementById('beachTrend'),{{
    type:'line',data:{{labels:lb,datasets:ds}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>`${{c.dataset.label}}: ${{c.parsed.y}}`}}}}}},
      scales:{{
        x:{{
          ticks:{{color:'#cccccc',font:{{size:13,weight:'bold'}},maxRotation:45,autoSkip:false}},
          grid:{{color:'rgba(255,255,255,0.08)'}},
          title:{{display:true,text:'תאריך',color:'#cccccc',font:{{size:12,weight:'bold'}}}}
        }},
        y:{{min:1,max:100,
          ticks:{{color:'#cccccc',font:{{size:13,weight:'bold'}},
            callback:function(v){{
              if(v===1) return 'מזוהם 1';
              if(v===25) return '25';
              if(v===50) return '50';
              if(v===75) return '75';
              if(v===100) return 'נקי 100';
              return '';
            }}}},
          grid:{{color:'rgba(255,255,255,0.08)'}},
          title:{{display:true,text:'איכות המים (WQI)',color:'#cccccc',font:{{size:12,weight:'bold'}}}}
        }}
      }}
    }}
  }});
  var el=document.getElementById('beachLegend');
  lg.forEach(function(item){{
    var r=document.createElement('div');
    r.style.cssText='display:flex;align-items:center;gap:5px;';
    r.innerHTML=`<span style="width:16px;height:2px;background:${{item.color}};flex-shrink:0;border-radius:1px;"></span>
      <span style="font-size:10px;color:#7fb3d3;flex:1;">${{item.name}}</span>
      <span style="font-size:11px;font-weight:600;color:${{item.wqiColor}};">${{item.wqi}}</span>`;
    el.appendChild(r);
  }});
}})();
</script></body></html>
"""
                    components.html(chart_html, height=740, scrolling=False)
                else:
                    st.caption("Zoom in to see beach comparison")

                # ── Monitoring Point Manager ──────────────────────────────────
                with st.expander("📍 Monitoring Points", expanded=bool(st.session_state.pending_point)):
                    # New point dialog
                    if st.session_state.pending_point:
                        pp = st.session_state.pending_point
                        st.info(f"📍 New point: {pp['lat']:.4f}, {pp['lon']:.4f}")
                        pt_name_inp = st.text_input("Name:", key="pt_name_inp", placeholder="e.g. Haifa Port")
                        cs, cc = st.columns(2)
                        with cs:
                            if st.button("💾 Save", use_container_width=True, key="save_pt"):
                                if pt_name_inp.strip():
                                    st.session_state.monitor_points[pt_name_inp.strip()] = {
                                        "lat": pp["lat"], "lon": pp["lon"], "wqi": None
                                    }
                                    save_points(st.session_state.monitor_points)
                                    st.session_state.pending_point = None
                                    st.rerun()
                        with cc:
                            if st.button("✕", use_container_width=True, key="cancel_pt"):
                                st.session_state.pending_point = None
                                st.rerun()

                    # List existing points
                    if st.session_state.monitor_points:
                        for pname in list(st.session_state.monitor_points.keys()):
                            pd_data = st.session_state.monitor_points[pname]
                            wv_str  = f"{pd_data['wqi']:.1f}" if pd_data.get("wqi") else "..."
                            cp, cd  = st.columns([3, 1])
                            with cp:
                                st.caption(f"📍 {pname} — WQI: {wv_str}")
                            with cd:
                                if st.button("🗑", key=f"del_pt_{pname}"):
                                    del st.session_state.monitor_points[pname]
                                    save_points(st.session_state.monitor_points)
                                    st.rerun()
                    else:
                        st.caption("Click anywhere on the sea to add a monitoring point")



# ── Global ────────────────────────────────────────────────────────────────────
else:
    available_dates = [(datetime.utcnow()-timedelta(days=d)).strftime('%Y-%m-%d') for d in range(1,8)]
    sel_date = st.sidebar.selectbox("Select acquisition date:",[f"🟢 {d}" for d in available_dates]).replace("🟢 ","")

    st.markdown("### 🌍 Global WQI - Sentinel-3")

    if "g_center" not in st.session_state:
        st.session_state.g_center=(24.0,-90.0); st.session_state.g_zoom=5
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
