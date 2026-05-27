"""
app.py
=============================================================================
AquaWatch Global — Water Quality Monitor Dashboard
Multi-satellite fusion: Sentinel-3, Sentinel-2, MODIS
=============================================================================
"""

import math
import json
import tempfile
import os
from typing import Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
import ee
from branca.element import MacroElement
from jinja2 import Template

st.set_page_config(page_title="AquaWatch Global — Sentinel WQI", layout="wide")

# ==============================================================================
# Analytics
# ==============================================================================
st.markdown('<meta name="google-site-verification" content="INSERT_GOOGLE_VERIFICATION_CODE_HERE" />', unsafe_allow_html=True)
components.html('<script async src="https://cloud.umami.is/script.js" data-website-id="07a48db1-5aa7-4d88-aaac-9cfb6fc2600d"></script>', height=0)
if "ga_loaded" not in st.session_state:
    st.session_state.ga_loaded = True
    components.html("""
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-K37THY2160"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
          gtag('config', 'G-K37THY2160');
        </script>""", height=0)

# =============================================================================
# 1. Atmospheric & Marine Context
# =============================================================================
_WB_CENTRES = {
    "🏖️ Mediterranean Coast": (32.40, 34.85),
    "🌊 Sea of Galilee":      (32.82, 35.59),
    "🧂 Dead Sea":            (31.50, 35.47),
    "🐠 Red Sea":             (29.55, 34.95),
}

def _empty_atm() -> dict:
    return {"wind_speed": None, "wind_dir_deg": None, "temp_c": None,
            "humidity": None, "precip_mm": None, "weather_code": None,
            "analysis_time": None, "centre_lat": None, "centre_lon": None,
            "_error": None, "_source": "Open-Meteo (GFS/ERA5)"}

@st.cache_data(ttl=3600)
def get_atmospheric_context(wb_key: str) -> dict:
    empty = _empty_atm()
    lat, lon = _WB_CENTRES.get(wb_key, (32.0, 35.0))
    try:
        import requests as _req
        resp = _req.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,weather_code",
            "wind_speed_unit": "ms", "forecast_days": 1}, timeout=10)
        resp.raise_for_status()
        cur = resp.json().get("current", {})
        ws, wd, tc = cur.get("wind_speed_10m"), cur.get("wind_direction_10m"), cur.get("temperature_2m")
        rh, pr, wc = cur.get("relative_humidity_2m"), cur.get("precipitation"), cur.get("weather_code", 0)
        return {"wind_speed": round(ws,1) if ws else None, "wind_dir_deg": round(wd,1) if wd else None,
                "temp_c": round(tc,1) if tc else None, "humidity": round(rh,0) if rh else None,
                "precip_mm": round(pr,2) if pr else None, "weather_code": wc,
                "analysis_time": cur.get("time","—"), "centre_lat": lat, "centre_lon": lon,
                "_error": None, "_source": "Open-Meteo (GFS/ERA5)"}
    except Exception as exc:
        empty["_error"] = f"Error: {exc}"; return empty

@st.cache_data(ttl=3600)
def get_atmospheric_context_by_coords(lat: float, lon: float) -> dict:
    empty = _empty_atm()
    try:
        import requests as _req
        resp = _req.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": round(lat,4), "longitude": round(lon,4),
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,weather_code",
            "wind_speed_unit": "ms", "forecast_days": 1}, timeout=10)
        resp.raise_for_status()
        cur = resp.json().get("current", {})
        ws, wd, tc = cur.get("wind_speed_10m"), cur.get("wind_direction_10m"), cur.get("temperature_2m")
        rh, pr, wc = cur.get("relative_humidity_2m"), cur.get("precipitation"), cur.get("weather_code", 0)
        return {"wind_speed": round(ws,1) if ws else None, "wind_dir_deg": round(wd,1) if wd else None,
                "temp_c": round(tc,1) if tc else None, "humidity": round(rh,0) if rh else None,
                "precip_mm": round(pr,2) if pr else None, "weather_code": wc,
                "analysis_time": cur.get("time","—"), "centre_lat": lat, "centre_lon": lon,
                "_error": None, "_source": "Open-Meteo (GFS/ERA5)"}
    except Exception as exc:
        empty["_error"] = f"Error: {exc}"; return empty

@st.cache_data(ttl=3600)
def get_marine_grid(lat_min, lat_max, lon_min, lon_max, steps=6) -> dict:
    import requests as _req
    lats = [round(lat_min+(lat_max-lat_min)*i/(steps-1),4) for i in range(steps)]
    lons = [round(lon_min+(lon_max-lon_min)*i/(steps-1),4) for i in range(steps)]
    points = [(la,lo) for la in lats for lo in lons]
    n = len(points)
    all_lats = ",".join(str(p[0]) for p in points)
    all_lons = ",".join(str(p[1]) for p in points)
    wind_u=[0.0]*n; wind_v=[0.0]*n; wave_u=[0.0]*n; wave_v=[0.0]*n
    try:
        r = _req.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": all_lats, "longitude": all_lons,
            "current": "wind_speed_10m,wind_direction_10m", "wind_speed_unit": "ms", "forecast_days": 1}, timeout=12)
        results = r.json()
        if isinstance(results, dict): results = [results]
        for i,res in enumerate(results):
            cur=res.get("current",{}); ws=cur.get("wind_speed_10m") or 0.0; wd=cur.get("wind_direction_10m") or 0.0
            rad=math.radians(wd); wind_u[i]=round(-ws*math.sin(rad),3); wind_v[i]=round(-ws*math.cos(rad),3)
    except Exception: pass
    try:
        r2 = _req.get("https://marine-api.open-meteo.com/v1/marine", params={
            "latitude": all_lats, "longitude": all_lons,
            "current": "wave_height,wave_direction", "forecast_days": 1}, timeout=12)
        results2 = r2.json()
        if isinstance(results2, dict): results2 = [results2]
        for i,res in enumerate(results2):
            cur=res.get("current",{}); wh=cur.get("wave_height") or 0.0; wdir=cur.get("wave_direction") or 0.0
            rad=math.radians(wdir); wave_u[i]=round(-wh*math.sin(rad),3); wave_v[i]=round(-wh*math.cos(rad),3)
    except Exception: pass
    def _lv(u,v):
        h={"parameterCategory":2,"parameterNumber":2,"parameterUnit":"m/s",
           "la1":lat_max,"la2":lat_min,"lo1":lon_min,"lo2":lon_max,"nx":steps,"ny":steps,
           "dx":round((lon_max-lon_min)/max(steps-1,1),4),"dy":round((lat_max-lat_min)/max(steps-1,1),4),
           "refTime":datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}
        return [{"header":h,"data":u},{"header":h,"data":v}]
    return {"wind":_lv(wind_u,wind_v),"waves":_lv(wave_u,wave_v)}

@st.cache_data(ttl=3600)
def get_sst_for_points(points: tuple) -> dict:
    import requests as _req
    if not points: return {}
    all_lats=",".join(str(p[0]) for p in points); all_lons=",".join(str(p[1]) for p in points)
    try:
        r=_req.get("https://marine-api.open-meteo.com/v1/marine",params={
            "latitude":all_lats,"longitude":all_lons,"current":"sea_surface_temperature","forecast_days":1},timeout=12)
        results=r.json()
        if isinstance(results,dict): results=[results]
        out={}
        for i,res in enumerate(results):
            sst=res.get("current",{}).get("sea_surface_temperature")
            out[points[i]]=round(sst,1) if sst is not None else None
        return out
    except Exception: return {p:None for p in points}

def haversine_km(lat1,lon1,lat2,lon2):
    R=6371.0; phi1,phi2=math.radians(lat1),math.radians(lat2)
    dphi=math.radians(lat2-lat1); dlam=math.radians(lon2-lon1)
    a=math.sin(dphi/2)**2+math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

def blend_atmospheric_penalty(df, atm, wb_key):
    def _pen(row):
        if row.get("wqi") is None: return 0.0
        p=0.0; ws=atm.get("wind_speed"); pr=atm.get("precip_mm"); rh=atm.get("humidity")
        if ws:
            if ws>15: p+=10.0
            elif ws>10: p+=7.0
            elif ws>7: p+=4.0
            elif ws>4: p+=1.5
        if pr:
            if pr>5: p+=8.0
            elif pr>2: p+=5.0
            elif pr>0.5: p+=2.0
        if wb_key in {"🌊 Sea of Galilee","🧂 Dead Sea","🐠 Red Sea"} and rh:
            if rh>85: p+=7.0
            elif rh>70: p+=3.0
        return min(p,25.0)
    df=df.copy(); df["atm_penalty"]=df.apply(_pen,axis=1)
    df["composite_with_atm"]=df.apply(lambda r: round(max(0.0,r["wqi"]-r["atm_penalty"]),1) if r["wqi"] is not None else None,axis=1)
    return df

# =============================================================================
# 2. GEE Auth
# =============================================================================
@st.cache_resource
def init_gee():
    creds_dict=dict(st.secrets["gee_credentials"]); creds_json=json.dumps(creds_dict)
    with tempfile.NamedTemporaryFile(mode="w",suffix=".json",delete=False) as f:
        f.write(creds_json); tmp_path=f.name
    credentials=ee.ServiceAccountCredentials(creds_dict["client_email"],tmp_path)
    ee.Initialize(credentials); os.unlink(tmp_path)
init_gee()

# =============================================================================
# 3. Geometries
# =============================================================================
HAIFA_CENTER=[32.4,34.85]
HAIFA_BBOX=ee.Geometry.Rectangle([34.20,31.20,35.20,33.20])
ISRAEL_TERRITORIAL=ee.Geometry.Polygon([[[34.95,33.10],[34.55,33.10],[34.15,32.50],[34.10,32.00],
    [34.15,31.50],[34.50,31.25],[34.75,31.25],[34.95,31.30],[35.02,31.60],[35.00,32.10],[35.05,32.60],[35.10,33.10],[34.95,33.10]]])
KINNERET_BBOX=ee.Geometry.Rectangle([35.48,32.70,35.68,32.95])
DEAD_SEA_BBOX=ee.Geometry.Rectangle([35.35,31.05,35.58,31.80])
RED_SEA_BBOX=ee.Geometry.Rectangle([34.85,29.40,35.02,29.60])
BEACHES=[
    {"name":"Rosh HaNikra","lat":33.0765,"lon":35.0983},{"name":"Nahariya","lat":33.0048,"lon":35.0832},
    {"name":"Acre","lat":32.9280,"lon":35.0680},{"name":"Haifa North","lat":32.8380,"lon":34.9820},
    {"name":"Atlit","lat":32.6892,"lon":34.9368},{"name":"Caesarea","lat":32.4948,"lon":34.8912},
    {"name":"Netanya","lat":32.3318,"lon":34.8512},{"name":"Herzliya","lat":32.1648,"lon":34.7962},
    {"name":"Tel Aviv Center","lat":32.0798,"lon":34.7618},{"name":"Ashdod","lat":31.7848,"lon":34.6248},
    {"name":"Ashkelon","lat":31.6548,"lon":34.5448},{"name":"Zikim","lat":31.6098,"lon":34.5198},
]
WATER_BODIES={
    "🏖️ Mediterranean Coast":{"center":HAIFA_CENTER,"zoom":8,"bbox":HAIFA_BBOX,"clip_geom":ISRAEL_TERRITORIAL,"points":BEACHES},
    "🌊 Sea of Galilee":{"center":[32.82,35.59],"zoom":12,"bbox":KINNERET_BBOX,"clip_geom":KINNERET_BBOX,"points":[{"name":"Tiberias","lat":32.794,"lon":35.534},{"name":"North Sea of Galilee","lat":32.920,"lon":35.595}]},
    "🧂 Dead Sea":{"center":[31.50,35.47],"zoom":11,"bbox":DEAD_SEA_BBOX,"clip_geom":DEAD_SEA_BBOX,"points":[{"name":"Ein Gedi","lat":31.462,"lon":35.388},{"name":"Ein Bokek","lat":31.198,"lon":35.352}]},
    "🐠 Red Sea":{"center":[29.55,34.95],"zoom":13,"bbox":RED_SEA_BBOX,"clip_geom":RED_SEA_BBOX,"points":[{"name":"Gulf of Eilat","lat":29.530,"lon":34.951}]},
}

# =============================================================================
# 4. Map UI Components
# =============================================================================
class OnMapAtmosphereControl(MacroElement):
    def __init__(self, atm):
        super().__init__()
        ws=atm.get("wind_speed"); wd=atm.get("wind_dir_deg"); tc=atm.get("temp_c")
        pr=atm.get("precip_mm"); rh=atm.get("humidity")
        ar=int(wd) if wd else 0
        ws_s=f"{ws:.1f} m/s" if ws else "—"; tc_s=f"{tc:.1f}°C" if tc else "—"
        pr_s=f"{pr:.1f} mm" if pr else "—"; rh_s=f"{int(rh)}%" if rh else "—"
        ri="🌧️" if pr and pr>0.5 else "☀️"
        bf=12
        if ws:
            for b,t in enumerate([0.3,1.6,3.4,5.5,8.0,10.8,13.9,17.2,20.8,24.5,28.5,32.7]):
                if ws<t: bf=b; break
        bc="#27AE60" if bf<4 else "#F39C12" if bf<7 else "#E74C3C"
        html=f"""<div style="background:rgba(255,255,255,0.93);border:1.5px solid #aaa;border-radius:10px;padding:10px 13px;font-family:Arial,sans-serif;font-size:13px;box-shadow:0 2px 10px rgba(0,0,0,0.18);min-width:155px;">
<div style="font-weight:bold;margin-bottom:7px;text-align:center;font-size:12px;color:#555;">🌍 Atmospheric Context</div>
<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;"><svg width="28" height="28" viewBox="0 0 28 28"><g transform="rotate({ar},14,14)"><polygon points="14,2 18,22 14,18 10,22" fill="#2980B9" opacity="0.85"/></g></svg>
<div><div style="font-size:13px;font-weight:bold;">{ws_s}</div><div style="font-size:11px;color:{bc};">Beaufort {bf}</div></div></div>
<hr style="margin:5px 0;border-color:#eee;">
<div style="display:flex;justify-content:space-between;margin-bottom:3px;"><span>🌡️</span><span style="font-weight:bold;">{tc_s}</span></div>
<div style="display:flex;justify-content:space-between;margin-bottom:3px;"><span>{ri}</span><span style="font-weight:bold;">{pr_s}</span></div>
<div style="display:flex;justify-content:space-between;"><span>💧</span><span style="font-weight:bold;">{rh_s}</span></div></div>"""
        self._template=Template("""{% macro script(this, kwargs) %}
var ac=L.control({position:'bottomleft'});
ac.onAdd=function(map){var d=L.DomUtil.create('div','ac');d.innerHTML=`"""+html.replace("`","'")+"""`;L.DomEvent.disableClickPropagation(d);return d;};
ac.addTo({{this._parent.get_name()}});{% endmacro %}""")

class OnMapWaterLegend(MacroElement):
    def __init__(self):
        super().__init__()
        self._template=Template("""{% macro script(this, kwargs) %}
var lg=L.control({position:'topright'});
lg.onAdd=function(map){var d=L.DomUtil.create('div','info legend');
d.style.cssText='background:rgba(255,255,255,0.9);padding:12px;border:2px solid #999;border-radius:8px;font-family:Arial,sans-serif;font-size:13px;box-shadow:0 0 15px rgba(0,0,0,0.2)';
d.innerHTML=`<div style="font-weight:bold;margin-bottom:8px;text-align:center;">Water Quality Index</div>
<div style="display:flex;align-items:center;gap:8px;"><div style="height:150px;width:16px;background:linear-gradient(to bottom,#00FF00,#FFFF00,#FF0000);border:1px solid #666;border-radius:3px;flex-shrink:0;"></div>
<div style="display:flex;flex-direction:column;justify-content:space-between;height:150px;font-size:11px;font-weight:bold;">
<span style="color:green;">Clean</span><span style="color:orange;">Moderate</span><span style="color:red;">Polluted</span></div></div>`;return d;};
lg.addTo({{this._parent.get_name()}});{% endmacro %}""")

class OnMapVelocityLayer(MacroElement):
    def __init__(self, velocity_data, layer_name="Wind", color="#ffffff"):
        super().__init__()
        dj=json.dumps(velocity_data)
        self._template=Template("""{% macro header(this, kwargs) %}
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet-velocity@2.1.2/dist/leaflet-velocity.min.css"/>
<script src="https://cdn.jsdelivr.net/npm/leaflet-velocity@2.1.2/dist/leaflet-velocity.min.js"></script>
{% endmacro %}{% macro script(this, kwargs) %}
(function(){var vd="""+dj+""";var vl=L.velocityLayer({displayValues:true,
displayOptions:{velocityType:\"""")+Template(layer_name+"""\",displayPosition:'bottomleft',displayEmptyString:'No data'},
data:vd,maxVelocity:15,velocityScale:0.012,
colorScale:['#3288bd','#66c2a5','#abdda4','#e6f598','#fee08b','#fdae61','#f46d43','#d53e4f'],opacity:0.85});
vl.addTo({{this._parent.get_name()}});})();{% endmacro %}""")

# =============================================================================
# 5. Multi-Satellite Fusion Pipeline
# =============================================================================
ISRAEL_COAST_BBOX={"lon_min":34.15,"lon_max":35.10,"lat_min":29.40,"lat_max":33.15}
GRID_STEP_DEG=0.0027
RESOLUTION_M={"S3":300,"MODIS":250,"S2":10}
GSW=ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(25)

def build_grid(bbox=None):
    if bbox is None: bbox=ISRAEL_COAST_BBOX
    step=bbox.get("step", GRID_STEP_DEG)
    lats=np.arange(bbox["lat_min"],bbox["lat_max"],step)
    lons=np.arange(bbox["lon_min"],bbox["lon_max"],step)
    records=[]
    for i,lat in enumerate(lats):
        for j,lon in enumerate(lons):
            records.append({"cell_id":f"{i:04d}_{j:04d}","lat":round(float(lat),6),"lon":round(float(lon),6)})
    return pd.DataFrame(records)

def _get_latest(cid, aoi, days_back=10):
    end=datetime.utcnow(); start=end-timedelta(days=days_back)
    coll=(ee.ImageCollection(cid).filterBounds(aoi)
          .filterDate(start.strftime("%Y-%m-%d"),end.strftime("%Y-%m-%d"))
          .sort("system:time_start",False))
    return None if coll.size().getInfo()==0 else coll.first()

def get_s3_fusion_layer(aoi):
    img=_get_latest("COPERNICUS/S3/OLCI",aoi,days_back=5)
    if img is None: return None
    img=img.updateMask(GSW)
    ndwi=img.normalizedDifference(["Oa06_radiance","Oa17_radiance"])
    b10,b11,b12=img.select("Oa10_radiance"),img.select("Oa11_radiance"),img.select("Oa12_radiance")
    mci=b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25))))
    turb=img.select("Oa08_radiance")
    wqi=ndwi.unitScale(-0.2,0.5).clamp(0,1).add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1)).add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1)).divide(3).multiply(100).rename("WQI_S3")
    cloud=img.select("Oa01_radiance").unitScale(0,150).clamp(0,1).rename("cloud_S3")
    valid=wqi.mask().rename("valid_S3").toFloat()
    return wqi.addBands(cloud).addBands(valid).set("source_time",img.get("system:time_start")).set("source_name","S3")

def get_modis_fusion_layer(aoi):
    img=_get_latest("MODIS/061/MOD09GA",aoi,days_back=3)
    if img is None: return None
    clear=img.select("state_1km").bitwiseAnd(0b11).eq(0)
    img=img.updateMask(clear).updateMask(GSW)
    b1,b2,b4=img.select("sur_refl_b01"),img.select("sur_refl_b02"),img.select("sur_refl_b04")
    wqi=b4.subtract(b2).divide(b4.add(b2)).unitScale(-0.3,0.3).clamp(0,1).add(b4.divide(b1.add(1e-6)).unitScale(0.8,2.5).clamp(0,1)).add(ee.Image(1).subtract(b1.unitScale(0,1500)).clamp(0,1)).divide(3).multiply(100).rename("WQI_MODIS")
    cloud=clear.Not().rename("cloud_MODIS").toFloat(); valid=wqi.mask().rename("valid_MODIS").toFloat()
    return wqi.addBands(cloud).addBands(valid).set("source_time",img.get("system:time_start")).set("source_name","MODIS")

def get_s2_fusion_layer(aoi):
    img=_get_latest("COPERNICUS/S2_SR_HARMONIZED",aoi,days_back=10)
    if img is None: return None
    clear=img.select("SCL").eq(6); img=img.updateMask(clear).updateMask(GSW)
    b3,b4,b5,b8,b8a=(img.select("B3").divide(10000),img.select("B4").divide(10000),img.select("B5").divide(10000),img.select("B8").divide(10000),img.select("B8A").divide(10000))
    wqi_10m=b3.subtract(b8).divide(b3.add(b8)).unitScale(-0.3,0.5).clamp(0,1).add(b5.divide(b4.add(1e-6)).unitScale(1.0,3.5).clamp(0,1)).add(ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0,0.15)).clamp(0,1)).divide(3).multiply(100)
    k=ee.Kernel.square(radius=15,units="pixels")
    wqi=wqi_10m.reduceNeighborhood(ee.Reducer.mean(),k).rename("WQI_S2")
    vr=clear.toFloat().reduceNeighborhood(ee.Reducer.mean(),k).rename("valid_S2")
    cloud=ee.Image(1).subtract(vr).rename("cloud_S2")
    return wqi.addBands(cloud).addBands(vr).set("source_time",img.get("system:time_start")).set("source_name","S2")

def _sample_layer(layer,grid_df,wb,cb,vb,sn,scale=300):
    if layer is None: return pd.DataFrame()
    features=[ee.Feature(ee.Geometry.Point([r["lon"],r["lat"]]),{"cell_id":r["cell_id"]}) for _,r in grid_df.iterrows()]
    sampled=layer.select([wb,cb,vb]).sampleRegions(collection=ee.FeatureCollection(features),scale=scale,geometries=False,tileScale=4)
    st_ms=layer.get("source_time").getInfo()
    records=[]
    for f in sampled.getInfo().get("features",[]):
        p=f["properties"]; wv=p.get(wb)
        if wv is None: continue
        records.append({"cell_id":p.get("cell_id"),"wqi":round(wv,2),"cloud_cover":round(p.get(cb,1.0),3),"valid_ratio":round(p.get(vb,0.0),3),"source":sn,"source_time_ms":st_ms})
    df=pd.DataFrame(records)
    if not df.empty and st_ms:
        df["source_dt"]=pd.to_datetime(df["source_time_ms"],unit="ms",utc=True)
        df["age_days"]=(pd.Timestamp.utcnow()-df["source_dt"]).dt.total_seconds()/86400
    return df

def _score(age,conf,res):
    return math.exp(-0.3*age)*(1.0/math.log10(max(res,10)))*conf

def _pick_winner(readings):
    best=None; bs=-1.0
    for r in readings:
        c=(1.0-r.get("cloud_cover",1.0))*r.get("valid_ratio",1.0)
        s=_score(r.get("age_days",99),c,RESOLUTION_M.get(r["source"],500))
        if s>bs: bs=s; best={**r,"score":round(s,4),"confidence":round(c,3)}
    return best

@st.cache_data(ttl=3600)
def run_fusion_pipeline() -> pd.DataFrame:
    aoi=ee.Geometry.Rectangle([ISRAEL_COAST_BBOX["lon_min"],ISRAEL_COAST_BBOX["lat_min"],ISRAEL_COAST_BBOX["lon_max"],ISRAEL_COAST_BBOX["lat_max"]])
    # Use 1km grid for demo — fast enough for Streamlit Cloud (~1500 cells vs 490K)
    grid=build_grid(bbox={**ISRAEL_COAST_BBOX, "step": 0.009})
    s3_layer    = get_s3_fusion_layer(aoi)
    modis_layer = get_modis_fusion_layer(aoi)
    s2_layer    = get_s2_fusion_layer(aoi)
    layers=[
        (s3_layer,    "WQI_S3",    "cloud_S3",    "valid_S3",    "S3",    1000),
        (modis_layer, "WQI_MODIS", "cloud_MODIS", "valid_MODIS", "MODIS", 1000),
        (s2_layer,    "WQI_S2",    "cloud_S2",    "valid_S2",    "S2",    1000),
    ]
    readings=[_sample_layer(l,grid,wb,cb,vb,sn,sc) for l,wb,cb,vb,sn,sc in layers if l is not None]
    valid=[df for df in readings if not df.empty]
    if not valid: return pd.DataFrame()
    all_r=pd.concat(valid,ignore_index=True)
    winners=[w for _,g in all_r.groupby("cell_id") if (w:=_pick_winner(g.to_dict("records")))]
    result=pd.DataFrame(winners).merge(grid[["cell_id","lat","lon"]],on="cell_id",how="left")
    result["health_label"]=result["wqi"].apply(lambda v: "🟢 Safe" if v>=70 else "🟡 Caution" if v>=55 else "🔴 Unsafe")
    return result.sort_values("cell_id").reset_index(drop=True)

# =============================================================================
# 6. Legacy S3 functions
# =============================================================================
@st.cache_data(ttl=14400)
def get_available_s3_dates(wb_key,days_back=30):
    wb=WATER_BODIES[wb_key]; end=datetime.utcnow(); start=end-timedelta(days=days_back)
    coll=(ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(wb["bbox"])
          .filterDate(start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d')))
    dl=coll.aggregate_array("system:time_start").getInfo()
    return sorted(list(set([datetime.utcfromtimestamp(d/1000).strftime("%Y-%m-%d") for d in dl])),reverse=True)

def process_s3_wqi_data(wb_key,target_date_str):
    wb=WATER_BODIES[wb_key]; t_date=ee.Date(target_date_str)
    wm=ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(25)
    coll=(ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(wb["bbox"])
          .filterDate(t_date.advance(-1,'day'),t_date.advance(1,'day')))
    if coll.size().getInfo()==0: return None,None,"No Sentinel-3 scan found for this date."
    img=coll.median().clip(wb["clip_geom"]).updateMask(wm)
    ndwi=img.normalizedDifference(['Oa06_radiance','Oa17_radiance']).rename('S3_NDWI')
    b10,b11,b12=img.select('Oa10_radiance'),img.select('Oa11_radiance'),img.select('Oa12_radiance')
    mci=b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25)))).rename('MCI')
    turb=img.select('Oa08_radiance').rename('S3_Turb')
    raw=ndwi.unitScale(-0.2,0.5).clamp(0,1).add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1)).add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1)).divide(3).multiply(100).rename('WQI')
    wqi=raw.reduceNeighborhood(reducer=ee.Reducer.mean(),kernel=ee.Kernel.square(radius=1,units='pixels')).rename('WQI').updateMask(wm)
    def _pt(pt):
        try:
            val=wqi.reduceRegion(reducer=ee.Reducer.mean(),geometry=ee.Geometry.Point([pt["lon"],pt["lat"]]).buffer(450),scale=300,bestEffort=True).getInfo()
            v=val.get('WQI'); return {**pt,"wqi":round(v,1) if v is not None else None}
        except: return {**pt,"wqi":None}
    with ThreadPoolExecutor(max_workers=4) as ex: pts=list(ex.map(_pt,wb["points"]))
    return wqi,pd.DataFrame(pts),None

def generate_coastal_points_in_bbox(lat_min,lat_max,lon_min,lon_max,spacing_deg=0.45):
    pts=[]; lat=lat_min
    while lat<=lat_max:
        lon=lon_min
        while lon<=lon_max:
            pts.append({"name":f"{lat:.2f},{lon:.2f}","lat":round(lat,4),"lon":round(lon,4)}); lon+=spacing_deg
        lat+=spacing_deg
    return pts

@st.cache_data(ttl=7200)
def filter_coastal_points_gee(points,bbox_rect):
    if not points: return []
    gsw=ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence"); cpts=[]
    for i in range(0,len(points),20):
        batch=points[i:i+20]
        fc=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([p["lon"],p["lat"]]).buffer(2000),{"idx":j}) for j,p in enumerate(batch)])
        try:
            feats=gsw.reduceRegions(collection=fc,reducer=ee.Reducer.max(),scale=300).getInfo().get("features",[])
            for f in feats:
                idx=f["properties"].get("idx"); val=f["properties"].get("max",0) or 0
                if val>=5 and idx is not None: cpts.append(batch[idx])
        except Exception: pass
    return cpts

@st.cache_data(ttl=14400)
def get_global_wqi_layer(target_date_str,bbox_rect):
    lon_min,lat_min,lon_max,lat_max=bbox_rect
    bbox=ee.Geometry.Rectangle([lon_min,lat_min,lon_max,lat_max])
    wm=ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(25)
    t=ee.Date(target_date_str)
    coll=(ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(bbox).filterDate(t.advance(-1,'day'),t.advance(1,'day')))
    if coll.size().getInfo()==0: return None,"No satellite data found for this area on the selected date."
    img=coll.median().clip(bbox).updateMask(wm)
    ndwi=img.normalizedDifference(['Oa06_radiance','Oa17_radiance']).rename('S3_NDWI')
    b10,b11,b12=img.select('Oa10_radiance'),img.select('Oa11_radiance'),img.select('Oa12_radiance')
    mci=b11.subtract(b10.add(b12.subtract(b10).multiply((708.75-681.25)/(753.75-681.25)))).rename('MCI')
    turb=img.select('Oa08_radiance').rename('S3_Turb')
    raw=ndwi.unitScale(-0.2,0.5).clamp(0,1).add(ee.Image(1).subtract(mci.unitScale(-2,12)).clamp(0,1)).add(ee.Image(1).subtract(turb.unitScale(10,80)).clamp(0,1)).divide(3).multiply(100).rename('WQI')
    return raw.reduceNeighborhood(reducer=ee.Reducer.mean(),kernel=ee.Kernel.square(radius=1,units='pixels')).rename('WQI').updateMask(wm),None

def get_bbox_from_map(map_data,zoom):
    if not map_data or not map_data.get("center"): return None
    lat=map_data["center"]["lat"]; lon=map_data["center"]["lng"]
    dp=360.0/(256*(2**zoom)); hw=dp*400; hh=dp*275
    return (max(-180,lon-hw),max(-85,lat-hh),min(180,lon+hw),min(85,lat+hh))

# =============================================================================
# 7. UI
# =============================================================================
st.title("🛰️ AquaWatch Global — Multi-Satellite Water Quality Monitor")
st.markdown("Composite WQI from Sentinel-3, Sentinel-2, and MODIS — dynamic fusion with confidence scoring.")

st.sidebar.header("🔧 Settings & Dates")
MODE_GLOBAL="🌍 Global"; MODE_FUSION="🔀 Israel Coast Fusion"
wb_selection=st.sidebar.selectbox("Select water body to monitor:",[MODE_GLOBAL,MODE_FUSION]+list(WATER_BODIES.keys()))
is_global=(wb_selection==MODE_GLOBAL); is_fusion=(wb_selection==MODE_FUSION)

wb_center_default=[32.0,34.9] if is_fusion else ([20.0,0.0] if is_global else WATER_BODIES[wb_selection]["center"])

if "atm_center" not in st.session_state or st.session_state.get("last_wb")!=wb_selection:
    st.session_state.atm_center=(wb_center_default[0],wb_center_default[1])
    st.session_state.atm_data=get_atmospheric_context("🏖️ Mediterranean Coast") if is_fusion else (_empty_atm() if is_global else get_atmospheric_context(wb_selection))
    st.session_state.last_wb=wb_selection; st.session_state.global_zoom=3
    st.session_state.global_bbox=None; st.session_state.global_coastal_pts=[]; st.session_state.global_wqi_layer=None

atm_data=st.session_state.atm_data

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌊 Environmental Overlays")
overlay_choice=st.sidebar.radio("Animated flow layer:",["None","💨 Wind","🌊 Waves"],horizontal=True)

marine_grid_data=None
if overlay_choice!="None":
    if is_fusion:
        g_lat_min,g_lat_max=ISRAEL_COAST_BBOX["lat_min"],ISRAEL_COAST_BBOX["lat_max"]
        g_lon_min,g_lon_max=ISRAEL_COAST_BBOX["lon_min"],ISRAEL_COAST_BBOX["lon_max"]
    elif not is_global:
        wb=WATER_BODIES[wb_selection]; bce=wb["bbox"].bounds().getInfo()["coordinates"][0]
        lons=[p[0] for p in bce]; lats=[p[1] for p in bce]
        g_lat_min,g_lat_max=min(lats),max(lats); g_lon_min,g_lon_max=min(lons),max(lons)
    else:
        bs=st.session_state.get("global_bbox")
        if bs: g_lon_min,g_lat_min,g_lon_max,g_lat_max=bs
        else: g_lat_min,g_lat_max,g_lon_min,g_lon_max=29.0,34.0,34.0,36.5
    with st.spinner("Fetching environmental flow data..."):
        marine_grid_data=get_marine_grid(g_lat_min,g_lat_max,g_lon_min,g_lon_max,steps=6)

if not is_global and not is_fusion:
    with st.spinner("Locating available Sentinel-3 pass dates..."):
        available_dates=get_available_s3_dates(wb_selection)
else:
    available_dates=[(datetime.utcnow()-timedelta(days=d)).strftime('%Y-%m-%d') for d in range(1,8)]

if available_dates:
    selected_date_str=st.sidebar.selectbox("Select satellite pass date:",[f"🟢 {d}" for d in available_dates]).replace("🟢 ","")
else:
    selected_date_str=(datetime.utcnow()-timedelta(days=1)).strftime('%Y-%m-%d')

# ---- Fusion tab ----
if is_fusion:
    st.subheader("🔀 Israel Coast — Multi-Satellite Fusion Map")
    st.markdown("Each 300m cell shows the best reading from S3, MODIS, or S2 — scored by freshness × confidence × resolution.")
    with st.spinner("Running fusion pipeline (S3 + MODIS + S2)..."):
        fusion_df=run_fusion_pipeline()
    if fusion_df.empty:
        st.warning("No satellite data available right now. Try again shortly.")
    else:
        col_map,col_info=st.columns([3.5,1.5])
        with col_map:
            m_f=folium.Map(location=[31.5,34.6],zoom_start=8)
            def _c(w): return "#27AE60" if w>=70 else "#F39C12" if w>=55 else "#E74C3C"
            for _,row in fusion_df.iterrows():
                folium.CircleMarker(location=[row["lat"],row["lon"]],radius=4,
                    color=_c(row["wqi"]),fill=True,fill_color=_c(row["wqi"]),
                    fill_opacity=row.get("confidence",0.7),
                    tooltip=f"<b>{row['health_label']}</b><br>WQI:{row['wqi']:.1f}<br>Source:{row['source']}<br>Age:{row.get('age_days',0):.1f}d<br>Confidence:{row.get('confidence',0):.0%}"
                ).add_to(m_f)
            m_f.add_child(OnMapWaterLegend())
            if marine_grid_data and overlay_choice!="None":
                key="waves" if overlay_choice=="🌊 Waves" else "wind"
                m_f.add_child(OnMapVelocityLayer(marine_grid_data[key],layer_name="Waves (m)" if key=="waves" else "Wind (m/s)"))
            st_folium(m_f,width=800,height=550,key="fusion_map_v1",returned_objects=[])
        with col_info:
            st.subheader("📊 Fusion Summary")
            for src,cnt in fusion_df["source"].value_counts().items(): st.metric(f"{src} cells",cnt)
            st.markdown("---"); st.subheader("🏖️ Health Status")
            for lbl,cnt in fusion_df["health_label"].value_counts().items(): st.write(f"{lbl}: **{cnt}** cells")
            st.markdown("---"); st.subheader("📋 Sample Data")
            st.dataframe(fusion_df[["lat","lon","wqi","source","health_label"]].head(20).rename(
                columns={"lat":"Lat","lon":"Lon","wqi":"WQI","source":"Satellite","health_label":"Status"}),
                use_container_width=True,hide_index=True)

# ---- Global tab ----
elif is_global:
    col_map,col_info=st.columns([4.0,1.0])
    with col_map:
        st.subheader("🌍 Global WQI Monitoring Map")
        zi=st.session_state.get("global_zoom",3); ci=list(st.session_state.atm_center)
        mg=folium.Map(location=ci,zoom_start=zi)
        cl=st.session_state.get("global_wqi_layer")
        if cl is not None:
            try:
                mid=ee.Image(cl).getMapId({'min':40,'max':85,'palette':['#FF0000','#FFFF00','#00FF00']})
                folium.TileLayer(tiles=mid['tile_fetcher'].url_format,attr='GEE Sentinel-3',name="WQI",overlay=True,control=False,opacity=0.85).add_to(mg)
            except Exception: pass
        cz=st.session_state.get("global_zoom",3); cp=st.session_state.get("global_coastal_pts",[])
        if cz>=7 and cp:
            for pt in cp:
                wv=pt.get("wqi"); cm="green" if wv and wv>65 else "orange" if wv and wv>45 else "gray"
                folium.CircleMarker(location=[pt["lat"],pt["lon"]],radius=5,color="black",weight=1,fill_color=cm,fill_opacity=0.85,fill=True).add_to(mg)
        mg.add_child(OnMapWaterLegend())
        if marine_grid_data and overlay_choice!="None":
            key="waves" if overlay_choice=="🌊 Waves" else "wind"
            mg.add_child(OnMapVelocityLayer(marine_grid_data[key],layer_name="Waves (m)" if key=="waves" else "Wind (m/s)"))
        mdg=st_folium(mg,width=800,height=550,key="global_map_v1",returned_objects=["center","zoom"])
        if mdg:
            nz=mdg.get("zoom") or zi; nc=mdg.get("center")
            if nc:
                nl,no=nc["lat"],nc["lng"]; pl,po=st.session_state.atm_center
                dk=haversine_km(pl,po,nl,no); pz=st.session_state.get("global_zoom",3)
                tc=(pz<7)!=(nz<7); mf=(dk>200)
                st.session_state.global_zoom=nz; st.session_state.atm_center=(nl,no)
                if tc or mf:
                    nb=get_bbox_from_map(mdg,nz); st.session_state.global_bbox=nb
                    if nb:
                        with st.spinner("Computing WQI for area..."):
                            gl,ge=get_global_wqi_layer(selected_date_str,nb)
                            st.session_state.global_wqi_layer=gl if not ge else None
                        if nz>=7:
                            lnm,lam,lnx,lax=nb
                            cands=generate_coastal_points_in_bbox(lam,lax,lnm,lnx)
                            with st.spinner("Locating coastal points..."): cpts=filter_coastal_points_gee(cands,nb)
                            if gl and cpts:
                                def _sp(pt):
                                    try:
                                        v=gl.reduceRegion(reducer=ee.Reducer.mean(),geometry=ee.Geometry.Point([pt["lon"],pt["lat"]]).buffer(2000),scale=300,bestEffort=True).getInfo()
                                        wv=v.get("WQI"); return {**pt,"wqi":round(wv,1) if wv else None}
                                    except: return {**pt,"wqi":None}
                                with ThreadPoolExecutor(max_workers=6) as ex: cpts=list(ex.map(_sp,cpts))
                            st.session_state.global_coastal_pts=cpts
                        else: st.session_state.global_coastal_pts=[]
                    st.rerun()
    with col_info:
        st.subheader("🏖️ Coastal Water Cleanliness Index")
        if cz<7: st.info("🔍 Zoom in to see measurement points")
        else:
            pts=st.session_state.get("global_coastal_pts",[])
            if pts:
                def _sg(s):
                    try: v=float(s)
                    except: return "❓ No Data"
                    return "🟢 Clean" if v>=70 else "🟡 Moderate" if v>=55 else "🔴 Polluted"
                df_g=pd.DataFrame(pts)[["lat","lon","wqi"]].copy()
                df_g["Status"]=df_g["wqi"].apply(_sg)
                st.dataframe(df_g.rename(columns={"lat":"Latitude","lon":"Longitude"})[["Latitude","Longitude","Status"]],use_container_width=True,hide_index=True)
            else: st.write("No coastal points found in the current area.")

# ---- Israel water bodies tab ----
else:
    with st.spinner("Computing composite values..."):
        wqi_layer,df_beaches,error_msg=process_s3_wqi_data(wb_selection,selected_date_str)
    if error_msg: st.error(error_msg)
    elif wqi_layer:
        df_beaches=blend_atmospheric_penalty(df_beaches,atm_data,wb_selection)
        pt_keys=tuple((r["lat"],r["lon"]) for _,r in df_beaches.iterrows())
        sst_map=get_sst_for_points(pt_keys)
        df_beaches["sst"]=df_beaches.apply(lambda r:sst_map.get((r["lat"],r["lon"])),axis=1)
        col_map,col_info=st.columns([4.0,1.0])
        with col_map:
            st.subheader(f"📍 Composite WQI Map: {wb_selection}")
            m=folium.Map(location=wb_center_default,zoom_start=WATER_BODIES[wb_selection]["zoom"])
            mid=ee.Image(wqi_layer).getMapId({'min':40,'max':85,'palette':['#FF0000','#FFFF00','#00FF00']})
            folium.TileLayer(tiles=mid['tile_fetcher'].url_format,attr='GEE Sentinel-3',name="WQI",overlay=True,control=False,opacity=0.85).add_to(m)
            for _,r in df_beaches.iterrows():
                sc=r['composite_with_atm'] if pd.notna(r['composite_with_atm']) else r['wqi']
                cm="green" if sc and sc>65 else "orange" if sc else "red"
                folium.CircleMarker(location=[r["lat"],r["lon"]],radius=6,color="black",weight=1,fill_color=cm,fill_opacity=0.9,fill=True).add_to(m)
            m.add_child(OnMapAtmosphereControl(atm_data)); m.add_child(OnMapWaterLegend())
            if marine_grid_data and overlay_choice!="None":
                key="waves" if overlay_choice=="🌊 Waves" else "wind"
                m.add_child(OnMapVelocityLayer(marine_grid_data[key],layer_name="Waves (m)" if key=="waves" else "Wind (m/s)"))
            md=st_folium(m,width=800,height=550,key="s3_map_v7",returned_objects=["center"])
            if md and md.get("center"):
                nl,no=md["center"]["lat"],md["center"]["lng"]; pl,po=st.session_state.atm_center
                st.session_state.atm_center=(nl,no)
                if haversine_km(pl,po,nl,no)>50:
                    st.session_state.atm_data=get_atmospheric_context_by_coords(nl,no); st.rerun()
        with col_info:
            st.subheader("🏖️ Coastal Water Cleanliness Index")
            if df_beaches is not None and not df_beaches.empty:
                df_d=df_beaches[["name","composite_with_atm","sst"]].copy()
                df_d.columns=["Station Name","_score","sst"]
                def _st(s):
                    try: v=float(s)
                    except: return "❓ No Data"
                    return "🟢 Clean" if v>=70 else "🟡 Moderate" if v>=55 else "🔴 Polluted"
                def _sst(s):
                    if s is None: return "—"
                    t=float(s); ic="🔴" if t>=28 else "🟠" if t>=24 else "🟡" if t>=20 else "🟢" if t>=16 else "🔵"
                    return f"{ic} {t:.1f}°C"
                df_d["Status"]=df_d["_score"].apply(_st); df_d["Sea Temp"]=df_d["sst"].apply(_sst)
                st.dataframe(df_d[["Station Name","Status","Sea Temp"]],use_container_width=True,hide_index=True)
            else: st.write("No defined stations found for this area.")
