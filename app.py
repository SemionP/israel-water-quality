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
