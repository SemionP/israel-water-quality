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

from config import (_T, PROFILES, SignalReading, MEDIResult, _normalize_wqi,
                    _signal_risk, _confidence_from_signals, _detect_drivers, compute_medi,
                    BEACHES, HAIFA_BBOX_COORDS, ISRAEL_CLIP_COORDS, PALETTE)

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

st.markdown("""<style>
#MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {display: none;}
footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)



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
.contact-btn{margin-left:auto;font-family:'Rajdhani',sans-serif;font-size:0.8rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--teal-bright);background:transparent;border:1px solid rgba(0,200,200,0.4);border-radius:3px;padding:5px 15px;cursor:pointer;display:inline-flex;align-items:center;gap:7px;transition:background 0.2s,border-color 0.2s;}
.contact-btn:hover{background:rgba(0,200,200,0.1);border-color:var(--teal-bright);}
.medi-modal-overlay{display:none;position:fixed;inset:0;background:rgba(2,13,24,0.88);align-items:center;justify-content:center;z-index:9999;}
.medi-modal-overlay.on{display:flex;}
.medi-modal{width:500px;background:#041e33;border:1px solid rgba(0,200,200,0.22);border-radius:6px;overflow:hidden;position:relative;}
.scan-lines{position:absolute;inset:0;pointer-events:none;background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,200,200,0.016) 3px,rgba(0,200,200,0.016) 4px);z-index:0;}
.scan-sweep{position:absolute;left:0;right:0;height:2px;background:rgba(0,200,200,0.18);animation:sweep 3.5s linear infinite;z-index:0;}
@keyframes sweep{0%{top:0;opacity:0.7;}100%{top:100%;opacity:0;}}
.modal-hdr{position:relative;z-index:1;padding:13px 18px;background:rgba(0,200,200,0.06);border-bottom:1px solid rgba(0,200,200,0.13);display:flex;align-items:center;justify-content:space-between;}
.modal-ttl{font-family:'Rajdhani',sans-serif;font-size:1rem;font-weight:700;color:var(--teal-bright);letter-spacing:0.08em;text-transform:uppercase;}
.modal-sub{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:var(--teal-dim);letter-spacing:0.14em;margin-top:2px;}
.modal-close{background:transparent;border:none;color:var(--text-dim);font-size:20px;cursor:pointer;line-height:1;padding:2px 4px;}
.modal-close:hover{color:var(--teal-bright);}
.modal-body{position:relative;z-index:1;padding:16px 18px;}
.m-sec{font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:rgba(0,200,200,0.4);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:8px;}
.m-row2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;}
.m-fg{margin-bottom:10px;}
.m-fl{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:var(--text-dim);letter-spacing:0.1em;text-transform:uppercase;display:block;margin-bottom:4px;}
.m-fi{width:100%;background:var(--ocean-surface);border:1px solid rgba(0,200,200,0.18);border-radius:3px;color:var(--text-primary);font-family:'Exo 2',sans-serif;font-size:0.8rem;padding:6px 9px;outline:none;transition:border-color 0.2s;}
.m-fi:focus{border-color:rgba(0,200,200,0.55);}
.m-fi::placeholder{color:rgba(214,234,248,0.28);}
select.m-fi{appearance:none;cursor:pointer;}
select.m-fi option{background:#041e33;}
textarea.m-fi{resize:none;height:80px;line-height:1.5;}
.m-div{height:1px;background:rgba(0,200,200,0.09);margin:10px 0;}
.modal-ftr{position:relative;z-index:1;padding:11px 18px;border-top:1px solid rgba(0,200,200,0.1);display:flex;align-items:center;justify-content:flex-end;}
.pulse-wrap{position:relative;display:inline-flex;}
.pulse-ring{position:absolute;inset:-4px;border-radius:5px;border:1px solid rgba(0,200,200,0.45);animation:pring 2s ease-out infinite;}
@keyframes pring{0%{opacity:0.7;transform:scale(1);}100%{opacity:0;transform:scale(1.1);}}
.m-sbtn{font-family:'Rajdhani',sans-serif;font-weight:700;font-size:0.82rem;letter-spacing:0.1em;text-transform:uppercase;color:#020d18;background:var(--teal-bright);border:none;border-radius:3px;padding:7px 20px;cursor:pointer;display:inline-flex;align-items:center;gap:6px;}
.m-sbtn:hover{background:#00dddd;}
.m-success{display:none;position:relative;z-index:1;padding:36px 18px;text-align:center;}
.m-success-icon{font-size:38px;color:var(--green-safe);margin-bottom:12px;}
.m-success-title{font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.1rem;color:var(--teal-bright);letter-spacing:0.08em;margin-bottom:8px;}
.m-success-msg{font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:var(--text-dim);letter-spacing:0.1em;line-height:1.8;}
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
/* Buttons — ensure readable text on dark background */
[data-testid="stButton"] button,
[data-testid="baseButton-secondary"],
[data-testid="stDownloadButton"] button{
  background:var(--ocean-surface)!important;
  color:var(--teal-bright)!important;
  border:1px solid rgba(0,200,200,0.4)!important;
  border-radius:4px!important;
  font-family:'Rajdhani',sans-serif!important;
  font-weight:600!important;
  letter-spacing:0.04em!important;
}
[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] button:hover{
  background:rgba(0,200,200,0.15)!important;
  border-color:var(--teal-bright)!important;
  color:#ffffff!important;
}
[data-testid="stButton"] button *,
[data-testid="stDownloadButton"] button *{color:inherit!important;}
/* File uploader */
[data-testid="stFileUploader"]{
  background:var(--ocean-mid)!important;
  border:1px dashed rgba(0,200,200,0.3)!important;
  border-radius:4px!important;
  padding:8px!important;
}
[data-testid="stFileUploader"] *{color:var(--text-primary)!important;}
[data-testid="stFileUploader"] button{
  background:var(--ocean-surface)!important;
  color:var(--teal-bright)!important;
  border:1px solid rgba(0,200,200,0.4)!important;
}
/* Text input */
[data-testid="stTextInput"] input{
  background:var(--ocean-surface)!important;
  color:var(--text-primary)!important;
  border:1px solid rgba(0,200,200,0.3)!important;
}
/* Expander */
[data-testid="stExpander"]{
  background:rgba(4,30,51,0.6)!important;
  border:1px solid rgba(0,200,200,0.2)!important;
  border-radius:6px!important;
}
[data-testid="stExpander"] summary{color:var(--teal-bright)!important;font-family:'Rajdhani',sans-serif!important;font-weight:600!important;}
[data-testid="stExpander"] summary *{color:var(--teal-bright)!important;}
</style>
""", unsafe_allow_html=True)

# ── Header with Contact Us button (pure Streamlit) ────────────────────────────
if "contact_modal_open" not in st.session_state:
    st.session_state.contact_modal_open = False
if "contact_sent" not in st.session_state:
    st.session_state.contact_sent = False

header_col, btn_col = st.columns([8, 1])
with header_col:
    st.markdown("""
<div class="medi-header">
  <div>
    <div class="logo-text">⬡ MEDI PLATFORM</div>
    <div class="logo-sub">Maritime Environmental Decision Intelligence</div>
  </div>
</div>
""", unsafe_allow_html=True)
with btn_col:
    st.markdown("<div style='padding-top:6px;'>", unsafe_allow_html=True)
    if st.button("✉ Contact Us", key="open_contact_modal", use_container_width=True):
        st.session_state.contact_modal_open = True
        st.session_state.contact_sent = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ── Contact Modal ──────────────────────────────────────────────────────────────
if st.session_state.contact_modal_open:
    st.markdown("""
<style>
.modal-backdrop{background:rgba(2,13,24,0.92);border:1px solid rgba(0,200,200,0.22);border-radius:6px;padding:0;margin-bottom:12px;position:relative;overflow:hidden;}
.scan-lines{position:absolute;inset:0;pointer-events:none;background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,200,200,0.016) 3px,rgba(0,200,200,0.016) 4px);z-index:0;}
.modal-inner{position:relative;z-index:1;}
.modal-hdr2{padding:13px 20px;background:rgba(0,200,200,0.06);border-bottom:1px solid rgba(0,200,200,0.13);display:flex;align-items:center;justify-content:space-between;}
.modal-ttl2{font-family:'Rajdhani',sans-serif;font-size:1.05rem;font-weight:700;color:#00c8c8;letter-spacing:0.08em;text-transform:uppercase;}
.modal-sub2{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#007f8a;letter-spacing:0.14em;margin-top:2px;}
.modal-body2{padding:16px 20px;}
.m-sec2{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:rgba(0,200,200,0.4);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:10px;margin-top:4px;}
.modal-divider{height:1px;background:rgba(0,200,200,0.09);margin:14px 0;}
.modal-success{padding:32px 20px;text-align:center;}
.modal-success-icon{font-size:36px;color:#1ecb7b;margin-bottom:10px;}
.modal-success-title{font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.1rem;color:#00c8c8;letter-spacing:0.08em;margin-bottom:6px;}
.modal-success-msg{font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#7fb3d3;letter-spacing:0.1em;line-height:1.8;}
.sweep-bar{height:2px;background:linear-gradient(90deg,transparent,rgba(0,200,200,0.4),transparent);animation:sweepbar 2.5s ease-in-out infinite;}
@keyframes sweepbar{0%,100%{opacity:0.3;}50%{opacity:1;}}
</style>
""", unsafe_allow_html=True)

    st.markdown('<div class="modal-backdrop"><div class="scan-lines"></div><div class="sweep-bar"></div><div class="modal-inner">', unsafe_allow_html=True)

    # Header row
    st.markdown("""
<div class="modal-hdr2">
  <div>
    <div class="modal-ttl2">⬡ Contact MEDI</div>
    <div class="modal-sub2">Maritime Data &amp; Research Inquiries</div>
  </div>
</div>
""", unsafe_allow_html=True)

    if st.session_state.contact_sent:
        st.markdown("""
<div class="modal-success">
  <div class="modal-success-icon">✓</div>
  <div class="modal-success-title">Inquiry sent</div>
  <div class="modal-success-msg">Your message is on its way.<br>We'll be in touch shortly.</div>
</div>
""", unsafe_allow_html=True)
        if st.button("✕  Close", key="close_success_modal"):
            st.session_state.contact_modal_open = False
            st.session_state.contact_sent = False
            st.rerun()
    else:
        st.markdown('<div class="modal-body2">', unsafe_allow_html=True)
        st.markdown('<div class="m-sec2">▸ your details</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            contact_name = st.text_input("Full name", placeholder="Dr. Jane Smith", key="cf_name", label_visibility="visible")
        with col2:
            contact_org = st.text_input("Organization", placeholder="Institute / Company", key="cf_org", label_visibility="visible")
        contact_email = st.text_input("Email", placeholder="you@organization.com", key="cf_email", label_visibility="visible")

        st.markdown('<div class="modal-divider"></div><div class="m-sec2">▸ your inquiry</div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            contact_use = st.selectbox("Use case", ["", "Research / Academia", "Port Operations", "Aquaculture", "Environmental Agency", "ESG / Compliance", "Maritime Surveillance", "Other"], key="cf_use")
        with col4:
            contact_time = st.selectbox("Timeline", ["", "Just exploring", "3–6 months", "Immediate need"], key="cf_time")
        contact_msg = st.text_area("What are you looking for?", placeholder="Describe your research, data needs, monitoring goals, or integration interest…", key="cf_msg", height=100)

        st.markdown('</div>', unsafe_allow_html=True)

        # Footer buttons
        btn_l, btn_r = st.columns([3, 1])
        with btn_l:
            if st.button("✕  Close", key="close_contact_modal"):
                st.session_state.contact_modal_open = False
                st.rerun()
        with btn_r:
            if st.button("▶ Send inquiry", key="send_contact", use_container_width=True):
                name  = contact_name  or "(not provided)"
                org   = contact_org   or "(not provided)"
                email = contact_email or "(not provided)"
                use   = contact_use   or "(not provided)"
                time  = contact_time  or "(not provided)"
                msg   = contact_msg   or "(not provided)"
                mailto_body = (
                    f"Name: {name}\nOrganization: {org}\nReply email: {email}\n"
                    f"Use Case: {use}\nTimeline: {time}\n\nInquiry:\n{msg}"
                )
                import urllib.parse
                subject_enc = urllib.parse.quote(f"MEDI Platform Inquiry — {use}")
                body_enc    = urllib.parse.quote(mailto_body)
                mailto_link = f"mailto:semion.polinov@gmail.com?subject={subject_enc}&body={body_enc}"
                st.markdown(f'<meta http-equiv="refresh" content="0;url={mailto_link}">', unsafe_allow_html=True)
                components.html(f'<script>window.open("{mailto_link}");</script>', height=0)
                st.session_state.contact_sent = True
                st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)

# Analytics
components.html('<script async src="https://cloud.umami.is/script.js" data-website-id="07a48db1-5aa7-4d88-aaac-9cfb6fc2600d"></script>', height=0)
if "ga_loaded" not in st.session_state:
    st.session_state.ga_loaded = True
    components.html("""<script async src="https://www.googletagmanager.com/gtag/js?id=G-K37THY2160"></script>
<script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','G-K37THY2160');</script>""", height=0)

# =============================================================================
# GEE Auth
# =============================================================================
from gee_processing import (init_gee, get_atm, get_sst, get_available_s3_dates,
    get_modis_sst_anomaly, process_modis_wqi, process_israel_s2,
    get_available_dates_combined, process_israel_wqi, compute_beach_history_7d,
    process_port_medi, get_global_wqi_layer, get_bbox_from_map, haversine_km,
    compute_point_wqi, compute_city_wqi, compute_beach_history_range,
    compute_zone_history_range, _empty_atm, MODE_ISRAEL, MODE_GLOBAL, mode,
    sample_pixel_spectra)
init_gee()


from storage import (load_zones, save_zones, load_zones_from_all, load_points, save_points)







# Geometries initialized in gee_processing.py
import ee
HAIFA_BBOX = ee.Geometry.Rectangle(HAIFA_BBOX_COORDS)
ISRAEL_CLIP = ee.Geometry.Polygon([ISRAEL_CLIP_COORDS])

ISRAEL_CLIP      = ee.Geometry.Polygon([[[34.95,33.10],[34.55,33.10],[34.15,32.50],[34.10,32.00],
    [34.15,31.50],[34.50,31.25],[34.75,31.25],[34.95,31.30],[35.02,31.60],[35.00,32.10],[35.05,32.60],[35.10,33.10],[34.95,33.10]]])

# BEACHES imported from config.py

# Map Components
# =============================================================================
class OnMapWaterLegend(MacroElement):
    def __init__(self):
        super().__init__()
        self._template=Template("""{% macro script(this, kwargs) %}
var lg=L.control({position:'bottomright'});
lg.onAdd=function(map){var d=L.DomUtil.create('div','info legend');
d.style.cssText='background:rgba(2,13,24,0.92);padding:12px;border:1px solid rgba(0,200,200,0.3);border-radius:6px;font-family:Arial,sans-serif;font-size:14px;color:#d6eaf8;';
d.innerHTML='<div style="font-weight:bold;margin-bottom:8px;text-align:center;color:#00c8c8;">Water Quality Index</div><div style="display:flex;align-items:center;gap:8px;"><div style="height:120px;width:14px;background:linear-gradient(to bottom,#4575b4,#74add1,#fdae61,#d73027);border-radius:3px;flex-shrink:0;"></div><div style="display:flex;flex-direction:column;justify-content:space-between;height:120px;font-size:13px;"><span style="color:#1ecb7b;font-weight:bold;">Clean</span><span style="color:#f0a500;font-weight:bold;">Moderate</span><span style="color:#e03c3c;font-weight:bold;">Polluted</span></div></div>';return d;};
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
        html=f'<div style="background:rgba(2,13,24,0.92);border:1px solid rgba(0,200,200,0.3);border-radius:8px;padding:10px 13px;font-family:Arial,sans-serif;font-size:14px;color:#d6eaf8;min-width:150px;"><div style="font-weight:bold;margin-bottom:7px;text-align:center;font-size:13px;color:#00c8c8;">🌍 Atmospheric Context</div><div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;"><svg width="24" height="24" viewBox="0 0 28 28"><g transform="rotate({ar},14,14)"><polygon points="14,2 18,22 14,18 10,22" fill="#2980B9" opacity="0.85"/></g></svg><div><div style="font-size:14px;font-weight:bold;">{ws_s}</div><div style="font-size:13px;color:{bc};">Beaufort {bf}</div></div></div><hr style="margin:5px 0;border-color:rgba(0,200,200,0.2);"><div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span>🌡️</span><span style="font-weight:bold;">{tc_s}</span></div><div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span>{ri}</span><span style="font-weight:bold;">{pr_s}</span></div><div style="display:flex;justify-content:space-between;"><span>💧</span><span style="font-weight:bold;">{rh_s}</span></div></div>'
        self._template=Template("""{% macro script(this, kwargs) %}
var ac=L.control({position:'bottomleft'});
ac.onAdd=function(map){var d=L.DomUtil.create('div','ac');d.innerHTML=`"""+html.replace("`","'")+"""`;L.DomEvent.disableClickPropagation(d);return d;};
ac.addTo({{this._parent.get_name()}});{% endmacro %}""")

# =============================================================================
# Data Functions
# =============================================================================
# GEE processing functions imported from gee_processing.py

medi_profile = "Beach Safety"  # default

# compute_point_wqi, compute_city_wqi, compute_beach_history_range,
# compute_zone_history_range imported from gee_processing.py



# Session state initialization
if "user_zones" not in st.session_state:
    st.session_state.user_zones = load_zones_from_all()
if "monitor_points" not in st.session_state:
    st.session_state.monitor_points = load_points()
if "pending_point" not in st.session_state:
    st.session_state.pending_point = None

if mode == MODE_ISRAEL:

    # ==========================================================================
    # SIDEBAR — Module selector (radio) + controls
    # ==========================================================================
    with st.sidebar:
        st.markdown("""
<div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
color:#007f8a;letter-spacing:0.12em;margin-bottom:8px;margin-top:4px;">
▸ SELECT MODULE
</div>""", unsafe_allow_html=True)

        if "active_module" not in st.session_state:
            st.session_state.active_module = None
        # Auto-select Water Quality
        st.session_state.active_module = "🌊  Water Quality"
        active_module = "🌊  Water Quality"
        # ── H3 WQI Monitor ───────────────────────────────────────────────
        st.markdown("### 🗺 H3 WQI Monitor")
        if st.button("🔄 Update WQI (S-3)", use_container_width=True):
            from update_wqi import run_update
            with st.spinner("Computing S-3 WQI..."):
                _snap = run_update(status_callback=st.write)
            if _snap:
                st.session_state["wqi_snapshot"] = _snap
                st.success(f"✅ {_snap['valid_count']}/{_snap['hex_count']} hex updated")
            else:
                st.error("No S-3 data available.")
        if st.button("🎯 Calibrate (1 year)", use_container_width=True):
            from calibrate_wqi import run_calibration
            with st.spinner("Self-calibrating from 365 days of S-3..."):
                _cal = run_calibration(status_callback=st.write)
            if _cal:
                st.success(f"✅ MCI: [{_cal['mci']['unit_scale_min']:.1f}, {_cal['mci']['unit_scale_max']:.1f}] | Turb: [{_cal['turbidity']['unit_scale_min']:.1f}, {_cal['turbidity']['unit_scale_max']:.1f}]")
            else:
                st.error("Calibration failed.")
        if "wqi_snapshot" not in st.session_state:
            try:
                from storage import load_snapshot
                st.session_state["wqi_snapshot"] = load_snapshot()
            except Exception:
                st.session_state["wqi_snapshot"] = None
        _snap = st.session_state.get("wqi_snapshot")
        if _snap:
            _ts = _snap.get("generated_utc","")[:16].replace("T"," ")
            st.caption(f"Last update: {_ts} UTC · {_snap.get('valid_count',0)} hex valid")
        st.divider()


        # ── Controls: SAR modules share date picker (disabled) ───────────────
        if False:  # SAR disabled
            st.markdown("### 📡 Sentinel-1 SAR")
            with st.spinner("Fetching S1 dates..."):
                from s1_processing import get_available_s1_dates as _get_s1_dates
                _s1_avail = _get_s1_dates(days_back=14)

            if not _s1_avail:
                st.warning("No S1 acquisitions in last 14 days.")
                st.stop()

            _s1_labels = [
                f"{d['date']}  ·  {d['orbit'][:3]}  ·  {d['age_h']:.0f}h ago"
                for d in _s1_avail
            ]
            _s1_idx = st.selectbox(
                "📅 Acquisition",
                range(len(_s1_labels)),
                format_func=lambda i: _s1_labels[i],
                key="s1_date_select",
            )
            _s1_sel_date  = _s1_avail[_s1_idx]["date"]
            _s1_sel_orbit = _s1_avail[_s1_idx]["orbit"]
            st.caption(f"Orbit: {_s1_sel_orbit}  ·  10m IW mode")
            st.divider()

            _run_label = (
                "🔍 Detect Oil Spills"
                if "Oil" in active_module
                else "🛸 Detect Vessels"
            )
            _run_sar = st.button(_run_label, use_container_width=True, type="primary")

    # ==========================================================================
    # SAR modules disabled
    if False:
        _is_oil     = "Oil" in active_module
        _module_key = "oil" if _is_oil else "vessels"
        _cache_key  = f"s1_result_{_module_key}_{_s1_sel_date}"

        st.markdown(
            f"## {'🛢️ Oil Spill Detection' if _is_oil else '🛸 Vessel Detection'}"
            f" — Sentinel-1 SAR"
        )

        # Load on button press or if cached result exists for this date
        if _run_sar or st.session_state.get(_cache_key):
            if _run_sar:
                # clear cache for new run
                for k in list(st.session_state.keys()):
                    if k.startswith("s1_result_"):
                        del st.session_state[k]

            if not st.session_state.get(_cache_key):
                with st.spinner(f"🛰 Processing S1 SAR · {_s1_sel_date}..."):
                    from s1_processing import (
                        get_s1_layers              as _gsl,
                        detect_oil_spills          as _dos,
                        detect_vessels             as _dv,
                        check_vessel_oil_proximity as _cvop,
                    )
                    _layers = _gsl(_s1_sel_date)
                    if _is_oil:
                        _oil_res = _dos(_s1_sel_date)
                        _ves_res = {"vessels": [], "n_vessels": 0}
                    else:
                        _ves_res = _dv(_s1_sel_date)
                        _oil_res = {"polygons": [], "n_anomalies": 0, "total_area_km2": 0}
                    _ves_res["vessels"] = _cvop(
                        _ves_res.get("vessels", []),
                        _oil_res.get("polygons", []),
                    )
                    st.session_state[_cache_key] = {
                        "layers":  _layers,
                        "oil":     _oil_res,
                        "vessels": _ves_res,
                        "date":    _s1_sel_date,
                    }

            _r    = st.session_state[_cache_key]
            _oil  = _r["oil"]
            _ves  = _r["vessels"]
            _date = _r["date"]

            # ── Map ─────────────────────────────────────────────────────────
            _m = folium.Map(location=[32.0, 34.85], zoom_start=8,
                            tiles="CartoDB dark_matter")

            # Layer 1: Raw VV (always shown, base context)
            if _r["layers"].get("vv"):
                folium.TileLayer(
                    _r["layers"]["vv"], attr="S1 VV", name="📡 SAR VV (dB)",
                    overlay=True, show=True, opacity=0.7).add_to(_m)

            # Layer 2: VV/VH ratio
            if _r["layers"].get("ratio"):
                folium.TileLayer(
                    _r["layers"]["ratio"], attr="VV/VH", name="📊 VV/VH Ratio",
                    overlay=True, show=False, opacity=0.6).add_to(_m)

            # Layer 3: ORM — Oil Risk Map (your colorBlend formula)
            if _r["layers"].get("orm"):
                folium.TileLayer(
                    _r["layers"]["orm"], attr="ORM", name="🛢 Oil Risk Map (ORM)",
                    overlay=True, show=True, opacity=0.8).add_to(_m)

            # Oil polygons — color by probability
            if _is_oil:
                for _p in _oil.get("polygons", []):
                    if _p.get("coords"):
                        _pc = _p.get("color", "#e24b4a")
                        _prob = _p.get("probability", "?")
                        _orm_v = _p.get("orm_mean")
                        folium.Polygon(
                            locations=[(c[1], c[0]) for c in _p["coords"]],
                            color=_pc, fill=True, fill_opacity=0.25, weight=2,
                            popup=folium.Popup(
                                f"<b>{_p['id']}</b><br>"
                                f"Area: {_p.get('area_km2_min','?')}–{_p.get('area_km2_max','?')} km²<br>"
                                f"Probability: <b>{_prob}%</b><br>"
                                f"Confidence: {_p.get('confidence','?')}<br>"
                                + (f"ORM mean: {_orm_v:.3f}" if _orm_v else ""),
                                max_width=220),
                        ).add_to(_m)
                        folium.Marker(
                            [_p["lat"], _p["lon"]],
                            icon=folium.DivIcon(html=(
                                f'<div style="color:{_pc};font-size:11px;font-weight:bold;'
                                f'white-space:nowrap;text-shadow:0 0 4px #000;'
                                f'background:rgba(0,0,0,0.5);padding:2px 4px;border-radius:3px;">'
                                f'🛢 {_prob}% · {_p.get("area_km2","?")} km²</div>'
                            ))
                        ).add_to(_m)

            # Vessel bounding boxes
            if not _is_oil:
                for _v in _ves.get("vessels", []):
                    if _v.get("bbox_coords"):
                        _vc = "#FAC775" if _v.get("near_oil") else "#c8e8f8"
                        folium.Polygon(
                            locations=[(c[1], c[0]) for c in _v["bbox_coords"]],
                            color=_vc, fill=False, weight=2,
                            popup=folium.Popup(
                                f"<b>{_v['id']}</b><br>"
                                f"Category: {_v['category']}<br>"
                                f"Length: {_v.get('length_m','?')} m<br>"
                                f"Width: {_v.get('width_m','?')} m<br>"
                                f"Confidence: {_v['confidence']}"
                                + (f"<br>⚠ Near {_v['near_oil_id']}" if _v.get("near_oil") else ""),
                                max_width=220),
                        ).add_to(_m)
                        folium.Marker(
                            [_v["lat"], _v["lon"]],
                            icon=folium.DivIcon(html=(
                                f'<div style="color:{_vc};font-size:10px;'
                                f'white-space:nowrap;text-shadow:0 0 4px #000;">'
                                f'⬡ {_v["id"]}</div>'
                            ))
                        ).add_to(_m)

            folium.LayerControl(collapsed=False).add_to(_m)
            st_folium(_m, use_container_width=True, height=520,
                      key=f"sar_map_{_date}_{_module_key}")

            # ── Results ──────────────────────────────────────────────────
            if _is_oil:
                st.markdown(f"### 🛢️ Oil Anomalies · {_oil.get('n_anomalies', 0)}"
                            + (f"  ·  Total {_oil.get('total_area_km2',0):.3f} km²"
                               if _oil.get('polygons') else ""))
                if _oil.get("polygons"):
                    _df_oil = pd.DataFrame([{
                        "ID":           p["id"],
                        "Area km²":     p.get("area_km2", "?"),
                        "Probability":  f"{p.get('probability','?')}%",
                        "Confidence":   p.get("confidence", "?"),
                        "ORM mean":     p.get("orm_mean", ""),
                        "Lat":          p["lat"],
                        "Lon":          p["lon"],
                    } for p in _oil["polygons"]])
                    st.dataframe(_df_oil, use_container_width=True, hide_index=True)
                else:
                    st.info("No oil anomalies detected for this date/area.")
                st.stop()

            # Vessels results
            _col1, _col2 = st.columns(2)
            with _col1:
                st.markdown(f"### 🛢️ Oil Anomalies · {_oil.get('n_anomalies', 0)}")
                st.info("Run Oil Spill Detection for oil results.")
            with _col2:
                _n_near = sum(1 for v in _ves.get("vessels", []) if v.get("near_oil"))
                st.markdown(f"### 🛸 Vessels · {_ves.get('n_vessels', 0)}"
                           + (f"  ·  ⚠ {_n_near} near oil" if _n_near else ""))
                if _ves.get("vessels"):
                    _df_ves = pd.DataFrame([{
                        "ID":         v["id"],
                        "Category":   v["category"],
                        "L min (m)":  v.get("length_min_m", "?"),
                        "L max (m)":  v.get("length_max_m", "?"),
                        "W min (m)":  v.get("width_min_m", "?"),
                        "W max (m)":  v.get("width_max_m", "?"),
                        "Confidence": v["confidence"],
                        "Near oil":   "⚠" if v.get("near_oil") else "",
                    } for v in _ves["vessels"]])
                    st.dataframe(_df_ves, use_container_width=True, hide_index=True)
                else:
                    st.info("No vessels detected.")

            st.caption(
                f"🛰 Sentinel-1 SAR · {_date} · {_s1_sel_orbit} orbit  "
                "·  ⚠ Oil detection requires optical validation  "
                "·  Vessel dimensions ±40%"
            )
        else:
            st.info(f"👈 Select a date and click **{_run_label}** in the sidebar.")

        # Stop here — don't render Water Quality below
        st.stop()

    # ==========================================================================
    # WATER QUALITY MODULE — lazy loading starts here
    # ==========================================================================

    # Landing page disabled
    if False:
        st.markdown("""
<div style="text-align:center;padding:48px 0 32px;">
  <div style="font-family:'Rajdhani',sans-serif;font-size:2.2rem;font-weight:700;
  color:#00c8c8;letter-spacing:0.12em;margin-bottom:12px;">⬡ MEDI PLATFORM</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.78rem;color:#7fb3d3;
  letter-spacing:0.16em;margin-bottom:40px;">SELECT A MODULE TO BEGIN</div>
</div>""", unsafe_allow_html=True)

        _lc1, _lc2, _lc3 = st.columns(3, gap="large")
        with _lc1:
            st.markdown("""<div style="background:rgba(0,200,200,0.06);border:1px solid rgba(0,200,200,0.2);
border-radius:10px;padding:32px 16px;text-align:center;margin-bottom:8px;">
<div style="font-size:2.4rem;margin-bottom:12px;">🌊</div>
<div style="font-family:'Rajdhani',sans-serif;font-size:1.1rem;font-weight:700;
color:#00c8c8;letter-spacing:0.06em;">Water Quality</div>
<div style="font-family:'Exo 2',sans-serif;font-size:0.75rem;color:#7fb3d3;margin-top:8px;">
S3 · S2 · MODIS · WQI</div></div>""", unsafe_allow_html=True)
            if st.button("Open →", key="land_wq", use_container_width=True):
                st.session_state.active_module = "🌊  Water Quality"
                st.rerun()

        with _lc2:
            st.markdown("""<div style="background:rgba(226,75,74,0.06);border:1px solid rgba(226,75,74,0.2);
border-radius:10px;padding:32px 16px;text-align:center;margin-bottom:8px;">
<div style="font-size:2.4rem;margin-bottom:12px;">🛢️</div>
<div style="font-family:'Rajdhani',sans-serif;font-size:1.1rem;font-weight:700;
color:#e24b4a;letter-spacing:0.06em;">Oil Spill Detection</div>
<div style="font-family:'Exo 2',sans-serif;font-size:0.75rem;color:#7fb3d3;margin-top:8px;">
Sentinel-1 SAR · Dark spot analysis</div></div>""", unsafe_allow_html=True)
            if st.button("Open →", key="land_oil", use_container_width=True):
                st.session_state.active_module = "🛢️  Oil Spill Detection"
                st.rerun()

        with _lc3:
            st.markdown("""<div style="background:rgba(55,138,221,0.06);border:1px solid rgba(55,138,221,0.2);
border-radius:10px;padding:32px 16px;text-align:center;margin-bottom:8px;">
<div style="font-size:2.4rem;margin-bottom:12px;">🛸</div>
<div style="font-family:'Rajdhani',sans-serif;font-size:1.1rem;font-weight:700;
color:#5aaacf;letter-spacing:0.06em;">Vessel Detection</div>
<div style="font-family:'Exo 2',sans-serif;font-size:0.75rem;color:#7fb3d3;margin-top:8px;">
Sentinel-1 SAR · Bright target detection</div></div>""", unsafe_allow_html=True)
            if st.button("Open →", key="land_ves", use_container_width=True):
                st.session_state.active_module = "🛸  Vessel Detection"
                st.rerun()

        st.stop()

    # Date selector — only runs after module selected
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
        all_candidates.sort(key=lambda x: x[0])  # freshest first (lowest age_hours)

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
        st.session_state.history_range = "30 ימים"

    if err:
        st.error(err)
    elif wqi_layer is not None:
        atm = get_atm(32.4, 34.85)

    # All monitoring areas now unified under user_zones
    city_wqi = {}

    # Compute user-defined zone WQI — LAZY: history only loads on demand
    user_zone_wqi = {}
    user_zone_history = {}
    if st.session_state.get("user_zones"):
        import json as _juz
        zones_json = _juz.dumps(st.session_state.user_zones)
        # Use cached history if already loaded, else leave empty until user requests
        if st.session_state.get("zone_history_loaded"):
            user_zone_history = st.session_state.get("zone_history_cache", {})
            for zname, zhistory in user_zone_history.items():
                vals = [e["wqi"] for e in zhistory if e["wqi"] is not None]
                user_zone_wqi[zname] = vals[-1] if vals else None

    # ── Current-date zone WQI — single reduceRegions call, no history needed ──
    # Bust any stale cache from previous broken runs
    for _stale_k in [k for k in st.session_state.keys() if k.startswith("zone_wqi_today_")]:
        del st.session_state[_stale_k]

    _debug_zone_wqi = st.expander("🔍 Debug", expanded=False)

    if st.session_state.get("user_zones") and wqi_layer is not None:
        _today_key = f"zone_wqi_today_{sel_date}_{data_source}"
        _debug_zone_wqi.write(f"Cache key: `{_today_key}`")
        _debug_zone_wqi.write(f"wqi_layer type: `{type(wqi_layer).__name__}`")
        _debug_zone_wqi.write(f"Zones: {list(st.session_state.user_zones.keys())}")

        if _today_key not in st.session_state:
            _debug_zone_wqi.write("→ Cache MISS — running GEE reduceRegions...")
            try:
                _wm  = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
                _img = wqi_layer.updateMask(_wm)

                # Log band names
                try:
                    _bands = _img.bandNames().getInfo()
                    _debug_zone_wqi.write(f"Band names: {_bands}")
                except Exception as _be:
                    _debug_zone_wqi.write(f"Band names error: {_be}")

                _features = []
                for zname, zdata in st.session_state.user_zones.items():
                    try:
                        coords = zdata.get("coords", [])
                        if not coords:
                            _debug_zone_wqi.write(f"⚠ {zname}: no coords, skipping")
                            continue
                        if zdata.get("type") == "point":
                            geom = ee.Geometry.Point(
                                [zdata["lon"], zdata["lat"]]
                            ).buffer(500)
                        else:
                            geom = ee.Geometry.Polygon(
                                [[[c[0], c[1]] for c in coords]]
                            )
                        _features.append(ee.Feature(geom, {"name": zname}))
                    except Exception as _fe:
                        _debug_zone_wqi.write(f"⚠ {zname} feature error: {_fe}")

                _debug_zone_wqi.write(f"Features built: {len(_features)}")

                if _features:
                    _fc  = ee.FeatureCollection(_features)
                    _res = _img.reduceRegions(
                        collection=_fc,
                        reducer=ee.Reducer.mean(),
                        scale=300
                    ).getInfo()

                    # Log raw GEE output for first feature
                    if _res.get("features"):
                        _debug_zone_wqi.write("Raw GEE props (first feature):")
                        _debug_zone_wqi.json(_res["features"][0].get("properties", {}))

                    _today = {}
                    for feat in _res.get("features", []):
                        props = feat.get("properties", {})
                        nm    = props.get("name")
                        # reduceRegions mean() → band name is the key
                        # Try all possible keys GEE might use
                        wv = None
                        for _k in ["WQI", "mean", "WQI_mean", "b1"]:
                            if props.get(_k) is not None:
                                wv = props[_k]
                                break
                        _today[nm] = round(float(wv), 1) if wv is not None else None

                    _debug_zone_wqi.write(f"Computed WQI values: {_today}")
                    st.session_state[_today_key] = _today
                else:
                    _debug_zone_wqi.write("⚠ No features built — zones may have no coords")
                    st.session_state[_today_key] = {}

            except Exception as _ze:
                _debug_zone_wqi.write(f"❌ GEE error: {_ze}")
                st.session_state[_today_key] = {}
        else:
            _debug_zone_wqi.write("→ Cache HIT")
            _debug_zone_wqi.write(st.session_state.get(_today_key, {}))

        # Merge into user_zone_wqi — history values take precedence if loaded
        for zname, wv in st.session_state.get(_today_key, {}).items():
            if zname not in user_zone_wqi:
                user_zone_wqi[zname] = wv

        _debug_zone_wqi.write(f"Final user_zone_wqi: {user_zone_wqi}")
    else:
        _debug_zone_wqi.write(f"Skipped: user_zones={bool(st.session_state.get('user_zones'))}, wqi_layer={wqi_layer is not None}")



    # Basemap definitions — full URL templates for both folium and L.tileLayer
    # 'sub' is the subdomains string for tiles using {s}; None = no subdomains.
    BASEMAPS = {
        "Satellite":  {"tile": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                       "attr": "Esri", "sub": None},
        "Ocean":      {"tile": "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
                       "attr": "Esri Ocean Base", "sub": None},
        "Bathymetry": {"tile": "https://tiles.emodnet-bathymetry.eu/2020/baselayer/web_mercator/{z}/{x}/{y}.png",
                       "attr": "EMODnet Bathymetry", "sub": None},
        "Terrain":    {"tile": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}",
                       "attr": "Esri Terrain", "sub": None},
        "NatGeo":     {"tile": "https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
                       "attr": "Esri / National Geographic", "sub": None},
        "Street":     {"tile": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                       "attr": "OpenStreetMap", "sub": None},
        "Dark":       {"tile": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
                       "attr": "CartoDB Dark", "sub": "abcd"},
        "Light":      {"tile": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
                       "attr": "CartoDB Positron", "sub": "abcd"},
    }
    if "basemap" not in st.session_state:
        st.session_state.basemap = "Satellite"

    if "show_zones_on_map" not in st.session_state:
        st.session_state.show_zones_on_map = True
    if "sat_panel_open" not in st.session_state:
        st.session_state.sat_panel_open = False
    if "sat_view_mode" not in st.session_state:
        st.session_state.sat_view_mode = "wqi"   # "wqi" | "true_color" | "swipe"
    if "sat_opacity" not in st.session_state:
        st.session_state.sat_opacity = 0.85
    if "inspect_mode" not in st.session_state:
        st.session_state.inspect_mode = False
    if "spectra_click" not in st.session_state:
        st.session_state.spectra_click = None
    if "spectra_result" not in st.session_state:
        st.session_state.spectra_result = None
    if "s1_mode" not in st.session_state:
        st.session_state["s1_mode"] = False
    if "s1_result" not in st.session_state:
        st.session_state["s1_result"] = None
    if "s1_date" not in st.session_state:
        st.session_state["s1_date"] = None

    @st.cache_data(ttl=7200)
    def _get_true_color_tile(source: str, target_date_str: str):
        """Return GEE tile URL for the raw (true-color) satellite image."""
        wm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(10)
        t  = ee.Date(target_date_str)
        # Use a wider display area — full Mediterranean coast + some inland
        DISPLAY_BOX = ee.Geometry.Rectangle([33.0, 31.2, 35.1, 33.2])  # Med coast only
        try:
            if source == "S3":
                coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                        .filterBounds(DISPLAY_BOX)
                        .filterDate(t.advance(-3, "day"), t.advance(1, "day")))
                if coll.size().getInfo() == 0:
                    return None
                img = coll.median().clip(DISPLAY_BOX)
                vis = {"bands": ["Oa08_radiance", "Oa06_radiance", "Oa04_radiance"],
                       "min": 0, "max": 120, "gamma": 1.4}
            elif source == "S2":
                coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(DISPLAY_BOX)
                        .filterDate(t.advance(-8, "day"), t.advance(1, "day"))
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
                        .sort("system:time_start", False))
                if coll.size().getInfo() == 0:
                    return None
                img = coll.mosaic().clip(DISPLAY_BOX)
                vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.3}
            else:  # MODIS
                terra = (ee.ImageCollection("MODIS/061/MOD09GA")
                         .filterBounds(DISPLAY_BOX)
                         .filterDate(t.advance(-3, "day"), t.advance(1, "day"))
                         .sort("system:time_start", False))
                if terra.size().getInfo() == 0:
                    return None
                img = terra.mosaic().clip(DISPLAY_BOX)
                vis = {"bands": ["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"],
                       "min": 0, "max": 3000, "gamma": 1.4}
            mid = img.getMapId(vis)
            return mid["tile_fetcher"].url_format
        except Exception:
            return None

    @st.cache_data(ttl=7200)
    def _get_wqi_tile(source: str, target_date_str: str):
        """Return GEE tile URL for WQI layer of given source — uses same pipeline as main map."""
        vis = {"min": 30, "max": 90,
               "palette": ["#d73027","#f46d43","#fdae61","#fee090",
                           "#e0f3f8","#abd9e9","#74add1","#4575b4"]}
        try:
            if source == "S3":
                layer, _, err, _ = process_israel_wqi(target_date_str)
            elif source == "S2":
                layer, _, err, _, _ = process_israel_s2(target_date_str)
            else:
                layer, _, err, _, _ = process_modis_wqi(target_date_str)
            if err or layer is None:
                return None
            mid = ee.Image(layer).getMapId(vis)
            return mid["tile_fetcher"].url_format
        except Exception:
            return None

    @st.cache_data(ttl=7200)
    def _get_spectral_index_tiles(source: str, target_date_str: str):
        """Return dict of {index_name: tile_url} for normalized spectral indices.
        Uses the SAME normalization as the WQI pipeline so values are meaningful 0-1."""
        DISPLAY_BOX = ee.Geometry.Rectangle([33.0, 31.2, 35.1, 33.2])  # Med coast only
        wm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(10)
        t  = ee.Date(target_date_str)
        # Shared palettes for normalized 0→1 indices
        PAL_WATER = ["#8B4513","#D2B48C","#FFFACD","#87CEEB","#0000CD"]  # brown→blue (0=land, 1=water)
        PAL_CHL   = ["#4575b4","#91bfdb","#ffffbf","#fc8d59","#d73027"]  # blue→red (0=low, 1=high CHL)
        PAL_TURB  = ["#4575b4","#74add1","#ffffbf","#f46d43","#8B4513"]  # blue→brown (0=clear, 1=turbid)
        result = {}
        try:
            if source == "S3":
                coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                        .filterBounds(DISPLAY_BOX)
                        .filterDate(t.advance(-3, "day"), t.advance(1, "day"))
                        .sort("system:time_start", False))
                if coll.size().getInfo() == 0:
                    return result
                img = coll.mosaic().clip(DISPLAY_BOX)
                # NDWI (normalized exactly like WQI pipeline)
                ndwi = img.normalizedDifference(["Oa06_radiance", "Oa17_radiance"])
                ndwi_n = ndwi.unitScale(-0.2, 0.5).clamp(0, 1).updateMask(wm)
                mid = ndwi_n.getMapId({"min": 0, "max": 1, "palette": PAL_WATER})
                result["NDWI"] = mid["tile_fetcher"].url_format
                # MCI (Oa10-Oa09, normalized as in WQI: unitScale -2 to 12)
                mci = img.select("Oa10_radiance").subtract(img.select("Oa09_radiance"))
                mci_n = mci.unitScale(-2, 12).clamp(0, 1).updateMask(wm)
                mid = mci_n.getMapId({"min": 0, "max": 1, "palette": PAL_CHL})
                result["MCI (Chlorophyll)"] = mid["tile_fetcher"].url_format
                # Turbidity (Oa08, normalized as in WQI: unitScale 10 to 80)
                turb = img.select("Oa08_radiance")
                turb_n = turb.unitScale(10, 80).clamp(0, 1).updateMask(wm)
                mid = turb_n.getMapId({"min": 0, "max": 1, "palette": PAL_TURB})
                result["Turbidity"] = mid["tile_fetcher"].url_format

            elif source == "S2":
                coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(DISPLAY_BOX)
                        .filterDate(t.advance(-8, "day"), t.advance(1, "day"))
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
                        .sort("system:time_start", False))
                if coll.size().getInfo() == 0:
                    return result
                img = coll.mosaic().clip(DISPLAY_BOX)
                b3 = img.select("B3").divide(10000)
                b4 = img.select("B4").divide(10000)
                b5 = img.select("B5").divide(10000)
                b8 = img.select("B8").divide(10000)
                b8a = img.select("B8A").divide(10000)
                # NDWI (normalized as in WQI: unitScale -0.3 to 0.5)
                ndwi = b3.subtract(b8).divide(b3.add(b8))
                ndwi_n = ndwi.unitScale(-0.3, 0.5).clamp(0, 1).updateMask(wm)
                mid = ndwi_n.getMapId({"min": 0, "max": 1, "palette": PAL_WATER})
                result["NDWI"] = mid["tile_fetcher"].url_format
                # CHL proxy (B5/B4, normalized: unitScale 1.0 to 3.5)
                chl = b5.divide(b4.add(0.000001))
                chl_n = chl.unitScale(1.0, 3.5).clamp(0, 1).updateMask(wm)
                mid = chl_n.getMapId({"min": 0, "max": 1, "palette": PAL_CHL})
                result["CHL Proxy"] = mid["tile_fetcher"].url_format
                # Turbidity ((B4+B8A)/2, normalized: unitScale 0 to 0.15)
                turb = b4.add(b8a).divide(2)
                turb_n = turb.unitScale(0, 0.15).clamp(0, 1).updateMask(wm)
                mid = turb_n.getMapId({"min": 0, "max": 1, "palette": PAL_TURB})
                result["Turbidity"] = mid["tile_fetcher"].url_format

            else:  # MODIS
                terra = (ee.ImageCollection("MODIS/061/MOD09GA")
                         .filterBounds(DISPLAY_BOX)
                         .filterDate(t.advance(-3, "day"), t.advance(1, "day"))
                         .sort("system:time_start", False))
                if terra.size().getInfo() == 0:
                    return result
                img = terra.mosaic().clip(DISPLAY_BOX)
                b1 = img.select("sur_refl_b01")
                b2 = img.select("sur_refl_b02")
                b4 = img.select("sur_refl_b04")
                # NDWI (normalized: unitScale -0.3 to 0.3)
                ndwi = b4.subtract(b2).divide(b4.add(b2))
                ndwi_n = ndwi.unitScale(-0.3, 0.3).clamp(0, 1).updateMask(wm)
                mid = ndwi_n.getMapId({"min": 0, "max": 1, "palette": PAL_WATER})
                result["NDWI"] = mid["tile_fetcher"].url_format
                # CHL proxy (B4/B1, normalized: unitScale 0.8 to 2.5)
                chl = b4.divide(b1.add(1))
                chl_n = chl.unitScale(0.8, 2.5).clamp(0, 1).updateMask(wm)
                mid = chl_n.getMapId({"min": 0, "max": 1, "palette": PAL_CHL})
                result["CHL Proxy"] = mid["tile_fetcher"].url_format
                # Turbidity (B1, normalized: unitScale 0 to 1500)
                turb_n = b1.unitScale(0, 1500).clamp(0, 1).updateMask(wm)
                mid = turb_n.getMapId({"min": 0, "max": 1, "palette": PAL_TURB})
                result["Turbidity"] = mid["tile_fetcher"].url_format
        except Exception:
            pass
        return result

    # ── MEDI Confidence Score (MCS) ─────────────────────────────────────────────
    @st.cache_data(ttl=7200)
    def compute_confidence_scores(source: str, date_str: str, zones_json: str,
                                  img_age_hours: float,
                                  s3_wqi_json: str, s2_wqi_json: str, mod_wqi_json: str,
                                  history_json: str):
        """Compute 0-100 confidence score per zone with 7 factor breakdown.
        Uses ONE reduceRegions GEE call for satellite quality factors + Python for the rest."""
        import json as _cj

        zones = _cj.loads(zones_json)
        s3_wqi = _cj.loads(s3_wqi_json) if s3_wqi_json else {}
        s2_wqi = _cj.loads(s2_wqi_json) if s2_wqi_json else {}
        mod_wqi = _cj.loads(mod_wqi_json) if mod_wqi_json else {}
        history = _cj.loads(history_json) if history_json else {}

        results = {}
        DISPLAY_BOX = ee.Geometry.Rectangle([33.0, 31.2, 35.1, 33.2])  # Med coast only
        wm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(10)
        t = ee.Date(date_str)

        # ── 1. Build FeatureCollection from zones ──────────────────────────
        features = []
        zone_names_ordered = []
        for name, zdata in zones.items():
            try:
                if zdata.get("type") == "point":
                    lat = zdata.get("lat", zdata.get("coords", [0,0])[1] if isinstance(zdata.get("coords"), list) else 0)
                    lon = zdata.get("lon", zdata.get("coords", [0,0])[0] if isinstance(zdata.get("coords"), list) else 0)
                    geom = ee.Geometry.Point([lon, lat]).buffer(500)
                else:
                    coords = zdata.get("coords", [])
                    if not coords:
                        continue
                    geom = ee.Geometry.Polygon([[[c[0], c[1]] for c in coords]])
                features.append(ee.Feature(geom, {"name": name}))
                zone_names_ordered.append(name)
            except Exception:
                continue

        if not features:
            return results

        fc = ee.FeatureCollection(features)

        # ── 2. Get satellite image + quality bands ─────────────────────────
        sun_zenith_score = 0.8  # default
        try:
            if source in ("S3", "Sentinel-3"):
                coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                        .filterBounds(DISPLAY_BOX)
                        .filterDate(t.advance(-3, "day"), t.advance(1, "day"))
                        .sort("system:time_start", False))
                img = coll.first()
                # Valid water: NOT cloud (bit 27) AND water mask
                qf = img.select("quality_flags")
                cloud = qf.bitwiseAnd(1 << 27).gt(0)
                valid = wm.And(cloud.Not()).rename("valid")
                # QA: also check sun glint (bit 22)
                glint = qf.bitwiseAnd(1 << 22).gt(0)
                qa_good = valid.And(glint.Not()).rename("qa")
                # Sun angle from metadata
                try:
                    sza = img.getNumber("SZA").getInfo()
                    sun_zenith_score = max(0, min(1, 1.0 - (sza - 20) / 60))  # 20°=best, 80°=worst
                except Exception:
                    pass

            elif source in ("S2", "Sentinel-2"):
                coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(DISPLAY_BOX)
                        .filterDate(t.advance(-8, "day"), t.advance(1, "day"))
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
                        .sort("system:time_start", False))
                img = coll.first()
                scl = img.select("SCL")
                water = scl.eq(6)
                cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
                valid = wm.And(cloud.Not()).rename("valid")
                qa_good = water.And(cloud.Not()).rename("qa")
                try:
                    sza = img.get("MEAN_SOLAR_ZENITH_ANGLE").getInfo()
                    sun_zenith_score = max(0, min(1, 1.0 - (sza - 20) / 60))
                except Exception:
                    pass

            else:  # MODIS
                coll = (ee.ImageCollection("MODIS/061/MOD09GA")
                        .filterBounds(DISPLAY_BOX)
                        .filterDate(t.advance(-3, "day"), t.advance(1, "day"))
                        .sort("system:time_start", False))
                img = coll.first()
                state = img.select("state_1km")
                cloud = state.bitwiseAnd(3).gt(0)  # bits 0-1: cloud state
                shadow = state.bitwiseAnd(4).gt(0)  # bit 2: cloud shadow
                valid = wm.And(cloud.Not()).And(shadow.Not()).rename("valid")
                qa_good = valid.rename("qa")  # MODIS QA is simpler
                try:
                    sza = img.get("SolarZenith").getInfo()
                    if sza: sun_zenith_score = max(0, min(1, 1.0 - (float(sza)/100 - 20) / 60))
                except Exception:
                    pass

            # ── 3. ONE reduceRegions call for all zones ────────────────────
            stack = valid.addBands(qa_good).addBands(ee.Image.constant(1).rename("total"))
            scale = 300 if source not in ("S2", "Sentinel-2") else 100

            stats = stack.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.sum().forEachBand(stack),
                scale=scale
            ).getInfo()

        except Exception:
            # If GEE fails, return basic scores from Python-only factors
            stats = {"features": []}

        # ── 4. Parse GEE results + compute all 7 factors ──────────────────
        gee_stats = {}
        for feat in stats.get("features", []):
            props = feat.get("properties", {})
            name = props.get("name", "")
            valid_px = props.get("valid", 0) or 0
            qa_px = props.get("qa", 0) or 0
            total_px = props.get("total", 0) or 0
            gee_stats[name] = {"valid_px": valid_px, "qa_px": qa_px, "total_px": total_px}

        for name in zone_names_ordered:
            gs = gee_stats.get(name, {"valid_px": 0, "qa_px": 0, "total_px": 1})
            total = max(gs["total_px"], 1)

            # Factor 1: Cloud-free ratio (25%)
            cloud_free = min(1.0, gs["valid_px"] / total)

            # Factor 2: Image age (20%) — exponential decay: 0h=1.0, 24h=0.7, 72h=0.3, 120h=0.1
            age_score = max(0, min(1, math.exp(-0.012 * img_age_hours)))

            # Factor 3: QA flag ratio (20%)
            qa_ratio = min(1.0, gs["qa_px"] / total)

            # Factor 4: Pixel count (15%) — log scale: 1px=0.1, 10px=0.5, 50px=0.85, 100+=1.0
            px_count = gs["valid_px"]
            px_score = min(1.0, math.log10(max(px_count, 1) + 1) / 2.0)

            # Factor 5: Sun angle (10%)
            sun_score = sun_zenith_score

            # Factor 6: Cross-sensor consistency (5%)
            vals = []
            if name in s3_wqi and s3_wqi[name] is not None: vals.append(s3_wqi[name])
            if name in s2_wqi and s2_wqi[name] is not None: vals.append(s2_wqi[name])
            if name in mod_wqi and mod_wqi[name] is not None: vals.append(mod_wqi[name])
            if len(vals) >= 2:
                spread = max(vals) - min(vals)
                cross_score = max(0, 1.0 - spread / 30.0)  # 0 spread=1.0, 30+ spread=0
            else:
                cross_score = 0.5  # unknown, neutral

            # Factor 7: Temporal consistency (5%)
            hist_vals = history.get(name, [])
            recent = [h for h in hist_vals if h is not None][-7:]
            if len(recent) >= 3:
                avg7 = sum(recent) / len(recent)
                std7 = (sum((v - avg7)**2 for v in recent) / len(recent)) ** 0.5
                current = s3_wqi.get(name) or s2_wqi.get(name) or mod_wqi.get(name)
                if current is not None and std7 > 0:
                    z_score = abs(current - avg7) / max(std7, 1)
                    temporal_score = max(0, 1.0 - z_score / 3.0)  # 0σ=1.0, 3σ=0
                else:
                    temporal_score = 0.5
            else:
                temporal_score = 0.5  # not enough history

            # Weighted MCS
            mcs = (cloud_free * 0.25 + age_score * 0.20 + qa_ratio * 0.20 +
                   px_score * 0.15 + sun_score * 0.10 + cross_score * 0.05 +
                   temporal_score * 0.05)
            mcs_pct = round(mcs * 100)

            results[name] = {
                "score": mcs_pct,
                "grade": "🟢" if mcs_pct >= 75 else "🟡" if mcs_pct >= 50 else "🔴",
                "factors": {
                    "cloud_free": round(cloud_free * 100),
                    "age": round(age_score * 100),
                    "qa_flags": round(qa_ratio * 100),
                    "pixels": int(px_count),
                    "px_score": round(px_score * 100),
                    "sun_angle": round(sun_score * 100),
                    "cross_sensor": round(cross_score * 100),
                    "temporal": round(temporal_score * 100),
                }
            }

        return results

    # Shared map builder
    def _build_map(selected_beach=None):
        bm      = st.session_state.get("basemap", "Satellite")
        bm_data = BASEMAPS.get(bm, BASEMAPS["Satellite"])
        m = folium.Map(
            location=[32.4, 34.85],
            zoom_start=8,
            tiles=bm_data["tile"],
            attr=bm_data["attr"]
        )
        # Add draw plugin — polygon + marker
        Draw(
            export=False,
            draw_options={
                "polygon":      {"allowIntersection": False},
                "rectangle":    True,
                "marker":       True,
                "circle":       False,
                "polyline":     False,
                "circlemarker": False,
            },
            edit_options={"edit": False}
        ).add_to(m)

        # Note: basemap tile layers are NOT added here; the custom JS basemap
        # button (topleft) manages basemap switching directly.
        vis = {'min':30,'max':90,'palette':['#d73027','#f46d43','#fdae61','#fee090','#e0f3f8','#abd9e9','#74add1','#4575b4']}
        wqi_tile_url = None

        # ── Collect ALL raster tile URLs for every available source ──────────────
        # Passed into the custom JS "Satellite Layers" panel button (below layers icon).
        # Only the active/visible layer is added via folium; others are toggled in JS.
        _raster_layers = []  # list of {id, label, date, url, visible}

        def _age_to_date(age_h):
            if age_h is None: return ""
            try:
                return (datetime.utcnow() - timedelta(hours=age_h)).strftime("%Y-%m-%d")
            except Exception:
                return ""

        # ── Determine active sensor and its date ────────────────────────────────
        _src_abbr = "S3" if data_source in ("S3","Sentinel-3") else \
                    "S2" if data_source in ("S2","Sentinel-2") else "MODIS"
        _active_age = s3_age if _src_abbr=="S3" else s2_age if _src_abbr=="S2" else mod_age
        _active_date = _age_to_date(_active_age)
        _active_layer = s3_layer if _src_abbr=="S3" else s2_layer if _src_abbr=="S2" else mod_layer

        # ── Only show products for the ACTIVE sensor ────────────────────────────
        # 1. WQI composite — use cached _get_wqi_tile (fresh GEE tile URL)
        try:
            _wqi_url = _get_wqi_tile(_src_abbr, sel_date)
            if _wqi_url:
                _wqi_vis = {"palette":["#d73027","#f46d43","#fdae61","#fee090","#e0f3f8","#abd9e9","#74add1","#4575b4"],"min":30,"max":90,"unit":"WQI","minLabel":"Polluted","maxLabel":"Clean"}
                _raster_layers.append({"id":"wqi_active","label":"WQI \u00b7 "+data_source,"date":_active_date,"url":_wqi_url,"visible":True,"vis":_wqi_vis})
                wqi_tile_url = _wqi_url
        except Exception:
            pass

        # 2. True Color (raw satellite image) — no legend
        try:
            _tc_url = _get_true_color_tile(_src_abbr, sel_date)
            if _tc_url:
                _raster_layers.append({"id":"tc_active","label":"True Color \u00b7 "+data_source,"date":_active_date,"url":_tc_url,"visible":False,"vis":None})
        except Exception:
            pass

        # 3. Spectral indices with vis metadata
        _idx_vis_map = {
            "NDWI": {"palette":["#8B4513","#D2B48C","#FFFACD","#87CEEB","#0000CD"],"min":0,"max":1,"unit":"NDWI (normalized)","minLabel":"Land/Dry","maxLabel":"Water"},
            "MCI (Chlorophyll)": {"palette":["#4575b4","#91bfdb","#ffffbf","#fc8d59","#d73027"],"min":0,"max":1,"unit":"MCI (normalized)","minLabel":"Low CHL","maxLabel":"High CHL"},
            "CHL Proxy": {"palette":["#4575b4","#91bfdb","#ffffbf","#fc8d59","#d73027"],"min":0,"max":1,"unit":"CHL (normalized)","minLabel":"Low","maxLabel":"High"},
            "Turbidity": {"palette":["#4575b4","#74add1","#ffffbf","#f46d43","#8B4513"],"min":0,"max":1,"unit":"Turbidity (normalized)","minLabel":"Clear","maxLabel":"Turbid"},
        }
        try:
            _idx_tiles = _get_spectral_index_tiles(_src_abbr, sel_date)
            for idx_name, idx_url in _idx_tiles.items():
                safe_id = "idx_" + idx_name.lower().replace(" ","_").replace("(","").replace(")","")
                idx_vis = _idx_vis_map.get(idx_name)
                _raster_layers.append({"id":safe_id,"label":idx_name+" \u00b7 "+data_source,"date":_active_date,"url":idx_url,"visible":False,"vis":idx_vis})
        except Exception:
            pass

        # ── Sentinel-1 SAR layers ────────────────────────────────────────────────
        _s1_mode_on = st.session_state.get("s1_mode", False)
        try:
            _s1r = st.session_state.get("s1_result")
            if _s1r and _s1r.get("layers"):
                _s1_lyr = _s1r["layers"]
                _s1_date_str = _s1r.get("date", "")
                _s1_vis_vv = {"palette":["#000014","#0a1520","#152840","#1e3a5a","#5aaacf","#c8e8f8"],"min":-25,"max":0,"unit":"VV backscatter (dB)","minLabel":"Low","maxLabel":"High"}
                _s1_vis_ratio = {"palette":["#041e33","#1D9E75","#fdae61","#d73027"],"min":0,"max":15,"unit":"VV/VH ratio (dB)","minLabel":"Low","maxLabel":"High"}
                if _s1_lyr.get("vv"):
                    _raster_layers.append({"id":"s1_vv","label":"VV backscatter · S1","date":_s1_date_str,"url":_s1_lyr["vv"],"visible":True,"vis":_s1_vis_vv})
                if _s1_lyr.get("vh"):
                    _raster_layers.append({"id":"s1_vh","label":"VH backscatter · S1","date":_s1_date_str,"url":_s1_lyr["vh"],"visible":False,"vis":None})
                if _s1_lyr.get("ratio"):
                    _raster_layers.append({"id":"s1_ratio","label":"VV/VH ratio · S1","date":_s1_date_str,"url":_s1_lyr["ratio"],"visible":False,"vis":_s1_vis_ratio})
                if _s1_lyr.get("rgb"):
                    _raster_layers.append({"id":"s1_rgb","label":"RGB composite · S1","date":_s1_date_str,"url":_s1_lyr["rgb"],"visible":False,"vis":None})
                # When S1 is active, hide all optical layers
                if _s1_mode_on:
                    for _rl in _raster_layers:
                        if not _rl["id"].startswith("s1_"):
                            _rl["visible"] = False
        except Exception:
            pass

        # All raster layers managed exclusively by JS (no folium TileLayer for rasters)
        # The folium.Map only has the basemap. JS pre-creates all tile layers and
        # adds/removes them based on checkbox state.

        # ── Custom Leaflet controls via MacroElement (script macro = runs AFTER map exists) ──
        # Topleft: 🗂 Basemaps | Topright: 🛰 Satellite Products | ⛶ Fullscreen | 📏 Ruler
        import json as _cjson
        _rl_json = _cjson.dumps(_raster_layers)
        _sel_date_js = sel_date
        _active_src_js = f"{data_source} \u00b7 {_active_date}"
        _bm_list_js = [{"id": name, "name": name, "url": data["tile"], "attr": data["attr"],
                        "sub": data.get("sub")}
                       for name, data in BASEMAPS.items()]
        _bm_json = _cjson.dumps(_bm_list_js)
        _active_bm_js = st.session_state.get("basemap", "Satellite")

        # Pre-substitute placeholder values into the JS body
        _js_body = """
(function() {
  try {
    var mapObj = __MAP_VAR__;
    if (!mapObj || typeof L === 'undefined') { console.warn('[MEDI] map or L not ready'); return; }

    var _rasterLayers = __RL_JSON__;
    var _sel_date = "__SEL_DATE__";
    var _activeSrc = "__ACTIVE_SRC__";
    var _basemaps = __BM_JSON__;
    var _activeBasemapId = "__ACTIVE_BM__";
    var _bmLayerRef = null;
    var _tileRegistry = {};
    var _opacity = 0.75;

    // Find the existing basemap tile layer added by folium and track it
    mapObj.eachLayer(function(l) {
      if (l._url && !l._isSatLayer && _bmLayerRef === null) {
        _bmLayerRef = l;
      }
    });

    // Pre-create ALL raster tile layers and add ALL to map (hidden ones at opacity 0)
    // This prevents st_folium from detecting layer changes and triggering reruns
    _rasterLayers.forEach(function(rl) {
      var l = L.tileLayer(rl.url, {opacity: rl.visible ? _opacity : 0, attribution: 'GEE', zIndex: 500});
      l._isSatLayer = true;
      _tileRegistry[rl.id] = l;
      l.addTo(mapObj);  // Always add — use opacity for visibility
    });

    function setLayerVisible(id, on) {
      var l = _tileRegistry[id]; if (!l) return;
      l.setOpacity(on ? _opacity : 0);  // Opacity toggle only — no add/remove
    }

    var BTN_STYLE = 'display:flex;align-items:center;justify-content:center;width:30px;height:30px;font-size:16px;text-decoration:none;background:rgba(2,13,24,0.92);color:#00c8c8;border:1px solid rgba(0,200,200,0.4);cursor:pointer;box-sizing:border-box;';
    var PANEL_RIGHT = 'position:absolute;right:36px;top:0;background:rgba(2,13,24,0.97);border:1px solid rgba(0,200,200,0.4);border-radius:6px;padding:10px 13px;width:280px;font-family:Arial,sans-serif;font-size:13px;color:#d6eaf8;z-index:9999;box-shadow:-4px 6px 20px rgba(0,0,0,0.7);';
    var PANEL_LEFT  = 'position:absolute;left:36px;top:0;background:rgba(2,13,24,0.97);border:1px solid rgba(0,200,200,0.4);border-radius:6px;padding:10px 13px;width:200px;font-family:Arial,sans-serif;font-size:13px;color:#d6eaf8;z-index:9999;box-shadow:4px 6px 20px rgba(0,0,0,0.7);';

    function setBasemap(id) {
      var bm = null;
      for (var i=0;i<_basemaps.length;i++) { if (_basemaps[i].id === id) { bm = _basemaps[i]; break; } }
      if (!bm) return;
      var opts = {attribution: bm.attr, zIndex: 1};
      if (bm.sub) opts.subdomains = bm.sub;
      var nl = L.tileLayer(bm.url, opts);
      nl.addTo(mapObj); nl.bringToBack();
      if (_bmLayerRef && _bmLayerRef !== nl) { try { mapObj.removeLayer(_bmLayerRef); } catch(e){} }
      _bmLayerRef = nl; _activeBasemapId = id;
    }

    // ── COMBINED LAYERS CONTROL (topleft) ── basemap + satellite in one bar ──
    var bmOpen = false, bmPanel = null;
    var satOpen = false, satPanel = null;
    var layersCtrl = L.control({position: 'topleft'});
    layersCtrl.onAdd = function() {
      var d = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
      d.style.marginTop = '4px';

      // -- Basemap button --
      var bmBtn = document.createElement('a');
      bmBtn.href = '#'; bmBtn.title = 'Background Maps';
      bmBtn.style.cssText = BTN_STYLE;
      bmBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00c8c8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>';
      bmBtn.addEventListener('click', function(e) {
        e.preventDefault();
        if (satOpen) { satOpen = false; satBtn.style.background = 'rgba(2,13,24,0.92)'; if (satPanel) { d.removeChild(satPanel); satPanel = null; } }
        bmOpen = !bmOpen;
        if (bmOpen) {
          bmBtn.style.background = 'rgba(0,200,200,0.25)';
          bmPanel = buildBmPanel(); d.appendChild(bmPanel);
        } else {
          bmBtn.style.background = 'rgba(2,13,24,0.92)';
          if (bmPanel) { d.removeChild(bmPanel); bmPanel = null; }
        }
      });
      d.appendChild(bmBtn);

      // -- Satellite button (below basemap in same bar) --
      var satBtn = document.createElement('a');
      satBtn.href = '#'; satBtn.title = 'Satellite Products';
      satBtn.style.cssText = BTN_STYLE + 'border-top:none;';
      satBtn.innerHTML = '\\ud83d\\udef0';
      satBtn.addEventListener('click', function(e) {
        e.preventDefault();
        if (bmOpen) { bmOpen = false; bmBtn.style.background = 'rgba(2,13,24,0.92)'; if (bmPanel) { d.removeChild(bmPanel); bmPanel = null; } }
        satOpen = !satOpen;
        if (satOpen) {
          satBtn.style.background = 'rgba(0,200,200,0.25)';
          satPanel = buildSatPanel(); d.appendChild(satPanel);
        } else {
          satBtn.style.background = 'rgba(2,13,24,0.92)';
          if (satPanel) { d.removeChild(satPanel); satPanel = null; }
        }
      });
      d.appendChild(satBtn);

      L.DomEvent.disableClickPropagation(d);
      return d;
    };
    function buildBmPanel() {
      var p = document.createElement('div');
      p.style.cssText = PANEL_LEFT;
      var rows = '';
      _basemaps.forEach(function(bm) {
        var chk = (bm.id === _activeBasemapId) ? 'checked' : '';
        rows += '<label style="display:flex;align-items:center;gap:7px;margin-bottom:6px;cursor:pointer;">' +
          '<input type="radio" name="bm_radio" value="' + bm.id + '" ' + chk + ' style="accent-color:#00c8c8;width:14px;height:14px;cursor:pointer;">' +
          '<span style="font-size:12px;color:#d6eaf8;">' + bm.name + '</span></label>';
      });
      p.innerHTML = '<div style="font-weight:bold;color:#00c8c8;font-size:12px;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px;border-bottom:1px solid rgba(0,200,200,0.18);padding-bottom:5px;">Background Map</div>' + rows;
      L.DomEvent.disableClickPropagation(p);
      setTimeout(function() {
        var radios = p.querySelectorAll('input[name="bm_radio"]');
        radios.forEach(function(r) { r.addEventListener('change', function() { if (r.checked) setBasemap(r.value); }); });
      }, 60);
      return p;
    }
    layersCtrl.addTo(mapObj);
    function buildSatPanel() {
      var p = document.createElement('div');
      p.style.cssText = PANEL_LEFT.replace('width:200px', 'width:300px');
      // Sort: active sensor's products first
      var srt = _rasterLayers.slice().sort(function(a, b){
        return (b.visible ? 1 : 0) - (a.visible ? 1 : 0);
      });
      var rows = '';
      if (srt.length === 0) {
        rows = '<div style="color:#7fb3d3;font-size:11px;padding:4px 0;">No raster data for this date</div>';
      } else {
        srt.forEach(function(rl) {
          var chk = rl.visible ? 'checked' : '';
          var dateBadge = rl.date ? '<span style="font-size:10px;color:#7fb3d3;background:rgba(0,200,200,0.08);padding:1px 5px;border-radius:3px;margin-left:auto;flex-shrink:0;">' + rl.date + '</span>' : '';
          var rowBg = rl.visible ? 'background:rgba(0,200,200,0.10);border-left:2px solid #00c8c8;' : 'border-left:2px solid transparent;';
          rows += '<label style="display:flex;align-items:center;gap:7px;margin-bottom:4px;padding:4px 6px;cursor:pointer;border-radius:3px;' + rowBg + '">' +
            '<input type="checkbox" id="rl_cb_' + rl.id + '" ' + chk + ' style="accent-color:#00c8c8;width:14px;height:14px;cursor:pointer;flex-shrink:0;">' +
            '<span style="font-size:12px;color:#d6eaf8;flex:1;min-width:0;">' + rl.label + '</span>' + dateBadge + '</label>';
        });
      }
      p.innerHTML = '<div style="font-weight:bold;color:#00c8c8;font-size:12px;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px;border-bottom:1px solid rgba(0,200,200,0.18);padding-bottom:5px;">\\ud83d\\udef0 ' + _activeSrc + '</div>' + rows + '<div style="border-top:1px solid rgba(0,200,200,0.15);margin-top:6px;padding-top:7px;"><label style="display:block;color:#7fb3d3;font-size:11px;margin-bottom:3px;">Opacity: <span id="satOpVal">' + Math.round(_opacity*100) + '%</span></label><input id="satOpSlider" type="range" min="10" max="100" value="' + Math.round(_opacity*100) + '" style="width:100%;accent-color:#00c8c8;"></div>';
      L.DomEvent.disableClickPropagation(p);
      setTimeout(function() {
        _rasterLayers.forEach(function(rl) {
          var cb = document.getElementById('rl_cb_' + rl.id);
          if (cb) cb.addEventListener('change', function() {
            setLayerVisible(rl.id, cb.checked);
            updateLegend();
          });
        });
        var sld = document.getElementById('satOpSlider');
        var lbl = document.getElementById('satOpVal');
        if (sld) sld.addEventListener('input', function() {
          _opacity = sld.value / 100; lbl.textContent = sld.value + '%';
          Object.keys(_tileRegistry).forEach(function(k) {
            var l = _tileRegistry[k];
            // Only update opacity for layers that are toggled ON (opacity > 0)
            if (l && l.options.opacity > 0) l.setOpacity(_opacity);
          });
        });
      }, 60);
      return p;
    }

    // ── FULLSCREEN (topright) ────────────────────────────────────────────
    var fsCtrl = L.control({position: 'topright'});
    fsCtrl.onAdd = function() {
      var d = L.DomUtil.create('div', 'leaflet-bar leaflet-control'); d.style.marginTop = '4px';
      var a = document.createElement('a'); a.href = '#'; a.title = 'Full Screen';
      a.style.cssText = BTN_STYLE; a.innerHTML = '\\u26f6';
      L.DomEvent.disableClickPropagation(d);
      var isFs = false;
      a.addEventListener('click', function(e) {
        e.preventDefault(); isFs = !isFs;
        var el = mapObj.getContainer();
        if (isFs) { (el.requestFullscreen||el.webkitRequestFullscreen||function(){}).call(el); a.innerHTML = '\\u2715'; }
        else { (document.exitFullscreen||document.webkitExitFullscreen||function(){}).call(document); a.innerHTML = '\\u26f6'; }
      });
      d.appendChild(a); return d;
    };
    fsCtrl.addTo(mapObj);

    // ── RULER (topright) ─────────────────────────────────────────────────
    var measuring = false, mpoints = [], mlines = [], mlabels = [], mtotal = null;
    function hkm(la1,lo1,la2,lo2) {
      var R=6371, dLa=(la2-la1)*Math.PI/180, dLo=(lo2-lo1)*Math.PI/180;
      var a=Math.sin(dLa/2)*Math.sin(dLa/2)+Math.cos(la1*Math.PI/180)*Math.cos(la2*Math.PI/180)*Math.sin(dLo/2)*Math.sin(dLo/2);
      return R*2*Math.atan2(Math.sqrt(a),Math.sqrt(1-a));
    }
    function fmt(km){ return km<1?(km*1000).toFixed(0)+' m':km.toFixed(2)+' km'; }
    function clearMeas(){
      mlines.forEach(function(l){mapObj.removeLayer(l);});
      mlabels.forEach(function(l){mapObj.removeLayer(l);});
      if(mtotal){mapObj.removeLayer(mtotal);mtotal=null;}
      mlines=[]; mlabels=[]; mpoints=[];
    }
    function onMeasClick(e){
      if(!measuring) return;
      var ll=e.latlng; mpoints.push(ll);
      var dot=L.circleMarker(ll,{radius:4,color:'#00c8c8',fillColor:'#00c8c8',fillOpacity:1,weight:2}).addTo(mapObj);
      mlabels.push(dot);
      if(mpoints.length>=2){
        var prev=mpoints[mpoints.length-2];
        var segKm=hkm(prev.lat,prev.lng,ll.lat,ll.lng);
        var line=L.polyline([prev,ll],{color:'#00c8c8',weight:2,dashArray:'6 4',opacity:0.9}).addTo(mapObj);
        mlines.push(line);
        var mid=[(prev.lat+ll.lat)/2,(prev.lng+ll.lng)/2];
        var sl=L.marker(mid,{icon:L.divIcon({html:'<div style="background:rgba(2,13,24,0.88);color:#00c8c8;border:1px solid rgba(0,200,200,0.45);border-radius:3px;padding:1px 5px;font-size:11px;font-family:monospace;white-space:nowrap;">'+fmt(segKm)+'</div>',className:'',iconAnchor:[0,0]})}).addTo(mapObj);
        mlabels.push(sl);
        var tot=0; for(var i=1;i<mpoints.length;i++) tot+=hkm(mpoints[i-1].lat,mpoints[i-1].lng,mpoints[i].lat,mpoints[i].lng);
        if(mtotal) mapObj.removeLayer(mtotal);
        if(mpoints.length>2){
          mtotal=L.marker(ll,{icon:L.divIcon({html:'<div style="background:rgba(2,13,24,0.95);color:#fff;border:1px solid #00c8c8;border-radius:3px;padding:2px 7px;font-size:11px;font-family:monospace;white-space:nowrap;margin-top:12px;">\\u03a3 '+fmt(tot)+'</div>',className:'',iconAnchor:[0,0]})}).addTo(mapObj);
        }
      }
    }
    var rulerCtrl = L.control({position:'topright'});
    rulerCtrl.onAdd = function(){
      var d=L.DomUtil.create('div','leaflet-bar leaflet-control'); d.style.marginTop='4px';
      var a=document.createElement('a'); a.href='#'; a.title='Measure Distance';
      a.style.cssText=BTN_STYLE; a.innerHTML='\\ud83d\\udccf';
      L.DomEvent.disableClickPropagation(d);
      a.addEventListener('click',function(e){
        e.preventDefault(); measuring=!measuring;
        if(measuring){clearMeas();a.style.background='rgba(0,200,200,0.25)';a.style.color='#fff';mapObj.getContainer().style.cursor='crosshair';mapObj.on('click',onMeasClick);}
        else{a.style.background='rgba(2,13,24,0.92)';a.style.color='#00c8c8';mapObj.getContainer().style.cursor='';mapObj.off('click',onMeasClick);clearMeas();}
      });
      d.appendChild(a); return d;
    };
    rulerCtrl.addTo(mapObj);

    // ── WQI LEGEND (bottomleft) ──────────────────────────────────────────
    var legendDiv = null;
    var legendCtrl = L.control({position: 'bottomleft'});
    legendCtrl.onAdd = function() {
      legendDiv = L.DomUtil.create('div', '');
      legendDiv.id = 'wqi-legend';
      L.DomEvent.disableClickPropagation(legendDiv);
      return legendDiv;
    };
    legendCtrl.addTo(mapObj);

    function updateLegend() {
      if (!legendDiv) return;
      // Collect vis params for all currently visible layers
      var visibleVis = [];
      _rasterLayers.forEach(function(rl) {
        if (!rl.vis) return;  // skip True Color (no legend)
        var cb = document.getElementById('rl_cb_' + rl.id);
        var isOn = cb ? cb.checked : rl.visible;
        var tl = _tileRegistry[rl.id];
        if (isOn || (tl && tl.options.opacity > 0)) {
          visibleVis.push({label: rl.label, vis: rl.vis});
        }
      });
      if (visibleVis.length === 0) {
        legendDiv.innerHTML = ''; legendDiv.style.display = 'none'; return;
      }
      legendDiv.style.display = 'block';
      var html = '';
      visibleVis.forEach(function(item) {
        var v = item.vis;
        var gradColors = v.palette.join(',');
        html +=
          '<div style="background:rgba(2,13,24,0.92);border:1px solid rgba(0,200,200,0.4);border-radius:6px;padding:8px 12px;font-family:Arial,sans-serif;min-width:180px;margin-bottom:4px;">' +
            '<div style="color:#00c8c8;font-size:11px;font-weight:bold;margin-bottom:5px;letter-spacing:0.5px;">' + v.unit + '</div>' +
            '<div style="height:14px;border-radius:3px;background:linear-gradient(to right,' + gradColors + ');"></div>' +
            '<div style="display:flex;justify-content:space-between;margin-top:3px;">' +
              '<span style="color:#d6eaf8;font-size:10px;">' + v.min + ' \\u2014 ' + (v.minLabel||'') + '</span>' +
              '<span style="color:#d6eaf8;font-size:10px;">' + (v.maxLabel||'') + ' \\u2014 ' + v.max + '</span>' +
            '</div>' +
          '</div>';
      });
      legendDiv.innerHTML = html;
    }
    // Initial legend check
    updateLegend();

    console.log('[MEDI] All controls added successfully');
  } catch(err) {
    console.error('[MEDI] Control init failed:', err);
  }
})();
"""
        _js_body = (_js_body
                    .replace("__RL_JSON__", _rl_json)
                    .replace("__SEL_DATE__", _sel_date_js)
                    .replace("__ACTIVE_SRC__", _active_src_js)
                    .replace("__BM_JSON__", _bm_json)
                    .replace("__ACTIVE_BM__", _active_bm_js))

        class MEDIControls(MacroElement):
            def __init__(self, body):
                super().__init__()
                self._body = body
                self._template = Template("{% macro script(this, kwargs) %}\n" +
                                          body.replace("__MAP_VAR__", "{{this._parent.get_name()}}") +
                                          "\n{% endmacro %}")

        m.add_child(MEDIControls(_js_body))

        m.add_child(folium.Element('<!-- WQI legend removed -->'))
        if False:  # zone markers hidden for demo
            _vis_grps = st.session_state.get("visible_groups", {""})
            for zname, zdata in st.session_state.get("user_zones", {}).items():
                # Skip zones from hidden groups
                zgrp = zdata.get("group", "")
                if zgrp not in _vis_grps:
                    continue
                ztype  = zdata.get("type", "polygon")
                coords = zdata.get("coords", [])
                # Detect territorial waters zone for special styling
                TW_KEYS = ["territorial","טריטוריאל","ים ישראל","israel water","tw_","terr_"]
                is_tw   = any(kw in zname.lower() for kw in TW_KEYS)
                color   = "#FFD700" if is_tw else "#00c8c8"
                weight  = 3 if is_tw else 2
                dash    = "8 4" if is_tw else None

                if ztype == "point" and zdata.get("lat") is not None:
                    folium.CircleMarker(
                        location=[zdata["lat"], zdata["lon"]],
                        radius=8,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.35,
                        weight=weight,
                        tooltip=folium.Tooltip(zname, sticky=True),
                    ).add_to(m)
                    folium.Marker(
                        location=[zdata["lat"], zdata["lon"]],
                        icon=folium.DivIcon(
                            html=f'<div style="font-size:13px;color:{color};font-weight:bold;'
                                 f'white-space:nowrap;text-shadow:0 0 4px #000,0 0 8px #000;'
                                 f'margin-top:-18px;margin-left:12px;">{zname}</div>',
                            icon_size=(0, 0), icon_anchor=(0, 0)
                        )
                    ).add_to(m)
                elif coords:
                    # coords are [[lon,lat],...] — folium needs [[lat,lon],...]
                    latlons = [[c[1], c[0]] for c in coords]
                    poly_kwargs = dict(
                        locations=latlons,
                        color=color,
                        weight=weight,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.08,
                        tooltip=folium.Tooltip(zname, sticky=True),
                    )
                    if dash:
                        poly_kwargs["dash_array"] = dash
                    folium.Polygon(**poly_kwargs).add_to(m)
                    # Centroid label
                    if latlons:
                        clat = sum(p[0] for p in latlons) / len(latlons)
                        clon = sum(p[1] for p in latlons) / len(latlons)
                        folium.Marker(
                            location=[clat, clon],
                            icon=folium.DivIcon(
                                html=f'<div style="font-size:13px;color:{color};font-weight:bold;'
                                     f'white-space:nowrap;text-shadow:0 0 4px #000,0 0 8px #000;'
                                     f'text-align:center;transform:translateX(-50%);">{zname}</div>',
                                icon_size=(0, 0), icon_anchor=(0, 0)
                            )
                        ).add_to(m)


        # ── H3 WQI snapshot layer ────────────────────────────────────────────
        _snap = st.session_state.get("wqi_snapshot")
        if _snap and _snap.get("hexes"):
            _hex_map = {h["hex_id"]: h for h in _snap["hexes"] if h.get("wqi") is not None}
            if "wqi_hex_geojson" not in st.session_state:
                try:
                    import json as _jh3, os as _os
                    _grid_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "medi_h3_grid_final_913.geojson")
                    with open(_grid_path) as _f:
                        st.session_state["wqi_hex_geojson"] = _jh3.load(_f)
                except Exception:
                    st.session_state["wqi_hex_geojson"] = None
            _hgeo = st.session_state.get("wqi_hex_geojson")
            if _hgeo:
                def _hex_style(feat):
                    hid = feat["properties"].get("hex_id","")
                    h   = _hex_map.get(hid)
                    if not h:
                        return {"fillColor":"#333","color":"#555","weight":0.3,"fillOpacity":0.15}
                    wqi = h["wqi"]
                    if wqi >= 80:   fc = "#4575b4"   # deep blue — very clean
                    elif wqi >= 65: fc = "#74add1"   # light blue — clean
                    elif wqi >= 50: fc = "#abd9e9"   # cyan — moderate
                    elif wqi >= 40: fc = "#fee090"   # yellow — fair
                    elif wqi >= 30: fc = "#f46d43"   # orange — poor
                    else:           fc = "#d73027"   # red — polluted
                    return {"fillColor":fc,"color":"#000","weight":0.3,"fillOpacity":0.55}

                def _hex_highlight(feat):
                    return {"weight":1.5,"color":"#00c8c8","fillOpacity":0.7}

                # Add WQI values to GeoJSON features for tooltip
                import copy as _copy
                _hgeo_rich = _copy.deepcopy(_hgeo)
                for _feat in _hgeo_rich["features"]:
                    _hid = _feat["properties"].get("hex_id","")
                    _hdata = _hex_map.get(_hid)
                    if _hdata:
                        _feat["properties"]["wqi"] = _hdata.get("wqi","N/A")
                        _feat["properties"]["chl"] = _hdata.get("chl","N/A")
                        _feat["properties"]["turb"] = _hdata.get("turb","N/A")
                    else:
                        _feat["properties"]["wqi"] = "N/A"
                        _feat["properties"]["chl"] = "N/A"
                        _feat["properties"]["turb"] = "N/A"

                folium.GeoJson(
                    _hgeo_rich,
                    name="🗺 H3 WQI Grid",
                    style_function=_hex_style,
                    highlight_function=_hex_highlight,
                    tooltip=folium.GeoJsonTooltip(
                        fields=["hex_id","wqi","chl","turb"],
                        aliases=["Hex ID:","WQI:","Chl-a:","Turbidity:"],
                        localize=True
                    ),
                    show=True,
                ).add_to(m)


        # ── WQI Legend ──────────────────────────────────────────────────────
        legend_js = """
        (function() {
            var legend = L.control({position: 'bottomleft'});
            legend.onAdd = function(map) {
                var div = L.DomUtil.create('div');
                div.innerHTML = '<div style="background:rgba(2,13,24,0.88);border:1px solid rgba(0,200,200,0.25);border-radius:6px;padding:10px 13px;font-family:monospace;">'
                    + '<div style="font-size:10px;color:#00c8c8;letter-spacing:0.1em;margin-bottom:8px;">WQI INDEX</div>'
                    + '<div style="display:flex;flex-direction:column;gap:4px;">'
                    + '<div style="display:flex;align-items:center;gap:7px;"><div style="width:14px;height:14px;border-radius:2px;background:#4575b4;"></div><span style="font-size:10px;color:#c8e8f8;">≥ 80 clean</span></div>'
                    + '<div style="display:flex;align-items:center;gap:7px;"><div style="width:14px;height:14px;border-radius:2px;background:#74add1;"></div><span style="font-size:10px;color:#c8e8f8;">≥ 65 good</span></div>'
                    + '<div style="display:flex;align-items:center;gap:7px;"><div style="width:14px;height:14px;border-radius:2px;background:#abd9e9;"></div><span style="font-size:10px;color:#c8e8f8;">≥ 50 moderate</span></div>'
                    + '<div style="display:flex;align-items:center;gap:7px;"><div style="width:14px;height:14px;border-radius:2px;background:#fee090;"></div><span style="font-size:10px;color:#c8e8f8;">≥ 40 fair</span></div>'
                    + '<div style="display:flex;align-items:center;gap:7px;"><div style="width:14px;height:14px;border-radius:2px;background:#f46d43;"></div><span style="font-size:10px;color:#c8e8f8;">≥ 30 poor</span></div>'
                    + '<div style="display:flex;align-items:center;gap:7px;"><div style="width:14px;height:14px;border-radius:2px;background:#d73027;"></div><span style="font-size:10px;color:#c8e8f8;">< 30 polluted</span></div>'
                    + '</div></div>';
                return div;
            };
            legend.addTo(__MAP_VAR__);
        })();
        """
        m.add_child(MEDIControls(legend_js))

        # Native folium.LayerControl removed — replaced by custom 30×30 basemap
        # button injected via JS (topleft, matching the other toolbar buttons).
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
                if "img_idx" not in st.session_state:
                    st.session_state.img_idx = 0

            # Load history for user zones
            history_days  = 30
            history_label = "30 ימים"
            beach_history = {}

            # ── S1 SAR loading (must be BEFORE col split so spinner shows full width) ──
            if st.session_state.get("s1_mode") and not st.session_state.get("s1_result"):
                with st.spinner("🛰 Loading Sentinel-1 SAR data..."):
                    try:
                        from s1_processing import (get_available_s1_dates as _gsd1,
                            get_s1_layers as _gsl1, detect_oil_spills as _dos1,
                            detect_vessels as _dv1, check_vessel_oil_proximity as _cvop1)
                        _s1_target = st.session_state.pop("s1_target_date", None)
                        if not _s1_target:
                            _s1_dates = _gsd1(days_back=7)
                            st.session_state["s1_avail_dates"] = _s1_dates
                            _s1_target = _s1_dates[0]["date"] if _s1_dates else sel_date
                        st.session_state["s1_date"] = _s1_target
                        _s1_layers  = _gsl1(_s1_target)
                        _s1_oil     = _dos1(_s1_target)
                        _s1_vessels = _dv1(_s1_target)
                        _s1_vessels["vessels"] = _cvop1(
                            _s1_vessels.get("vessels", []),
                            _s1_oil.get("polygons", [])
                        )
                        st.session_state["s1_result"] = {
                            "layers":  _s1_layers,
                            "oil":     _s1_oil,
                            "vessels": _s1_vessels,
                            "date":    _s1_target,
                        }
                        st.rerun()
                    except Exception as _s1e:
                        st.warning(f"SAR load failed: {_s1e}")
                        st.session_state["s1_mode"]   = False
                        st.session_state["s1_result"] = None

            col_map, col_info = st.columns([3, 1], gap="small")
            with col_map:
                pass  # buttons hidden for demo

                # ── S1 SAR mode active indicator ──────────────────────────
                if st.session_state.get("s1_mode") and not st.session_state.get("s1_result"):
                    st.info("🛰 Loading SAR data...")
                # Ensure snapshot loaded before building map
                if "wqi_snapshot" not in st.session_state:
                    try:
                        from storage import load_snapshot as _ls
                        st.session_state["wqi_snapshot"] = _ls()
                    except Exception:
                        st.session_state["wqi_snapshot"] = None

                map_data_wqi = st_folium(
                    _build_map(),
                    use_container_width=True, height=740,
                    key=f"israel_map_wqi_{st.session_state.get('img_idx',0)}_{st.session_state.get('s1_date','')}",
                    returned_objects=["bounds","last_active_drawing","last_clicked"]
                )
  
                if st.session_state.spectra_result:
                    _sp = st.session_state.spectra_result
                    _lat_str, _lon_str = st.session_state.spectra_click.split(",")
                    st.markdown(
                        f'<div style="font-size:12px;color:#7fb3d3;margin:4px 0 2px;">'
                        f'🔬 Spectra · {data_source} · {_lat_str}°N {_lon_str}°E</div>',
                        unsafe_allow_html=True)
                    _max_val = max(_sp.values()) if _sp else 1
                    _bar_html = '<div style="display:flex;align-items:flex-end;gap:3px;height:80px;padding:4px;background:rgba(0,200,200,0.04);border:1px solid rgba(0,200,200,0.15);border-radius:5px;">'
                    for _wl, _rv in _sp.items():
                        _pct = int((_rv / _max_val) * 100) if _max_val else 0
                        _wl_num = int(_wl.replace("nm","")) if "nm" in _wl else 500
                        _col = "#8B00FF" if _wl_num<450 else "#0055FF" if _wl_num<500 else "#00AA00" if _wl_num<570 else "#FF4400" if _wl_num<700 else "#880000"
                        _bar_html += f'<div title="{_wl}: {_rv}" style="flex:1;min-width:8px;height:{_pct}%;background:{_col};border-radius:2px 2px 0 0;cursor:help;"></div>'
                    _bar_html += '</div>'
                    _bar_html += '<div style="display:flex;gap:3px;overflow-x:auto;">'
                    for _wl, _rv in _sp.items():
                        _bar_html += f'<div style="flex:1;min-width:8px;text-align:center;font-size:9px;color:#7fb3d3;">{_rv}</div>'
                    _bar_html += '</div>'
                    _bar_html += '<div style="display:flex;gap:3px;overflow-x:auto;margin-bottom:4px;">'
                    for _wl in _sp.keys():
                        _bar_html += f'<div style="flex:1;min-width:8px;text-align:center;font-size:9px;color:#7fb3d3;">{_wl}</div>'
                    _bar_html += '</div>'
                    st.markdown(_bar_html, unsafe_allow_html=True)
                    if st.button("✕ Clear spectra", key="clear_spectra"):
                        st.session_state.spectra_result = None
                        st.session_state.spectra_click = None
                        st.rerun()
            with col_info:
                # ── S1 SAR PANEL ──────────────────────────────────────────
                if st.session_state.get("s1_mode") and st.session_state.get("s1_result"):
                    _s1r  = st.session_state.get("s1_result", {})
                    _oil  = _s1r.get("oil", {})
                    _ves  = _s1r.get("vessels", {})
                    _n_oil  = _oil.get("n_anomalies", 0)
                    _n_ves  = _ves.get("n_vessels", 0)
                    _n_near = sum(1 for v in _ves.get("vessels", []) if v.get("near_oil"))
                    _s1_date_str = _s1r.get("date", sel_date)

                    st.markdown(f'<div style="font-size:12px;color:#00c8c8;font-family:monospace;margin-bottom:4px;">🛰 Sentinel-1 SAR · {_s1_date_str}</div>', unsafe_allow_html=True)
                    # Date navigator
                    _s1_avail = st.session_state.get("s1_avail_dates", [])
                    if not _s1_avail:
                        try:
                            from s1_processing import get_available_s1_dates as _gsd
                            _s1_avail = _gsd(days_back=14)
                            st.session_state["s1_avail_dates"] = _s1_avail
                        except Exception:
                            _s1_avail = []
                    if _s1_avail and len(_s1_avail) > 1:
                        _s1_date_opts = [d["date"] for d in _s1_avail]
                        _s1_cur_idx = _s1_date_opts.index(_s1_date_str) if _s1_date_str in _s1_date_opts else 0
                        _sn1, _sn2, _sn3 = st.columns([1, 4, 1])
                        with _sn1:
                            if st.button("◀", key="s1_prev"):
                                if _s1_cur_idx < len(_s1_date_opts)-1:
                                    st.session_state["s1_target_date"] = _s1_date_opts[_s1_cur_idx + 1]
                                    st.session_state["s1_result"] = None
                                    st.rerun()
                        with _sn2:
                            st.markdown(f'<div style="text-align:center;font-size:12px;color:#7fb3d3;padding:4px 0;">{_s1_date_str} ({_s1_cur_idx+1}/{len(_s1_date_opts)})</div>', unsafe_allow_html=True)
                        with _sn3:
                            if st.button("▶", key="s1_next"):
                                if _s1_cur_idx > 0:
                                    st.session_state["s1_target_date"] = _s1_date_opts[_s1_cur_idx - 1]
                                    st.session_state["s1_result"] = None
                                    st.rerun()
                    _s1c1, _s1c2, _s1c3 = st.columns(3)
                    with _s1c1:
                        st.markdown(f'<div style="background:rgba(55,138,221,0.08);border:1px solid rgba(55,138,221,0.2);border-radius:6px;padding:8px;text-align:center;"><div style="font-size:11px;color:#7fb3d3;">Vessels</div><div style="font-size:22px;font-weight:600;color:#c8e8f8;">{_n_ves}</div></div>', unsafe_allow_html=True)
                    with _s1c2:
                        st.markdown(f'<div style="background:rgba(226,75,74,0.08);border:1px solid rgba(226,75,74,0.2);border-radius:6px;padding:8px;text-align:center;"><div style="font-size:11px;color:#7fb3d3;">Oil anomalies</div><div style="font-size:22px;font-weight:600;color:#f09595;">{_n_oil}</div></div>', unsafe_allow_html=True)
                    with _s1c3:
                        _wc = "#FAC775" if _n_near > 0 else "#7fb3d3"
                        st.markdown(f'<div style="background:rgba(239,159,39,0.08);border:1px solid rgba(239,159,39,0.2);border-radius:6px;padding:8px;text-align:center;"><div style="font-size:11px;color:#7fb3d3;">Near oil</div><div style="font-size:22px;font-weight:600;color:{_wc};">{_n_near}</div></div>', unsafe_allow_html=True)

                    if _ves.get("vessels"):
                        st.markdown('<div style="font-size:12px;color:#00c8c8;margin:8px 0 4px;font-family:monospace;">📍 Detected vessels</div>', unsafe_allow_html=True)
                        for _v in _ves["vessels"]:
                            _vc = "rgba(239,159,39,0.15)" if _v.get("near_oil") else "rgba(4,30,51,0.6)"
                            _vb = "rgba(239,159,39,0.4)" if _v.get("near_oil") else "rgba(0,200,200,0.15)"
                            _alert = f'⚠ near {_v["near_oil_id"]}' if _v.get("near_oil") else ""
                            st.markdown(
                                f'<div style="background:{_vc};border:1px solid {_vb};border-radius:5px;padding:6px 10px;margin-bottom:4px;">'
                                f'<span style="font-size:12px;color:#c8e8f8;font-weight:500;">{_v["id"]}</span>'
                                f'<span style="font-size:11px;color:#EF9F27;margin-left:8px;">{_alert}</span><br>'
                                f'<span style="font-size:11px;color:#7fb3d3;">{_v["lat"]}°N {_v["lon"]}°E &nbsp;·&nbsp; {_v["category"]} &nbsp;·&nbsp; ~{_v["length_min_m"]}–{_v["length_max_m"]}m × {_v["width_min_m"]}–{_v["width_max_m"]}m &nbsp;·&nbsp; {_v["confidence"]}</span>'
                                f'</div>', unsafe_allow_html=True)

                    if _oil.get("polygons"):
                        st.markdown('<div style="font-size:12px;color:#00c8c8;margin:8px 0 4px;font-family:monospace;">⚠ Oil anomalies</div>', unsafe_allow_html=True)
                        for _o in _oil["polygons"]:
                            _cc = {"High":"#f09595","Medium":"#FAC775","Low":"#B4B2A9"}.get(_o["confidence"],"#7fb3d3")
                            _nv = [v["id"] for v in _ves.get("vessels",[]) if v.get("near_oil_id")==_o["id"]]
                            _ns = f'⚠ {", ".join(_nv)} nearby' if _nv else ""
                            st.markdown(
                                f'<div style="background:rgba(226,75,74,0.08);border:1px solid rgba(226,75,74,0.2);border-radius:5px;padding:6px 10px;margin-bottom:4px;">'
                                f'<span style="font-size:12px;color:{_cc};font-weight:500;">{_o["id"]}</span>'
                                f'<span style="font-size:11px;color:#EF9F27;margin-left:8px;">{_ns}</span><br>'
                                f'<span style="font-size:11px;color:#7fb3d3;">{_o["lat"]}°N {_o["lon"]}°E &nbsp;·&nbsp; {_o["area_km2_min"]}–{_o["area_km2_max"]} km² &nbsp;·&nbsp; {_o["confidence"]}</span>'
                                f'</div>', unsafe_allow_html=True)

                    if not _ves.get("vessels") and not _oil.get("polygons"):
                        st.info("No vessels or oil anomalies detected for this date.")

                    st.markdown('<div style="font-size:10px;color:#7fb3d3;padding:5px 0;border-top:1px solid rgba(0,200,200,0.1);margin-top:6px;">⚠ SAR detection only. Oil requires optical validation. Vessel size ±40%.</div>', unsafe_allow_html=True)
                    if st.button("🗑 Clear SAR", key="clear_s1"):
                        st.session_state["s1_mode"]   = False
                        st.session_state["s1_result"] = None
                        st.rerun()
                    st.markdown("---")

                # ── WQI Dashboard (gauge + anomalies + history trend) ──────────
                if not st.session_state.get("s1_mode"):
                    _snap_h = st.session_state.get("wqi_snapshot")
                    if _snap_h and _snap_h.get("hexes"):
                        import numpy as _np
                        _wqi_vals = [h["wqi"] for h in _snap_h["hexes"] if h.get("wqi") is not None]
                        if _wqi_vals:
                            _mean   = round(float(_np.mean(_wqi_vals)), 1)
                            _med    = round(float(_np.median(_wqi_vals)), 1)
                            _n_clean  = sum(1 for v in _wqi_vals if v >= 65)
                            _n_mod    = sum(1 for v in _wqi_vals if 40 <= v < 65)
                            _n_poor   = sum(1 for v in _wqi_vals if v < 40)

                            # Gauge color by mean WQI
                            _gc = "#4575b4" if _mean >= 80 else "#74add1" if _mean >= 65 else "#abd9e9" if _mean >= 50 else "#fee090" if _mean >= 40 else "#f46d43" if _mean >= 30 else "#d73027"

                            # History trend (last 10 days)
                            _hist_trend = ""
                            try:
                                from storage import load_history
                                _history = load_history()
                                if _history:
                                    _last10 = _history[-10:]
                                    _h_dates = [h["date"][-5:] for h in _last10]
                                    _h_means = [h["mean_wqi"] for h in _last10]
                                    _h_max = max(_h_means) if _h_means else 100
                                    _h_min = min(_h_means) if _h_means else 0
                                    _h_range = max(_h_max - _h_min, 10)
                                    # Sparkline SVG
                                    _sw, _sh = 280, 60
                                    _spark = f'<svg width="100%" viewBox="0 0 {_sw} {_sh}" xmlns="http://www.w3.org/2000/svg" style="display:block;">'
                                    _pts = []
                                    for _hi, _hv in enumerate(_h_means):
                                        _hx = 10 + _hi * (_sw - 20) / max(len(_h_means) - 1, 1)
                                        _hy = _sh - 20 - int((_hv - _h_min) / _h_range * (_sh - 28))
                                        _pts.append((_hx, _hy))
                                    if len(_pts) > 1:
                                        _pstr = " ".join(f"{x:.0f},{y:.0f}" for x,y in _pts)
                                        _spark += f'<polyline points="{_pstr}" fill="none" stroke="#00c8c8" stroke-width="1.5" stroke-linejoin="round"/>'
                                    for _hx, _hy in _pts:
                                        _spark += f'<circle cx="{_hx:.0f}" cy="{_hy:.0f}" r="3" fill="#00c8c8"/>'
                                    # Labels
                                    if _pts:
                                        _spark += f'<text x="{_pts[0][0]:.0f}" y="{_sh-4}" text-anchor="middle" font-size="8" fill="#4a7fa5">{_h_dates[0]}</text>'
                                        _spark += f'<text x="{_pts[-1][0]:.0f}" y="{_sh-4}" text-anchor="middle" font-size="8" fill="#4a7fa5">{_h_dates[-1]}</text>'
                                    _spark += '</svg>'
                                    # Delta vs 10d ago
                                    _delta = round(_mean - _h_means[0], 1) if len(_h_means) > 1 else 0
                                    _delta_col = "#74add1" if _delta >= 0 else "#f46d43"
                                    _delta_sign = "+" if _delta >= 0 else ""
                                    _hist_trend = f"""
                                        <div style="margin-top:10px;border-top:1px solid rgba(0,200,200,0.12);padding-top:8px;">
                                        <div style="font-size:10px;color:#00c8c8;font-family:monospace;letter-spacing:0.08em;margin-bottom:4px;">10-DAY TREND</div>
                                        {_spark}
                                        <div style="font-size:10px;color:{_delta_col};margin-top:2px;">vs 10d ago: {_delta_sign}{_delta}</div>
                                        </div>"""
                            except Exception:
                                pass

                            # Top 3 worst hex with coordinates
                            _sorted_hex = sorted(
                                [h for h in _snap_h["hexes"] if h.get("wqi") is not None],
                                key=lambda x: x["wqi"]
                            )[:3]

                            # Build anomaly rows with coordinates from geojson
                            _anom_html = ""
                            try:
                                import json as _jj
                                if "wqi_hex_geojson" in st.session_state:
                                    _geo = st.session_state["wqi_hex_geojson"]
                                    _coord_map = {f["properties"]["hex_id"]: (f["properties"].get("lat",0), f["properties"].get("lng",0)) for f in _geo["features"]}
                                    for _ah in _sorted_hex:
                                        _awqi = _ah["wqi"]
                                        _acid = _ah["hex_id"]
                                        _alat, _alng = _coord_map.get(_acid, (0, 0))
                                        _ac = "#d73027" if _awqi < 30 else "#f46d43" if _awqi < 40 else "#fee090"
                                        _abc = "rgba(215,48,39,0.12)" if _awqi < 30 else ("rgba(244,109,67,0.10)" if _awqi < 40 else "rgba(254,224,144,0.10)")
                                        _anom_html += f'<div style="font-size:11px;color:#c8e8f8;background:{_abc};border:0.5px solid {_ac};border-radius:5px;padding:5px 8px;margin-bottom:4px;">WQI {_awqi:.0f} &nbsp;·&nbsp; <span style="color:#7fb3d3;">{_alat:.2f}°N {_alng:.2f}°E</span></div>'
                            except Exception:
                                for _ah in _sorted_hex:
                                    _awqi = _ah["wqi"]
                                    _ac = "#d73027" if _awqi < 30 else "#f46d43"
                                    _anom_html += f'<div style="font-size:11px;color:#c8e8f8;border:0.5px solid {_ac};border-radius:5px;padding:5px 8px;margin-bottom:4px;">WQI {_awqi:.0f}</div>'

                            # Gauge SVG
                            _gw, _gh = 160, 100
                            _angle = (_mean / 100) * 180
                            import math as _math
                            _rad = _math.radians(180 - _angle)
                            _nx = 80 + 55 * _math.cos(_rad)
                            _ny = 85 - 55 * _math.sin(_rad)
                            _gauge_svg = f"""<svg width="100%" viewBox="0 0 {_gw} {_gh}" xmlns="http://www.w3.org/2000/svg" style="display:block;">
                              <path d="M 15 85 A 65 65 0 0 1 145 85" fill="none" stroke="rgba(0,200,200,0.12)" stroke-width="12" stroke-linecap="round"/>
                              <path d="M 15 85 A 65 65 0 0 1 {_nx:.1f} {_ny:.1f}" fill="none" stroke="{_gc}" stroke-width="12" stroke-linecap="round"/>
                              <text x="80" y="72" text-anchor="middle" font-size="22" font-weight="bold" fill="#d6eaf8">{_mean:.0f}</text>
                              <text x="80" y="84" text-anchor="middle" font-size="8" fill="#7fb3d3">mean WQI</text>
                              <text x="18" y="98" text-anchor="middle" font-size="8" fill="#7fb3d3">0</text>
                              <text x="142" y="98" text-anchor="middle" font-size="8" fill="#7fb3d3">100</text>
                            </svg>"""

                            # Metric cards
                            _cards_html = f"""
                            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin:8px 0;">
                              <div style="background:rgba(69,117,180,0.12);border:0.5px solid rgba(69,117,180,0.3);border-radius:6px;padding:6px;text-align:center;">
                                <div style="font-size:10px;color:#7fb3d3;">clean</div>
                                <div style="font-size:18px;font-weight:500;color:#74add1;">{_n_clean}</div>
                              </div>
                              <div style="background:rgba(254,224,144,0.08);border:0.5px solid rgba(254,224,144,0.25);border-radius:6px;padding:6px;text-align:center;">
                                <div style="font-size:10px;color:#7fb3d3;">moderate</div>
                                <div style="font-size:18px;font-weight:500;color:#fee090;">{_n_mod}</div>
                              </div>
                              <div style="background:rgba(215,48,39,0.08);border:0.5px solid rgba(215,48,39,0.25);border-radius:6px;padding:6px;text-align:center;">
                                <div style="font-size:10px;color:#7fb3d3;">poor</div>
                                <div style="font-size:18px;font-weight:500;color:#d73027;">{_n_poor}</div>
                              </div>
                            </div>"""

                            st.markdown(
                                f'<div style="font-size:10px;color:#00c8c8;font-family:monospace;letter-spacing:0.08em;margin-bottom:4px;margin-top:4px;">WQI OVERVIEW · {len(_wqi_vals)} hex</div>',
                                unsafe_allow_html=True
                            )
                            st.markdown(_gauge_svg, unsafe_allow_html=True)
                            st.markdown(_cards_html, unsafe_allow_html=True)

                            # ── Today vs 10d average ───────────────────────
                            try:
                                from storage import load_history
                                _history = load_history()
                                if len(_history) >= 2:
                                    _h_means = [h["mean_wqi"] for h in _history[-10:]]
                                    _avg10 = round(sum(_h_means) / len(_h_means), 1)
                                    _delta = round(_mean - _avg10, 1)
                                    _dsign = "+" if _delta >= 0 else ""
                                    _dcol = "#74add1" if _delta >= 0 else "#f46d43"
                                    _trend_html = f"""
                                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;border-top:1px solid rgba(0,200,200,0.12);padding-top:10px;">
                                      <div style="background:rgba(0,20,40,0.3);border-radius:6px;padding:10px;text-align:center;">
                                        <div style="font-size:10px;color:#7fb3d3;margin-bottom:4px;">today</div>
                                        <div style="font-size:24px;font-weight:500;color:#d6eaf8;">{_mean:.0f}</div>
                                        <div style="font-size:9px;color:#4a7fa5;">mean WQI</div>
                                      </div>
                                      <div style="background:rgba(0,20,40,0.3);border-radius:6px;padding:10px;text-align:center;">
                                        <div style="font-size:10px;color:#7fb3d3;margin-bottom:4px;">10d average</div>
                                        <div style="font-size:24px;font-weight:500;color:#d6eaf8;">{_avg10:.0f}</div>
                                        <div style="font-size:9px;color:{_dcol};">{_dsign}{_delta} vs today</div>
                                      </div>
                                    </div>"""
                                    st.markdown(_trend_html, unsafe_allow_html=True)
                            except Exception:
                                pass

                # Detect drawings → unified pending_zone (point or polygon)
                last_clicked = map_data_wqi.get("last_clicked") if map_data_wqi else None
                last_drawing  = map_data_wqi.get("last_active_drawing") if map_data_wqi else None

                # Track which drawings have already been processed (by coords hash)
                if "saved_drawing_hashes" not in st.session_state:
                    st.session_state.saved_drawing_hashes = set()

                if last_drawing:
                    geom  = last_drawing.get("geometry", {})
                    gtype = geom.get("type", "")
                    if gtype in ["Polygon", "Rectangle"]:
                        coords = geom["coordinates"][0]
                        draw_hash = str(coords)
                        pending   = st.session_state.get("pending_zone")
                        # Only trigger if this drawing hasn't been saved yet AND isn't already pending
                        already_pending = pending and pending.get("coords") == coords
                        already_saved   = draw_hash in st.session_state.saved_drawing_hashes
                        if not already_pending and not already_saved:
                            st.session_state["pending_zone"] = {"type": "polygon", "coords": coords, "hash": draw_hash}
                            st.rerun()
                    elif gtype == "Point":
                        raw = geom.get("coordinates", [])
                        if raw:
                            lon, lat  = round(raw[0], 5), round(raw[1], 5)
                            draw_hash = f"{lat},{lon}"
                            pending   = st.session_state.get("pending_zone")
                            already_pending = pending and pending.get("lat") == lat and pending.get("lon") == lon
                            already_saved   = draw_hash in st.session_state.saved_drawing_hashes
                            if not already_pending and not already_saved:
                                st.session_state["pending_zone"] = {"type": "point", "lat": lat, "lon": lon, "hash": draw_hash}
                                st.rerun()
                elif last_clicked and last_clicked.get("lat"):
                    clat = round(last_clicked["lat"], 5)
                    clon = round(last_clicked["lng"], 5)
                    draw_hash = f"{clat},{clon}"
                    already_saved = draw_hash in st.session_state.saved_drawing_hashes
                    if st.session_state.inspect_mode and not already_saved:
                        _sc_key = f"{round(last_clicked['lat'],4)},{round(last_clicked['lng'],4)}"
                        if st.session_state.spectra_click != _sc_key:
                            st.session_state.spectra_click = _sc_key
                            with st.spinner("🔬 Sampling..."):
                                st.session_state.spectra_result = sample_pixel_spectra(
                                    last_clicked["lat"], last_clicked["lng"], sel_src, sel_date)
                            st.rerun()
                    elif not st.session_state.inspect_mode:
                        pending = st.session_state.get("pending_zone")
                        already_pending = pending and pending.get("lat") == clat and pending.get("lon") == clon
                        if not already_pending and not already_saved:
                            st.session_state["pending_zone"] = {"type": "point", "lat": clat, "lon": clon, "hash": draw_hash}
                            st.rerun()

                if False:  # Dashboard hidden — demo mode
                    pass
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

    mg.add_child(folium.Element('<!-- WQI legend removed -->'))
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
