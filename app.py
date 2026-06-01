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
@st.cache_resource
def init_gee():
    creds=dict(st.secrets["gee_credentials"])
    with tempfile.NamedTemporaryFile(mode="w",suffix=".json",delete=False) as f:
        f.write(json.dumps(creds)); tmp=f.name
    ee.Initialize(ee.ServiceAccountCredentials(creds["client_email"],tmp))
    os.unlink(tmp)
init_gee()

# =============================================================================
# Persistent Zone Storage — Google Drive (primary) + /tmp fallback
# =============================================================================
ZONES_KEY       = "medi-zones-v1"
GDRIVE_FILENAME = "medi_zones.json"
GDRIVE_FOLDER   = "1VU11P0UCzeMiVsn0k1RiIHuEu8bBLUFH"
GDRIVE_FILE_ID  = "1KTI_oRHIrvRJNtfZYrWMkhNi2D-34Y9v"  # hardcoded for fast startup load

@st.cache_resource
def _gdrive_token():
    """Get OAuth2 access token for service account using only stdlib."""
    try:
        import time, json as _j, base64, urllib.request, urllib.parse
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding as _pad
        creds    = dict(st.secrets["gee_credentials"])
        sa_email = creds["client_email"]
        priv_key = creds["private_key"]
        now = int(time.time())
        def b64(d): return base64.urlsafe_b64encode(d).rstrip(b"=")
        hdr = b64(_j.dumps({"alg":"RS256","typ":"JWT"}).encode())
        pay = b64(_j.dumps({"iss":sa_email,
            "scope":"https://www.googleapis.com/auth/drive",
            "aud":"https://oauth2.googleapis.com/token",
            "iat":now,"exp":now+3600}).encode())
        msg = hdr + b"." + pay
        key = serialization.load_pem_private_key(priv_key.encode(), password=None)
        sig = b64(key.sign(msg, _pad.PKCS1v15(), hashes.SHA256()))
        jwt = (msg + b"." + sig).decode()
        body = urllib.parse.urlencode({
            "grant_type":"urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion":jwt}).encode()
        req = urllib.request.Request("https://oauth2.googleapis.com/token", data=body,
            headers={"Content-Type":"application/x-www-form-urlencoded"})
        return _j.loads(urllib.request.urlopen(req, timeout=10).read())["access_token"]
    except:
        return None

def _gdrive_file_id(token) -> str | None:
    import urllib.request, urllib.parse, json as _j
    q   = f"name='{GDRIVE_FILENAME}' and '{GDRIVE_FOLDER}' in parents and trashed=false"
    url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(
        {"q":q,"fields":"files(id)","pageSize":"1"})
    req = urllib.request.Request(url, headers={"Authorization":f"Bearer {token}"})
    try:
        res = _j.loads(urllib.request.urlopen(req, timeout=10).read())
        return res["files"][0]["id"] if res.get("files") else None
    except:
        return None

def load_zones() -> dict:
    import json as _j, urllib.request, urllib.parse
    try:
        token = _gdrive_token()
        if not token:
            return {}
        q   = f"name='{GDRIVE_FILENAME}' and '{GDRIVE_FOLDER}' in parents and trashed=false"
        url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(
            {"q": q, "fields": "files(id,name)", "pageSize": "5",
             "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"})
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        res = _j.loads(urllib.request.urlopen(req, timeout=10).read())
        files = res.get("files", [])
        if files:
            fid = files[0]["id"]
            req2 = urllib.request.Request(
                f"https://www.googleapis.com/drive/v3/files/{fid}?alt=media&supportsAllDrives=true",
                headers={"Authorization": f"Bearer {token}"})
            raw = urllib.request.urlopen(req2, timeout=10).read().decode()
            return _j.loads(raw)
    except:
        pass
    # fallbacks
    try:
        raw = st.secrets.get("saved_zones", None)
        if raw: return _j.loads(raw)
    except:
        pass
    try:
        return _j.loads(open("/tmp/medi_zones.json").read())
    except:
        return {}

def save_zones(zones: dict):
    import json as _j, urllib.request, urllib.error
    data = _j.dumps(zones, ensure_ascii=False).encode()
    # Always save to /tmp
    try:
        with open("/tmp/medi_zones.json","w") as f: f.write(data.decode())
    except:
        pass
    # Save to Google Drive — SA can only UPDATE files you own, not create new ones
    try:
        token = _gdrive_token()
        if not token:
            return
        import urllib.parse
        q   = f"name='{GDRIVE_FILENAME}' and '{GDRIVE_FOLDER}' in parents and trashed=false"
        url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(
            {"q": q, "fields": "files(id)", "pageSize": "5",
             "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"})
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        res = _j.loads(urllib.request.urlopen(req, timeout=10).read())
        files = res.get("files", [])
        if not files:
            return
        fid = files[0]["id"]
        url2 = f"https://www.googleapis.com/upload/drive/v3/files/{fid}?uploadType=media&supportsAllDrives=true"
        req2 = urllib.request.Request(url2, data=data, method="PATCH", headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json"})
        urllib.request.urlopen(req2, timeout=15)
    except:
        pass

def load_zones_from_all() -> dict: return load_zones()
def load_points() -> dict: return {}
def save_points(points: dict): pass






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
    """Compute WQI history for N days. S3+S2+MODIS for all ranges."""
    end   = datetime.utcnow()
    start = end - timedelta(days=days_back+1)
    wide  = ee.Geometry.Rectangle([34.0,29.0,36.0,33.5])
    wm    = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    fmt   = "%Y-%m-%d"

    s3_coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
               .filterBounds(wide)
               .filterDate(start.strftime(fmt), end.strftime(fmt))
               .sort("system:time_start",False))
    s3_ts  = s3_coll.aggregate_array("system:time_start").getInfo()
    s3_set = set(datetime.utcfromtimestamp(ts/1000).strftime(fmt) for ts in s3_ts)

    s2_coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(wide)
               .filterDate(start.strftime(fmt), end.strftime(fmt))
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
               .sort("system:time_start",False))
    s2_ts  = s2_coll.aggregate_array("system:time_start").getInfo()
    s2_set = set(datetime.utcfromtimestamp(ts/1000).strftime(fmt) for ts in s2_ts)

    # Every calendar day in window, best sensor priority S3>S2>MODIS
    all_days = [(end-timedelta(days=i)).strftime(fmt) for i in range(days_back+1)]
    seen=set(); date_ts=[]
    for d in all_days:
        if d not in seen:
            seen.add(d)
            if d in s3_set:   date_ts.append((d,"S3"))
            elif d in s2_set: date_ts.append((d,"S2"))
            else:             date_ts.append((d,"MODIS"))

    if not date_ts: return {}

    def _wqi_for_date(args):
        date_str,source=args
        try:
            t=ee.Date(date_str)
            if source=="S3":
                coll=(ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(wide)
                      .filterDate(t,t.advance(1,"day")))
                if coll.size().getInfo()==0:
                    coll=(ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(wide)
                          .filterDate(t.advance(-1,"day"),t.advance(2,"day")))
                    if coll.size().getInfo()==0: return date_str,source,None
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
                return date_str,source,wqi
            elif source=="S2":
                coll=(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(wide)
                      .filterDate(t,t.advance(1,"day"))
                      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",40))
                      .sort("system:time_start",False))
                if coll.size().getInfo()==0: return date_str,source,None
                img=coll.first().updateMask(wm)
                b3,b4,b5,b8,b8a=(img.select("B3").divide(10000),img.select("B4").divide(10000),
                                  img.select("B5").divide(10000),img.select("B8").divide(10000),
                                  img.select("B8A").divide(10000))
                wqi=(b3.subtract(b8).divide(b3.add(b8)).unitScale(-0.3,0.5).clamp(0,1)
                     .add(b5.divide(b4.add(1e-6)).unitScale(1.0,3.5).clamp(0,1))
                     .add(ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0,0.15)).clamp(0,1))
                     .divide(3).multiply(100).rename("WQI").clip(ISRAEL_CLIP).updateMask(wm))
                return date_str,source,wqi
            else:
                t2=ee.ImageCollection("MODIS/061/MOD09GA").filterBounds(wide).filterDate(t,t.advance(1,"day"))
                a2=ee.ImageCollection("MODIS/061/MYD09GA").filterBounds(wide).filterDate(t,t.advance(1,"day"))
                qa=t2.merge(a2).sort("system:time_start",False)
                if qa.size().getInfo()==0: return date_str,source,None
                im=qa.first(); cl=im.select("state_1km").bitwiseAnd(0b11).eq(0)
                im=im.updateMask(cl).updateMask(wm)
                b1,b2,b4=im.select("sur_refl_b01"),im.select("sur_refl_b02"),im.select("sur_refl_b04")
                wqi=(b4.subtract(b2).divide(b4.add(b2)).unitScale(-0.3,0.3).clamp(0,1)
                     .add(b4.divide(b1.add(1e-6)).unitScale(0.8,2.5).clamp(0,1))
                     .add(ee.Image(1).subtract(b1.unitScale(0,1500)).clamp(0,1))
                     .divide(3).multiply(100).rename("WQI").clip(ISRAEL_CLIP).updateMask(wm))
                return date_str,source,wqi
        except: return date_str,source,None

    def _sample(args):
        beach,date_str,source,wqi=args
        if wqi is None: return beach["name"],date_str,source,None
        try:
            v=wqi.reduceRegion(reducer=ee.Reducer.mean(),
              geometry=ee.Geometry.Point([beach["lon"],beach["lat"]]).buffer(450),
              scale=300,bestEffort=True).getInfo()
            wv=v.get("WQI")
            return beach["name"],date_str,source,round(wv,1) if wv else None
        except: return beach["name"],date_str,source,None

    with ThreadPoolExecutor(max_workers=4) as ex:
        wqi_images=list(ex.map(_wqi_for_date,date_ts))  # list of (date,source,img)
    img_map={(d,s):img for d,s,img in wqi_images}
    tasks=[(b,d,s,img_map.get((d,s))) for b in BEACHES for d,s in date_ts]
    with ThreadPoolExecutor(max_workers=6) as ex:
        results=list(ex.map(_sample,tasks))

    history={b["name"]:[] for b in BEACHES}
    for bn,ds,src,wv in results:
        if wv is not None:
            history[bn].append({"date":ds,"wqi":wv,"source":src})
    for n in history:
        history[n]=sorted(history[n],key=lambda x:x["date"])
    return history







@st.cache_data(ttl=3600)
def compute_zone_history_range(zones_json: str, days_back: int):
    """
    Compute WQI history for user-defined polygon zones over N days.
    Uses S3 + S2 + MODIS. Stores source (S3/S2/MODIS) per data point for tooltip.
    zones_json: JSON string of {name: {coords: [...]}} to make it cache-friendly.
    Returns dict: {zone_name: [{date, wqi, source}, ...]}
    """
    import json as _j
    zones = _j.loads(zones_json)
    if not zones:
        return {}

    end   = datetime.utcnow()
    start = end - timedelta(days=days_back+1)
    wide  = ee.Geometry.Rectangle([34.0, 29.0, 36.0, 33.5])
    wm    = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(30)
    fmt   = "%Y-%m-%d"

    # ── Discover available dates per sensor ──────────────────────────────────
    s3_coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
               .filterBounds(wide)
               .filterDate(start.strftime(fmt), end.strftime(fmt))
               .sort("system:time_start", False))
    s3_ts  = s3_coll.aggregate_array("system:time_start").getInfo()
    # Keep individual timestamps so we can build one image per exact day
    s3_dates = {}  # date_str -> list of timestamps
    for ts in s3_ts:
        d = datetime.utcfromtimestamp(ts/1000).strftime(fmt)
        s3_dates.setdefault(d, []).append(ts)

    s2_coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(wide)
               .filterDate(start.strftime(fmt), end.strftime(fmt))
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
               .sort("system:time_start", False))
    s2_ts  = s2_coll.aggregate_array("system:time_start").getInfo()
    s2_dates = {}
    for ts in s2_ts:
        d = datetime.utcfromtimestamp(ts/1000).strftime(fmt)
        s2_dates.setdefault(d, []).append(ts)

    # Every calendar day in window: S3 > S2 > MODIS priority
    all_days = [(end - timedelta(days=i)).strftime(fmt) for i in range(days_back + 1)]
    seen = set(); date_ts = []
    for d in all_days:
        if d not in seen:
            seen.add(d)
            if d in s3_dates:
                date_ts.append((d, "S3"))
            elif d in s2_dates:
                date_ts.append((d, "S2"))
            else:
                date_ts.append((d, "MODIS"))

    if not date_ts:
        return {name: [] for name in zones}

    def _wqi_for_date(args):
        date_str, source = args
        try:
            t = ee.Date(date_str)
            if source == "S3":
                # Filter to the exact calendar day only (not ±2) to avoid merging adjacent days
                day_start = t
                day_end   = t.advance(1, "day")
                coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                        .filterBounds(wide)
                        .filterDate(day_start, day_end))
                if coll.size().getInfo() == 0:
                    # Fallback: try ±1 day in case of UTC boundary shift
                    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
                            .filterBounds(wide)
                            .filterDate(t.advance(-1,"day"), t.advance(2,"day")))
                    if coll.size().getInfo() == 0:
                        return date_str, source, None
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
                    kernel=ee.Kernel.square(radius=1, units="pixels")
                ).rename("WQI").updateMask(wm)
                return date_str, source, wqi
            elif source == "S2":
                coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(wide)
                        .filterDate(t, t.advance(1,"day"))
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
                        .sort("system:time_start", False))
                if coll.size().getInfo() == 0:
                    return date_str, source, None
                img   = coll.first().updateMask(wm)
                b3,b4,b5,b8,b8a = (img.select("B3").divide(10000), img.select("B4").divide(10000),
                                    img.select("B5").divide(10000), img.select("B8").divide(10000),
                                    img.select("B8A").divide(10000))
                wqi = (b3.subtract(b8).divide(b3.add(b8)).unitScale(-0.3,0.5).clamp(0,1)
                       .add(b5.divide(b4.add(1e-6)).unitScale(1.0,3.5).clamp(0,1))
                       .add(ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0,0.15)).clamp(0,1))
                       .divide(3).multiply(100).rename("WQI").clip(ISRAEL_CLIP).updateMask(wm))
                return date_str, source, wqi
            else:  # MODIS
                t2 = ee.ImageCollection("MODIS/061/MOD09GA").filterBounds(wide).filterDate(t, t.advance(1,"day"))
                a2 = ee.ImageCollection("MODIS/061/MYD09GA").filterBounds(wide).filterDate(t, t.advance(1,"day"))
                qa = t2.merge(a2).sort("system:time_start", False)
                if qa.size().getInfo() == 0:
                    return date_str, source, None
                im = qa.first(); cl = im.select("state_1km").bitwiseAnd(0b11).eq(0)
                im = im.updateMask(cl).updateMask(wm)
                b1,b2,b4 = im.select("sur_refl_b01"),im.select("sur_refl_b02"),im.select("sur_refl_b04")
                wqi = (b4.subtract(b2).divide(b4.add(b2)).unitScale(-0.3,0.3).clamp(0,1)
                       .add(b4.divide(b1.add(1e-6)).unitScale(0.8,2.5).clamp(0,1))
                       .add(ee.Image(1).subtract(b1.unitScale(0,1500)).clamp(0,1))
                       .divide(3).multiply(100).rename("WQI").clip(ISRAEL_CLIP).updateMask(wm))
                return date_str, source, wqi
        except:
            return date_str, source, None

    with ThreadPoolExecutor(max_workers=4) as ex:
        raw_results = list(ex.map(_wqi_for_date, date_ts))

    # Sample each zone polygon on each date's WQI image
    history = {name: [] for name in zones}
    for date_str, source, wqi_img in raw_results:
        if wqi_img is None:
            continue
        for zname, zdata in zones.items():
            try:
                poly = ee.Geometry.Polygon([zdata["coords"]])
                val  = wqi_img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=poly, scale=300, bestEffort=True
                ).getInfo()
                wv = val.get("WQI")
                if wv is not None:
                    history[zname].append({
                        "date": date_str,
                        "wqi": round(float(wv), 1),
                        "source": source   # ← stored for tooltip
                    })
            except:
                pass

    for name in history:
        history[name] = sorted(history[name], key=lambda x: x["date"])

    return history


# Session state initialization
if "user_zones" not in st.session_state:
    st.session_state.user_zones = load_zones_from_all()
if "monitor_points" not in st.session_state:
    st.session_state.monitor_points = load_points()
if "pending_point" not in st.session_state:
    st.session_state.pending_point = None

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
        st.session_state.history_range = "30 ימים"

    if err:
        st.error(err)
    elif wqi_layer is not None:
        atm = get_atm(32.4, 34.85)

    # All monitoring areas now unified under user_zones
    city_wqi = {}

    # Compute user-defined zone WQI (today + 30-day history)
    user_zone_wqi = {}
    user_zone_history = {}
    if st.session_state.get("user_zones"):
        import json as _juz
        zones_json = _juz.dumps(st.session_state.user_zones)
        with st.spinner("Loading zone history (30 days)..."):
            user_zone_history = compute_zone_history_range(zones_json, 30)
        # Extract latest value per zone for current stats
        for zname, zhistory in user_zone_history.items():
            vals = [e["wqi"] for e in zhistory if e["wqi"] is not None]
            user_zone_wqi[zname] = vals[-1] if vals else None



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

    @st.cache_data(ttl=7200)
    def _get_true_color_tile(source: str, target_date_str: str):
        """Return GEE tile URL for the raw (true-color) satellite image."""
        wm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(10)
        t  = ee.Date(target_date_str)
        # Use a wider display area — full Mediterranean coast + some inland
        DISPLAY_BOX = ee.Geometry.Rectangle([33.5, 29.5, 36.5, 33.5])
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

        _s3_date_str  = _age_to_date(s3_age)
        _s2_date_str  = _age_to_date(s2_age)
        _mod_date_str = _age_to_date(mod_age)

        # Sentinel-3 WQI
        try:
            if s3_layer is not None:
                _s3_mid = ee.Image(s3_layer).getMapId(vis)
                _s3_url = _s3_mid['tile_fetcher'].url_format
                is_active_s3 = (data_source in ("S3", "Sentinel-3"))
                _raster_layers.append({"id":"wqi_s3","label":"WQI \u00b7 Sentinel-3","date":_s3_date_str,"url":_s3_url,"visible":is_active_s3})
                if is_active_s3:
                    wqi_tile_url = _s3_url
        except Exception:
            pass

        # Sentinel-3 True Color
        try:
            _s3_tc = _get_true_color_tile("S3", sel_date)
            if _s3_tc:
                _raster_layers.append({"id":"tc_s3","label":"True Color \u00b7 Sentinel-3","date":_s3_date_str,"url":_s3_tc,"visible":False})
        except Exception:
            pass

        # Sentinel-2 WQI
        try:
            if s2_layer is not None:
                _s2_mid = ee.Image(s2_layer).getMapId(vis)
                _s2_url = _s2_mid['tile_fetcher'].url_format
                is_active_s2 = (data_source in ("S2", "Sentinel-2"))
                _raster_layers.append({"id":"wqi_s2","label":"WQI \u00b7 Sentinel-2","date":_s2_date_str,"url":_s2_url,"visible":is_active_s2})
                if is_active_s2:
                    wqi_tile_url = _s2_url
        except Exception:
            pass

        # Sentinel-2 True Color
        try:
            _s2_tc = _get_true_color_tile("S2", sel_date)
            if _s2_tc:
                _raster_layers.append({"id":"tc_s2","label":"True Color \u00b7 Sentinel-2","date":_s2_date_str,"url":_s2_tc,"visible":False})
        except Exception:
            pass

        # MODIS WQI
        try:
            if mod_layer is not None:
                _mod_mid = ee.Image(mod_layer).getMapId(vis)
                _mod_url = _mod_mid['tile_fetcher'].url_format
                is_active_mod = (data_source not in ("S3", "Sentinel-3", "S2", "Sentinel-2"))
                _raster_layers.append({"id":"wqi_mod","label":"WQI \u00b7 MODIS","date":_mod_date_str,"url":_mod_url,"visible":is_active_mod})
                if is_active_mod:
                    wqi_tile_url = _mod_url
        except Exception:
            pass

        # MODIS True Color
        try:
            _mod_tc = _get_true_color_tile("MODIS", sel_date)
            if _mod_tc:
                _raster_layers.append({"id":"tc_mod","label":"True Color \u00b7 MODIS","date":_mod_date_str,"url":_mod_tc,"visible":False})
        except Exception:
            pass

        # Add only the active (visible) rasters to folium map; JS panel controls the rest
        for _rl in _raster_layers:
            if _rl["visible"]:
                folium.TileLayer(
                    tiles=_rl["url"], attr=f'GEE {_rl["label"]}',
                    name=_rl["label"], overlay=True, control=False, opacity=0.75,
                ).add_to(m)

        # ── Custom Leaflet controls via MacroElement (script macro = runs AFTER map exists) ──
        # Topleft: 🗂 Basemaps | Topright: 🛰 Satellite Products | ⛶ Fullscreen | 📏 Ruler
        import json as _cjson
        _rl_json = _cjson.dumps(_raster_layers)
        _sel_date_js = sel_date
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

    // Pre-create all raster tile layers (only add visible ones)
    _rasterLayers.forEach(function(rl) {
      var l = L.tileLayer(rl.url, {opacity: _opacity, attribution: 'GEE', zIndex: 500});
      l._isSatLayer = true;
      _tileRegistry[rl.id] = l;
      if (rl.visible) l.addTo(mapObj);
    });

    function setLayerVisible(id, on) {
      var l = _tileRegistry[id]; if (!l) return;
      if (on) { if (!mapObj.hasLayer(l)) l.addTo(mapObj); l.setOpacity(_opacity); }
      else    { if (mapObj.hasLayer(l)) mapObj.removeLayer(l); }
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
      p.innerHTML = '<div style="font-weight:bold;color:#00c8c8;font-size:12px;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px;border-bottom:1px solid rgba(0,200,200,0.18);padding-bottom:5px;">\\ud83d\\udef0 Satellite Products</div>' + rows + '<div style="border-top:1px solid rgba(0,200,200,0.15);margin-top:6px;padding-top:7px;"><label style="display:block;color:#7fb3d3;font-size:11px;margin-bottom:3px;">Opacity: <span id="satOpVal">' + Math.round(_opacity*100) + '%</span></label><input id="satOpSlider" type="range" min="10" max="100" value="' + Math.round(_opacity*100) + '" style="width:100%;accent-color:#00c8c8;"></div>';
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
            var l = _tileRegistry[k]; if (l && mapObj.hasLayer(l)) l.setOpacity(_opacity);
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
      // Show legend if any WQI layer is visible (check both checkbox state and tile registry)
      var anyWqi = false;
      _rasterLayers.forEach(function(rl) {
        if (rl.id.indexOf('wqi_') === 0) {
          var cb = document.getElementById('rl_cb_' + rl.id);
          var isOn = cb ? cb.checked : rl.visible;
          var tl = _tileRegistry[rl.id];
          if (isOn || (tl && mapObj.hasLayer(tl))) { anyWqi = true; }
        }
      });
      if (!anyWqi) { legendDiv.innerHTML = ''; legendDiv.style.display = 'none'; return; }
      legendDiv.style.display = 'block';
      legendDiv.innerHTML =
        '<div style="background:rgba(2,13,24,0.92);border:1px solid rgba(0,200,200,0.4);border-radius:6px;padding:8px 12px;font-family:Arial,sans-serif;min-width:180px;">' +
          '<div style="color:#00c8c8;font-size:11px;font-weight:bold;margin-bottom:6px;letter-spacing:0.5px;">WQI</div>' +
          '<div style="display:flex;height:14px;border-radius:3px;overflow:hidden;">' +
            '<div style="flex:1;background:#d73027;"></div>' +
            '<div style="flex:1;background:#f46d43;"></div>' +
            '<div style="flex:1;background:#fdae61;"></div>' +
            '<div style="flex:1;background:#fee090;"></div>' +
            '<div style="flex:1;background:#e0f3f8;"></div>' +
            '<div style="flex:1;background:#abd9e9;"></div>' +
            '<div style="flex:1;background:#74add1;"></div>' +
            '<div style="flex:1;background:#4575b4;"></div>' +
          '</div>' +
          '<div style="display:flex;justify-content:space-between;margin-top:3px;">' +
            '<span style="color:#f46d43;font-size:10px;">30 \\u2014 Polluted</span>' +
            '<span style="color:#74add1;font-size:10px;">Clean \\u2014 90</span>' +
          '</div>' +
        '</div>';
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
        if st.session_state.get("show_zones_on_map", True):
            for zname, zdata in st.session_state.get("user_zones", {}).items():
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
                        f'<div style="text-align:center;font-size:13px;color:#7fb3d3;padding:5px 0;">' +
                        dots_html +
                        f' <b style="color:#d6eaf8;">{cur[5]}</b> · {cur_dt} · {cur[0]:.0f}h ago</div>',
                        unsafe_allow_html=True
                    )
                with nav_r:
                    if st.button("▶", key="nav_next", use_container_width=True):
                        n = len(all_candidates)
                        st.session_state.img_idx = (st.session_state.img_idx-1)%n
                        st.rerun()

            # Load history for user zones
            history_days  = 30
            history_label = "30 ימים"
            beach_history = {}
            col_map, col_info = st.columns([1, 1], gap="small")
            with col_map:
                # ── Task 3: Show/Hide zones toggle ────────────────────────────
                z_icon = "👁️" if st.session_state.show_zones_on_map else "👁️‍🗨️"
                z_label = f"{z_icon} הסתר אזורים" if st.session_state.show_zones_on_map else f"{z_icon} הצג אזורים"
                if st.button(z_label, key="toggle_zones_map", use_container_width=False):
                    st.session_state.show_zones_on_map = not st.session_state.show_zones_on_map
                    st.rerun()
                map_data_wqi = st_folium(
                    _build_map(),
                    use_container_width=True, height=740,
                    key=f"israel_map_wqi_{st.session_state.get('img_idx',0)}",
                    returned_objects=["bounds","last_active_drawing","last_clicked"]
                )
            with col_info:
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
                    pending   = st.session_state.get("pending_zone")
                    already_pending = pending and pending.get("lat") == clat and pending.get("lon") == clon
                    already_saved   = draw_hash in st.session_state.saved_drawing_hashes
                    if not already_pending and not already_saved:
                        st.session_state["pending_zone"] = {"type": "point", "lat": clat, "lon": clon, "hash": draw_hash}
                        st.rerun()

                # All monitoring zones → visible in chart
                # Include zones from session_state even if history not yet computed
                visible_beaches = list(user_zone_history.keys())
                for zname in st.session_state.user_zones:
                    if zname not in visible_beaches:
                        visible_beaches.append(zname)

                # Pre-merge zone history into beach_history so all_dates is correct
                for zname, zhistory in user_zone_history.items():
                    if zname not in beach_history:
                        beach_history[zname] = []
                    existing = {e["date"] for e in beach_history[zname]}
                    for entry in zhistory:
                        if entry["date"] not in existing and entry["wqi"] is not None:
                            beach_history[zname].append(entry)
                            existing.add(entry["date"])

                # Build comparison chart
                if visible_beaches:
                    import json as _json

                    # Add current df as latest data point if history missing
                    if df is not None and not df.empty:
                        for _, row in df.iterrows():
                            if row["name"] in beach_history and row["wqi"]:
                                existing_dates = {e["date"] for e in beach_history[row["name"]]}
                                if sel_date not in existing_dates:
                                    beach_history[row["name"]].append({"date": sel_date, "wqi": row["wqi"], "source": data_source[:2]})
                            elif row["name"] not in beach_history and row["wqi"]:
                                beach_history[row["name"]] = [{"date": sel_date, "wqi": row["wqi"], "source": data_source[:2]}]

                    # Merge city_wqi into beach_history for chart
                    for city_name, cwqi in (city_wqi or {}).items():
                        if cwqi is not None:
                            if city_name not in beach_history:
                                beach_history[city_name] = []
                            existing = {e["date"] for e in beach_history[city_name]}
                            if sel_date not in existing:
                                beach_history[city_name].append({"date": sel_date, "wqi": cwqi, "source": "S3"})

                    all_dates = sorted(set(
                        e["date"] for name in visible_beaches
                        for e in beach_history.get(name, [])
                    ))

                    def _get_current(name):
                        if city_wqi and name in city_wqi and city_wqi[name] is not None:
                            return float(city_wqi[name])
                        if user_zone_wqi and name in user_zone_wqi and user_zone_wqi[name] is not None:
                            return float(user_zone_wqi[name])
                        hist_vals = [e["wqi"] for e in beach_history.get(name,[]) if e["wqi"] and str(e["wqi"]) != "nan"]
                        return hist_vals[-1] if hist_vals else None

                    PALETTE = ["#1D9E75","#378ADD","#7F77DD","#BA7517","#D4537E","#E24B4A","#639922","#D85A30"]

                    # ── Task 8: Group-aware chart building ────────────────────
                    chart_view_mode = st.session_state.get("chart_view_mode", "All zones (individual)")
                    selected_group  = None
                    if chart_view_mode.startswith("Group: "):
                        selected_group = chart_view_mode[len("Group: "):]

                    if selected_group:
                        # Build one averaged line per group (only show selected group's zones aggregated)
                        # Also show zones NOT in any group individually, and other groups as their average
                        grp_map = {}   # group_name -> [zone_names]
                        ungrouped = []
                        for zn in visible_beaches:
                            zg = st.session_state.user_zones.get(zn, {}).get("group", "")
                            if zg:
                                grp_map.setdefault(zg, []).append(zn)
                            else:
                                ungrouped.append(zn)

                        # For the selected group: one line per member zone (individual)
                        # For other groups: one averaged line
                        # For ungrouped: individual lines
                        chart_names = []
                        chart_names += ungrouped
                        for g, members in grp_map.items():
                            if g == selected_group:
                                chart_names += members   # individual lines for selected group
                            else:
                                chart_names.append(f"[{g}] avg")  # averaged line for other groups

                        beach_colors = {}
                        for i, nm in enumerate(chart_names):
                            beach_colors[nm] = PALETTE[i % len(PALETTE)]

                        datasets = []
                        for nm in chart_names:
                            if nm.startswith("[") and nm.endswith("] avg"):
                                # Averaged group line
                                grp_name = nm[1:nm.index("] avg")]
                                members  = grp_map.get(grp_name, [])
                                data = []
                                srcs = []
                                for d in all_dates:
                                    vals = [beach_history[m][0]["wqi"] if m in beach_history else None
                                            for m in members]
                                    # per-date lookup
                                    day_vals = []
                                    day_srcs = []
                                    for m in members:
                                        hm = {e["date"]: e for e in beach_history.get(m, [])}
                                        if d in hm and hm[d]["wqi"] is not None:
                                            day_vals.append(hm[d]["wqi"])
                                            day_srcs.append(hm[d].get("source",""))
                                    avg = round(sum(day_vals)/len(day_vals), 1) if day_vals else None
                                    data.append(avg)
                                    srcs.append(",".join(set(day_srcs)) if day_srcs else "")
                                datasets.append({
                                    "label": nm,
                                    "data": data,
                                    "sources": srcs,
                                    "borderColor": beach_colors[nm],
                                    "borderDash": [4, 2],
                                    "_isGroupAvg": True,
                                })
                            else:
                                hmap = {e["date"]: e for e in beach_history.get(nm, [])}
                                data = [hmap[d]["wqi"] if d in hmap else None for d in all_dates]
                                srcs = [hmap[d].get("source","") if d in hmap else "" for d in all_dates]
                                datasets.append({
                                    "label": nm,
                                    "data": data,
                                    "sources": srcs,
                                    "borderColor": beach_colors[nm],
                                    "borderDash": [],
                                })

                        current_vals = {}
                        for nm in chart_names:
                            if nm.startswith("[") and nm.endswith("] avg"):
                                grp_name = nm[1:nm.index("] avg")]
                                members  = grp_map.get(grp_name, [])
                                mvs = [_get_current(m) for m in members if _get_current(m)]
                                current_vals[nm] = round(sum(mvs)/len(mvs),1) if mvs else None
                            else:
                                current_vals[nm] = _get_current(nm)

                        display_names = chart_names

                    else:
                        # Original individual mode
                        beach_colors = {name: PALETTE[i % len(PALETTE)] for i,name in enumerate(visible_beaches)}
                        current_vals = {n: _get_current(n) for n in visible_beaches}
                        datasets = []
                        for name in visible_beaches:
                            hist_map = {e["date"]: e for e in beach_history.get(name,[])}
                            data = [hist_map[d]["wqi"] if d in hist_map else None for d in all_dates]
                            src_map = [hist_map[d].get("source","") if d in hist_map else "" for d in all_dates]
                            datasets.append({
                                "label": name,
                                "data": data,
                                "sources": src_map,
                                "borderColor": beach_colors[name],
                                "borderDash": [5,3] if (current_vals.get(name,100) or 100) < 30 else [],
                            })
                        display_names = visible_beaches

                    valid_vals = {n:v for n,v in current_vals.items() if v}
                    best  = max(valid_vals, key=valid_vals.get) if valid_vals else None
                    worst = min(valid_vals, key=valid_vals.get) if valid_vals else None

                    legend_items = []
                    for name in display_names:
                        v   = current_vals.get(name)
                        col = "#1ecb7b" if v and v>=70 else "#f0a500" if v and v>=55 else "#e03c3c" if v else "#888"
                        legend_items.append({
                            "name": name,
                            "color": beach_colors[name],
                            "wqi": round(v,1) if v else "---",
                            "wqiColor": col,
                        })

                    # Task 6: Identify territorial waters zone — must come before chart_json
                    TW_KEYWORDS = ["territorial", "טריטוריאל", "ים ישראל", "israel water",
                                   "territorial water", "tw_", "terr_"]
                    tw_zone_name = None
                    for zn in display_names:
                        if any(kw in zn.lower() for kw in TW_KEYWORDS):
                            tw_zone_name = zn
                            break

                    # Build source label for chart subtitle from actual data used
                    sources_used = sorted(set(
                        e.get("source","") for name in display_names
                        for e in beach_history.get(name, []) if e.get("source")
                    ))
                    src_label = " · ".join(sources_used) if sources_used else "S3 · S2 · MODIS"

                    # Task 8: Add group mode indicator to subtitle
                    if selected_group:
                        src_label = f"Group: {selected_group} · " + src_label

                    chart_json  = _json.dumps(datasets)
                    labels_json = _json.dumps(all_dates)  # full YYYY-MM-DD for tooltip
                    labels_short_json = _json.dumps([d[5:].replace("-","/") for d in all_dates])  # MM/DD for axis
                    legend_json = _json.dumps(legend_items)

                    # Task 6: compute territorial waters series after beach_history is ready
                    tw_avg_json = "null"
                    tw_label_js = "null"
                    if tw_zone_name and tw_zone_name in beach_history:
                        tw_map = {e["date"]: e["wqi"] for e in beach_history[tw_zone_name] if e["wqi"] is not None}
                        tw_series = [tw_map.get(d) for d in all_dates]
                        tw_avg_json = _json.dumps(tw_series)
                        tw_label_js = _json.dumps(tw_zone_name)

                    best_name   = best or "---"
                    best_val    = round(valid_vals[best],1) if best else "---"
                    worst_name  = worst or "---"
                    worst_val   = round(valid_vals[worst],1) if worst else "---"
                    n_beaches   = len(visible_beaches)

                    # Coast statistics — pull from current_vals (includes user zones + cities)
                    # and exclude territorial-waters reference zone from "cleanest/most-polluted"
                    # comparisons so the benchmark line doesn't dominate.
                    TW_KEYS = ["territorial","טריטוריאל","ים ישראל","israel water","tw_","terr_"]
                    def _is_tw(n):
                        return any(k in n.lower() for k in TW_KEYS)

                    # Combine: user zones (current_vals) + predefined cities (city_wqi)
                    combined = {}
                    for k, v in (current_vals or {}).items():
                        if v is not None and not _is_tw(k) and not (k.startswith("[") and k.endswith("] avg")):
                            combined[k] = float(v)
                    for k, v in (city_wqi or {}).items():
                        if v is not None and k not in combined:
                            combined[k] = float(v)

                    cst_valid   = combined
                    cst_avg     = f"{sum(cst_valid.values())/len(cst_valid):.1f}" if cst_valid else "N/A"
                    cst_best    = max(cst_valid, key=cst_valid.get) if cst_valid else "N/A"
                    cst_best_v  = f"{cst_valid[cst_best]:.1f}" if cst_valid else ""
                    cst_worst   = min(cst_valid, key=cst_valid.get) if cst_valid else "N/A"
                    cst_worst_v = f"{cst_valid[cst_worst]:.1f}" if cst_valid else ""
                    cst_nclean  = sum(1 for v in cst_valid.values() if v>=70)
                    cst_nmod    = sum(1 for v in cst_valid.values() if 50<=v<70)
                    cst_npoll   = sum(1 for v in cst_valid.values() if v<50)

                    chart_html = f"""
<!DOCTYPE html><html style="height:100%;"><body style="margin:0;padding:0;background:#020d18;overflow:hidden;height:100%;">
<div style="padding:0.4rem 0.5rem 0.25rem;height:100vh;box-sizing:border-box;display:flex;flex-direction:column;gap:0;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
    <p style="font-size:14px;color:#7fb3d3;margin:0;">איכות פני המים · {history_label} · {src_label}</p>
    <div style="display:flex;align-items:center;gap:8px;">
      <p style="font-size:13px;color:#7fb3d3;margin:0;">{n_beaches} אזורים</p>
      <button id="chartFsBtn" onclick="(function(){{var el=document.documentElement;if(!document.fullscreenElement){{el.requestFullscreen&&el.requestFullscreen();document.getElementById('chartFsBtn').textContent='✕';}}else{{document.exitFullscreen&&document.exitFullscreen();document.getElementById('chartFsBtn').textContent='⛶';}}}})()" style="background:rgba(0,200,200,0.1);border:1px solid rgba(0,200,200,0.35);border-radius:4px;color:#00c8c8;cursor:pointer;font-size:15px;padding:2px 7px;line-height:1.4;" title="Full Screen">⛶</button>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-bottom:7px;">
    <div style="background:rgba(0,200,200,0.06);border:1px solid rgba(0,200,200,0.15);border-radius:5px;padding:5px;text-align:center;">
      <p style="font-size:12px;color:#7fb3d3;margin:0;">ממוצע חוף ישראל</p>
      <p style="font-size:24px;font-weight:700;margin:1px 0;color:#d6eaf8;">{cst_avg}</p>
      <p style="font-size:12px;color:#7fb3d3;margin:0;">WQI</p>
    </div>
    <div style="background:rgba(69,117,180,0.08);border:1px solid rgba(69,117,180,0.2);border-radius:5px;padding:5px;text-align:center;">
      <p style="font-size:12px;color:#7fb3d3;margin:0;">הכי נקי</p>
      <p style="font-size:13px;font-weight:600;margin:1px 0;color:#4575b4;">{cst_best}</p>
      <p style="font-size:18px;font-weight:700;margin:0;color:#4575b4;">{cst_best_v}</p>
    </div>
    <div style="background:rgba(215,48,39,0.08);border:1px solid rgba(215,48,39,0.2);border-radius:5px;padding:5px;text-align:center;">
      <p style="font-size:12px;color:#7fb3d3;margin:0;">הכי מזוהם</p>
      <p style="font-size:13px;font-weight:600;margin:1px 0;color:#d73027;">{cst_worst}</p>
      <p style="font-size:18px;font-weight:700;margin:0;color:#d73027;">{cst_worst_v}</p>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-bottom:8px;">
    <div style="background:rgba(69,117,180,0.12);border-radius:4px;padding:3px;text-align:center;">
      <span style="font-size:18px;font-weight:700;color:#4575b4;">{cst_nclean}</span>
      <span style="font-size:12px;color:#7fb3d3;"> נקיים</span>
    </div>
    <div style="background:rgba(253,174,97,0.12);border-radius:4px;padding:3px;text-align:center;">
      <span style="font-size:18px;font-weight:700;color:#fdae61;">{cst_nmod}</span>
      <span style="font-size:12px;color:#7fb3d3;"> בינוניים</span>
    </div>
    <div style="background:rgba(215,48,39,0.12);border-radius:4px;padding:3px;text-align:center;">
      <span style="font-size:18px;font-weight:700;color:#d73027;">{cst_npoll}</span>
      <span style="font-size:12px;color:#7fb3d3;"> מזוהמים</span>
    </div>
  </div>
  <div style="display:flex;gap:0;align-items:flex-start;flex:1;min-height:0;">
    <div style="position:relative;flex:1;min-height:0;height:100%;padding-bottom:40px;overflow:hidden;">
      <canvas id="beachTrend" role="img" aria-label="Water quality trends for {n_beaches} beaches" style="width:100%;height:100%;"></canvas>
    </div>
    <div id="beachLegend" style="display:flex;flex-direction:column;justify-content:flex-start;gap:4px;overflow-y:auto;min-width:170px;max-width:180px;padding:4px 6px;max-height:calc(100vh - 170px);"></div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
(function(){{
  var ds    = {chart_json};
  var lb    = {labels_json};
  var ls    = {labels_short_json};
  var lg    = {legend_json};
  var twAvg = {tw_avg_json};   // Task 6: territorial waters series or null
  var twLbl = {tw_label_js};   // Task 6: label string or null

  // ── Task 2: hidden-state tracking ──────────────────────────────────────────
  var hiddenMap = {{}};  // label -> true if hidden
  var chartRef  = null;

  // ── Task 6: Inject territorial-waters reference dataset if present ──────────
  if (twAvg) {{
    ds = ds.filter(function(d){{ return d.label !== twLbl; }});
    ds.push({{
      label: twLbl,
      data: twAvg,
      sources: twAvg.map(function(){{ return 'TW'; }}),
      borderColor: '#FFD700',
      borderDash: [],
      borderWidth: 3.5,
      pointRadius: 3,
      pointBackgroundColor: '#FFD700',
      backgroundColor: 'transparent',
      tension: 0.35,
      spanGaps: true,
      _isTW: true
    }});
  }}

  // ── Style all datasets ──────────────────────────────────────────────────────
  ds = ds.map(function(d) {{
    var isTW = d._isTW || false;
    return Object.assign({{}}, d, {{
      backgroundColor: 'transparent',
      tension: 0.35,
      pointRadius: isTW ? 3 : 4,
      pointBackgroundColor: d.borderColor,
      borderWidth: isTW ? 3.5 : 2,
      spanGaps: true
    }});
  }});

  // ── Auto-range Y axis: 10% padding top/bottom, clamped to [0, 100] ──────────
  var _allVals = [];
  ds.forEach(function(d) {{
    (d.data || []).forEach(function(v) {{ if (v !== null && v !== undefined) _allVals.push(v); }});
  }});
  var yMin = 0, yMax = 100;   // fallback
  if (_allVals.length > 0) {{
    var minV = Math.min.apply(null, _allVals);
    var maxV = Math.max.apply(null, _allVals);
    var range = Math.max(maxV - minV, 1);
    var pad = range * 0.10;
    yMin = Math.max(0,   Math.floor(minV - pad));
    yMax = Math.min(100, Math.ceil (maxV + pad));
    // Guarantee a minimum visible band of 10 units
    if (yMax - yMin < 10) {{
      var mid = (yMax + yMin) / 2;
      yMin = Math.max(0,   Math.floor(mid - 5));
      yMax = Math.min(100, Math.ceil (mid + 5));
    }}
  }}

  // ── Task 5: end-of-line label plugin ───────────────────────────────────────
  var endLabelPlugin = {{
    id: 'endLabel',
    afterDatasetsDraw: function(chart) {{
      var ctx = chart.ctx;
      chart.data.datasets.forEach(function(dataset, i) {{
        var meta = chart.getDatasetMeta(i);
        if (meta.hidden) return;
        // Find the last visible (non-null) point
        var lastPt = null;
        for (var j = dataset.data.length - 1; j >= 0; j--) {{
          if (dataset.data[j] !== null && dataset.data[j] !== undefined) {{
            var el = meta.data[j];
            if (el) {{ lastPt = {{ x: el.x, y: el.y, val: dataset.data[j] }}; break; }}
          }}
        }}
        if (!lastPt) return;
        var isTW = dataset._isTW || false;
        var label = dataset.label;
        // Truncate long names
        if (label.length > 18) label = label.substring(0, 16) + '…';
        ctx.save();
        ctx.font = isTW ? 'bold 11px Arial' : '10px Arial';
        ctx.fillStyle = dataset.borderColor;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        // Draw small connecting tick
        ctx.beginPath();
        ctx.strokeStyle = dataset.borderColor;
        ctx.lineWidth = 1;
        ctx.moveTo(lastPt.x, lastPt.y);
        ctx.lineTo(lastPt.x + 5, lastPt.y);
        ctx.stroke();
        ctx.fillText(label, lastPt.x + 7, lastPt.y);
        ctx.restore();
      }});
    }}
  }};

  // ── Build chart ─────────────────────────────────────────────────────────────
  chartRef = new Chart(document.getElementById('beachTrend'), {{
    type: 'line',
    data: {{ labels: ls, datasets: ds }},
    plugins: [endLabelPlugin],
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      layout: {{ padding: {{ right: 110, left: 4, top: 4, bottom: 4 }} }},  // room for end labels
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            title: function(items) {{ return lb[items[0].dataIndex]; }},
            label: function(c) {{
              var src = c.dataset.sources ? c.dataset.sources[c.dataIndex] : '';
              var srcLabel = src ? (' · ' + src) : '';
              return c.dataset.label + ': ' + c.parsed.y + srcLabel;
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          ticks: {{
            color: '#ffffff', font: {{size:13, weight:'600'}},
            maxRotation: 45, minRotation: 0, autoSkip: true,
            maxTicksLimit: 10, padding: 4
          }},
          grid: {{ color: 'rgba(255,255,255,0.08)' }},
          border: {{ color: 'rgba(255,255,255,0.2)' }},
          title: {{ display: false }}
        }},
        y: {{
          min: yMin, max: yMax,
          ticks: {{
            color: '#cccccc', font: {{size:14, weight:'bold'}},
            maxTicksLimit: 6,
            callback: function(v, idx, ticks) {{
              // Bottom-most visible tick → label "מזוהם N"
              if (idx === 0)                  return 'מזוהם ' + Math.round(v);
              // Top-most visible tick → label "נקי N"
              if (idx === ticks.length - 1)   return 'נקי ' + Math.round(v);
              return Math.round(v);
            }}
          }},
          grid: {{ color: 'rgba(255,255,255,0.08)' }},
          title: {{ display:true, text:'איכות המים (WQI)', color:'#cccccc', font:{{size:14,weight:'bold'}} }}
        }}
      }}
    }}
  }});

  // ── Task 2: Clickable legend with toggle ────────────────────────────────────
  var el = document.getElementById('beachLegend');
  lg.forEach(function(item, idx) {{
    var r = document.createElement('div');
    r.style.cssText = 'display:flex;align-items:center;gap:5px;cursor:pointer;padding:2px 4px;border-radius:3px;transition:opacity 0.2s;user-select:none;';
    r.dataset.label = item.name;

    // Find matching dataset index (including TW which may have been appended)
    function getDatasetIdx(label) {{
      for (var i=0; i<chartRef.data.datasets.length; i++) {{
        if (chartRef.data.datasets[i].label === label) return i;
      }}
      return -1;
    }}

    var isTW = twLbl && item.name === twLbl;
    var lineW = isTW ? '20px' : '16px';
    var lineH = isTW ? '3px' : '2px';
    r.innerHTML =
      '<span style="width:'+lineW+';height:'+lineH+';background:'+item.color+';flex-shrink:0;border-radius:1px;'+(isTW?'box-shadow:0 0 4px '+item.color+';':'')+'" class="leg-line"></span>' +
      '<span style="font-size:13px;color:#7fb3d3;flex:1;" class="leg-name">'+(isTW?'<b>'+item.name+'</b>':item.name)+'</span>' +
      '<span style="font-size:13px;font-weight:600;color:'+item.wqiColor+';" class="leg-wqi">'+item.wqi+'</span>';

    r.addEventListener('click', function() {{
      var label = this.dataset.label;
      var dsIdx = getDatasetIdx(label);
      if (dsIdx === -1) return;
      var meta  = chartRef.getDatasetMeta(dsIdx);
      meta.hidden = !meta.hidden;
      hiddenMap[label] = meta.hidden;
      // Visual feedback on legend row
      this.style.opacity = meta.hidden ? '0.35' : '1.0';
      this.querySelector('.leg-line').style.opacity = meta.hidden ? '0.3' : '1';
      chartRef.update();
    }});

    el.appendChild(r);
  }});

  // Also add TW entry to legend if it exists but wasn't in original lg
  if (twLbl) {{
    var twInLg = lg.some(function(i){{ return i.name === twLbl; }});
    if (!twInLg) {{
      var twValid = (twAvg||[]).filter(function(v){{ return v!==null&&v!==undefined; }});
      var twCurr  = twValid.length ? twValid[twValid.length-1] : null;
      var twCol   = twCurr>=70?'#1ecb7b':twCurr>=55?'#f0a500':'#e03c3c';
      var r2 = document.createElement('div');
      r2.style.cssText='display:flex;align-items:center;gap:5px;cursor:pointer;padding:2px 4px;border-radius:3px;transition:opacity 0.2s;user-select:none;margin-top:6px;border-top:1px solid rgba(255,215,0,0.2);padding-top:6px;';
      r2.dataset.label = twLbl;
      r2.innerHTML =
        '<span style="width:20px;height:3px;background:#FFD700;flex-shrink:0;border-radius:1px;box-shadow:0 0 4px #FFD700;" class="leg-line"></span>' +
        '<span style="font-size:13px;color:#FFD700;flex:1;font-weight:bold;" class="leg-name">'+twLbl+'</span>' +
        '<span style="font-size:13px;font-weight:600;color:'+twCol+';" class="leg-wqi">'+(twCurr?twCurr.toFixed(1):'---')+'</span>';
      r2.addEventListener('click', function() {{
        var dsIdx = -1;
        for (var i=0; i<chartRef.data.datasets.length; i++) {{
          if (chartRef.data.datasets[i].label === twLbl) {{ dsIdx=i; break; }}
        }}
        if (dsIdx===-1) return;
        var meta = chartRef.getDatasetMeta(dsIdx);
        meta.hidden = !meta.hidden;
        this.style.opacity = meta.hidden ? '0.35' : '1.0';
        chartRef.update();
      }});
      el.appendChild(r2);
    }}
  }}

}})();
</script></body></html>
"""
                    components.html(chart_html, height=740, scrolling=False)
                else:
                    if st.session_state.user_zones:
                        st.info(f"⏳ Loading history for {len(st.session_state.user_zones)} zones...")
                    else:
                        st.caption("Draw a shape on the map to start monitoring")

                # ── Monitoring Areas (unified: points + polygons) ─────────────
                pending_zone = st.session_state.get("pending_zone")
                with st.expander("📍 Monitoring Areas", expanded=bool(pending_zone)):
                    if pending_zone:
                        if pending_zone["type"] == "polygon":
                            st.info(f"🟦 New polygon: {len(pending_zone['coords'])} vertices")
                        else:
                            st.info(f"📍 New point: {pending_zone['lat']:.4f}, {pending_zone['lon']:.4f}")
                        zone_name_inp = st.text_input("Name:", key="zone_name_inp", placeholder="e.g. Haifa Anchorage")

                        # Task 8: Group assignment
                        existing_groups = sorted(set(
                            z.get("group","") for z in st.session_state.user_zones.values()
                            if z.get("group","")
                        ))
                        group_options = ["— No group —"] + existing_groups + ["+ New group…"]
                        grp_sel = st.selectbox("Group:", group_options, key="zone_group_sel")
                        if grp_sel == "+ New group…":
                            zone_group_inp = st.text_input("New group name:", key="zone_group_new_inp",
                                                            placeholder="e.g. Ports")
                        elif grp_sel == "— No group —":
                            zone_group_inp = ""
                        else:
                            zone_group_inp = grp_sel

                        zc1, zc2 = st.columns(2)
                        with zc1:
                            if st.button("💾 Save", use_container_width=True, key="save_zone"):
                                if zone_name_inp.strip():
                                    zn    = zone_name_inp.strip()
                                    grp   = zone_group_inp.strip() if zone_group_inp else ""
                                    if pending_zone["type"] == "polygon":
                                        st.session_state.user_zones[zn] = {"coords": pending_zone["coords"], "type": "polygon", "group": grp}
                                    else:
                                        lat, lon = pending_zone["lat"], pending_zone["lon"]
                                        d = 0.005
                                        box = [[lon-d,lat-d],[lon+d,lat-d],[lon+d,lat+d],[lon-d,lat+d],[lon-d,lat-d]]
                                        st.session_state.user_zones[zn] = {"coords": box, "type": "point", "lat": lat, "lon": lon, "group": grp}
                                    save_zones(st.session_state.user_zones)
                                    h = pending_zone.get("hash")
                                    if h:
                                        st.session_state.saved_drawing_hashes.add(h)
                                    st.session_state.pop("pending_zone", None)
                                    compute_zone_history_range.clear()
                                    st.rerun()
                        with zc2:
                            if st.button("✕ Discard", use_container_width=True, key="cancel_zone"):
                                h = pending_zone.get("hash")
                                if h:
                                    st.session_state.saved_drawing_hashes.add(h)
                                st.session_state.pop("pending_zone", None)
                                st.rerun()

                    if st.session_state.user_zones:
                        # Task 8: Group filter + chart view mode
                        all_zone_groups = sorted(set(
                            z.get("group","") for z in st.session_state.user_zones.values()
                            if z.get("group","")
                        ))
                        if all_zone_groups:
                            st.markdown("<div style='font-size:13px;color:#7fb3d3;margin:6px 0 2px;'>📊 Chart view</div>",
                                        unsafe_allow_html=True)
                            view_opts = ["All zones (individual)"] + [f"Group: {g}" for g in all_zone_groups]
                            if "chart_view_mode" not in st.session_state:
                                st.session_state.chart_view_mode = "All zones (individual)"
                            new_view = st.selectbox("", view_opts, key="chart_view_sel",
                                                    index=view_opts.index(st.session_state.chart_view_mode)
                                                    if st.session_state.chart_view_mode in view_opts else 0,
                                                    label_visibility="collapsed")
                            if new_view != st.session_state.chart_view_mode:
                                st.session_state.chart_view_mode = new_view
                                st.rerun()

                        for zname in list(st.session_state.user_zones.keys()):
                            zwqi  = user_zone_wqi.get(zname)
                            zwqi_str = f"{zwqi:.1f}" if zwqi is not None else "..."
                            ztype = st.session_state.user_zones[zname].get("type", "polygon")
                            zgrp  = st.session_state.user_zones[zname].get("group", "")
                            icon  = "📍" if ztype == "point" else "🟦"
                            grp_badge = f' <span style="font-size:12px;background:rgba(0,200,200,0.15);color:#00c8c8;border-radius:3px;padding:1px 5px;">{zgrp}</span>' if zgrp else ""
                            zc, zd = st.columns([3, 1])
                            with zc:
                                st.markdown(f'<div style="font-size:14px;color:#d6eaf8;padding:2px 0;">{icon} {zname}{grp_badge} <span style="color:#7fb3d3;">WQI: {zwqi_str}</span></div>',
                                            unsafe_allow_html=True)
                            with zd:
                                if st.button("🗑", key=f"del_zone_{zname}"):
                                    del st.session_state.user_zones[zname]
                                    save_zones(st.session_state.user_zones)
                                    compute_zone_history_range.clear()
                                    st.rerun()

                        # ── Export ──────────────────────────────────────────
                        import json as _jex
                        zones_export = _jex.dumps(st.session_state.user_zones, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="⬇️ Export zones",
                            data=zones_export,
                            file_name="medi_zones.json",
                            mime="application/json",
                            use_container_width=True,
                            key="export_zones"
                        )
                    else:
                        st.caption("Click on the map or draw a shape to add a monitoring area")

                    # ── Import ──────────────────────────────────────────────
                    st.markdown("<hr style='margin:6px 0;border-color:rgba(0,200,200,0.15)'>", unsafe_allow_html=True)
                    uploaded_zones = st.file_uploader("⬆️ Import zones", type="json", key="import_zones",
                                                       label_visibility="collapsed")
                    if uploaded_zones:
                        try:
                            import json as _jim
                            imported = _jim.loads(uploaded_zones.read())
                            st.session_state.user_zones.update(imported)
                            save_zones(st.session_state.user_zones)
                            compute_zone_history_range.clear()
                            st.success(f"✅ Imported {len(imported)} zones")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Import failed: {e}")
                    else:
                        st.caption("⬆️ Import zones from a previously exported JSON file")



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
