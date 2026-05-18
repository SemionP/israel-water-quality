import ee
import json
import os
import tempfile
from datetime import datetime, timedelta
from urllib.parse import quote

import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium


st.set_page_config(page_title="ניטור איכות מי ים — ישראל", page_icon="🌊", layout="wide")

CONTACT_EMAIL = "your-email@example.com"


def inject_analytics():
    components.html(
        '<script async src="https://cloud.umami.is/script.js" data-website-id="07a48db1-5aa7-4d88-aaac-9cfb6fc2600d"></script>',
        height=0,
    )

    if "ga_loaded" not in st.session_state:
        st.session_state.ga_loaded = True

        components.html(
            """
            <!-- Google tag (gtag.js) -->
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
# אימות GEE עם Streamlit Secrets
# ==============================
@st.cache_resource
def init_gee():
    creds_dict = dict(st.secrets["gee_credentials"])
    creds_json = json.dumps(creds_dict)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(creds_json)
        tmp_path = f.name

    try:
        service_account = creds_dict["client_email"]
        credentials = ee.ServiceAccountCredentials(service_account, tmp_path)
        ee.Initialize(credentials)
    finally:
        os.unlink(tmp_path)


# ==============================
# הגדרות
# ==============================
HAIFA_CENTER = [32.4, 34.85]
HAIFA_BBOX = ee.Geometry.Rectangle([34.50, 31.55, 35.15, 33.10])

ISRAEL_TERRITORIAL = ee.Geometry.Polygon([[
    [34.95, 33.10], [34.60, 33.10],
    [34.20, 32.60], [34.15, 32.00],
    [34.20, 31.55], [34.55, 31.30],
    [34.75, 31.25], [34.95, 31.30],
    [35.00, 31.55], [35.00, 32.00],
    [35.10, 32.60], [35.10, 33.10],
    [34.95, 33.10],
]])

BEACHES = [
    {"name": "ראש הנקרה", "lat": 33.074, "lon": 35.100},
    {"name": "נהריה", "lat": 33.005, "lon": 35.088},
    {"name": "עכו", "lat": 32.927, "lon": 35.065},
    {"name": "קריית ים", "lat": 32.865, "lon": 35.058},
    {"name": "חיפה צפון", "lat": 32.846, "lon": 34.972},
    {"name": "חיפה מרכז", "lat": 32.819, "lon": 34.960},
    {"name": "חיפה דרום", "lat": 32.783, "lon": 34.950},
    {"name": "עתלית", "lat": 32.693, "lon": 34.938},
    {"name": "זיכרון יעקב", "lat": 32.571, "lon": 34.918},
    {"name": "קיסריה", "lat": 32.497, "lon": 34.893},
    {"name": "נתניה", "lat": 32.334, "lon": 34.855},
    {"name": "הרצליה", "lat": 32.163, "lon": 34.796},
    {"name": "תל אביב צפון", "lat": 32.108, "lon": 34.768},
    {"name": "תל אביב מרכז", "lat": 32.080, "lon": 34.762},
    {"name": "תל אביב דרום", "lat": 32.051, "lon": 34.757},
    {"name": "בת ים", "lat": 32.017, "lon": 34.749},
    {"name": "ראשון לציון", "lat": 31.973, "lon": 34.737},
    {"name": "אשדוד צפון", "lat": 31.844, "lon": 34.658},
