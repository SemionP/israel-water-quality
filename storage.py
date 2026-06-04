"""MEDI Platform — Zone Storage (Google Drive + fallback)"""

import json, tempfile, os, urllib.request, urllib.parse, urllib.error
import streamlit as st

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
