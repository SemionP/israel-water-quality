"""MEDI Platform — Zone Storage (Google Drive + fallback)"""

import json, os, time, urllib.request, urllib.parse, urllib.error
import streamlit as st

ZONES_KEY       = "medi-zones-v1"
GDRIVE_FILENAME = "medi_zones.json"
GDRIVE_FOLDER   = "1VU11P0UCzeMiVsn0k1RiIHuEu8bBLUFH"
GDRIVE_FILE_ID  = "1KTI_oRHIrvRJNtfZYrWMkhNi2D-34Y9v"

def _gdrive_token() -> str | None:
    cache = st.session_state.setdefault("_gdrive_token_cache", {})
    now   = int(time.time())
    if cache.get("token") and cache.get("exp", 0) - now > 300:
        return cache["token"]
    try:
        import base64, json as _j
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding as _pad

        creds    = dict(st.secrets["gee_credentials"])
        sa_email = creds["client_email"]
        priv_key = creds["private_key"]

        def b64(d): return base64.urlsafe_b64encode(d).rstrip(b"=")

        iat = now
        exp = now + 3600
        hdr = b64(_j.dumps({"alg": "RS256", "typ": "JWT"}).encode())
        pay = b64(_j.dumps({
            "iss":   sa_email,
            "scope": "https://www.googleapis.com/auth/drive",
            "aud":   "https://oauth2.googleapis.com/token",
            "iat":   iat,
            "exp":   exp,
        }).encode())
        msg = hdr + b"." + pay
        key = serialization.load_pem_private_key(priv_key.encode(), password=None)
        sig = b64(key.sign(msg, _pad.PKCS1v15(), hashes.SHA256()))
        jwt = (msg + b"." + sig).decode()

        body = urllib.parse.urlencode({
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion":  jwt,
        }).encode()
        req = urllib.request.Request(
            "https://oauth2.googleapis.com/token",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        token = _j.loads(urllib.request.urlopen(req, timeout=10).read())["access_token"]
        cache["token"] = token
        cache["exp"]   = exp
        return token
    except Exception:
        return None


def _find_file_id(token: str) -> str | None:
    q   = f"name='{GDRIVE_FILENAME}' and '{GDRIVE_FOLDER}' in parents and trashed=false"
    url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode({
        "q": q, "fields": "files(id)", "pageSize": "5",
        "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
    })
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        res = json.loads(urllib.request.urlopen(req, timeout=10).read())
        files = res.get("files", [])
        return files[0]["id"] if files else None
    except Exception:
        return None


def _create_file(token: str, data: bytes) -> str | None:
    meta = json.dumps({"name": GDRIVE_FILENAME, "parents": [GDRIVE_FOLDER]}).encode()
    req1 = urllib.request.Request(
        "https://www.googleapis.com/drive/v3/files?supportsAllDrives=true",
        data=meta, method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        res1 = json.loads(urllib.request.urlopen(req1, timeout=10).read())
        fid  = res1.get("id")
        if not fid:
            return None
        url2 = f"https://www.googleapis.com/upload/drive/v3/files/{fid}?uploadType=media&supportsAllDrives=true"
        req2 = urllib.request.Request(
            url2, data=data, method="PATCH",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        urllib.request.urlopen(req2, timeout=15)
        return fid
    except Exception:
        return None


# ── Generic Drive helpers ──────────────────────────────────────────────────────

def _drive_find(token: str, filename: str) -> str | None:
    q   = f"name='{filename}' and '{GDRIVE_FOLDER}' in parents and trashed=false"
    url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode({
        "q": q, "fields": "files(id)", "pageSize": "5",
        "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
    })
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        res   = json.loads(urllib.request.urlopen(req, timeout=10).read())
        files = res.get("files", [])
        return files[0]["id"] if files else None
    except Exception:
        return None


def _drive_read(token: str, fid: str) -> str | None:
    req = urllib.request.Request(
        f"https://www.googleapis.com/drive/v3/files/{fid}?alt=media&supportsAllDrives=true",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        return urllib.request.urlopen(req, timeout=15).read().decode()
    except Exception:
        return None


def _drive_write(token: str, filename: str, data: bytes, fid: str | None = None):
    if fid:
        url = f"https://www.googleapis.com/upload/drive/v3/files/{fid}?uploadType=media&supportsAllDrives=true"
        req = urllib.request.Request(
            url, data=data, method="PATCH",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=30)
        return fid
    else:
        meta = json.dumps({"name": filename, "parents": [GDRIVE_FOLDER]}).encode()
        req1 = urllib.request.Request(
            "https://www.googleapis.com/drive/v3/files?supportsAllDrives=true",
            data=meta, method="POST",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        res1    = json.loads(urllib.request.urlopen(req1, timeout=10).read())
        new_fid = res1.get("id")
        if new_fid:
            url2 = f"https://www.googleapis.com/upload/drive/v3/files/{new_fid}?uploadType=media&supportsAllDrives=true"
            req2 = urllib.request.Request(
                url2, data=data, method="PATCH",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            urllib.request.urlopen(req2, timeout=30)
        return new_fid


# ── Zones ─────────────────────────────────────────────────────────────────────

def load_zones() -> dict:
    try:
        token = _gdrive_token()
        if not token:
            raise RuntimeError("no token")
        fid = _find_file_id(token)
        if fid:
            raw = _drive_read(token, fid)
            if raw:
                return json.loads(raw)
    except Exception:
        pass
    try:
        with open("/tmp/medi_zones.json") as f:
            return json.loads(f.read())
    except Exception:
        pass
    try:
        raw = st.secrets.get("saved_zones", None)
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return {}


def save_zones(zones: dict):
    data = json.dumps(zones, ensure_ascii=False, indent=2).encode()
    try:
        with open("/tmp/medi_zones.json", "w") as f:
            f.write(data.decode())
    except Exception:
        pass
    try:
        token = _gdrive_token()
        if not token:
            return
        fid = _find_file_id(token)
        if fid:
            _drive_write(token, GDRIVE_FILENAME, data, fid)
        else:
            _create_file(token, data)
    except Exception:
        pass


def load_zones_from_all() -> dict:
    return load_zones()


def load_points() -> dict:
    return {}


def save_points(points: dict):
    pass


# ── WQI Snapshot ──────────────────────────────────────────────────────────────

SNAPSHOT_FILENAME = "medi_wqi_snapshot.json"


def load_snapshot() -> dict | None:
    try:
        token = _gdrive_token()
        if not token:
            return None
        fid = _drive_find(token, SNAPSHOT_FILENAME)
        if fid:
            raw = _drive_read(token, fid)
            if raw:
                return json.loads(raw)
    except Exception:
        pass
    try:
        with open("/tmp/medi_wqi_snapshot.json") as f:
            return json.loads(f.read())
    except Exception:
        return None


def save_snapshot(snapshot: dict):
    data = json.dumps(snapshot, ensure_ascii=False).encode()
    try:
        with open("/tmp/medi_wqi_snapshot.json", "w") as f:
            f.write(data.decode())
    except Exception:
        pass
    try:
        token = _gdrive_token()
        if not token:
            return
        fid = _drive_find(token, SNAPSHOT_FILENAME)
        _drive_write(token, SNAPSHOT_FILENAME, data, fid)
    except Exception:
        pass


# ── WQI Calibration ───────────────────────────────────────────────────────────

CALIBRATION_FILENAME = "medi_calibration.json"


def load_calibration() -> dict | None:
    try:
        token = _gdrive_token()
        if not token:
            return None
        fid = _drive_find(token, CALIBRATION_FILENAME)
        if fid:
            raw = _drive_read(token, fid)
            if raw:
                return json.loads(raw)
    except Exception:
        pass
    try:
        with open("/tmp/medi_calibration.json") as f:
            return json.loads(f.read())
    except Exception:
        return None


def save_calibration(calibration: dict):
    data = json.dumps(calibration, ensure_ascii=False, indent=2).encode()
    try:
        with open("/tmp/medi_calibration.json", "w") as f:
            f.write(data.decode())
    except Exception:
        pass
    try:
        token = _gdrive_token()
        if not token:
            return
        fid = _drive_find(token, CALIBRATION_FILENAME)
        _drive_write(token, CALIBRATION_FILENAME, data, fid)
    except Exception:
        pass


# ── WQI History (daily mean WQI log) ──────────────────────────────────────────

HISTORY_FILENAME = "medi_wqi_history.json"


def load_history() -> list:
    """Load WQI daily history. Returns list of dicts [{date, mean_wqi, valid_hex, source}]."""
    try:
        token = _gdrive_token()
        if not token:
            return []
        fid = _drive_find(token, HISTORY_FILENAME)
        if fid:
            raw = _drive_read(token, fid)
            if raw:
                return json.loads(raw)
    except Exception:
        pass
    try:
        with open("/tmp/medi_wqi_history.json") as f:
            return json.loads(f.read())
    except Exception:
        return []


def append_history(entry: dict):
    """
    Append one entry to WQI history. Entry format:
    {"date": "2026-06-22", "mean_wqi": 82.3, "median_wqi": 87.0,
     "valid_hex": 739, "source": "Sentinel-3 OLCI"}
    Deduplicates by date — only one entry per date kept (latest wins).
    Keeps last 365 entries.
    """
    history = load_history()

    # Remove existing entry for same date if exists
    date = entry.get("date", "")
    history = [h for h in history if h.get("date") != date]

    history.append(entry)
    history.sort(key=lambda x: x.get("date", ""))
    history = history[-365:]  # keep max 1 year

    data = json.dumps(history, ensure_ascii=False).encode()

    try:
        with open("/tmp/medi_wqi_history.json", "w") as f:
            f.write(data.decode())
    except Exception:
        pass

    try:
        token = _gdrive_token()
        if not token:
            return
        fid = _drive_find(token, HISTORY_FILENAME)
        _drive_write(token, HISTORY_FILENAME, data, fid)
    except Exception:
        pass
