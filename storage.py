"""MEDI Platform — Zone Storage (Google Drive + fallback)"""

import json, os, time, urllib.request, urllib.parse, urllib.error
import streamlit as st

# =============================================================================
# Persistent Zone Storage — Google Drive (primary) + /tmp fallback
# =============================================================================
ZONES_KEY       = "medi-zones-v1"
GDRIVE_FILENAME = "medi_zones.json"
GDRIVE_FOLDER   = "1VU11P0UCzeMiVsn0k1RiIHuEu8bBLUFH"
GDRIVE_FILE_ID  = "1KTI_oRHIrvRJNtfZYrWMkhNi2D-34Y9v"

# -----------------------------------------------------------------------------
# Token — cached in session_state with expiry, NOT @st.cache_resource
# (cache_resource keeps the token forever; Drive tokens expire after 1 hour)
# -----------------------------------------------------------------------------
def _gdrive_token() -> str | None:
    """Return a valid OAuth2 access token, refreshing when within 5 min of expiry."""
    cache = st.session_state.setdefault("_gdrive_token_cache", {})
    now   = int(time.time())
    # Reuse cached token if it has more than 5 minutes left
    if cache.get("token") and cache.get("exp", 0) - now > 300:
        return cache["token"]
    # Fetch a fresh token
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


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _find_file_id(token: str) -> str | None:
    """Search Drive for medi_zones.json in the target folder."""
    q   = f"name='{GDRIVE_FILENAME}' and '{GDRIVE_FOLDER}' in parents and trashed=false"
    url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode({
        "q":                        q,
        "fields":                   "files(id)",
        "pageSize":                 "5",
        "supportsAllDrives":        "true",
        "includeItemsFromAllDrives":"true",
    })
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        res = json.loads(urllib.request.urlopen(req, timeout=10).read())
        files = res.get("files", [])
        return files[0]["id"] if files else None
    except Exception:
        return None


def _create_file(token: str, data: bytes) -> str | None:
    """Create medi_zones.json in the target folder. Returns new file ID or None."""
    # Step 1: create metadata
    meta = json.dumps({
        "name":    GDRIVE_FILENAME,
        "parents": [GDRIVE_FOLDER],
    }).encode()
    req1 = urllib.request.Request(
        "https://www.googleapis.com/drive/v3/files?supportsAllDrives=true",
        data=meta,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        },
    )
    try:
        res1  = json.loads(urllib.request.urlopen(req1, timeout=10).read())
        fid   = res1.get("id")
        if not fid:
            return None
        # Step 2: upload content
        url2  = f"https://www.googleapis.com/upload/drive/v3/files/{fid}?uploadType=media&supportsAllDrives=true"
        req2  = urllib.request.Request(
            url2, data=data, method="PATCH",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        urllib.request.urlopen(req2, timeout=15)
        return fid
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def load_zones() -> dict:
    """Load zones from Google Drive, falling back to /tmp then st.secrets."""
    try:
        token = _gdrive_token()
        if not token:
            raise RuntimeError("no token")
        fid = _find_file_id(token)
        if fid:
            req = urllib.request.Request(
                f"https://www.googleapis.com/drive/v3/files/{fid}?alt=media&supportsAllDrives=true",
                headers={"Authorization": f"Bearer {token}"},
            )
            raw = urllib.request.urlopen(req, timeout=10).read().decode()
            return json.loads(raw)
    except Exception:
        pass

    # /tmp fallback (survives within the same container session)
    try:
        with open("/tmp/medi_zones.json") as f:
            return json.loads(f.read())
    except Exception:
        pass

    # st.secrets fallback (manually pasted backup)
    try:
        raw = st.secrets.get("saved_zones", None)
        if raw:
            return json.loads(raw)
    except Exception:
        pass

    return {}


def save_zones(zones: dict):
    """
    Persist zones to:
      1. /tmp (always, instant)
      2. Google Drive (primary, durable)
         - Updates existing file if found
         - Creates new file if not found (handles first-time or missing file)
    """
    data = json.dumps(zones, ensure_ascii=False, indent=2).encode()

    # Always write to /tmp first (fast, no network)
    try:
        with open("/tmp/medi_zones.json", "w") as f:
            f.write(data.decode())
    except Exception:
        pass

    # Drive
    try:
        token = _gdrive_token()
        if not token:
            return

        fid = _find_file_id(token)

        if fid:
            # Update existing file
            url = f"https://www.googleapis.com/upload/drive/v3/files/{fid}?uploadType=media&supportsAllDrives=true"
            req = urllib.request.Request(
                url, data=data, method="PATCH",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type":  "application/json",
                },
            )
            urllib.request.urlopen(req, timeout=15)
        else:
            # File doesn't exist yet — create it
            _create_file(token, data)

    except Exception:
        pass  # Drive write failed; /tmp copy still intact for this session


def load_zones_from_all() -> dict:
    return load_zones()


def load_points() -> dict:
    return {}


def save_points(points: dict):
    pass
