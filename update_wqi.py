"""
update_wqi.py — MEDI pre-computed WQI snapshot generator (H3 grid)
===================================================================
Computes WQI for each of the 913 H3 R7 hexagons over Israeli
Mediterranean territorial waters, then saves a JSON snapshot that
the Streamlit app loads instantly (no live GEE calls at view time).

Run:  python update_wqi.py
Output: medi_wqi_snapshot.json  (hex_id -> wqi, chl, turb, coverage, ts)

Design (agreed in planning):
- Pre-computed snapshot model (manual run now, cron later)
- Batched GEE requests (50 hex/batch) with retry
- Coverage threshold: <30% valid pixels -> wqi=None
- Temporal fallback handled at merge time (keeps last good value)
- Reuses the S-2 WQI formula from gee_processing.py
"""

import ee
import json
import time
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
GRID_FILE       = "medi_h3_grid_final_913.geojson"
SNAPSHOT_FILE   = "medi_wqi_snapshot.json"
BATCH_SIZE      = 50          # hex per GEE batch
MAX_WORKERS     = 4          # parallel point samples within a batch
LOOKBACK_DAYS   = 10         # S-2 mosaic window
CLOUD_MAX       = 30         # max cloudy pixel %
COVERAGE_MIN    = 0.30       # min fraction of valid pixels in hex
HEX_SAMPLE_M    = 650        # sample radius (m) ~ R7 hex inradius
RETRY           = 3


# ----------------------------------------------------------------------
# GEE init (reuses service-account auth from environment)
# ----------------------------------------------------------------------
def init_gee():
    """Initialize Earth Engine. Expects GEE creds already configured
    (service account via env, same as the Streamlit app)."""
    try:
        ee.Initialize()
    except Exception:
        # Fall back to service account if available
        sa = os.environ.get("GEE_SERVICE_ACCOUNT")
        key = os.environ.get("GEE_KEY_FILE")
        if sa and key:
            creds = ee.ServiceAccountCredentials(sa, key)
            ee.Initialize(creds)
        else:
            raise


# ----------------------------------------------------------------------
# WQI image (Sentinel-2, 10m) — same formula as gee_processing.process_israel_s2
# ----------------------------------------------------------------------
def build_wqi_image(aoi):
    """Return (wqi_image, age_hours) for the most recent S-2 mosaic."""
    now   = datetime.utcnow()
    end   = ee.Date(now.strftime("%Y-%m-%d")).advance(1, "day")
    start = ee.Date((now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d"))

    coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_MAX))
            .sort("system:time_start", False))

    if coll.size().getInfo() == 0:
        return None, None

    img_time_ms = coll.first().get("system:time_start").getInfo()
    img_dt      = datetime.utcfromtimestamp(img_time_ms / 1000)
    age_hours   = (datetime.utcnow() - img_dt).total_seconds() / 3600

    img = coll.mosaic()
    b3  = img.select("B3").divide(10000)
    b4  = img.select("B4").divide(10000)
    b5  = img.select("B5").divide(10000)
    b8  = img.select("B8").divide(10000)
    b8a = img.select("B8A").divide(10000)

    ndwi_raw = b3.subtract(b8).divide(b3.add(b8).add(1e-6))
    ndwi_n   = ndwi_raw.unitScale(-0.3, 0.5).clamp(0, 1)
    chl_n    = b5.divide(b4.add(1e-6)).unitScale(1.0, 3.5).clamp(0, 1)
    turb_n   = ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0, 0.15)).clamp(0, 1)

    wqi = (ndwi_n.add(chl_n).add(turb_n)
           .divide(3).multiply(100)
           .clip(aoi)
           .rename("WQI")
           .updateMask(ndwi_raw.gt(-0.1)))

    # Also expose chl & turb proxies for per-hex reporting
    chl  = chl_n.multiply(100).rename("CHL").updateMask(ndwi_raw.gt(-0.1))
    turb = turb_n.multiply(100).rename("TURB").updateMask(ndwi_raw.gt(-0.1))

    return wqi.addBands(chl).addBands(turb), round(age_hours, 1)


# ----------------------------------------------------------------------
# Per-hex sampling
# ----------------------------------------------------------------------
def sample_hex(img, hex_feat):
    """Sample WQI/CHL/TURB at the hex center. Returns dict."""
    props = hex_feat["properties"]
    lat, lng = props["lat"], props["lng"]
    pt = ee.Geometry.Point([lng, lat]).buffer(HEX_SAMPLE_M)

    for attempt in range(RETRY):
        try:
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.count(), sharedInputs=True),
                geometry=pt, scale=10, bestEffort=True).getInfo()

            wqi  = stats.get("WQI_mean")
            chl  = stats.get("CHL_mean")
            turb = stats.get("TURB_mean")
            cnt  = stats.get("WQI_count") or 0

            # Coverage: count of valid pixels vs nominal pixels in buffer
            nominal = (3.14159 * HEX_SAMPLE_M**2) / (10*10)
            coverage = min(1.0, cnt / nominal) if nominal else 0

            if coverage < COVERAGE_MIN or wqi is None:
                return {"hex_id": props["hex_id"], "wqi": None,
                        "chl": None, "turb": None,
                        "coverage": round(coverage, 2)}

            return {"hex_id": props["hex_id"],
                    "wqi": round(wqi, 1),
                    "chl": round(chl, 1) if chl else None,
                    "turb": round(turb, 1) if turb else None,
                    "coverage": round(coverage, 2)}
        except Exception as e:
            if attempt == RETRY - 1:
                return {"hex_id": props["hex_id"], "wqi": None,
                        "chl": None, "turb": None, "coverage": 0,
                        "error": str(e)[:80]}
            time.sleep(1.5 * (attempt + 1))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("MEDI WQI snapshot generator")
    print("=" * 50)

    init_gee()
    print("GEE initialized.")

    with open(GRID_FILE) as f:
        grid = json.load(f)
    hexes = grid["features"]
    print(f"Loaded {len(hexes)} hexagons from {GRID_FILE}")

    # Build WQI image over the full grid bbox
    lats = [h["properties"]["lat"] for h in hexes]
    lngs = [h["properties"]["lng"] for h in hexes]
    aoi = ee.Geometry.Rectangle([min(lngs)-0.1, min(lats)-0.1,
                                  max(lngs)+0.1, max(lats)+0.1])

    print("Building S-2 WQI mosaic...")
    img, age_hours = build_wqi_image(aoi)
    if img is None:
        print("No S-2 data available in window. Aborting.")
        return
    print(f"WQI image ready (data age: {age_hours}h)")

    # Process in batches
    results = []
    n_batches = (len(hexes) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi in range(n_batches):
        batch = hexes[bi*BATCH_SIZE:(bi+1)*BATCH_SIZE]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            batch_res = list(ex.map(lambda h: sample_hex(img, h), batch))
        results.extend(batch_res)
        ok = sum(1 for r in batch_res if r["wqi"] is not None)
        print(f"  Batch {bi+1}/{n_batches}: {ok}/{len(batch)} valid "
              f"({time.time()-t0:.1f}s)")

    # Stats
    valid = [r for r in results if r["wqi"] is not None]
    print(f"\nDone: {len(valid)}/{len(results)} hex with valid WQI")

    # Temporal fallback: merge with previous snapshot
    if os.path.exists(SNAPSHOT_FILE):
        with open(SNAPSHOT_FILE) as f:
            prev = json.load(f)
        prev_map = {h["hex_id"]: h for h in prev.get("hexes", [])}
        merged = 0
        for r in results:
            if r["wqi"] is None and r["hex_id"] in prev_map:
                old = prev_map[r["hex_id"]]
                if old.get("wqi") is not None:
                    r["wqi"]    = old["wqi"]
                    r["chl"]    = old.get("chl")
                    r["turb"]   = old.get("turb")
                    r["stale"]  = True
                    merged += 1
        print(f"Temporal fallback: {merged} hex filled from previous snapshot")

    # Save
    snapshot = {
        "generated_utc": datetime.utcnow().isoformat(),
        "data_age_hours": age_hours,
        "source": "Sentinel-2",
        "hex_count": len(results),
        "valid_count": len([r for r in results if r["wqi"] is not None]),
        "hexes": results,
    }
    with open(SNAPSHOT_FILE, "w") as f:
        json.dump(snapshot, f)
    print(f"Saved snapshot -> {SNAPSHOT_FILE} "
          f"({os.path.getsize(SNAPSHOT_FILE)//1024} KB)")


if __name__ == "__main__":
    main()
