"""
update_wqi.py — MEDI WQI snapshot via Sentinel-3 OLCI
======================================================
Computes WQI for 913 H3 R7 hexagons using S-3 OLCI (300m).
Saves result to Google Drive via storage.save_snapshot().

Called from app.py sidebar button OR run standalone:
  python update_wqi.py
"""

import ee, json, time, os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

GRID_FILE    = "medi_h3_grid_final_913.geojson"
BATCH_SIZE   = 50
MAX_WORKERS  = 4
LOOKBACK     = 5     # days back for S-3 composite
COVERAGE_MIN = 0.30  # min valid pixel fraction per hex
HEX_RADIUS   = 650   # metres (R7 inradius)
RETRY        = 3


def build_s3_wqi(aoi):
    """S-3 OLCI WQI image — same formula as gee_processing.process_israel_wqi"""
    now   = datetime.utcnow()
    end   = ee.Date(now.strftime("%Y-%m-%d")).advance(1, "day")
    start = ee.Date((now - timedelta(days=LOOKBACK)).strftime("%Y-%m-%d"))

    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
            .filterBounds(aoi)
            .filterDate(start, end)
            .sort("system:time_start", False))

    if coll.size().getInfo() == 0:
        return None, None

    img_ms    = coll.first().get("system:time_start").getInfo()
    img_dt    = datetime.utcfromtimestamp(img_ms / 1000)
    age_hours = (datetime.utcnow() - img_dt).total_seconds() / 3600

    img  = coll.median()
    # S-3 OLCI band indices (same as gee_processing.py)
    b6   = img.select("Oa06_radiance").divide(65535)   # ~560nm green
    b8   = img.select("Oa08_radiance").divide(65535)   # ~665nm red
    b11  = img.select("Oa11_radiance").divide(65535)   # ~709nm red-edge
    b17  = img.select("Oa17_radiance").divide(65535)   # ~865nm NIR

    # NDWI (water mask proxy)
    ndwi = b6.subtract(b17).divide(b6.add(b17).add(1e-6))
    wm   = ndwi.gt(0)  # water mask

    # WQI components (normalised 0-1, higher = better quality)
    ndwi_n = ndwi.unitScale(-0.2, 0.5).clamp(0, 1)
    mci_n  = ee.Image(1).subtract(
                b11.subtract(b8).unitScale(-2, 12).clamp(0, 1))  # MCI: low=good
    turb_n = ee.Image(1).subtract(
                b8.unitScale(0, 0.08).clamp(0, 1))               # turbidity: low=good

    wqi  = (ndwi_n.add(mci_n).add(turb_n)
            .divide(3).multiply(100)
            .rename("WQI").updateMask(wm).clip(aoi))
    chl  = mci_n.multiply(100).rename("CHL").updateMask(wm)
    turb = turb_n.multiply(100).rename("TURB").updateMask(wm)

    return wqi.addBands(chl).addBands(turb), round(age_hours, 1)


def sample_hex(img, feat):
    props = feat["properties"]
    lat, lng = props["lat"], props["lng"]
    pt = ee.Geometry.Point([lng, lat]).buffer(HEX_RADIUS)

    for attempt in range(RETRY):
        try:
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.count(), sharedInputs=True),
                geometry=pt, scale=300, bestEffort=True).getInfo()

            wqi  = stats.get("WQI_mean")
            chl  = stats.get("CHL_mean")
            turb = stats.get("TURB_mean")
            cnt  = stats.get("WQI_count") or 0
            nominal  = (3.14159 * HEX_RADIUS**2) / (300*300)
            coverage = min(1.0, cnt / nominal) if nominal else 0

            if coverage < COVERAGE_MIN or wqi is None:
                return {"hex_id": props["hex_id"], "wqi": None,
                        "chl": None, "turb": None,
                        "coverage": round(coverage, 2)}
            return {"hex_id": props["hex_id"],
                    "wqi":      round(wqi, 1),
                    "chl":      round(chl, 1) if chl else None,
                    "turb":     round(turb, 1) if turb else None,
                    "coverage": round(coverage, 2)}
        except Exception as e:
            if attempt == RETRY - 1:
                return {"hex_id": props["hex_id"], "wqi": None,
                        "chl": None, "turb": None, "coverage": 0,
                        "error": str(e)[:80]}
            time.sleep(1.5 * (attempt + 1))


def run_update(status_callback=None):
    """
    Main update function. Called from:
    - CLI: python update_wqi.py
    - Streamlit button: run_update(status_callback=st.write)
    Returns snapshot dict.
    """
    def log(msg):
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    log("Initializing GEE...")
    try:
        ee.Initialize()
    except Exception:
        sa  = os.environ.get("GEE_SERVICE_ACCOUNT")
        key = os.environ.get("GEE_KEY_FILE")
        if sa and key:
            ee.Initialize(ee.ServiceAccountCredentials(sa, key))
        else:
            raise

    with open(GRID_FILE) as f:
        grid = json.load(f)
    hexes = grid["features"]
    log(f"Loaded {len(hexes)} hexagons.")

    lats = [h["properties"]["lat"] for h in hexes]
    lngs = [h["properties"]["lng"] for h in hexes]
    aoi  = ee.Geometry.Rectangle([min(lngs)-0.1, min(lats)-0.1,
                                   max(lngs)+0.1, max(lats)+0.1])

    log("Building S-3 OLCI WQI mosaic...")
    img, age_hours = build_s3_wqi(aoi)
    if img is None:
        log("No S-3 data available. Aborting.")
        return None

    log(f"S-3 mosaic ready (age: {age_hours:.1f}h). Processing {len(hexes)} hex...")

    results = []
    n_batches = (len(hexes) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi in range(n_batches):
        batch = hexes[bi*BATCH_SIZE:(bi+1)*BATCH_SIZE]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            batch_res = list(ex.map(lambda h: sample_hex(img, h), batch))
        results.extend(batch_res)
        ok = sum(1 for r in batch_res if r["wqi"] is not None)
        log(f"  Batch {bi+1}/{n_batches}: {ok}/{len(batch)} valid ({time.time()-t0:.1f}s)")

    valid = [r for r in results if r["wqi"] is not None]
    log(f"Valid WQI: {len(valid)}/{len(results)} hexagons")

    snapshot = {
        "generated_utc": datetime.utcnow().isoformat(),
        "data_age_hours": age_hours,
        "source": "Sentinel-3 OLCI",
        "hex_count": len(results),
        "valid_count": len(valid),
        "hexes": results,
    }

    # Save
    try:
        from storage import save_snapshot
        save_snapshot(snapshot)
        log("Saved to Google Drive.")
    except Exception:
        # CLI fallback — save locally
        with open("medi_wqi_snapshot.json", "w") as f:
            json.dump(snapshot, f)
        log("Saved locally: medi_wqi_snapshot.json")

    return snapshot


if __name__ == "__main__":
    run_update()
