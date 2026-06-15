"""
update_wqi.py — MEDI WQI snapshot via Sentinel-3 OLCI
======================================================
Uses SAME formula as gee_processing.process_israel_wqi (line 329)
to ensure consistent WQI values across the platform.
"""

import ee, json, time, os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

GRID_FILE    = "medi_h3_grid_final_913.geojson"
BATCH_SIZE   = 50
MAX_WORKERS  = 4
LOOKBACK     = 5
COVERAGE_MIN = 0.30
HEX_RADIUS   = 650
RETRY        = 3


def build_s3_wqi(aoi):
    """S-3 OLCI WQI — identical formula to gee_processing.process_israel_wqi"""
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

    img = coll.median()

    # Water mask
    wm = img.normalizedDifference(['Oa06_radiance', 'Oa17_radiance']).gt(0)

    # NDWI — same as process_israel_wqi
    ndwi = img.normalizedDifference(['Oa06_radiance', 'Oa17_radiance'])
    ndwi_n = ndwi.unitScale(-0.2, 0.5).clamp(0, 1)

    # MCI (Maximum Chlorophyll Index) — same formula
    b10 = img.select('Oa10_radiance')
    b11 = img.select('Oa11_radiance')
    b12 = img.select('Oa12_radiance')
    mci = b11.subtract(b10.add(b12.subtract(b10).multiply((708.75 - 681.25) / (753.75 - 681.25))))
    mci_n = ee.Image(1).subtract(mci.unitScale(-2, 12).clamp(0, 1))

    # Turbidity — same as process_israel_wqi (NO divide by 65535)
    turb = img.select('Oa08_radiance')
    turb_n = ee.Image(1).subtract(turb.unitScale(10, 80).clamp(0, 1))

    # WQI composite
    wqi = (ndwi_n.add(mci_n).add(turb_n)
           .divide(3).multiply(100)
           .rename("WQI")
           .updateMask(wm)
           .clip(aoi))

    # Also export individual components
    chl  = mci_n.multiply(100).rename("CHL").updateMask(wm)
    turb_out = turb_n.multiply(100).rename("TURB").updateMask(wm)

    return wqi.addBands(chl).addBands(turb_out), round(age_hours, 1)


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
    def log(msg):
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    log("Initializing GEE...")
    from gee_processing import init_gee
    init_gee()

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
        log("No S-3 data available.")
        return None

    log(f"Mosaic ready ({age_hours:.1f}h old). Processing {len(hexes)} hex...")

    results = []
    n_batches = (len(hexes) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi in range(n_batches):
        batch = hexes[bi*BATCH_SIZE:(bi+1)*BATCH_SIZE]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            batch_res = list(ex.map(lambda h: sample_hex(img, h), batch))
        results.extend(batch_res)
        ok = sum(1 for r in batch_res if r["wqi"] is not None)
        log(f"Batch {bi+1}/{n_batches}: {ok}/{len(batch)} valid ({time.time()-t0:.1f}s)")

    valid = [r for r in results if r["wqi"] is not None]
    log(f"Done: {len(valid)}/{len(results)} hex valid")

    snapshot = {
        "generated_utc": datetime.utcnow().isoformat(),
        "data_age_hours": age_hours,
        "source": "Sentinel-3 OLCI",
        "hex_count": len(results),
        "valid_count": len(valid),
        "hexes": results,
    }

    try:
        from storage import save_snapshot
        save_snapshot(snapshot)
        log("Saved to Google Drive.")
    except Exception as _e:
        log(f"Drive save failed: {_e}")

    return snapshot


if __name__ == "__main__":
    run_update()
