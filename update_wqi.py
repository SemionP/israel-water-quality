"""
update_wqi.py — MEDI WQI snapshot via Sentinel-3 OLCI
======================================================
- NDWI removed from WQI (used only as water mask)
- WQI = (Chl_norm + Turb_norm) / 2 × 100
- Uses calibrated unitScale from calibration JSON
- Falls back to defaults if no calibration exists
- Appends daily mean WQI to history on each run
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

DEFAULT_MCI_MIN  = -2
DEFAULT_MCI_MAX  = 12
DEFAULT_TURB_MIN = 10
DEFAULT_TURB_MAX = 80


def load_cal():
    try:
        from storage import load_calibration
        cal = load_calibration()
        if cal:
            return {
                "mci_min":  cal["mci"]["unit_scale_min"],
                "mci_max":  cal["mci"]["unit_scale_max"],
                "turb_min": cal["turbidity"]["unit_scale_min"],
                "turb_max": cal["turbidity"]["unit_scale_max"],
            }
    except Exception:
        pass
    return {
        "mci_min": DEFAULT_MCI_MIN, "mci_max": DEFAULT_MCI_MAX,
        "turb_min": DEFAULT_TURB_MIN, "turb_max": DEFAULT_TURB_MAX,
    }


def build_s3_wqi(aoi, cal):
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

    # Mask fill values before band math
    def mask_fill(i):
        return i.updateMask(i.lt(10000))

    b10 = mask_fill(img.select('Oa10_radiance'))
    b11 = mask_fill(img.select('Oa11_radiance'))
    b12 = mask_fill(img.select('Oa12_radiance'))
    b08 = mask_fill(img.select('Oa08_radiance'))

    mci = b11.subtract(b10.add(b12.subtract(b10).multiply(0.39)))
    mci_n = ee.Image(1).subtract(
        mci.unitScale(cal["mci_min"], cal["mci_max"]).clamp(0, 1))

    turb_n = ee.Image(1).subtract(
        b08.unitScale(cal["turb_min"], cal["turb_max"]).clamp(0, 1))

    wqi = (mci_n.add(turb_n)
           .divide(2).multiply(100)
           .rename("WQI")
           .updateMask(wm)
           .clip(aoi))

    chl      = mci_n.multiply(100).rename("CHL").updateMask(wm)
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

    cal = load_cal()
    log(f"Calibration: MCI [{cal['mci_min']:.1f}, {cal['mci_max']:.1f}] | Turb [{cal['turb_min']:.1f}, {cal['turb_max']:.1f}]")

    with open(GRID_FILE) as f:
        grid = json.load(f)
    hexes = grid["features"]
    log(f"Loaded {len(hexes)} hexagons.")

    lats = [h["properties"]["lat"] for h in hexes]
    lngs = [h["properties"]["lng"] for h in hexes]
    aoi  = ee.Geometry.Rectangle([min(lngs)-0.1, min(lats)-0.1,
                                   max(lngs)+0.1, max(lats)+0.1])

    log("Building S-3 OLCI WQI mosaic (calibrated, no NDWI)...")
    img, age_hours = build_s3_wqi(aoi, cal)
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
        "calibrated": True,
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

    # ── Append to daily history ────────────────────────────────────────────────
    if valid:
        import statistics
        wqi_vals = [r["wqi"] for r in valid]
        mean_wqi   = round(sum(wqi_vals) / len(wqi_vals), 1)
        median_wqi = round(statistics.median(wqi_vals), 1)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        history_entry = {
            "date":       today,
            "mean_wqi":   mean_wqi,
            "median_wqi": median_wqi,
            "valid_hex":  len(valid),
            "source":     "Sentinel-3 OLCI",
        }
        try:
            from storage import append_history
            append_history(history_entry)
            log(f"History updated: {today} mean={mean_wqi} median={median_wqi}")
        except Exception as _e:
            log(f"History save failed: {_e}")

    return snapshot


if __name__ == "__main__":
    run_update()
