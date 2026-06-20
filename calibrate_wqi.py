"""
calibrate_wqi.py — Self-calibration of WQI formula
===================================================
Fixed: uses 30-day window, smaller AOI tiles, proper MCI computation
"""

import ee, json, os
from datetime import datetime, timedelta


def run_calibration(status_callback=None):
    def log(msg):
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    log("Initializing GEE...")
    from gee_processing import init_gee
    init_gee()

    # Israeli coastal waters (tight AOI, avoid land)
    aoi = ee.Geometry.Rectangle([34.2, 31.3, 34.9, 33.1])

    now = datetime.utcnow()
    end = ee.Date(now.strftime("%Y-%m-%d"))
    start = ee.Date((now - timedelta(days=30)).strftime("%Y-%m-%d"))

    log("Loading S-3 OLCI (last 30 days)...")
    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
            .filterBounds(aoi)
            .filterDate(start, end))

    count = coll.size().getInfo()
    log(f"Found {count} images.")
    if count == 0:
        log("No data.")
        return None

    # Use single recent image instead of median (cleaner)
    img = coll.sort("system:time_start", False).first()

    # Water mask
    wm = img.normalizedDifference(['Oa06_radiance', 'Oa17_radiance']).gt(0.1)

    # MCI
    b10 = img.select('Oa10_radiance')
    b11 = img.select('Oa11_radiance')
    b12 = img.select('Oa12_radiance')
    mci = b11.subtract(b10.add(b12.subtract(b10).multiply(0.39))).updateMask(wm).rename('mci')

    # Turbidity
    turb = img.select('Oa08_radiance').updateMask(wm).rename('turb')

    log("Computing percentiles...")

    # Sample points instead of reduceRegion (more reliable)
    combined = mci.addBands(turb)
    sample = combined.sample(
        region=aoi,
        scale=300,
        numPixels=5000,
        seed=42,
        geometries=False
    )

    # Get as lists
    mci_values = sample.aggregate_array('mci').getInfo()
    turb_values = sample.aggregate_array('turb').getInfo()

    log(f"Sampled {len(mci_values)} water pixels for MCI")
    log(f"Sampled {len(turb_values)} water pixels for Turb")

    if not mci_values or not turb_values:
        log("No valid water pixels found.")
        return None

    # Compute percentiles in Python (more reliable than GEE)
    import numpy as np
    mci_arr = np.array(mci_values)
    turb_arr = np.array(turb_values)

    cal = {
        "generated_utc": datetime.utcnow().isoformat(),
        "period": "30 days, single image",
        "sample_pixels": len(mci_values),
        "mci": {
            "p5":  round(float(np.percentile(mci_arr, 5)), 2),
            "p25": round(float(np.percentile(mci_arr, 25)), 2),
            "p50": round(float(np.percentile(mci_arr, 50)), 2),
            "p75": round(float(np.percentile(mci_arr, 75)), 2),
            "p95": round(float(np.percentile(mci_arr, 95)), 2),
            "unit_scale_min": round(float(np.percentile(mci_arr, 5)), 2),
            "unit_scale_max": round(float(np.percentile(mci_arr, 95)), 2),
        },
        "turbidity": {
            "p5":  round(float(np.percentile(turb_arr, 5)), 2),
            "p25": round(float(np.percentile(turb_arr, 25)), 2),
            "p50": round(float(np.percentile(turb_arr, 50)), 2),
            "p75": round(float(np.percentile(turb_arr, 75)), 2),
            "p95": round(float(np.percentile(turb_arr, 95)), 2),
            "unit_scale_min": round(float(np.percentile(turb_arr, 5)), 2),
            "unit_scale_max": round(float(np.percentile(turb_arr, 95)), 2),
        }
    }

    log(f"MCI range: {cal['mci']['unit_scale_min']} to {cal['mci']['unit_scale_max']}")
    log(f"Turb range: {cal['turbidity']['unit_scale_min']} to {cal['turbidity']['unit_scale_max']}")

    try:
        from storage import save_calibration
        save_calibration(cal)
        log("Saved to Google Drive.")
    except Exception:
        pass

    try:
        with open("medi_calibration.json", "w") as f:
            json.dump(cal, f, indent=2)
        log("Saved locally.")
    except Exception:
        pass

    log("Calibration complete!")
    return cal


if __name__ == "__main__":
    run_calibration()
