"""
calibrate_wqi.py — Self-calibration using hex centers
======================================================
Samples MCI and Turbidity at 200 random hex centers
(already confirmed as sea-only locations).
"""

import ee, json, os, random
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

    now = datetime.utcnow()
    end = ee.Date(now.strftime("%Y-%m-%d"))
    start = ee.Date((now - timedelta(days=10)).strftime("%Y-%m-%d"))

    aoi = ee.Geometry.Rectangle([34.0, 31.0, 35.2, 33.4])

    log("Loading S-3 OLCI (last 10 days)...")
    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
            .filterBounds(aoi)
            .filterDate(start, end)
            .sort("system:time_start", False))

    count = coll.size().getInfo()
    log(f"Found {count} images.")
    if count == 0:
        log("No data.")
        return None

    img = coll.first()

    # MCI
    b10 = img.select('Oa10_radiance')
    b11 = img.select('Oa11_radiance')
    b12 = img.select('Oa12_radiance')
    mci = b11.subtract(b10.add(b12.subtract(b10).multiply(0.39))).rename('mci')

    # Turbidity
    turb = img.select('Oa08_radiance').rename('turb')

    # Load hex grid and pick 200 random centers
    with open("medi_h3_grid_final_913.geojson") as f:
        grid = json.load(f)
    all_hex = grid["features"]
    sample_hex = random.sample(all_hex, min(200, len(all_hex)))
    log(f"Sampling {len(sample_hex)} hex centers...")

    mci_values = []
    turb_values = []
    errors = 0

    for i, feat in enumerate(sample_hex):
        lat = feat["properties"]["lat"]
        lng = feat["properties"]["lng"]
        pt = ee.Geometry.Point([lng, lat])
        try:
            vals = mci.addBands(turb).reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=pt,
                scale=300
            ).getInfo()
            m = vals.get("mci")
            t = vals.get("turb")
            if m is not None and t is not None:
                mci_values.append(m)
                turb_values.append(t)
        except Exception:
            errors += 1

        if (i + 1) % 50 == 0:
            log(f"  Sampled {i+1}/{len(sample_hex)}...")

    log(f"Got {len(mci_values)} valid samples ({errors} errors)")

    if len(mci_values) < 10:
        log("Not enough valid samples.")
        return None

    import numpy as np
    mci_arr = np.array(mci_values)
    turb_arr = np.array(turb_values)

    log(f"MCI raw range: {mci_arr.min():.2f} to {mci_arr.max():.2f}")
    log(f"Turb raw range: {turb_arr.min():.2f} to {turb_arr.max():.2f}")

    cal = {
        "generated_utc": datetime.utcnow().isoformat(),
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

    log(f"MCI unitScale: [{cal['mci']['unit_scale_min']}, {cal['mci']['unit_scale_max']}]")
    log(f"Turb unitScale: [{cal['turbidity']['unit_scale_min']}, {cal['turbidity']['unit_scale_max']}]")

    try:
        from storage import save_calibration
        save_calibration(cal)
        log("Saved to Google Drive.")
    except Exception:
        pass

    try:
        with open("medi_calibration.json", "w") as f:
            json.dump(cal, f, indent=2)
    except Exception:
        pass

    log("Calibration complete!")
    return cal


if __name__ == "__main__":
    run_calibration()
