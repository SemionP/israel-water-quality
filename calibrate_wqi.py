"""
calibrate_wqi.py — Self-calibration of WQI formula for Israeli EEZ
===================================================================
Computes percentile 5% and 95% for MCI and Turbidity bands
from S-3 OLCI data over the past year. Saves calibration JSON
that update_wqi.py uses for unitScale bounds.
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

    # Israeli EEZ bounding box
    aoi = ee.Geometry.Rectangle([34.0, 31.0, 35.2, 33.4])

    now = datetime.utcnow()
    end = ee.Date(now.strftime("%Y-%m-%d"))
    start = ee.Date((now - timedelta(days=365)).strftime("%Y-%m-%d"))

    log("Loading S-3 OLCI data for past 365 days...")
    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
            .filterBounds(aoi)
            .filterDate(start, end))

    count = coll.size().getInfo()
    log(f"Found {count} images.")

    if count == 0:
        log("No data available.")
        return None

    img = coll.median()

    # Water mask
    wm = img.normalizedDifference(['Oa06_radiance', 'Oa17_radiance']).gt(0)

    # MCI
    b10 = img.select('Oa10_radiance')
    b11 = img.select('Oa11_radiance')
    b12 = img.select('Oa12_radiance')
    mci = b11.subtract(b10.add(b12.subtract(b10).multiply(0.39))).updateMask(wm)

    # Turbidity (Oa08)
    turb = img.select('Oa08_radiance').updateMask(wm)

    log("Computing percentiles (this may take 1-2 minutes)...")

    # Compute percentiles
    mci_stats = mci.rename('mci').reduceRegion(
        reducer=ee.Reducer.percentile([5, 25, 50, 75, 95]),
        geometry=aoi,
        scale=1000,
        bestEffort=True,
        maxPixels=1e8
    ).getInfo()

    turb_stats = turb.rename('turb').reduceRegion(
        reducer=ee.Reducer.percentile([5, 25, 50, 75, 95]),
        geometry=aoi,
        scale=1000,
        bestEffort=True,
        maxPixels=1e8
    ).getInfo()

    log(f"MCI stats: {mci_stats}")
    log(f"Turb stats: {turb_stats}")

    calibration = {
        "generated_utc": datetime.utcnow().isoformat(),
        "period": "365 days",
        "images_used": count,
        "mci": {
            "p5": mci_stats.get("mci_p5"),
            "p25": mci_stats.get("mci_p25"),
            "p50": mci_stats.get("mci_p50"),
            "p75": mci_stats.get("mci_p75"),
            "p95": mci_stats.get("mci_p95"),
            "unit_scale_min": mci_stats.get("mci_p5"),
            "unit_scale_max": mci_stats.get("mci_p95"),
        },
        "turbidity": {
            "p5": turb_stats.get("turb_p5"),
            "p25": turb_stats.get("turb_p25"),
            "p50": turb_stats.get("turb_p50"),
            "p75": turb_stats.get("turb_p75"),
            "p95": turb_stats.get("turb_p95"),
            "unit_scale_min": turb_stats.get("turb_p5"),
            "unit_scale_max": turb_stats.get("turb_p95"),
        }
    }

    # Save to Google Drive
    try:
        from storage import save_calibration
        save_calibration(calibration)
        log("Saved calibration to Google Drive.")
    except Exception:
        pass

    # Also save locally
    try:
        with open("medi_calibration.json", "w") as f:
            json.dump(calibration, f, indent=2)
        log("Saved locally: medi_calibration.json")
    except Exception:
        pass

    log(f"Calibration complete!")
    log(f"MCI range: {calibration['mci']['unit_scale_min']:.2f} to {calibration['mci']['unit_scale_max']:.2f}")
    log(f"Turb range: {calibration['turbidity']['unit_scale_min']:.1f} to {calibration['turbidity']['unit_scale_max']:.1f}")

    return calibration


if __name__ == "__main__":
    run_calibration()
