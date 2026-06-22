"""
calibrate_wqi.py — Self-calibration v6
=======================================
No physical range filter — uses IQR only.
Logs raw band values for diagnostics.
"""

import ee, json, os, random, time
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

    img = coll.median()
    raw = img.select(['Oa08_radiance', 'Oa10_radiance', 'Oa11_radiance', 'Oa12_radiance'])

    with open("medi_h3_grid_final_913.geojson") as f:
        grid = json.load(f)
    sample_hex = random.sample(grid["features"], min(150, len(grid["features"])))
    log(f"Sampling {len(sample_hex)} hex centers...")

    samples = []
    logged_first = False
    for i, feat in enumerate(sample_hex):
        lat = feat["properties"]["lat"]
        lng = feat["properties"]["lng"]
        pt = ee.Geometry.Point([lng, lat]).buffer(500)
        try:
            vals = raw.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=pt,
                scale=300,
                bestEffort=True
            ).getInfo()
            oa08 = vals.get("Oa08_radiance")
            oa10 = vals.get("Oa10_radiance")
            oa11 = vals.get("Oa11_radiance")
            oa12 = vals.get("Oa12_radiance")
            # Log first valid sample for diagnostics
            if not logged_first and all(v is not None for v in [oa08, oa10, oa11, oa12]):
                log(f"Sample bands: Oa08={oa08:.1f} Oa10={oa10:.1f} Oa11={oa11:.1f} Oa12={oa12:.1f}")
                logged_first = True
            if all(v is not None for v in [oa08, oa10, oa11, oa12]):
                samples.append({"oa08": oa08, "oa10": oa10, "oa11": oa11, "oa12": oa12})
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            log(f"  Sampled {i+1}/{len(sample_hex)}... ({len(samples)} valid)")
            time.sleep(0.5)

    log(f"Valid samples: {len(samples)}")
    if len(samples) < 10:
        log("Not enough samples.")
        return None

    import numpy as np
    mci_values = []
    turb_values = []
    for s in samples:
        mci = s["oa11"] - (s["oa10"] + (s["oa12"] - s["oa10"]) * 0.39)
        mci_values.append(mci)
        turb_values.append(s["oa08"])

    mci_arr = np.array(mci_values)
    turb_arr = np.array(turb_values)

    log(f"MCI raw: min={mci_arr.min():.2f} max={mci_arr.max():.2f} median={np.median(mci_arr):.2f}")
    log(f"Turb raw: min={turb_arr.min():.2f} max={turb_arr.max():.2f} median={np.median(turb_arr):.2f}")

    # IQR outlier removal
    def iqr_filter(arr):
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        return arr[(arr >= q1 - 3*iqr) & (arr <= q3 + 3*iqr)]

    mci_clean = iqr_filter(mci_arr)
    turb_clean = iqr_filter(turb_arr)
    log(f"After IQR: MCI n={len(mci_clean)} [{mci_clean.min():.2f}, {mci_clean.max():.2f}]")
    log(f"After IQR: Turb n={len(turb_clean)} [{turb_clean.min():.2f}, {turb_clean.max():.2f}]")

    cal = {
        "generated_utc": datetime.utcnow().isoformat(),
        "sample_count": len(samples),
        "mci": {
            "p5":  round(float(np.percentile(mci_clean, 5)), 4),
            "p50": round(float(np.percentile(mci_clean, 50)), 4),
            "p95": round(float(np.percentile(mci_clean, 95)), 4),
            "unit_scale_min": round(float(np.percentile(mci_clean, 5)), 4),
            "unit_scale_max": round(float(np.percentile(mci_clean, 95)), 4),
        },
        "turbidity": {
            "p5":  round(float(np.percentile(turb_clean, 5)), 2),
            "p50": round(float(np.percentile(turb_clean, 50)), 2),
            "p95": round(float(np.percentile(turb_clean, 95)), 2),
            "unit_scale_min": round(float(np.percentile(turb_clean, 5)), 2),
            "unit_scale_max": round(float(np.percentile(turb_clean, 95)), 2),
        }
    }

    log(f"✅ MCI unitScale: [{cal['mci']['unit_scale_min']}, {cal['mci']['unit_scale_max']}]")
    log(f"✅ Turb unitScale: [{cal['turbidity']['unit_scale_min']}, {cal['turbidity']['unit_scale_max']}]")

    try:
        from storage import save_calibration
        save_calibration(cal)
        log("Saved to Google Drive.")
    except Exception:
        pass

    log("Calibration complete!")
    return cal


if __name__ == "__main__":
    run_calibration()
