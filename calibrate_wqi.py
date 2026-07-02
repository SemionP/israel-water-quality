"""
calibrate_wqi.py — Self-calibration v5
=======================================
Samples raw S-3 OLCI bands at hex centers,
computes MCI and Turbidity in Python (avoids GEE band math issues).

v4 changes (fix for Chl-a ~100 everywhere bug):
- Sample from the SAME median composite that update_wqi.py uses for
  production (was: coll.first(), a single raw scene) — calibration
  and production must see the same distribution.
- Apply the same NDWI water mask before sampling raw bands.

v5 changes (fix for MCI raw min = -12,792,597 bug):
- Mask S-3 OLCI fill values (2^22 = 4,194,304) on EACH image in the
  collection BEFORE building the median composite, not after. This
  is the same confirmed-working pattern already used elsewhere in
  this project: coll.map(mask_fill).median(). Without this, a single
  fill-value pixel anywhere in the 5-day window corrupts the whole
  median composite at that location, and any hex sample that lands
  on it drags the percentile range down to millions of units off —
  which is what collapsed unitScale and made CHL clamp to 100
  everywhere in production.
"""

import ee, json, os, random, time
from datetime import datetime, timedelta

LOOKBACK = 5  # must match update_wqi.py LOOKBACK
FILL_THRESHOLD = 10000  # S-3 OLCI fill value is 2^22; anything sane is well under this


def mask_fill(img):
    """Mask S-3 OLCI fill-value pixels. Must run per-image, before median()."""
    return img.updateMask(img.lt(FILL_THRESHOLD))


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
    end = ee.Date(now.strftime("%Y-%m-%d")).advance(1, "day")
    start = ee.Date((now - timedelta(days=LOOKBACK)).strftime("%Y-%m-%d"))
    aoi = ee.Geometry.Rectangle([34.0, 31.0, 35.2, 33.4])

    log(f"Loading S-3 OLCI (last {LOOKBACK} days, fill-masked median composite)...")
    coll = (ee.ImageCollection("COPERNICUS/S3/OLCI")
            .filterBounds(aoi)
            .filterDate(start, end)
            .sort("system:time_start", False))

    count = coll.size().getInfo()
    log(f"Found {count} images.")
    if count == 0:
        log("No data.")
        return None

    # v5: mask fill values per-image, BEFORE the median composite
    # v4: median composite, same as production build_s3_wqi()
    img = coll.map(mask_fill).median()

    # v4: water mask, same NDWI test as production, applied BEFORE
    # sampling raw bands
    wm = img.normalizedDifference(['Oa06_radiance', 'Oa17_radiance']).gt(0)

    # Select raw bands — no band math in GEE
    raw = img.select(['Oa08_radiance', 'Oa10_radiance', 'Oa11_radiance', 'Oa12_radiance']).updateMask(wm)

    # Load hex grid, pick 150 random centers
    with open("medi_h3_grid_final_913.geojson") as f:
        grid = json.load(f)
    sample_hex = random.sample(grid["features"], min(150, len(grid["features"])))
    log(f"Sampling {len(sample_hex)} hex centers (raw bands, fill+water-masked)...")

    samples = []
    for i, feat in enumerate(sample_hex):
        lat = feat["properties"]["lat"]
        lng = feat["properties"]["lng"]
        pt = ee.Geometry.Point([lng, lat])
        try:
            vals = raw.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=pt,
                scale=300
            ).getInfo()
            oa08 = vals.get("Oa08_radiance")
            oa10 = vals.get("Oa10_radiance")
            oa11 = vals.get("Oa11_radiance")
            oa12 = vals.get("Oa12_radiance")
            if all(v is not None for v in [oa08, oa10, oa11, oa12]):
                samples.append({"oa08": oa08, "oa10": oa10, "oa11": oa11, "oa12": oa12})
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            log(f"  Sampled {i+1}/{len(sample_hex)}... ({len(samples)} valid)")
            time.sleep(0.5)

    log(f"Valid samples: {len(samples)} (masked out: {len(sample_hex) - len(samples)})")
    if len(samples) < 10:
        log("Not enough samples (too many hex centers fell outside water/fill mask or had no data).")
        return None

    # Compute MCI and Turbidity in Python
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

    # v5 sanity check: fill masking should keep raw values in a
    # physically plausible radiance range. If this still fires, the
    # fill mask isn't catching everything and needs a closer look.
    if abs(mci_arr.min()) > 1000 or abs(turb_arr.min()) > 1000:
        log("⚠️ WARNING: raw values still look like fill-value contamination "
            "(magnitude > 1000). Do not trust this calibration — investigate "
            "before saving.")

    cal = {
        "generated_utc": datetime.utcnow().isoformat(),
        "sample_count": len(samples),
        "source_composite": f"median_{LOOKBACK}day_fill_and_water_masked",
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
