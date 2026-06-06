"""
s1_processing.py - MEDI Platform
Sentinel-1 SAR: oil spill detection, vessel detection

Scope: Israeli Mediterranean coast only.
Excluded: Kinneret, Dead Sea, Red Sea / Gulf of Eilat.
"""

import math
from datetime import datetime, timedelta

import streamlit as st
import ee
from geometry_constants import (
    MED_BBOX_COORDS, get_med_bbox, get_israel_med_sea,
    sea_mask as _sea_mask, AOI_PRESETS,
)

# Default AOI — Israeli Mediterranean only (no Kinneret / Dead Sea / Red Sea)
MED_SEA_BOX = get_med_bbox()


@st.cache_data(ttl=7200)
def _get_s1_img_cached(date_str: str, aoi_coords: list, days_back: int = 6):
    """
    Fetch and cache S1 GRD mosaic for the given AOI and date window.
    Returns (image_or_None, actual_date_str_or_None, age_hours).
    aoi_coords: [lon_min, lat_min, lon_max, lat_max]
    """
    aoi = ee.Geometry.Rectangle(aoi_coords)
    t   = ee.Date(date_str)

    coll = (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(t.advance(-days_back, "day"), t.advance(1, "day"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .sort("system:time_start", False))

    # Use aggregate to avoid size().getInfo() round-trip
    ts_list = coll.aggregate_array("system:time_start").getInfo()
    if not ts_list:
        return None, None, None

    actual_ts  = ts_list[0]
    actual_dt  = datetime.utcfromtimestamp(actual_ts / 1000)
    actual_date = actual_dt.strftime("%Y-%m-%d")
    age_h      = (datetime.utcnow() - actual_dt).total_seconds() / 3600

    img = coll.mosaic().reproject(crs="EPSG:4326", scale=10)
    return img, actual_date, round(age_h, 1)


@st.cache_data(ttl=14400)
def get_available_s1_dates(days_back: int = 30) -> list:
    """List available S1 acquisition dates over the Israeli Mediterranean."""
    end   = datetime.utcnow()
    start = end - timedelta(days=days_back)
    try:
        coll = (ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(MED_SEA_BOX)
                .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                .filter(ee.Filter.eq("instrumentMode", "IW")))

        ts_list = coll.aggregate_array("system:time_start").getInfo()
        orbits  = coll.aggregate_array("orbitProperties_pass").getInfo()

        results = []
        seen    = set()
        for ts, orb in sorted(zip(ts_list, orbits), reverse=True):
            d   = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            key = f"{d}_{orb}"
            if key not in seen:
                seen.add(key)
                age_h = (datetime.utcnow() - datetime.utcfromtimestamp(ts / 1000)).total_seconds() / 3600
                results.append({
                    "date":  d,
                    "orbit": orb,
                    "age_h": round(age_h, 1),
                    "ts":    ts,
                })
        return results
    except Exception:
        return []


@st.cache_data(ttl=7200)
def get_s1_layers(date_str: str, aoi_coords: list = None) -> dict:
    """Return GEE tile URLs for SAR visualisation layers."""
    _aoi_coords = aoi_coords or [33.0, 31.2, 35.0, 33.15]
    result = {}
    try:
        img, _, _ = _get_s1_img_cached(date_str, _aoi_coords)
        if img is None:
            return result

        vv    = img.select("VV")
        vh    = img.select("VH")
        ratio = vv.subtract(vh)

        mid = vv.getMapId({"min": -25, "max": 0,
                           "palette": ["#000014","#0a1520","#152840","#1e3a5a","#5aaacf","#c8e8f8"]})
        result["vv"] = mid["tile_fetcher"].url_format

        mid = ratio.getMapId({"min": 0, "max": 15,
                              "palette": ["#041e33","#1D9E75","#fdae61","#d73027"]})
        result["ratio"] = mid["tile_fetcher"].url_format

    except Exception:
        pass
    return result


@st.cache_data(ttl=7200)
def detect_oil_spills(date_str: str, aoi_coords: list = None,
                      days_back: int = 6,
                      vv_threshold: float = -16.0,
                      min_area_m2: int = 20000,
                      max_area_m2: int = 80000000) -> dict:
    """
    Detect oil spill candidates in a Sentinel-1 SAR image.

    Algorithm (standard EMSA/ESA approach):
    1. Dark spot: VV < vv_threshold  (oil dampens Bragg scattering)
    2. Low cross-pol ratio: VV - VH < 10 dB  (oil suppresses both, unlike wind shadows)
    3. Sea mask: JRC permanent water only
    4. Size filter: min_area_m2 to max_area_m2

    Parameters
    ----------
    date_str     : target date (searches ±days_back)
    aoi_coords   : [lon_min, lat_min, lon_max, lat_max]
    days_back    : look-back window in days
    vv_threshold : detection threshold in dB (default -16 is standard for Med Sea)
    min_area_m2  : minimum polygon area (removes noise)
    max_area_m2  : maximum polygon area (removes cloud/land confusion)
    """
    _aoi_coords = aoi_coords or [33.0, 31.2, 35.0, 33.15]
    aoi = ee.Geometry.Rectangle(_aoi_coords)
    wm  = _sea_mask()

    result = {
        "polygons":      [],
        "tile_url":      None,
        "tile_url_raw":  None,
        "total_area_km2": 0,
        "n_anomalies":   0,
        "actual_date":   None,
        "age_h":         None,
        "error":         None,
    }

    try:
        img, actual_date, age_h = _get_s1_img_cached(date_str, _aoi_coords, days_back)
        if img is None:
            result["error"] = f"No S1 data found within {days_back} days of {date_str}"
            return result

        result["actual_date"] = actual_date
        result["age_h"]       = age_h

        vv = img.select("VV")
        vh = img.select("VH")

        # ── Detection mask ────────────────────────────────────────────────
        # Criterion 1: dark spot (oil dampens backscatter)
        dark = vv.lt(vv_threshold)

        # Criterion 2: low cross-pol difference (distinguishes oil from wind shadows)
        # Wind shadows: VV low but VH also very low → ratio small
        # Oil: VV low but VH slightly less suppressed → ratio moderate
        # Using VV-VH < 10 follows EMSA OSN guidelines
        ratio_ok = vv.subtract(vh).lt(10.0)

        oil_mask = dark.And(ratio_ok).And(wm).clip(aoi)

        # ── Tile URL for map overlay ──────────────────────────────────────
        mid = oil_mask.updateMask(oil_mask).getMapId(
            {"min": 0, "max": 1, "palette": ["#ff4444"]})
        result["tile_url"] = mid["tile_fetcher"].url_format

        # Raw VV layer for context
        mid_raw = vv.getMapId(
            {"min": -25, "max": 0,
             "palette": ["#000014","#0a1520","#152840","#1e3a5a","#5aaacf","#c8e8f8"]})
        result["tile_url_raw"] = mid_raw["tile_fetcher"].url_format

        # ── Vectorise ─────────────────────────────────────────────────────
        vectors = (oil_mask
                   .updateMask(oil_mask)
                   .reduceToVectors(
                       geometry=aoi,
                       scale=40,           # 40m — faster than 10m, fine for oil polygons
                       maxPixels=1e8,
                       bestEffort=True,
                       geometryType="polygon",
                       eightConnected=True)
                   .map(lambda f: f.set("area_m2", f.geometry().area(50))))

        vectors = vectors.filter(ee.Filter.And(
            ee.Filter.gt("area_m2", min_area_m2),
            ee.Filter.lt("area_m2", max_area_m2),
        ))

        feats = vectors.getInfo().get("features", [])

        total_area = 0.0
        for f in feats[:15]:   # cap at 15 polygons
            geom   = f.get("geometry", {})
            props  = f.get("properties", {})
            area_m2 = props.get("area_m2", 0) or 0
            area_km2 = area_m2 / 1e6

            coords = geom.get("coordinates", [[]])[0]
            if not coords:
                continue

            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            clon = sum(lons) / len(lons)
            clat = sum(lats) / len(lats)

            # Sanity: centroid must be in sea (west of Israeli coast)
            if clon > 35.15 or clon < 32.0:
                continue

            # Confidence based on area and VV intensity
            try:
                vv_mean_val = (vv
                               .reduceRegion(
                                   reducer=ee.Reducer.mean(),
                                   geometry=f.get("geometry"),
                                   scale=100,
                                   bestEffort=True)
                               .get("VV").getInfo())
            except Exception:
                vv_mean_val = None

            if vv_mean_val is not None and vv_mean_val < -20:
                conf = "High"
            elif area_km2 > 0.5:
                conf = "High"
            elif area_km2 > 0.1:
                conf = "Medium"
            else:
                conf = "Low"

            result["polygons"].append({
                "id":          f"OIL-{len(result['polygons']) + 1}",
                "area_km2":    round(area_km2, 3),
                "area_km2_min": round(area_km2 * 0.7, 3),
                "area_km2_max": round(area_km2 * 1.3, 3),
                "confidence":  conf,
                "lat":         round(clat, 4),
                "lon":         round(clon, 4),
                "coords":      coords,
                "vv_mean_db":  round(vv_mean_val, 1) if vv_mean_val else None,
                "date":        actual_date,
                "age_h":       age_h,
            })
            total_area += area_km2

        result["total_area_km2"] = round(total_area, 3)
        result["n_anomalies"]    = len(result["polygons"])

    except Exception as e:
        result["error"] = str(e)

    return result


@st.cache_data(ttl=7200)
def detect_vessels(date_str: str, aoi_coords: list = None, days_back: int = 6) -> dict:
    """Detect vessels as bright SAR targets."""
    _aoi_coords = aoi_coords or [33.0, 31.2, 35.0, 33.15]
    aoi = ee.Geometry.Rectangle(_aoi_coords)
    wm  = _sea_mask()
    result = {"vessels": [], "tile_url": None, "n_vessels": 0, "error": None}

    try:
        img, actual_date, age_h = _get_s1_img_cached(date_str, _aoi_coords, days_back)
        if img is None:
            result["error"] = "No S1 data available"
            return result

        vv = img.select("VV")

        vessel_mask = vv.gt(-13).And(wm).clip(aoi)

        mid = vessel_mask.updateMask(vessel_mask).getMapId(
            {"min": 0, "max": 1, "palette": ["#c8e8f8"]})
        result["tile_url"] = mid["tile_fetcher"].url_format

        vectors = (vessel_mask
                   .updateMask(vessel_mask)
                   .reduceToVectors(
                       geometry=aoi, scale=10, maxPixels=1e9,
                       bestEffort=True, geometryType="polygon",
                       eightConnected=True)
                   .map(lambda f: f.set("area_m2", f.geometry().area(10)))
                   .filter(ee.Filter.And(
                       ee.Filter.gt("area_m2", 100),
                       ee.Filter.lt("area_m2", 150000))))

        feats = vectors.getInfo().get("features", [])

        for f in feats[:25]:
            geom   = f.get("geometry", {})
            props  = f.get("properties", {})
            area_m2 = props.get("area_m2", 0) or 0
            coords = geom.get("coordinates", [[]])[0]
            if not coords:
                continue

            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            clon = sum(lons) / len(lons)
            clat = sum(lats) / len(lats)

            # Real bbox dimensions
            lon_span_m = (max(lons) - min(lons)) * 92500
            lat_span_m = (max(lats) - min(lats)) * 111000
            length_m   = round(max(lon_span_m, lat_span_m), 1)
            width_m    = round(min(lon_span_m, lat_span_m), 1)

            if length_m > 150:
                cat = "Large vessel"
            elif length_m > 50:
                cat = "Medium vessel"
            else:
                cat = "Small vessel"

            conf = "High" if area_m2 > 5000 else "Medium" if area_m2 > 1000 else "Low"

            result["vessels"].append({
                "id":       f"V{len(result['vessels']) + 1}",
                "lat":      round(clat, 4),
                "lon":      round(clon, 4),
                "category": cat,
                "length_m": length_m,
                "width_m":  width_m,
                "area_m2":  round(area_m2, 0),
                "confidence": conf,
                "date":     actual_date,
                "bbox_coords": [
                    [min(lons), min(lats)],
                    [max(lons), min(lats)],
                    [max(lons), max(lats)],
                    [min(lons), max(lats)],
                    [min(lons), min(lats)],
                ],
            })

        result["n_vessels"] = len(result["vessels"])

    except Exception as e:
        result["error"] = str(e)

    return result


def check_vessel_oil_proximity(vessels: list, oil_polygons: list,
                                threshold_km: float = 2.0) -> list:
    """Flag vessels within threshold_km of an oil polygon centroid."""
    def _hav(lat1, lon1, lat2, lon2):
        R  = 6371.0
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        a  = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
              + math.cos(p1) * math.cos(p2)
              * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    for v in vessels:
        v["near_oil"]    = False
        v["near_oil_id"] = None
        for oil in oil_polygons:
            if _hav(v["lat"], v["lon"], oil["lat"], oil["lon"]) <= threshold_km:
                v["near_oil"]    = True
                v["near_oil_id"] = oil["id"]
                break
    return vessels


def age_to_color(age_h: float) -> str:
    """
    Map detection age (hours) to a hex color.
    0-7 days  → red (#e03c3c)
    7-14 days → orange → yellow
    14+ days  → blue (#2255cc), fading with age
    """
    age_days = age_h / 24.0
    if age_days <= 7:
        # Red, intensity scales with freshness
        t   = 1.0 - (age_days / 7.0)   # 1=fresh, 0=week old
        r   = 220
        g   = int(60 * (1 - t))
        b   = int(60 * (1 - t))
        return f"#{r:02x}{g:02x}{b:02x}"
    elif age_days <= 14:
        # Orange → yellow transition
        t = (age_days - 7) / 7.0       # 0=7days, 1=14days
        r = 220
        g = int(60 + 120 * t)
        b = 60
        return f"#{r:02x}{g:02x}{b:02x}"
    else:
        # Blue, fading with age (max ~60 days)
        t   = min(1.0, (age_days - 14) / 46.0)  # 0=14days, 1=60days
        r   = int(60 * (1 - t))
        g   = int(80 * (1 - t))
        b   = int(180 + 40 * t)
        return f"#{r:02x}{g:02x}{b:02x}"
