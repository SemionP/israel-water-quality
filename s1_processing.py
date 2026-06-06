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
    """
    Return GEE tile URLs for SAR visualisation layers.
    Layers:
      vv    — raw VV backscatter (dB), greyscale
      ratio — VV/VH ratio
      orm   — Oil Risk Map composite (your colorBlend formula)
    """
    _aoi_coords = aoi_coords or [33.0, 31.2, 35.0, 33.15]
    result = {}
    try:
        img, _, _ = _get_s1_img_cached(date_str, _aoi_coords)
        if img is None:
            return result

        vv = img.select("VV")
        vh = img.select("VH")

        # ── Layer 1: Raw VV (dB) ──────────────────────────────────────────
        mid = vv.getMapId({"min": -25, "max": 0,
                           "palette": ["#000014","#0a1520","#152840",
                                       "#1e3a5a","#5aaacf","#c8e8f8"]})
        result["vv"] = mid["tile_fetcher"].url_format

        # ── Layer 2: VV/VH ratio ──────────────────────────────────────────
        mid = vv.subtract(vh).getMapId({"min": 0, "max": 15,
                              "palette": ["#041e33","#1D9E75","#fdae61","#d73027"]})
        result["ratio"] = mid["tile_fetcher"].url_format

        # ── Layer 3: ORM — Oil Risk Map ───────────────────────────────────
        # Your formula: ORM = log(0.01 / (0.01 + VV_linear * 2))
        # Displayed only where: ORM < 0 AND VV_lin < 0.018 AND VH_lin < 0.00126
        vv_lin = ee.Image(10).pow(vv.divide(10))   # dB → linear
        vh_lin = ee.Image(10).pow(vh.divide(10))
        orm    = (ee.Image(0.01)
                  .divide(ee.Image(0.01).add(vv_lin.multiply(2)))
                  .log())

        # Oil candidate mask (your exact thresholds)
        oil_candidate = (orm.lt(0)
                         .And(vv_lin.lt(0.018))
                         .And(vh_lin.lt(0.00126)))

        # ORM colorBlend: [-1.6, -1.4, -1.2, -1, -.8, -.6, -.4, -.2, 0]
        # Colors: dark blue → blue → red → orange → yellow → green
        orm_vis = orm.updateMask(oil_candidate)
        mid_orm = orm_vis.getMapId({
            "min": -1.6, "max": 0,
            "palette": [
                "#00001a",  # -1.6  dark blue
                "#0000cc",  # -1.4  blue
                "#cc0080",  # -1.2  purple-red (1,0,.5)
                "#ff0000",  # -1.0  red
                "#ff8033",  # -0.8  orange
                "#ffcc33",  # -0.6  yellow-orange
                "#ffff66",  # -0.4  yellow
                "#80cc4d",  # -0.2  yellow-green
                "#80cc4d",  #  0.0  green
            ]
        })
        result["orm"] = mid_orm["tile_fetcher"].url_format

    except Exception:
        pass
    return result


@st.cache_data(ttl=7200)
def detect_oil_spills(date_str: str, aoi_coords: list = None,
                      days_back: int = 6,
                      min_area_m2: int = 20000,
                      max_area_m2: int = 80000000) -> dict:
    """
    Detect oil spill candidates using ORM (Oil Risk Map) index.

    Algorithm based on user-provided formula:
        ORM = log(0.01 / (0.01 + VV_linear * 2))
        Candidate if: ORM < 0  AND  VV_linear < 0.018  AND  VH_linear < 0.00126

    Probability is derived from ORM value:
        ORM in [-0.2,  0.0) → 30% (Low)
        ORM in [-0.6, -0.2) → 55% (Medium)
        ORM in [-1.0, -0.6) → 80% (High)
        ORM <  -1.0         → 95% (Very High)
    """
    _aoi_coords = aoi_coords or [33.0, 31.2, 35.0, 33.15]
    aoi = ee.Geometry.Rectangle(_aoi_coords)
    wm  = _sea_mask()

    result = {
        "polygons":       [],
        "tile_url_orm":   None,
        "tile_url_raw":   None,
        "total_area_km2": 0,
        "n_anomalies":    0,
        "actual_date":    None,
        "age_h":          None,
        "error":          None,
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

        # ── Convert dB → linear ───────────────────────────────────────────
        vv_lin = ee.Image(10).pow(vv.divide(10))
        vh_lin = ee.Image(10).pow(vh.divide(10))

        # ── ORM index (your formula) ──────────────────────────────────────
        orm = (ee.Image(0.01)
               .divide(ee.Image(0.01).add(vv_lin.multiply(2)))
               .log())

        # ── Oil candidate mask (your exact thresholds) ────────────────────
        oil_mask = (orm.lt(0)
                    .And(vv_lin.lt(0.018))
                    .And(vh_lin.lt(0.00126))
                    .And(wm)
                    .clip(aoi))

        # ── ORM visualization tile (colorBlend) ───────────────────────────
        orm_vis = orm.updateMask(oil_mask)
        mid_orm = orm_vis.getMapId({
            "min": -1.6, "max": 0,
            "palette": ["#00001a","#0000cc","#6600cc","#cc0080",
                        "#ff0000","#ff8033","#ffcc33","#ffff66","#80cc4d"]
        })
        result["tile_url_orm"] = mid_orm["tile_fetcher"].url_format

        # Raw VV for context
        mid_raw = vv.getMapId({
            "min": -25, "max": 0,
            "palette": ["#000014","#0a1520","#152840","#1e3a5a","#5aaacf","#c8e8f8"]
        })
        result["tile_url_raw"] = mid_raw["tile_fetcher"].url_format

        # ── Vectorise oil_mask → polygons ─────────────────────────────────
        vectors = (oil_mask
                   .updateMask(oil_mask)
                   .reduceToVectors(
                       geometry=aoi,
                       scale=40,
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
        for f in feats[:15]:
            geom    = f.get("geometry", {})
            props   = f.get("properties", {})
            area_m2 = props.get("area_m2", 0) or 0
            area_km2 = area_m2 / 1e6

            coords = geom.get("coordinates", [[]])[0]
            if not coords:
                continue

            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            clon = sum(lons) / len(lons)
            clat = sum(lats) / len(lats)

            # Centroid must be in Mediterranean sea
            if clon > 35.15 or clon < 32.0:
                continue

            # ── Sample mean ORM value inside polygon ──────────────────────
            try:
                orm_mean = (orm
                            .reduceRegion(
                                reducer=ee.Reducer.mean(),
                                geometry=f["geometry"],
                                scale=100,
                                bestEffort=True)
                            .get("constant").getInfo())
            except Exception:
                orm_mean = None

            # ── Probability from ORM value ────────────────────────────────
            if orm_mean is None:
                prob, conf, color = 50, "Medium", "#f0a500"
            elif orm_mean < -1.0:
                prob, conf, color = 95, "Very High", "#e03c3c"
            elif orm_mean < -0.6:
                prob, conf, color = 80, "High",      "#e03c3c"
            elif orm_mean < -0.2:
                prob, conf, color = 55, "Medium",    "#f0a500"
            else:
                prob, conf, color = 30, "Low",       "#5aaacf"

            result["polygons"].append({
                "id":           f"OIL-{len(result['polygons']) + 1}",
                "area_km2":     round(area_km2, 3),
                "area_km2_min": round(area_km2 * 0.7, 3),
                "area_km2_max": round(area_km2 * 1.3, 3),
                "probability":  prob,       # 30 / 55 / 80 / 95
                "confidence":   conf,       # Low / Medium / High / Very High
                "color":        color,      # hex for bounding box
                "orm_mean":     round(orm_mean, 3) if orm_mean else None,
                "lat":          round(clat, 4),
                "lon":          round(clon, 4),
                "coords":       coords,
                "date":         actual_date,
                "age_h":        age_h,
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
