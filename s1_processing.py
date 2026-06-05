"""
s1_processing.py - MEDI Platform
Sentinel-1 SAR processing: layers, oil detection, vessel detection
"""

import math
from datetime import datetime, timedelta
import streamlit as st
import ee

# Full display box
DISPLAY_BOX = ee.Geometry.Rectangle([33.5, 29.5, 36.5, 33.5])

# Mediterranean sea only — excludes inland water bodies
MED_SEA_BOX = ee.Geometry.Rectangle([33.5, 29.5, 35.2, 33.3])

# Inland water masks to exclude (Kinneret + Dead Sea)
INLAND_EXCLUDE = [
    ee.Geometry.Rectangle([35.3, 32.6, 35.7, 33.0]),  # Kinneret
    ee.Geometry.Rectangle([35.3, 31.0, 35.7, 31.9]),  # Dead Sea
]

def _get_sea_mask():
    """Water mask: permanent ocean/sea only, excluding inland lakes."""
    wm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(80)
    # Exclude inland water bodies
    inland = ee.Image(0)
    for geom in INLAND_EXCLUDE:
        inland = inland.Or(ee.Image(1).clip(geom))
    return wm.And(inland.Not())


@st.cache_data(ttl=14400)
def get_available_s1_dates(days_back: int = 14) -> list:
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
        seen = set()
        for ts, orb in sorted(zip(ts_list, orbits), reverse=True):
            d = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            key = f"{d}_{orb}"
            if key not in seen:
                seen.add(key)
                results.append({"date": d, "orbit": orb,
                                 "age_h": round((datetime.utcnow() - datetime.utcfromtimestamp(ts/1000)).total_seconds()/3600, 1)})
        return results
    except Exception:
        return []


@st.cache_data(ttl=7200)
def get_s1_layers(date_str: str, orbit: str = "ASCENDING") -> dict:
    t = ee.Date(date_str)
    result = {}
    try:
        coll = (ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(MED_SEA_BOX)
                .filterDate(t.advance(-3, "day"), t.advance(1, "day"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                .filter(ee.Filter.eq("instrumentMode", "IW")))
        if coll.size().getInfo() == 0:
            return result

        img = coll.mosaic().clip(MED_SEA_BOX)
        vv  = img.select("VV")
        vh  = img.select("VH")

        mid = vv.getMapId({"min": -25, "max": 0,
                           "palette": ["#000014","#0a1520","#152840","#1e3a5a","#5aaacf","#c8e8f8"]})
        result["vv"] = mid["tile_fetcher"].url_format

        mid = vh.getMapId({"min": -30, "max": -5,
                           "palette": ["#000014","#0a1520","#152840","#1e3a5a","#5aaacf","#c8e8f8"]})
        result["vh"] = mid["tile_fetcher"].url_format

        ratio = vv.subtract(vh)
        mid = ratio.getMapId({"min": 0, "max": 15,
                              "palette": ["#041e33","#1D9E75","#fdae61","#d73027"]})
        result["ratio"] = mid["tile_fetcher"].url_format

        rgb_img = ee.Image.cat([
            vv.unitScale(-25, 0).clamp(0, 1),
            vh.unitScale(-30, -5).clamp(0, 1),
            ratio.unitScale(0, 15).clamp(0, 1)
        ])
        mid = rgb_img.getMapId({"bands": ["VV","VH","VV"], "min": 0, "max": 1, "gamma": 1.2})
        result["rgb"] = mid["tile_fetcher"].url_format

        urban = vv.gt(-5).selfMask()
        mid = urban.getMapId({"min": 0, "max": 1, "palette": ["#c8e8f8"]})
        result["urban"] = mid["tile_fetcher"].url_format

    except Exception:
        pass
    return result


@st.cache_data(ttl=7200)
def detect_oil_spills(date_str: str, bbox_coords: list = None) -> dict:
    t   = ee.Date(date_str)
    # Use Med sea box — avoids inland false positives
    aoi = ee.Geometry.Rectangle(bbox_coords) if bbox_coords else MED_SEA_BOX
    wm  = _get_sea_mask()
    result = {"polygons": [], "tile_url": None, "total_area_km2": 0, "n_anomalies": 0}

    try:
        coll = (ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(aoi)
                .filterDate(t.advance(-3, "day"), t.advance(1, "day"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                .filter(ee.Filter.eq("instrumentMode", "IW")))
        if coll.size().getInfo() == 0:
            return result

        img = coll.mosaic()
        vv  = img.select("VV")
        vh  = img.select("VH")

        # Oil: low VV on sea surface + VV-VH ratio check
        oil_mask = (vv.lt(-18)
                    .And(vv.subtract(vh).lt(8))
                    .And(wm)
                    .clip(aoi))

        mid = oil_mask.updateMask(oil_mask).getMapId({"min": 0, "max": 1, "palette": ["#e24b4a"]})
        result["tile_url"] = mid["tile_fetcher"].url_format

        vectors = oil_mask.updateMask(oil_mask).reduceToVectors(
            geometry=aoi,
            scale=100,
            maxPixels=1e8,
            bestEffort=True,
            geometryType="polygon",
            eightConnected=True
        )

        vectors = vectors.map(lambda f: f.set("area_m2", f.geometry().area(100)))
        vectors = vectors.filter(ee.Filter.And(
            ee.Filter.gt("area_m2", 50000),
            ee.Filter.lt("area_m2", 50000000)
        ))

        feats = vectors.getInfo().get("features", [])
        total_area = 0
        for i, f in enumerate(feats[:10]):
            geom   = f.get("geometry", {})
            props  = f.get("properties", {})
            area_m2 = props.get("area_m2", 0) or 0
            area_km2 = area_m2 / 1e6

            coords = geom.get("coordinates", [[]])[0]
            if coords:
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                clon = sum(lons) / len(lons)
                clat = sum(lats) / len(lats)
            else:
                clat, clon = 32.8, 34.9

            # Skip if centroid is not in Mediterranean (lon < 34.0 = definitely sea)
            if clon > 35.0:
                continue

            conf = "High" if area_km2 > 0.5 else "Medium" if area_km2 > 0.2 else "Low"
            area_min = round(area_km2 * 0.7, 2)
            area_max = round(area_km2 * 1.3, 2)
            total_area += area_km2

            result["polygons"].append({
                "id": f"OIL-{len(result['polygons'])+1}",
                "area_km2_min": area_min,
                "area_km2_max": area_max,
                "confidence": conf,
                "lat": round(clat, 4),
                "lon": round(clon, 4),
                "coords": coords,
            })

        result["total_area_km2"] = round(total_area, 2)
        result["n_anomalies"]    = len(result["polygons"])

    except Exception:
        pass
    return result


@st.cache_data(ttl=7200)
def detect_vessels(date_str: str, bbox_coords: list = None) -> dict:
    t   = ee.Date(date_str)
    aoi = ee.Geometry.Rectangle(bbox_coords) if bbox_coords else MED_SEA_BOX
    wm  = _get_sea_mask()
    result = {"vessels": [], "tile_url": None, "n_vessels": 0}

    try:
        coll = (ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(aoi)
                .filterDate(t.advance(-3, "day"), t.advance(1, "day"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                .filter(ee.Filter.eq("instrumentMode", "IW")))
        if coll.size().getInfo() == 0:
            return result

        img = coll.mosaic()
        vv  = img.select("VV")
        vh  = img.select("VH")

        # Vessels = bright targets on sea only
        energy = vv.add(vh).divide(2)
        vessel_mask = (energy.gt(-8)
                       .And(wm)
                       .clip(aoi))

        mid = vessel_mask.updateMask(vessel_mask).getMapId({"min": 0, "max": 1, "palette": ["#c8e8f8"]})
        result["tile_url"] = mid["tile_fetcher"].url_format

        vectors = vessel_mask.updateMask(vessel_mask).reduceToVectors(
            geometry=aoi,
            scale=20,
            maxPixels=1e8,
            bestEffort=True,
            geometryType="polygon",
            eightConnected=True
        )

        vectors = vectors.map(lambda f: f.set("area_m2", f.geometry().area(20)))
        vectors = vectors.filter(ee.Filter.And(
            ee.Filter.gt("area_m2", 200),
            ee.Filter.lt("area_m2", 100000)
        ))

        feats = vectors.getInfo().get("features", [])
        for i, f in enumerate(feats[:20]):
            geom    = f.get("geometry", {})
            props   = f.get("properties", {})
            area_m2 = props.get("area_m2", 0) or 0

            coords = geom.get("coordinates", [[]])[0]
            if coords:
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                clon = sum(lons) / len(lons)
                clat = sum(lats) / len(lats)
                # Skip if not in sea area
                if clon > 35.0:
                    continue
                lon_range = max(lons) - min(lons)
                lat_range = max(lats) - min(lats)
                px = lon_range * 92000
                py = lat_range * 111000
                length_px = max(px, py)
                width_px  = min(px, py)
            else:
                clat, clon = 32.8, 34.9
                length_px, width_px = 60, 15

            if length_px > 150:
                cat = "Large"; l_min, l_max = 120, 250; w_min, w_max = 20, 45; conf = "High"
            elif length_px > 60:
                cat = "Medium"; l_min, l_max = 50, 130; w_min, w_max = 12, 30; conf = "High"
            elif length_px > 25:
                cat = "Small"; l_min, l_max = 20, 80; w_min, w_max = 8, 22; conf = "Medium"
            else:
                cat = "Small"; l_min, l_max = 10, 45; w_min, w_max = 5, 15; conf = "Low"

            result["vessels"].append({
                "id": f"V{len(result['vessels'])+1}",
                "lat": round(clat, 4),
                "lon": round(clon, 4),
                "category": cat,
                "length_min_m": l_min,
                "length_max_m": l_max,
                "width_min_m":  w_min,
                "width_max_m":  w_max,
                "confidence":   conf,
            })

        result["n_vessels"] = len(result["vessels"])

    except Exception:
        pass
    return result


def check_vessel_oil_proximity(vessels: list, oil_polygons: list, threshold_km: float = 2.0) -> list:
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        a = math.sin(math.radians(lat2-lat1)/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(math.radians(lon2-lon1)/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    for v in vessels:
        v["near_oil"]    = False
        v["near_oil_id"] = None
        for oil in oil_polygons:
            d = _haversine(v["lat"], v["lon"], oil["lat"], oil["lon"])
            if d <= threshold_km:
                v["near_oil"]    = True
                v["near_oil_id"] = oil["id"]
                break
    return vessels
