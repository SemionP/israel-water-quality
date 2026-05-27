"""
gee_pipeline.py
===============================================================================
AquaWatch — Multi-Satellite Fusion Pipeline (Israel Coast POC)
-------------------------------------------------------------------------------
Stage 1 (GEE): Pull latest imagery from S3, MODIS, S2 → compute per-cell:
  - WQI raw value
  - cloud_cover
  - valid_pixel_ratio

Stage 2 (Python): For each cell, compute dynamic score and pick winner.

Stage 3: Output GeoJSON / DataFrame ready for Streamlit map.
===============================================================================
"""

import ee
import math
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


# ==============================================================================
# 0. GEE Init
# ==============================================================================
def init_gee_from_secrets(secrets: dict):
    import tempfile, os
    creds_json = json.dumps(secrets)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(creds_json)
        tmp_path = f.name
    credentials = ee.ServiceAccountCredentials(secrets["client_email"], tmp_path)
    ee.Initialize(credentials)
    os.unlink(tmp_path)


# ==============================================================================
# 1. Grid Definition — 300m over Israel coast
# ==============================================================================

ISRAEL_COAST_BBOX = {
    "lon_min": 34.15,
    "lon_max": 35.10,
    "lat_min": 29.40,   # Eilat
    "lat_max": 33.15,   # Rosh HaNikra
}

# 300m ≈ 0.0027 degrees at Israel's latitude
GRID_STEP_DEG = 0.0027


def build_grid(bbox: dict = ISRAEL_COAST_BBOX) -> pd.DataFrame:
    """
    Builds a static 300m grid over the AOI.
    Returns DataFrame: cell_id, lat, lon
    Water cells only — filtered downstream by GEE GSW mask.
    """
    lats = np.arange(bbox["lat_min"], bbox["lat_max"], GRID_STEP_DEG)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"], GRID_STEP_DEG)

    records = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            cell_id = f"{i:04d}_{j:04d}"
            records.append({
                "cell_id": cell_id,
                "lat":     round(float(lat), 6),
                "lon":     round(float(lon), 6),
            })

    df = pd.DataFrame(records)
    print(f"[Grid] Total cells before water mask: {len(df):,}")
    return df


# ==============================================================================
# 2. GEE — Per-satellite WQI + QA extraction
# ==============================================================================

AOI = ee.Geometry.Rectangle([
    ISRAEL_COAST_BBOX["lon_min"],
    ISRAEL_COAST_BBOX["lat_min"],
    ISRAEL_COAST_BBOX["lon_max"],
    ISRAEL_COAST_BBOX["lat_max"],
])

# Permanent / seasonal water mask
GSW = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(25)

RESOLUTION_M = {"S3": 300, "MODIS": 250, "S2": 10}


def _get_latest(collection_id: str, aoi, days_back: int = 10) -> Optional[ee.Image]:
    """Returns the most recent image from a collection within days_back."""
    end   = datetime.utcnow()
    start = end - timedelta(days=days_back)
    coll  = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        .sort("system:time_start", False)   # newest first
    )
    if coll.size().getInfo() == 0:
        return None
    return coll.first()


# ------------------------------------------------------------------------------
# 2a. Sentinel-3 OLCI  (300m)
# ------------------------------------------------------------------------------
def get_s3_layer() -> Optional[ee.Image]:
    """
    WQI from: NDWI (Oa06/Oa17) + MCI (Oa10-12) + Turbidity (Oa08)
    Cloud proxy: Oa01 brightness (400 nm).
    """
    img = _get_latest("COPERNICUS/S3/OLCI", AOI, days_back=5)
    if img is None:
        return None

    img = img.updateMask(GSW)

    ndwi = img.normalizedDifference(["Oa06_radiance", "Oa17_radiance"]).rename("ndwi")
    b10  = img.select("Oa10_radiance")
    b11  = img.select("Oa11_radiance")
    b12  = img.select("Oa12_radiance")
    mci  = b11.subtract(
        b10.add(b12.subtract(b10).multiply((708.75 - 681.25) / (753.75 - 681.25)))
    ).rename("mci")
    turb = img.select("Oa08_radiance").rename("turb")

    ndwi_n = ndwi.unitScale(-0.2, 0.5).clamp(0, 1)
    mci_n  = ee.Image(1).subtract(mci.unitScale(-2, 12)).clamp(0, 1)
    turb_n = ee.Image(1).subtract(turb.unitScale(10, 80)).clamp(0, 1)

    wqi = ndwi_n.add(mci_n).add(turb_n).divide(3).multiply(100).rename("WQI_S3")

    # Cloud proxy — high blue radiance = cloud/sunglint
    cloud_cover = img.select("Oa01_radiance").unitScale(0, 150).clamp(0, 1).rename("cloud_S3")
    valid       = wqi.mask().rename("valid_S3").toFloat()

    return (
        wqi.addBands(cloud_cover).addBands(valid)
        .set("source_time", img.get("system:time_start"))
        .set("source_name", "S3")
    )


# ------------------------------------------------------------------------------
# 2b. MODIS Terra MOD09GA  (250-500m, daily)
# ------------------------------------------------------------------------------
def get_modis_layer() -> Optional[ee.Image]:
    """
    WQI from MODIS surface reflectance:
      Chl proxy:  B4 (545nm) / B1 (645nm)
      Turbidity:  B1 (645nm)
      NDWI:       (B4 - B2) / (B4 + B2)
    Cloud mask from state_1km QA.
    """
    img = _get_latest("MODIS/061/MOD09GA", AOI, days_back=3)
    if img is None:
        return None

    qa    = img.select("state_1km")
    clear = qa.bitwiseAnd(0b11).eq(0)
    img   = img.updateMask(clear).updateMask(GSW)

    b1 = img.select("sur_refl_b01")   # 645 nm red
    b2 = img.select("sur_refl_b02")   # 859 nm NIR
    b4 = img.select("sur_refl_b04")   # 545 nm green

    ndwi_n = b4.subtract(b2).divide(b4.add(b2)).unitScale(-0.3, 0.3).clamp(0, 1)
    chl_n  = b4.divide(b1.add(1e-6)).unitScale(0.8, 2.5).clamp(0, 1)
    turb_n = ee.Image(1).subtract(b1.unitScale(0, 1500)).clamp(0, 1)

    wqi = ndwi_n.add(chl_n).add(turb_n).divide(3).multiply(100).rename("WQI_MODIS")

    cloud_cover = clear.Not().rename("cloud_MODIS").toFloat()
    valid       = wqi.mask().rename("valid_MODIS").toFloat()

    return (
        wqi.addBands(cloud_cover).addBands(valid)
        .set("source_time", img.get("system:time_start"))
        .set("source_name", "MODIS")
    )


# ------------------------------------------------------------------------------
# 2c. Sentinel-2 MSI  (10m → aggregated to ~300m)
# ------------------------------------------------------------------------------
def get_s2_layer() -> Optional[ee.Image]:
    """
    WQI from S2 bands:
      Chl:      B5 / B4  (Red Edge)
      Turbidity: B4 + B8A
      NDWI:     (B3 - B8) / (B3 + B8)
    Aggregated from 10m → ~300m via focal mean (30-pixel kernel).
    valid_ratio = fraction of water-clear S2 pixels in each kernel window.
    """
    img = _get_latest("COPERNICUS/S2_SR_HARMONIZED", AOI, days_back=10)
    if img is None:
        return None

    scl   = img.select("SCL")
    clear = scl.eq(6)                          # SCL class 6 = water
    img   = img.updateMask(clear).updateMask(GSW)

    b3  = img.select("B3").divide(10000)
    b4  = img.select("B4").divide(10000)
    b5  = img.select("B5").divide(10000)
    b8  = img.select("B8").divide(10000)
    b8a = img.select("B8A").divide(10000)

    ndwi_n = b3.subtract(b8).divide(b3.add(b8)).unitScale(-0.3, 0.5).clamp(0, 1)
    chl_n  = b5.divide(b4.add(1e-6)).unitScale(1.0, 3.5).clamp(0, 1)
    turb_n = ee.Image(1).subtract(b4.add(b8a).divide(2).unitScale(0, 0.15)).clamp(0, 1)

    wqi_10m = ndwi_n.add(chl_n).add(turb_n).divide(3).multiply(100)

    # Aggregate 10m → 300m: 30-pixel radius focal mean
    kernel      = ee.Kernel.square(radius=15, units="pixels")
    wqi         = wqi_10m.reduceNeighborhood(ee.Reducer.mean(), kernel).rename("WQI_S2")
    valid_ratio = clear.toFloat().reduceNeighborhood(ee.Reducer.mean(), kernel).rename("valid_S2")
    cloud_cover = ee.Image(1).subtract(valid_ratio).rename("cloud_S2")

    return (
        wqi.addBands(cloud_cover).addBands(valid_ratio)
        .set("source_time", img.get("system:time_start"))
        .set("source_name", "S2")
    )


# ==============================================================================
# 3. Sample all layers onto the 300m grid
# ==============================================================================

def sample_layer(
    layer: ee.Image,
    grid_df: pd.DataFrame,
    wqi_band: str,
    cloud_band: str,
    valid_band: str,
    source_name: str,
    scale: int = 300,
) -> pd.DataFrame:
    """
    Samples an EE image at every grid cell center.
    Returns DataFrame: cell_id, wqi, cloud_cover, valid_ratio, source, age_days
    """
    if layer is None:
        return pd.DataFrame()

    features = [
        ee.Feature(
            ee.Geometry.Point([row["lon"], row["lat"]]),
            {"cell_id": row["cell_id"]},
        )
        for _, row in grid_df.iterrows()
    ]
    fc = ee.FeatureCollection(features)

    sampled = layer.select([wqi_band, cloud_band, valid_band]).sampleRegions(
        collection=fc,
        scale=scale,
        geometries=False,
        tileScale=4,
    )

    source_time_ms = layer.get("source_time").getInfo()

    feats = sampled.getInfo().get("features", [])
    records = []
    for f in feats:
        p       = f["properties"]
        wqi_val = p.get(wqi_band)
        if wqi_val is None:
            continue
        records.append({
            "cell_id":        p.get("cell_id"),
            "wqi":            round(wqi_val, 2),
            "cloud_cover":    round(p.get(cloud_band, 1.0), 3),
            "valid_ratio":    round(p.get(valid_band, 0.0), 3),
            "source":         source_name,
            "source_time_ms": source_time_ms,
        })

    df = pd.DataFrame(records)
    if not df.empty and source_time_ms:
        df["source_dt"] = pd.to_datetime(df["source_time_ms"], unit="ms", utc=True)
        df["age_days"]  = (
            pd.Timestamp.utcnow() - df["source_dt"]
        ).dt.total_seconds() / 86400

    return df


# ==============================================================================
# 4. Dynamic Score + Winner Selection  (pure Python, no GEE)
# ==============================================================================

def compute_score(age_days: float, confidence: float, resolution_m: int) -> float:
    """
    Dynamic score = freshness × confidence × resolution_score

    freshness:        exp decay — halves every ~2.3 days
    confidence:       (1 - cloud_cover) × valid_ratio
    resolution_score: log-inverse of pixel size (finer = better)
    """
    freshness = math.exp(-0.3 * age_days)
    res_score = 1.0 / math.log10(max(resolution_m, 10))
    return freshness * confidence * res_score


def pick_winner(cell_readings: list) -> Optional[dict]:
    """
    Given readings for one cell from multiple satellites,
    returns the reading with the highest dynamic score.
    """
    best       = None
    best_score = -1.0

    for r in cell_readings:
        confidence = (1.0 - r.get("cloud_cover", 1.0)) * r.get("valid_ratio", 1.0)
        score      = compute_score(
            age_days     = r.get("age_days", 99),
            confidence   = confidence,
            resolution_m = RESOLUTION_M.get(r["source"], 500),
        )
        if score > best_score:
            best_score = score
            best       = {**r, "score": round(score, 4), "confidence": round(confidence, 3)}

    return best


def fuse(readings_dfs: list, grid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges all satellite readings, picks winner per cell using dynamic scoring.
    Returns: cell_id, lat, lon, wqi, source, age_days, score, confidence, health_label
    """
    valid_dfs = [df for df in readings_dfs if not df.empty]
    if not valid_dfs:
        return pd.DataFrame()

    all_readings = pd.concat(valid_dfs, ignore_index=True)

    winners = []
    for cell_id, group in all_readings.groupby("cell_id"):
        winner = pick_winner(group.to_dict("records"))
        if winner:
            winners.append(winner)

    result = pd.DataFrame(winners)
    result = result.merge(grid_df[["cell_id", "lat", "lon"]], on="cell_id", how="left")

    def _label(wqi):
        if wqi >= 70: return "🟢 Safe"
        if wqi >= 55: return "🟡 Caution"
        return "🔴 Unsafe"

    result["health_label"] = result["wqi"].apply(_label)

    return result.sort_values("cell_id").reset_index(drop=True)


# ==============================================================================
# 5. Main entry point
# ==============================================================================

def run_pipeline(gee_secrets: dict = None) -> pd.DataFrame:
    """
    Full pipeline: GEE pull → sample → fuse → return DataFrame.
    Pass gee_secrets dict (from st.secrets) or call after ee.Initialize().
    """
    if gee_secrets:
        init_gee_from_secrets(gee_secrets)
    else:
        ee.Initialize()

    print("[Pipeline] Building 300m grid...")
    grid = build_grid()

    print("[Pipeline] Pulling latest satellite layers...")
    s3_layer    = get_s3_layer()
    modis_layer = get_modis_layer()
    s2_layer    = get_s2_layer()

    print(f"  S3 available:    {s3_layer    is not None}")
    print(f"  MODIS available: {modis_layer is not None}")
    print(f"  S2 available:    {s2_layer    is not None}")

    print("[Pipeline] Sampling onto grid...")
    readings = []

    if s3_layer:
        df = sample_layer(s3_layer, grid,
                          wqi_band="WQI_S3", cloud_band="cloud_S3",
                          valid_band="valid_S3", source_name="S3", scale=300)
        print(f"  S3    → {len(df):,} cells")
        readings.append(df)

    if modis_layer:
        df = sample_layer(modis_layer, grid,
                          wqi_band="WQI_MODIS", cloud_band="cloud_MODIS",
                          valid_band="valid_MODIS", source_name="MODIS", scale=500)
        print(f"  MODIS → {len(df):,} cells")
        readings.append(df)

    if s2_layer:
        df = sample_layer(s2_layer, grid,
                          wqi_band="WQI_S2", cloud_band="cloud_S2",
                          valid_band="valid_S2", source_name="S2", scale=300)
        print(f"  S2    → {len(df):,} cells")
        readings.append(df)

    print("[Pipeline] Fusing with dynamic scoring...")
    result = fuse(readings, grid)

    print(f"[Pipeline] Done — {len(result):,} fused cells")
    if not result.empty:
        print(f"  Source breakdown:\n{result['source'].value_counts().to_string()}")

    return result


# ==============================================================================
# 6. Export helpers
# ==============================================================================

def to_geojson(df: pd.DataFrame) -> dict:
    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["lon"], row["lat"]],
            },
            "properties": {
                "cell_id":      row["cell_id"],
                "wqi":          row["wqi"],
                "source":       row["source"],
                "age_days":     round(row.get("age_days", 0), 2),
                "score":        row.get("score"),
                "confidence":   row.get("confidence"),
                "health_label": row.get("health_label"),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def to_csv(df: pd.DataFrame, path: str):
    cols = ["cell_id", "lat", "lon", "wqi", "source",
            "age_days", "score", "confidence", "health_label"]
    df[cols].to_csv(path, index=False)
    print(f"[Export] {len(df):,} rows → {path}")


# ==============================================================================
# Quick local test — no GEE required
# ==============================================================================
if __name__ == "__main__":
    print("=== Grid test ===")
    grid = build_grid()
    print(grid.head())
    print(f"Total cells: {len(grid):,}")

    print("\n=== Dynamic score test ===")
    test_cases = [
        {"source": "S3",    "age_days": 1,  "cloud_cover": 0.4, "valid_ratio": 0.8, "wqi": 68},
        {"source": "MODIS", "age_days": 0,  "cloud_cover": 0.1, "valid_ratio": 1.0, "wqi": 71},
        {"source": "S2",    "age_days": 3,  "cloud_cover": 0.0, "valid_ratio": 1.0, "wqi": 74},
    ]
    for t in test_cases:
        conf  = (1 - t["cloud_cover"]) * t["valid_ratio"]
        score = compute_score(t["age_days"], conf, RESOLUTION_M[t["source"]])
        print(f"  {t['source']:6s}  age={t['age_days']}d  cloud={t['cloud_cover']}  → score={score:.4f}")

    print("\n=== Winner selection ===")
    winner = pick_winner(test_cases)
    print(f"  Winner: {winner['source']}  WQI={winner['wqi']}  score={winner['score']}")
