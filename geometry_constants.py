"""
geometry_constants.py - MEDI Platform
======================================
Single source of truth for all geographic boundaries.

Scope: Israeli Mediterranean Sea ONLY.
Explicitly excluded:
  - Kinneret (Sea of Galilee)   — 32.7–32.9°N, 35.5–35.7°E
  - Dead Sea                    — 31.0–31.8°N, 35.4–35.6°E
  - Red Sea / Gulf of Eilat     — south of 31.2°N

All coordinates: [lon_min, lat_min, lon_max, lat_max] or polygon rings [lon, lat].
"""

import ee

# =============================================================================
# BOUNDING BOXES (Rectangle format: [lon_min, lat_min, lon_max, lat_max])
# =============================================================================

# Full Israeli Mediterranean operational area
# North: Lebanese border ~33.1°N | South: Gaza/Egypt border ~31.2°N
# East: coastline ~34.9-35.0°E   | West: open Mediterranean
MED_BBOX_COORDS = [33.0, 31.2, 35.0, 33.15]

# Wider area for satellite pass detection (S1/S3 coverage check)
MED_WIDE_COORDS = [32.5, 31.0, 35.1, 33.2]

# Port-specific bounding boxes
HAIFA_PORT_BBOX_COORDS  = [34.75, 32.70, 35.00, 32.95]
ASHDOD_PORT_BBOX_COORDS = [34.45, 31.65, 34.80, 31.95]

# Legacy alias used by gee_processing.py — same as MED_BBOX_COORDS
HAIFA_BBOX_COORDS = [34.20, 31.20, 35.20, 33.20]  # kept for backward compat


# =============================================================================
# POLYGON BOUNDARIES
# =============================================================================

# Israeli Mediterranean coast polygon (sea-side only)
# Follows the coastline from Gaza to Rosh HaNikra, then closes westward over open sea
ISRAEL_MED_SEA_COORDS = [
    [33.0,  31.2],   # SW — open Mediterranean
    [35.0,  31.2],   # SE — Gaza/Egypt maritime border
    [35.0,  31.55],  # Ashkelon offshore
    [34.90, 31.70],  # Ashdod approaches
    [34.85, 32.10],  # Tel Aviv offshore
    [34.80, 32.35],  # Netanya offshore
    [34.85, 32.55],  # Caesarea offshore
    [34.90, 32.82],  # Haifa offshore
    [35.00, 33.05],  # Rosh HaNikra
    [35.00, 33.15],  # NE — Lebanese border
    [33.0,  33.15],  # NW — open Mediterranean
    [33.0,  31.2],   # close polygon
]

# Israel land clip (for WQI optical processing — clips inland water)
# Used by gee_processing.py
ISRAEL_CLIP_COORDS = [
    [34.95, 33.10], [34.55, 33.10], [34.15, 32.50], [34.10, 32.00],
    [34.15, 31.50], [34.50, 31.25], [34.75, 31.25], [34.95, 31.30],
    [35.02, 31.60], [35.00, 32.10], [35.05, 32.60], [35.10, 33.10],
    [34.95, 33.10],
]

# =============================================================================
# AOI PRESETS for Oil Spill Detection UI
# =============================================================================
AOI_PRESETS = {
    "🇮🇱 Territorial Waters (12nm)": {
        "coords":      [33.8, 31.2, 34.95, 33.1],
        "description": "Israeli territorial waters — 12 nautical miles from baseline",
        "color":       "#00c8c8",
    },
    "🌊 Contiguous Zone (24nm)": {
        "coords":      [33.5, 31.2, 34.95, 33.1],
        "description": "Israeli contiguous zone — 24 nautical miles",
        "color":       "#5aaacf",
    },
    "🗺️ EEZ (approx.)": {
        "coords":      [33.0, 31.2, 34.95, 33.15],
        "description": "Israeli EEZ in the Mediterranean — approximate per Cyprus agreement",
        "color":       "#3a7abf",
    },
    "🚢 Haifa Port vicinity": {
        "coords":      HAIFA_PORT_BBOX_COORDS,
        "description": "Haifa port and approaches (~25 km radius)",
        "color":       "#1ecb7b",
    },
    "⚓ Ashdod Port vicinity": {
        "coords":      ASHDOD_PORT_BBOX_COORDS,
        "description": "Ashdod port and approaches (~25 km radius)",
        "color":       "#1ecb7b",
    },
    "✏️ Draw custom AOI": {
        "coords":      None,
        "description": "Draw a polygon on the map",
        "color":       "#f0a500",
    },
}

# =============================================================================
# GEE Geometry objects (lazy — only built when ee is initialized)
# =============================================================================

def get_med_bbox():
    return ee.Geometry.Rectangle(MED_BBOX_COORDS)

def get_med_wide():
    return ee.Geometry.Rectangle(MED_WIDE_COORDS)

def get_israel_med_sea():
    return ee.Geometry.Polygon([ISRAEL_MED_SEA_COORDS])

def get_israel_clip():
    return ee.Geometry.Polygon([ISRAEL_CLIP_COORDS])

def get_haifa_bbox():
    return ee.Geometry.Rectangle(HAIFA_BBOX_COORDS)

def sea_mask():
    """
    Mediterranean sea-only binary mask.
    Kinneret / Dead Sea / Red Sea excluded by clipping to ISRAEL_MED_SEA polygon.
    """
    return (ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
            .select("occurrence")
            .gte(50)
            .clip(get_israel_med_sea()))
