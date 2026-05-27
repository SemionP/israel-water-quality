"""
medi_engine.py
==============================================================================
MEDI Platform — Risk Engine (Hidden Layer)
==============================================================================
Input:  raw signal values (WQI, SST, turbidity, chlorophyll, vessel_density)
Output: risk_score, risk_level, trend, confidence, drivers[]
        → passed to Claude for explanation generation
==============================================================================
"""

import math
from dataclasses import dataclass, field
from typing import Optional

# ==============================================================================
# Signal thresholds — PROPRIETARY (not exposed to users)
# ==============================================================================
_T = {
    "wqi":             {"low": 70, "mid": 50, "high": 35},
    "turbidity":       {"low": 0.3, "mid": 0.55, "high": 0.75},
    "chlorophyll":     {"low": 0.25, "mid": 0.5,  "high": 0.7},
    "sst_anomaly":     {"low": 1.5,  "mid": 3.0,  "high": 5.0},
    "vessel_density":  {"low": 0.3,  "mid": 0.55, "high": 0.75},
    "oil_proxy":       {"low": 0.2,  "mid": 0.4,  "high": 0.6},
    "night_activity":  {"low": 0.25, "mid": 0.5,  "high": 0.7},
    "coastline_change":{"low": 0.2,  "mid": 0.45, "high": 0.65},
}

# ==============================================================================
# Risk profiles — PUBLIC (users choose)
# ==============================================================================
PROFILES = {
    "Port Operations": {
        "signals":     ["wqi", "turbidity", "vessel_density", "oil_proxy"],
        "weights":     [0.25,  0.25,         0.30,             0.20],
        "description": "Focused on vessel traffic, discharge risk, and water intake quality.",
    },
    "Beach Safety": {
        "signals":     ["wqi", "turbidity", "chlorophyll"],
        "weights":     [0.45,  0.30,         0.25],
        "description": "Focused on bathing water quality and algae/bloom risk.",
    },
    "Aquaculture": {
        "signals":     ["wqi", "chlorophyll", "sst_anomaly", "turbidity"],
        "weights":     [0.30,  0.35,           0.25,          0.10],
        "description": "Focused on bloom conditions, oxygen stress, and feed disruption.",
    },
    "ESG Compliance": {
        "signals":     ["wqi", "turbidity", "oil_proxy", "vessel_density", "coastline_change"],
        "weights":     [0.20,  0.20,         0.25,        0.20,             0.15],
        "description": "Broad environmental footprint monitoring for reporting.",
    },
    "Maritime Surveillance": {
        "signals":     ["vessel_density", "night_activity", "oil_proxy", "turbidity"],
        "weights":     [0.35,             0.30,             0.25,        0.10],
        "description": "Focused on illegal activity, dark vessels, and discharge events.",
    },
}

# ==============================================================================
# Data classes
# ==============================================================================
@dataclass
class SignalReading:
    """One signal measurement at a point in time."""
    name:       str
    value:      float           # normalized 0–1 (higher = worse)
    raw_value:  Optional[float] = None
    unit:       str = ""
    age_days:   float = 0.0
    confidence: float = 1.0

@dataclass
class MEDIResult:
    """Full MEDI output — public-facing fields only."""
    risk_score:   float          # 0–100
    risk_level:   str            # LOW / MODERATE / ELEVATED / HIGH / CRITICAL
    risk_color:   str            # hex color
    trend:        str            # RISING / STABLE / FALLING
    trend_delta:  Optional[float] = None   # % change vs previous
    confidence:   float = 0.0   # 0–1
    drivers:      list  = field(default_factory=list)   # list of driver strings
    profile:      str   = ""
    explanation:  str   = ""     # filled by Claude
    recommendation: str = ""     # filled by Claude
    zone:         str   = ""

# ==============================================================================
# Core engine — HIDDEN LOGIC
# ==============================================================================
def _normalize_wqi(wqi_0_100: float) -> float:
    """Convert WQI (higher=better) to risk signal (higher=worse)."""
    return max(0.0, min(1.0, 1.0 - (wqi_0_100 / 100.0)))

def _signal_risk(value: float, thresholds: dict) -> float:
    """
    Convert a normalized signal value to a risk score 0–1.
    Uses sigmoid-like scaling around thresholds.
    """
    lo, mid, hi = thresholds["low"], thresholds["mid"], thresholds["high"]
    if value <= lo:
        return value / lo * 0.33
    elif value <= mid:
        return 0.33 + (value - lo) / (mid - lo) * 0.34
    elif value <= hi:
        return 0.67 + (value - mid) / (hi - mid) * 0.20
    else:
        # Beyond high threshold — sigmoid push toward 1.0
        excess = (value - hi) / (1.0 - hi + 1e-6)
        return 0.87 + 0.13 * (1 - math.exp(-3 * excess))

def _confidence_from_signals(signals: list[SignalReading]) -> float:
    """Confidence = weighted average of signal confidences, penalised by age."""
    if not signals:
        return 0.0
    total = 0.0
    for s in signals:
        age_penalty = math.exp(-0.2 * s.age_days)
        total += s.confidence * age_penalty
    return round(total / len(signals), 2)

def _detect_drivers(signal_risks: dict, threshold: float = 0.55) -> list[str]:
    """Return signals that are above the driver threshold, sorted by severity."""
    labels = {
        "wqi":              "water quality degradation",
        "turbidity":        "turbidity anomaly",
        "chlorophyll":      "algae/bloom signal",
        "sst_anomaly":      "sea surface temperature anomaly",
        "vessel_density":   "elevated vessel density",
        "oil_proxy":        "oil/discharge signal",
        "night_activity":   "anomalous night-time activity",
        "coastline_change": "coastline dynamic change",
    }
    drivers = [
        (labels.get(k, k), v)
        for k, v in signal_risks.items()
        if v >= threshold
    ]
    drivers.sort(key=lambda x: x[1], reverse=True)
    return [d[0] for d in drivers]

def compute_medi(
    signals:        dict[str, SignalReading],
    profile_name:   str,
    previous_score: Optional[float] = None,
    zone:           str = "",
) -> MEDIResult:
    """
    Main MEDI computation.
    
    Parameters
    ----------
    signals       : dict of signal name → SignalReading
    profile_name  : one of PROFILES keys
    previous_score: MEDI score from previous period (for trend)
    zone          : label for the geographic area
    
    Returns
    -------
    MEDIResult (explanation and recommendation filled later by Claude)
    """
    profile = PROFILES.get(profile_name, PROFILES["Beach Safety"])
    profile_signals = profile["signals"]
    profile_weights = profile["weights"]

    # --- Compute per-signal risk scores ---
    signal_risks = {}
    active_signals = []

    for sig_name, weight in zip(profile_signals, profile_weights):
        reading = signals.get(sig_name)
        if reading is None:
            continue

        # WQI is inverted (higher = better → convert to risk)
        val = _normalize_wqi(reading.value) if sig_name == "wqi" else reading.value
        val = max(0.0, min(1.0, val))

        t = _T.get(sig_name, {"low": 0.3, "mid": 0.55, "high": 0.75})
        risk = _signal_risk(val, t)
        signal_risks[sig_name] = risk
        active_signals.append(reading)

    if not signal_risks:
        return MEDIResult(
            risk_score=0, risk_level="UNKNOWN", risk_color="#888888",
            trend="STABLE", confidence=0.0, profile=profile_name, zone=zone
        )

    # --- Weighted fusion (weights normalised to active signals) ---
    total_weight = sum(
        w for sn, w in zip(profile_signals, profile_weights)
        if sn in signal_risks
    )
    weighted_sum = sum(
        signal_risks[sn] * w
        for sn, w in zip(profile_signals, profile_weights)
        if sn in signal_risks
    )
    base_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    # --- Worst-case amplifier: if any signal is critical, boost score ---
    max_risk = max(signal_risks.values())
    if max_risk > 0.85:
        base_score = base_score * 0.6 + max_risk * 0.4   # pull toward worst case

    risk_score = round(base_score * 100, 1)

    # --- Risk level & color ---
    if risk_score < 25:
        level, color = "LOW",      "#1ecb7b"
    elif risk_score < 45:
        level, color = "MODERATE", "#7ecb1e"
    elif risk_score < 62:
        level, color = "ELEVATED", "#f0a500"
    elif risk_score < 78:
        level, color = "HIGH",     "#e07b00"
    else:
        level, color = "CRITICAL", "#e03c3c"

    # --- Trend ---
    if previous_score is None:
        trend, delta = "STABLE", None
    else:
        delta = round(risk_score - previous_score, 1)
        if delta > 4:
            trend = "RISING"
        elif delta < -4:
            trend = "FALLING"
        else:
            trend = "STABLE"

    # --- Drivers ---
    drivers = _detect_drivers(signal_risks)

    # --- Confidence ---
    confidence = _confidence_from_signals(active_signals)

    return MEDIResult(
        risk_score   = risk_score,
        risk_level   = level,
        risk_color   = color,
        trend        = trend,
        trend_delta  = delta,
        confidence   = confidence,
        drivers      = drivers,
        profile      = profile_name,
        zone         = zone,
    )
