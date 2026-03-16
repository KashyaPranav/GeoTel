"""
Year-to-year comparison framework for multi-temporal environmental analysis.

Implements Section IV of the methodology:
  - Mathematical change detection (delta I)
  - Percentage change analysis
  - Physics-based interpretation
  - Statistical aggregation
"""

import numpy as np
from src.indices import compute_all_indices

EPS = 1e-10

# Physics-based interpretation rules
INTERPRETATIONS = {
    "si": {
        "name": "Salinity Index",
        "increase": "Enhanced salt accumulation, potentially due to seawater intrusion or reduced drainage",
        "decrease": "Reduced surface salinity, possibly from improved drainage or freshwater recharge",
        "stable": "Salinity levels remain relatively stable",
        "unit": "",
    },
    "ndvi": {
        "name": "Vegetation Index (NDVI)",
        "increase": "Vegetation recovery or improved soil fertility",
        "decrease": "Vegetation stress or soil fertility degradation",
        "stable": "Vegetation health remains relatively stable",
        "unit": "",
    },
    "ndbi": {
        "name": "Built-up Index (NDBI)",
        "increase": "Expansion of impervious and built-up surfaces",
        "decrease": "Reduction in built-up area or increase in vegetation cover",
        "stable": "Built-up extent remains relatively stable",
        "unit": "",
    },
    "ndwi": {
        "name": "Water Index (NDWI)",
        "increase": "Expansion of water bodies or increased surface water presence",
        "decrease": "Reduction in water coverage, possibly from sedimentation or land reclamation",
        "stable": "Water body extent remains relatively stable",
        "unit": "",
    },
    "mndwi": {
        "name": "Modified Water Index (MNDWI)",
        "increase": "Enhanced water detection, better separation from built-up areas",
        "decrease": "Reduced water presence or increased land cover over former water areas",
        "stable": "Modified water index remains relatively stable",
        "unit": "",
    },
}

CHANGE_THRESHOLD = 0.005  # minimum absolute mean change to count as non-stable


def pixel_change(baseline_arr, current_arr):
    """
    Compute pixel-wise change: delta_I = I_current - I_baseline.
    """
    return current_arr - baseline_arr


def percentage_change(baseline_arr, current_arr):
    """
    Compute pixel-wise percentage change:
        %delta_I = (I_current - I_baseline) / I_baseline * 100

    Pixels where baseline is near zero are set to NaN.
    """
    safe_base = np.where(np.abs(baseline_arr) > EPS, baseline_arr, np.nan)
    return (current_arr - baseline_arr) / safe_base * 100.0


def _spatial_distribution(delta_arr):
    """
    Calculate fraction of pixels showing positive, negative, and no change.
    """
    valid = np.isfinite(delta_arr)
    total = valid.sum()
    if total == 0:
        return {"positive_frac": 0.0, "negative_frac": 0.0, "neutral_frac": 1.0}
    pos = (delta_arr[valid] > CHANGE_THRESHOLD).sum()
    neg = (delta_arr[valid] < -CHANGE_THRESHOLD).sum()
    return {
        "positive_frac": float(pos / total),
        "negative_frac": float(neg / total),
        "neutral_frac": float((total - pos - neg) / total),
    }


def compute_statistics(baseline_arr, current_arr, index_key):
    """
    Compute full statistics for one index between baseline and current year.

    Returns a dict containing:
        - mean_baseline, mean_current
        - mean_change (delta I)
        - pct_change (percentage change of means)
        - std_baseline, std_current
        - spatial_distribution (fraction of pixels increasing / decreasing)
        - interpretation (physics-based text)
        - delta_map, pct_map (pixel-wise arrays)
    """
    delta = pixel_change(baseline_arr, current_arr)
    pct = percentage_change(baseline_arr, current_arr)

    mean_b = float(np.nanmean(baseline_arr))
    mean_c = float(np.nanmean(current_arr))
    mean_delta = float(np.nanmean(delta))

    std_b = float(np.nanstd(baseline_arr))
    std_c = float(np.nanstd(current_arr))
    median_b = float(np.nanmedian(baseline_arr))
    median_c = float(np.nanmedian(current_arr))
    min_b = float(np.nanmin(baseline_arr))
    min_c = float(np.nanmin(current_arr))
    max_b = float(np.nanmax(baseline_arr))
    max_c = float(np.nanmax(current_arr))

    # Coefficient of variation (CV = std / |mean|)
    cv_b = (std_b / abs(mean_b) * 100.0) if abs(mean_b) > EPS else 0.0
    cv_c = (std_c / abs(mean_c) * 100.0) if abs(mean_c) > EPS else 0.0

    # Root mean square of the change field
    rms_change = float(np.sqrt(np.nanmean(delta ** 2)))

    # percentage change of the mean values
    if abs(mean_b) > EPS:
        pct_of_mean = (mean_c - mean_b) / mean_b * 100.0
    else:
        pct_of_mean = 0.0

    spatial = _spatial_distribution(delta)

    # Interpretation
    interp_info = INTERPRETATIONS[index_key]
    if abs(mean_delta) < CHANGE_THRESHOLD:
        interpretation = interp_info["stable"]
    elif mean_delta > 0:
        interpretation = interp_info["increase"]
    else:
        interpretation = interp_info["decrease"]

    return {
        "mean_baseline": mean_b,
        "mean_current": mean_c,
        "std_baseline": std_b,
        "std_current": std_c,
        "median_baseline": median_b,
        "median_current": median_c,
        "min_baseline": min_b,
        "min_current": min_c,
        "max_baseline": max_b,
        "max_current": max_c,
        "cv_baseline": cv_b,
        "cv_current": cv_c,
        "mean_change": mean_delta,
        "rms_change": rms_change,
        "pct_change": pct_of_mean,
        "spatial_distribution": spatial,
        "interpretation": interpretation,
        "delta_map": delta,
        "pct_map": pct,
    }


def compare_years(data_baseline, data_current):
    """
    Run full year-to-year comparison for all three indices.

    Parameters
    ----------
    data_baseline : dict
        Band data for the baseline year (blue, red, nir, etc.)
    data_current : dict
        Band data for the current year.

    Returns
    -------
    dict keyed by index name ('si', 'ndvi', 'ndbi'), each containing:
        - 'baseline': 2D array of index values
        - 'current': 2D array of index values
        - 'stats': statistics dict from compute_statistics
    """
    idx_b = compute_all_indices(data_baseline)
    idx_c = compute_all_indices(data_current)

    results = {}
    for key in ("si", "ndvi", "ndbi", "ndwi", "mndwi"):
        stats = compute_statistics(idx_b[key], idx_c[key], key)
        results[key] = {
            "baseline": idx_b[key],
            "current": idx_c[key],
            "stats": stats,
        }

    return results


def multi_location_summary(all_results):
    """
    Aggregate comparison results across multiple locations into a summary table.

    Parameters
    ----------
    all_results : dict
        {location_name: compare_years result, ...}

    Returns
    -------
    list of dicts suitable for pandas DataFrame construction.
    """
    rows = []
    for loc, result in all_results.items():
        for idx_key in ("si", "ndvi", "ndbi", "ndwi", "mndwi"):
            s = result[idx_key]["stats"]
            rows.append(
                {
                    "Location": loc,
                    "Index": INTERPRETATIONS[idx_key]["name"],
                    "Mean Baseline": round(s["mean_baseline"], 6),
                    "Mean Current": round(s["mean_current"], 6),
                    "Mean Change": round(s["mean_change"], 6),
                    "% Change": round(s["pct_change"], 2),
                    "RMS Change": round(s["rms_change"], 6),
                    "Pixels Increased (%)": round(s["spatial_distribution"]["positive_frac"] * 100, 1),
                    "Pixels Decreased (%)": round(s["spatial_distribution"]["negative_frac"] * 100, 1),
                    "Interpretation": s["interpretation"],
                }
            )
    return rows
