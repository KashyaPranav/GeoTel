"""
Spectral index computation for coastal environmental analysis.

Implements physics-based spectral indices derived from Sentinel-2 band
reflectance values:
  - Salinity Index (SI)
  - Normalized Difference Vegetation Index (NDVI)
  - Normalized Difference Built-up Index (NDBI)
"""

import numpy as np

EPS = 1e-10  # numerical guard against division by zero


def salinity_index(blue, red):
    """
    Compute Salinity Index.

    SI = sqrt(Blue * Red)

    Higher values indicate increased surface salinity (salt stress in soil),
    often associated with seawater intrusion or salt accumulation.
    """
    return np.sqrt(np.maximum(blue * red, 0.0))


def ndvi(nir, red):
    """
    Compute Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    Healthy vegetation strongly reflects NIR and absorbs red light,
    giving higher NDVI. Values range from -1 to +1.
    """
    denom = nir + red
    return np.where(denom > EPS, (nir - red) / denom, 0.0)


def ndbi(red, nir):
    """
    Compute Normalized Difference Built-up Index.

    NDBI = (Red - NIR) / (Red + NIR)

    Positive values correspond to urban or impervious surfaces.
    Equivalent to -NDVI; separated for semantic clarity.
    """
    denom = red + nir
    return np.where(denom > EPS, (red - nir) / denom, 0.0)


def compute_all_indices(band_data):
    """
    Compute SI, NDVI, and NDBI from a band data dictionary.

    Parameters
    ----------
    band_data : dict
        Must contain 'blue', 'red', 'nir' keys with np.ndarray values.

    Returns
    -------
    dict with 'si', 'ndvi', 'ndbi' arrays.
    """
    blue = band_data["blue"]
    red = band_data["red"]
    nir = band_data["nir"]

    return {
        "si": salinity_index(blue, red),
        "ndvi": ndvi(nir, red),
        "ndbi": ndbi(red, nir),
    }
