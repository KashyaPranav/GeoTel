"""
Spectral index computation for coastal environmental analysis.

Implements physics-based spectral indices derived from Sentinel-2 band
reflectance values:
  - Salinity Index (SI)
  - Normalized Difference Vegetation Index (NDVI)
  - Normalized Difference Built-up Index (NDBI) using SWIR1
  - Normalized Difference Water Index (NDWI)
  - Modified Normalized Difference Water Index (MNDWI)
"""

import numpy as np

EPS = 1e-10  # numerical guard against division by zero


def salinity_index(blue, red):
    """
    Compute Salinity Index.

    SI = sqrt(Blue * Red)

    Higher values indicate increased surface salinity (salt stress in soil),
    often associated with seawater intrusion or salt accumulation.

    Reference bands: B2 (490nm), B4 (665nm)
    """
    return np.sqrt(np.maximum(blue * red, 0.0))


def ndvi(nir, red):
    """
    Compute Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    Healthy vegetation strongly reflects NIR and absorbs red light,
    giving higher NDVI. Values range from -1 to +1.

    Reference bands: B8 (842nm), B4 (665nm)
    """
    denom = nir + red
    return np.where(denom > EPS, (nir - red) / denom, 0.0)


def ndbi(swir, nir):
    """
    Compute Normalized Difference Built-up Index.

    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)

    Built-up and impervious surfaces reflect more SWIR radiation relative
    to NIR compared to vegetation. Positive values typically indicate
    urban or constructed areas.

    Reference: Zha, Gao & Ni (2003)
    Reference bands: B11 (1610nm), B8 (842nm)
    """
    denom = swir + nir
    return np.where(denom > EPS, (swir - nir) / denom, 0.0)


def ndwi(green, nir):
    """
    Compute Normalized Difference Water Index.

    NDWI = (Green - NIR) / (Green + NIR)

    Positive values indicate water bodies. Higher values suggest
    open water surfaces.

    Reference: McFeeters (1996)
    Reference bands: B3 (560nm), B8 (842nm)
    """
    denom = green + nir
    return np.where(denom > EPS, (green - nir) / denom, 0.0)


def mndwi(green, swir):
    """
    Compute Modified Normalized Difference Water Index.

    MNDWI = (Green - SWIR1) / (Green + SWIR1)

    More effective than NDWI at distinguishing water from built-up
    areas due to SWIR absorption by water.

    Reference: Xu (2006)
    Reference bands: B3 (560nm), B11 (1610nm)
    """
    denom = green + swir
    return np.where(denom > EPS, (green - swir) / denom, 0.0)


def compute_all_indices(band_data):
    """
    Compute SI, NDVI, NDBI, NDWI, and MNDWI from a band data dictionary.

    Parameters
    ----------
    band_data : dict
        Must contain 'blue', 'green', 'red', 'nir', 'swir' keys with np.ndarray values.

    Returns
    -------
    dict with 'si', 'ndvi', 'ndbi', 'ndwi', 'mndwi' arrays.
    """
    blue = band_data["blue"]
    green = band_data["green"]
    red = band_data["red"]
    nir = band_data["nir"]
    swir = band_data["swir"]

    return {
        "si": salinity_index(blue, red),
        "ndvi": ndvi(nir, red),
        "ndbi": ndbi(swir, nir),
        "ndwi": ndwi(green, nir),
        "mndwi": mndwi(green, swir),
    }
