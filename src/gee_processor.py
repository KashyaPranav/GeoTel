"""
Google Earth Engine processor for Sentinel-2 MSI Level-2A surface reflectance data.

Handles satellite data acquisition, preprocessing, spatial clipping, and
median composite generation for multi-temporal coastal analysis.
"""

import numpy as np
import ee
import warnings

warnings.filterwarnings("ignore")

# Chennai coastal study area locations with geographic bounds [west, south, east, north]
# Each region covers ~2-3 km on a side to capture meaningful land cover variation
LOCATIONS = {
    "Marina Beach": [80.2650, 13.0400, 80.2900, 13.0620],
    "Besant Nagar": [80.2400, 13.0000, 80.2700, 13.0250],
    "Elliot Beach": [80.2550, 12.9800, 80.2800, 13.0000],
    "Royapuram": [80.2900, 13.1100, 80.3200, 13.1350],
    "Cooum River": [80.2500, 13.0550, 80.2850, 13.0800],
}

# Target scale in metres (Sentinel-2 native for 10m bands)
SCALE = 10


def init_gee(project=None):
    """Initialize Google Earth Engine. Returns True if successful."""
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        return True
    except Exception:
        try:
            ee.Authenticate()
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
            return True
        except Exception:
            return False


def _cloud_mask_s2(image):
    """Apply cloud and cirrus mask using the SCL band (Scene Classification)."""
    scl = image.select("SCL")
    # Keep: vegetation(4), bare soil(5), water(6), snow/ice(11)
    clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(11))
    return image.updateMask(clear)


def coords_to_bounds(lat, lon, radius_km=1.5):
    """
    Create a [west, south, east, north] bounding box from a centre point.

    Parameters
    ----------
    lat, lon : float
        Centre of the region in decimal degrees.
    radius_km : float
        Half-width of the bounding box in kilometres (default 1.5 km).

    Returns
    -------
    list of four floats: [west, south, east, north].
    """
    deg_lat = radius_km / 111.32
    deg_lon = radius_km / (111.32 * np.cos(np.radians(lat)))
    return [lon - deg_lon, lat - deg_lat, lon + deg_lon, lat + deg_lat]


def fetch_sentinel_data(location_name=None, year=2024, month=1, month_end=12,
                        bounds=None):
    """
    Fetch Sentinel-2 Level-2A surface reflectance data from GEE.

    Uses a date window of [year-month, year-month_end] and builds a
    median composite after cloud masking. The composite is reprojected
    to EPSG:4326 at SCALE metres before pixel extraction, ensuring a
    proper raster grid is returned.

    Parameters
    ----------
    location_name : str or None
        Key from LOCATIONS dict. Ignored if *bounds* is provided.
    year : int
        Target year.
    month : int
        Start month of the acquisition window (default 1).
    month_end : int
        End month of the acquisition window (default 12).
    bounds : list or None
        Explicit [west, south, east, north] bounding box. Takes
        priority over *location_name*.

    Returns
    -------
    dict with keys 'blue', 'green', 'red', 'nir' (np.ndarray), 'source', 'count'.
    Returns None if no images are found.
    """
    if bounds is None:
        bounds = LOCATIONS.get(location_name, LOCATIONS["Marina Beach"])
    region = ee.Geometry.Rectangle(bounds)

    start_date = f"{year}-{month:02d}-01"
    if month_end == 12:
        end_date = f"{year}-12-31"
    else:
        end_date = f"{year}-{month_end:02d}-28"

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .select(["B2", "B3", "B4", "B8", "SCL"])
    )

    count = collection.size().getInfo()
    if count == 0:
        return None

    # Apply cloud mask, take median composite, select reflectance bands
    composite = (
        collection.map(_cloud_mask_s2)
        .median()
        .select(["B2", "B3", "B4", "B8"])
    )

    # Reproject to a fixed CRS and scale so sampleRectangle returns a
    # proper pixel grid instead of a single aggregated value
    crs = "EPSG:4326"
    composite = composite.reproject(crs=crs, scale=SCALE)

    # Clip to region and unmask (fill masked pixels with 0)
    composite = composite.clip(region).unmask(0)

    # Download pixel data as numpy arrays
    band_data = composite.sampleRectangle(region=region, defaultValue=0)

    blue = np.array(band_data.get("B2").getInfo(), dtype=np.float64) / 10000.0
    green = np.array(band_data.get("B3").getInfo(), dtype=np.float64) / 10000.0
    red = np.array(band_data.get("B4").getInfo(), dtype=np.float64) / 10000.0
    nir = np.array(band_data.get("B8").getInfo(), dtype=np.float64) / 10000.0

    # Validate we got a meaningful raster
    if blue.size <= 1:
        return None

    return {
        "blue": blue,
        "green": green,
        "red": red,
        "nir": nir,
        "source": "Google Earth Engine",
        "count": count,
    }


def generate_synthetic_data(location_name, year):
    """
    Generate physically plausible synthetic Sentinel-2 data for demo/fallback.

    Simulates coastal features: water, vegetation, urban, and bare soil patches
    with year-dependent salinity drift to demonstrate temporal change detection.
    """
    rng = np.random.RandomState(hash(location_name + str(year)) % 2**31)
    shape = (128, 128)

    # Base reflectance (bare soil / mixed land)
    blue = rng.uniform(0.06, 0.14, shape)
    green = rng.uniform(0.08, 0.18, shape)
    red = rng.uniform(0.08, 0.20, shape)
    nir = rng.uniform(0.20, 0.40, shape)

    # Spatial gradient: left side more marine-influenced
    x_grad = np.broadcast_to(np.linspace(0, 1, shape[1])[None, :], shape)

    # Water body (left edge / coast)
    water = x_grad < 0.15
    blue[water] = rng.uniform(0.06, 0.12, water.sum())
    green[water] = rng.uniform(0.04, 0.08, water.sum())
    red[water] = rng.uniform(0.02, 0.05, water.sum())
    nir[water] = rng.uniform(0.01, 0.04, water.sum())

    # Dense vegetation (right / inland)
    veg = x_grad > 0.7
    blue[veg] = rng.uniform(0.03, 0.06, veg.sum())
    green[veg] = rng.uniform(0.06, 0.12, veg.sum())
    red[veg] = rng.uniform(0.03, 0.06, veg.sum())
    nir[veg] = rng.uniform(0.30, 0.55, veg.sum())

    # Urban patch (center)
    urban = (x_grad > 0.35) & (x_grad < 0.55)
    blue[urban] = rng.uniform(0.10, 0.18, urban.sum())
    green[urban] = rng.uniform(0.10, 0.16, urban.sum())
    red[urban] = rng.uniform(0.12, 0.22, urban.sum())
    nir[urban] = rng.uniform(0.15, 0.25, urban.sum())

    # Temporal drift: salinity increases, vegetation degrades, buildup expands
    drift = (year - 2015) * 0.003
    blue += drift * 0.5
    red += drift * 0.8
    nir -= drift * 0.3  # vegetation stress

    # Expand urban zone slightly each year
    urban_expand = (year - 2015) * 0.005
    expanding = (x_grad > (0.55 - urban_expand)) & (x_grad < (0.55 + urban_expand))
    red[expanding] += 0.02
    nir[expanding] -= 0.02

    # Clip to physical bounds
    blue = np.clip(blue, 0, 1)
    green = np.clip(green, 0, 1)
    red = np.clip(red, 0, 1)
    nir = np.clip(nir, 0, 1)

    return {
        "blue": blue,
        "green": green,
        "red": red,
        "nir": nir,
        "source": "Synthetic (Demo)",
        "count": 0,
    }
