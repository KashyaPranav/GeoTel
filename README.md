# GeoSpace

Multi-Temporal Coastal Environmental Change Analysis using Sentinel-2 imagery and spectral indices.

## Overview

GeoSpace is a Streamlit-based dashboard for monitoring environmental changes in coastal regions using satellite remote sensing. It processes Sentinel-2 MSI Level-2A surface reflectance data via Google Earth Engine to compute spectral indices and perform year-to-year change detection.

## Features

- **5 Spectral Indices**: SI (Salinity), NDVI (Vegetation), NDBI (Built-up), NDWI (Water), MNDWI (Modified Water)
- **Multi-Temporal Analysis**: Compare baseline and current years with optional multi-year time series
- **Geographic Map Overlays**: Folium-based maps with index overlays on real basemaps
- **Area Calculations**: Change extent computed in sq km
- **Cross-Index Correlation**: Pearson correlation between index change maps
- **Export**: PDF reports, GeoTIFF rasters, CSV statistics
- **Global Search**: Geocoding support for any location worldwide
- **Preset Locations**: 5 Chennai coastal sites (Marina Beach, Besant Nagar, Elliot Beach, Royapuram, Cooum River)
- **Data Caching**: Cached GEE fetches to avoid redundant downloads

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Earth Engine (optional - synthetic data fallback available)
earthengine authenticate

# Run the dashboard
streamlit run app/main.py
```

## Project Structure

```
GeoSpace/
  app/
    main.py              # Streamlit dashboard
  src/
    __init__.py
    indices.py           # Spectral index computation (SI, NDVI, NDBI, NDWI, MNDWI)
    gee_processor.py     # Google Earth Engine data acquisition
    analysis.py          # Year-to-year comparison framework
  tests/
    test_indices.py      # Unit tests for spectral indices
    test_analysis.py     # Unit tests for analysis framework
  requirements.txt
  README.md
```

## Running Tests

```bash
pytest tests/ -v
```

## Data Source

ESA Copernicus Sentinel-2 MSI Level-2A surface reflectance (COPERNICUS/S2_SR_HARMONIZED) via Google Earth Engine.
