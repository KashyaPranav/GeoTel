"""
GeoSpace: Multi-Temporal Coastal Environmental Change Analysis

Streamlit dashboard implementing the full analytical framework described in:
"Multi-Temporal Coastal Environmental Change Analysis Using Sentinel-2
 Imagery and Spectral Indices"

Pipeline:
  1. Satellite data acquisition (GEE / synthetic fallback)
  2. Preprocessing and spatial clipping
  3. Spectral index computation (SI, NDVI, NDBI)
  4. Year-to-year comparison and differencing
  5. Statistical summarisation and interpretation
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from geopy.geocoders import Nominatim

from src.gee_processor import (
    init_gee,
    fetch_sentinel_data,
    generate_synthetic_data,
    coords_to_bounds,
    LOCATIONS,
)
from src.analysis import compare_years, multi_location_summary, INTERPRETATIONS

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoSpace - Coastal Environmental Analysis",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .metric-label { font-size: 0.85rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Geocoding helper
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def geocode_query(query):
    """Return list of (display_name, lat, lon, boundingbox) for a query."""
    geolocator = Nominatim(user_agent="geospace_app")
    try:
        results = geolocator.geocode(query, exactly_one=False, limit=5)
    except Exception:
        return []
    if not results:
        return []
    out = []
    for r in results:
        out.append(
            {
                "display": r.address,
                "lat": r.latitude,
                "lon": r.longitude,
                "bbox": r.raw.get("boundingbox"),  # [south, north, west, east]
            }
        )
    return out


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("GeoSpace")
st.sidebar.caption("Multi-Temporal Coastal Environmental Change Analysis")
st.sidebar.markdown("---")

location_mode = st.sidebar.radio(
    "Location Mode",
    ["Preset (Chennai)", "Search Global"],
)

# ---- location selection depending on mode ----
selected_locations = []
custom_bounds = {}  # loc_name -> [w, s, e, n]

if location_mode == "Preset (Chennai)":
    locations_available = list(LOCATIONS.keys())
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Single Location", "Multi-Location Comparison"],
        help="Single location shows detailed maps; multi-location produces a summary table.",
    )
    if analysis_mode == "Single Location":
        selected_locations = [
            st.sidebar.selectbox("Location", locations_available)
        ]
    else:
        selected_locations = st.sidebar.multiselect(
            "Locations",
            locations_available,
            default=locations_available,
        )
        if not selected_locations:
            selected_locations = locations_available

else:  # Search Global
    analysis_mode = "Single Location"
    search_query = st.sidebar.text_input(
        "Type a location name",
        placeholder="e.g. Mumbai coast, Tokyo Bay, Miami Beach",
    )

    if search_query:
        suggestions = geocode_query(search_query)
        if suggestions:
            options = [s["display"] for s in suggestions]
            chosen = st.sidebar.selectbox("Select from results", options)
            chosen_data = suggestions[options.index(chosen)]

            # Build bounding box from geocoded result
            if chosen_data["bbox"]:
                bb = chosen_data["bbox"]  # [south, north, west, east] as strings
                bounds = [
                    float(bb[2]),
                    float(bb[0]),
                    float(bb[3]),
                    float(bb[1]),
                ]
                # If the bbox is very large (e.g. whole country), constrain it
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                if width > 0.06 or height > 0.06:
                    bounds = coords_to_bounds(
                        chosen_data["lat"], chosen_data["lon"], radius_km=1.5
                    )
            else:
                bounds = coords_to_bounds(
                    chosen_data["lat"], chosen_data["lon"], radius_km=1.5
                )

            loc_label = chosen.split(",")[0].strip()
            selected_locations = [loc_label]
            custom_bounds[loc_label] = bounds

            st.sidebar.caption(
                f"Lat: {chosen_data['lat']:.4f}, Lon: {chosen_data['lon']:.4f}"
            )
        else:
            st.sidebar.warning("No results found. Try a different query.")

st.sidebar.markdown("---")
st.sidebar.subheader("Temporal Settings")

col_y1, col_y2 = st.sidebar.columns(2)
with col_y1:
    baseline_year = st.number_input("Baseline Year", 2017, 2025, 2019)
with col_y2:
    current_year = st.number_input("Current Year", 2017, 2025, 2024)

month_range = st.sidebar.slider(
    "Month Window",
    1,
    12,
    (1, 12),
    help="Wider windows yield more cloud-free composites. Use the same window for both years to minimise seasonal bias.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Google Earth Engine")
gee_project = st.sidebar.text_input(
    "GCP Project ID",
    value="geospace-488623",
    help="Your Google Cloud project ID with Earth Engine API enabled.",
)

st.sidebar.markdown("---")

# ---- SEARCH BUTTON ----
search_clicked = st.sidebar.button("Search", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data**: Sentinel-2 MSI L2A (GEE)  \n"
    "**Indices**: SI, NDVI, NDBI  \n"
    "**Method**: Median composite + pixel-wise differencing"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INDEX_CMAPS = {
    "si": "YlOrRd",
    "ndvi": "RdYlGn",
    "ndbi": "RdBu_r",
}

INDEX_LABELS = {
    "si": "Salinity Index (SI)",
    "ndvi": "Vegetation Index (NDVI)",
    "ndbi": "Built-up Index (NDBI)",
}


def _heatmap_fig(arr, title, colorscale, zmin=None, zmax=None):
    """Create a Plotly heatmap figure for a 2D index array."""
    fig = go.Figure(
        data=go.Heatmap(
            z=np.flipud(arr),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Value"),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=10, r=10, t=35, b=10),
        height=350,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _delta_heatmap(delta_arr, title):
    """Heatmap for the change (delta) map, centred on zero."""
    absmax = max(abs(np.nanmin(delta_arr)), abs(np.nanmax(delta_arr)), 1e-6)
    fig = go.Figure(
        data=go.Heatmap(
            z=np.flipud(delta_arr),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-absmax,
            zmax=absmax,
            colorbar=dict(title="Delta"),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=10, r=10, t=35, b=10),
        height=350,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _histogram_fig(delta_arr, index_label):
    """Histogram of pixel-wise change values."""
    flat = delta_arr.flatten()
    flat = flat[np.isfinite(flat)]
    fig = go.Figure(data=go.Histogram(x=flat, nbinsx=60, marker_color="#4a90d9"))
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title=dict(text=f"Distribution of Change in {index_label}", font=dict(size=14)),
        xaxis_title="Delta I",
        yaxis_title="Pixel Count",
        margin=dict(l=40, r=10, t=35, b=40),
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("Multi-Temporal Coastal Environmental Change Analysis")
st.caption("Sentinel-2 MSI imagery | Spectral indices | Year-to-year comparison")

# ---------------------------------------------------------------------------
# Data acquisition (only when Search is clicked)
# ---------------------------------------------------------------------------

if search_clicked:
    if not selected_locations:
        st.warning("Please select or search for a location first.")
        st.stop()

    # Initialise GEE
    gee_ok = init_gee(project=gee_project if gee_project else None)
    if gee_ok:
        st.success("Google Earth Engine initialised - using real Sentinel-2 data")
    else:
        st.warning(
            "Google Earth Engine unavailable. Using synthetic demo data.  \n"
            "Run `earthengine authenticate` in your terminal to enable real data."
        )

    all_results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = len(selected_locations) * 2
    step = 0

    for loc in selected_locations:
        loc_bounds = custom_bounds.get(loc)  # None for presets

        # Baseline
        status_text.text(f"Fetching baseline ({baseline_year}) for {loc}...")
        if gee_ok:
            data_b = fetch_sentinel_data(
                location_name=loc,
                year=baseline_year,
                month=month_range[0],
                month_end=month_range[1],
                bounds=loc_bounds,
            )
        else:
            data_b = None
        if data_b is None:
            data_b = generate_synthetic_data(loc, baseline_year)
        step += 1
        progress_bar.progress(step / total_steps)

        # Current
        status_text.text(f"Fetching current ({current_year}) for {loc}...")
        if gee_ok:
            data_c = fetch_sentinel_data(
                location_name=loc,
                year=current_year,
                month=month_range[0],
                month_end=month_range[1],
                bounds=loc_bounds,
            )
        else:
            data_c = None
        if data_c is None:
            data_c = generate_synthetic_data(loc, current_year)
        step += 1
        progress_bar.progress(step / total_steps)

        # Ensure matching spatial dimensions
        min_h = min(data_b["blue"].shape[0], data_c["blue"].shape[0])
        min_w = min(data_b["blue"].shape[1], data_c["blue"].shape[1])
        for k in ("blue", "green", "red", "nir", "swir"):
            data_b[k] = data_b[k][:min_h, :min_w]
            data_c[k] = data_c[k][:min_h, :min_w]

        all_results[loc] = compare_years(data_b, data_c)

    progress_bar.empty()
    status_text.empty()

    # Store results in session state so they persist across reruns
    st.session_state["all_results"] = all_results
    st.session_state["selected_locations"] = selected_locations
    st.session_state["analysis_mode"] = analysis_mode
    st.session_state["baseline_year"] = baseline_year
    st.session_state["current_year"] = current_year
    st.session_state["month_range"] = month_range

# ---------------------------------------------------------------------------
# Display results (from session state)
# ---------------------------------------------------------------------------

if "all_results" not in st.session_state:
    st.info("Configure your location and parameters in the sidebar, then click **Search** to begin analysis.")
    st.stop()

all_results = st.session_state["all_results"]
stored_locations = st.session_state["selected_locations"]
stored_mode = st.session_state["analysis_mode"]
stored_baseline = st.session_state["baseline_year"]
stored_current = st.session_state["current_year"]
stored_months = st.session_state["month_range"]

if stored_mode == "Multi-Location Comparison":
    # -----------------------------------------------------------------------
    # Multi-location summary table
    # -----------------------------------------------------------------------
    st.header("Cross-Location Comparison Summary")
    st.caption(f"Baseline: {stored_baseline} | Current: {stored_current} | Months: {stored_months[0]}-{stored_months[1]}")

    rows = multi_location_summary(all_results)
    df = pd.DataFrame(rows)

    def _color_change(row):
        """Apply red/green to Mean Change based on index type."""
        val = row["Mean Change"]
        idx = row["Index"]
        if not isinstance(val, (int, float)):
            return ""
        # NDVI: increase is good (green), decrease is bad (red)
        # SI & NDBI: increase is bad (red), decrease is good (green)
        if idx == "Vegetation Index (NDVI)":
            return "color: green" if val > 0 else ("color: red" if val < 0 else "")
        else:
            return "color: red" if val > 0 else ("color: green" if val < 0 else "")

    st.dataframe(
        df.style.format(
            {
                "Mean Baseline": "{:.6f}",
                "Mean Current": "{:.6f}",
                "Mean Change": "{:+.6f}",
                "% Change": "{:+.2f}",
                "RMS Change": "{:.6f}",
                "Pixels Increased (%)": "{:.1f}",
                "Pixels Decreased (%)": "{:.1f}",
            }
        ).apply(
            lambda row: [""] * len(row) if "Mean Change" not in row.index
            else [
                _color_change(row) if col == "Mean Change" or col == "% Change" else ""
                for col in row.index
            ],
            axis=1,
        ),
        use_container_width=True,
        height=min(35 * len(rows) + 38, 600),
    )

    # Grouped bar chart
    st.subheader("Percentage Change by Location")
    fig = px.bar(
        df,
        x="Location",
        y="% Change",
        color="Index",
        barmode="group",
        color_discrete_sequence=["#d9534f", "#5cb85c", "#5bc0de"],
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=400, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    # Interpretations
    st.subheader("Physics-Based Interpretations")
    for loc in stored_locations:
        with st.expander(f"{loc}", expanded=False):
            for idx_key in ("si", "ndvi", "ndbi"):
                s = all_results[loc][idx_key]["stats"]
                st.markdown(
                    f"**{INDEX_LABELS[idx_key]}**: {s['interpretation']}  \n"
                    f"Mean change: `{s['mean_change']:+.4f}` ({s['pct_change']:+.2f}%)"
                )

else:
    # -----------------------------------------------------------------------
    # Single location detailed view
    # -----------------------------------------------------------------------
    loc = stored_locations[0]
    result = all_results[loc]

    st.header(f"Analysis: {loc}")
    st.caption(f"Baseline: {stored_baseline} | Current: {stored_current} | Months: {stored_months[0]}-{stored_months[1]}")

    # Top-level metrics
    cols = st.columns(3)
    for i, idx_key in enumerate(("si", "ndvi", "ndbi")):
        s = result[idx_key]["stats"]
        with cols[i]:
            label = INDEX_LABELS[idx_key]
            delta_str = f"{s['pct_change']:+.2f}%"
            delta_color = "inverse" if idx_key in ("si", "ndbi") else "normal"
            st.metric(label, f"{s['mean_current']:.6f}", delta=delta_str, delta_color=delta_color)

    st.markdown("---")

    # Tabs per index
    tabs = st.tabs([INDEX_LABELS[k] for k in ("si", "ndvi", "ndbi")])

    for tab, idx_key in zip(tabs, ("si", "ndvi", "ndbi")):
        with tab:
            s = result[idx_key]["stats"]
            cmap = INDEX_CMAPS[idx_key]
            label = INDEX_LABELS[idx_key]

            formulas = {
                "si": "SI = sqrt(Blue x Red)",
                "ndvi": "NDVI = (NIR - Red) / (NIR + Red)",
                "ndbi": "NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)",
            }
            st.markdown(f"**Formula**: `{formulas[idx_key]}`")

            # Row 1: Baseline | Current | Change maps
            c1, c2, c3 = st.columns(3)
            vmin = min(np.nanmin(result[idx_key]["baseline"]), np.nanmin(result[idx_key]["current"]))
            vmax = max(np.nanmax(result[idx_key]["baseline"]), np.nanmax(result[idx_key]["current"]))

            with c1:
                st.plotly_chart(
                    _heatmap_fig(result[idx_key]["baseline"], f"Baseline ({stored_baseline})", cmap, vmin, vmax),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    _heatmap_fig(result[idx_key]["current"], f"Current ({stored_current})", cmap, vmin, vmax),
                    use_container_width=True,
                )
            with c3:
                st.plotly_chart(
                    _delta_heatmap(s["delta_map"], f"Change (Delta {label.split('(')[1].rstrip(')')} )"),
                    use_container_width=True,
                )

            # Row 2: Statistics
            st.subheader("Statistical Aggregation")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown("**Baseline**")
                st.metric("Mean", f"{s['mean_baseline']:.6f}")
                st.metric("Median", f"{s['median_baseline']:.6f}")
                st.metric("Std Dev", f"{s['std_baseline']:.6f}")
            with m2:
                st.markdown("**Current**")
                st.metric("Mean", f"{s['mean_current']:.6f}")
                st.metric("Median", f"{s['median_current']:.6f}")
                st.metric("Std Dev", f"{s['std_current']:.6f}")
            with m3:
                st.markdown("**Range**")
                st.metric("Min (Baseline)", f"{s['min_baseline']:.6f}")
                st.metric("Max (Baseline)", f"{s['max_baseline']:.6f}")
                st.metric("Min (Current)", f"{s['min_current']:.6f}")
                st.metric("Max (Current)", f"{s['max_current']:.6f}")
            with m4:
                st.markdown("**Change**")
                st.metric("Mean Delta I", f"{s['mean_change']:+.6f}")
                st.metric("RMS Change", f"{s['rms_change']:.6f}")
                st.metric("% Change", f"{s['pct_change']:+.2f}%")
            with m5:
                st.markdown("**Spatial**")
                sd = s["spatial_distribution"]
                st.metric("Pixels Increased", f"{sd['positive_frac']*100:.1f}%")
                st.metric("Pixels Decreased", f"{sd['negative_frac']*100:.1f}%")
                st.metric("CV Baseline", f"{s['cv_baseline']:.1f}%")
                st.metric("CV Current", f"{s['cv_current']:.1f}%")

            # Row 3: Histogram
            st.plotly_chart(
                _histogram_fig(s["delta_map"], label),
                use_container_width=True,
            )

            # Row 4: Interpretation
            if s["mean_change"] > 0:
                icon = "🔴" if idx_key in ("si", "ndbi") else "🟢"
            elif s["mean_change"] < 0:
                icon = "🟢" if idx_key in ("si", "ndbi") else "🔴"
            else:
                icon = "⚪"
            st.info(f"{icon} **Interpretation**: {s['interpretation']}")

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.header("Summary Report")

    summary_cols = st.columns(3)
    for i, idx_key in enumerate(("si", "ndvi", "ndbi")):
        s = result[idx_key]["stats"]
        with summary_cols[i]:
            st.subheader(INDEX_LABELS[idx_key])
            if s["mean_change"] > 0:
                direction = "INCREASED"
                # SI/NDBI increase is bad (red), NDVI increase is good (green)
                color = "red" if idx_key in ("si", "ndbi") else "green"
            elif s["mean_change"] < 0:
                direction = "DECREASED"
                color = "green" if idx_key in ("si", "ndbi") else "red"
            else:
                direction = "STABLE"
                color = "gray"
            st.markdown(
                f"**Trend**: :{color}[**{direction}**]  \n"
                f"**Delta I**: `{s['mean_change']:+.6f}`  \n"
                f"**% Change**: `{s['pct_change']:+.2f}%`  \n"
                f"**Interpretation**: {s['interpretation']}"
            )

    # Key Findings
    st.subheader("Key Findings")
    findings = []
    si_s = result["si"]["stats"]
    ndvi_s = result["ndvi"]["stats"]
    ndbi_s = result["ndbi"]["stats"]

    if si_s["mean_change"] > 0.005:
        findings.append(
            f"Salinity increased by {si_s['pct_change']:+.2f}% - "
            f"{si_s['interpretation']}"
        )
    elif si_s["mean_change"] < -0.005:
        findings.append(
            f"Salinity decreased by {abs(si_s['pct_change']):.2f}% - "
            f"{si_s['interpretation']}"
        )

    if ndvi_s["mean_change"] > 0.005:
        findings.append(
            f"NDVI improved by {ndvi_s['pct_change']:+.2f}% - "
            f"{ndvi_s['interpretation']}"
        )
    elif ndvi_s["mean_change"] < -0.005:
        findings.append(
            f"NDVI declined by {abs(ndvi_s['pct_change']):.2f}% - "
            f"{ndvi_s['interpretation']}"
        )

    if ndbi_s["mean_change"] > 0.005:
        findings.append(
            f"Built-up area expanded by {ndbi_s['pct_change']:+.2f}% - "
            f"{ndbi_s['interpretation']}"
        )
    elif ndbi_s["mean_change"] < -0.005:
        findings.append(
            f"Built-up area reduced by {abs(ndbi_s['pct_change']):.2f}% - "
            f"{ndbi_s['interpretation']}"
        )

    if findings:
        for f in findings:
            st.markdown(f"- {f}")
    else:
        st.markdown("All indices remain relatively stable between the two periods.")

st.markdown("---")
st.caption(
    f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"Data: ESA Copernicus Sentinel-2 L2A | Framework: GeoSpace"
)
