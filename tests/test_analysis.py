"""Unit tests for the year-to-year analysis framework."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis import (
    pixel_change,
    percentage_change,
    compute_statistics,
    compare_years,
)


class TestPixelChange:
    def test_basic(self):
        baseline = np.array([[1.0, 2.0], [3.0, 4.0]])
        current = np.array([[1.5, 2.5], [2.5, 4.5]])
        delta = pixel_change(baseline, current)
        expected = np.array([[0.5, 0.5], [-0.5, 0.5]])
        np.testing.assert_allclose(delta, expected)

    def test_no_change(self):
        arr = np.ones((3, 3))
        delta = pixel_change(arr, arr)
        np.testing.assert_allclose(delta, 0.0)


class TestPercentageChange:
    def test_basic(self):
        baseline = np.array([[1.0, 2.0]])
        current = np.array([[1.1, 2.4]])
        pct = percentage_change(baseline, current)
        np.testing.assert_allclose(pct, [[10.0, 20.0]])

    def test_zero_baseline(self):
        baseline = np.array([[0.0]])
        current = np.array([[1.0]])
        pct = percentage_change(baseline, current)
        assert np.isnan(pct[0, 0])


class TestComputeStatistics:
    def test_keys(self):
        baseline = np.random.uniform(0.1, 0.3, (10, 10))
        current = np.random.uniform(0.1, 0.3, (10, 10))
        stats = compute_statistics(baseline, current, "si")
        required_keys = [
            "mean_baseline", "mean_current", "std_baseline", "std_current",
            "median_baseline", "median_current", "min_baseline", "min_current",
            "max_baseline", "max_current", "cv_baseline", "cv_current",
            "mean_change", "rms_change", "pct_change",
            "spatial_distribution", "interpretation", "delta_map", "pct_map",
        ]
        for k in required_keys:
            assert k in stats, f"Missing key: {k}"

    def test_increase_interpretation(self):
        baseline = np.full((5, 5), 0.1)
        current = np.full((5, 5), 0.2)
        stats = compute_statistics(baseline, current, "si")
        assert "accumulation" in stats["interpretation"].lower() or "salt" in stats["interpretation"].lower()

    def test_spatial_fractions_sum_to_one(self):
        baseline = np.random.uniform(0, 1, (10, 10))
        current = np.random.uniform(0, 1, (10, 10))
        stats = compute_statistics(baseline, current, "ndvi")
        sd = stats["spatial_distribution"]
        total = sd["positive_frac"] + sd["negative_frac"] + sd["neutral_frac"]
        assert total == pytest.approx(1.0)


class TestCompareYears:
    def test_returns_all_indices(self):
        bands = {
            "blue": np.random.uniform(0, 0.3, (10, 10)),
            "green": np.random.uniform(0, 0.3, (10, 10)),
            "red": np.random.uniform(0, 0.3, (10, 10)),
            "nir": np.random.uniform(0.1, 0.5, (10, 10)),
            "swir": np.random.uniform(0.1, 0.4, (10, 10)),
        }
        result = compare_years(bands, bands)
        assert set(result.keys()) == {"si", "ndvi", "ndbi", "ndwi", "mndwi"}
