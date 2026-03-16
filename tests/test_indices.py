"""Unit tests for spectral index computation."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.indices import salinity_index, ndvi, ndbi, ndwi, mndwi, compute_all_indices


class TestSalinityIndex:
    def test_basic(self):
        blue = np.array([[0.1, 0.2], [0.3, 0.4]])
        red = np.array([[0.2, 0.3], [0.1, 0.5]])
        result = salinity_index(blue, red)
        expected = np.sqrt(blue * red)
        np.testing.assert_allclose(result, expected)

    def test_zero_inputs(self):
        blue = np.zeros((2, 2))
        red = np.zeros((2, 2))
        result = salinity_index(blue, red)
        np.testing.assert_allclose(result, 0.0)

    def test_negative_clamped(self):
        blue = np.array([[-0.1]])
        red = np.array([[0.2]])
        result = salinity_index(blue, red)
        assert result[0, 0] == 0.0


class TestNDVI:
    def test_healthy_vegetation(self):
        nir = np.array([[0.5]])
        red = np.array([[0.1]])
        result = ndvi(nir, red)
        assert result[0, 0] == pytest.approx((0.5 - 0.1) / (0.5 + 0.1))

    def test_bare_soil(self):
        nir = np.array([[0.2]])
        red = np.array([[0.2]])
        result = ndvi(nir, red)
        assert result[0, 0] == pytest.approx(0.0)

    def test_range(self):
        nir = np.random.uniform(0, 1, (10, 10))
        red = np.random.uniform(0, 1, (10, 10))
        result = ndvi(nir, red)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_zero_denominator(self):
        nir = np.array([[0.0]])
        red = np.array([[0.0]])
        result = ndvi(nir, red)
        assert result[0, 0] == 0.0


class TestNDBI:
    def test_urban(self):
        swir = np.array([[0.3]])
        nir = np.array([[0.15]])
        result = ndbi(swir, nir)
        assert result[0, 0] == pytest.approx((0.3 - 0.15) / (0.3 + 0.15))

    def test_vegetation_negative(self):
        swir = np.array([[0.1]])
        nir = np.array([[0.5]])
        result = ndbi(swir, nir)
        assert result[0, 0] < 0


class TestNDWI:
    def test_water_positive(self):
        green = np.array([[0.1]])
        nir = np.array([[0.02]])
        result = ndwi(green, nir)
        assert result[0, 0] > 0

    def test_land_negative(self):
        green = np.array([[0.1]])
        nir = np.array([[0.4]])
        result = ndwi(green, nir)
        assert result[0, 0] < 0


class TestMNDWI:
    def test_water_positive(self):
        green = np.array([[0.1]])
        swir = np.array([[0.01]])
        result = mndwi(green, swir)
        assert result[0, 0] > 0

    def test_buildup_negative(self):
        green = np.array([[0.1]])
        swir = np.array([[0.3]])
        result = mndwi(green, swir)
        assert result[0, 0] < 0


class TestComputeAllIndices:
    def test_all_keys(self):
        band_data = {
            "blue": np.random.uniform(0, 0.3, (5, 5)),
            "green": np.random.uniform(0, 0.3, (5, 5)),
            "red": np.random.uniform(0, 0.3, (5, 5)),
            "nir": np.random.uniform(0.1, 0.5, (5, 5)),
            "swir": np.random.uniform(0.1, 0.4, (5, 5)),
        }
        result = compute_all_indices(band_data)
        assert set(result.keys()) == {"si", "ndvi", "ndbi", "ndwi", "mndwi"}
        for v in result.values():
            assert v.shape == (5, 5)
