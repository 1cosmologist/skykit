from math import comb, sqrt

import numpy as np
import pytest

from skykit import (bump_steerable_2d, generate_filter_bank,
                    wavelet_transform_tile)


def test_half_plane_support_peak_and_angular_normalization():
    size = 40
    directions = 4
    bank = generate_filter_bank(size, size, J=2, L=directions,
                                wavelet_type="bump_steerable")
    filters = [np.asarray(item["val"]) for item in bank["psi"][:directions]]
    expected_peak = sqrt(4 ** (directions - 1) /
                         (directions * comb(2 * (directions - 1), directions - 1)))

    # Bin 9 of 40 is 0.225 cycles/pixel = 0.45*pi radians/pixel.
    np.testing.assert_allclose(filters[0][9, 0], expected_peak, rtol=2e-6)
    assert filters[0][31, 0] == 0  # opposite half-plane
    assert filters[0][0, 0] == 0   # zero DC
    assert filters[0][20, 0] == 0  # beyond the 0.9*pi radial support

    # The L directions and their conjugate antipodes have unit angular energy
    # at the radial peak. This is not a multiscale tight-frame assertion.
    energy = sum(abs(f[9, 0]) ** 2 + abs(f[31, 0]) ** 2 for f in filters)
    np.testing.assert_allclose(energy, 1.0, rtol=2e-6)


def test_complex_response_and_dyadic_scaling():
    size = 80
    bank = generate_filter_bank(size, size, J=2, L=4,
                                wavelet_type="bump_steerable")
    first = np.asarray(bank["psi"][0]["val"])
    second = np.asarray(bank["psi"][4]["val"])
    # Peak frequencies are at FFT bins 18 and 9 at successive scales.
    np.testing.assert_allclose(first[18, 0], second[9, 0], rtol=2e-6)
    spatial_filter = np.fft.ifft2(first)
    assert np.max(np.abs(spatial_filter.imag)) > 1e-5

    tile = np.random.default_rng(20).normal(size=(size, size))
    features = wavelet_transform_tile(tile, bank, order=1)
    assert np.iscomplexobj(features["U1"])
    assert np.max(np.abs(features["U1"][0].imag)) > 1e-5


def test_parameter_validation():
    with pytest.raises(ValueError, match="L"):
        bump_steerable_2d(16, 16, j=0, theta=0, L=1)
    with pytest.raises(ValueError, match="xi0"):
        bump_steerable_2d(16, 16, j=0, theta=0, L=4, xi0=0.85 * np.pi)
