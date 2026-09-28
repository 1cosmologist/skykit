import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from skykit import (FeatureMap, FeatureMapSet, ScatteringStatistics, TileSet,
                    generate_filter_bank, open_feature_maps, scattering_transform,
                    wavelet_transform_tile)


def test_single_tile_paths_and_values():
    size = 32
    bank = generate_filter_bank(size, size, J=2, L=4)
    tile = np.random.default_rng(8).normal(size=(size, size))
    features = wavelet_transform_tile(tile, bank, order=2)
    assert isinstance(features, FeatureMap)
    assert features["U1"].shape == (8, size, size)
    assert features["U2"].shape == (16, size, size)
    assert features.paths["U1"][0] == (0, 0.0)
    assert features.paths["U2"][0] == (0, 0.0, 1, 0.0)
    psi1 = np.asarray(bank["psi"][0]["val"])
    psi2 = np.asarray(bank["psi"][4]["val"])
    u1 = np.abs(np.fft.ifft2(np.fft.fft2(tile) * psi1))
    u2 = np.abs(np.fft.ifft2(np.fft.fft2(u1) * psi2))
    np.testing.assert_allclose(features.feature("U1", (0, 0.0)), u1, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(features.feature("U2", (0, 0.0, 1, 0.0)), u2,
                               rtol=2e-5, atol=2e-6)
    stats = scattering_transform(features)
    assert isinstance(stats, ScatteringStatistics)
    assert stats["U1"].shape == (8,)
    assert stats["U2"].shape == (16,)
    np.testing.assert_allclose(stats.statistic("U1", (0, 0.0)), u1.mean(),
                               rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(stats["U2"], features["U2"].mean(axis=(-2, -1)),
                               rtol=2e-5, atol=2e-6)


def test_tileset_and_cross_statistics():
    size = 16
    bank = generate_filter_bank(size, size, J=2, L=2)
    data = np.random.default_rng(9).normal(size=(3, 2, size, size))
    tileset = TileSet(data, nside=size, tile_nside=size, margin=0, pol=True)
    first = wavelet_transform_tile(tileset, bank, order=2, batch_size=2)
    second = wavelet_transform_tile(tileset, bank, order=1, batch_size=1)
    assert isinstance(first, FeatureMapSet)
    assert first["U1"].shape == (3, 2, 4, size, size)
    assert first["U2"].shape == (3, 2, 4, size, size)
    assert set(second.keys()) == {"U1"}
    stats = scattering_transform(first, second, operation1=jnp.square,
                                 operation2=jnp.abs, reduction="variance",
                                 path_batch_size=2)
    assert set(stats.keys()) == {"U1_U1", "U2_U1"}
    assert stats["U1_U1"].shape == (3, 2, 4, 4)
    expected = np.var(first["U1"][:, :, :, None] ** 2 *
                      np.abs(second["U1"][:, :, None]), axis=(-2, -1))
    np.testing.assert_allclose(stats["U1_U1"], expected, rtol=2e-5, atol=2e-6)
    path1 = first.paths["U1"][0]
    path2 = second.paths["U1"][1]
    assert stats.statistic("U1_U1", path1, path2, tile=1, stokes=0) == stats["U1_U1"][1, 0, 0, 1]


def test_hdf5_round_trip_single_and_set():
    size = 16
    bank = generate_filter_bank(size, size, J=2, L=2)
    rng = np.random.default_rng(13)
    tile = rng.normal(size=(2, size, size))
    tileset = TileSet(rng.normal(size=(3, size, size)),
                      nside=size, tile_nside=size, margin=0)
    with tempfile.TemporaryDirectory() as directory:
        tile_path = Path(directory) / "tile.h5"
        set_path = Path(directory) / "set.h5"
        memory = wavelet_transform_tile(tile, bank, order=2)
        memory.to_hdf5(tile_path)
        with open_feature_maps(tile_path) as loaded:
            assert isinstance(loaded, FeatureMap)
            assert loaded.pol
            assert loaded.paths == memory.paths
            np.testing.assert_allclose(loaded["U1"][:], memory["U1"])
            assert loaded.feature("U1", memory.paths["U1"][0], stokes=1).shape == (size, size)
        with wavelet_transform_tile(tile, bank, order=1, filepath=tile_path) as streamed_tile:
            assert isinstance(streamed_tile, FeatureMap)
            assert set(streamed_tile.keys()) == {"U1"}
            np.testing.assert_allclose(streamed_tile["U1"][:], memory["U1"],
                                       rtol=2e-5, atol=2e-6)
        streamed = wavelet_transform_tile(tileset, bank, order=2,
                                          filepath=set_path, batch_size=2)
        assert isinstance(streamed, FeatureMapSet)
        assert streamed.metadata["nside"] == size
        assert streamed["U1"].chunks[0] == 1
        disk_stats = scattering_transform(streamed)
        memory_stats = scattering_transform(wavelet_transform_tile(tileset, bank, order=2))
        np.testing.assert_allclose(disk_stats["U1"], memory_stats["U1"],
                                   rtol=2e-5, atol=2e-6)
        streamed.close()
        assert scattering_transform(set_path)["U2"].shape == (3, 4)
        assert scattering_transform(set_path, set_path)["U1_U1"].shape == (3, 4, 4)


def test_empty_second_order_and_validation():
    size = 16
    bank = generate_filter_bank(size, size, J=1, L=2)
    feature = wavelet_transform_tile(np.ones((size, size)), bank, order=2)
    assert feature["U2"].shape == (0, size, size)
    assert feature.paths["U2"] == ()
    assert scattering_transform(feature)["U2"].shape == (0,)
    with pytest.raises(ValueError, match="order"):
        wavelet_transform_tile(np.ones((size, size)), bank, order=3)
    with pytest.raises(ValueError, match="matching tile layouts"):
        scattering_transform(feature, wavelet_transform_tile(
            TileSet(np.ones((1, size, size)), nside=size, tile_nside=size, margin=0), bank))
