"""Undecimated wavelet feature maps for tiles and TileSets."""

from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from .feature_maps import FeatureMap, FeatureMapSet, _create_map_dataset, open_feature_maps
from .tileset import TileSet


def _path_metadata(psi, order):
    paths = {"U1": tuple((int(f["j"]), float(f["theta"])) for f in psi)}
    if order == 2:
        paths["U2"] = tuple(
            (int(first["j"]), float(first["theta"]),
             int(second["j"]), float(second["theta"]))
            for first in psi for second in psi if second["j"] > first["j"])
    return paths


def wavelet_transform_tile(tile_or_tileset, filter_bank, order=2, *,
                           filepath=None, batch_size=None):
    """Compute U1/U2 modulus features and their scale/orientation paths.

    ``tile_or_tileset`` is one ``(H, W)`` or polarized ``(2, H, W)`` tile,
    or a :class:`TileSet`. ``order`` is 1 or 2. With ``filepath``, write
    tile-major HDF5 datasets as batches complete and return a lazy file-backed
    :class:`FeatureMap` or :class:`FeatureMapSet`. Close that object after use.
    Without a path, return the same object type backed by NumPy arrays.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    psi = filter_bank["psi"]
    if not psi:
        raise ValueError("filter bank must contain at least one wavelet")
    height, width = np.shape(psi[0]["val"])
    if any(np.shape(f["val"]) != (height, width) for f in psi):
        raise ValueError("all wavelets must share tile dimensions")
    paths = _path_metadata(psi, order)
    psi_vals = jnp.stack([f["val"] for f in psi])
    second_indices = tuple(tuple(i for i, f in enumerate(psi) if f["j"] > first["j"])
                           for first in psi)

    @jax.jit
    def transform_batch(batch):
        x_fft = jnp.fft.fft2(batch)
        u1 = jnp.abs(jnp.fft.ifft2(x_fft[:, None] * psi_vals[None]))
        result = {"U1": u1}
        if order == 2:
            u1_fft = jnp.fft.fft2(u1)
            groups = [jnp.abs(jnp.fft.ifft2(
                u1_fft[:, i, None] * psi_vals[jnp.array(indices)][None]))
                for i, indices in enumerate(second_indices) if indices]
            result["U2"] = (jnp.concatenate(groups, axis=1) if groups else
                            jnp.empty((batch.shape[0], 0, height, width), dtype=u1.dtype))
        return result

    is_set = isinstance(tile_or_tileset, TileSet)
    if is_set:
        data = tile_or_tileset.data
        pol = bool(tile_or_tileset.pol)
        metadata = {"nside": tile_or_tileset.nside,
                    "tile_nside": tile_or_tileset.tile_nside,
                    "margin": tile_or_tileset.margin}
    else:
        data = np.asarray(tile_or_tileset)[None]
        pol = data.ndim == 4
        metadata = {}
    if data.ndim != (4 if pol else 3) or data.shape[-2:] != (height, width):
        raise ValueError("input tile shape does not match the filter bank")
    n_tiles = data.shape[0]
    if n_tiles < 1:
        raise ValueError("input must contain at least one tile")
    components = data.shape[1] if pol else 1
    chunk_size = batch_size or (1 if filepath is not None else n_tiles)
    maps = {}
    filepath = Path(filepath) if filepath is not None else None
    handle = h5py.File(filepath, "w") if filepath is not None else None
    try:
        if handle is not None:
            handle.attrs["single_tile"] = not is_set
            handle.attrs["pol"] = pol
            for key, value in metadata.items():
                handle.attrs[key] = value
            for key, group in paths.items():
                width_of_path = 2 if key == "U1" else 4
                handle.create_dataset(f"{key}_paths",
                                      data=np.asarray(group, dtype=np.float64).reshape(-1, width_of_path))
        for start in range(0, n_tiles, chunk_size):
            stop = min(start + chunk_size, n_tiles)
            batch = data[start:stop]
            flat = batch.reshape(((stop - start) * components, height, width)) if pol else batch
            result = transform_batch(jnp.asarray(flat))
            for key, value in result.items():
                array = np.asarray(value)
                if pol:
                    array = array.reshape((stop - start, components) + array.shape[1:])
                if not is_set:
                    array = array[0]
                if key not in maps:
                    shape = ((n_tiles,) + array.shape[1:]) if is_set else array.shape
                    maps[key] = (_create_map_dataset(handle, key, shape, array.dtype,
                                                     not is_set, pol) if handle is not None
                                 else np.empty(shape, dtype=array.dtype))
                if is_set:
                    maps[key][start:stop] = array
                else:
                    maps[key][...] = array
    finally:
        if handle is not None:
            handle.close()
    if filepath is not None:
        return open_feature_maps(filepath)
    cls = FeatureMapSet if is_set else FeatureMap
    return cls(maps, paths, pol=pol, metadata=metadata)
