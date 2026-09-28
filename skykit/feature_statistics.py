"""Tile statistics of precomputed wavelet feature maps."""

from contextlib import ExitStack
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np


def _create_dataset(handle, key, shape, dtype, *, single_tile, polarized):
    """Create a tile-major dataset, chunked for individual tile/path reads."""
    path_axis = len(shape) - 3
    chunks = None
    if all(shape):
        chunks = tuple(1 if axis < path_axis else min(8, size) if axis == path_axis
                       else size for axis, size in enumerate(shape))
    dataset = handle.create_dataset(key, shape=shape, dtype=dtype, chunks=chunks)
    leading = [] if single_tile else ["tile"]
    if polarized:
        leading.append("stokes")
    for axis, label in enumerate(leading + ["path", "x", "y"]):
        dataset.dims[axis].label = label
    return dataset


def save_feature_maps(feature_maps, filepath, *, single_tile=None):
    """Write U1/U2 feature maps to one HDF5 file, including a single tile.

    ``feature_maps`` is a transform result or a dictionary of feature arrays.
    Set ``single_tile=True`` for a polarized single tile, whose four-dimensional
    shape otherwise looks like a scalar tileset. Arrays are copied one tile at
    a time. Returns the destination path.
    """
    groups = {key: feature_maps[key] for key in ("U1", "U2") if key in feature_maps}
    if not groups:
        raise ValueError("feature_maps must contain U1 or U2")
    first = next(iter(groups.values()))
    if single_tile is None:
        single_tile = first.ndim == 3
    expected_ranks = (3, 4) if single_tile else (4, 5)
    if first.ndim not in expected_ranks:
        raise ValueError("feature map shape is inconsistent with single_tile")
    polarized = first.ndim == (4 if single_tile else 5)
    _validate_groups(groups)
    filepath = Path(filepath)
    with h5py.File(filepath, "w") as handle:
        handle.attrs["single_tile"] = bool(single_tile)
        handle.attrs["pol"] = polarized
        for key, source in groups.items():
            dataset = _create_dataset(handle, key, source.shape, source.dtype,
                                      single_tile=single_tile, polarized=polarized)
            if single_tile:
                dataset[...] = source
            else:
                for tile in range(source.shape[0]):
                    dataset[tile] = source[tile]
    return filepath


def open_feature_maps(filepath):
    """Open an HDF5 feature-map set for lazy reads; use as a context manager.

    The returned ``h5py.File`` exposes ``U1`` and/or ``U2`` datasets and must
    be closed after use, preferably with ``with open_feature_maps(path) as maps``.
    """
    handle = h5py.File(filepath, "r")
    if not any(key in handle for key in ("U1", "U2")):
        handle.close()
        raise ValueError("HDF5 file must contain U1 or U2")
    try:
        _groups(handle)
    except Exception:
        handle.close()
        raise
    return handle


def _validate_groups(groups):
    batch_shape = None
    spatial_shape = None
    for key, value in groups.items():
        if not hasattr(value, "shape") or not 3 <= len(value.shape) <= 5:
            raise ValueError(f"{key} must have shape (..., paths, height, width)")
        if batch_shape is None:
            batch_shape = value.shape[:-3]
            spatial_shape = value.shape[-2:]
        elif value.shape[:-3] != batch_shape or value.shape[-2:] != spatial_shape:
            raise ValueError("feature map groups must have matching tile and pixel shapes")
    return batch_shape, spatial_shape


def _groups(feature_maps):
    if isinstance(feature_maps, dict):
        groups = {key: value for key, value in feature_maps.items()
                  if key in ("U1", "U2")}
    elif isinstance(feature_maps, h5py.File):
        groups = {key: feature_maps[key] for key in ("U1", "U2") if key in feature_maps}
    else:
        groups = {"U1": feature_maps}
    if not groups:
        raise ValueError("feature maps must contain U1 or U2")
    batch_shape, spatial_shape = _validate_groups(groups)
    return groups, batch_shape, spatial_shape


def compute_scattering_statistics(feature_maps1, feature_maps2=None, *,
                                  operation1=None, operation2=None,
                                  reduction="mean", path_batch_size=8):
    """Reduce operated feature maps over all pixels of each tile.

    Inputs may be transform results containing ``U1``/``U2``, dictionaries of
    arrays, or HDF5 files written by :func:`save_feature_maps`.
    A bare array is interpreted as ``U1``. Map shape is
    ``(..., n_paths, height, width)``; leading axes are retained in the result.

    With one set, each path gives one statistic of ``operation1(map)``.
    With two sets, only cross statistics are returned: for every pair of
    paths, reduce ``operation1(map1) * operation2(map2)``. The two path axes
    preserve the filter bank's order (scale, then orientation for ``U1``;
    valid ``j2 > j1`` paths for ``U2``). ``variance`` is the population
    variance across pixels. Operations must accept and return JAX arrays of
    the same shape; ``None`` means identity. Output keys are ``U1``/``U2``
    for one set and, for example, ``U1_U2`` for two sets.
    """
    if reduction not in ("mean", "variance"):
        raise ValueError("reduction must be 'mean' or 'variance'")
    if not isinstance(path_batch_size, int) or path_batch_size < 1:
        raise ValueError("path_batch_size must be a positive integer")
    operation1 = operation1 or (lambda x: x)
    operation2 = operation2 or (lambda x: x)
    with ExitStack() as stack:
        if isinstance(feature_maps1, (str, Path)):
            feature_maps1 = stack.enter_context(open_feature_maps(feature_maps1))
        if isinstance(feature_maps2, (str, Path)):
            feature_maps2 = stack.enter_context(open_feature_maps(feature_maps2))
        return _compute_statistics(feature_maps1, feature_maps2, operation1,
                                   operation2, reduction, path_batch_size)


def _compute_statistics(feature_maps1, feature_maps2, operation1, operation2,
                        reduction, path_batch_size):
    first, batch_shape, pixels = _groups(feature_maps1)
    second = None
    if feature_maps2 is not None:
        second, other_batch, other_pixels = _groups(feature_maps2)
        if other_batch != batch_shape or other_pixels != pixels:
            raise ValueError("feature map sets must have matching tile and pixel shapes")
    reduce = jnp.mean if reduction == "mean" else jnp.var
    result = {}
    tile_count = int(np.prod(batch_shape)) if batch_shape else 1

    def read_paths(array, tile, start, stop):
        leading = np.unravel_index(tile, batch_shape) if batch_shape else ()
        return array[leading + (slice(start, stop), slice(None), slice(None))]

    for name1, maps1 in first.items():
        n1 = maps1.shape[-3]
        if second is None:
            output = None
            for tile in range(tile_count):
                for start in range(0, n1, path_batch_size):
                    stop = min(start + path_batch_size, n1)
                    operated = operation1(jnp.asarray(read_paths(maps1, tile, start, stop)))
                    if operated.shape != (stop - start,) + pixels:
                        raise ValueError("operation1 must preserve feature map shape")
                    reduced = np.asarray(reduce(operated, axis=(-2, -1)))
                    if output is None:
                        output = np.empty((tile_count, n1), dtype=reduced.dtype)
                    output[tile, start:stop] = reduced
            if output is None:
                output = np.empty((tile_count, n1), dtype=np.float32)
            result[name1] = output.reshape(batch_shape + (n1,))
            continue
        for name2, maps2 in second.items():
            n2 = maps2.shape[-3]
            output = None
            for tile in range(tile_count):
                for start1 in range(0, n1, path_batch_size):
                    stop1 = min(start1 + path_batch_size, n1)
                    operated1 = operation1(jnp.asarray(read_paths(maps1, tile, start1, stop1)))
                    if operated1.shape != (stop1 - start1,) + pixels:
                        raise ValueError("operation1 must preserve feature map shape")
                    for start2 in range(0, n2, path_batch_size):
                        stop2 = min(start2 + path_batch_size, n2)
                        operated2 = operation2(jnp.asarray(read_paths(maps2, tile, start2, stop2)))
                        if operated2.shape != (stop2 - start2,) + pixels:
                            raise ValueError("operation2 must preserve feature map shape")
                        product = operated1[:, None] * operated2[None, :]
                        reduced = np.asarray(reduce(product, axis=(-2, -1)))
                        if output is None:
                            output = np.empty((tile_count, n1, n2), dtype=reduced.dtype)
                        output[tile, start1:stop1, start2:stop2] = reduced
            if output is None:
                output = np.empty((tile_count, n1, n2), dtype=np.float32)
            result[f"{name1}_{name2}"] = output.reshape(batch_shape + (n1, n2))
    return result
